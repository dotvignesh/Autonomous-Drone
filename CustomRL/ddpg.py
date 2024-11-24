import torch
import torch.nn as nn
import numpy as np
from torch.distributions import MultivariateNormal


class MemoryReplayBuffer(object):
    def __init__(self, max_size, input_shape, action_size):
        
        """
        Initialize memory replay buffer. (State, Action, Reward, Next State, and Terminal flag)

        Args:
            max_size (int): Maximum size of buffer.
            input_shape (tuple): Shape of state input.
            action_size (int): Size of action space.
        """

        self.max_size = max_size
        self.memory_index = 0
        self.current_memory = 0
        self.state_mem = np.zeros((max_size, input_shape))
        self.action_mem = np.zeros((max_size, action_size))
        self.new_state_mem = np.zeros((max_size, input_shape))
        self.reward_mem = np.zeros(max_size)
        self.terminal_mem = np.zeros(max_size)

    def store_transition(self, state: np.ndarray, action: np.ndarray, reward: float, next_state: np.ndarray, terminal: bool) -> None:
        """
        Store a transition in the buffer. Dynamically updates the memory count (with a modulo)

        Args:
            state (np.ndarray): The state.
            action (np.ndarray): The action.
            reward (float): The reward.
            next_state (np.ndarray): The next state.
            done (bool): Whether the episode is done.
        """
        index = self.memory_index % self.max_size
        self.state_mem[index] = state
        self.action_mem[index] = action
        self.reward_mem[index] = reward
        self.new_state_mem[index] = next_state
        self.terminal_mem[index] = 1 - terminal
        self.memory_index = (self.memory_index + 1) % self.max_size
        self.current_memory += 1 if self.current_memory < self.max_size else 0

    def sample(self, batch_size):
        """
        Sample a batch of transitions from the buffer.

        Args:
            batch_size (int): The batch size.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]: A tuple containing the sampled states, actions, rewards, next states, and done flags.
        """
        current_memory = min(self.current_memory, self.max_size)
        choices = np.random.choice(current_memory, batch_size)

        states = self.state_mem[choices]
        actions = self.action_mem[choices]
        rewards = self.reward_mem[choices]
        next_states = self.new_state_mem[choices]
        dones = self.terminal_mem[choices]

        return states, actions, rewards, next_states, dones
    

######## Implementation for OU Noise based from https://soeren-kirchner.medium.com/deep-deterministic-policy-gradient-ddpg-with-and-without-ornstein-uhlenbeck-process-e6d272adfc3
class OUNoise(object):
    """Ornstein-Uhlenbeck process."""

    def __init__(self, mu, theta=0.15, sigma=0.2, dt=1e-2):
        """Initialize parameters and noise process."""
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.dt = dt
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = np.copy(self.mu) 

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) * self.dt+ self.sigma * np.sqrt(self.dt) * np.random.normal(scale=0.5,size=self.mu.shape)
        self.state = x + dx
        return self.state
    
class CriticNetwork(nn.Module):
    def __init__(self, lr_critic, state_dim, hidden_dim1, hidden_dim2, action_size):
        super(CriticNetwork, self).__init__()

        self.hidden_dim1 = hidden_dim1
        self.hidden_dim2 = hidden_dim2
        self.relu = nn.ReLU()
        self.action_size = action_size

        ##### Fully connected layer 1
        self.hiddenlayer1 = nn.Linear(state_dim, hidden_dim1)
        # Simple Truncated Normal Distribution initialization https://www.pinecone.io/learn/weight-initialization/
        h1 = 1./np.sqrt(self.hiddenlayer1.weight.data.size()[0])
        # Ensure uniform distribution between -h1 and h1 for each weight and bias
        torch.nn.init.uniform_(self.hiddenlayer1.weight.data, -h1, h1)
        torch.nn.init.uniform_(self.hiddenlayer1.bias.data, -h1, h1)
        # Layer Normalization to improve training stability https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html
        self.bn1 = nn.LayerNorm(self.hidden_dim1)

        
        ##### Fully connected layer 2
        self.hiddenlayer2 = nn.Linear(hidden_dim1, hidden_dim2)
        # Simple Truncated Normal Distribution initialization
        h2 = 1./np.sqrt(self.hiddenlayer2.weight.data.size()[0])
        # Ensure uniform distribution between -h2 and h2 for each weight and bias
        torch.nn.init.uniform_(self.hiddenlayer2.weight.data, -h2, h2)
        torch.nn.init.uniform_(self.hiddenlayer2.bias.data, -h2, h2)
        # Layer Normalization
        self.bn2 = nn.LayerNorm(self.hidden_dim2)


        ##### Action Value Layer
        self.act_value = nn.Linear(action_size, hidden_dim2)

        ##### Final Layer to get q value of state action pair
        fq = 0.005
        self.finalq = nn.Linear(hidden_dim2, 1)
        torch.nn.init.uniform_(self.finalq.weight.data, -fq, fq)
        torch.nn.init.uniform_(self.finalq.bias.data, -fq, fq)

        ##### Optimizer
        # self.optimizer = torch.optim.Adam(self.parameters(), lr=lr_critic)

        ##### Device
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    def forward(self, state, action):
        
        # First Layer
        x = self.hiddenlayer1(state)
        x = self.bn1(x)
        x = self.relu(x)

        # Second Layer
        x = self.hiddenlayer2(x)
        x = self.bn2(x)
        x = self.relu(x)

        # Action Value
        action_value = self.relu(self.act_value(action))

        # Add state value and action value together and RELU activation
        x = torch.add(x, action_value)
        x = self.relu(x)

        # Final Layer to get q value of state action pair
        x = self.finalq(x)

        return x


class ActorNetwork(nn.Module):
    def __init__(self, lr_actor, state_dim, hidden_dim1, hidden_dim2, action_size):
        super(ActorNetwork, self).__init__()

        self.hidden_dim1 = hidden_dim1
        self.hidden_dim2 = hidden_dim2
        self.action_size = action_size
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

        ##### Fully connected layer 1
        self.hiddenlayer1 = nn.Linear(state_dim, hidden_dim1)
        # Simple Truncated Normal Distribution initialization https://www.pinecone.io/learn/weight-initialization/
        h1 = 1./np.sqrt(self.hiddenlayer1.weight.data.size()[0])
        # Ensure uniform distribution between -h1 and h1 for each weight and bias
        torch.nn.init.uniform_(self.hiddenlayer1.weight.data, -h1, h1)
        torch.nn.init.uniform_(self.hiddenlayer1.bias.data, -h1, h1)
        # Layer Normalization to improve training stability https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html
        self.bn1 = nn.LayerNorm(self.hidden_dim1)

        
        ##### Fully connected layer 2
        self.hiddenlayer2 = nn.Linear(hidden_dim1, hidden_dim2)
        # Simple Truncated Normal Distribution initialization
        h2 = 1./np.sqrt(self.hiddenlayer2.weight.data.size()[0])
        # Ensure uniform distribution between -h2 and h2 for each weight and bias
        torch.nn.init.uniform_(self.hiddenlayer2.weight.data, -h2, h2)
        torch.nn.init.uniform_(self.hiddenlayer2.bias.data, -h2, h2)
        # Layer Normalization
        self.bn2 = nn.LayerNorm(self.hidden_dim2)


        ##### Final Layer to get action values
        self.actions = nn.Linear(hidden_dim2, action_size)
        action_weight = 0.005
        torch.nn.init.uniform_(self.actions.weight.data, -action_weight, action_weight)
        torch.nn.init.uniform_(self.actions.bias.data, -action_weight, action_weight)


        ##### Optimizer
        # self.optimizer = torch.optim.Adam(self.parameters(), lr=lr_actor)

        ##### Device
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    def forward(self, state):
        # First Layer
        x = self.hiddenlayer1(state)
        x = self.bn1(x)
        x = self.relu(x)

        # Second Layer
        x = self.hiddenlayer2(x)
        x = self.bn2(x)
        x = self.relu(x)

        # Final Layer to get action values
        x = self.actions(x)
        x = self.tanh(x)

        return x

class DDPG:
    def __init__(self, state_dim, action_size, lr_actor, lr_critic, gamma, tau, hidden_dim1=64, hidden_dim2=128, memory_size=100000, batch_size=64):
        self.gamma = gamma
        self.tau = tau
        self.action_size = action_size
        self.memory_size = memory_size
        self.batch_size = batch_size

        self.experience_replay_buffer = MemoryReplayBuffer(memory_size, state_dim, action_size)

        self.actor = ActorNetwork(lr_actor, state_dim, hidden_dim1, hidden_dim2, action_size)
        self.actor_target = ActorNetwork(lr_actor, state_dim, hidden_dim1, hidden_dim2, action_size)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr_actor)

        self.critic = CriticNetwork(lr_critic, state_dim, hidden_dim1, hidden_dim2, action_size)
        self.critic_target = CriticNetwork(lr_critic, state_dim, hidden_dim1, hidden_dim2, action_size)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr_critic)

        self.noise = OUNoise(np.zeros(action_size))

        self.loss_fn = nn.MSELoss()
        self.update_network_parameters(tau=1.0)

    def remember_transition(self, state, action, reward, next_state, terminal):
        self.experience_replay_buffer.store_transition(state, action, reward, next_state, terminal)
    
    def select_action(self, state):
        self.actor.eval()
        state = torch.from_numpy(state).float().to(self.actor.device)
        # print("State: ", state)
        
        action = self.actor(state).detach().data.numpy().flatten()
        # print("Action: ", action)

        # Add noise to action to encourage exploration
        action_with_noise = action + self.noise.sample()
        action = np.clip(action_with_noise, -1, 1)
        # print("Action with noise: ", action)
        
        self.actor.train()

        return action
    
    def learn(self):
        
        # only learn if we have enough experience
        if self.experience_replay_buffer.current_memory < self.batch_size:
            return

        # sample from replay buffer using batch training
        state, action, reward, next_state, terminal = self.experience_replay_buffer.sample(self.batch_size)
        # convert to tensors
        state = torch.from_numpy(state).float().to(self.actor.device)
        action = torch.from_numpy(action).float().to(self.actor.device)
        reward = torch.from_numpy(reward).float().to(self.actor.device)
        next_state = torch.from_numpy(next_state).float().to(self.actor.device)
        terminal = torch.from_numpy(terminal).float().to(self.actor.device)

        # Turn both networks into evaluation mode
        self.actor.eval()
        self.critic.eval()
        # Calculate target q value and current q value
        with torch.no_grad():
            current_q_value = self.critic(state, action)
            next_action = self.actor_target(next_state)
            next_q_value = self.critic_target(next_state, next_action)
            
            # print("shape of next_q_value: ", next_q_value.shape)
            # print("shape of reward: ", reward.shape)
            # print("shape of terminal: ", terminal.shape)

            # print("1 - terminal: ", 1 - terminal)
            # print("next_q_value: ", next_q_value)
            # print("reward: ", reward)

            # calculate target q value
            target_q_value = reward.unsqueeze(1) + self.gamma * next_q_value * (1 - terminal.unsqueeze(1))
            target_q_value = target_q_value.detach()

        # Turn both networks into training mode for gradient updates
        self.critic.train()
        self.actor.train()

        current_q_value = self.critic(state, action)

        current_q_value = current_q_value.reshape(-1)
        target_q_value = target_q_value.reshape(-1)

        # Update critic network
        critic_loss = self.loss_fn(current_q_value, target_q_value)
        # print("critric loss: ", critic_loss)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Update actor network
        actor_loss = -self.critic(state, self.actor(state))
        # print("actor loss: ", actor_loss)
        actor_loss = torch.mean(actor_loss)
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Soft update on the target netowrk parameters
        self.update_network_parameters()

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        actor_params = self.actor.named_parameters()
        critic_params = self.critic.named_parameters()

        actor_state_dict = dict(actor_params)
        critic_state_dict = dict(critic_params)

        for name in actor_state_dict:
            actor_state_dict[name] = tau*actor_state_dict[name].clone() + (1-tau)*self.actor_target.state_dict()[name].clone()

        for name in critic_state_dict:
            critic_state_dict[name] = tau*critic_state_dict[name].clone() + (1-tau)*self.critic_target.state_dict()[name].clone()

        self.actor_target.load_state_dict(actor_state_dict)
        self.critic_target.load_state_dict(critic_state_dict)

    def save(self, checkpoint_path):
        torch.save(self.actor.state_dict(), checkpoint_path)

    def load(self, checkpoint_path):
        state_dict = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
        self.actor.load_state_dict(state_dict)