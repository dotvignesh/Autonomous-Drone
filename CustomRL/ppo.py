import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal

class RolloutBuffer:
    """Stores the transitions collected during training"""
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.state_values = []
        self.is_terminals = []

    def clear(self):
        """Clear all stored transitions"""
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.state_values[:]
        del self.is_terminals[:]

class ActorNetwork(nn.Module):
    """Actor network for continuous action spaces"""
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, action_dim),
            nn.Tanh()
        )
    
    def forward(self, state):
        return self.network(state)

class CriticNetwork(nn.Module):
    """Critic network that estimates state values"""
    def __init__(self, state_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
    
    def forward(self, state):
        return self.network(state)

class ActorCritic(nn.Module):
    """Combined actor-critic network"""
    def __init__(self, state_dim, action_dim, action_std_init):
        super().__init__()
        self.device = torch.device('cpu')
        self.action_dim = action_dim
        self.action_var = torch.full((action_dim,), action_std_init * action_std_init).to(self.device)
        
        self.actor = ActorNetwork(state_dim, action_dim)
        self.critic = CriticNetwork(state_dim)

    def set_action_std(self, new_action_std):
        """Update the action standard deviation"""
        self.action_var = torch.full((self.action_dim,), new_action_std * new_action_std).to(self.device)

    def act(self, state):
        """Select an action given a state"""
        action_mean = self.actor(state)
        cov_mat = torch.diag(self.action_var).unsqueeze(dim=0)
        dist = MultivariateNormal(action_mean, cov_mat)

        action = dist.sample()
        action_logprob = dist.log_prob(action)
        state_val = self.critic(state)

        return action.detach(), action_logprob.detach(), state_val.detach()

    def evaluate(self, state, action):
        """Evaluate actions given states"""
        action_mean = self.actor(state)
        action_var = self.action_var.expand_as(action_mean)
        cov_mat = torch.diag_embed(action_var).to(self.device)
        dist = MultivariateNormal(action_mean, cov_mat)

        if self.action_dim == 1:
            action = action.reshape(-1, self.action_dim)

        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(state)

        return action_logprobs, state_values, dist_entropy

class PPO:
    """Proximal Policy Optimization algorithm"""
    def __init__(self, state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, action_std_init=0.6):
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.action_std = action_std_init
        self.device = torch.device('cpu')

        self.buffer = RolloutBuffer()
        self.policy = ActorCritic(state_dim, action_dim, action_std_init).to(self.device)
        self.policy_old = ActorCritic(state_dim, action_dim, action_std_init).to(self.device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.optimizer = torch.optim.Adam([
            {'params': self.policy.actor.parameters(), 'lr': lr_actor},
            {'params': self.policy.critic.parameters(), 'lr': lr_critic}
        ])
        
        self.mse_loss = nn.MSELoss()

    def select_action(self, state):
        """Select an action for the given state"""
        with torch.no_grad():
            state = torch.FloatTensor(state).to(self.device)
            action, action_logprob, state_val = self.policy_old.act(state)

        self._store_transition(state, action, action_logprob, state_val)
        return action.detach().cpu().numpy().flatten()

    def _store_transition(self, state, action, action_logprob, state_val):
        """Store transition in buffer"""
        self.buffer.states.append(state)
        self.buffer.actions.append(action)
        self.buffer.logprobs.append(action_logprob)
        self.buffer.state_values.append(state_val)

    def _compute_returns(self):
        """Compute discounted returns"""
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
        return self._normalize_rewards(torch.tensor(rewards, dtype=torch.float32).to(self.device))

    def _normalize_rewards(self, rewards):
        """Normalize rewards"""
        return (rewards - rewards.mean()) / (rewards.std() + 1e-7)

    def update(self):
        """Update policy parameters"""
        rewards = self._compute_returns()
        
        # Convert buffer data to tensors
        old_states = torch.squeeze(torch.stack(self.buffer.states, dim=0)).detach()
        old_actions = torch.squeeze(torch.stack(self.buffer.actions, dim=0)).detach()
        old_logprobs = torch.squeeze(torch.stack(self.buffer.logprobs, dim=0)).detach()
        old_state_values = torch.squeeze(torch.stack(self.buffer.state_values, dim=0)).detach()
        
        # Calculate advantages
        advantages = rewards - old_state_values

        # Optimize policy for K epochs
        for _ in range(self.K_epochs):
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)
            state_values = torch.squeeze(state_values)
            
            # Calculate ratios and surrogate losses
            ratios = torch.exp(logprobs - old_logprobs.detach())
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages
            
            # Calculate total loss
            loss = -torch.min(surr1, surr2) + 0.5 * self.mse_loss(state_values, rewards) - 0.01 * dist_entropy
            
            # Update policy
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        # Update old policy
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.buffer.clear()

    def decay_action_std(self, action_std_decay_rate, min_action_std):
        """Decay action standard deviation"""
        self.action_std = max(self.action_std - action_std_decay_rate, min_action_std)
        print(f"Action std decayed to: {self.action_std}")
        self.policy.set_action_std(self.action_std)
        self.policy_old.set_action_std(self.action_std)

    def save(self, checkpoint_path):
        """Save policy state"""
        torch.save(self.policy_old.state_dict(), checkpoint_path)

    def load(self, checkpoint_path):
        """Load policy state"""
        state_dict = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
        self.policy_old.load_state_dict(state_dict)
        self.policy.load_state_dict(state_dict)