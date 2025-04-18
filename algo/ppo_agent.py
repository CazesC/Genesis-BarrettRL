import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import argparse
from network.ppo import PPO

class PPOAgent:
    def __init__(self, input_dim, output_dim, lr, gamma, clip_epsilon, device, load=False, num_envs=1, hidden_dim=64, checkpoint_path=None):
        self.device = device
        self.num_envs = num_envs
        self.model = PPO(input_dim, output_dim, hidden_dim).to(self.device)
        self.checkpoint_path = checkpoint_path
        if load: 
            self.load_checkpoint()
            print("Loaded model from checkpoint")
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.gamma = gamma
        self.clip_epsilon = clip_epsilon
        self.entropy_coef = 0.01  # Coefficient for entropy regularization
        self.exploration_rate = 0.2  # 20% chance of random action
        self.exploration_decay = 0.995  # Decay rate for exploration
        self.min_exploration = 0.05  # Minimum exploration rate

        self.logged_ratios = []
        self.logged_advantages = []
        self.loss_log = []

        self.unclipped_log = []
        self.clipped_log = []


    def save_checkpoint(self):
        checkpoint = {
            'model_state_dict': self.model.state_dict()
        }
        torch.save(checkpoint, self.checkpoint_path)
        print(f"Checkpoint saved to {self.checkpoint_path}")

    def load_checkpoint(self):
        checkpoint = torch.load(self.checkpoint_path, map_location=torch.device(self.device))
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        print(f"Checkpoint loaded from {self.checkpoint_path}")

    def select_action(self, state):
        # Random exploration
        if torch.rand(1, device=self.device) < self.exploration_rate:
            return torch.randint(0, self.model.network[-1].out_features, (state.shape[0],), device=self.device)
        
        with torch.no_grad():
            logits = self.model(state)
        probs = nn.functional.softmax(logits, dim=-1)
        dist = Categorical(probs)
        action = dist.sample()
        
        # Decay exploration rate
        self.exploration_rate = max(self.min_exploration, self.exploration_rate * self.exploration_decay)
        
        return action

    def train(self, states, actions, rewards, dones):
        # Convert lists to tensors
        states_tensor = torch.stack(states).to(self.device)
        actions_tensor = torch.stack(actions).to(self.device)

        # Calculate discounted rewards
        discounted_rewards = []
        R = 0
        for reward in reversed(rewards):
            R = reward + self.gamma * R * (~dones[-1])
            discounted_rewards.insert(0, R)

        discounted_rewards_tensor = torch.stack(discounted_rewards).to(self.device)

        # Normalize the rewards
        advantages = discounted_rewards_tensor - discounted_rewards_tensor.mean()

        # Compute "old policy" distribution ONCE, before the PPO update loop
        with torch.no_grad():
            logits_old = self.model(states_tensor)
            probs_old = nn.functional.softmax(logits_old, dim=-1)
            dist_old = Categorical(probs_old)

        
        # Update policy using PPO
        for _ in range(10):  # Number of epochs for each batch update
            
            logits_new = self.model(states_tensor)
            probs_new = nn.functional.softmax(logits_new, dim=-1)

            
            dist_new = Categorical(probs_new)

            ratio = dist_new.log_prob(actions_tensor) - dist_old.log_prob(actions_tensor)
            ratio = ratio.exp()
            print(torch.allclose(logits_old, logits_new, atol=1e-5))

            
            

            # Log for visualization (detach to prevent gradient issues)
            self.logged_ratios.append(ratio.detach().cpu())
            self.logged_advantages.append(advantages.detach().cpu())

            # Calculate surrogate loss
            surrogate_loss_1 = ratio * advantages
            surrogate_loss_2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
            
            # Add entropy regularization to encourage exploration
            entropy = -(probs_new * torch.log(probs_new + 1e-10)).sum(dim=-1).mean()
            
            # Combined loss with entropy regularization
            loss = -torch.min(surrogate_loss_1, surrogate_loss_2).mean() - self.entropy_coef * entropy
            print(loss)

            # Perform optimization step
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            self.loss_log.append(loss.item())
            print(f"loss: {loss}")