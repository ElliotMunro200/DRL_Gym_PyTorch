import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from torch.optim import Adam
import numpy as np
import gym

# get batch of experience
# update policy - clip loss
# update value function - MSE loss on returns

class PPO_Agent(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_dim):
        super(PPO_Agent, self).__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.hidden_dim = hidden_dim
        self.logits_net = nn.Sequential(
            nn.Linear(self.obs_dim, self.hidden_dim),
            nn.Tanh(),
            nn.Linear(self.hidden_dim, self.act_dim)
        )
        self.critic = nn.Sequential(
            nn.Linear(self.obs_dim, self.hidden_dim),
            nn.Tanh(),
            nn.Linear(self.hidden_dim, 1)
        )
        self.optimizer = Adam([
            {'params': self.logits_net.parameters(), 'lr': 1e-2},
            {'params': self.critic.parameters(), 'lr': 1e-2}
        ])
        self.clip_param = 0.2
        self.MseLoss = nn.MSELoss()
        self.batch_obs = []
        self.batch_acts = []
        self.batch_rews = []
        self.batch_advantages = []
        self.old_logp = []

    def buffer(self):
        batch_obs = torch.tensor(np.array(self.batch_obs))
        batch_acts = torch.tensor(np.array(self.batch_acts))
        batch_rews = torch.tensor(np.array(self.batch_rews))
        batch_advantages = torch.tensor(np.array(self.batch_advantages))
        old_logp = torch.tensor(np.array(self.old_logp))
        return batch_obs, batch_acts, batch_rews, batch_advantages, old_logp

    def get_policy(self, obs):
        obs = torch.tensor(obs, dtype=torch.float32)
        logits = self.logits_net(obs)
        policy = Categorical(logits=logits)
        return policy

    def action_select(self, obs):
        action = self.get_policy(obs).sample().item()
        return action

    def update(self):
        batch_obs, batch_acts, batch_rews, batch_advantages, old_logp = self.buffer()
        batch_rets = []
        # calculate returns

        logp = self.get_policy(batch_obs).log_prob(batch_acts)
        state_values = self.critic(batch_obs)
        ratio = torch.exp(logp - old_logp)
        surr1 = ratio * batch_advantages
        surr2 = torch.clamp(ratio, 1 - self.clip_param, 1 + self.clip_param) * batch_advantages
        loss = torch.max(surr1, surr2) + 0.5 * self.MseLoss(state_values, batch_rets)

        self.optimizer.zero_grad()
        loss.mean().backward()
        self.optimizer.step()


