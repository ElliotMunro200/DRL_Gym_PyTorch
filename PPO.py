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
        self.batch_old_logp = []
        self.batch_rews = []
        self.batch_advantages = []

    def empty_buffer(self):
        self.batch_obs = []
        self.batch_acts = []
        self.batch_old_logp = []
        self.batch_rews = []
        self.batch_advantages = []

    def get_buffer_data(self):
        batch_obs = torch.tensor(np.array(self.batch_obs)).detach()
        batch_acts = torch.tensor(np.array(self.batch_acts)).detach()
        batch_old_logp = torch.tensor(np.array(self.batch_old_logp)).detach()
        batch_rews = torch.tensor(np.array(self.batch_rews)).detach()
        batch_advantages = torch.tensor(np.array(self.batch_advantages)).detach()
        return batch_obs, batch_acts, batch_old_logp, batch_rews, batch_advantages

    def get_policy(self, obs_tensor):
        logits = self.logits_net(obs_tensor)
        policy = Categorical(logits=logits)
        return policy

    def action_select(self, obs):
        obs_tensor = torch.from_numpy(obs).type(torch.float32)
        dist = self.get_policy(obs_tensor)
        action = dist.sample()
        logp = dist.log_prob(action)
        return action.item(), logp.item()

    def rewards_to_go(self):
        episode_rewards = self.batch_rews.copy()
        rews_to_go = [sum(episode_rewards[i:]) for i in range(len(episode_rewards))]
        return rews_to_go

    def update(self):
        batch_obs, batch_acts, batch_old_logp, batch_rews, batch_advantages = self.get_buffer_data()
        batch_rets = torch.tensor(np.array(self.rewards_to_go()), dtype=torch.float32)
        state_values = self.critic(batch_obs).squeeze()
        batch_advantages = batch_rets.detach() - state_values.detach()
        self.batch_advantages.append(batch_advantages)

        logp = self.get_policy(batch_obs).log_prob(batch_acts)
        ratio = torch.exp(logp - batch_old_logp.detach())
        surr1 = ratio * batch_advantages
        surr2 = torch.clamp(ratio, 1 - self.clip_param, 1 + self.clip_param) * batch_advantages
        loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, batch_rets)

        self.optimizer.zero_grad()
        loss.mean().backward()
        self.optimizer.step()

        return loss.mean(), state_values.mean()

def train(env_name='CartPole-v1', hidden_size=32):
    env = gym.make(env_name)
    agent = PPO_Agent(env.observation_space.shape[0], env.action_space.n, hidden_size)
    max_episodes = 500
    training_rewards_by_episode = []
    for ep in range(max_episodes):
        agent.empty_buffer()
        obs = env.reset()[0]
        done, trunc = False, False
        t = 0
        while not (done or trunc):
            action, logp = agent.action_select(obs)
            agent.batch_obs.append(obs)
            agent.batch_acts.append(action)
            agent.batch_old_logp.append(logp)
            obs, rew, done, trunc, _ = env.step(action)
            t += 1
            agent.batch_rews.append(rew)
        episode_rewards = agent.batch_rews
        total_ep_rews = sum(episode_rewards)
        training_rewards_by_episode.append(total_ep_rews)
        loss_mean, values_mean = agent.update()
        print(f"| Trained Episode {ep} | Rewards: {total_ep_rews:<5} | Ep Len: {t:<3} | Loss: {loss_mean:<5.1f} | Value: {values_mean:.1f} |")
    return training_rewards_by_episode


def plot(ep_rews):
    import matplotlib.pyplot as plt
    plt.plot(ep_rews)
    plt.title(f"CartPole-v1, PPO, hidden_dim=32, episodes=500")
    plt.xlabel("Episode")
    plt.ylabel("Total rewards")
    plt.show()


if __name__ == '__main__':
    import time
    start_time = time.time()
    ep_rews = train()
    end_time = time.time()
    print(f"END_TIME: {end_time-start_time}")
    plot(ep_rews)
