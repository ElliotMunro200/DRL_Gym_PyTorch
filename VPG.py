import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
import torch.optim as optim
import gym
import numpy as np

# repeat for number of updates:
# collect experience until buffer is full: -
# init buffer lists -
# obs = env.reset() -
# action = agent.action_select() - via stochastic policy
# obs, rew, done = env.step(action) -
# add (obs, rew, done) to buffer lists -
# when buffer is full execute VPG update: -

class VPG_Agent(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_dim):
        super(VPG_Agent, self).__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.hidden_dim = hidden_dim
        self.fc1 = nn.Linear(self.obs_dim, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, self.act_dim)
        self.logits_net = nn.Sequential(
            nn.Linear(self.obs_dim, self.hidden_dim),
            nn.Tanh(),
            nn.Linear(self.hidden_dim, self.act_dim)
        )
        self.optimizer = optim.Adam(self.logits_net.parameters(), lr=1e-2)

    def get_policy(self, obs):
        obs = torch.tensor(obs, dtype=torch.float32)
        logits = self.logits_net(obs)
        policy = Categorical(logits=logits)
        return policy

    def action_select(self, obs):
        action = self.get_policy(obs).sample().item()
        return action

    def loss(self, batch_obs, batch_acts, batch_rets):
        logp = self.get_policy(batch_obs).log_prob(batch_acts)
        batch_rets = torch.tensor(batch_rets, dtype=torch.float32)
        return -(logp * batch_rets).mean()

    def update(self, batch_obs, batch_acts, batch_rets):
        batch_obs = torch.tensor(np.array(batch_obs))
        batch_acts = torch.tensor(np.array(batch_acts))
        batch_rets = torch.tensor(np.array(batch_rets))
        self.optimizer.zero_grad()
        batch_loss = self.loss(batch_obs, batch_acts, batch_rets)
        batch_loss.backward()
        self.optimizer.step()

def plot(all_episode_rews):
    import matplotlib.pyplot as plt
    plt.plot(all_episode_rews)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title(f"CartPole-v1, VPG, hidden_dim=32, epochs=50, batch_size=2000")
    plt.show()

def rewards_to_go(ep_rews):
    ep_rews_to_go = []
    for i in range(len(ep_rews)):
        ep_rews_to_go.append(sum(ep_rews[i:]))
    return ep_rews_to_go

if __name__ == "__main__":
    import time
    start_time = time.time()
    env = gym.make('CartPole-v1')
    print(f"env.action_space: {env.action_space}")
    print(f"env.observation_space: {env.observation_space}")
    print(f"env.observation_space.high: {env.observation_space.high}")
    print(f"env.observation_space.low: {env.observation_space.low}")
    print(f"env.action_space.n: {env.action_space.n}")
    print(f"env.observation_space.shape[0]: {env.observation_space.shape[0]}")
    print(f"env._max_episode_steps: {env._max_episode_steps}")

    agent = VPG_Agent(env.observation_space.shape[0], env.action_space.n, 32)

    all_episode_lens = []
    all_episode_rews = []
    for i_batch in range(50):
        batch_obs = []
        batch_acts = []
        batch_rews = []
        observation = env.reset()[0]
        t = 0
        ep_rews = []
        while True:
            #env.render()
            batch_obs.append(observation.copy())
            action = agent.action_select(observation)
            observation, reward, done, trunc, info = env.step(action)
            t += 1
            batch_acts.append(action)
            ep_rews.append(reward)
            if done or trunc:
                all_episode_lens.append(t+1)
                total_episode_reward = sum(ep_rews)
                all_episode_rews.append(total_episode_reward)
                ep_rews_to_go = rewards_to_go(ep_rews)
                batch_rews.extend(ep_rews_to_go)
                print(f"Batch {i_batch + 1}, episode {len(all_episode_lens)} finished after {t} timesteps, total reward: {total_episode_reward}")
                if len(batch_obs) >= 2000:
                    agent.update(batch_obs, batch_acts, batch_rews)
                    print(f"Updating after episode {len(all_episode_lens)} of batch {i_batch+1}.")
                    break
                observation = env.reset()[0]
                t = 0
                ep_rews = []
    print(f"Average episode length was: {sum(all_episode_lens)/len(all_episode_lens):.1f}")
    end_time = time.time()
    print(f"END_TIME: {end_time - start_time}")
    plot(all_episode_rews)