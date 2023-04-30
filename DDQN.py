import torch
from torch import nn
from torch.optim import Adam
import numpy as np
import random
import math
import gym

class DDQN_Agent(nn.Module):
    def __init__(self, obs_shape, action_shape, hidden_size):
        # Network parameters
        super().__init__()
        self.obs_shape = obs_shape
        self.action_shape = action_shape
        self.hidden_size = hidden_size
        self.QNetwork = nn.Sequential(
            nn.Linear(self.obs_shape, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.action_shape)
        )
        self.QTarget = self.QNetwork
        self.QTarget.load_state_dict(self.QNetwork.state_dict())
        # Training parameters
        self.gamma = 0.99
        self.optimizer = Adam(self.QNetwork.parameters(), lr=1e-2)
        # Episode buffer lists
        self.batch_obs = []
        self.batch_acts = []
        self.batch_rews = []
        self.batch_dones = []
        self.batch_next_obs = []
        self.batch_len = 128
        self.epsilon = 0.9
        self.eps_end = 0.05
        self.eps_start = 0.9
        self.MseLoss = nn.MSELoss()

    def epsilon_decay(self, t):
        epsilon = self.eps_end + (self.eps_start - self.eps_end) * math.exp(-1*t*0.0001)
        self.epsilon = epsilon

    def action_select(self, obs, t):
        if random.random() < self.epsilon:
            action = np.random.choice(self.action_shape)
        else:
            with torch.no_grad():
                obs_tensor = torch.from_numpy(np.array(obs))
                logits = self.QNetwork(obs_tensor)
                action = torch.argmax(logits).item()
        self.epsilon_decay(t)
        return action

    def empty_buffer(self):
        self.batch_obs = []
        self.batch_acts = []
        self.batch_rews = []
        self.batch_dones = []
        self.batch_next_obs = []

    def get_buffer_data(self):
        buffer_len = len(self.batch_obs)
        s = self.batch_len
        batch_indicies = [random.randint(0, buffer_len-1) for _ in range(s)]
        batch_obs = [self.batch_obs[i] for i in batch_indicies]
        batch_acts = [self.batch_acts[i] for i in batch_indicies]
        batch_rews = [self.batch_rews[i] for i in batch_indicies]
        batch_dones = [self.batch_dones[i] for i in batch_indicies]
        batch_next_obs = [self.batch_next_obs[i] for i in batch_indicies]
        batch_obs = torch.tensor(np.array(batch_obs)).detach()
        batch_acts = torch.tensor(np.array(batch_acts)).detach()
        batch_rews = torch.tensor(np.array(batch_rews)).detach()
        batch_dones = torch.tensor(np.array(batch_dones)).detach()
        batch_next_obs = torch.tensor(np.array(batch_next_obs)).detach()
        return batch_obs, batch_acts, batch_rews, batch_dones, batch_next_obs

    def update(self):
        batch_obs, batch_acts, batch_rews, batch_dones, batch_next_obs = self.get_buffer_data()
        maxQ = torch.max(self.QTarget(batch_next_obs), dim=1).values.detach()
        TD_target = batch_rews + (1 - batch_dones.int()) * self.gamma * maxQ
        TD_target = TD_target.type(torch.float32)
        Qsa = self.QNetwork(batch_obs)
        Qvals = torch.gather(Qsa, 1, batch_acts.unsqueeze(dim=1)).squeeze() # (x,2)[x] --> (x)
        QLoss = self.MseLoss(TD_target, Qvals)

        self.optimizer.zero_grad()
        QLoss.mean().backward()
        self.optimizer.step()

        self.QTarget.load_state_dict(self.QNetwork.state_dict())
        return QLoss


def train(env_id="CartPole-v1", hidden_size=32):
    # Init env and agent
    env = gym.make(env_id)
    agent = DDQN_Agent(env.observation_space.shape[0], env.action_space.n, hidden_size)
    # Training loop over episodes and timesteps
    num_episodes = 200
    total_rews_by_ep = []
    total_timesteps = 0
    for episode in range(num_episodes):
        obs = env.reset()[0]
        done, trunc = False, False
        t = 0
        while not (done or trunc):
            action = agent.action_select(obs, total_timesteps)
            new_obs, rew, done, trunc, _ = env.step(action)
            t += 1
            total_timesteps += 1
            agent.batch_obs.append(obs)
            agent.batch_acts.append(action)
            agent.batch_rews.append(rew)
            agent.batch_dones.append(done)
            agent.batch_next_obs.append(new_obs)
            obs = new_obs
            if len(agent.batch_obs) >= agent.batch_len:
                Qloss = agent.update()
                if t % 100 == 0:
                    print(f"| Episode {episode:<3} done | Total timesteps: {total_timesteps:<5} | Len: {t:<3} | Rewards: {ep_rew:<4.1f} | Q-Loss: {Qloss:<4.1f} |")
        ep_rew = sum(agent.batch_rews[-t:])
        total_rews_by_ep.append(ep_rew)

    return total_rews_by_ep


def plot(rews):
    import matplotlib.pyplot as plt
    plt.plot(rews)
    plt.title("CartPole-v1, DDQN, hidden_size=32, episodes=200")
    plt.ylabel("Total Rewards")
    plt.xlabel("Episode")
    plt.show()


if __name__ == "__main__":
    import time
    start_time = time.time()
    ep_rews = train()
    end_time = time.time()
    print(f"END_TIME: {end_time - start_time}")
    plot(ep_rews)