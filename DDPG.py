import torch
from torch import nn
from torch.optim import Adam
from copy import deepcopy
from torch.distributions.normal import Normal
import numpy as np
import random
import gym

class QNet(nn.Module):
    def __init__(self, obs_shape, action_shape, hidden_size):
        super().__init__()
        self.obs_shape = obs_shape
        self.action_shape = action_shape
        self.hidden_size = hidden_size
        self.Q = nn.Sequential(
            nn.Linear(self.obs_shape+self.action_shape, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, 1)
        )

    def forward(self, obsacts):
        return self.Q(obsacts)

class PiNet(nn.Module):
    def __init__(self, obs_shape, action_shape, hidden_size, act_high):
        super().__init__()
        self.obs_shape = obs_shape
        self.action_shape = action_shape
        self.hidden_size = hidden_size
        self.act_high = act_high
        self.Pi = nn.Sequential(
            nn.Linear(self.obs_shape, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.action_shape),
            nn.Tanh()
        )

    def forward(self, obs):
        return self.Pi(obs) * self.act_high

class DDPG_Agent(nn.Module):
    def __init__(self, obs_shape, action_shape, hidden_size, act_high):
        super().__init__()
        # S,A (Env-Agent) params
        self.obs_shape = obs_shape
        self.action_shape = action_shape
        self.act_high = act_high
        # Networks
        self.hidden_size = hidden_size
        self.QNetwork = QNet(obs_shape, action_shape, hidden_size)
        self.PiNetwork = PiNet(obs_shape, action_shape, hidden_size, act_high)

        self.QTarget = deepcopy(self.QNetwork)
        self.PiTarget = deepcopy(self.PiNetwork)
        for p in self.QTarget.parameters():
            p.requires_grad = False
        for p in self.PiTarget.parameters():
            p.requires_grad = False
        # Buffer lists
        self.batch_obs = []
        self.batch_acts = []
        self.batch_rews = []
        self.batch_dones = []
        self.batch_next_obs = []
        # Training params
        self.batch_size = 128
        self.gamma = 0.99
        self.tau = 0.995

        self.MseLoss = nn.MSELoss()
        self.optimizerQ = Adam(self.QNetwork.parameters(), lr=1e-3)
        self.optimizerPi = Adam(self.PiNetwork.parameters(), lr=1e-3)

    def action_select(self, obs):
        obs_tensor = torch.from_numpy(obs)
        mean = self.PiNetwork(obs_tensor).item()
        noise = Normal(torch.tensor([0.0]), torch.tensor([1.0])).sample()
        action = torch.clip(mean+noise*0.1, -2.0, 2.0).item()
        return action

    def soft_target_weight_update(self):
        QNetwork_weights = self.QNetwork.state_dict()
        QTarget_weights = self.QTarget.state_dict()
        for key in QNetwork_weights:
            QTarget_weights[key] = QTarget_weights[key] * self.tau + QNetwork_weights[key] * (1 - self.tau)
        self.QTarget.load_state_dict(QTarget_weights)

        PiNetwork_weights = self.PiNetwork.state_dict()
        PiTarget_weights = self.PiTarget.state_dict()
        for key in PiNetwork_weights:
            PiTarget_weights[key] = PiTarget_weights[key] * self.tau + PiNetwork_weights[key] * (1 - self.tau)
        self.PiTarget.load_state_dict(PiTarget_weights)

    def get_buffer_data(self):
        buffer_len = len(self.batch_obs)
        s = self.batch_size
        batch_indicies = [random.randint(0, buffer_len-1) for _ in range(s)]
        batch_obs = [self.batch_obs[i] for i in batch_indicies]
        batch_acts = [self.batch_acts[i] for i in batch_indicies]
        batch_rews = [self.batch_rews[i] for i in batch_indicies]
        batch_dones = [self.batch_dones[i] for i in batch_indicies]
        batch_next_obs = [self.batch_next_obs[i] for i in batch_indicies]
        batch_obs = torch.tensor(np.array(batch_obs)).detach()
        batch_acts = torch.tensor(np.array(batch_acts), dtype=torch.float32).detach()
        batch_rews = torch.tensor(np.array(batch_rews), dtype=torch.float32).detach()
        batch_dones = torch.tensor(np.array(batch_dones)).detach()
        batch_next_obs = torch.tensor(np.array(batch_next_obs)).detach()
        return batch_obs, batch_acts, batch_rews, batch_dones, batch_next_obs

    def update(self):
        batch_obs, batch_acts, batch_rews, batch_dones, batch_next_obs = self.get_buffer_data()

        Qtargs = self.QTarget(torch.cat((batch_next_obs, self.PiTarget(batch_next_obs)), dim=1)).squeeze()
        targets = batch_rews + self.gamma * (1 - batch_dones.int()) * Qtargs
        Qvals = self.QNetwork(torch.cat((batch_obs, batch_acts.unsqueeze(dim=1)), dim=1)).squeeze()

        self.optimizerQ.zero_grad()
        QLoss = self.MseLoss(Qvals, targets)
        QLoss.backward()
        self.optimizerQ.step()

        self.optimizerPi.zero_grad()
        PiLoss = -1 * self.QNetwork(torch.cat((batch_obs, self.PiNetwork(batch_obs)), dim=1)).mean()
        PiLoss.backward()
        self.optimizerPi.step()

        with torch.no_grad():
            self.soft_target_weight_update()

        return QLoss, PiLoss


def train(env_id="Pendulum-v1", hidden_size=64):
    env = gym.make(env_id)
    agent = DDPG_Agent(env.observation_space.shape[0], env.action_space.shape[0], hidden_size, env.action_space.high[0])
    num_episodes = 200
    total_rews_by_ep = []
    for episode in range(num_episodes):
        obs, done, trunc, t = env.reset()[0], False, False, 0
        while not (done or trunc):
            if episode >= 5:
                with torch.no_grad():
                    action = agent.action_select(obs)
            else:
                action = env.action_space.sample()[0]
            np_action = np.array([action])
            new_obs, rew, done, trunc, _ = env.step(np_action)
            t += 1
            agent.batch_obs.append(obs)
            agent.batch_acts.append(action)
            agent.batch_rews.append(rew)
            agent.batch_dones.append(done)
            agent.batch_next_obs.append(new_obs)
            obs = new_obs
        if len(agent.batch_obs) >= agent.batch_size:
            for _ in range(env._max_episode_steps): # t
                QLoss, PiLoss = agent.update()
        ep_rew = sum(agent.batch_rews[-t:])
        total_rews_by_ep.append(ep_rew)
        print(f"| Episode {episode:<3} done | Len: {t:<3} | Rewards: {ep_rew:<4.1f} | Q-Loss: {QLoss:<4.1f} | Pi-Loss: {PiLoss:<4.1f} |")
    return total_rews_by_ep

def plot(ep_rews):
    import matplotlib.pyplot as plt
    plt.plot(ep_rews)
    plt.title("Pendulum-v1, DDPG, hidden_size=64, episodes=500")
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
