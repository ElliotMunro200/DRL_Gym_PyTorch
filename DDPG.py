import torch
from torch import nn
from torch.optim import Adam
from copy import deepcopy
from torch.distributions.normal import Normal
import numpy as np
import random
import gym

from gym.spaces import Box
from DDPG_base import combined_shape, MLPActorCritic_DDPG
from utils import get_args, printing

class DDPG_Buffer():
    def __init__(self, args, obs_dim, act_dim):
        self.args = args
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.obss = np.zeros(combined_shape(1, self.obs_dim))
        self.acts = np.zeros(combined_shape(1, self.act_dim))
        self.rews = np.zeros((1,))
        self.dones = np.zeros_like(self.rews, dtype=bool)
        self.next_obss = np.zeros(combined_shape(1, self.obs_dim))
        self.num_store = 0
        self.rews_by_ep = []
        self.rews_current_ep = np.zeros((1,))

    def add(self, obss, acts, rews, dones, next_obss):
        self.obss[self.num_store] = obss
        self.acts[self.num_store] = acts
        self.rews[self.num_store] = rews
        self.dones[self.num_store] = dones
        self.next_obss[self.num_store] = next_obss
        self.num_store += 1
        return

    def get(self):
        # converting to torch.float32 tensors
        obss = torch.from_numpy(self.obss[:, :-1]).type(torch.float32)
        acts = torch.from_numpy(self.acts[:, :-1]).type(torch.float32)
        rets = torch.from_numpy(self.rets).type(torch.float32)
        advs = torch.from_numpy(self.advs).type(torch.float32)
        return obss, acts, rets, advs


class DDPG_Agent(nn.Module):
    def __init__(self, obs_space, act_space, args):
        super().__init__()
        self.obs_space = obs_space
        self.obs_dim = self.obs_space.shape[0]
        self.act_space = act_space
        assert isinstance(act_space, Box)
        self.act_dim = self.act_space.shape[0]
        self.args = args
        self.h_sizes = self.args.hidden_sizes
        self.DDPG_ac = MLPActorCritic_DDPG(self.obs_space, self.act_space, hidden_sizes=self.h_sizes)
        with torch.no_grad():
            self.DDPG_ac_target = deepcopy(self.DDPG_ac)
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
        self.optimizer = Adam(self.DDPG_ac.parameters(), lr=1e-3)

    def action_select(self, obs):
        obs_tensor = torch.from_numpy(obs)
        mean = self.DDPG_ac.pi(obs_tensor).item()
        noise = Normal(torch.tensor([0.0]), torch.tensor([1.0])).sample()
        action = torch.clip(mean+noise*0.1, -2.0, 2.0).item()
        return action

    def soft_target_weight_update(self):
        Q_weights = self.DDPG_ac.q1.state_dict()
        QTarget_weights = self.DDPG_ac_target.q1.state_dict()
        for key in Q_weights:
            QTarget_weights[key] = QTarget_weights[key] * self.tau + Q_weights[key] * (1 - self.tau)
        self.DDPG_ac_target.q1.load_state_dict(QTarget_weights)

        Pi_weights = self.DDPG_ac.pi.state_dict()
        PiTarget_weights = self.DDPG_ac_target.pi.state_dict()
        for key in Pi_weights:
            PiTarget_weights[key] = PiTarget_weights[key] * self.tau + Pi_weights[key] * (1 - self.tau)
        self.DDPG_ac_target.pi.load_state_dict(PiTarget_weights)

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

        Q_from_target = self.DDPG_ac_target.q1(batch_next_obs, self.DDPG_ac_target.pi(batch_next_obs))
        targets = batch_rews + self.gamma * (1 - batch_dones.int()) * Q_from_target
        Qvals = self.DDPG_ac.q1(batch_obs, batch_acts.unsqueeze(dim=1))

        self.optimizer.zero_grad()
        loss = self.MseLoss(Qvals, targets) + -1 * self.DDPG_ac.q1(batch_obs, self.DDPG_ac.pi(batch_obs)).mean()
        loss.backward()
        self.optimizer.step()

        with torch.no_grad():
            self.soft_target_weight_update()

        return loss


def train(args):
    env = gym.make(args.env_id)
    agent = DDPG_Agent(env.observation_space, env.action_space, args)
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
                loss = agent.update()
        ep_rew = sum(agent.batch_rews[-t:])
        total_rews_by_ep.append(ep_rew)
        print(f"| Episode {episode:<3} done | Len: {t:<3} | Rewards: {ep_rew:<4.1f} | Loss: {loss:<4.1f} |")
    return total_rews_by_ep


def plot(ep_rews, exp_info):
    import matplotlib.pyplot as plt
    plt.plot(ep_rews)
    plt.title(exp_info)
    plt.ylabel("Total Rewards")
    plt.xlabel("Episode")
    plt.show()

if __name__ == "__main__":
    import time
    args = get_args()
    exp_info = printing(args, gym.make(args.env_id))
    start_time = time.time()
    ep_rews = train(args)
    end_time = time.time()
    print(f"TOTAL TRAINING TIME: {end_time - start_time:.2f}s")
    plot(ep_rews, exp_info)

# TODO:
# 1) make a numpy array buffer
# 2) make agent step function
# 3) change the step loop
# 4) put time into the agent
# 5) put episode reward tracking to the agent

# actions = agent.step(envs.reset()[0])
# env_T = 0 # to remove
# while agent.T < args.training_steps:
#     obss, rews, dones, truncs, infos = envs.step(actions)
#     env_T += 1  # to remove
#     print(f"agent.T: {agent.T}, env_T: {env_T}")
#     assert agent.T == env_T  # to remove
#     actions = agent.step(obss, rews, dones, truncs, infos)
# rewards_by_episode = agent.rewards_by_episode
# return rewards_by_episode