# A2C is a specific type of AC (VPG that bootstraps to estimate the returns)
# algorithm that both:
# 1) uses A to update pi and V (advantage function used)
# 2) uses parallel workers to collect experience (distributed RL)
# it is a synchronous version of A3C - an AC algorithm that uses asynchronous actor-learners

# build synchronous VectorEnv on CPU.
# build main AC agent on GPU, and a deepcopy of just the policy on the CPU.
# for T updates:
    # for set number of timesteps:
        # execute synchronous stepping with VectorEnv (appending data to local buffer each step).
        # calculate the returns on CPU.
        # pass all data to the GPU for the state value computations, gradient computation, and the weight update.
        # perform synchronous update of the AC weights.

import torch
import torch.nn as nn
from torch.optim import Adam

import gym
import time
import numpy as np
from copy import deepcopy
from utils import get_args
from AC_base import MLPActorCritic

class A2C_Buffer():
    def __init__(self, master_device):
        self.master_device = master_device
        self.obss = np.zeros()

    def __len__(self):
        return self.obss.shape[0]

    def add(self, obss, rews, dones, truncs, infos):
        return

    def get(self):
        obss = torch.from_numpy(self.obss)
        rews = torch.from_numpy(self.rews)
        dones = torch.from_numpy(self.dones)
        truncs = torch.from_numpy(self.truncs)
        infos = torch.from_numpy(self.infos)
        if self.master_device != "cpu":
            obss.to(self.master_device)
            rews.to(self.master_device)
            dones.to(self.master_device)
            truncs.to(self.master_device)
            infos.to(self.master_device)
        return obss, rews, dones, truncs, infos

class A2C_Agent(nn.Module):
    def __init__(self, obs_space, act_space, args):
        self.obs_space = obs_space
        self.act_space = act_space
        self.args = args
        self.hidden_size = args.hidden_size
        self.device = args.device
        self.ac_master = MLPActorCritic(self.obs_space, self.act_space).to(self.device)
        with torch.no_grad():
            self.pi_workers = deepcopy(self.ac_master.pi)
        self.buffer_workers = A2C_Buffer(self.device)
        self.T = 0
        self.rewards_by_episode = []

    def step(self, obss, rews=0.0, dones=False, truncs=False, infos=None):
        actions, values, logp_actions = self.ac_master.step(obss)
        self.buffer_workers.add(obss, values, actions, logp_actions, rews, dones, truncs, infos)
        self.T += 1
        if self.T % args.num_steps_in_batch == 0:
            self.batch_update()
            with torch.no_grad():
                self.pi_workers.load_state_dict(self.ac_master.pi.state_dict())
        return actions, self.T

    def bootstrapped_returns(self):
        return

    def batch_update(self):
        returns = self.bootstrapped_returns()
        obss, rews, dones, truncs, infos = self.buffer_cpu.get()
        return

def train(args):
    envs = gym.vector.make(args.env_id, num_envs=args.n_env)
    agent = A2C_Agent(envs.single_observation_space, envs.single_action_space, args)
    actions = agent.step(lambda: envs.reset()[0])
    env_T = 0 # to remove
    while agent.T < args.training_steps:
        obss, rews, dones, truncs, infos = envs.step(actions)
        actions = agent.step(obss, rews, dones, truncs, infos)
        env_T += 1 # to remove
        assert agent.T == env_T # to remove
    rewards_by_episode = agent.rewards_by_episode
    return rewards_by_episode

def plot(ep_rews, args):
    import matplotlib.pyplot as plt
    plt.plot(ep_rews)
    plt.title(f"{args.env_id}, A2C, hidden_dim={args.hidden_size}, epochs={args.num_batches}")
    plt.xlabel("Episode")
    plt.ylabel("Total rewards")
    plt.show()


if __name__ == "__main__":
    args = get_args()
    start_time = time.time()
    rewards_by_episode = train(args)
    end_time = time.time()
    print(f"TOTAL TRAINING TIME: {end_time - start_time}")
    plot(rewards_by_episode, args)