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
from gym.spaces import Discrete, Box
import time
import numpy as np
from copy import deepcopy
from utils import get_args, printing
from GAE import GAE
from AC_base import MLPActorCritic

class A2C_Buffer():
    def __init__(self, args, obs_dim, act_dim):
        self.args = args
        self.master_device = args.device
        self.n_env = self.args.n_env
        self.batch_steps = self.args.num_steps_in_batch
        self.GAE = self.args.GAE
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.obss = np.zeros((self.n_env, self.batch_steps, self.obs_dim))
        self.vals = np.zeros((self.n_env, self.batch_steps))
        self.acts = np.zeros((self.n_env, self.batch_steps, self.act_dim))
        self.rews = np.zeros((self.n_env, self.batch_steps))
        self.dones = np.zeros((self.n_env, self.batch_steps))
        self.truncs = np.zeros((self.n_env, self.batch_steps))
        self.rets = np.zeros((self.n_env, self.batch_steps))
        self.advs = np.zeros((self.n_env, self.batch_steps))

    def add(self, obss, vals, acts, rews, dones, truncs): #todo
        return

    def compute_rets(self):  # todo
        return

    def execute_GAE(self, batch_rews, state_values):  # todo
        ep_len = batch_rews.shape[1]  # GAE
        episode_GAE = GAE(self.n_env, ep_len, self.gamma, self.lambda_)  # GAE
        dones_np = np.zeros((1, ep_len - 1))  # GAE
        dones_np = np.append(dones_np, 1.0)  # GAE
        dones_np = np.expand_dims(dones_np, axis=0)  # GAE
        state_values_np = state_values.detach().numpy()  # GAE
        state_values_np = np.append(state_values_np, 0.0)  # GAE
        state_values_np = np.expand_dims(state_values_np, axis=0)  # GAE
        batch_rews_np = batch_rews.detach().numpy()  # GAE
        batch_rews_np = np.expand_dims(batch_rews_np, axis=0)  # GAE
        batch_advantages = episode_GAE(dones_np, batch_rews_np, state_values_np)  # GAE
        batch_advantages = torch.from_numpy(batch_advantages)  # GAE
        return batch_advantages

    def compute_advs(self): #todo
        if not self.GAE:
            self.advs = self.rets - self.vals
        elif self.GAE:
            self.advs = self.execute_GAE(self.rews, self.vals)

    def get(self):
        obss = torch.from_numpy(self.obss)
        acts = torch.from_numpy(self.acts)
        rets = torch.from_numpy(self.rets)
        advs = torch.from_numpy(self.advs)
        if self.master_device != "cpu":
            obss.to(self.master_device)
            acts.to(self.master_device)
            rets.to(self.master_device)
            advs.to(self.master_device)
        print(f"MASTER: {obss.device}, {self.master_device}")
        return obss, acts, rets, advs

class A2C_Agent(nn.Module):
    def __init__(self, obs_space, act_space, args):
        super().__init__()
        self.obs_space = obs_space
        self.obs_dim = self.obs_space.shape[0]
        self.act_space = act_space
        if isinstance(self.act_space, Discrete):
            self.act_dim = self.act_space.n
        elif isinstance(self.act_space, Box):
            self.act_dim = self.act_space.shape[0]
        self.args = args
        self.h_sizes = args.hidden_sizes
        self.device = args.device
        self.ac_master = MLPActorCritic(self.obs_space, self.act_space, hidden_sizes=(self.h_sizes))
        with torch.no_grad():
            self.ac_workers = deepcopy(self.ac_master)
        self.ac_master.to(self.device)
        self.n_env = args.n_env
        self.batch_steps = args.num_steps_in_batch
        self.buffer_workers = A2C_Buffer(self.args, self.obs_dim, self.act_dim)
        self.GAE = False
        if args.GAE:
            self.GAE = True
            self.gamma = 0.99
            self.lambda_ = 0.9
        self.T = 0
        self.rewards_by_episode = []  # todo

    def step(self, obss, rews=0.0, dones=False, truncs=False, infos=None):
        obss_tensor = torch.from_numpy(obss)
        actions, values, logp_actions = self.ac_workers.step(obss_tensor)
        self.buffer_workers.add(obss, values, actions, rews, dones, truncs) #todo
        self.T += 1
        if self.T % args.num_steps_in_batch == 0:
            self.batch_update()
            with torch.no_grad():
                self.ac_workers.pi.load_state_dict(self.ac_master.pi.state_dict())
                self.ac_workers.v.load_state_dict(self.ac_master.v.state_dict())
        return actions


    def batch_update(self):
        self.buffer_workers.compute_rets()
        self.buffer_workers.compute_advs()
        obss, acts, rets, advs = self.buffer_workers.get()
        print(f"DEVICES: {obss.device}, {acts.device}")
        _, logp_as = self.ac_master.pi(obss, acts)
        loss = -(logp_as * advs) + 0.5 * self.MseLoss(rets, self.ac_master.v(obss))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return

def train(args):
    envs = gym.vector.make(args.env_id, num_envs=args.n_env)
    run_name = printing(args, gym.make(args.env_id))
    agent = A2C_Agent(envs.single_observation_space, envs.single_action_space, args)
    actions = agent.step(envs.reset()[0])
    env_T = 0 # to remove
    while agent.T < args.training_steps:
        obss, rews, dones, truncs, infos = envs.step(actions)
        env_T += 1  # to remove
        print(f"agent.T: {agent.T}, env_T: {env_T}")
        assert agent.T == env_T  # to remove
        actions = agent.step(obss, rews, dones, truncs, infos)
    rewards_by_episode = agent.rewards_by_episode
    return rewards_by_episode

def plot(ep_rews, args):
    import matplotlib.pyplot as plt
    plt.plot(ep_rews)
    plt.title(f"{args.env_id}, A2C, hidden_dim={args.hidden_sizes}, epochs={args.num_batches}")
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