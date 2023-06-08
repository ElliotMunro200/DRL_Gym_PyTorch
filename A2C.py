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
from GAE_edited import GAE
from AC_base import combined_shape, MLPActorCritic

class A2C_Buffer():
    def __init__(self, args, obs_dim, act_dim):
        self.args = args
        self.master_device = args.device
        self.n_env = self.args.n_env
        self.batch_steps = self.args.num_steps_in_batch
        self.gamma = args.gamma
        self.GAE = self.args.GAE
        if self.GAE:
            self.lambda_ = args.lambda_
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.obss = np.zeros((self.n_env,) + combined_shape(self.batch_steps+1, self.obs_dim))
        self.vals = np.zeros((self.n_env, self.batch_steps+1))
        self.acts = np.zeros((self.n_env,) + combined_shape(self.batch_steps+1, self.act_dim))
        self.rews = np.zeros((self.n_env, self.batch_steps+1))
        self.dones = np.zeros_like(self.rews, dtype=bool)
        self.truncs = np.zeros((self.n_env, self.batch_steps+1))
        self.rets = np.zeros((self.n_env, self.batch_steps))
        self.advs = np.zeros((self.n_env, self.batch_steps))
        self.num_store = 0
        self.rews_by_ep = []
        self.rews_current_ep = np.zeros((self.n_env,))

    def refresh(self):
        # Putting last elements of last batch into first position of the new batch since we are still on that step
        obss_last = self.obss[:, -1, ...]
        vals_last = self.vals[:, -1]
        acts_last = self.acts[:, -1, ...]
        rews_last = self.rews[:, -1]
        dones_last = self.dones[:, -1]
        truncs_last = self.truncs[:, -1]
        self.obss = np.zeros((self.n_env,) + combined_shape(self.batch_steps+1, self.obs_dim))
        self.vals = np.zeros((self.n_env, self.batch_steps+1))
        self.acts = np.zeros((self.n_env,) + combined_shape(self.batch_steps+1, self.act_dim))
        self.rews = np.zeros((self.n_env, self.batch_steps+1))
        self.dones = np.zeros_like(self.rews, dtype=bool)
        self.truncs = np.zeros((self.n_env, self.batch_steps+1))
        self.rets = np.zeros((self.n_env, self.batch_steps))
        self.advs = np.zeros((self.n_env, self.batch_steps))
        self.obss[:, 0, ...] = obss_last
        self.vals[:, 0] = vals_last
        self.acts[:, 0, ...] = acts_last
        self.rews[:, 0] = rews_last
        self.dones[:, 0] = dones_last
        self.truncs[:, 0] = truncs_last
        self.num_store = 1

    def add(self, obss, vals, acts, rews, dones, truncs):
        self.obss[:, self.num_store] = obss
        self.vals[:, self.num_store] = vals
        self.acts[:, self.num_store] = acts
        self.rews[:, self.num_store] = rews
        self.dones[:, self.num_store] = dones
        self.truncs[:, self.num_store] = truncs
        self.num_store += 1
        return

    def tally_rewards(self):
        # For each worker
        ep_end_array = np.logical_or(self.dones, self.truncs)
        for w in range(self.n_env):
            # If no dones, add all rewards to current tallies
            if True not in ep_end_array[w, :]:
                self.rews_current_ep[w] += sum(self.rews[w, :])
            # Else, find where the dones are and add the rewards accordingly
            else:
                inds = np.where(ep_end_array[w, :] == True)[0]
                # Adding rewards to end the previously ongoing tally, and adding to episodic rewards list
                r_end_last = sum(self.rews[w, 0:inds[0]+1])
                self.rews_current_ep[w] += r_end_last
                self.rews_by_ep.append(self.rews_current_ep[w])
                # Making new ongoing tally from the last rewards in the batch
                r_last = sum(self.rews[w, inds[-1]+1:])
                self.rews_current_ep[w] = r_last
                # Adding full episodes in batch straight to episodic rewards list
                rew_tots = [sum(self.rews[w, inds[i]+1:inds[i+1]+1]) for i in range(len(inds)-1)]
                for ep_rew in rew_tots:
                    self.rews_by_ep.append(ep_rew)

    def compute_n_step_rets(self):
        dones = deepcopy(self.dones)
        rewards = deepcopy(self.rews)
        ret_t = np.array([0.0 if d else self.vals[i, -1] for i, d in enumerate(self.dones[:, -1])])
        for t in reversed(range(self.batch_steps)):
            mask = 1.0 - dones[:, t+1]
            ret_t = rewards[:, t+1] + self.gamma * mask * ret_t
            self.rets[:, t] = ret_t
        return

    def execute_GAE(self):
        batch_len = self.rews.shape[1]
        episode_GAE = GAE(self.n_env, batch_len, self.gamma, self.lambda_)
        dones = deepcopy(self.dones)
        state_values = deepcopy(self.vals)
        batch_rews = deepcopy(self.rews)
        batch_advantages = episode_GAE(dones, batch_rews, state_values)
        return batch_advantages

    def compute_advs(self):
        if not self.GAE:
            self.advs = self.rets - self.vals[:, :-1]
        elif self.GAE:
            self.advs = self.execute_GAE()

    def get(self):
        # converting to torch.float32 tensors
        obss = torch.from_numpy(self.obss[:, :-1]).type(torch.float32)
        acts = torch.from_numpy(self.acts[:, :-1]).type(torch.float32)
        rets = torch.from_numpy(self.rets).type(torch.float32)
        advs = torch.from_numpy(self.advs).type(torch.float32)
        # flattening
        obss = torch.flatten(obss, start_dim=0, end_dim=1)
        acts = torch.flatten(acts, start_dim=0, end_dim=1)
        rets = torch.flatten(rets, start_dim=0, end_dim=1)
        advs = torch.flatten(advs, start_dim=0, end_dim=1)
        # advantage normalization trick (for faster learning)
        advs_mean = torch.mean(advs)
        advs_std = torch.std(advs)
        advs = (advs - advs_mean) / advs_std
        # put on GPU if using it
        if self.master_device != "cpu":
            obss = obss.to(self.master_device)
            acts = acts.to(self.master_device)
            rets = rets.to(self.master_device)
            advs = advs.to(self.master_device)
        return obss, acts, rets, advs

class A2C_Agent(nn.Module):
    def __init__(self, obs_space, act_space, args):
        super().__init__()
        self.obs_space = obs_space
        self.obs_dim = self.obs_space.shape[0]
        self.act_space = act_space
        self.act_dim = self.act_space.shape
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
        self.optimizer = Adam(self.ac_master.parameters(), lr=1e-2)
        self.MseLoss = nn.MSELoss()
        self.T = 0
        self.rewards_by_episode = self.buffer_workers.rews_by_ep

    def step(self, obss, rews=np.array([0.0]), dones=np.array([False]), truncs=np.array([False]), infos=np.array([None])):
        obss_tensor = torch.from_numpy(obss)
        actions, values, logp_actions = self.ac_workers.step(obss_tensor)
        self.buffer_workers.add(obss, values, actions, rews, dones, truncs)
        self.T += 1
        if self.T % self.batch_steps == 0:
            self.batch_update()
            with torch.no_grad():
                self.ac_workers.pi.load_state_dict(self.ac_master.pi.state_dict())
                self.ac_workers.v.load_state_dict(self.ac_master.v.state_dict())
        return actions


    def batch_update(self):
        self.buffer_workers.compute_n_step_rets()
        self.buffer_workers.compute_advs()
        # .get() flattens out all it gets so the loss can be calculated as usual.
        # We can do this since each timestep transition is used as an independent sample.
        obss, acts, rets, advs = self.buffer_workers.get()
        _, logp_as = self.ac_master.pi(obss, acts)
        loss = -(logp_as * advs).mean() + 0.5 * self.MseLoss(rets, self.ac_master.v(obss))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.buffer_workers.tally_rewards()
        self.buffer_workers.refresh()
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
    plt.title(f"{args.env_id}, A2C, hidden_dim={args.hidden_sizes}, batch size={args.num_steps_in_batch}, GAE: {args.GAE}")
    plt.xlabel("Episode")
    plt.ylabel("Total rewards")
    plt.show()


if __name__ == "__main__":
    args = get_args()
    start_time = time.time()
    rewards_by_episode = train(args)
    end_time = time.time()
    print(f"TOTAL TRAINING TIME: {end_time - start_time:.2f}s")
    plot(rewards_by_episode, args)
