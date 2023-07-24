import torch
from torch import nn
from torch.optim import Adam
from copy import deepcopy
from torch.distributions.normal import Normal
import numpy as np
import random
import gym

from gym.spaces import Box
from PG_base import MLPActorCritic_DDPG, PG_OffPolicy_Buffer
from utils import get_args, printing, plot


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
        # Buffer+Rewards lists
        self.batch_size = args.num_steps_in_batch
        self.buffer_size = args.buffer_size
        self.buffer = PG_OffPolicy_Buffer(self.obs_dim, self.act_dim, self.batch_size, self.buffer_size)
        self.total_rews_by_ep = []
        # Training params
        self.gamma = args.gamma
        self.tau = args.tau
        self.warmup_period = args.warmup_period
        self.update_period = args.update_period
        self.MseLoss = nn.MSELoss()
        self.last_loss = 100000
        self.optimizer = Adam(self.DDPG_ac.parameters(), lr=args.learning_rate)
        self.t = 0
        self.ep_t = 0
        self.ep = len(self.total_rews_by_ep)

    def action_from_obs(self, obs):
        obs_tensor = torch.from_numpy(obs).type(torch.float32)
        mean = self.DDPG_ac.pi(obs_tensor).squeeze()
        noise = Normal(torch.tensor([0.0]), torch.tensor([1.0])).sample()
        action = torch.clip(mean+noise*0.1, -2.0, 2.0).squeeze()
        return action

    def action_select(self, obss):
        if self.ep >= 1:
            with torch.no_grad():
                action = self.action_from_obs(obss)
                if action.ndim == 0:  # for MCC-v0 with act-dim of 1 (comes from dim error in action_from_obs)
                    action = action.unsqueeze(dim=0)
                action = action.numpy()
        else:
            action = self.act_space.sample()
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

    def update(self):
        data = self.buffer.sample_batch()
        b_obs, b_acts, b_rews, b_terms, b_next_obs = data['obs'], data['act'], data['rew'], data['term'], data['obs2']

        Q_from_target = self.DDPG_ac_target.q1(b_next_obs, self.DDPG_ac_target.pi(b_next_obs))
        targets = b_rews + self.gamma * (1 - b_terms.int()) * Q_from_target
        Qvals = self.DDPG_ac.q1(b_obs, b_acts) #.unsqueeze(dim=1)

        self.optimizer.zero_grad()
        loss = self.MseLoss(Qvals, targets) + -1 * self.DDPG_ac.q1(b_obs, self.DDPG_ac.pi(b_obs)).mean()
        loss.backward()
        self.optimizer.step()
        self.last_loss = loss

        with torch.no_grad():
            self.soft_target_weight_update()

        return loss

    def after_episode(self):
        bcs = self.buffer.curr_size
        ep_rews = self.buffer.rew_buf[(bcs-self.ep_t):bcs]
        ep_rew = sum(ep_rews)
        self.total_rews_by_ep.append(ep_rew)
        self.ep = len(self.total_rews_by_ep)
        print(f"| Episode {self.ep:<3} done | Steps: {self.ep_t:<3} | Rewards: {ep_rew:<4.1f} | Last Loss: {self.last_loss:<4.1f} |")
        self.ep_t = 0

    def step(self, obss, rews=0.0, terms=False, trunc=False):
        acts = self.action_select(obss)
        self.buffer.store(obss, acts, rews, terms)
        self.t += 1
        self.ep_t += 1
        # only update if buffer is big enough and time to update
        if (self.buffer.curr_size >= self.warmup_period) and (self.t % self.update_period == 0):
            # one update for each timestep since last updating
            for _ in range(self.update_period):
                loss = self.update()
        if (terms or trunc):
            self.after_episode()
        return acts


def train(args):
    env = gym.make(args.env_id)
    agent = DDPG_Agent(env.observation_space, env.action_space, args)
    action, env_t = agent.step(env.reset()[0]), 0
    while agent.t < args.training_steps:
        obss, rews, terms, truncs, _ = env.step(action)
        if (terms or truncs):
            obss, _ = env.reset()
        env_t += 1
        #print(f"env_t: {env_t}, agent.T: {agent.t}")
        assert agent.t == env_t  # to remove
        action = agent.step(obss, rews, terms, truncs)
    total_rews_by_ep = agent.total_rews_by_ep
    return total_rews_by_ep


if __name__ == "__main__":
    import time
    args = get_args()
    exp_info = printing(args, gym.make(args.env_id))
    start_time = time.time()
    ep_rews = train(args)
    end_time = time.time()
    print(f"TOTAL TRAINING TIME: {end_time - start_time:.2f}s")
    plot(ep_rews, exp_info)
