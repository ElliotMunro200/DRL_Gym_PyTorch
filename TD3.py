import torch
from torch import nn
from torch.optim import Adam
from copy import deepcopy
from torch.distributions.normal import Normal
import numpy as np
import itertools
import gym

from gym.spaces import Box
from PG_base import MLPActorCritic_TD3, PG_OffPolicy_Buffer
from utils import get_args, printing, plot, EnvTask


class TD3_Agent(nn.Module):
    def __init__(self, obs_space, act_space, act_limit, args):
        super().__init__()
        self.obs_space = obs_space
        self.obs_dim = self.obs_space.shape[0]
        self.act_space = act_space
        assert isinstance(act_space, Box)
        self.act_dim = self.act_space.shape[0]
        self.act_limit = act_limit
        self.args = args
        self.h_sizes = self.args.hidden_sizes
        self.TD3_ac = MLPActorCritic_TD3(self.obs_space, self.act_space, hidden_sizes=self.h_sizes)
        with torch.no_grad():
            self.TD3_ac_target = deepcopy(self.TD3_ac)
        # Buffer+Rewards lists
        self.batch_size = args.num_steps_in_batch
        self.buffer_size = args.buffer_size
        self.buffer = PG_OffPolicy_Buffer(self.obs_dim, self.act_dim, self.batch_size, self.buffer_size)
        self.total_rews_by_ep = []
        # Training params
        self.gamma = args.gamma
        self.tau = args.tau
        self.target_noise = args.target_noise
        self.noise_clip = args.noise_clip
        self.warmup_period = args.warmup_period
        self.update_period = args.update_period
        self.delayed_update_period = args.delayed_update_period
        self.MseLoss = nn.MSELoss()
        self.last_Q_loss = 100000
        self.optimizerPi = Adam(self.TD3_ac.pi.parameters(), lr=args.learning_rate)
        self.q_params = itertools.chain(self.TD3_ac.q1.parameters(), self.TD3_ac.q2.parameters())
        self.optimizerQ = Adam(self.q_params, lr=args.learning_rate)
        # Step counters
        self.t = 0
        self.ep_t = 0
        self.ep = len(self.total_rews_by_ep)
        self.n_update = 0

    def action_from_obs(self, obs):
        obs_tensor = torch.from_numpy(obs).type(torch.float32)
        mean = self.TD3_ac.pi(obs_tensor).squeeze()
        noise = Normal(torch.tensor([0.0]), torch.tensor([1.0])).sample()
        action = torch.clip(mean+noise*0.1, -2.0, 2.0).squeeze()
        return action

    def action_select(self, obss):
        if self.ep >= 5:
            with torch.no_grad():
                action = self.action_from_obs(obss)
                if action.ndim == 0: # for MCC-v0 with act-dim of 1 (comes from dim error in action_from_obs)
                   action = action.unsqueeze(dim=0)
                action = action.numpy()
        else:
            action = self.act_space.sample()
        return action

    def soft_target_weight_update(self):
        Q_weights = self.TD3_ac.q1.state_dict()
        QTarget_weights = self.TD3_ac_target.q1.state_dict()
        for key in Q_weights:
            QTarget_weights[key] = QTarget_weights[key] * self.tau + Q_weights[key] * (1 - self.tau)
        self.TD3_ac_target.q1.load_state_dict(QTarget_weights)

        Q_weights = self.TD3_ac.q2.state_dict()
        QTarget_weights = self.TD3_ac_target.q2.state_dict()
        for key in Q_weights:
            QTarget_weights[key] = QTarget_weights[key] * self.tau + Q_weights[key] * (1 - self.tau)
        self.TD3_ac_target.q2.load_state_dict(QTarget_weights)

        Pi_weights = self.TD3_ac.pi.state_dict()
        PiTarget_weights = self.TD3_ac_target.pi.state_dict()
        for key in Pi_weights:
            PiTarget_weights[key] = PiTarget_weights[key] * self.tau + Pi_weights[key] * (1 - self.tau)
        self.TD3_ac_target.pi.load_state_dict(PiTarget_weights)

    def update(self):
        data = self.buffer.sample_batch()
        b_obs, b_acts, b_rews, b_terms, b_obs_2 = data['obs'], data['act'], data['rew'], data['term'], data['obs2']

        Q1 = self.TD3_ac.q1(b_obs, b_acts)
        Q2 = self.TD3_ac.q2(b_obs, b_acts)

        with torch.no_grad():
            Pi_targ = self.TD3_ac_target.pi(b_obs_2)
            epsilon = torch.randn_like(Pi_targ) * self.target_noise
            epsilon = torch.clamp(epsilon, -self.noise_clip, self.noise_clip)
            b_acts_2 = Pi_targ + epsilon
            b_acts_2 = torch.clamp(b_acts_2, -self.act_limit, self.act_limit)
            Q1_pi_target = self.TD3_ac_target.q1(b_obs_2, b_acts_2)
            Q2_pi_target = self.TD3_ac_target.q2(b_obs_2, b_acts_2)
            Q_pi_target = torch.min(Q1_pi_target, Q2_pi_target)
            targets = b_rews + self.gamma * (1 - b_terms.int()) * Q_pi_target

        Q1Loss = self.MseLoss(Q1, targets)
        Q2Loss = self.MseLoss(Q2, targets)
        QLoss = Q1Loss + Q2Loss

        self.optimizerQ.zero_grad()
        QLoss.backward()
        self.optimizerQ.step()

        self.last_Q_loss = QLoss

        if self.n_update % self.delayed_update_period == 0:

            for p in self.q_params:
                p.requires_grad = False

            self.optimizerPi.zero_grad()
            PiLoss = -1 * self.TD3_ac.q1(b_obs, self.TD3_ac.pi(b_obs)).mean()
            PiLoss.backward()
            self.optimizerPi.step()
            with torch.no_grad():
                self.soft_target_weight_update()

            for p in self.q_params:
                p.requires_grad = True

        self.n_update += 1

        return QLoss

    def after_episode(self):
        bcs = self.buffer.curr_size
        ep_rews = self.buffer.rew_buf[(bcs-self.ep_t):bcs]
        ep_rew = sum(ep_rews)
        self.total_rews_by_ep.append(ep_rew)
        self.ep = len(self.total_rews_by_ep)
        print(f"| Episode {self.ep:<3} done | Steps: {self.ep_t:<3} | Rewards: {ep_rew:<4.1f} | Last QLoss: {self.last_Q_loss:<4.1f} |")
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
                QLoss = self.update()
        if (terms or trunc):
            self.after_episode()
        return acts


def train(args, env, agent):
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
    envtask = EnvTask(args)
    env = envtask.env
    agent = TD3_Agent(env.observation_space, env.action_space, env.action_space.high[0], args)
    exp_info = printing(args, env)
    start_time = time.time()
    ep_rews = train(args, env, agent)
    end_time = time.time()
    print(f"TOTAL TRAINING TIME: {end_time - start_time:.2f}s")
    plot(ep_rews, exp_info)
