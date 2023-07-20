import torch
from torch import nn
from torch.optim import Adam
from copy import deepcopy
from torch.distributions.normal import Normal
import numpy as np
import itertools
import gym

from gym.spaces import Box
from PG_base import MLPActorCritic_SAC, PG_OffPolicy_Buffer
from utils import get_args, printing, plot


class SAC_Agent(nn.Module):
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
        self.SAC_ac = MLPActorCritic_SAC(self.obs_space, self.act_space, hidden_sizes=self.h_sizes)
        with torch.no_grad():
            self.SAC_ac_target = deepcopy(self.SAC_ac)
        # Buffer+Rewards lists
        self.batch_size = args.num_steps_in_batch
        self.buffer_size = args.buffer_size
        self.buffer = PG_OffPolicy_Buffer(self.obs_dim, self.act_dim, self.batch_size, self.buffer_size)
        self.total_rews_by_ep = []
        # Training params
        self.alpha = args.alpha
        self.gamma = args.gamma
        self.tau = args.tau
        self.target_noise = args.target_noise
        self.noise_clip = args.noise_clip
        self.warmup_period = args.warmup_period
        self.update_period = args.update_period
        self.delayed_update_period = 1
        self.MseLoss = nn.MSELoss()
        self.last_Q_loss = 100000
        self.optimizerPi = Adam(self.SAC_ac.pi.parameters(), lr=args.learning_rate)
        self.q_params = itertools.chain(self.SAC_ac.q1.parameters(), self.SAC_ac.q2.parameters())
        self.optimizerQ = Adam(self.q_params, lr=args.learning_rate)
        # Step counters
        self.t = 0
        self.ep_t = 0
        self.ep = len(self.total_rews_by_ep)
        self.n_update = 0

    def action_from_obs(self, obs):
        obs_tensor = torch.from_numpy(obs)
        action, _ = self.SAC_ac.pi(obs_tensor, with_logprob=False)
        return action.item()

    def action_select(self, obss):
        if self.ep >= 5:
            with torch.no_grad():
                action = np.array([self.action_from_obs(obss)])
        else:
            action = self.act_space.sample()
        return action

    def soft_target_weight_update(self):
        Q_weights = self.SAC_ac.q1.state_dict()
        QTarget_weights = self.SAC_ac_target.q1.state_dict()
        for key in Q_weights:
            QTarget_weights[key] = QTarget_weights[key] * self.tau + Q_weights[key] * (1 - self.tau)
        self.SAC_ac_target.q1.load_state_dict(QTarget_weights)

        Q_weights = self.SAC_ac.q2.state_dict()
        QTarget_weights = self.SAC_ac_target.q2.state_dict()
        for key in Q_weights:
            QTarget_weights[key] = QTarget_weights[key] * self.tau + Q_weights[key] * (1 - self.tau)
        self.SAC_ac_target.q2.load_state_dict(QTarget_weights)

    def update(self):
        data = self.buffer.sample_batch()
        b_obs, b_acts, b_rews, b_terms, b_obs_2 = data['obs'], data['act'], data['rew'], data['term'], data['obs2']

        Q1 = self.SAC_ac.q1(b_obs, b_acts)
        Q2 = self.SAC_ac.q2(b_obs, b_acts)

        with torch.no_grad():
            b_acts_2, log_b_acts_2 = self.SAC_ac.pi(b_obs_2)
            Q1_pi_target = self.SAC_ac_target.q1(b_obs_2, b_acts_2)
            Q2_pi_target = self.SAC_ac_target.q2(b_obs_2, b_acts_2)
            Q_pi_target = torch.min(Q1_pi_target, Q2_pi_target)
            targets = b_rews + self.gamma * (1 - b_terms.int()) * (Q_pi_target - self.alpha * log_b_acts_2)

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
            b_act_pi, log_b_act_pi = self.SAC_ac.pi(b_obs)
            Q1_pi = self.SAC_ac.q1(b_obs, b_act_pi)
            Q2_pi = self.SAC_ac.q2(b_obs, b_act_pi)
            Q_pi = torch.min(Q1_pi, Q2_pi)
            PiLoss = (self.alpha * log_b_act_pi - Q_pi).mean()
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


def train(args):
    env = gym.make(args.env_id)
    agent = SAC_Agent(env.observation_space, env.action_space, env.action_space.high[0], args)
    actions, env_t = agent.step(env.reset()[0]), 0
    while agent.t < args.training_steps:
        obss, rews, terms, truncs, _ = env.step(actions)
        if (terms or truncs):
            obss, _ = env.reset()
        env_t += 1
        #print(f"env_t: {env_t}, agent.T: {agent.t}")
        assert agent.t == env_t  # to remove
        actions = agent.step(obss, rews, terms, truncs)
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
