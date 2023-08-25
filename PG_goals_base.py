import numpy as np
import scipy.signal

import time
import torch
import torch.nn as nn
from copy import deepcopy


def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)


def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j + 1]), act()]
    return nn.Sequential(*layers)


def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])


class MLPActor(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation, act_limit):
        super().__init__()
        pi_sizes = [obs_dim] + list(hidden_sizes) + [act_dim]
        self.pi = mlp(pi_sizes, activation, nn.Tanh)
        self.act_limit = act_limit

    def forward(self, obs):
        # Return output from network scaled to action space limits.
        return self.act_limit * self.pi(obs)


class MLPQFunction(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        self.q = mlp([obs_dim + act_dim] + list(hidden_sizes) + [1], activation)

    def forward(self, obs, act):
        q = self.q(torch.cat([obs, act], dim=-1))
        return torch.squeeze(q, -1)  # Critical to ensure q has right shape.


class MLPActorCritic_DDPG(nn.Module):

    def __init__(self, observation_space, action_space, hidden_sizes=(256, 256),
                 activation=nn.ReLU):
        super().__init__()

        obs_dim = observation_space.shape[0]
        act_dim = action_space.shape[0]
        act_limit = action_space.high[0]

        # build policy and value functions
        self.pi = MLPActor(obs_dim, act_dim, hidden_sizes, activation, act_limit)
        self.q1 = MLPQFunction(obs_dim, act_dim, hidden_sizes, activation)

    def act(self, obs):
        with torch.no_grad():
            return self.pi(obs).numpy()


class MLPActorCritic_TD3(nn.Module):

    def __init__(self, observation_space, action_space, hidden_sizes=(256, 256),
                 activation=nn.ReLU):
        super().__init__()

        obs_dim = observation_space.shape[0]
        act_dim = action_space.shape[0]
        act_limit = action_space.high[0]

        # build policy and value functions
        self.pi = MLPActor(obs_dim, act_dim, hidden_sizes, activation, act_limit)
        self.q1 = MLPQFunction(obs_dim, act_dim, hidden_sizes, activation)
        self.q2 = MLPQFunction(obs_dim, act_dim, hidden_sizes, activation)

    def act(self, obs):
        with torch.no_grad():
            return self.pi(obs).numpy()


class MLP_GoalActorCritic_TD3(nn.Module):

    def __init__(self, obs_dim, goal_dim, action_space, hidden_sizes=(256, 256),
                 activation=nn.ReLU):
        super().__init__()

        act_dim = action_space.shape[0]
        act_limit = action_space.high[0]

        # build policy and value functions
        self.pi = MLP_GoalActor(obs_dim, goal_dim, act_dim, hidden_sizes, activation, act_limit)
        self.q1 = MLP_GoalQFunction(obs_dim, goal_dim, act_dim, hidden_sizes, activation)
        self.q2 = MLP_GoalQFunction(obs_dim, goal_dim, act_dim, hidden_sizes, activation)

    def act(self, obs, goal):
        with torch.no_grad():
            return self.pi(obs, goal).numpy()


class DDPG_Buffer():
    def __init__(self, obs_dim, act_dim, batch_size, buffer_size):
        self.obs_buf = np.zeros(combined_shape(buffer_size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(combined_shape(buffer_size, act_dim), dtype=np.float32)
        self.rew_buf = np.zeros(buffer_size, dtype=np.float32)
        self.term_buf = np.zeros(buffer_size, dtype=np.float32)
        self.ptr, self.curr_size = 0, 0
        self.batch_size, self.max_size = batch_size, buffer_size

    def store(self, obs, act, rew, term):
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.term_buf[self.ptr] = term
        self.ptr = (self.ptr + 1) % self.max_size
        self.curr_size = min(self.curr_size + 1, self.max_size)

    def sample_batch(self):
        idxs = np.random.randint(0, self.curr_size - 1, size=self.batch_size)
        batch = dict(obs=self.obs_buf[idxs],
                     obs2=self.obs_buf[idxs + 1],
                     act=self.act_buf[idxs],
                     rew=self.rew_buf[idxs + 1],
                     term=self.term_buf[idxs + 1])
        return {k: torch.as_tensor(v, dtype=torch.float32) for k, v in batch.items()}

# default max buffer size is 1,000,000.
class PG_Goal_OffPolicy_Buffer():
    def __init__(self, obs_dim, subg_dim, act_dim, batch_size, buffer_size, update_s_term):
        self.obs_buf = np.zeros(combined_shape(buffer_size, obs_dim), dtype=np.float32)
        self.rew_buf = np.zeros(buffer_size, dtype=np.float32)
        self.done_buf = np.zeros(buffer_size, dtype=np.float32)
        self.subg_buf = np.zeros(combined_shape(buffer_size, subg_dim), dtype=np.float32)
        self.act_buf = np.zeros(combined_shape(buffer_size, act_dim), dtype=np.float32)
        self.ptr, self.curr_size = 0, 0
        self.batch_size, self.max_size = batch_size, buffer_size
        self.update_s_term = update_s_term

    def store(self, obs, rew, done, subg, act):
        self.obs_buf[self.ptr] = obs
        self.rew_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.subg_buf[self.ptr] = subg
        self.act_buf[self.ptr] = act
        self.ptr = (self.ptr + 1) % self.max_size
        self.curr_size = min(self.curr_size + 1, self.max_size)

    # TODO: test also updating the termination timesteps manually.
    def sample_batch(self):
        # all indexes
        indxs = np.arange(0, self.curr_size - 1)

        # if not wanting to manually update Q(s_terminal,.), apply dones mask to indexes before sampling.
        if not self.update_s_term:
            # find where dones are in the buffer and make a mask
            done_indxs = np.where(self.done_buf)
            done_mask = np.isin(indxs, done_indxs)
            # apply mask to full index array to remove dones from available first steps in the transition
            indxs = indxs[~done_mask]

        # final available timestep indexes
        indxs = np.random.choice(indxs, size=self.batch_size)
        timestep_1 = dict(obs=self.obs_buf[indxs],
                          rew=self.rew_buf[indxs],
                          done=self.done_buf[indxs],
                          subg=self.subg_buf[indxs],
                          act=self.act_buf[indxs])

        # if updating Q(s_term,a) manually, then for every True in d, d' must also be True for target values to be = 0.
        done_next = deepcopy(self.done_buf[indxs + 1])
        if self.update_s_term:
            sample_done_indxs = np.where(self.done_buf[indxs])
            done_next[sample_done_indxs] = 1.

        timestep_2 = dict(obs2=self.obs_buf[indxs + 1],
                          rew2=self.rew_buf[indxs + 1],
                          done2=done_next,
                          subg2=self.subg_buf[indxs + 1],
                          act2=self.act_buf[indxs + 1])

        timestep_double = timestep_1.copy()
        timestep_double.update(timestep_2)
        return {k: torch.as_tensor(v, dtype=torch.float32) for k, v in timestep_double.items()}


class MLP_GoalActor(nn.Module):

    def __init__(self, obs_dim, goal_dim, act_dim, hidden_sizes, activation, act_limit):
        super().__init__()
        pi_sizes = [obs_dim + goal_dim] + list(hidden_sizes) + [act_dim]
        self.pi = mlp(pi_sizes, activation, nn.Tanh)
        self.act_limit = act_limit

    def forward(self, obs, goal):
        # Return output from network scaled to action space limits.
        return self.act_limit * self.pi(torch.cat([obs, goal], dim=-1))


class MLP_GoalQFunction(nn.Module):

    def __init__(self, obs_dim, goal_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        self.q = mlp([obs_dim + goal_dim + act_dim] + list(hidden_sizes) + [1], activation)

    def forward(self, obs, goal, act):
        q = self.q(torch.cat([obs, goal, act], dim=-1))
        return torch.squeeze(q, -1)  # Critical to ensure q has right shape.


class MLP_GeneralActor(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation, act_limit, goal_dim=0):
        super().__init__()
        pi_sizes = [obs_dim + goal_dim] + list(hidden_sizes) + [act_dim]
        self.pi = mlp(pi_sizes, activation, nn.Tanh)
        self.act_limit = act_limit

    def forward(self, obs, goal=None):
        # input to the policy is left as obs by the concatenation if there is no goal supplied
        if goal is None:
            goal = torch.empty(obs.shape[:-1] + (0,))
        # Return output from network scaled to action space limits.
        return self.act_limit * self.pi(torch.cat([obs, goal], dim=-1))


class MLP_GeneralQFunction(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation, goal_dim=0):
        super().__init__()
        self.q = mlp([obs_dim + goal_dim + act_dim] + list(hidden_sizes) + [1], activation)

    def forward(self, obs, act, goal=None):
        if goal is None:
            goal = torch.empty(obs.shape[:-1] + (0,))
        q = self.q(torch.cat([obs, goal, act], dim=-1))
        return torch.squeeze(q, -1)  # Critical to ensure q has right shape.
