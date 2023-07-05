import numpy as np
import scipy.signal

import torch
import torch.nn as nn


def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)

def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
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
        return torch.squeeze(q, -1) # Critical to ensure q has right shape.


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
        return torch.squeeze(q, -1) # Critical to ensure q has right shape.


class MLP_GeneralActor(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation, act_limit, goal_dim=0):
        super().__init__()
        pi_sizes = [obs_dim + goal_dim] + list(hidden_sizes) + [act_dim]
        self.pi = mlp(pi_sizes, activation, nn.Tanh)
        self.act_limit = act_limit

    def forward(self, obs, goal=None):
        # input to the policy is left as obs by the concatenation if there is no goal supplied
        if goal is None:
            goal = torch.empty(obs.shape[:-1]+(0,))
        # Return output from network scaled to action space limits.
        return self.act_limit * self.pi(torch.cat([obs, goal], dim=-1))


class MLP_GeneralQFunction(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation, goal_dim=0):
        super().__init__()
        self.q = mlp([obs_dim + goal_dim + act_dim] + list(hidden_sizes) + [1], activation)

    def forward(self, obs, act, goal=None):
        if goal is None:
            goal = torch.empty(obs.shape[:-1]+(0,))
        q = self.q(torch.cat([obs, goal, act], dim=-1))
        return torch.squeeze(q, -1) # Critical to ensure q has right shape.


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

    def __init__(self, observation_space, action_space, hidden_sizes=(256,256),
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

    def __init__(self, observation_space, action_space, hidden_sizes=(256,256),
                 activation=nn.ReLU):
        super().__init__()

        obs_dim = observation_space.shape[0]
        goal_dim = obs_dim
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
        self.ptr = (self.ptr+1) % self.max_size
        self.curr_size = min(self.curr_size+1, self.max_size)

    def sample_batch(self):
        idxs = np.random.randint(0, self.curr_size-1, size=self.batch_size)
        batch = dict(obs=self.obs_buf[idxs],
                     obs2=self.obs_buf[idxs+1],
                     act=self.act_buf[idxs],
                     rew=self.rew_buf[idxs+1],
                     term=self.term_buf[idxs+1])
        return {k: torch.as_tensor(v, dtype=torch.float32) for k, v in batch.items()}