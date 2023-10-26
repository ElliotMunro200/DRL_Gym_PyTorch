class Rectangle:
    def __init__(self, length, width):
        self.length = length
        self.width = width

    def area(self):
        return self.length * self.width

    def perimeter(self):
        return 2 * self.length + 2 * self.width

# Here we declare that the Square class inherits from the Rectangle class
class Square(Rectangle):
    pass
    #def __init__(self, length):
    #    super().__init__(length, length)

if __name__ == "__main__":
    sq = Square(4,5)
    print(dir(Rectangle))


def train(args, env, agent):
    obs, rews, done = env.reset()
    action = agent.step(obs)
    while env.step_count < args.training_steps:
        if not done:
            obs, rews, done, _ = env.step(action)
        action = agent.step(obs, rews, done)
        if done:
            obs, rews, done = env.reset()


def train(args, env, agent):
    env.timestep = 0
    while env.timestep < args.training_steps:
        # every env.step() increments env.step_count by 1, env.reset() does not increment the env.timestep.
        # means that the timestep is correct when reaching agent.step().
        if env.timestep % args.max_episode_steps == 0:
            obs, rews, done, _ = env.reset()
        else:
            obs, rews, done, _ = env.step(action)
        # agent can do an internal calculation of the time remaining in the episode
        #  - from the given env.max_episode_steps info and the env.timestep.
        # even if the state is terminal the agent makes choices and saves to the buffer within agent.step().
        # only on the first agent.step() of the training does the agent sore nothing in the buffer.
        action = agent.step(obs, rews, env.timestep)


# keeping done signal to accommodate early terminations for generality.
def train(args, env, agent):
    done = True
    env.t = -1
    while env.t < args.training_steps:
        # every env.reset() and env.step() is counted as a timestep and so increments env.step_count by 1.
        if done:
            obs, rews, done, _ = env.reset()
        elif not done:
            obs, rews, done, _ = env.step(action)
        # agent saves (s,r,d,g,a) timestep data to the buffer within agent.step() every time.
        # agent observes ep_t_left = obs[-1] and then calculates -> ep_t -> ep_num -> t.
        # at the end of the episode N_sum_prev_full_ep_lens is updated by ep_t.
        action = agent.step(obs, rews, done)


# 4 module controller architecture - executed by higher-level policy

from copy import deepcopy
import torch.nn as nn
class GTD3_Controller(nn.Module):
    def __init__(self, args):
        self.args = args
        # The AC module
        self.GTD3_ac = GTD3_ac(self.args)
        # The replay buffer module
        self.buffer = Buffer_GC_OffPolicy_LowLevel(self.args)
        # The WM module
        self.WM = WM_GC_LowLevel(self.args)

    def update(self):
        pass

    # Logic sub-controller - controls the whole Controller
    def step(self, obs, goal, rews, done):
        # update step counters

        # action selection (specifically action_from_obs, depends on off-policy algo type)
        acts = self.action_select(obs, goal)

        # buffer storage
        self.buffer.store(obs, rews, done, goal, acts)

        # possible update (specifically Q+pi Loss functions used, depends on off-policy algo type)
        self.update()

        # possible after-episode function (reward counting + counter iteration) + logging
        self.after_episode()


import matplotlib.pyplot as plt
from collections import deque
import random

# Create a fixed-length deque to store the data points
data_points = deque(maxlen=50)

# Create an empty plot
fig, ax = plt.subplots()
line, = ax.plot([])

# Set the x-axis and y-axis limits
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)

# Iterate through the data points and update the plot
for i in range(100):
    # Generate and add data points to the deque
    new_x = i
    new_y = random.randint(0, 100)
    data_points.append((new_x, new_y))

    # Update the plot with the new data points
    x_values = [x for x, y in data_points]
    y_values = [y for x, y in data_points]
    line.set_data(x_values, y_values)
    plt.pause(0.001)

    # Clear the plot for the next set of values
    line.set_data([], [])

# Show the plot
plt.show()