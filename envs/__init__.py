"""Random policy on an environment."""

import numpy as np
import argparse

import envs.create_maze_env


def get_goal_sample_fn(env_name, evaluate):
    if env_name == 'AntMaze':
        # NOTE: When evaluating (i.e. the metrics shown in the paper)
        # we use the commented out goal sampling function.    The uncommented
        # one is only used for training. TODO: check if evaluate variable should take value from args.eval or not
        if evaluate:
            return lambda: np.array([0., 16.]) #np.array([0., 16.])
        else:
            return lambda: np.array([0., 16.]) #np.random.uniform((-4, -4), (20, 20))
    elif env_name == 'AntPush':
        return lambda: np.array([0., 19.])
    elif env_name == 'AntFall':
        return lambda: np.array([0., 27., 4.5])
    else:
        assert False, 'Unknown env'


def get_reward_fn(env_name):
    if env_name == 'AntMaze':
        return lambda obs, goal: -np.sum(np.square(obs[:2] - goal)) ** 0.5
    elif env_name == 'AntPush':
        return lambda obs, goal: -np.sum(np.square(obs[:2] - goal)) ** 0.5
    elif env_name == 'AntFall':
        return lambda obs, goal: -np.sum(np.square(obs[:3] - goal)) ** 0.5
    else:
        assert False, 'Unknown env'


def success_fn(last_reward):
    return last_reward > -5.0


class EnvWithGoal(object):
    def __init__(self, base_env, env_name, max_ep_steps, eval=False):
        self.base_env = base_env

        self.env_name = env_name
        self.evaluate = eval
        self.reward_fn = get_reward_fn(env_name)
        self.goal = None
        self.distance_threshold = 5
        self.t = -1
        self.state_dim = self.base_env.observation_space.shape[0] + 1
        self.action_dim = self.base_env.action_space.shape[0]
        self._max_episode_steps = max_ep_steps
        #self.reward_range = (-1000.0, 1000.0)
        self.reward_range = self.base_env.reward_range
        self.metadata = self.base_env.metadata
        self.metadata = {'render.modes': ['human', 'rgb_array']}
        print(self.metadata)
        self.spec = self.base_env.spec

    def seed(self, seed):
        self.base_env.seed(seed)

    def reset(self):
        # self.viewer_setup()
        self.goal_sample_fn = get_goal_sample_fn(self.env_name, self.evaluate)
        obs = self.base_env.reset()
        self.t += 1
        self.time_rem = self._max_episode_steps
        self.goal = self.goal_sample_fn()
        next_obs = {
            # add timestep
            'observation': np.r_[obs.copy(), self.goal, self.time_rem],
            'achieved_goal': obs[:2],
            'desired_goal': self.goal,
        }
        return next_obs, 0.0, False

    # the step function of the currently used EnvWithGoal class.
    # concats the base_env obs with time_rem to get time-gnostic states as necessary for finite-horizon MDPs.
    def step(self, a):
        obs, _, done, info = self.base_env.step(a)
        reward = self.reward_fn(obs, self.goal)
        self.t += 1
        self.time_rem -= 1
        next_obs = {
            # add timestep
            'observation': np.r_[obs.copy(), self.goal, self.time_rem],
            'achieved_goal': obs[:2],
            'desired_goal': self.goal,
        }
        return next_obs, reward, (done or self.time_rem == 0), info

    def render(self, mode="human"):
        self.base_env.render(mode=mode)

    def get_image(self):
        self.render()
        data = self.base_env.viewer.get_image()

        img_data = data[0]
        width = data[1]
        height = data[2]

        tmp = np.fromstring(img_data, dtype=np.uint8)
        image_obs = np.reshape(tmp, [height, width, 3])
        image_obs = np.flipud(image_obs)

        return image_obs

    @property
    def action_space(self):
        return self.base_env.action_space

    @property
    def observation_space(self):
        return self.base_env.observation_space

def run_environment(env_name, episode_length, num_episodes):
    env = EnvWithGoal(
            create_maze_env.create_maze_env(env_name),
            env_name)

    def action_fn(obs):
        action_space = env.action_space
        action_space_mean = (action_space.low + action_space.high) / 2.0
        action_space_magn = (action_space.high - action_space.low) / 2.0
        random_action = (action_space_mean +
            action_space_magn *
            np.random.uniform(low=-1.0, high=1.0,
            size=action_space.shape))

        return random_action

    rewards = []
    successes = []
    for ep in range(num_episodes):
        rewards.append(0.0)
        successes.append(False)
        obs = env.reset()
        for _ in range(episode_length):
            env.render()
            print(env.get_image().shape)
            obs, reward, done, _ = env.step(action_fn(obs))
            rewards[-1] += reward
            successes[-1] = success_fn(reward)
            if done:
                break
        
        print('Episode {} reward: {}, Success: {}'.format(ep + 1, rewards[-1], successes[-1]))

    print('Average Reward over {} episodes: {}'.format(num_episodes, np.mean(rewards)))
    print('Average Success over {} episodes: {}'.format(num_episodes, np.mean(successes)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", default="AntEnv", type=str)               
    parser.add_argument("--episode_length", default=500, type=int)      
    parser.add_argument("--num_episodes", default=100, type=int)

    args = parser.parse_args()
    run_environment(args.env_name, args.episode_length, args.num_episodes)