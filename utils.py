# get_args()
import torch
import torch.nn as nn

# printing()
from gym.spaces import Discrete, Box

# wandb_init()
import wandb

# make_env()
import gym
from envs import EnvWithGoal
from envs.create_maze_env import create_maze_env

# evaluate_policy()
import time
import numpy as np

def get_args():
    import argparse
    parser = argparse.ArgumentParser(description='DRL_Gym_PyTorch args')
    # General Arguments
    parser.add_argument('-env', '--env_id', type=str, default="CartPole-v1", help='The RL environment (default: CartPole-v1)')
    parser.add_argument('--algo', type=str, default="VPG", help='The RL agent (default: VPG)')
    parser.add_argument('-hs', '--hidden_sizes', nargs='+', type=int, default=[300, 300], help='The agent hidden size (default: 32)')
    parser.add_argument('--seed', type=int, default=1, help='Random seed (default: 1)')
    parser.add_argument('-t', '--training_steps', type=int, default=20000, help='# of total training steps (default: 10000)')
    parser.add_argument('-mes', '--max_ep_steps', type=int, default=299, help='# of total training steps (default: 10000)')
    parser.add_argument('-b', '--buffer_size', type=int, default=500, help='# off-policy buffer size (default: 1e6)')
    parser.add_argument('-ne', '--num_episodes', type=int, default=50, help='Number of episodes (default: 50)')
    parser.add_argument('-nb', '--num_batches', type=int, default=50, help='Number of batches/epochs (default: 50)')
    parser.add_argument('--num_eps_in_batch', type=int, default=4, help='# eps of warmup pre training (default: 4)')
    parser.add_argument('-up', '--update_period', type=int, default=10, help='# of steps per update (default: 10)')
    parser.add_argument('-bs', '--batch_size', type=int, default=50, help='# batch_size (default: 50)')
    parser.add_argument('--delayed_update_period', type=int, default=2, help='# of critic updates per target+policy updates (default: 2)')
    parser.add_argument('-tfe', '--training_first_ep_num', type=int, default=5, help='# episode number to start training on')
    parser.add_argument('--GAE', action='store_true', default=False, help='enables use of GAE advantage estimation')
    parser.add_argument('--alpha', type=float, default=0.2, help='alpha value (default: 0.2)')
    parser.add_argument('--gamma', type=float, default=0.99, help='gamma value (default: 0.99)')
    parser.add_argument('--lambda_', type=float, default=0.90, help='lambda value (default: 0.90)')
    parser.add_argument('--tau', type=float, default=0.005, help='tau value (default: 0.995)')
    parser.add_argument('-alr', '--actor_learning_rate', type=float, default=1e-3, help='alr value (default: 1e-3)')
    parser.add_argument('-clr', '--critic_learning_rate', type=float, default=1e-4, help='clr value (default: 1e-4)')
    parser.add_argument('--target_noise', type=float, default=1.0, help='noise added to target actions (default: 0.2)')
    parser.add_argument('--noise_clip', type=float, default=1.0, help='target action noise clip value (default: 0.5)')
    parser.add_argument('-sd', '--subgoal_dim', default=15, type=int)
    # Experiment Execution Arguments
    parser.add_argument('--train', action='store_true', default=False, help='trains the agent on the supplied env')
    parser.add_argument('--plot', action='store_true', default=False, help='plot the rewards gained in training')
    parser.add_argument('--eval', action='store_true', default=False, help='evaluates the agent on the supplied env')
    parser.add_argument('--render', action='store_true', default=False, help='renders eval episodes')
    parser.add_argument('--save_video', action='store_true', default=False, help='saves rendered eval episodes')
    parser.add_argument('--n_env', type=int, default=2, help='# number of parallel env processes (default: 2)')
    parser.add_argument('--cuda', action='store_true', default=False, help='enables CUDA training')
    parser.add_argument('--wandb', action='store_true', default=False, help='enables WandB experiment tracking')
    parser.add_argument('--wandb_project_name', type=str, default="DRL_Gym_PyTorch", help="the WandB's project name")
    parser.add_argument('--wandb_entity', type=str, default=None, help="the entity (team) of WandB's project")
    parser.add_argument('--wandb_resume', type=str, default="allow", help="Resume setting. auto, auto-resume without id given; allow, requires give previous run id or starts a new run; must, requires id and crashes if not the same as a previous run, ensuring resuming.")
    parser.add_argument('--wandb_id', type=str, default="new", help="use WandB auto id generation or user-provided id.")
    parser.add_argument('--checkpoint_model_file', type=str, default="model_checkpoint.pt", help="where to locally save checkpoint tarball")
    args = parser.parse_args()
    args.cuda = args.cuda and torch.cuda.is_available()
    args.device = 'cuda:0' if torch.cuda else 'cpu'
    return args

def printing(args, env):
    print(f"env.action_space: {env.action_space}")
    print(f"env.observation_space: {env.observation_space}")
    print(f"env.observation_space.high: {env.observation_space.high}")
    print(f"env.observation_space.low: {env.observation_space.low}")
    if isinstance(env.action_space, Discrete):
        print(f"action space Discrete shape: {env.action_space.shape}, num discrete actions: {env.action_space.n}")
    if isinstance(env.action_space, Box):
        print(f"action space Box shape: {env.action_space.shape[0]} | action space high: {env.action_space.high[0]}")
    print(f"env.observation_space.shape[0]: {env.observation_space.shape[0]}")
    print(f"env._max_episode_steps: {env._max_episode_steps}")
    run_name = f"{args.env_id}, {args.algo}, hidden_sizes={args.hidden_sizes}, training steps={args.training_steps}, batch_size={args.batch_size}"
    print(f"RUN_NAME: {run_name}")
    return run_name


def make_env(args):
    if args.env_id in ["AntMaze", "AntPush", "AntFall"]:
        env = EnvWithGoal(create_maze_env(args.env_id), args.env_id, eval=args.eval)
    else:
        env = gym.make(args.env_id)
    return env

def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)

class EnvTask:
    def __init__(self, args):
        self.args = args
        self.env = self.make_env()
        self._max_episode_steps = self.env._max_episode_steps
    def make_env(self): # env = gym.wrappers.RecordVideo(env, "/home/elliot/DRL_Gym_PyTorch/wandb/videos/")
        if self.args.env_id in ["AntMaze", "AntPush", "AntFall"]:
            from utils import EnvWithGoal
            from envs.create_maze_env import create_maze_env
            env = EnvWithGoal(create_maze_env(self.args.env_id), self.args.env_id, self.args.max_ep_steps, eval=self.args.eval)
        else:
            import gym
            env = gym.make(self.args.env_id)
        return env


class SubgoalActionSpace(object):
    def __init__(self, dim):
        # positions (and orientations) of ant torso and joints.
        limits = np.array([-10, -10, -0.5, -1, -1, -1, -1,
                    -0.5, -0.3, -0.5, -0.3, -0.5, -0.3, -0.5, -0.3])
        self.shape = (dim,1)
        self.low = limits[:dim]
        self.high = -self.low

    def sample(self):
        return (self.high - self.low) * np.random.sample(self.high.shape) + self.low

class Subgoal(object):
    def __init__(self, dim=15):
        self.action_space = SubgoalActionSpace(dim)
        self.action_dim = self.action_space.shape[0]

# Run evaluation of current agent policy for some number of episodes with option to render and/or save videos.
def evaluate_policy(args, env, agent, eval_episodes=10, sleep=-1):
    # wrapping env with automatic video saving of rendered video if wanting to save rendered videos.
    if args.save_video:
        from OpenGL import GL
        env = gym.wrappers.Monitor(env, directory='video',
                                   write_upon_reset=True, force=True, resume=True, mode='evaluation')
        render = False

    # setting up evaluation
    success = 0
    rewards = []
    # changing the training time final goal distribution (full square) to the evaluation goal distribution (0,16).
    env.evaluate = True
    # loop over episodes
    for e in range(eval_episodes):
        obs, r, d = env.reset()
        fg = obs['desired_goal']
        s = obs['observation']
        done = False
        reward_episode_sum = 0

        agent.fg = fg

        while not done:
            if args.render:
                env.render()
            if sleep > 0:
                time.sleep(sleep)

            a = agent.action_select(s, fg)
            n_s, r, done, _ = env.step(a)
            reward_episode_sum += r
            n_s = n_s['observation']
            s = n_s

        # If done, find whether episode success or fail in reaching final goal by calculating the Euclidean error.
        else:
            error = np.sqrt(np.sum(np.square(fg - s[:2])))
            print(f"Goal (x,y): ({fg[0]:.2f}, {fg[1]:.2f}) | End (x,y): ({s[0]:.2f}, {s[1]:.2f}) | Error: {error:.2f}")
            rewards.append(reward_episode_sum)
            success += 1 if error <= 1 else 0

    # reverting to the training time final goal distribution (full square) from the evaluation goal distribution (0,16).
    env.evaluate = False
    return np.round(np.array(rewards), 0), success / eval_episodes

def rewards_to_go(ep_rews):
    ep_rews_to_go = []
    for i in range(len(ep_rews)):
        ep_rews_to_go.append(sum(ep_rews[i:]))
    return ep_rews_to_go

def wandb_init(args):
    if args.wandb_id == "new":
        args.wandb_id = wandb.util.generate_id()
    run = wandb.init(
        project=args.wandb_project_name,
        entity=args.wandb_entity,
        group=args.algo,
        resume=args.wandb_resume,
        id=args.wandb_id,
        sync_tensorboard=True,
        config=vars(args),
        monitor_gym=True,
        save_code=True,
    )
    return run


def plot(ep_rews, exp_info):
    import matplotlib.pyplot as plt
    plt.plot(ep_rews)
    plt.title(exp_info)
    plt.ylabel("Total Rewards")
    plt.xlabel("Episode")
    plt.show()

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'