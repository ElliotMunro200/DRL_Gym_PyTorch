import torch
import torch.nn as nn

def get_args():
    import argparse
    parser = argparse.ArgumentParser(description='DRL_Gym_PyTorch args')
    # General Arguments
    parser.add_argument('-env', '--env_id', type=str, default="CartPole-v1", help='The RL environment (default: CartPole-v1)')
    parser.add_argument('--algo', type=str, default="VPG", help='The RL agent (default: VPG)')
    parser.add_argument('-hs', '--hidden_sizes', nargs='+', type=int, default=[32, 32], help='The agent hidden size (default: 32)')
    parser.add_argument('--seed', type=int, default=1, help='Random seed (default: 1)')
    parser.add_argument('-ne', '--num_episodes', type=int, default=50, help='Number of episodes (default: 50)')
    parser.add_argument('-nb', '--num_batches', type=int, default=50, help='Number of batches/epochs (default: 50)')
    parser.add_argument('--num_eps_in_batch', type=int, default=4, help='# eps of warmup pre training (default: 4)')
    parser.add_argument('-nsb', '--num_steps_in_batch', type=int, default=50, help='# of steps before updating (default: 50)')
    parser.add_argument('-t', '--training_steps', type=int, default=20000, help='# of total training steps (default: 10000)')
    parser.add_argument('--buffer_size', type=int, default=1000000, help='# off-policy buffer size (default: 1e6)')
    parser.add_argument('--update_period', type=int, default=10, help='# of steps per update (default: 10)')
    parser.add_argument('--delayed_update_period', type=int, default=2, help='# of critic updates per target+policy updates (default: 2)')
    parser.add_argument('--warmup_period', type=int, default=500, help='# of steps before first update (default: 500)')
    parser.add_argument('--GAE', action='store_true', default=False, help='enables use of GAE advantage estimation')
    parser.add_argument('--gamma', type=float, default=0.99, help='gamma value (default: 0.99)')
    parser.add_argument('--lambda_', type=float, default=0.90, help='lambda value (default: 0.90)')
    parser.add_argument('--tau', type=float, default=0.995, help='tau value (default: 0.995)')
    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-3, help='optimizer lr value (default: 1e-3)')
    parser.add_argument('--target_noise', type=float, default=0.2, help='noise added to target actions (default: 0.2)')
    parser.add_argument('--noise_clip', type=float, default=0.5, help='target action noise clip value (default: 0.5)')
    parser.add_argument('--subgoal_dim', default=15, type=int)
    # Experiment Execution Arguments
    parser.add_argument('--train', action='store_true', default=False, help='trains the agent on the supplied env')
    parser.add_argument('--eval', action='store_true', default=False, help='evaluates the agent on the supplied env')
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


class EnvTask:
    def __init__(self, args):
        self.args = args
        self.env = self.make_env()
    def make_env(self): # env = gym.wrappers.RecordVideo(env, "/home/elliot/DRL_Gym_PyTorch/wandb/videos/")
        if self.args.env_id in ["AntMaze", "AntPush", "AntFall"]:
            from utils import EnvWithGoal
            from envs.create_maze_env import create_maze_env
            env = EnvWithGoal(create_maze_env(self.args.env_id), self.args.env_id, eval=self.args.eval)
        else:
            import gym
            env = gym.make(self.args.env_id)
        return env


class Agent(nn.Module):
    def __init__(self, args, envtask):
        self.args = args
        self.envtask = envtask

        from utils import build_network
        from utils import algo_update
        self.policy = build_network(self.args)
        if args.off_policy:
            from utils import Replay_Buffer
            self.rb = Replay_Buffer(self.args)
        if args.log:
            from utils import WandB_Logger
            self.log = WandB_Logger(self.args)

    def step(self):
        return 5

    def train(self):
        data = self.envtask.env.reset()
        for t in range(args.training_steps):
            action = self.step(data)
            data = self.envtask.env.step(action)
        return 5

    def evaluate(self):
        return 5


if __name__ == "__main__":
    args = get_args()
    envtask = EnvTask(args)
    agent = Agent(args, envtask)
    _ = agent.train()
    _ = agent.evaluate()
