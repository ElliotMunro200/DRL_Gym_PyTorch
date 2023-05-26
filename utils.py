import torch
from gym.spaces import Discrete, Box
import wandb

def get_args():
    import argparse
    parser = argparse.ArgumentParser(description='DRL_Gym_PyTorch args')
    # General Arguments
    parser.add_argument('--env_id', type=str, default="CartPole-v1", help='The RL environment (default: CartPole-v1)')
    parser.add_argument('--algo', type=str, default="VPG", help='The RL agent (default: VPG)')
    parser.add_argument('-hs', '--hidden_size', type=int, default=32, help='The agent hidden size (default: 32)')
    parser.add_argument('--seed', type=int, default=1, help='Random seed (default: 1)')
    parser.add_argument('-nb', '--num_batches', type=int, default=50, help='Number of batches/epochs (default: 50)')
    parser.add_argument('--num_eps_in_batch', type=int, default=4, help='# eps of warmup pre training (default: 4)')
    parser.add_argument('--num_steps_in_batch', type=int, default=100, help='# number of steps (default: 100)')
    parser.add_argument('--training_steps', type=int, default=10000, help='# number of total steps (default: 10000)')
    parser.add_argument('--GAE', action='store_true', default=False, help='enables use of GAE advantage estimation')
    # Experiment Execution Arguments
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
        print(f"env Discrete shape: {env.action_space.n}")
    if isinstance(env.action_space, Box):
        print(f"env Box shape: {env.action_space.shape[0]}")
    print(f"env.observation_space.shape[0]: {env.observation_space.shape[0]}")
    print(f"env._max_episode_steps: {env._max_episode_steps}")
    run_name = f"{args.env_id}, {args.algo}, hidden_size={args.hidden_size}, num_batches={args.num_batches}, batch_size={args.batch_size}"
    print(f"RUN_NAME: {run_name}")
    return run_name

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

def rewards_to_go(ep_rews):
    ep_rews_to_go = []
    for i in range(len(ep_rews)):
        ep_rews_to_go.append(sum(ep_rews[i:]))
    return ep_rews_to_go