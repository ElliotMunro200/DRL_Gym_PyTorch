import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
import torch.optim as optim
import gym
import numpy as np

# repeat for number of updates:
# collect experience until buffer is full: -
# init buffer lists -
# obs = env.reset() -
# action = agent.action_select() - via stochastic policy
# obs, rew, done = env.step(action) -
# add (obs, rew, done) to buffer lists -
# when buffer is full execute VPG update: -

class VPG_Agent(nn.Module):
    def __init__(self, obs_dim, action_space, hidden_dim):
        super(VPG_Agent, self).__init__()
        self.obs_dim = obs_dim
        self.action_space = action_space
        self.act_dim = self.action_space.n
        self.hidden_dim = hidden_dim
        self.fc1 = nn.Linear(self.obs_dim, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, self.act_dim)
        self.logits_net = nn.Sequential(
            nn.Linear(self.obs_dim, self.hidden_dim),
            nn.Tanh(),
            nn.Linear(self.hidden_dim, self.act_dim)
        )
        self.optimizer = optim.Adam(self.logits_net.parameters(), lr=1e-2)

    def get_policy(self, obs):
        obs = obs.to(torch.float32)
        logits = self.logits_net(obs)
        policy = Categorical(logits=logits)
        return policy

    def action_select(self, obs):
        obs = torch.tensor(obs)
        action = self.get_policy(obs).sample().item()
        return action

    def loss(self, batch_obs, batch_acts, batch_rets):
        logp = self.get_policy(batch_obs).log_prob(batch_acts)
        batch_rets = batch_rets.to(torch.float32)
        return -(logp * batch_rets).mean()

    def update(self, batch_obs, batch_acts, batch_rets):
        batch_obs = torch.tensor(np.array(batch_obs))
        batch_acts = torch.tensor(np.array(batch_acts))
        batch_rets = torch.tensor(np.array(batch_rets))
        self.optimizer.zero_grad()
        batch_loss = self.loss(batch_obs, batch_acts, batch_rets)
        batch_loss.backward()
        self.optimizer.step()

def plot(run_name, all_episode_rews):
    import matplotlib.pyplot as plt
    plt.plot(all_episode_rews)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title(run_name)
    plt.show()

def rewards_to_go(ep_rews):
    ep_rews_to_go = []
    for i in range(len(ep_rews)):
        ep_rews_to_go.append(sum(ep_rews[i:]))
    return ep_rews_to_go

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
    # Experiment Execution Arguments
    parser.add_argument('--cuda', action='store_true', default=False, help='enables CUDA training')
    parser.add_argument('--wandb', action='store_true', default=False, help='enables WandB experiment tracking')
    parser.add_argument('--wandb_project_name', type=str, default="DRL_Gym_PyTorch", help="the WandB's project name")
    parser.add_argument('--wandb_entity', type=str, default=None, help="the entity (team) of WandB's project")
    parser.add_argument('--wandb_resume', type=str, default="auto", help="Resume setting. auto, auto-resume without id given; allow, requires give previous run id or starts a new run; must, requires id and crashes if not the same as a previous run, ensuring resuming.")
    args = parser.parse_args()
    args.cuda = args.cuda and torch.cuda.is_available()
    return args

def printing(args, env):
    print(f"env.action_space: {env.action_space}")
    print(f"env.observation_space: {env.observation_space}")
    print(f"env.observation_space.high: {env.observation_space.high}")
    print(f"env.observation_space.low: {env.observation_space.low}")
    print(f"env.action_space.n: {env.action_space.n}")
    print(f"env.observation_space.shape[0]: {env.observation_space.shape[0]}")
    print(f"env._max_episode_steps: {env._max_episode_steps}")
    run_name = f"{args.env_id}, {args.algo}, hidden_size={args.hidden_size}, num_batches={args.num_batches}, batch_size={args.batch_size}"
    print(f"RUN_NAME: {run_name}")
    return run_name

def wandb_init(args):
    wandb.init(
        project=args.wandb_project_name,
        entity=args.wandb_entity,
        group=args.algo,
        resume=args.wandb_resume,
        sync_tensorboard=True,
        config=vars(args),
        monitor_gym=True,
        save_code=True,
    )

def train(env, agent, args):

    all_episode_lens = []
    all_episode_rews = []
    for i_batch in range(args.num_batches):
        batch_obs = []
        batch_acts = []
        batch_rews = []
        observation = env.reset(seed=args.seed)[0]
        t = 0
        ep_rews = []
        while True:
            # env.render()
            batch_obs.append(observation.copy())
            action = agent.action_select(observation)
            observation, reward, done, trunc, info = env.step(action)
            t += 1
            batch_acts.append(action)
            ep_rews.append(reward)
            if done or trunc:
                all_episode_lens.append(t + 1)
                ep_num = len(all_episode_lens)
                total_episode_reward = sum(ep_rews)
                all_episode_rews.append(total_episode_reward)
                if args.wandb:
                    wandb.log({"episodic reward": total_episode_reward, "episode number": ep_num})
                ep_rews_to_go = rewards_to_go(ep_rews)
                batch_rews.extend(ep_rews_to_go)
                print(f"Batch {i_batch + 1}, episode {ep_num} finished after {t} timesteps, total reward: {total_episode_reward}")
                if len(batch_obs) >= (args.num_eps_in_batch * env._max_episode_steps):
                    agent.update(batch_obs, batch_acts, batch_rews)
                    print(f"Updating after episode {ep_num} of batch {i_batch + 1}.")
                    break
                observation = env.reset(seed=args.seed)[0]
                t = 0
                ep_rews = []
    print(f"Average episode length was: {sum(all_episode_lens) / ep_num:.1f} timesteps")
    return all_episode_rews

if __name__ == "__main__":
    import time
    args = get_args()
    env = gym.make(args.env_id, render_mode="rgb_array")
    agent = VPG_Agent(env.observation_space.shape[0], env.action_space, args.hidden_size)
    args.max_ep_steps = env._max_episode_steps
    args.batch_size = args.num_eps_in_batch * args.max_ep_steps
    run_name = printing(args, env)
    if args.wandb:
        import wandb
        wandb_init(args)
    # env = gym.wrappers.RecordVideo(env, "/home/elliot/DRL_Gym_PyTorch/wandb/videos/")
    start_time = time.time()
    all_episode_rews = train(env, agent, args)
    end_time = time.time()
    print(f"TRAINING TIME: {end_time - start_time:.2f} seconds")
    plot(run_name, all_episode_rews)
    if args.wandb:
        wandb.finish()