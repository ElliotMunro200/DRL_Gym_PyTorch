import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from torch.distributions.normal import Normal
from torch.optim import Adam
import gym
from gym.spaces import Box, Discrete
import numpy as np
from utils import get_args, printing, mlp, wandb_init, rewards_to_go, plot

# repeat for number of updates:
# collect experience until buffer is full: -
# init buffer lists -
# obs = env.reset() -
# action = agent.action_select() - via stochastic policy
# obs, rew, done = env.step(action) -
# add (obs, rew, done) to buffer lists -
# when buffer is full execute VPG update: -

class VPG_Agent(nn.Module):
    def __init__(self, args, envtask):
        super(VPG_Agent, self).__init__()
        self.args = args
        self.envtask = envtask
        self.env = self.envtask.env
        self.obs_space = self.env.observation_space
        self.obs_dim = self.obs_space.shape[0]
        self.action_space = self.env.action_space
        if isinstance(self.action_space, Discrete):
            self.act_dim = self.action_space.n
        elif isinstance(self.action_space, Box):
            self.act_dim = self.action_space.shape[0]
        self.hidden_sizes = self.args.hidden_sizes
        self.run_name = printing(self.args, self.env)
        self.pi_net = mlp([self.obs_dim] + list(self.hidden_sizes) + [self.act_dim], nn.Tanh)
        #self.pi_net = nn.Sequential(
        #    nn.Linear(self.obs_dim, self.hidden_dim),
        #    nn.Tanh(),
        #    nn.Linear(self.hidden_dim, self.act_dim)
        #)
        self.optimizer = Adam(self.pi_net, lr=1e-2)

    def get_policy(self, obs):
        obs = obs.to(torch.float32)
        logits = self.pi_net(obs)
        if isinstance(self.action_space, Discrete):
            policy = Categorical(logits=logits)
        elif isinstance(self.action_space, Box):
            policy = Normal(logits, 1)
        return policy

    def action_select(self, obs):
        obs = torch.tensor(obs)
        if isinstance(self.action_space, Discrete):
            action = self.get_policy(obs).sample().item()
        elif isinstance(self.action_space, Box):
            action = self.get_policy(obs).sample().numpy()
        return action

    def loss(self, batch_obs, batch_acts, batch_rets):
        if isinstance(self.action_space, Discrete):
            logp = self.get_policy(batch_obs).log_prob(batch_acts)
        elif isinstance(self.action_space, Box):
            logp = self.get_policy(batch_obs).log_prob(batch_acts).sum(axis=-1)
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
        return batch_loss

    def step(self, obss, rews, terms, truncs):
        return

    def train(self):

        if args.wandb:
            import wandb
            run = wandb_init(args)

        import time
        start_time = time.time()

        actions, env_t = agent.step(self.env.reset()[0]), 0
        while agent.t < args.training_steps:
            obss, rews, terms, truncs, _ = self.env.step(actions)
            if (terms or truncs):
                obss, _ = self.env.reset()
            env_t += 1
            # print(f"env_t: {env_t}, agent.T: {agent.t}")
            assert agent.t == env_t  # to remove
            actions = agent.step(obss, rews, terms, truncs)

        end_time = time.time()
        print(f"TRAINING TIME: {end_time - start_time:.2f} seconds")

        if args.wandb:
            wandb.finish()

        if self.args.plot:
            plot(self.total_rews_by_ep, self.run_name)

def train(env, agent, args):

    all_episode_lens = []
    all_episode_rews = []
    i_batch = 0
    # checking if wandb ID is the same as a previous run --> resumes training
    if args.wandb:
        if wandb.run.resumed:
            f = wandb.restore(args.checkpoint_model_file)
            print(f)
            print(type(f))
            checkpoint = torch.load(f, encoding='utf-8')
            agent.pi_net.load_state_dict(checkpoint['model_state_dict'])
            agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            i_batch = checkpoint['epoch']
            loss = checkpoint['loss']

    # looping of batches/epochs (size = n * episode_size, for chosen n)
    while i_batch < args.num_batches:
        batch_obs = []
        batch_acts = []
        batch_rews = []
        observation = env.reset()[0]  # env.reset(seed=args.seed)[0]
        t = 0
        ep_rews = []
        # continue to loop in the epoch until buffer is sufficiently filled, then train, and empty buffer for new epoch.
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
                #print(f"Batch {i_batch + 1}, episode {ep_num} finished after {t} timesteps, total reward: {total_episode_reward:.0f}")
                if len(batch_obs) >= (args.num_eps_in_batch * env._max_episode_steps):
                    loss = agent.update(batch_obs, batch_acts, batch_rews)
                    if args.wandb:
                        wandb.log({'loss': loss.item()}, step=i_batch)
                        if ((i_batch+1) % (args.num_batches // 5)) == 0:
                            torch.save({
                                'epoch': i_batch,
                                'model_state_dict': agent.pi_net.state_dict(),
                                'optimizer_state_dict': agent.optimizer.state_dict(),
                                'loss': loss,
                                }, args.checkpoint_model_file)
                            wandb.save(args.checkpoint_model_file)
                    print(f"Batch {i_batch + 1}, episode {ep_num}, timesteps {t}, last reward: {total_episode_reward:.0f}, loss: {loss:.2f}.")
                    i_batch += 1
                    break
                observation = env.reset(seed=args.seed)[0]
                t = 0
                ep_rews = []
    print(f"Average episode length was: {sum(all_episode_lens) / ep_num:.1f} timesteps")
    return all_episode_rews


if __name__ == "__main__":
    args = get_args()
    from utils import EnvTask
    envtask = EnvTask(args)
    agent = VPG_Agent(args, envtask)
    _ = agent.train()
    _ = agent.evaluate()