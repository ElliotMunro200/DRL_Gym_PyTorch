import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from torch.distributions.normal import Normal
from torch.optim import Adam

from gym.spaces import Box, Discrete
import numpy as np
from utils import get_args, printing, mlp, wandb_init, plot
from PG_base import PG_OnPolicy_Buffer
import time

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
        super().__init__()
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
        self.optimizer = Adam(self.pi_net.parameters(), lr=1e-2)
        print(self.action_space.shape)
        self.buffer = PG_OnPolicy_Buffer(self.obs_dim, self.action_space.shape, self.args.num_steps_in_batch)
        #self.all_episode_lens = []
        self.all_episode_rews = []
        self.ep_rews = []
        self.total_episode_reward = 0.0
        self.t = 0
        self.i_batch = 0
        self.ep_num = 0
        self.loss = 0

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

    def loss_func(self, b_obs, b_acts, b_rets):
        if isinstance(self.action_space, Discrete):
            logp = self.get_policy(b_obs).log_prob(b_acts)
        elif isinstance(self.action_space, Box):
            logp = self.get_policy(b_obs).log_prob(b_acts).sum(axis=-1)
        loss = -(logp * b_rets).mean()
        self.loss = loss
        return loss

    def rewards_to_go(self, ep_rews):
        ep_rews_to_go = []
        for i in range(len(ep_rews)):
            ep_rews_to_go.append(sum(ep_rews[i:]))
        ep_rews_to_go = torch.as_tensor(ep_rews_to_go, dtype=torch.float32)
        return ep_rews_to_go

    def update(self):
        data = self.buffer.sample_batch()
        b_obs, b_acts, b_rews = data['obs'], data['act'], data['rew']
        b_rets = self.rewards_to_go(b_rews)
        self.optimizer.zero_grad()
        loss = self.loss_func(b_obs, b_acts, b_rets)
        loss.backward()
        self.optimizer.step()
        return loss

    def after_episode(self):
        #self.all_episode_lens.append(self.t - self.all_episode_lens[-1])
        self.ep_num = len(self.all_episode_rews)
        self.total_episode_reward = sum(self.ep_rews)
        self.all_episode_rews.append(self.total_episode_reward)
        if self.args.wandb:
            wandb.log({"episodic reward": self.total_episode_reward, "episode number": self.ep_num})
        self.ep_rews = []
        print(f"Episode {self.ep_num}, batch {self.i_batch}, timesteps {self.t - 1}, last reward: {self.total_episode_reward:.0f}, loss: {self.loss:.2f}.")

    def after_update(self):
        self.buffer.empty()
        self.i_batch += 1
        if self.args.wandb:
            wandb.log({'loss': self.loss.item()}, step=self.i_batch)
            if ((self.i_batch + 1) % (self.args.num_batches // 5)) == 0:
                torch.save({
                    'epoch': self.i_batch,
                    'model_state_dict': agent.pi_net.state_dict(),
                    'optimizer_state_dict': agent.optimizer.state_dict(),
                    'loss': self.loss,
                }, args.checkpoint_model_file)
                wandb.save(args.checkpoint_model_file)

    def after_training(self):
        self.end_time = time.time()
        print(f"TRAINING TIME: {self.end_time - self.start_time:.2f} seconds")
        #print(sum(self.all_episode_lens))
        print(self.ep_num)
        print(f"Average episode length was: {self.t / self.ep_num:.1f} timesteps")
        if self.args.wandb:
            wandb.finish()
        if self.args.plot:
            plot(self.all_episode_rews, self.run_name)


    def before_training(self):
        # setting up wandb logging and doing any training state restoration
        if self.args.wandb:
            run = wandb_init(self.args)
            # checking if wandb ID is the same as a previous run --> resumes training
            if wandb.run.resumed:
                f = wandb.restore(self.args.checkpoint_model_file)
                print(f)
                print(type(f))
                checkpoint = torch.load(f, encoding='utf-8')
                self.pi_net.load_state_dict(checkpoint['model_state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                self.i_batch = checkpoint['epoch']
                self.loss = checkpoint['loss']
                self.t = self.i_batch * self.args.num_steps_in_batch
        self.start_time = time.time()

    def step(self, obs, rew=0.0, term=False, trunc=False):
        act = agent.action_select(obs)
        self.buffer.store(obs, act, rew, term)
        self.t += 1
        self.ep_rews.append(rew)
        if self.buffer.curr_size >= self.args.num_steps_in_batch:
            loss = self.update()
            self.after_update()
        if (term or trunc):
            self.after_episode()
        return act

    def train(self):
        self.before_training()
        act, env_t = self.step(self.env.reset()[0]), 0
        while self.t < self.args.training_steps:
            obs, rew, term, trunc, _ = self.env.step(act)
            if (term or trunc):
                obs, _ = self.env.reset()
                self.t += 1
                env_t += 1
            env_t += 1
            # print(f"env_t: {env_t}, agent.T: {agent.t}")
            assert self.t == env_t  # to remove
            act = self.step(obs, rew, term, trunc)
        self.after_training()

    def evaluate(self):
        return 5


if __name__ == "__main__":
    args = get_args()
    if args.wandb:
        import wandb
    from utils import EnvTask
    envtask = EnvTask(args)
    agent = VPG_Agent(args, envtask)
    _ = agent.train()
    _ = agent.evaluate()