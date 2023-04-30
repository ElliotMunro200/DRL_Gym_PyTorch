import torch
from torch import nn
from torch.optim import Adam
from torch.distributions import Categorical
import gym


class ActorCritic_Agent(nn.Module):
    def __init__(self, obs_shape, action_shape, hidden_size):
        super().__init__()
        # Make logits net (for building the policy)
        self.obs_shape = obs_shape
        self.action_shape = action_shape
        self.hidden_size = hidden_size
        self.logits_net = nn.Sequential(
            nn.Linear(self.obs_shape, self.hidden_size),
            nn.Tanh(),
            nn.Linear(self.hidden_size, self.action_shape)
        )
        # Make the Critic
        self.critic = nn.Sequential(
            nn.Linear(self.obs_shape, self.hidden_size),
            nn.Tanh(),
            nn.Linear(self.hidden_size, 1)
        )
        # Make optimizers
        self.critic_optimizer = Adam(self.critic.parameters(), lr=1e-2)
        self.policy_optimizer = Adam(self.logits_net.parameters(), lr=1e-2)
        # Make learning parameters
        self.gamma = 0.99
        self.error = torch.empty(1)
        self.Igamma = 1.0
        self.critic_lr = 0.1
        self.policy_lr = 0.1

    # Make get policy and action select functions
    def get_policy(self, obs):
        logits = self.logits_net(obs)
        dist = Categorical(logits=logits)
        return dist

    def action_select(self, obs):
        dist = self.get_policy(obs)
        action = dist.sample()
        logp = dist.log_prob(action)
        return action.item(), logp.item()

    # Make update function [+ loss function]
    def calculate_value_error(self, rew, done, new_obs, obs):
        new_value = torch.tensor([0.0]) if done else self.critic(new_obs)
        rew = torch.tensor(rew, requires_grad=False)
        error = rew + self.gamma * new_value.detach() - self.critic(obs).detach()
        self.error = error

    def update_critic(self, obs):
        value_loss = -1 * self.critic_lr * self.error * self.critic(obs)

        self.critic_optimizer.zero_grad()
        value_loss.backward()
        self.critic_optimizer.step()
        return value_loss.item()

    def update_policy(self, obs, action):
        logp = self.get_policy(obs).log_prob(action)
        policy_loss = -1 * self.policy_lr * self.Igamma * self.error * logp
        self.Igamma *= self.gamma

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()
        return policy_loss.item()


def train(env_name="CartPole-v1", hidden_size=32):
    env = gym.make(env_name)
    agent = ActorCritic_Agent(env.observation_space.shape[0], env.action_space.n, hidden_size)
    # Make training loops over episodes and timesteps
    num_episodes = 300
    total_rews_by_ep = []
    for episode in range(num_episodes):
        obs = env.reset()[0]
        obs_tensor = torch.from_numpy(obs)
        done, trunc = False, False
        t = 0
        agent.Igamma = 1.0
        ep_rewards = []
        # For each timestep: select+execute action, calculate TD-error, update pi+V.
        while not (done or trunc):
            action, _ = agent.action_select(obs_tensor)
            new_obs, rew, done, trunc, _ = env.step(action)
            t += 1
            ep_rewards.append(rew)
            action_tensor = torch.tensor(action)
            new_obs_tensor = torch.from_numpy(new_obs)
            agent.calculate_value_error(rew, done, new_obs_tensor, obs_tensor)
            value_loss = agent.update_critic(obs_tensor)
            policy_loss = agent.update_policy(obs_tensor, action_tensor)
            obs_tensor = new_obs_tensor
        total_ep_rews = sum(ep_rewards)
        total_rews_by_ep.append(total_ep_rews)
        print(
            f"| Episode {episode:<3} done | Length: {t:<3} | Rewards {total_ep_rews:<5} | Value loss: {value_loss:<5.1f} | Policy loss: {policy_loss:<4.1f} |")
    return total_rews_by_ep


def plot(rews):
    import matplotlib.pyplot as plt
    plt.plot(rews)
    plt.title("CartPole-v1, AC, hidden_size=32, episodes=300")
    plt.ylabel("Total Rewards")
    plt.xlabel("Episode")
    plt.show()


if __name__ == "__main__":
    import time
    start_time = time.time()
    total_rews_by_ep = train()
    end_time = time.time()
    print(f"END_TIME: {end_time - start_time}")
    plot(total_rews_by_ep)
