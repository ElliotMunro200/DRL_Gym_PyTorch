import torch
from torch import nn
from torch.optim import Adam
from copy import deepcopy
from torch.distributions.normal import Normal
import numpy as np
import itertools

import time
from gym.spaces import Box
from PG_goals_base import MLP_GoalActorCritic_TD3, PG_Goal_OffPolicy_Buffer
from utils import get_args, printing, make_env, Subgoal, evaluate_policy, plot, wandb_init, bcolors


class TD3_Goal_Agent(nn.Module):
    def __init__(self, args, env):
        super().__init__()
        # args, env inputs
        self.args = args
        self.env = env
        self.env_max_ep_steps = env._max_episode_steps
        # state, action, goal spaces
        self.obs_space = env.observation_space
        # [qpos(15)|qvel(14)|fg(2)|ep_t_left(1)] = dim of 33.
        self.obs_dim = self.obs_space.shape[0]+2+1
        self.act_space = env.action_space
        assert isinstance(self.act_space, Box)
        # Ant: 2 joints per leg, 4 legs -> act dim = 8
        self.act_dim = self.act_space.shape[0]
        self.act_limit = env.action_space.high[0]
        self.subgoal_dim = args.subgoal_dim
        self.subgoal = Subgoal(self.subgoal_dim)
        # testing without changing subgoals
        self.subtask_length = self.env_max_ep_steps
        self.sg = None
        self.fg = None
        # Networks
        self.h_sizes = self.args.hidden_sizes
        self.TD3_ac = MLP_GoalActorCritic_TD3(self.obs_dim, self.subgoal_dim, self.act_space, hidden_sizes=self.h_sizes)
        with torch.no_grad():
            self.TD3_ac_target = deepcopy(self.TD3_ac)
        # Buffer+Rewards lists
        self.batch_size = args.batch_size
        self.buffer_size = args.buffer_size
        self.buffer = PG_Goal_OffPolicy_Buffer(self.obs_dim, self.subgoal_dim, self.act_dim, self.batch_size, self.buffer_size)
        self.total_rews_by_ep = []
        self.ep_rew = 0
        # Training params
        self.gamma = args.gamma
        self.tau = args.tau
        self.target_noise = args.target_noise
        self.noise_clip = args.noise_clip
        self.training_first_ep_num = args.training_first_ep_num
        self.update_period = args.update_period
        self.delayed_update_period = args.delayed_update_period
        self.MseLoss = nn.MSELoss()
        self.last_Q_loss = 100000
        self.last_Pi_loss = 100000
        self.optimizerPi = Adam(self.TD3_ac.pi.parameters(), lr=args.actor_learning_rate)
        self.q_params = itertools.chain(self.TD3_ac.q1.parameters(), self.TD3_ac.q2.parameters())
        self.optimizerQ = Adam(self.q_params, lr=args.critic_learning_rate)
        # Evaluation params
        self.evaluate = evaluate_policy
        # Step counters
        self.t = 0
        self.ep_t = 0
        self.ep_t_left = self.env_max_ep_steps
        self.n_update = 0
        self.N_sum_prev_ep_lens = 0
        self.ep_num = 0

    # soft update of target networks
    def soft_target_weight_update(self):
        # Q1 update
        Q1_weights = self.TD3_ac.q1.state_dict()
        Q1_Target_weights = self.TD3_ac_target.q1.state_dict()
        for key in Q1_Target_weights:
            Q1_Target_weights[key] = Q1_Target_weights[key] * (1 - self.tau) + Q1_weights[key] * self.tau
        self.TD3_ac_target.q1.load_state_dict(Q1_Target_weights)
        # Q2 update
        Q2_weights = self.TD3_ac.q2.state_dict()
        Q2_Target_weights = self.TD3_ac_target.q2.state_dict()
        for key in Q2_Target_weights:
            Q2_Target_weights[key] = Q2_Target_weights[key] * (1 - self.tau) + Q2_weights[key] * self.tau
        self.TD3_ac_target.q2.load_state_dict(Q2_Target_weights)
        # Pi update
        Pi_weights = self.TD3_ac.pi.state_dict()
        Pi_Target_weights = self.TD3_ac_target.pi.state_dict()
        for key in Pi_Target_weights:
            Pi_Target_weights[key] = Pi_Target_weights[key] * (1 - self.tau) + Pi_weights[key] * self.tau
        self.TD3_ac_target.pi.load_state_dict(Pi_Target_weights)

    # extracts rewards from the buffer for the past episode and appends the total to the episodic rewards list,
    # prints the episode rewards, resets episode time counter
    # TODO: check this logic, especially finding ep_rews.
    def after_episode(self):
        # slicing most recent rewards from the reward buffer, taking into account cyclic deque index ptr.
        index = self.buffer.ptr if self.buffer.ptr > 0 else self.buffer.curr_size
        ep_rews = self.buffer.rew_buf[(index-(self.ep_t+1)):index]
        self.ep_rew = sum(ep_rews)
        self.total_rews_by_ep.append(self.ep_rew)
        self.N_sum_prev_ep_lens += (self.ep_t + 1)

    # Updates the TD3_ac (pi, Q1, Q2), with a form of GPI, and then copies softly to TD3_ac_targ.
    # Common DDPG failure mode is dramatically overestimating Q-values, which TD3 addresses directly in 3 ways.
    # Use target-policy smoothing, twin critics (for minimum Q value estimate), delayed updates of policy + target nets.
    # Both Q1 and Q2 are updated with MSELoss on calculated Q-targets - more accurate and self-consistent (from r, Q).
    # Pi is updated from the loss function L=-Q1(s,g,pi(s,g)) - which is improving Pi for the current Q1.
    def update(self):
        # pulls a batch of random and unordered timestep-tuple pairs from the buffer.
        # the first timestep-tuple cannot have a done signal
        data = self.buffer.sample_batch()
        b_obs, b_rews, b_dones, b_goals, b_acts = data['obs'], data['rew'], data['done'], data['subg'], data['act']
        b_obs_2, b_rews_2, b_dones_2, b_goals_2, b_acts_2 = data['obs2'], data['rew2'], data['done2'], data['subg2'], data['act2']
        #print(f"A2 buffer: {b_acts_2[0]}")

        # Q1 and Q2 vals from buffer for updating parameters of
        Q1 = self.TD3_ac.q1(b_obs, b_goals, b_acts)
        Q2 = self.TD3_ac.q2(b_obs, b_goals, b_acts)

        with torch.no_grad():
            pi_targ_act = self.TD3_ac_target.pi(b_obs_2, b_goals_2)
            #print(f"pi_targ_act: {pi_targ_act[0]}")
            epsilon = torch.randn_like(pi_targ_act) * self.target_noise
            b_acts_2 = pi_targ_act + epsilon
            #print(f"A2 target sum: {b_acts_2[0]}")
            b_acts_2 = torch.clip(b_acts_2, -self.act_limit, self.act_limit)
            #print(f"A2 target clip: {b_acts_2[0]}")
            Q1_pi_target = self.TD3_ac_target.q1(b_obs_2, b_goals_2, b_acts_2)
            Q2_pi_target = self.TD3_ac_target.q2(b_obs_2, b_goals_2, b_acts_2)
            Q_pi_target = torch.min(Q1_pi_target, Q2_pi_target)
            targets = b_rews_2 + self.gamma * (1 - b_dones_2.int()) * Q_pi_target

        Q1Loss = self.MseLoss(Q1, targets)
        Q2Loss = self.MseLoss(Q2, targets)
        QLoss = Q1Loss + Q2Loss

        self.optimizerQ.zero_grad()
        QLoss.backward()
        self.optimizerQ.step()

        self.last_Q_loss = QLoss

        if self.n_update % self.delayed_update_period == 0:

            for p in self.q_params:
                p.requires_grad = False

            self.optimizerPi.zero_grad()
            PiLoss = -1 * self.TD3_ac.q1(b_obs, b_goals, self.TD3_ac.pi(b_obs, b_goals)).mean()
            PiLoss.backward()
            self.optimizerPi.step()

            self.last_Pi_loss = PiLoss

            with torch.no_grad():
                self.soft_target_weight_update()

            for p in self.q_params:
                p.requires_grad = True

        self.n_update += 1

    # taking in and outputting np.float32, converts to tensors to get the action means from the policy
    # adds noise, and clips action values to range (-1.0, 1.0).
    def action_from_obs(self, obs, subg):
        obs_tensor = torch.from_numpy(obs).type(torch.float32)
        subg_tensor = torch.from_numpy(subg).type(torch.float32)
        mean = self.TD3_ac.pi(obs_tensor, subg_tensor)
        noise = Normal(torch.tensor([0.0]), torch.tensor([1.0])).sample()
        action = torch.clip(mean+noise, -self.act_limit, self.act_limit).numpy()
        return action

    def action_select(self, obs, subg):
        if self.ep_num >= self.training_first_ep_num:
            with torch.no_grad():
                action = self.action_from_obs(obs, subg)
        else:
            action = self.act_space.sample()
        return action

    def subgoal_select(self, obs):
        #self.sg = self.subgoal.action_space.sample()
        self.sg = deepcopy(self.fg)

    # The logic function/controller of this controller level in the hierarchy. Has multiple functions per step:
    # 1) subgoal selection; 2) action selection + time increments; 3) storing in buffer;
    # 4) updating nets (+QLoss logging to W+B); 5) printing episode rewards + local logging (+logging to W+B);
    # 6) returning selected action back to the loop in the external train() function.
    def step(self, obs, rews=0.0, done=False):
        # TODO: updating step counters.
        self.ep_t_left = int(obs[-1])
        self.ep_t = self.env_max_ep_steps - self.ep_t_left
        if self.ep_t == 0 and self.N_sum_prev_ep_lens > 0:
            self.ep_num += 1
        self.t = self.ep_t + self.N_sum_prev_ep_lens
        # print(f"ep_t_left: {self.ep_t_left} | ep_t: {self.ep_t} | t: {self.t} | N_sum: {self.N_sum_prev_ep_lens}")

        # subgoal selection
        if self.ep_t % self.subtask_length == 0:
            self.subgoal_select(obs)
        # action selection + time increments
        acts = self.action_select(obs, deepcopy(self.sg))

        # store timestep tuple (s,r,d,g,a) on every single step.
        # TODO: test MCMC idea of throwing away warmup episode data from buffer after training on it.
        self.buffer.store(obs, rews, done, deepcopy(self.sg), acts)

        # update from specified first training episode number (just filling buffer with random actions before) if also:
        # not in the first steps of training, and the specified period of time has elapsed since the last update.
        # TODO: test the effect of the warmup data collection length on training.
        if (self.ep_num >= self.training_first_ep_num) and \
                (self.t >= self.update_period) and ((self.ep_t+1) % self.update_period == 0):
            # one update for each timestep since last updating, and logging
            for _ in range(self.update_period):
                self.update()
                if args.wandb:
                    wandb.log({"QLoss": self.last_Q_loss, "PiLoss": self.last_Pi_loss}, step=self.t)

        # when episode is over, do the following:
        if done:
            self.after_episode()
            print(f"| Episode {self.ep_num:<3} done | Steps: {self.ep_t:<3} | Rewards: {self.ep_rew:<4.1f} "
                  f"| Last QLoss: {self.last_Q_loss:<4.1f} | Last PiLoss: {self.last_Pi_loss:<4.1f} |")
            if args.wandb:
                wandb.log({"episodic reward": self.ep_rew, "episode number": self.ep_num})
        return acts


# the training function
def train(args, env, agent):
    start_training_time = time.time()
    done = True
    while env.t < args.training_steps:
        # every env.reset() and env.step() is counted as a timestep and so increments env.step_count by 1.
        if done:
            obs, rews, done = env.reset()
            agent.fg = obs['desired_goal']
            obs = obs['observation']

        elif not done:
            obs, rews, done, _ = env.step(action)
            obs = obs['observation']
        # agent saves (s,r,d,g,a) timestep data to the buffer within agent.step() every time.
        # agent observes ep_t_left = obs[-1] and then calculates -> ep_t -> ep_num -> t.
        # at the end of the episode N_sum_prev_full_ep_lens is updated by ep_t.
        #print(f"{env.t} | {obs[:2]} | {obs[-3:]} | {rews} | {done}")
        action = agent.step(obs, rews, done)
        #print(f"agent.t: {agent.t}, env.t: {env.t}")
        assert agent.t == env.t
    total_rews_by_ep = agent.total_rews_by_ep

    end_training_time = time.time()
    print(f"TOTAL TRAINING TIME: {end_training_time - start_training_time:.2f}s")

    return total_rews_by_ep


if __name__ == "__main__":
    # define the experiment: args --> env|task --> agent, and get info about it
    args = get_args()
    env = make_env(args)
    agent = TD3_Goal_Agent(args, env)
    exp_info = printing(args, env)

    # logging
    if args.wandb:
        import wandb
        run = wandb_init(args)
    # env = gym.wrappers.RecordVideo(env, "/home/elliot/DRL_Gym_PyTorch/wandb/videos/")

    # execute training and/or evaluation phases
    if args.train:
        ep_rews = train(args, env, agent)
        if args.plot:
            plot(ep_rews, exp_info)
    if args.eval:
        rewards_array, success_rate = agent.evaluate(args, env, agent, render=True, save_video=True)
        print(f"{bcolors.OKGREEN}Eval rewards: {rewards_array} | Success rate: {success_rate}{bcolors.ENDC}")

    if args.wandb:
        wandb.finish()
