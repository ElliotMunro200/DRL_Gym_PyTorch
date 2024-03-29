import gymnasium as gym
from gymnasium.spaces import Box
import cv2
import wandb
import os
import argparse
import math as m
import numpy as np
import torch
import torch.nn as nn

from PG_base import MLPActor
from TD3 import TD3_Agent
from utils import get_args
from CPG_hopper import CPG_hopper

"""
Executes an episode of Hopper-v4 with actions generated by the imported CPG_hopper class.
CPG parameters om, mu, dp are explicitly defined in the script. 
[Optional with argparse] (1) video saving and playback (2) W+B rewards and video logging 
(3) plotting the CPG set points x = r*cos(theta).
"""

def video_playback_loop(video_path):
    while True:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("Error: Could not open video.")
            break

        while True:
            ret, frame = cap.read()
            if not ret:
                break  # Reached the end of the video
            cv2.imshow('Hopper - Gymnasium', frame)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break  # Quit if 'q' is pressed

        # Ask user if they want to replay the video
        replay = input("Replay the video? [y/n]: ")
        if replay.lower() != 'y':
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # Set up argparse
    parser = argparse.ArgumentParser(description="Run Hopper environment and log with Weights & Biases")
    parser.add_argument("-ngs", "--num_global_steps", type=int, default=200000, help="number of global steps")
    parser.add_argument("-w", "--wandb", action="store_true", help="Enable logging to Weights & Biases")
    parser.add_argument("-v", "--visualize", action="store_true", help="Enable video playback in a popup window")
    parser.add_argument("-p", "--plotCPG", action="store_true", help="Plot CPG oscillations for each joint")
    args = parser.parse_args()

    # Set up the Hopper environment
    env = gym.make('Hopper-v4', render_mode='rgb_array', healthy_angle_range=(-2.0, 2.0))
    supported_modes = env.metadata.get('render_modes')
    if 'rgb_array' in supported_modes:
        print("rgb_array mode is supported!")
    else:
        print("rgb_array mode is NOT supported. Available modes:", supported_modes)

    # Initialize Weights & Biases if CLI flags -w is provided
    if args.wandb:
        wandb.init(project="CPG_Hopper", entity=None)

    # defining initial CPG parameters before beginning of execution/training. # TODO: tidy up
    mu = np.ones((3, 1))
    om = 2 * m.pi * np.array([[2], [4], [8]])
    dp = np.array([m.pi, m.pi])
    dt = 0.008
    hopper = CPG_hopper(mu, om, dp, dt)

    # initializing NN
    obs_space = env.observation_space
    obs_dim = obs_space.shape[0]
    act_dim = len(mu) + len(om) + len(dp)
    act_space = Box(low=-1.0, high=1.0, shape=(8,), dtype=np.float32)
    assert act_space.shape[0] == act_dim
    act_limit = int(act_space.high[0])
    hidden_sizes = [300, 300]
    activation = nn.ReLU
    NN = MLPActor(obs_dim, act_dim, hidden_sizes, activation, act_limit)
    #argsTD3 = get_args() # TODO: make args work
    #TD3 = TD3_Agent(obs_space, act_space, act_limit, argsTD3) # TODO: init TD3

    # reinitializing CPG from NN
    obs = env.reset()[0]
    obs_tensor = torch.from_numpy(obs).type(torch.float32)
    cpg_params = NN(obs_tensor).detach().numpy() # TODO: change to TD3.action_from_obs()
    #cpg_params = TD3.action_select(obs)
    hopper.mu = (np.expand_dims(cpg_params[0:3], axis=1) + 1) / 2
    hopper.om = (np.expand_dims(cpg_params[3:6], axis=1) + 1) * 10 * m.pi
    hopper.dp = cpg_params[6:8] * m.pi

    # loop over global timesteps
    gNs = 0
    Ne = 0
    while gNs <= args.num_global_steps:
        # Run one episode
        t = 0
        Ns = 0
        observation = env.reset()[0]
        term = False
        trunc = False
        total_reward = 0
        while not (term or trunc):
            # choose CPG parameters (with NN) here

            obs_tensor = torch.from_numpy(observation).type(torch.float32)
            cpg_params = NN(obs_tensor).detach().numpy() # TODO: change to TD3.action_from_obs()
            #cpg_params = TD3.action_from_obs(observation)
            hopper.mu = (np.expand_dims(cpg_params[0:3], axis=1) + 1) / 2
            hopper.om = (np.expand_dims(cpg_params[3:6], axis=1) + 1) * 10 * m.pi
            hopper.dp = cpg_params[6:8] * m.pi

            # NN-CPG action selection
            action = hopper.hop()

            # rounding for printing observation and action
            obs = [round(float(i), 3) for i in observation[0:5]]
            act = [round(float(i), 3) for i in action]
            print(f"[gNs={gNs} | Ne={Ne} | Ns={Ns} | t={t:.3f}] obs: {obs}, action: {act}")

            # env step
            observation, reward, term, trunc, info = env.step(action)
            #TD3.buffer.store(observation, action, reward, term) # TODO: change to TD3.buffer.store(...)

            #if Ns % TD3.update_period == 0:
            #    for _ in range(TD3.update_period):
            #        QLoss = TD3.update() # TODO: TD3.update()

            # increment episode rewards, step counts, and episode time (s).
            total_reward += reward
            gNs += 1
            Ns += 1
            t = Ns * dt

        print(f"[gNs={gNs} | Ne={Ne} | Ns={Ns} | t={t:.3f}] obs: {obs}")

        # end of episode reward logging
        if args.wandb:
            wandb.log({'total_reward': total_reward})
        Ne += 1

    # Test episode: visualization, W+B logging, plotting
    if args.visualize:
        # Directory to save videos
        video_dir = './videos'
        os.makedirs(video_dir, exist_ok=True)
        video_path = os.path.join(video_dir, 'hopper_episode.mp4')

        # Set up the video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = None
        if video_writer is None:
            _ = env.reset()
            frame = env.render()
            height, width, _ = frame.shape
            video_writer = cv2.VideoWriter(video_path, fourcc, 125.0, (width, height))
            print("success making env frame and video_writer for test episode!")

    # resetting CPG agent and episode variables before executing test episode.
    # reinitializing CPG from NN
    observation = env.reset()[0]
    obs_tensor = torch.from_numpy(observation).type(torch.float32)
    cpg_params = NN(obs_tensor).detach().numpy()
    #cpg_params = TD3.action_from_obs(observation) # TODO: change to TD3.action_from_obs()
    hopper.mu = (np.expand_dims(cpg_params[0:3], axis=1) + 1) / 2
    hopper.om = (np.expand_dims(cpg_params[3:6], axis=1) + 1) * 10 * m.pi
    hopper.dp = cpg_params[6:8] * m.pi
    dt = 0.008
    hopper = CPG_hopper(mu, om, dp, dt)
    Ne = "T"
    t = 0
    Ns = 0
    term = False
    trunc = False
    test_reward = 0

    # run a test episode for visualization
    while not (term or trunc):

        # CPG parameter selection from NN
        obs_tensor = torch.from_numpy(observation).type(torch.float32)
        cpg_params = NN(obs_tensor).detach().numpy()
        #cpg_params = TD3.action_from_obs(observation) # TODO: change to TD3.action_from_obs()
        hopper.mu = (np.expand_dims(cpg_params[0:3], axis=1) + 1) / 2
        hopper.om = (np.expand_dims(cpg_params[3:6], axis=1) + 1) * 10 * m.pi
        hopper.dp = cpg_params[6:8] * m.pi

        # NN-CPG action selection
        action = hopper.hop()

        # rounding for printing observation and action
        obs = [round(float(i), 3) for i in observation[0:5]]
        act = [round(float(i), 3) for i in action]
        print(f"[gNs={gNs} | Ne={Ne} | Ns={Ns} | t={t:.3f}] obs: {obs}, action: {act}")

        # env step
        observation, reward, term, trunc, info = env.step(action)

        # Render to rgb array and write to video
        if args.visualize:
            frame = env.render()
            video_writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

        # increment episode rewards, step counts, and episode time (s).
        test_reward += reward
        gNs += 1
        Ns += 1
        t = Ns * dt

    print(f"[gNs={gNs} | Ne={Ne} | Ns={Ns} | t={t:.3f}] obs: {obs}")

    # plot CPG oscillations for last episode if CLI flag -p is given.
    if args.plotCPG:
        hopper.time = t
        # plot subjoints for test episode.
        # Only blocking script from continuing when not visualizing (video holds plots open, but blocked by plots).
        hopper.subplot_joints(not args.visualize)

    if args.visualize:
        # Release the video writer and close the environment.
        video_writer.release()
        env.close()
        # Play back the video of the test episode in a popup window on repeat.
        video_playback_loop(video_path)
        # Upload the test episode video to wandb.
        if args.wandb:
            wandb.log({"hopper_episode": wandb.Video(video_path, fps=125, format="mp4")})

    # Test episode reward logging and run closing.
    if args.wandb:
        wandb.log({'test_reward': test_reward})
        wandb.finish()