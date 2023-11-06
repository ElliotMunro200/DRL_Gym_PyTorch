import gymnasium as gym
import cv2
import wandb
import os
import argparse

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
    parser.add_argument("--wandb", action="store_true", help="Enable logging to Weights & Biases")
    parser.add_argument("--visualize", action="store_true", help="Enable video playback in a popup window")
    args = parser.parse_args()

    # Set up the Hopper environment
    env = gym.make('Hopper-v4', render_mode='rgb_array', healthy_angle_range=(-2.0, 2.0))
    supported_modes = env.metadata.get('render_modes')
    if 'rgb_array' in supported_modes:
        print("rgb_array mode is supported!")
    else:
        print("rgb_array mode is NOT supported. Available modes:", supported_modes)
    t = 0
    observation = env.reset()

    # Initialize Weights & Biases if the --wandb flag is provided
    if args.wandb:
        wandb.init(project="CPG_Hopper", entity=None)

    if args.visualize:
        # Directory to save videos
        video_dir = './videos'
        os.makedirs(video_dir, exist_ok=True)
        video_path = os.path.join(video_dir, 'hopper_episode.mp4')

        # Set up the video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')

        video_writer = None
        if video_writer is None:
            frame = env.render()
            height, width, _ = frame.shape
            video_writer = cv2.VideoWriter(video_path, fourcc, 50.0, (width, height))
            print("success making frame and video_writer!")

    # Run one episode
    term = False
    trunc = False
    total_reward = 0
    while not (term or trunc):
        action = env.action_space.sample()  # Replace with your own policy
        if t != 0:
            obs = [round(float(i), 3) for i in observation[0:5]]
        elif t == 0:
            obs = [round(float(i), 3) for i in observation[0][0:5]]
        act = [round(float(i), 3) for i in action]
        print(f"obs_(t={t}): {obs}, action_(t={t}): {act}")
        observation, reward, term, trunc, info = env.step(action)
        # rewards
        if args.wandb:
            wandb.log({'step_reward': reward})
        total_reward += reward

        # Render to rgb array and write to video
        frame = env.render()
        video_writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

        t += 1

    print(f"obs_(t={t}): {obs}")

    # Release the video writer and close the environment
    video_writer.release()
    env.close()

    # end of episode logging
    if args.wandb:
        # Log the total reward to wandb
        wandb.log({'total_reward': total_reward})

        # Upload the video to wandb
        wandb.log({"hopper_episode": wandb.Video(video_path, fps=50, format="mp4")})

        # Close the wandb run
        wandb.finish()

    # Play back the video in a popup window on repeat
    if args.visualize:
        video_playback_loop(video_path)