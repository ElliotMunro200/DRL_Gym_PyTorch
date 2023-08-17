class Rectangle:
    def __init__(self, length, width):
        self.length = length
        self.width = width

    def area(self):
        return self.length * self.width

    def perimeter(self):
        return 2 * self.length + 2 * self.width

# Here we declare that the Square class inherits from the Rectangle class
class Square(Rectangle):
    pass
    #def __init__(self, length):
    #    super().__init__(length, length)

if __name__ == "__main__":
    sq = Square(4,5)
    print(dir(Rectangle))


def train(args, env, agent):
    obs, rews, done = env.reset()
    action = agent.step(obs)
    while env.step_count < args.training_steps:
        if not done:
            obs, rews, done, _ = env.step(action)
        action = agent.step(obs, rews, done)
        if done:
            obs, rews, done = env.reset()


def train(args, env, agent):
    done = True
    env.step_count = -1
    while env.step_count < args.training_steps:
        # every env.reset() and env.step() increments env.step_count by 1.
        if done:
            obs, rews, done, _ = env.reset()
        elif not done:
            obs, rews, done, _ = env.step(action)
        # even if the state is terminal the agent makes choices and saves to the buffer.
        # only on the first agent.step of the training does the agent sore nothing in the buffer.
        action = agent.step(obs, rews, done)


def train(args, env, agent):
    env.timestep = 0
    while env.timestep < args.training_steps:
        # every env.step() increments env.step_count by 1, env.reset() does not increment the env.timestep.
        # means that the timestep is correct when reaching agent.step().
        if env.timestep % args.max_episode_steps == 0:
            obs, rews, _, _ = env.reset()
        else:
            obs, rews, _, _ = env.step(action)
        # agent can do an internal calculation of the time remaining in the episode
        #  - from the given env.max_episode_steps info and the env.timestep.
        # even if the state is terminal the agent makes choices and saves to the buffer within agent.step().
        # only on the first agent.step() of the training does the agent sore nothing in the buffer.
        action = agent.step(obs, rews, env.timestep)
