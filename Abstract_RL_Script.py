# RL Script: defines a possible set of Experiments by housing a variable set of args + a defined Algorithm(args)
#
# Script
#     - Inst. args (after call to python Script.py --[given args])
#     ~ def Algorithm(args)
#             - Inst. env(args)
#             - Inst. agent(args, env)
#             for t in range(num_timesteps):
#                 data = env.step(act=None)
#                 action = agent.step(data)
#
# Experiment = Algorithm(args)
# Task = env(args)

class Env():

    def __init__(self, args):
        self.args = args
        self.data_0 = self.step()
        if args.agent:
            agent, action = self.Agent(args, self, self.data_0)

    def step(self, act=None):
        data = act
        return data

    class Agent():
        def __new__(cls, args, env, data_0):
            instance = super().__new__(cls)
            action = instance.step(data_0)
            return instance, action

        def __init__(self, args, env, data_0):
            self.args = args
            # trigger for data=env.step(action), when self.step(data)

        def step(self, data):
            action = data
            return action