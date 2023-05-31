# RL Script: defines a possible set of Experiments by housing a variable set of args + a defined Algorithm(args)

# Script
    # - Inst. args (after call to python Script.py --[given args])
    # ~ def Algorithm(args)
            # - Inst. env(args)
            # - Inst. agent(args, env, data_0)

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

        # trigger for data=env.step(action), when self.step(data)

        def step(self, data):
            return action

# For each script arg:
    # Take note of what it is and what it does.
        # If it is optional to be active, take note of when it should be enabled/disabled (development phases).
# If args(trainable phase) ~ args(debugging phase) and/or if unlikely that script actually trains (do next 2 in one).
# Find empirical wall-clock time per timestep (args chosen for trainable phase).
# If a*E[wall-clock time per timestep] < empirical wall-clock time per timestep (for chosen a: a >= 1):
    # Find E[wall-clock time per training-loop-part per timestep] and record (args chosen for trainable phase).
    # For each part:
        # If long:
            # Fix its time issue.
# Check to see if there is functional performance tracking (for trainable phase). If not:
    # Make it, and verify that it works on a much smaller, quick problem. Note how it connects to args, and when to use.
# Train for b*E[# of timesteps] (for chosen b: 0 < b =< 1). Make sure to use the correct args (trainable phase).

# Find E[wall-clock time per training-loop-part per timestep] and record (args chosen for debugging phase).
# Use this info to order tests of experiment aspects: external, simple --> core, complex. I.e., args --> training loop \
# --> env/task --> agent structure --> agent update functionality/logic.
# Once the offending parts have been debugged and retested, scale the experiment up again and retrain.