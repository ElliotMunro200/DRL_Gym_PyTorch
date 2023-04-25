# A2C is a specific type of AC (VPG that bootstraps to update value function)
# algorithm that both:
# 1) uses A to update pi and V (advantage function used)
# 2) uses parallel workers to collect experience (distributed RL)
# it is a synchronous version of A3C - an AC algorithm that uses asynchronous actor-learners