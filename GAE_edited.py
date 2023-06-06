"""
EDITED FROM https://github.com/labmlai/annotated_deep_learning_paper_implementations/blob/master/labml_nn/rl/ppo/gae.py
"""

import numpy as np


class GAE:
    def __init__(self, n_workers: int, worker_steps: int, gamma: float, lambda_: float):
        self.lambda_ = lambda_
        self.gamma = gamma
        self.worker_steps = worker_steps
        self.n_workers = n_workers

    def __call__(self, done: np.ndarray, rewards: np.ndarray, values: np.ndarray) -> np.ndarray:

        # advantages table
        advantages = np.zeros((self.n_workers, self.worker_steps-1), dtype=np.float32)
        gae_t = 0

        for t in reversed(range(self.worker_steps-1)):
            mask = 1.0 - done[:, t+1]
            delta_t = rewards[:, t+1] + self.gamma * mask * values[:, t+1] - values[:, t]
            gae_t = delta_t + self.gamma * self.lambda_ * mask * gae_t
            advantages[:, t] = gae_t

        return advantages