import math as m
import numpy as np

class oscillator():
    def __init__(self, timestep, a, R, v, w, phi):
        self.dt = timestep
        self.a = a
        self.R = R
        self.v = v
        self.w = w
        self.phi = phi
        self.dr2dt = np.zeros((7, 2))
        self.drdt = np.zeros((7, 2))
        self.r = np.zeros((7, 2))
        self.dOdt = np.zeros((7, 2))
        self.O = np.zeros((7, 2))
        self.x = np.zeros((7, 2))


    def dr2_dt(self):
        self.dr2dt = self.a * (self.a/4 * (self.R - self.dr2dt) - self.drdt)

    def dr_dt(self):
        self.drdt += self.dr2dt * self.dt

    def new_r(self):
        self.r += self.drdt * self.dt

    def dO_dt(self):
        rflat = self.r.flatten()
        oflat = self.O.flatten()
        couplesum = 0
        for j, r in enumerate(rflat):
            couplesum += r * self.w[:, j] * m.sin(oflat[j] - oflat - self.phi[:, j])
        self.dOdt = 2 * m.pi * self.v + couplesum

    def new_O(self):
        self.O += self.dOdt * self.dt

    def new_x(self):
        self.x = self.r * (1 + m.cos(self.O))

class amphibot_CPG(oscillator):
    def __init__(self, R, v, w, phi, t=0.001, a=100):
        super().__init__(t, a, R, v, w, phi)
        self.


    def step(self):



    def plot_joints(self):
        import matplotlib.pyplot as plt
        plt.plot(self.joints)
        plt.title(self.exp_info)
        plt.ylabel("Total Rewards")
        plt.xlabel("Episode")
        plt.show()

def make_phi_w():
    phi = np.zeros((14, 14))
    w = np.zeros((14, 14))


    return phi, w

if __name__ == "__main__":
    R = 1
    v = 1
    phi, w = make_phi_w()
    sea_snake = amphibot_CPG(R, v, w, phi)
    sea_snake.plot_joints()