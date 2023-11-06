import math as m
import numpy as np


class oscillator():
    def __init__(self, dt, a, R, v, dp):
        # defining the CPG oscillator behaviour.
        self.dt = dt
        self.a = a
        self.R = R
        self.v = v
        self.dp = dp
        self.make_phi_w()
        self.dr2dt = np.zeros((7, 2))
        self.drdt = np.zeros((7, 2))
        self.r = np.zeros((7, 2))
        self.dOdt = np.zeros((7, 2))
        self.O = np.zeros((7, 2))
        self.x = np.full((7, 2), 0.5)
        self.x_rec = np.copy(np.expand_dims(self.x[:, 0], axis=1))
        self.sp = np.zeros((7, 1))
        self.sp_rec = np.zeros((7, 1))

    def new_r(self):
        # integrating for new r values from r'' with Euler integration.
        self.dr2dt = self.a * (self.a / 4 * (self.R - self.r) - self.drdt)
        #print(f"dr2dt: {self.dr2dt}")
        self.drdt += self.dr2dt * self.dt
        self.r += self.drdt * self.dt

    def new_O(self):
        # integrating for new theta values from O'' with Euler integration.
        # Flattening used for dot product with slices of weight coupling matrix w.
        rflat = self.r.flatten('F')
        oflat = self.O.flatten('F')
        couplesum = 0
        for j, r in enumerate(rflat):
            phase = oflat[j] - oflat - self.phi[:, j]
            couplesum += r * self.w[:, j] * np.sin(phase)
        self.dOdt = 2 * m.pi * self.v + couplesum
        #print(f"dOdt: {self.dOdt}")
        self.dOdt = self.dOdt.reshape(2, -1).T
        self.O += self.dOdt * self.dt

    def new_x(self):
        # calculating new x values from newly updated r and theta. Recording in x_rec.
        self.x = self.r * (1 + np.cos(self.O))
        self.x_rec = np.append(self.x_rec, np.expand_dims(self.x[:, 0], axis=1), axis=1)
        #print(np.expand_dims(self.x[:, 0], axis=1))
        #print(self.x_rec)

    def new_sp_PD(self):
        # calculating the new setpoint from adjacent oscillators modulating the same joint motor. Recording in sp_rec.
        self.sp = self.x[:, 0] - self.x[:, 1]
        self.sp_rec = np.append(self.sp_rec, np.expand_dims(self.sp, axis=1), axis=1)

    def make_phi_w(self):
        # building new phi and w matrices for a given delta phi value.
        self.phi = np.zeros((14, 14))
        self.w = np.zeros((14, 14))
        for i in range(self.phi.shape[0]):
            for j in range(self.phi.shape[1]):
                if i == j + 1 and i != 7:
                    self.phi[i, j] = -self.dp
                    self.w[i, j] = 1
                elif j == i + 1 and j != 7:
                    self.phi[i, j] = self.dp
                    self.w[i, j] = 1
                elif i == j + 7 or j == i + 7:
                    self.phi[i, j] = m.pi
                    self.w[i, j] = 1


class amphibot_CPG(oscillator):
    def __init__(self, R, v, dp, dt=0.1, time=5, a=100):
        # defining CPG class attributes.
        super().__init__(dt, a, R, v, dp)
        self.exp_info = f"| AmphiBot_II | R = {R} | v = {v} | dp = {dp} |"
        self.time = time
        self.steps = 0

    def step(self):
        # updating the CPG state for a new timestep.
        self.new_r()
        self.new_O()
        self.new_x()
        self.new_sp_PD()
        self.steps += 1
        return

    def swim(self):
        # looping over timesteps of 0.001s to generate CPG oscillator behaviours for different CPG parameter values.
        t = 0
        while t < self.time:
            self.step()
            t = self.steps * self.dt
            if t % 1 == 0:
                print(f"time {t}")
            if t == 5:
                self.R = 1.5
                self.v = 1
            if t == 10:
                self.R = 0.75
                self.v = 1
            if t == 15:
                self.v = 1
                self.dp = -1
                self.R = 0.75
                self.make_phi_w()

    def plot_joints(self):
        # plotting CPG oscillator behaviour.
        import matplotlib.pyplot as plt
        #print(self.x_rec[0, :])
        times = np.linspace(0, self.time, num=int(self.time / self.dt + 1))
        plt.plot(times, self.x_rec[0, :])
        #plt.plot(times, self.sp_rec[0, :])
        plt.title(self.exp_info)
        plt.ylabel("x")
        plt.xlabel("Time (sec)")
        plt.show()

if __name__ == "__main__":
    # defining CPG parameters and running CPG oscillation, then plotting.
    R = 1
    v = 1
    dp = 1
    dt = 0.01
    sea_snake = amphibot_CPG(R, v, dp, dt, time=20)
    sea_snake.swim()
    sea_snake.plot_joints()
