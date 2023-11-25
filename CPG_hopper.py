import math as m
import numpy as np

"""
Defines a CPG oscillator class for the 1-leg, 3-joint Hopper agent: CPG_hopper.
Parameters given are mu (r target), om (theta'), dp (phase couplings), dt (integration step size). 
Has functionality for plotting CPG set points x = r * cos(theta).
"""

class chain_oscillator():
    def __init__(self, mu, om, dp, dt, a):
        # defining the CPG oscillator behaviour.
        self.mu = mu
        self.om = om
        self.dp = dp
        self.dt = dt
        self.a = a
        self.make_phi_w()
        self.dr2dt = np.zeros((3, 1))
        self.drdt = np.zeros((3, 1))
        self.r = np.zeros((3, 1))
        self.dOdt = np.zeros((3, 1))
        self.O = np.zeros((3, 1))
        self.x = np.full((3, 1), 0.5)
        self.x_rec = np.copy(np.expand_dims(self.x[:, 0], axis=1))

    def new_r(self):
        # integrating for new r values from r'' with Euler integration.
        self.dr2dt = self.a * (self.a / 4 * (self.mu - self.r) - self.drdt)
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
        couplesum = np.expand_dims(couplesum, axis=1)
        self.dOdt = self.om + couplesum
        self.O += self.dOdt * self.dt

    def new_x(self):
        # calculating new x values from newly updated r and theta. Recording in x_rec.
        self.x = self.r * (np.cos(self.O))
        self.x_rec = np.append(self.x_rec, np.expand_dims(self.x[:, 0], axis=1), axis=1)

    def make_phi_w(self):
        # building new phi and w matrices for a given delta phi value.
        self.phi = np.zeros((3, 3))
        self.w = np.zeros((3, 3))
        for i in range(self.phi.shape[0]):
            for j in range(self.phi.shape[1]):
                if i == j + 1:
                    self.phi[i, j] = -self.dp[j]
                    self.w[i, j] = 1
                elif j == i + 1:
                    self.phi[i, j] = self.dp[i]
                    self.w[i, j] = 1


class CPG_hopper(chain_oscillator):
    def __init__(self, mu, om, dp, dt=0.01, time=5, a=100):
        # defining CPG class attributes.
        super().__init__(mu, om, dp, dt, a)
        mu_round = [round(float(i), 1) for i in mu]
        om_round = [round(float(i), 1) for i in om]
        dp_round = [round(float(i), 1) for i in dp]
        self.exp_info = f"CPG_Hopper Torques (Nm) | mu = {mu_round} | om ~ {om_round} | dp ~ {dp_round} | dt = {dt}"
        self.time = time
        self.steps = 0

    def hop(self):
        # updating the CPG state for a new timestep.
        self.new_r()
        self.new_O()
        self.new_x()
        self.steps += 1
        return self.x[:, 0]

    def hopping(self):
        # looping over timesteps of 0.001s to generate CPG oscillator behaviours for different CPG parameter values.
        t = 0
        while t < self.time:
            self.hop()
            t = self.steps * self.dt
            if t % 1 == 0:
                print(f"time {t}")
            if t == 5:
                self.mu = np.array([[2], [2], [2]])
                self.om = 2 * m.pi * np.array([[0.5], [0.5], [0.5]])
            if t == 10:
                pass
            if t == 15:
                self.dp += 1
                self.make_phi_w()

    def subplot_joints(self, block):
        # plotting CPG oscillator behaviour.
        import matplotlib.pyplot as plt

        times = np.linspace(0, self.time, num=int(self.time / self.dt + 1))
        hip_joint = self.x_rec[0, :]
        knee_joint = self.x_rec[1, :]
        ankle_joint = self.x_rec[2, :]

        fig, axs = plt.subplots(3, 1, figsize=(10,8))

        # Plotting for the hip_joint
        axs[0].plot(times, hip_joint, 'tab:blue')
        axs[0].set_title('Hip Joint')
        axs[0].set_ylabel('Set Point')

        # Plotting for the knee_joint
        axs[1].plot(times, knee_joint, 'tab:orange')
        axs[1].set_title('Knee Joint')
        axs[1].set_ylabel('Set Point')

        # Plotting for the ankle_joint
        axs[2].plot(times, ankle_joint, 'tab:green')
        axs[2].set_title('Ankle Joint')
        axs[2].set_ylabel('Set Point')
        axs[2].set_xlabel('Time (s)')

        # Adjust the layout to prevent overlap
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        # Figure scale details
        plt.suptitle(self.exp_info, fontsize=12)
        plt.show(block=block)

if __name__ == "__main__":
    # defining CPG parameters and running CPG oscillation, then plotting.
    mu = np.ones((3, 1))
    om = 2 * m.pi * np.ones((3, 1))
    dp = m.pi * np.ones(2)
    dt = 0.01
    hopper = CPG_hopper(mu, om, dp, dt, time=20)
    hopper.hopping()
    hopper.subplot_joints(True)