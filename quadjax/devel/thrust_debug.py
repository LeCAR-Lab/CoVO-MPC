import numpy as np
from matplotlib import pyplot as plt

def main():
    dt = 0.002
    steps = 2000
    tau_thrust = 0.1
    alpha = dt / tau_thrust

    kp = 1.0
    ki = 1.0
    ki_limit = 1.0

    x = 0.0
    x_des = 10.0

    integral = 0.0
    xs = []

    for i in range(steps):
        x_err = x_des - x
        integral += x_err * dt
        integral = np.clip(integral, -ki_limit, ki_limit)
        u = kp * x_des + ki * integral
        x = (1 - alpha) * x + alpha * u
        xs.append(x)

    plt.plot(xs)
    plt.plot([x_des] * steps, label='x_des', linestyle='--')
    plt.legend()
    plt.savefig('thrust_debug.png')

if __name__ == '__main__':
    main()