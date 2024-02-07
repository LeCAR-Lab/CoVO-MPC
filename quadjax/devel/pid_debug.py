import numpy as np
from matplotlib import pyplot as plt

def main():
    I = 1.7e-5
    dt = 0.002
    steps = 200
    tau_thrust = 0.1
    alpha = dt / tau_thrust

    kp = 100.0 # tau = 0.01s
    kd = 1.0
    ki = 0.0

    x = 0.0
    x_des = 0.2
    x_dot = 0.0
    last_error = 0.0
    integral = 0.0
    xs = []
    x_dots = []
    x_dot_dess = []

    for i in range(steps):
        x_err = x_des - x
        x_dot_err = (x_err - last_error) / dt
        last_error = x_err
        x_dot_des = np.clip(kp * x_err + kd * x_dot_err + ki * integral, -200, 200) 
        integral += (x_des - x) * dt
        x_dot_dess.append(x_dot_des)
        x_dot = alpha * x_dot_des + (1 - alpha) * x_dot
        x_dots.append(x_dot)
        x = x + x_dot * dt
        xs.append(x)

    plt.plot(xs)
    # plt.plot(x_dot_dess, label='x_dot_des')
    # plt.plot(x_dots, label='x_dot')
    plt.plot([x_des] * steps, label='x_des', linestyle='--')
    plt.legend()
    plt.savefig('pid_debug.png')

if __name__ == '__main__':
    main()