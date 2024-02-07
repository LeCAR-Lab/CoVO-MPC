import pandas as pd
from matplotlib import pyplot as plt

def main():
    data = pd.read_csv('../../results/state_seq_quad3d_fine_tracking_pid_pwm.csv')
    omega_x = data['omega_0']
    omega_x_des = data['omega_tar_0']
    thrust = data['last_thrust']

    print(data.keys())

    # set figure size to 10x10
    plt.figure(figsize=(15, 5))
    # plt.plot(omega_x[:1000], label='omega_x')
    # plt.plot(omega_x_des[:1000], label='omega_x_des')
    plt.plot(thrust[:1000]/0.5, label='thrust')
    plt.legend()
    plt.savefig('state_seq.png')

if __name__ == '__main__':
    main()