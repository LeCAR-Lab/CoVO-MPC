import pandas as pd
from matplotlib import pyplot as plt

def main():
    data = pd.read_csv('../../results/state_seq_.csv')
    omega_x = data['omega_0']
    omega_x_des = data['omega_tar_0']

    # set figure size to 10x10
    plt.figure(figsize=(15, 5))
    plt.plot(omega_x[:1000], label='omega_x')
    plt.plot(omega_x_des[:1000], label='omega_x_des')
    plt.legend()
    plt.savefig('state_seq.png')

if __name__ == '__main__':
    main()