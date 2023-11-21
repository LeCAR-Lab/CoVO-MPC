import os
from matplotlib import pyplot as plt
import numpy as np
import pickle
from jax import numpy as np
import seaborn as sns
import pandas as pd

# find all .pkl file in ../../results/init/
os.chdir('../../results/l4dc_stable_N')
files = os.listdir()
files = [f for f in files if f.endswith('.pkl')]
files = [f for f in files if f.startswith('eval_err_pos_quad3d_tracking_zigzag_')]
# files = [f for f in files if 'zigzag' not in f]

# load all files
all_data = pd.DataFrame(
    columns=['method', 'error', 'N', 'H', 'lam']
)

for file in files:
    file_name, lam, _ = file.split(".")
    exp_name = file_name.split("tracking_zigzag_")[-1]
    method, rest = exp_name.split("_N")
# if ('ppo' in[ method) or ('zero' in method):
#     continue]
    if method == 'mppi_zeji_mean':
        method = 'CoVO-Online'
    elif method == 'mppi_zeji_pid':
        method = 'CoVO-Offline'
    N = int(rest.split("_")[0])
    H = int(rest.split("_")[1][1:])
    lam = float(f'0.{lam}')
    if lam != 0.01:
        continue
    print(f'loading method={method} exp_name={exp_name} N={N}, H={H}, lam={lam}')
    with open(file, 'rb') as f:
        data = pickle.load(f)
    data = np.array(data).flatten()
    # append N and error to all_data
    all_data = pd.concat(
        [
            all_data,
            pd.DataFrame(
                {
                    'method': [method]*len(data),
                    'N': [N]*len(data),
                    'H': [H]*len(data),
                    'lam': [lam]*len(data),
                    'error': data*100,
                }
            )
        ]
    )

# plot in a order from largest mean to smallest mean
sns.set_theme(style="whitegrid")
fig, ax = plt.subplots(figsize=(10, 5))
# set color set mppi to grey and CoVO-Offline to light red and CoVO-Online to red
# baseline_colors = sns.light_palette("grey", n_colors=1, reverse=True)
our_method_colors = sns.light_palette("red", n_colors=3)
color_set = our_method_colors
print('start plotting...')
sns.boxplot(
    x='N',
    y='error',
    hue='method',
    data=all_data,
    ax=ax,
    palette=color_set,
)
print('done')
ax.set_xlabel('Sampling number (N)')
ax.set_ylabel('Position tracking error (cm)')

# save the figure
fig.savefig('plot.png', dpi=300)