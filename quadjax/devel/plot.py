import os
from matplotlib import pyplot as plt
import numpy as np
import pickle
from jax import numpy as np
import seaborn as sns
import pandas as pd

# find all .pkl file in ../../results/init/
os.chdir('../../results/')
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
        method = 'CoVO-MPC'
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
                    'error': data,
                }
            )
        ]
    )

# plot in a order from largest mean to smallest mean
sns.set_theme(style="whitegrid")
fig, ax = plt.subplots(figsize=(10, 5))
print('start plotting...')
sns.boxplot(
    x='N',
    y='error',
    hue='method',
    data=all_data,
    ax=ax,
)
print('done')
ax.set_xlabel('N')
ax.set_ylabel('error')

# save the figure
fig.savefig('plot.png', dpi=300)