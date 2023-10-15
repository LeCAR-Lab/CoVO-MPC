import os
from matplotlib import pyplot as plt
import numpy as np
import pickle
from jax import numpy as np
import seaborn as sns
import pandas as pd

# find all .pkl file in ../../results/init/
os.chdir('../../results/rma/')
files = os.listdir()
files = [f for f in files if f.endswith('.pkl')]
files = [f for f in files if f.startswith('eval_err_pos')]
files = [f for f in files if 'DR' not in f]
files = ['eval_err_pos_A-DATT.pkl']

# load all files
all_data = pd.DataFrame(
    columns=['algo', 'error', 'mean', 'std']
)

for file in files:
    file_name = file.split(".")[0]
    exp_name = file_name[13:]
    with open(file, 'rb') as f:
        data = pickle.load(f)
    data = np.array(data).flatten()
    # append N and error to all_data
    all_data = pd.concat(
        [
            all_data,
            pd.DataFrame(
                {
                    'algo': [f'{exp_name} \n mu:{np.mean(data):.2f} \n std:{np.std(data):.2f}']*len(data),
                    'error': data,
                    'mean': [np.mean(data)]*len(data),
                    'std': [np.std(data)]*len(data),
                }
            )
        ]
    )

# plot in a order from largest mean to smallest mean
sns.set_theme(style="whitegrid")
fig, ax = plt.subplots(figsize=(10, 5))
print('start plotting...')
sns.violinplot(x='algo', y='error', data=all_data, ax=ax)
print('done')
ax.set_xlabel('algo')
ax.set_ylabel('error')