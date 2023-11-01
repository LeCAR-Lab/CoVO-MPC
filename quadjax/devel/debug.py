import matplotlib.pyplot as plt
import jax
from jax import numpy as jnp

H = 6
action_dim = 2
N = 1000
key = jax.random.PRNGKey(0)

a_mean = jnp.linspace(0, 1, action_dim*H).reshape((H, action_dim))
a_cov = jnp.tile(jnp.eye(action_dim)*0.04, (H, 1, 1))

sample_key, key = jax.random.split(key)
keys = jax.random.split(sample_key, N)
def single_sample(key, traj_mean, traj_cov):
    subkeys = jax.random.split(key, H)
    return jax.vmap(lambda key, mean, cov: jax.random.multivariate_normal(key, mean, cov))(subkeys, traj_mean, traj_cov)
# repeat single_sample N times to get N samples
a_sampled = jax.vmap(single_sample, in_axes=(0, None, None))(keys, a_mean, a_cov)
a_sampled_flattened = jnp.reshape(a_sampled, (N, action_dim*H))

def single_sample(key):
    return jax.random.multivariate_normal(key, a_mean.flatten(), jnp.eye(2*H)*0.04)
sample_key, key = jax.random.split(key)
keys = jax.random.split(sample_key, N)
a_sampled_flattened_new = jax.vmap(single_sample)(keys)

# get covariance matrix
a_old_cov = jnp.cov(a_sampled_flattened.T)
# plot a_old_cov as a heatmap
plt.imshow(a_old_cov)
plt.colorbar()
plt.savefig('a_old_cov.png')

a_new_cov = jnp.cov(a_sampled_flattened_new.T)
# plot a_new_cov as a heatmap
import matplotlib.pyplot as plt
plt.imshow(a_new_cov)
plt.savefig('a_new_cov.png')