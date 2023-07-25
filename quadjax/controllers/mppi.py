import jax
from jax import numpy as jnp
from typing import Sequence, NamedTuple, Any

from quadjax.quad3d import Quad3D
from quadjax.dynamics.dataclass import EnvParams3D, EnvState3D


def quad3d_free_mppi_policy(
    obs: jnp.ndarray,
    env_state: EnvState3D,
    env_params: EnvParams3D,
    rng: jax.random.PRNGKey,
    env: Quad3D,
    old_a_mean: jnp.ndarray, 
    old_a_sigma: jnp.ndarray, 
):
    horizon = 10
    sample_num = 128
    discount = 0.99
    lam = 0.003  # temperature
    gamma_mean = 0.9  # learning rate
    gamma_sigma = 0.0  # learning rate
    action_dim = 4

    # set new action by shift old_a_mean and old_a_sigma to the right for 1 step
    a_mean = jnp.concatenate([old_a_mean[1:], jnp.zeros([1, action_dim], dtype=jnp.float32)], axis=0)
    a_sigma = jnp.concatenate([old_a_sigma[1:], jnp.eye(action_dim, dtype=jnp.float32)[None, ...]], axis=0)
    # sample action
    rng, _rng = jax.random.split(rng)
    def sample_for_each_batch(rng):
        rngs = jax.random.split(rng, horizon)
        samples = jax.vmap(jax.random.multivariate_normal)(rngs, a_mean, a_sigma)
        return samples
    actions_sampled = jax.vmap(sample_for_each_batch)(jax.random.split(_rng, sample_num))
    actions_sampled = jnp.transpose(actions_sampled, (1, 0, 2))  # (horizon, sample_num, action_dim)
    actions_sampled = jnp.clip(actions_sampled, -1.0, 1.0)

    def _env_step(runner_state, action):
        env_state, rng, env_params = runner_state

        # STEP ENV
        rng, _rng = jax.random.split(rng)
        rng_step = jax.random.split(_rng, sample_num)
        obsv, env_state, reward, done, info = jax.vmap(env.step)(
            rng_step, env_state, action, env_params
        )
        # resample environment parameters if done
        rng_params = jax.random.split(_rng, sample_num)
        new_env_params = jax.vmap(env.sample_params)(rng_params)

        def map_fn(done, x, y):
            reshaped_done = done.reshape([done.shape[0]] + [1] * (x.ndim - 1))
            return reshaped_done * x + (1 - reshaped_done) * y

        env_params = jax.tree_map(
            lambda x, y: map_fn(done, x, y), new_env_params, env_params
        )

        runner_state = (env_state, rng, env_params)
        return runner_state, (reward, done)

    # rollout environment
    # repeat env_state for sample_num times
    env_state_batch = jax.tree_map(lambda x: jnp.repeat(x[None, ...], sample_num, axis=0), env_state)
    # repeat env_params for sample_num times
    env_params_batch = jax.tree_map(lambda x: jnp.repeat(x[None, ...], sample_num, axis=0), env_params)
    runner_state, traj_batch = jax.lax.scan(_env_step, (env_state_batch, rng, env_params_batch), actions_sampled)
    reward_batch = traj_batch[0]
    done_batch = traj_batch[1]

    # compute discounted cost
    terminate_mask = jnp.cumsum(done_batch, axis=0).astype(bool)
    first_terminate_mask = (jnp.cumsum(terminate_mask, axis=0) == 1)
    reward_terminated = reward_batch * (1 - terminate_mask) + jnp.cumsum(reward_batch * first_terminate_mask, axis=0)
    discount_factor = discount ** jnp.arange(horizon)
    cost = - jnp.sum(reward_terminated * discount_factor[:, None], axis=0)
    # compute weight
    cost -= jnp.min(cost)
    cost = jnp.exp(-1.0 / lam * cost)
    weight = cost / jnp.sum(cost)  # tensor, (N,)

    # get new action
    actions_sampled_T = actions_sampled.transpose(1,0,2)
    samples_T = actions_sampled_T - a_mean[None, :, :] # (batch, horizon, action_dim)
    a_mean_new = (1 - gamma_mean) * a_mean + gamma_mean * jnp.sum(weight[:, None, None] * actions_sampled_T, axis=0) # (horizon, action_dim)
    a_sigma_new = (1 - gamma_sigma) * a_sigma + gamma_sigma * jnp.sum(weight[:, None, None, None] * jnp.einsum("...i,...j->...ij", samples_T, samples_T), axis=0) # (horizon, action_dim, action_dim)

    return a_mean_new[0], {
        "a_mean": a_mean_new,
        "a_sigma": a_sigma_new,
    }