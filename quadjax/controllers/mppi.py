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
):
    sample_num = 128
    discount = 0.99
    def _env_step(runner_state, action):
        train_state, env_state, last_obs, rng, env_params = runner_state

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

        runner_state = (train_state, env_state, obsv, rng, env_params)
        return runner_state, (reward, done)
    # sample action
    rng, _rng = jax.random.split(rng)
    rng_action = jax.random.split(_rng, sample_num)
    action = jax.vmap(env.sample_action)(rng_action)
    # rollout environment
    runner_state, traj_batch = jax.lax.scan(
        _env_step, runner_state, None, config["NUM_STEPS"]
    )
    reward_batch = traj_batch[0]
    done_batch = traj_batch[1]
    # compute discounted reward
    def _discount_reward(reward, done):
        return jax.lax.scan(
            lambda c, x: (x[0] + discount * c[0], x[1]),
            (0.0, False),
            (reward, done),
        )[0]
    discounted_reward_batch = jax.vmap(_discount_reward)(reward_batch, done_batch)
    