import jax
import chex
from flax import struct
from functools import partial
from jax import lax
from jax import numpy as jnp

from quadjax import controllers
from quadjax.dynamics import EnvParams2D, EnvState2D

@struct.dataclass
class MPPIParams:
    lam: float # temperature
    H: int # horizon
    N: int # number of samples
    gamma_mean: float # mean of gamma
    gamma_sigma: float # std of gamma
    discount: float # discount factor
    sample_sigma: float # std of sampling

    a_mean: jnp.ndarray # mean of action
    a_cov: jnp.ndarray # covariance matrix of action


class MPPIController2D(controllers.BaseController):
    def __init__(self, env) -> None:
        super().__init__(env)


    @partial(jax.jit, static_argnums=(0,))
    def __call__(self, obs:jnp.ndarray, state: EnvState2D, env_params: EnvParams2D, rng_act: chex.PRNGKey, control_params: MPPIParams) -> jnp.ndarray:
        # shift operator
        a_mean_old = control_params.a_mean
        a_cov_old = control_params.a_cov
        control_params = control_params.replace(a_mean=jnp.concatenate([a_mean_old[1:], a_mean_old[-1:]]),
                                                 a_cov=jnp.concatenate([a_cov_old[1:], a_cov_old[-1:]]))

        # sample action with mean and covariance, repeat for N times to get N samples with shape (N, H, action_dim)
        # a_mean shape (H, action_dim), a_cov shape (H, action_dim, action_dim)
        rng_act, act_key = jax.random.split(rng_act)
        def single_sample(key, traj_mean, traj_cov):
            return jax.vmap(lambda mean, cov: jax.random.multivariate_normal(key, mean, cov))(traj_cov, traj_mean)
        # repeat single_sample N times to get N samples
        a_sampled = jax.vmap(single_sample, in_axes=(0, None, None))(act_key, control_params.a_mean, control_params.a_cov)
        a_sampled = jnp.clip(a_sampled, -1.0, 1.0)
        # rollout to get reward with lax.scan
        rng_act, step_key = jax.random.split(rng_act)
        def rollout_fn(carry, action):
            state, params, reward_before, done_before = carry
            obs, state, reward, done, info = self.env.step_env(step_key, state, action, params)
            reward = jnp.where(done_before, reward_before, reward)
            return (state, params, reward, done | done_before), reward
        # repeat state each element to match the sample size N
        state_repeat = jax.tree_map(lambda x: jnp.repeat(x[None, ...], control_params.N, axis=0), state)
        env_params_repeat = jax.tree_map(lambda x: jnp.repeat(x[None, ...], control_params.N, axis=0), env_params)
        done_repeat = jnp.repeat(False, control_params.N)
        reward_repeat = jnp.repeat(0.0, control_params.N)
        _, rewards = lax.scan(rollout_fn, (state_repeat, env_params_repeat, done_repeat, reward_repeat), a_sampled, length=control_params.H)
        # get discounted reward sum over horizon (axis=1)
        discounted_rewards = jnp.sum(rewards * jnp.power(control_params.discount, jnp.arange(control_params.H)), axis=1)
        # get cost
        cost = -discounted_rewards

        # get trajectory weight
        cost_exp = jnp.exp(-(cost-jnp.min(cost)) / control_params.lam)
        weight = cost_exp / jnp.sum(cost_exp)

        # update trajectory mean and covariance with weight
        a_mean = jnp.sum(weight[:, None] * a_sampled, axis=0)
        a_cov = jnp.sum(weight[:, None, None] * (a_sampled - a_mean) * (a_sampled - a_mean)[:, None, :], axis=0)
        control_params = control_params.replace(a_mean=a_mean, a_cov=a_cov)

        # get action
        u = control_params.a_mean[0]

        return u, control_params