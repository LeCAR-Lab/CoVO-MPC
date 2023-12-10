import jax
import chex
from flax import struct
from functools import partial
from jax import lax
from jax import numpy as jnp
import pickle

from quadjax import controllers
from quadjax.dynamics import EnvParams3D, EnvState3D


@struct.dataclass
class CoVOParams:
    gamma_mean: float  # mean of gamma
    gamma_sigma: float  # std of gamma
    discount: float  # discount factor
    sample_sigma: float  # std of sampling

    a_mean: jnp.ndarray  # mean of action
    a_cov: jnp.ndarray  # covariance matrix of action
    a_cov_offline: jnp.ndarray  # covariance matrix of action


class CoVOController(controllers.BaseController):
    def __init__(
        self, env, control_params, N: int, H: int, lam: float, mode: str = "online"
    ) -> None:
        super().__init__(env, control_params)
        self.N = N  # NOTE: N is the number of saples, set here as a static number
        self.H = H
        self.lam = lam
        self.action_dim = self.env.action_dim
        if mode == "online":
            # Key method
            def get_sigma_covo(control_params, env_state, env_params, key):
                R = self.get_hessian(
                    env_state, env_params, control_params, control_params.a_mean, key
                )
                sigma = self.optimize_sigma(R, control_params)
                return sigma

            self.get_sigma_covo = get_sigma_covo
        elif mode == "offline":
            assert (
                env.action_dim == 4
            ), "only support 4D action space Quadrotor environment for now"
            expansion_control_params = controllers.PIDParams(
                Kp=10.0,
                Kd=5.0,
                Ki=0.0,
                Kp_att=10.0,
            )
            expansion_controller = controllers.PIDController(
                env, control_params=control_params
            )

            def mppi_rollout_fn(carry, unused):
                env_state, env_params, key = carry
                rng_act, key = jax.random.split(key)
                obs = self.env.get_obs(env_state, env_params)
                action, _, _ = expansion_controller(
                    obs, env_state, env_params, rng_act, expansion_control_params
                )
                action = lax.stop_gradient(action)
                rng_step, key = jax.random.split(key)
                _, env_state, _, _, _ = self.env.step_env(
                    rng_step, env_state, action, env_params, deterministic=True
                )
                return (env_state, env_params, key), action

            def get_single_a_cov_offline(carry, unused):
                env_state, env_params, key = carry
                _, a_mean = lax.scan(
                    mppi_rollout_fn, (env_state, env_params, key), None, length=self.H
                )
                R = self.get_hessian(env_state, env_params, control_params, a_mean, key)
                a_cov = self.optimize_sigma(R, control_params)
                # step forward with lqr
                rng_step, key = jax.random.split(key)
                obs = self.env.get_obs(env_state, env_params)
                action, _, _ = expansion_controller(
                    obs, env_state, env_params, rng_step, expansion_control_params
                )
                action = lax.stop_gradient(action)
                rng_step, key = jax.random.split(key)
                _, env_state, _, _, _ = self.env.step_env(
                    rng_step, env_state, action, env_params
                )
                return (env_state, env_params, key), a_cov

            if mode == "offline":

                def get_a_cov_offline(env_state, env_params, key):
                    _, a_cov_offline = lax.scan(
                        get_single_a_cov_offline,
                        (env_state, env_params, key),
                        None,
                        length=self.env.default_params.max_steps_in_episode,
                    )
                    return a_cov_offline

            elif mode == "online":

                def get_a_cov_offline(env_state, env_params, key):
                    _, a_cov = get_single_a_cov_offline(
                        (env_state, env_params, key), None
                    )
                    a_cov_offline = jnp.repeat(
                        a_cov[None, ...],
                        self.env.default_params.max_steps_in_episode,
                        axis=0,
                    )
                    return a_cov_offline

            else:
                raise NotImplementedError

            def reset_a_cov_offline(env_state, env_params, control_params, key):
                a_cov_offline = get_a_cov_offline(env_state, env_params, key)
                control_params = control_params.replace(a_cov_offline=a_cov_offline)
                return control_params

            # Key method
            def get_sigma_covo(control_params, env_state, env_params, key):
                return control_params.a_cov_offline[env_state.time]

            self.get_sigma_covo = get_sigma_covo
            # overwrite reset function
            self.reset = reset_a_cov_offline
        else:
            raise NotImplementedError

    def optimize_sigma(self, R: jnp.ndarray, control_params: CoVOParams):
        R = (R + R.T) / 2.0
        eigns, u = jnp.linalg.eigh(R)

        min_eign = jnp.min(eigns)
        offset = -min_eign + 1e-2
        eigns = eigns + offset

        log_o = jnp.log(eigns)
        element_num = self.action_dim * self.H
        log_det_a_cov = element_num * (jnp.log(control_params.sample_sigma) * 2)
        log_const = (log_det_a_cov * 2 + jnp.sum(log_o)) / element_num
        log_s = 0.5 * log_const - 0.5 * log_o

        a_cov = u @ jnp.diag(jnp.exp(log_s)) @ u.T

        return (a_cov + a_cov.T) / 2.0  # make it symmetric

    def get_hessian(
        self,
        env_state: EnvState3D,
        env_params: EnvParams3D,
        control_params: CoVOParams,
        a_mean: jnp.ndarray,
        rng_act: chex.PRNGKey = None,
    ):
        def single_rollout_fn(carry, action):
            (
                env_state,
                env_params,
                reward_before,
                done_before,
                key,
                cumulated_reward,
            ) = carry
            rng_act, key = jax.random.split(key)
            _, env_state, reward, done, _ = self.env.step_env(
                rng_act, env_state, action, env_params, deterministic=True
            )
            cumulated_reward = cumulated_reward + reward
            return (
                env_state,
                env_params,
                reward,
                done | done_before,
                key,
                cumulated_reward,
            ), reward

        def get_cumulated_cost(a_mean_flattened, carry):
            a_mean = a_mean_flattened.reshape(self.H, -1)
            env_state, env_params, rng_act = carry

            # NOTE: use for loop instead of lax.scan to avoid the gradient disappearing problem
            cumulated_reward = 0.0
            carry = (env_state, env_params, 0.0, False, rng_act, 0.0)
            for i in range(self.H):
                carry, reward = single_rollout_fn(carry, a_mean[i])
                cumulated_reward = cumulated_reward + reward

            cumulated_reward = cumulated_reward + self.env.reward_fn(
                env_state, env_params
            )

            return -cumulated_reward

        # calculate the hessian of get_cumulated_reward
        jabobian_fn = jax.jacfwd(get_cumulated_cost, argnums=0)
        hessian_fn = jax.jacfwd(jabobian_fn, argnums=0)
        return hessian_fn(a_mean.flatten(), (env_state, env_params, rng_act))

    @partial(jax.jit, static_argnums=(0,))
    def __call__(
        self,
        obs: jnp.ndarray,
        env_state,
        env_params,
        rng_act: chex.PRNGKey,
        control_params: CoVOParams,
        info,
    ) -> jnp.ndarray:
        # inject noise to state elements
        env_state = info["noisy_state"]

        # shift operator
        a_mean_old = control_params.a_mean
        a_mean = jnp.concatenate([a_mean_old[1:], a_mean_old[-1:]])
        control_params = control_params.replace(a_mean=a_mean)

        # optimize sigma with CoVO (The key difference between CoVO and MPPI)
        a_cov = self.get_sigma_covo(control_params, env_state, env_params, rng_act)

        control_params = control_params.replace(a_cov=a_cov)

        # sample action with mean and covariance, repeat for N times to get N samples with shape (N, H, action_dim)
        # a_mean shape (H, action_dim), a_cov shape (H, action_dim, action_dim)
        rng_act, act_key = jax.random.split(rng_act)
        act_keys = jax.random.split(act_key, self.N)

        def single_sample(key):
            return jax.random.multivariate_normal(
                key, control_params.a_mean.flatten(), control_params.a_cov
            )

        a_sampled_flattened = jax.vmap(single_sample)(act_keys)
        a_sampled = a_sampled_flattened.reshape(self.N, self.H, -1)

        # rollout to get reward with lax.scan
        a_sampled = jnp.clip(a_sampled, -1.0, 1.0)  # (N, H, action_dim)
        rng_act, step_key = jax.random.split(rng_act)

        def rollout_fn(carry, action):
            env_state, env_params, reward_before, done_before = carry
            # obs, env_state, reward, done, info = jax.vmap(lambda s, a, p: self.env.step_env_wocontroller(step_key, s, a, p))(env_state, action, env_params)
            obs, env_state, reward, done, info = jax.vmap(
                lambda s, a, p: self.env.step_env(step_key, s, a, p, True)
            )(env_state, action, env_params)
            reward = jnp.where(done_before, reward_before, reward)
            return (env_state, env_params, reward, done | done_before), (
                reward,
                env_state.pos,
            )

        # repeat env_state each element to match the sample size N
        state_repeat = jax.tree_map(
            lambda x: jnp.repeat(jnp.asarray(x)[None, ...], self.N, axis=0), env_state
        )
        env_params_repeat = jax.tree_map(
            lambda x: jnp.repeat(jnp.asarray(x)[None, ...], self.N, axis=0), env_params
        )
        done_repeat = jnp.full(self.N, False)
        reward_repeat = jnp.full(self.N, 0.0)

        _, (rewards, poses) = lax.scan(
            rollout_fn,
            (state_repeat, env_params_repeat, reward_repeat, done_repeat),
            a_sampled.transpose(1, 0, 2),
            length=self.H,
        )
        # get discounted reward sum over horizon (axis=1)
        rewards = rewards.transpose(1, 0)  # (H, N) -> (N, H)
        discounted_rewards = jnp.sum(
            rewards * jnp.power(control_params.discount, jnp.arange(self.H)),
            axis=1,
            keepdims=False,
        )
        # get cost
        cost = -discounted_rewards

        # get trajectory weight
        cost_exp = jnp.exp(-(cost - jnp.min(cost)) / self.lam)

        weight = cost_exp / jnp.sum(cost_exp)

        # update trajectory mean and covariance with weight
        a_mean = jnp.sum(
            weight[:, None, None] * a_sampled, axis=0
        ) * control_params.gamma_mean + control_params.a_mean * (
            1 - control_params.gamma_mean
        )
        control_params = control_params.replace(a_mean=a_mean)

        # get action
        u = control_params.a_mean[0]

        # debug values
        info = {"pos_mean": jnp.mean(poses, axis=1), "pos_std": jnp.std(poses, axis=1)}

        return u, control_params, info