import jax
import jax.numpy as jnp
from jax import lax
from gymnax.environments import environment, spaces
from typing import Tuple, Optional
import chex
from flax import struct
import tyro
from dataclasses import dataclass as pydataclass
import time

from quadjax.envs.base import BaseEnvironment
from quadjax import controllers

@struct.dataclass
class CartPoleState:
    x: float
    x_dot: float
    theta: float
    theta_dot: float
    time: int
    last_action: float


@struct.dataclass
class CartPoleParams:
    gravity: float = 9.8
    masscart: float = 1.0
    masspole: float = 0.1
    total_mass: float = 1.0 + 0.1  # (masscart + masspole)
    length: float = 0.5
    polemass_length: float = 0.05  # (masspole * length)
    force_mag: float = 10.0
    tau: float = 0.02
    theta_threshold_radians: float = 12 * 2 * jnp.pi / 360
    x_threshold: float = 2.4
    max_steps_in_episode: int = 500  # v0 had only 200 steps!


class CartPole(BaseEnvironment):
    def __init__(self):
        super().__init__()
        self.action_dim = 1
        # MPPI compatiblility function
        self.step_env_wocontroller = self.step_env
        self.step_env_wocontroller_gradient = self.step_env

    @property
    def default_params(self) -> CartPoleParams:
        return CartPoleParams()

    def step_env(
        self,
        key: chex.PRNGKey,
        state: CartPoleState,
        action: jnp.array,
        params: CartPoleParams,
    ) -> Tuple[chex.Array, CartPoleState, float, bool, dict]:
        """Performs step transitions in the environment."""
        action = jnp.clip(action, -1.0, 1.0)
        force = params.force_mag * action[0]
        costheta = jnp.cos(state.theta)
        sintheta = jnp.sin(state.theta)

        temp = (
            force + params.polemass_length * state.theta_dot**2 * sintheta
        ) / params.total_mass
        thetaacc = (params.gravity * sintheta - costheta * temp) / (
            params.length
            * (4.0 / 3.0 - params.masspole * costheta**2 / params.total_mass)
        )
        xacc = temp - params.polemass_length * thetaacc * costheta / params.total_mass

        # Only default Euler integration option available here!
        x = state.x + params.tau * state.x_dot
        x_dot = state.x_dot + params.tau * xacc
        theta = state.theta + params.tau * state.theta_dot
        theta_dot = state.theta_dot + params.tau * thetaacc

        # Important: Reward is based on previous step state
        reward = self.get_reward(state, params)

        # Update state dict and evaluate termination conditions
        state = CartPoleState(x, x_dot, theta, theta_dot, state.time + 1, action[0])
        done = self.is_terminal(state, params)

        return (
            self.get_obs(state),
            state,
            reward,
            done,
            {},
        )

    def reset_env(
        self, key: chex.PRNGKey, params: CartPoleParams
    ) -> Tuple[chex.Array, CartPoleState]:
        """Performs resetting of environment."""
        init_state = jax.random.uniform(key, minval=-0.05, maxval=0.05, shape=(4,))
        state = CartPoleState(
            x=init_state[0],
            x_dot=init_state[1],
            theta=init_state[2],
            theta_dot=init_state[3],
            time=0,
            last_action=0.0,
        )
        return self.get_obs(state), state

    def get_reward(self, state: CartPoleState, params: CartPoleParams) -> float:
        """Returns reward for given state."""
        reward = (
            -1.0 * state.theta**2
            - 0.1 * state.theta_dot**2
            - 0.001 * state.x**2
            - 0.001 * state.x_dot**2
            - 0.1 * state.last_action**2
        )
        return reward

    def get_obs(self, state: CartPoleState) -> chex.Array:
        """Applies observation function to state."""
        return jnp.array([state.x, state.x_dot, state.theta, state.theta_dot])

    def is_terminal(self, state: CartPoleState, params: CartPoleParams) -> bool:
        """Check whether state is terminal."""
        # Check termination criteria
        done1 = jnp.logical_or(
            state.x < -params.x_threshold,
            state.x > params.x_threshold,
        )
        done2 = jnp.logical_or(
            state.theta < -params.theta_threshold_radians,
            state.theta > params.theta_threshold_radians,
        )

        # Check number of steps in episode termination condition
        done_steps = state.time >= params.max_steps_in_episode
        done = jnp.logical_or(jnp.logical_or(done1, done2), done_steps)
        return done


@pydataclass
class Args:
    task: str = "cartpole"
    controller: str = "mppi"  # fixed
    debug: bool = False


def main(args: Args):
    # setup environment
    if args.task == "cartpole":
        env = CartPole()
    else:
        raise

    # setup controller
    if args.controller == "pid":
        pass
    elif args.controller == "mppi":
        sigma = 0.5
        N = 1024 if not args.debug else 2
        H = 32 if not args.debug else 2
        lam = 0.01

        a_mean = jnp.tile(jnp.zeros(env.action_dim), (H, 1))
        sigmas = jnp.array([sigma] * env.action_dim)
        a_cov_per_step = jnp.diag(sigmas**2)
        a_cov = jnp.tile(a_cov_per_step, (H, 1, 1))
        control_params = controllers.MPPIParams(
            gamma_mean=1.0,
            gamma_sigma=0.0,
            discount=1.0,
            sample_sigma=sigma,
            a_mean=a_mean,
            a_cov=a_cov,
            obs_noise_scale=0.0,
        )
        controller = controllers.MPPIController(
            env=env, control_params=control_params, N=N, H=H, lam=lam
        )
    elif "covo" in args.controller:
        if 'offline' in args.controller:
            expansion_mode = 'pid'
        else:
            expansion_mode = 'mean'
        
    else:
        raise NotImplementedError
    
    # run experiment
    rng = jax.random.PRNGKey(0)
    def run_one_ep(rng):
        # reset env and controller
        env_params = env.default_params
        rng_reset, rng = jax.random.split(rng)
        obs, env_state = env.reset_env(rng_reset, env_params)
        rng_control, rng = jax.random.split(rng)
        control_params = controller.reset(
            env_state, env_params, controller.init_control_params, rng_control
        )

        def run_one_step(carry, _):
            obs, env_state, rng, env_params, control_params = carry
            rng, rng_act, rng_step = jax.random.split(rng, 3)
            action, control_params, control_info = controller(
                obs, env_state, env_params, rng_act, control_params
            )
            next_obs, next_env_state, reward, done, info = env.step_env(
                rng_step, env_state, action, env_params
            )
            return (next_obs, next_env_state, rng, env_params, control_params), (reward, done)
        # run one episode
        (obs, env_state, rng, env_params, control_params), (reward, dones) = lax.scan(
            run_one_step,
            (obs, env_state, rng, env_params, control_params),
            jnp.arange(env.default_params.max_steps_in_episode),
        )
        return reward.mean()
    # run multiple episodes
    rngs = jax.random.split(rng, 100)
    t0 = time.time()
    rewards = jax.vmap(run_one_ep)(rngs)
    reward = run_one_ep(rng)
    print(f'time: {time.time() - t0:.2f}s')

    print(f'reward: ${rewards.mean():.2f} \pm {rewards.std():.2f}$')


if __name__ == "__main__":
    main(tyro.cli(Args))