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
from gymnax.visualize import Visualizer
from functools import partial
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
import control as pycontrol
from scipy.linalg import solve_continuous_are

import quadjax
from quadjax.envs.base import BaseEnvironment
from quadjax import controllers


@struct.dataclass
class PendulumState:
    theta: float
    theta_dot: float
    last_u: float  # Only needed for rendering
    time: int


@struct.dataclass
class PendulumParams:
    dt: float = 0.05
    m: float = 1.0  # mass
    b: float = 0.1  # friction
    l: float = 1.0  # length
    g: float = 9.8  # gravity
    max_torque: float = 2.0
    max_steps_in_episode: int = 200

    q1: float = 10.0
    q2: float = 1.0
    r: float = 1.0


class Pendulum(BaseEnvironment):
    """
    JAX Compatible version of Pendulum-v0 OpenAI gym environment. Source:
    github.com/openai/gym/blob/master/gym/envs/classic_control/pendulum.py
    """

    def __init__(self):
        super().__init__()
        self.obs_dim = 2
        self.action_dim = 1
        # MPPI compatiblility function
        self.step_env_wocontroller = self.step_env
        self.step_env_wocontroller_gradient = self.step_env
        self.reward_fn = self.get_reward

    @property
    def default_params(self) -> PendulumParams:
        """Default environment parameters for Pendulum-v0."""
        return PendulumParams()

    def get_reward(self, state: PendulumState, params: PendulumParams) -> float:
        return -(
            # params.q1 * jnp.sin(state.theta)**2
            params.q1 * (jnp.cos(state.theta)-1)**2
            + params.q2 * state.theta_dot**2
            + params.r * state.last_u**2
        ).squeeze()
        # return -(
        #     angle_normalize(state.theta) ** 2
        #     + 0.1 * state.theta_dot**2
        #     + 0.001 * (state.last_u**2)
        # ).squeeze()

    def step_env(
        self,
        key: chex.PRNGKey,
        state: PendulumState,
        action: float,
        params: PendulumParams,
    ) -> Tuple[chex.Array, PendulumState, float, bool, dict]:
        """Integrate pendulum ODE and return transition."""
        action = action * params.max_torque
        u = jnp.clip(action, -params.max_torque, params.max_torque)
        state = state.replace(last_u=u)

        reward = self.get_reward(state, params)

        newthdot = state.theta_dot + (
            (
                params.m * params.g * params.l * jnp.sin(state.theta)
                + 1.0 * u 
                - 1.0 * params.b * state.theta_dot
            ) / (params.m * params.l**2)
            * params.dt
        )

        # newthdot = jnp.clip(newthdot, -params.max_speed, params.max_speed)
        newth = state.theta + newthdot * params.dt

        # Update state dict and evaluate termination conditions
        state = PendulumState(
            newth.squeeze(), newthdot.squeeze(), u.reshape(), state.time + 1
        )
        done = self.is_terminal(state, params)

        return (
            lax.stop_gradient(self.get_obs(state)),
            lax.stop_gradient(state),
            reward,
            done,
            {"err_pos": 0.0, "err_vel": 0.0, "hit_wall": False, "pass_wall": False},
        )

    def reset_env(
        self, key: chex.PRNGKey, params: PendulumParams
    ) -> Tuple[chex.Array, PendulumState]:
        """Reset environment state by sampling theta, theta_dot."""
        high = jnp.array([-1, 1])
        state = jax.random.uniform(key, shape=(2,), minval=-high, maxval=high)
        state = PendulumState(
            theta=state[0] * 0.05 + jnp.pi/2, theta_dot=0.0, last_u=0.0, time=0
        )
        return (
            self.get_obs(state),
            {"err_pos": 0.0, "err_vel": 0.0, "hit_wall": False, "pass_wall": False},
            state,
        )

    def get_obs(self, state: PendulumState) -> chex.Array:
        """Return angle in polar coordinates and change."""
        return jnp.array(
            [
                jnp.cos(state.theta),
                jnp.sin(state.theta),
                state.theta_dot,
            ]
        ).squeeze()

    def is_terminal(self, state: PendulumState, params: PendulumParams) -> bool:
        """Check whether state is terminal."""
        return state.time >= params.max_steps_in_episode


def angle_normalize(x: float) -> float:
    """Normalize the angle - radians."""
    return ((x + jnp.pi) % (2 * jnp.pi)) - jnp.pi


@pydataclass
class Args:
    controller: str = "mppi"  # fixed, energy
    debug: bool = False


def main(args: Args):
    # setup environment
    env = Pendulum()

    sigma = 0.5 #0.5
    N = 1024  # 16384
    H = 64 #64
    lam = 0.01 #0.01

    a_mean = jnp.tile(jnp.zeros(env.action_dim), (H, 1))
    # load a_mean
    # a_mean = jnp.load("../../results/action_seq.npy")[:H]
    # other controllers
    if args.controller == "mppi":
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
    elif args.controller == "mppi_spline":
        sigmas = jnp.array([sigma] * env.action_dim)
        a_cov_per_step = jnp.diag(sigmas**2)
        N = 32
        h = 5
        n = 10
        H = (h-1)*n + 1
        a_cov = jnp.tile(a_cov_per_step, (h, 1, 1))

        control_params = controllers.MPPISplineParams(
            gamma_mean=1.0,
            gamma_sigma=0.0,
            discount=1.0,
            sample_sigma=sigma,
            a_mean=jnp.zeros((H, env.action_dim)),
            a_cov=a_cov,
            obs_noise_scale=0.0,
        )
        controller = controllers.MPPISplineController(
            env=env, control_params=control_params, N=N, h=h, lam=1e-2, n=n,
        )
    elif "covo" in args.controller:
        expansion_mode = "feedback" if "offline" in args.controller else "mean"
        a_cov = jnp.diag(jnp.ones(H * env.action_dim) * sigma**2)
        control_params = controllers.MPPIZejiParams(
            gamma_mean=1.0,
            gamma_sigma=0.0,
            discount=1.0,
            sample_sigma=sigma,
            a_mean=a_mean,
            a_cov=a_cov,
            a_cov_offline=jnp.zeros((H, env.action_dim, env.action_dim)),
            obs_noise_scale=0.0,
        )
        controller = controllers.MPPIZejiController(
            env=env,
            control_params=control_params,
            N=N,
            H=H,
            lam=lam,
            expansion_mode=expansion_mode,
        )
    else:
        raise NotImplementedError

    # run experiment
    rng = jax.random.PRNGKey(0)

    def run_one_ep(rng):
        # reset env and controller
        env_params = env.default_params
        rng_reset, rng = jax.random.split(rng)
        obs, env_info, env_state = env.reset_env(rng_reset, env_params)
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
            return (next_obs, next_env_state, rng, env_params, control_params), (
                reward,
                done,
            )

        # run one episode
        (obs, env_state, rng, env_params, control_params), (reward, dones) = lax.scan(
            run_one_step,
            (obs, env_state, rng, env_params, control_params),
            jnp.arange(env.default_params.max_steps_in_episode),
        )
        return reward.mean()

    # run multiple episodes
    if args.debug:
        state_seq, reward_seq, info_seq, action_seq = [], [], [], []
        rng, rng_reset = jax.random.split(rng)
        env_params = env.default_params
        obs, _, env_state = env.reset_env(rng_reset, env_params)
        step_cnt = 0
        while True:
            state_seq.append(env_state)
            rng, rng_act, rng_step = jax.random.split(rng, 3)
            action, control_params, control_info = controller(
                obs, env_state, env_params, rng_act, control_params
            )

            # theta = env_state.theta
            # theta_dot = env_state.theta_dot
            # theta_repeat = jnp.repeat(jnp.asarray(theta)[None, ...], N, axis=0)[jnp.newaxis, ...]
            # theta_dot_repeat = jnp.repeat(jnp.asarray(theta_dot)[None, ...], N, axis=0)[jnp.newaxis, ...]
            # thetas = control_info["thetas"]
            # theta_dots = control_info["theta_dots"]
            # thetas = jnp.concatenate([theta_repeat, thetas], axis=0)
            # theta_dots = jnp.concatenate([theta_dot_repeat, theta_dots], axis=0)

            # jnp.save(f"../../results/highpen_thetas_{step_cnt}.npy", thetas)
            # jnp.save(f"../../results/highpen_theta_dots_{step_cnt}.npy", theta_dots)

            # save control_info to a file
            # cost = control_info["cost"]
            # a_sampled = control_info["a_sampled"]
            # import pickle
            # with open("../../results/cost.pkl", "wb") as f:
            #     pickle.dump(cost, f)
            # with open("../../results/a_sampled.pkl", "wb") as f:
            #     pickle.dump(a_sampled, f)
            # with open("../../results/a_mean.pkl", "wb") as f:
            #     pickle.dump(a_mean, f)
            # with open("../../results/a_cov.pkl", "wb") as f:
            #     pickle.dump(a_cov, f)
            # exit()

            next_obs, next_env_state, reward, done, info = env.step_env(
                rng_step, env_state, action, env_params
            )
            action_seq.append(action)
            info_seq.append(control_info)
            reward_seq.append(reward)
            step_cnt += 1
            if done:
                break
            else:
                obs = next_obs
                env_state = next_env_state
        cum_rewards = jnp.cumsum(jnp.array(reward_seq))
        # vis = Visualizer(env, env_params, state_seq, cum_rewards)
        # plot theta, x, E_normed_error
        theta = jnp.array([state.theta for state in state_seq])
        theta_dot = jnp.array([state.theta_dot for state in state_seq])
        action_seq = jnp.array(action_seq)
        plt.figure(figsize=(5, 7))
        plt.subplot(3, 1, 1)
        plt.plot(theta)
        plt.ylabel("theta")
        plt.subplot(3, 1, 2)
        plt.plot(theta_dot)
        plt.ylabel("theta_dot")
        plt.subplot(3, 1, 3)
        plt.plot(action_seq[:, 0])
        plt.ylabel("action")
        plt.savefig("../../results/theta_x.png")
        # create animation with the state sequence
        l = env_params.l

        print("saving animation...")

        def update_plot(frame_num):
            plt.gca().clear()
            plt.scatter(
                l * jnp.sin(state_seq[frame_num].theta),
                l * jnp.cos(state_seq[frame_num].theta),
                marker="o",
                color="blue",
            )
            plt.xlim(-l, l)
            plt.ylim(-l, l)
            plt.gca().set_aspect("equal", adjustable="box")

        plt.figure()
        anim = FuncAnimation(plt.gcf(), update_plot, frames=len(state_seq), interval=50)
        anim.save("../../results/anim.gif", dpi=80, writer="imagemagick", fps=20)
        # save action_seq
        # jnp.save("../../results/action_seq.npy", action_seq)
    else:
        rngs = jax.random.split(rng, 100)
        t0 = time.time()
        rewards = jax.vmap(run_one_ep)(rngs)
        rewards = rewards
        print(f"time: {time.time() - t0:.2f}s")
        print(f"cost: ${-rewards.mean():.2f} \pm {rewards.std():.2f}$")


if __name__ == "__main__":
    main(tyro.cli(Args))