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
    # gravity: float = 1.0
    # masscart: float = 1.0
    # masspole: float = 1.0
    # total_mass: float = 1.0 + 1.0  # (masscart + masspole)
    # length: float = 1.0
    # polemass_length: float = 1.0 # (masspole * length)
    # force_mag: float = 1.0
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
    max_steps_in_episode: int = 50 * 6


class CartPole(BaseEnvironment):
    def __init__(self):
        super().__init__()
        self.obs_dim = 4
        self.action_dim = 1
        # MPPI compatiblility function
        self.step_env_wocontroller = self.step_env
        self.step_env_wocontroller_gradient = self.step_env
        self.reward_fn = self.get_reward

    @property
    def name(self) -> str:
        return "CartPole-v1"

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
        # action = jnp.clip(action, -1.0, 1.0)

        # force = params.force_mag * action[0]
        # costheta = jnp.cos(state.theta)
        # sintheta = jnp.sin(state.theta)

        # temp = (
        #     force + params.polemass_length * state.theta_dot**2 * sintheta
        # ) / params.total_mass
        # thetaacc = (params.gravity * sintheta - costheta * temp) / (
        #     params.length
        #     * (4.0 / 3.0 - params.masspole * costheta**2 / params.total_mass)
        # )
        # xacc = temp - params.polemass_length * thetaacc * costheta / params.total_mass

        fx = action[0] * params.force_mag
        mc = params.masscart
        mp = params.masspole
        l = params.length
        g = params.gravity
        theta = state.theta
        theta_dot = state.theta_dot
        s = jnp.sin(theta)
        c = jnp.cos(theta)

        temp = mc + mp * s**2
        xacc = (fx + mp * s * (l * theta_dot**2 + g * c)) / temp
        thetaacc = (-fx * c - mp * l * theta_dot**2 * c * s - (mc + mp) * g * s) / (
            l * temp
        )

        # Only default Euler integration option available here!
        x = state.x + params.tau * state.x_dot
        x_dot = state.x_dot + params.tau * xacc
        theta = state.theta + params.tau * state.theta_dot
        theta = (theta) % (2 * jnp.pi)
        theta_dot = state.theta_dot + params.tau * thetaacc

        # Important: Reward is based on previous step state
        reward = self.get_reward(state, params)

        # Update state dict and evaluate termination conditions
        state = CartPoleState(x, x_dot, theta, theta_dot, state.time + 1, action[0])
        done = self.is_terminal(state, params)

        return (
            self.get_obs(state, params),
            state,
            reward,
            done,
            {"err_pos": 0.0, "err_vel": 0.0, "hit_wall": False, "pass_wall": False},
        )

    def reset_env(
        self, key: chex.PRNGKey, params: CartPoleParams
    ) -> Tuple[chex.Array, CartPoleState]:
        """Performs resetting of environment."""
        init_state = jax.random.uniform(key, minval=-0.05, maxval=0.05, shape=(4,))
        state = CartPoleState(
            x=init_state[0],
            x_dot=init_state[1],
            theta=init_state[2]*20,
            theta_dot=init_state[3],
            time=0,
            last_action=0.0
        )
        return (
            self.get_obs(state, params),
            {"err_pos": 0.0, "err_vel": 0.0, "hit_wall": False, "pass_wall": False},
            state,
        )

    def get_reward(self, state: CartPoleState, params: CartPoleParams) -> float:
        """Returns reward for given state."""
        x = jnp.clip(state.x, -1, 1)
        x_dot_normed = jnp.clip(state.x_dot / 2, -1, 1)
        theta_normed = jnp.clip(jnp.abs(state.theta-jnp.pi) / jnp.pi, -1, 1)
        theta_dot_normed = jnp.clip(state.theta_dot / 4, -1, 1)
        reward = (
            1.0
            - (
                1.0 * theta_normed**2
                # + 0.3 * theta_dot_normed**2
                + 0.5 * x**2
                # + 0.3 * x_dot_normed**2
            )
            / 1.0
        )
        return reward

    def get_obs(self, state: CartPoleState, params: CartPoleParams) -> chex.Array:
        """Applies observation function to state."""
        return jnp.array([state.x, state.x_dot, state.theta, state.theta_dot])

    def is_terminal(self, state: CartPoleState, params: CartPoleParams) -> bool:
        """Check whether state is terminal."""
        # Check termination criteria
        done1 = jnp.logical_or(
            state.x < -params.x_threshold,
            state.x > params.x_threshold,
        )
        # done2 = jnp.logical_or(
        #     state.theta < -params.theta_threshold_radians,
        #     state.theta > params.theta_threshold_radians,
        # )

        # Check number of steps in episode termination condition
        done_steps = state.time >= params.max_steps_in_episode
        done = jnp.logical_or(done1, done_steps)
        # done = jnp.logical_or(jnp.logical_or(done1, done2), done_steps)
        return done

    def sample_params(self, key: chex.PRNGKey) -> CartPoleParams:
        """Samples random parameters."""
        return self.default_params


class EnergyController(controllers.BaseController):
    def __init__(self, env, control_params) -> None:
        super().__init__(env, control_params)
        self.ke = 0.1
        self.kp = 1.0
        self.kd = 1.0

        env_params = env.default_params
        l = env_params.length
        g = env_params.gravity
        mp = env_params.masspole
        mc = env_params.masscart
        c = jnp.cos(jnp.pi)
        s = jnp.sin(jnp.pi)
        M = jnp.array([[mc + mp, mp * l * c], [mp * l * c, mp * l**2]])
        M_inv = jnp.linalg.inv(M)
        C = jnp.zeros((2, 2))
        tau_g_dq = jnp.array([[0, 0], [0, mp * g * l]])
        B = jnp.array([[1], [0]])
        A00 = jnp.zeros((2, 2))
        A01 = jnp.eye(2)
        A10 = M_inv @ tau_g_dq
        A11 = -M_inv @ C
        A = jnp.block([[A00, A01], [A10, A11]])
        B = jnp.block([[jnp.zeros((2, 1))], [M_inv @ B]])
        Q = jnp.diag(jnp.array([10, 10, 1, 1]))
        R = jnp.eye(1)
        S = solve_continuous_are(A, B, Q, R)
        self.K_lqr = jnp.linalg.inv(R) @ B.T @ S

    @partial(jax.jit, static_argnums=(0,))
    def __call__(
        self,
        obs,
        state: CartPoleState,
        env_params,
        rng_act,
        control_params,
        env_info=None,
    ) -> jnp.ndarray:
        """Computes action for given state."""
        l = env_params.length
        g = env_params.gravity
        mp = env_params.masspole
        mc = env_params.masscart
        lam = mc / mp
        w0 = jnp.sqrt(g / l)
        theta = state.theta
        theta_dot = state.theta_dot
        x = state.x
        x_dot = state.x_dot
        z = state.x / l
        z_dot = state.x_dot / l
        c = jnp.cos(theta)
        s = jnp.sin(theta)

        # Compute normalized energy
        E_normed = 0.5 * (theta_dot**2) - (w0**2) * c
        E_normed_desired = w0**2
        E_normed_error = E_normed - E_normed_desired

        # Compute action
        z_ddot_desired = (
            self.ke * theta_dot * c * E_normed_error - self.kp * z - self.kd * z_dot
        )
        u = (lam + 1 - c**2) * z_ddot_desired - w0**2 * s * c - theta_dot**2 * s
        force = jnp.array([u * mp * l])
        action = force / env_params.force_mag

        near_equilibrium = (jnp.abs(x) < 0.15) & (jnp.abs(theta - jnp.pi) < jnp.pi / 30)
        near_equilibrium = near_equilibrium | control_params
        x = jnp.array([x, theta - jnp.pi, x_dot, theta_dot])
        lqr_action = -self.K_lqr @ x / env_params.force_mag

        action = jnp.where(near_equilibrium, lqr_action, action)

        control_params = near_equilibrium

        return (
            action,
            control_params,
            {"E_normed_error": E_normed_error, "near_equilibrium": near_equilibrium},
        )

    def reset(self, env_state=None, env_params=None, control_params=None, key=None):
        return False

@pydataclass
class Args:
    controller: str = "mppi"  # fixed, energy
    debug: bool = False


def main(args: Args):
    # setup environment
    env = CartPole()

    # setup controller
    # shared MPPI parameters
    sigma = 0.2
    N = 1024 if not args.debug else 2
    H = 32 if not args.debug else 2
    lam = 0.01
    a_mean = jnp.tile(jnp.zeros(env.action_dim), (H, 1))
    # other controllers
    if args.controller == "feedback":
        control_params = controllers.FeedbackParams(
            K=jnp.array([[-0.1, -0.3, -5.0, -1.0]])
        )
        controller = controllers.FeedbackController(
            env=env, control_params=control_params
        )
    elif args.controller == "mppi":
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
    elif args.controller == "energy":
        controller = EnergyController(env=env, control_params=False)
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
        control_params=False
        while True:
            state_seq.append(env_state)
            rng, rng_act, rng_step = jax.random.split(rng, 3)
            action, control_params, control_info = controller(
                obs, env_state, env_params, rng_act, control_params
            )
            next_obs, next_env_state, reward, done, info = env.step_env(
                rng_step, env_state, action, env_params
            )
            action_seq.append(action)
            info_seq.append(control_info)
            reward_seq.append(reward)
            if done:
                break
            else:
                obs = next_obs
                env_state = next_env_state
        cum_rewards = jnp.cumsum(jnp.array(reward_seq))
        # vis = Visualizer(env, env_params, state_seq, cum_rewards)
        # plot theta, x, E_normed_error
        theta = jnp.array([state.theta for state in state_seq])
        x = jnp.array([state.x for state in state_seq])
        theta_dot = jnp.array([state.theta_dot for state in state_seq])
        x_dot = jnp.array([state.x_dot for state in state_seq])
        E_normed_error = jnp.array([info["E_normed_error"] for info in info_seq])
        plt.figure(figsize=(5, 7))
        plt.subplot(6, 1, 1)
        plt.plot(theta)
        plt.ylabel("theta")
        plt.subplot(6, 1, 2)
        plt.plot(x)
        plt.ylabel("x")
        plt.subplot(6, 1, 3)
        plt.plot(theta_dot)
        plt.ylabel("theta_dot")
        plt.subplot(6, 1, 4)
        plt.plot(x_dot)
        plt.ylabel("x_dot")
        plt.subplot(6, 1, 5)
        plt.plot(E_normed_error)
        plt.ylabel("E_normed_error")
        plt.subplot(6, 1, 6)
        plt.plot(jnp.array(action_seq)[:, 0])
        plt.ylabel("action")
        plt.savefig("../../results/theta_x.png")
        # create animation with the state sequence
        l = env_params.length

        def update_plot(frame_num):
            plt.gca().clear()
            plt.scatter(state_seq[frame_num].x, 0, marker="o", color="red")
            near_equilibrium = info_seq[frame_num]["near_equilibrium"]
            plt.scatter(
                state_seq[frame_num].x + l * jnp.sin(state_seq[frame_num].theta),
                -l * jnp.cos(state_seq[frame_num].theta),
                marker="o",
                color="green" if near_equilibrium else "blue",
            )
            plt.xlim(-env_params.x_threshold, env_params.x_threshold)
            plt.ylim(-env_params.length, env_params.length)
            plt.gca().set_aspect("equal", adjustable="box")

        plt.figure()
        anim = FuncAnimation(plt.gcf(), update_plot, frames=len(state_seq), interval=20)
        anim.save("../../results/anim.gif", dpi=80, writer="imagemagick", fps=50)
    else:
        rngs = jax.random.split(rng, 100)
        t0 = time.time()
        rewards = jax.vmap(run_one_ep)(rngs)
        rewards = rewards * 1000
        print(f"time: {time.time() - t0:.2f}s")
        print(f"cost: ${-rewards.mean():.2f} \pm {rewards.std():.2f}$")


if __name__ == "__main__":
    main(tyro.cli(Args))
