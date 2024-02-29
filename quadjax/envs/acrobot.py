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
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from quadjax.envs.base import BaseEnvironment
from quadjax import controllers


@struct.dataclass
class AcrobotState:
    joint_angle1: float
    joint_angle2: float
    velocity_1: float
    velocity_2: float
    time: int
    last_action: float


@struct.dataclass
class AcrobotParams:
    dt: float = 0.02
    link_length_1: float = 1.0
    link_length_2: float = 1.0
    link_mass_1: float = 1.0
    link_mass_2: float = 1.0
    link_com_pos_1: float = 0.5
    link_com_pos_2: float = 0.5
    link_moi: float = 1.0
    max_vel_1: float = 4 * jnp.pi
    max_vel_2: float = 9 * jnp.pi
    torque_noise_max: float = 0.0
    max_steps_in_episode: int = 500


class Acrobot(BaseEnvironment):
    def __init__(self):
        super().__init__()
        self.action_dim = 1
        # MPPI compatiblility function
        self.step_env_wocontroller = self.step_env
        self.step_env_wocontroller_gradient = self.step_env
        self.reward_fn = self.get_reward

    @property
    def default_params(self) -> AcrobotParams:
        return AcrobotParams()

    def step_env(
        self,
        key: chex.PRNGKey,
        state: AcrobotState,
        action: jnp.array,
        params: AcrobotParams,
    ) -> Tuple[chex.Array, AcrobotState, float, bool, dict]:
        """Performs step transitions in the environment."""
        action_clip = jnp.clip(action[0], -1, 1)
        torque = action_clip * 3.0

        # Augment state with force action so it can be passed to ds/dt
        s_augmented = jnp.array(
            [
                state.joint_angle1+jnp.pi,
                state.joint_angle2,
                state.velocity_1,
                state.velocity_2,
                torque,
            ]
        )
        ns = rk4(s_augmented, params)
        joint_angle1 = (ns[0]) % (2 * jnp.pi) - jnp.pi
        joint_angle2 = (ns[1] + jnp.pi) % (2 * jnp.pi) - jnp.pi
        velocity_1 = jnp.clip(ns[2], -params.max_vel_1, params.max_vel_1)
        velocity_2 = jnp.clip(ns[3], -params.max_vel_2, params.max_vel_2)

        reward = self.get_reward(state, params)

        # Update state dict and evaluate termination conditions
        state = AcrobotState(
            joint_angle1,
            joint_angle2,
            velocity_1,
            velocity_2,
            state.time + 1,
            last_action=action_clip,
        )
        done = self.is_terminal(state, params)
        return (
            self.get_obs(state, params),
            state,
            reward,
            done,
            {},
        )

    def reset_env(
        self, key: chex.PRNGKey, params: AcrobotParams
    ) -> Tuple[chex.Array, AcrobotState]:
        """Reset environment state by sampling initial position."""
        init_state = jax.random.uniform(key, shape=(4,), minval=-0.05, maxval=0.05)
        state = AcrobotState(
            joint_angle1=init_state[0] + jnp.pi,
            joint_angle2=init_state[1],
            velocity_1=init_state[2],
            velocity_2=init_state[3],
            time=0,
            last_action=0.0,
        )
        return self.get_obs(state, params), {}, state

    def get_reward(self, state: AcrobotState, params: AcrobotParams) -> float:
        """Returns reward for given state."""
        reward = jnp.cos(state.joint_angle1) + jnp.cos(state.joint_angle1 + state.joint_angle2)
        return reward

    def get_obs(self, state: AcrobotState, params: AcrobotParams) -> chex.Array:
        """Applies observation function to state."""
        return jnp.array(
            [
                state.joint_angle1,
                state.joint_angle2,
                state.velocity_1,
                state.velocity_2,
            ]
        )

    def is_terminal(self, state: AcrobotState, params: AcrobotParams) -> bool:
        """Check whether state is terminal."""
        # Check number of steps in episode termination condition
        done = state.time >= params.max_steps_in_episode
        return done


def dsdt(s_augmented: chex.Array, t: float, params: AcrobotParams) -> chex.Array:
    """Compute time derivative of the state change - Use for ODE int."""
    m1, m2 = params.link_mass_1, params.link_mass_2
    l1 = params.link_length_1
    lc1, lc2 = params.link_com_pos_1, params.link_com_pos_2
    I1, I2 = params.link_moi, params.link_moi
    g = 9.8
    a = s_augmented[-1]
    s = s_augmented[:-1]
    theta1, theta2, dtheta1, dtheta2 = s
    d1 = (
        m1 * lc1 ** 2
        + m2 * (l1 ** 2 + lc2 ** 2 + 2 * l1 * lc2 * jnp.cos(theta2))
        + I1
        + I2
    )
    d2 = m2 * (lc2 ** 2 + l1 * lc2 * jnp.cos(theta2)) + I2
    phi2 = m2 * lc2 * g * jnp.cos(theta1 + theta2 - jnp.pi / 2.0)
    phi1 = (
        -m2 * l1 * lc2 * dtheta2 ** 2 * jnp.sin(theta2)
        - 2 * m2 * l1 * lc2 * dtheta2 * dtheta1 * jnp.sin(theta2)
        + (m1 * lc1 + m2 * l1) * g * jnp.cos(theta1 - jnp.pi / 2)
        + phi2
    )
    ddtheta2 = (
        a
        + d2 / d1 * phi1
        - m2 * l1 * lc2 * dtheta1 ** 2 * jnp.sin(theta2)
        - phi2
    ) / (m2 * lc2 ** 2 + I2 - d2 ** 2 / d1)
    ddtheta1 = -(d2 * ddtheta2 + phi1) / d1
    return jnp.array([dtheta1, dtheta2, ddtheta1, ddtheta2, 0.0])


def wrap(x: float, m: float, M: float) -> float:
    """For example, m = -180, M = 180 (degrees), x = 360 --> returns 0."""
    diff = M - m
    go_up = x < m     # Wrap if x is outside the left bound
    go_down = x >= M  # Wrap if x is outside OR on the right bound

    how_often = (
        go_up * jnp.ceil((m - x) / diff)           # if m - x is an integer, keep it
        + go_down * jnp.floor((x - M) / diff + 1)  # if x - M is an integer, round up
    )
    x_out = x - how_often * diff * go_down + how_often * diff * go_up
    return x_out


def rk4(y0: chex.Array, params: AcrobotParams):
    """Runge-Kutta integration of ODE - Difference to OpenAI: Only 1 step!"""
    dt2 = params.dt / 2.0
    k1 = dsdt(y0, 0, params)
    k2 = dsdt(y0 + dt2 * k1, dt2, params)
    k3 = dsdt(y0 + dt2 * k2, dt2, params)
    k4 = dsdt(y0 + params.dt * k3, params.dt, params)
    yout = y0 + params.dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
    return yout

@pydataclass
class Args:
    controller: str = "mppi"  # fixed
    debug: bool = False


def main(args: Args):
    # setup environment
    env = Acrobot()

    # setup controller
    # shared MPPI parameters
    sigma = 0.2
    N = 1024 if not args.debug else 2
    H = 32 if not args.debug else 2
    lam = 0.01

    sigma = 0.5
    N = 1024 if not args.debug else 32
    H = 32  # 64 #128
    lam = 0.01

    a_mean = jnp.tile(jnp.zeros(env.action_dim), (H, 1))
    # load a_mean
    # a_mean = jnp.load("../../results/action_seq.npy")[:H]
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
    elif args.controller == "mppi_spline":
        sigmas = jnp.array([sigma] * env.action_dim)
        a_cov_per_step = jnp.diag(sigmas**2)
        N = 32
        h = 10
        n = 10
        H = (h - 1) * n + 1
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
            env=env,
            control_params=control_params,
            N=N,
            h=h,
            lam=1e-2,
            n=n,
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
    elif args.controller == "zero":
        control_params = None
        controller = lambda *args, **kwargs: (jnp.zeros(env.action_dim), None, None)
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
        # create animation with the state sequence
        l1 = env_params.link_length_1
        l2 = env_params.link_length_2

        # plot joint angle 1 and 2
        plt.figure()
        plt.plot([s.joint_angle1 for s in state_seq], label="joint angle 1")
        plt.plot([s.joint_angle2 for s in state_seq], label="joint angle 2")
        plt.legend()
        plt.savefig("../../results/joint_angle.png")
        # plot action
        plt.cla()
        plt.plot([a[0] for a in action_seq], label="action")
        plt.savefig("../../results/action.png")

        print("saving animation...")

        def update_plot(frame_num):
            plt.gca().clear()
            theta1 = state_seq[frame_num].joint_angle1
            theta2 = state_seq[frame_num].joint_angle2
            p1 = jnp.array([-l1 * jnp.sin(theta1), l1 * jnp.cos(theta1)])
            p2 = p1 + jnp.array([-l2 * jnp.sin(theta1+theta2), l2 * jnp.cos(theta1+theta2)])
            plt.plot([0, p1[0], p2[0]], [0, p1[1], p2[1]], "k-o")
            plt.xlim(-(l1 + l2), l1 + l2)
            plt.ylim(-(l1 + l2), l1 + l2)
            plt.gca().set_aspect("equal", adjustable="box")

        plt.figure()
        anim = FuncAnimation(plt.gcf(), update_plot, frames=len(state_seq), interval=20)
        anim.save("../../results/anim.gif", dpi=80, writer="imagemagick", fps=50)
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