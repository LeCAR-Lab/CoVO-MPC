import jax
import jax.numpy as jnp
from jax import lax
from gymnax.environments import environment, spaces
from typing import Tuple, Optional
import chex
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
from functools import partial
from dataclasses import dataclass as pydataclass
import tyro

from quadjax.dynamics import geom, EnvParamsDual2D, EnvStateDual2D, ActionDual2D, get_hit_penalty, get_dual_taut_dynamics_2d

class DualQuad2D(environment.Environment):
    """
    JAX Compatible version of DualQuad2D-v0 OpenAI gym environment.
    """

    def __init__(self, task: str = "tracking_zigzag"):
        super().__init__()
        self.task = task
        # reference trajectory function
        if task == "tracking":
            self.generate_traj = self.generate_lissa_traj
        elif task == "tracking_zigzag":
            self.generate_traj = self.generate_zigzag_traj
        elif task in ["jumping", "hovering"]:
            self.generate_traj = self.generate_fixed_traj
        else:
            raise NotImplementedError
        # dynamics
        self.taut_dynamics = get_dual_taut_dynamics_2d()
        self.loose_dynamics = None
        self.dynamic_transfer = None

    @property
    def default_params(self) -> EnvParamsDual2D:
        """Default environment parameters for DualQuad2D-v0."""
        return EnvParamsDual2D()
    
    def step_env(
        self,
        key: chex.PRNGKey,
        state: EnvStateDual2D,
        action: float,
        params: EnvParamsDual2D,
    ) -> Tuple[chex.Array, EnvStateDual2D, float, bool, dict]:
        thrust0 = (action[0] + 1.0) / 2.0 * params.max_thrust
        tau0 = action[1] * params.max_torque
        thrust1 = (action[2] + 1.0) / 2.0 * params.max_thrust
        tau1 = action[3] * params.max_torque
        err_pos = jnp.sqrt((state.y_tar - state.y_obj) ** 2 + (state.z_tar - state.z_obj) ** 2)
        err_vel = jnp.sqrt((state.y_dot_tar - state.y_obj_dot) ** 2 + (state.z_dot_tar - state.z_obj_dot) ** 2) 
        if self.task == "jumping":
            drone_panelty = get_hit_penalty(state.y0, state.z0) * 3.0
            drone_panelty = get_hit_penalty(state.y1, state.z1) * 3.0
            obj_panelty = get_hit_penalty(state.y_obj, state.z_obj) * 3.0
            reward = 1.0 - 0.6 * err_pos - 0.15 * err_vel \
                + (drone_panelty + obj_panelty)
        elif self.task == 'hovering':
            reward = 1.0 - 0.6 * err_pos - 0.1 * err_vel
        else:
            reward = 1.0 - 0.8 * err_pos - 0.05 * err_vel
        reward = reward.squeeze()
        env_action = ActionDual2D(thrust0=thrust0, tau0=tau0, thrust1=thrust1, tau1=tau1)

        new_state = self.taut_dynamics(params, state, env_action)

        done = self.is_terminal(state, params)
        return (
            lax.stop_gradient(self.get_obs(new_state, params)),
            lax.stop_gradient(new_state),
            reward,
            done,
            {
                "discount": self.discount(new_state, params), 
                "err_pos": err_pos,
                "err_vel": err_vel,
            },
        )

    def reset_env(
        self, key: chex.PRNGKey, params: EnvParamsDual2D
    ) -> Tuple[chex.Array, EnvStateDual2D]:
        """Reset environment state by sampling theta, theta_dot."""
        # generate reference trajectory by adding a few sinusoids together
        y_traj, z_traj, y_dot_traj, z_dot_traj = self.generate_traj(key)

        state = EnvStateDual2D(
            y0 = y_traj[0]-params.delta_yh0, z0 = z_traj[0]-params.delta_zh0 + params.l0, theta0 = 0.0, phi0 = 0.0, 
            y0_dot=0.0, z0_dot=0.0, theta0_dot=0.0, phi0_dot=0.0,
            last_thrust0=0.0, last_tau0=0.0,
            y_hook0=y_traj[0], z_hook0=z_traj[0] + params.l0, y_hook0_dot=0.0, z_hook0_dot=0.0,
            f_rope0=0.0, f_rope0_y=0.0, f_rope0_z=0.0, l_rope0=params.l0,

            y1 = y_traj[0]-params.delta_yh1, z1 = z_traj[0]-params.delta_zh1 + params.l1, theta1 = 0.0, phi1 = 0.0,
            y1_dot=0.0, z1_dot=0.0, theta1_dot=0.0, phi1_dot=0.0,
            last_thrust1=0.0, last_tau1=0.0,
            y_hook1=y_traj[0], z_hook1=z_traj[0]+params.l1, y_hook1_dot=0.0, z_hook1_dot=0.0,
            f_rope1=0.0, f_rope1_y=0.0, f_rope1_z=0.0, l_rope1=params.l1,

            y_obj=y_traj[0], z_obj=z_traj[0], y_obj_dot=0.0, z_obj_dot=0.0,

            time = 0,
            y_traj = y_traj, z_traj = z_traj, y_dot_traj = y_dot_traj, z_dot_traj = z_dot_traj,
            y_tar = y_traj[0], z_tar = z_traj[0], y_dot_tar = y_dot_traj[0], z_dot_tar = z_dot_traj[0],
        )
        return self.get_obs(state, params), state
    
    @partial(jax.jit, static_argnums=(0,))
    def sample_params(self, key: chex.PRNGKey) -> EnvParamsDual2D:
        """Sample environment parameters."""
        
        params_key, key = jax.random.split(key)

        m_min, m_max = 0.025, 0.04
        I_min, I_max = 2.5e-5, 3.5e-5
        mo_min, mo_max = 0.003, 0.01
        l_min, l_max = 0.2, 0.4
        delta_yh_min, delta_yh_max = -0.04, 0.04
        delta_zh_min, delta_zh_max = -0.06, 0.00

        params_min = jnp.array([m_min, I_min, l_min, delta_yh_min, delta_zh_min]*2 + [mo_min])
        params_max = jnp.array([m_max, I_max, l_max, delta_yh_max, delta_zh_max]*2 + [mo_max])
        params = jax.random.uniform(params_key, shape=(11,), minval=params_min, maxval=params_max)
        
        return EnvParamsDual2D(
            m0=params[0], I0=params[1], l0=params[2], delta_yh0=params[3], delta_zh0=params[4],
            m1=params[5], I1=params[6], l1=params[7], delta_yh1=params[8], delta_zh1=params[9],
            mo=params[10],
        )
    
    @partial(jax.jit, static_argnums=(0,))
    def generate_fixed_traj(self, key: chex.PRNGKey) -> Tuple[chex.Array, chex.Array, chex.Array, chex.Array]:
        ts = jnp.arange(
            0, self.default_params.max_steps_in_episode + 50, self.default_params.dt
        )
        key_y, key_z, key_sign = jax.random.split(key, 3)
        sign = jax.random.choice(key_sign, jnp.array([-1.0, 1.0]))
        y = jax.random.uniform(key_y, shape=(), minval=0.12, maxval=1.0)
        z = jax.random.uniform(key_z, shape=(), minval=-1.0, maxval=1.0)
        y_traj = jnp.zeros_like(ts) + sign * y
        z_traj = jnp.zeros_like(ts) + z
        y_dot_traj = jnp.zeros_like(ts)
        z_dot_traj = jnp.zeros_like(ts)
        return y_traj, z_traj, y_dot_traj, z_dot_traj

    @partial(jax.jit, static_argnums=(0,))
    def generate_lissa_traj(self, key: chex.PRNGKey) -> chex.Array:
        # get random attitude and phase
        key_amp_y, key_phase_y, key_amp_z, key_phase_z = jax.random.split(key, 4)
        rand_amp_y = jax.random.uniform(key_amp_y, shape=(2,), minval=-1.0, maxval=1.0)
        rand_amp_z = jax.random.uniform(key_amp_z, shape=(2,), minval=-1.0, maxval=1.0)
        rand_phase_y = jax.random.uniform(
            key_phase_y, shape=(2,), minval=-jnp.pi, maxval=jnp.pi
        )
        rand_phase_z = jax.random.uniform(
            key_phase_z, shape=(2,), minval=-jnp.pi, maxval=jnp.pi
        )
        # get trajectory
        scale = 0.8
        ts = jnp.arange(
            0, self.default_params.max_steps_in_episode + 50, self.default_params.dt
        )  # NOTE: do not use params for jax limitation
        w1 = 2 * jnp.pi * 0.25
        w2 = 2 * jnp.pi * 0.5
        y_traj = scale * rand_amp_y[0] * jnp.sin(
            w1 * ts + rand_phase_y[0]
        ) + scale * rand_amp_y[1] * jnp.sin(w2 * ts + rand_phase_y[1])
        z_traj = scale * rand_amp_z[0] * jnp.sin(
            w1 * ts + rand_phase_z[0]
        ) + scale * rand_amp_z[1] * jnp.sin(w2 * ts + rand_phase_z[1])
        y_dot_traj = scale * rand_amp_y[0] * w1 * jnp.cos(
            w1 * ts + rand_phase_y[0]
        ) + scale * rand_amp_y[1] * w2 * jnp.cos(w2 * ts + rand_phase_y[1])
        z_dot_traj = scale * rand_amp_z[0] * w1 * jnp.cos(
            w1 * ts + rand_phase_z[0]
        ) + scale * rand_amp_z[1] * w2 * jnp.cos(w2 * ts + rand_phase_z[1])
        return y_traj, z_traj, y_dot_traj, z_dot_traj
    
    @partial(jax.jit, static_argnums=(0,))
    def generate_zigzag_traj(self, key: chex.PRNGKey) -> chex.Array:
        point_per_seg = 40
        num_seg = self.default_params.max_steps_in_episode // point_per_seg + 1

        key_keypoints = jax.random.split(key, num_seg)
        key_angles = jax.random.split(key, num_seg)

        # sample from 2d -1.5 to 1.5
        prev_y, prev_z = jax.random.uniform(key_keypoints[0], shape=(2,), minval=-1.5, maxval=1.5)

        def update_fn(carry, i):
            key_keypoint, key_angle, prev_y, prev_z = carry

            # Calculate the previous point angle to the center
            prev_angle = jnp.arctan2(prev_z, prev_y) + jnp.pi

            # Sample a random displacement angle from [-pi/3, pi/3]
            delta_angle = jax.random.uniform(key_angle, minval=-jnp.pi/3, maxval=jnp.pi/3)

            # Calculate the new angle
            angle = prev_angle + delta_angle

            # Sample the distance from [1.5, 2.5]
            distance = jax.random.uniform(key_keypoint, minval=1.5, maxval=2.5)

            # Calculate the new point
            next_y = prev_y + distance * jnp.cos(angle)
            next_z = prev_z + distance * jnp.sin(angle)

            y_traj_seg = jnp.linspace(prev_y, next_y, point_per_seg, endpoint=False)
            z_traj_seg = jnp.linspace(prev_z, next_z, point_per_seg, endpoint=False)
            y_dot_traj_seg = (next_y - prev_y) / (point_per_seg + 1) / self.default_params.dt * jnp.ones(point_per_seg)
            z_dot_traj_seg = (next_z - prev_z) / (point_per_seg + 1) / self.default_params.dt * jnp.ones(point_per_seg)

            carry = (key_keypoints[i+1], key_angles[i+1], next_y, next_z)
            return carry, (y_traj_seg, z_traj_seg, y_dot_traj_seg, z_dot_traj_seg)

        initial_carry = (key_keypoints[1], key_angles[1], prev_y, prev_z)
        _, (y_traj_segs, z_traj_segs, y_dot_traj_segs, z_dot_traj_segs) = lax.scan(update_fn, initial_carry, jnp.arange(1, num_seg))

        y_traj = jnp.concatenate(y_traj_segs)
        z_traj = jnp.concatenate(z_traj_segs)
        y_dot_traj = jnp.concatenate(y_dot_traj_segs)
        z_dot_traj = jnp.concatenate(z_dot_traj_segs)

        return y_traj, z_traj, y_dot_traj, z_dot_traj


    def get_obs(self, state: EnvStateDual2D, params: EnvParamsDual2D) -> chex.Array:
        """Return angle in polar coordinates and change."""
        traj_obs_len, traj_obs_gap = self.default_params.traj_obs_len, self.default_params.traj_obs_gap
        indices = state.time + 1 + jnp.arange(traj_obs_len) * traj_obs_gap
        
        obs_elements = [
            # state
            state.y0, state.z0, state.y0_dot / 4.0, state.z0_dot / 4.0, state.theta0, state.theta0_dot / 40.0, state.phi0, state.phi0_dot / 10.0, 
            state.y1, state.z1, state.y1_dot / 4.0, state.z1_dot / 4.0, state.theta1, state.theta1_dot / 40.0, state.phi1, state.phi1_dot / 10.0,
            state.y_obj, state.z_obj, state.y_obj_dot / 4.0, state.z_obj_dot / 4.0,
            # reference
            state.y_tar, state.z_tar,
            state.y_dot_tar / 4.0, state.z_dot_tar / 4.0,
            state.y_tar - state.y_obj, state.z_tar - state.z_obj,
            (state.y_dot_tar - state.y_obj_dot) / 4.0,
            (state.z_dot_tar - state.z_obj_dot) / 4.0,
            *state.y_traj[indices].flatten(), 
            *state.z_traj[indices].flatten(), 
            *(state.y_dot_traj[indices].flatten() / 4.0),
            *(state.z_dot_traj[indices].flatten() / 4.0),
            # params
            (params.m0-0.025)/(0.04-0.025) * 2.0 - 1.0,
            (params.I0-2.5e-5)/(3.5e-5 - 2.5e-5) * 2.0 - 1.0,
            (params.l0-0.2)/(0.4-0.2) * 2.0 - 1.0,
            (params.delta_yh0-(-0.04))/(0.04-(-0.04)) * 2.0 - 1.0,
            (params.delta_zh0-(-0.06))/(0.0-(-0.06)) * 2.0 - 1.0,
            (params.m1-0.025)/(0.04-0.025) * 2.0 - 1.0,
            (params.I1-2.5e-5)/(3.5e-5 - 2.5e-5) * 2.0 - 1.0,
            (params.l1-0.2)/(0.4-0.2) * 2.0 - 1.0,
            (params.delta_yh1-(-0.04))/(0.04-(-0.04)) * 2.0 - 1.0,
            (params.delta_zh1-(-0.06))/(0.0-(-0.06)) * 2.0 - 1.0,
            (params.mo-0.003)/(0.01-0.003) * 2.0 - 1.0,
        ]

        return jnp.array(obs_elements).squeeze()

    def is_terminal(self, state: EnvStateDual2D, params: EnvParamsDual2D) -> bool:
        """Check whether state is terminal."""
        # Check number of steps in episode termination condition
        done = (
            (state.time >= params.max_steps_in_episode)
            | (jnp.abs(state.y0) > 2.0)
            | (jnp.abs(state.z0) > 2.0)
            | (jnp.abs(state.y1) > 2.0)
            | (jnp.abs(state.z1) > 2.0)
            | (jnp.abs(state.theta0_dot) > 100.0)
            | (jnp.abs(state.phi0_dot) > 100.0)
            | (jnp.abs(state.theta1_dot) > 100.0)
            | (jnp.abs(state.phi1_dot) > 100.0)
        )
        return done

def test_env(env: DualQuad2D, policy, render_video=False):
    rng = jax.random.PRNGKey(1)
    rng, rng_params = jax.random.split(rng)
    env_params = env.sample_params(rng_params)

    state_seq, param_seq, obs_seq, reward_seq = [], [], [], []
    rng, rng_reset = jax.random.split(rng)
    obs, env_state = env.reset(rng_reset, env_params)
    n_dones = 0
    while True:
        state_seq.append(env_state)
        param_seq.append(env_params)
        rng, rng_act, rng_step = jax.random.split(rng, 3)
        action = policy(obs, env_state, env_params, rng_act)
        next_obs, next_env_state, reward, done, info = env.step(
            rng_step, env_state, action, env_params
        )
        if done:
            rng, rng_params = jax.random.split(rng)
            env_params = env.sample_params(rng_params)
        reward_seq.append(reward)
        obs_seq.append(obs)
        if done:
            n_dones += 1
        obs = next_obs
        env_state = next_env_state
        if n_dones >= 1:
            break
    
    # plot trajectory
    def update_plot(frame_num):
        plt.gca().clear()
        frame_list = np.arange(np.max([0, frame_num - 200]), frame_num + 1)
        plt.plot([s.y_obj for s in state_seq[frame_list[0]:frame_list[-1]]], [s.z_obj for s in state_seq[frame_list[0]:frame_list[-1]]], alpha=0.5)
        plt.plot([s.y_tar for s in state_seq], [s.z_tar for s in state_seq], "--", alpha=0.3)

        if env.task == 'jumping':
            hy, hz = 0.05, 0.3
            square1 = [[hy, hz], [hy, 2.0], [-hy, 2.0], [-hy, hz], [hy, hz]]
            square2 = [[hy, -hz], [hy, -2.0], [-hy, -2.0], [-hy, -hz], [hy, -hz]]
            for square in [square1, square2]:
                x, y = zip(*square)
                plt.plot(x, y, linestyle='-')
        
        start = max(0, frame_num)
        for i in range(start, frame_num + 1):
            num_steps = max(frame_num - start, 1)
            alpha = 1 if i == frame_num else ((i-start) / num_steps * 0.1)
            # quadrotor 0 with blue arrow
            plt.arrow(
                state_seq[i].y0,
                state_seq[i].z0,
                -0.1 * jnp.sin(state_seq[i].theta0),
                0.1 * jnp.cos(state_seq[i].theta0),
                width=0.01,
                color="b",
                alpha=alpha,
            )
            # quadrotor 1 with green arrow
            plt.arrow(
                state_seq[i].y1,
                state_seq[i].z1,
                -0.1 * jnp.sin(state_seq[i].theta1),
                0.1 * jnp.cos(state_seq[i].theta1),
                width=0.01,
                color="g",
                alpha=alpha,
            )
            # plot object as point
            plt.plot(state_seq[i].y_obj, state_seq[i].z_obj, "o", color="b", alpha=alpha)
            # plot hook as cross
            plt.plot(state_seq[i].y_hook0, state_seq[i].z_hook0, "x", color="b", alpha=alpha)
            plt.plot(state_seq[i].y_hook1, state_seq[i].z_hook1, "x", color="g", alpha=alpha)
            # plot rope as line (gree if slack, red if taut)
            plt.arrow(
                state_seq[i].y_hook0,
                state_seq[i].z_hook0,
                state_seq[i].y_obj - state_seq[i].y_hook0,
                state_seq[i].z_obj - state_seq[i].z_hook0,
                width=0.01,
                color="r" if state_seq[i].l_rope0 > (param_seq[i].l0 - env_params.rope_taut_therehold) else "g",
                alpha=alpha,
            )
            plt.arrow(
                state_seq[i].y_hook1,
                state_seq[i].z_hook1,
                state_seq[i].y_obj - state_seq[i].y_hook1,
                state_seq[i].z_obj - state_seq[i].z_hook1,
                width=0.01,
                color="r" if state_seq[i].l_rope1 > (param_seq[i].l1 - env_params.rope_taut_therehold) else "g",
                alpha=alpha,
            )
            # plot y_tar and z_tar with red dot
            plt.plot(state_seq[i].y_tar, state_seq[i].z_tar, "ro", alpha=alpha)
        plt.xlabel("y")
        plt.ylabel("z")
        plt.xlim([-2, 2])
        plt.ylim([-2, 2])

    if render_video:
        plt.figure(figsize=(4, 4))
        anim = FuncAnimation(plt.gcf(), update_plot, frames=len(state_seq), interval=20)
        anim.save(filename="../../results/anim.gif", writer="imagemagick", fps=int(1.0/env_params.dt))

    num_figs = len(state_seq[0].__dict__) + 2
    time = [s.time * env_params.dt for s in state_seq]

    # calculate number of rows needed
    num_rows = int(jnp.ceil(num_figs / 6))

    # create num_figs subplots
    plt.subplots(num_rows, 6, figsize=(4 * 6, 2 * num_rows))

    # plot reward
    plt.subplot(num_rows, 6, 1)
    plt.plot(time, reward_seq)
    plt.ylabel("reward")

    # plot obs
    # plot 10 obs in a subplot
    current_fig = 2
    for i in range(len(obs_seq[0])//10+1):
        plt.subplot(num_rows, 6, current_fig)
        current_fig += 1
        for j in range(10):
            idx = i*10+j
            plt.plot(time, [o[idx] for o in obs_seq], label=f"{idx}")
        plt.ylabel("obs")
        plt.legend(fontsize=6, ncol=2)

    # plot state
    for i, (name, value) in enumerate(state_seq[0].__dict__.items()):
        if name in ["y_traj", "z_traj", "y_dot_traj", "z_dot_traj", "theta_traj"]:
            continue
        current_fig += 1
        plt.subplot(num_rows, 6, current_fig)
        plt.plot(time, [getattr(s, name) for s in state_seq])
        if name in ["y_obj", "z_obj", "y_obj_dot", "z_obj_dot"]:
            plt.plot(time, [s.__dict__[name[0] + "_tar"] for s in state_seq], "--")
            plt.legend(["actual", "target"], fontsize=3)
        plt.ylabel(name)

    plt.xlabel("time")
    plt.savefig("../../results/plot.png")

@pydataclass
class Args:
    task: str = "tracking"
    render: bool = False

def main(args: Args):
    env = DualQuad2D(task=args.task)

    def pid_policy(obs, state, params, rng):
        # get drone target force
        # w0 = 8.0
        # zeta = 0.95
        # kp = m * (w0**2)
        # kd = m * 2.0 * zeta * w0
        # target_force_y = kp * (y_drone_tar - y) + kd * (y_dot_tar - y_dot) + target_force_y_obj
        # target_force_z = (
        #     kp * (z_drone_tar - z)
        #     + kd * (z_dot_tar - z_dot)
        #     + m * 9.81
        # ) + target_force_z_obj
        # thrust = -target_force_y * jnp.sin(theta) + target_force_z * jnp.cos(theta)
        # thrust = jnp.sqrt(target_force_y**2 + target_force_z**2)
        # target_theta = -jnp.arctan2(target_force_y, target_force_z)

        w0 = 30.0
        zeta = 0.95
        tau0 = params.I0 * (
            (w0**2) * (0.0 - state.theta0) + 2.0 * zeta * w0 * (0.0 - state.theta0_dot)
        )
        tau1 = params.I1 * (
            (w0**2) * (0.0 - state.theta1) + 2.0 * zeta * w0 * (0.0 - state.theta1_dot)
        )

        # convert into action space
        thrust0_normed = jnp.clip(
            (params.mo * 0.5 + params.m0) * params.g / env.default_params.max_thrust * 2.0 - 1.0, -1.0, 1.0
        )
        tau0_normed = jnp.clip(tau0 / env.default_params.max_torque, -1.0, 1.0)
        thrust1_normed = jnp.clip(
            (params.mo * 0.5 + params.m1) * params.g / env.default_params.max_thrust * 2.0 - 1.0, -1.0, 1.0
        )
        tau1_normed = jnp.clip(tau1 / env.default_params.max_torque, -1.0, 1.0)
        return jnp.array([thrust0_normed, tau0_normed, thrust1_normed, tau1_normed])

    print('starting test...')
    # with jax.disable_jit():
    test_env(env, policy=pid_policy, render_video=args.render)

if __name__ == "__main__":
    main(tyro.cli(Args))