import jax
import jax.numpy as jnp
from jax import lax
from gymnax.environments import environment, spaces
from typing import Tuple, Optional
import chex
import matplotlib.pyplot as plt
import numpy as np
from functools import partial
from dataclasses import dataclass as pydataclass
import tyro
import pickle
import time as time_module

from adaptive_control_gym.envs.jax_env.dynamics import geom
from adaptive_control_gym.envs.jax_env.dynamics.utils import get_hit_penalty, EnvParams3D, EnvState3D, Action3D
from adaptive_control_gym.envs.jax_env.dynamics.loose import get_loose_dynamics_3d


class Quad3D(environment.Environment):
    """
    JAX Compatible version of Quad3D-v0 OpenAI gym environment. Source:
    github.com/openai/gym/blob/master/gym/envs/classic_control/Quad3D.py
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
        self.taut_dynamics = None
        self.loose_dynamics = get_loose_dynamics_3d()
        self.dynamic_transfer = None
        # controllers

    @property
    def default_params(self) -> EnvParams3D:
        """Default environment parameters for Quad3D-v0."""
        return EnvParams3D()

    def step_env(
        self,
        key: chex.PRNGKey,
        state: EnvState3D,
        action: float,
        params: EnvParams3D,
    ) -> Tuple[chex.Array, EnvState3D, float, bool, dict]:
        thrust = (action[0] + 1.0) / 2.0 * params.max_thrust
        torque = action[1:4] * params.max_torque
        err_pos = jnp.linalg.norm(state.pos_tar - state.pos)
        err_vel = jnp.linalg.norm(state.vel_tar - state.vel)
        if self.task == "jumping":
            raise NotImplementedError
            drone_panelty = get_hit_penalty(state.y, state.z) * 3.0
            obj_panelty = get_hit_penalty(state.y_obj, state.z_obj) * 3.0
            reward = 1.0 - 0.6 * err_pos - 0.15 * err_vel \
                + (drone_panelty + obj_panelty)
        elif self.task == 'hovering':
            reward = 1.0 - 0.6 * err_pos - 0.1 * err_vel
        else:
            raise NotImplementedError
            reward = 1.0 - 0.8 * err_pos - 0.05 * err_vel
        reward = reward.squeeze()
        env_action = Action3D(thrust=thrust, torque=torque)

        # old_loose_state = state.l_rope < (
        #     params.l - params.rope_taut_therehold)
        # taut_state = self.taut_dynamics(params, state, env_action)
        # loose_state = self.loose_dynamics(params, state, env_action)
        # new_state = self.dynamic_transfer(
        #     params, loose_state, taut_state, old_loose_state)

        new_state = self.loose_dynamics(params, state, env_action)

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
        self, key: chex.PRNGKey, params: EnvParams3D
    ) -> Tuple[chex.Array, EnvState3D]:
        """Reset environment state by sampling theta, theta_dot."""
        traj_key, pos_key, key = jax.random.split(key, 3)
        # generate reference trajectory by adding a few sinusoids together
        pos_traj, vel_traj = self.generate_traj(traj_key)

        if self.task == 'jumping':
            pos = jnp.array([-1.0, 0.0, 0.0])
        else:
            pos = jax.random.uniform(
                pos_key, shape=(3,), minval=-1.0, maxval=1.0)
        pos_hook = pos + params.hook_offset
        pos_obj = pos + jnp.array([0.0, 0.0, -params.l])
        zeros3 = jnp.zeros(3)
        state = EnvState3D(
            # drone
            pos=pos, vel=zeros3, omega=zeros3, quat=jnp.concatenate(
                [zeros3, jnp.array([1.0])]),
            # object
            pos_obj=pos_obj, vel_obj=zeros3,
            # hook
            pos_hook=pos_hook, vel_hook=zeros3,
            # rope
            l_rope=params.l, zeta=jnp.array([0.0, 0.0, -1.0]), zeta_dot=zeros3,
            f_rope=zeros3, f_rope_norm=0.0,
            # trajectory
            pos_tar=pos_traj[0], vel_tar=vel_traj[0],
            pos_traj=pos_traj, vel_traj=vel_traj,
            # debug value
            last_thrust=0.0, last_torque=zeros3,
            # step
            time=0,
        )
        return self.get_obs(state, params), state

    @partial(jax.jit, static_argnums=(0,))
    def sample_params(self, key: chex.PRNGKey) -> EnvParams3D:
        """Sample environment parameters."""

        param_key = jax.random.split(key)[0]
        rand_val = jax.random.uniform(
            param_key, shape=(9,), minval=0.0, maxval=1.0)

        m = 0.025 + 0.015 * rand_val[0]
        I = jnp.array([1.2e-5, 1.2e-5, 2.0e-5]) + 0.5e-5 * rand_val[1:4]
        mo = 0.005 + 0.005 * rand_val[4]
        l = 0.2 + 0.2 * rand_val[5]
        hook_offset = rand_val[6:9] * 0.04

        return EnvParams3D(m=m, I=I, mo=mo, l=l, hook_offset=hook_offset)

    @partial(jax.jit, static_argnums=(0,))
    def generate_fixed_traj(self, key: chex.PRNGKey) -> Tuple[chex.Array, chex.Array, chex.Array, chex.Array]:
        zeros = jnp.zeros((self.default_params.max_steps_in_episode, 3))
        key_pos = jax.random.split(key)[0]
        pos = jax.random.uniform(
            key_pos, shape=(3,), minval=-1.0, maxval=1.0)
        pos_traj = zeros + pos
        vel_traj = zeros
        return pos_traj, vel_traj

    @partial(jax.jit, static_argnums=(0,))
    def generate_lissa_traj(self, key: chex.PRNGKey) -> chex.Array:
        # get random amplitude and phase
        key_amp, key_phase = jax.random.split(key, 2)
        rand_amp = jax.random.uniform(
            key_amp, shape=(3, 2), minval=-1.0, maxval=1.0)
        rand_phase = jax.random.uniform(
            key_phase, shape=(3, 2), minval=-jnp.pi, maxval=jnp.pi
        )
        # get trajectory
        scale = 0.8
        ts = jnp.arange(
            0, self.default_params.max_steps_in_episode + 50, self.default_params.dt
        )  # NOTE: do not use params for jax limitation
        w1 = 2 * jnp.pi * 0.25
        w2 = 2 * jnp.pi * 0.5

        pos_traj = scale * jnp.stack([
            rand_amp[i, 0] * jnp.sin(w1 * ts + rand_phase[i, 0]) +
            rand_amp[i, 1] * jnp.sin(w2 * ts + rand_phase[i, 1])
            for i in range(3)
        ])

        vel_traj = scale * jnp.stack([
            rand_amp[i, 0] * w1 * jnp.cos(w1 * ts + rand_phase[i, 0]) +
            rand_amp[i, 1] * w2 * jnp.cos(w2 * ts + rand_phase[i, 1])
            for i in range(3)
        ])

        return pos_traj, vel_traj

    @partial(jax.jit, static_argnums=(0,))
    def generate_zigzag_traj(self, key: chex.PRNGKey) -> chex.Array:
        point_per_seg = 40
        num_seg = self.default_params.max_steps_in_episode // point_per_seg + 1

        key_keypoints = jax.random.split(key, num_seg)
        key_angles = jax.random.split(key, num_seg)

        # sample from 3d -1.5 to 1.5
        prev_point = jax.random.uniform(
            key_keypoints[0], shape=(3,), minval=-1.5, maxval=1.5)

        def update_fn(carry, i):
            key_keypoint, key_angle, prev_point = carry

            # Calculate the unit vector pointing to the center
            vec_to_center = - prev_point / jnp.linalg.norm(prev_point)

            # Sample random rotation angles for theta and phi from [-pi/3, pi/3]
            delta_theta, delta_phi = jax.random.uniform(
                key_angle, shape=(2,), minval=-jnp.pi/3, maxval=jnp.pi/3)

            # Calculate new direction
            theta = jnp.arccos(vec_to_center[2]) + delta_theta
            phi = jnp.arctan2(vec_to_center[1], vec_to_center[0]) + delta_phi
            new_direction = jnp.array(
                [jnp.sin(theta) * jnp.cos(phi), jnp.sin(theta) * jnp.sin(phi), jnp.cos(theta)])

            # Sample the distance from [1.5, 2.5]
            distance = jax.random.uniform(key_keypoint, minval=1.5, maxval=2.5)

            # Calculate the new point
            next_point = prev_point + distance * new_direction

            point_traj_seg = jnp.stack([jnp.linspace(
                prev, next_p, point_per_seg, endpoint=False) for prev, next_p in zip(prev_point, next_point)], axis=-1)
            point_dot_traj_seg = (next_point - prev_point) / \
                (point_per_seg + 1) * jnp.ones((point_per_seg, 3))

            carry = (key_keypoints[i+1], key_angles[i+1], next_point)
            return carry, (point_traj_seg, point_dot_traj_seg)

        initial_carry = (key_keypoints[1], key_angles[1], prev_point)
        _, (point_traj_segs, point_dot_traj_segs) = lax.scan(
            update_fn, initial_carry, jnp.arange(1, num_seg))

        return jnp.concatenate(point_traj_segs, axis=-1), jnp.concatenate(point_dot_traj_segs, axis=-1)

    def get_obs(self, state: EnvState3D, params: EnvParams3D) -> chex.Array:
        """Return angle in polar coordinates and change."""
        obs_elements = [
            # drone
            state.pos, state.vel/4.0, state.quat, state.omega/40.0,  # 3*3+4=13
            # object
            state.pos_obj, state.vel_obj/4.0,  # 3*2=6
            # hook
            state.pos_hook, state.vel_hook/4.0,  # 3*2=6
            # rope
            jnp.expand_dims(state.l_rope, axis=0), state.zeta, state.zeta_dot,  # 3*3=9
            state.f_rope, jnp.expand_dims(state.f_rope_norm, axis=0),  # 3+1=4
            # trajectory
            state.pos_tar, state.vel_tar/4.0  # 3*2=6
        ]  # 13+6+6+9+4+6=44
        # future trajectory observation
        # NOTE: use statis default value rather than params for jax limitation
        traj_obs_len, traj_obs_gap = self.default_params.traj_obs_len, self.default_params.traj_obs_gap
        for i in range(traj_obs_len):  # 6*traj_obs_len
            idx = state.time + 1 + i * traj_obs_gap
            obs_elements.append(state.pos_traj[idx]),
            obs_elements.append(state.vel_traj[idx]/4.0)

        # parameter observation
        param_elements = [
            jnp.array([
                (params.m-0.025)/(0.04-0.025) * 2.0 - 1.0,
                (params.mo-0.005)/0.05 * 2.0 - 1.0,
                (params.l-0.2)/(0.4-0.2) * 2.0 - 1.0]),  # 3
            (params.I-1.2e-5)/0.5e-5 * 2.0 - 1.0,  # 3
            (params.hook_offset-0.0)/0.04 * 2.0 - 1.0,  # 3
        ]  # 4+3=7

        return jnp.concatenate(obs_elements+param_elements).squeeze()

    def is_terminal(self, state: EnvState3D, params: EnvParams3D) -> bool:
        """Check whether state is terminal."""
        # Check number of steps in episode termination condition
        done = (
            (state.time >= params.max_steps_in_episode)
            | (jnp.abs(state.pos) > 2.0).any()
            | (jnp.abs(state.pos_obj) > 2.0).any()
            | (jnp.abs(state.omega) > 100.0).any()
        )
        return done

    @property
    def name(self) -> str:
        """Environment name."""
        return "Quad3D-v1"

    @property
    def num_actions(self) -> int:
        """Number of actions possible in environment."""
        return 2

    def action_space(self, params: Optional[EnvParams3D] = None) -> spaces.Box:
        """Action3D space of the environment."""
        if params is None:
            params = self.default_params
        return spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(4,),
            dtype=jnp.float32,
        )

    def observation_space(self, params: EnvParams3D) -> spaces.Box:
        """Observation space of the environment."""
        # NOTE: use default params for jax limitation
        return spaces.Box(-1.0, 1.0, shape=(44+self.default_params.traj_obs_len*6+9,), dtype=jnp.float32)


def test_env(env: Quad3D, policy):
    # running environment
    t0 = time_module.time()
    rng = jax.random.PRNGKey(1)
    rng, rng_params = jax.random.split(rng)
    env_params = env.sample_params(rng_params)

    state_seq, obs_seq, reward_seq = [], [], []
    rng, rng_reset = jax.random.split(rng)
    obs, env_state = env.reset(rng_reset, env_params)
    n_dones = 0
    while True:
        state_seq.append(env_state)
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
    print(f'env running time: {time_module.time()-t0:.2f}s')

    t0 = time_module.time()
    # plot results
    num_figs = len(state_seq[0].__dict__) + 20
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
        if name in ["pos_traj", "vel_traj"]:
            continue
        elif (('pos' in name) or ('vel' in name)) and ('tar' not in name):
            xyz = np.array([getattr(s, name) for s in state_seq])
            xyz_tar = np.array([getattr(s, name[:3] + "_tar")
                               for s in state_seq])
            for i, subplot_name in zip(range(3), ['x', 'y', 'z']):
                current_fig += 1
                plt.subplot(num_rows, 6, current_fig)
                plt.plot(time, xyz[:, i], label=f"{subplot_name}")
                plt.plot(time, xyz_tar[:, i], '--', label=f"{subplot_name}_tar")
                plt.ylabel(name+"_"+subplot_name)
                plt.legend()
        else:
            current_fig += 1
            plt.subplot(num_rows, 6, current_fig)
            plt.plot(time, [getattr(s, name) for s in state_seq])
            plt.ylabel(name)

    plt.xlabel("time")
    plt.savefig("../results/plot.png")
    print(f'plotting time: {time_module.time()-t0:.2f}s')

    # save state_seq (which is a list of EnvState3D:flax.struct.dataclass)
    with open("../results/state_seq.pkl", "wb") as f:
        pickle.dump(state_seq, f)


@pydataclass
class Args:
    task: str = "hovering"

def main(args: Args):
    env = Quad3D(task=args.task)

    def pid_policy(obs:jnp.ndarray, env_state:EnvState3D, env_params:EnvParams3D, rng:jax.random.PRNGKey):
        # get object target force
        w0 = 8.0
        zeta = 0.95
        kp = env_params.mo * (w0**2)
        kd = env_params.mo * 2.0 * zeta * w0
        target_force_obj = kp * (env_state.pos_tar - env_state.pos_obj) + \
            kd * (env_state.vel_tar - env_state.vel_obj) + \
            env_params.mo * jnp.array([0.0, 0.0, env_params.g]) 
        target_force_obj_norm = jnp.linalg.norm(target_force_obj)
        zeta_target = - target_force_obj / target_force_obj_norm
        pos_tar_quad = env_state.pos_obj - env_params.l * zeta_target
        vel_tar_quad = env_state.vel_tar

        # DEBUG
        pos_tar_quad = env_state.pos_tar
        vel_tar_quad = env_state.vel_tar
        target_force_obj *= 0.0
        target_force_obj_norm = 0.0

        # get drone target force
        w0 = 10.0
        zeta = 0.95
        kp = env_params.m * (w0**2)
        kd = env_params.m * 2.0 * zeta * w0
        target_force = kp * (pos_tar_quad - env_state.pos) + \
            kd * (vel_tar_quad - env_state.vel) + \
            env_params.m * jnp.array([0.0, 0.0, env_params.g]) + \
            target_force_obj
        thrust = jnp.linalg.norm(target_force)
        target_unitvec = target_force / thrust
        # target_unitvec = jnp.array([jnp.sin(jnp.pi/6), 0.0, jnp.cos(jnp.pi/6)]) # DEBUG
        target_unitvec_local = geom.rotate_with_quat(
            target_unitvec, geom.conjugate_quat(env_state.quat))

        w0 = 10.0
        zeta = 0.95
        kp = env_params.I[0] * (w0**2)
        kd = env_params.I[0] * 2.0 * zeta * w0
        current_unitvec_local = jnp.array([0.0, 0.0, 1.0])
        rot_axis = jnp.cross(current_unitvec_local, target_unitvec_local)
        rot_angle = jnp.arccos(jnp.dot(current_unitvec_local, target_unitvec_local))
        omega_local = geom.rotate_with_quat(
            env_state.omega, geom.conjugate_quat(env_state.quat))
        torque = kp * rot_angle * rot_axis / jnp.linalg.norm(rot_axis) + \
            kd * (-omega_local)
        
        # convert into action space
        thrust_normed = jnp.clip(
            thrust / env.default_params.max_thrust * 2.0 - 1.0, -1.0, 1.0
        )
        tau_normed = jnp.clip(torque / env.default_params.max_torque, -1.0, 1.0)
        return jnp.array([thrust_normed, tau_normed[0], tau_normed[1], 0.0])

    def random_policy(obs, state, params, rng): return env.action_space(
        env.default_params).sample(rng)
    
    def fixed_policy(obs, state, params, rng): 
        return jnp.array([params.g*params.m/params.max_thrust * 2.0 - 1.0, 0.0, 0.0, 0.0])

    print('starting test...')
    with jax.disable_jit():
        test_env(env, policy=pid_policy)


if __name__ == "__main__":
    main(tyro.cli(Args))
