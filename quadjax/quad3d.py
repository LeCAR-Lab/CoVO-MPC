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

from quadjax.dynamics import geom
from quadjax.dynamics.utils import get_hit_penalty
from quadjax.dynamics.dataclass import EnvParams3D, EnvState3D, Action3D
from quadjax.dynamics import make_hybrid_rope_dyn_3d, make_free_dyn_3d

# for debug purpose
from icecream import install

install()


class Quad3D(environment.Environment):
    """
    JAX Compatible version of Quad3D-v0 OpenAI gym environment. Source:
    github.com/openai/gym/blob/master/gym/envs/classic_control/Quad3D.py
    """

    def __init__(self, task: str = "tracking", dynamic_model: str = "hybrid"):
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
        if dynamic_model == "hybrid":
            self.dynamics_fn = make_hybrid_rope_dyn_3d()
        elif dynamic_model == "free":
            self.dynamics_fn = make_free_dyn_3d()
        else:
            raise NotImplementedError
        # controllers

    @property
    def default_params(self) -> EnvParams3D:
        """Default environment parameters for Quad3D-v0."""
        return EnvParams3D()

    def step_env(
        self,
        key: chex.PRNGKey,
        state: EnvState3D,
        action: jnp.ndarray,
        params: EnvParams3D,
    ) -> Tuple[chex.Array, EnvState3D, float, bool, dict]:
        thrust = (action[0] + 1.0) / 2.0 * params.max_thrust
        torque = action[1:4] * params.max_torque
        err_pos = jnp.linalg.norm(state.pos_tar - state.pos_obj)
        err_vel = jnp.linalg.norm(state.vel_tar - state.vel_obj)
        if self.task == "jumping":
            raise NotImplementedError
            drone_panelty = get_hit_penalty(state.y, state.z) * 3.0
            obj_panelty = get_hit_penalty(state.y_obj, state.z_obj) * 3.0
            reward = (
                1.0 - 0.6 * err_pos - 0.15 * err_vel + (drone_panelty + obj_panelty)
            )
        elif self.task == "hovering":
            reward = 1.0 - 0.6 * err_pos - 0.1 * err_vel
        else:
            reward = 1.0 - 0.8 * err_pos - 0.1 * err_vel
        reward = reward.squeeze()
        env_action = Action3D(thrust=thrust, torque=torque)

        new_state = self.dynamics_fn(state, env_action, params)

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

        if self.task == "jumping":
            pos = jnp.array([-1.0, 0.0, 0.0])
        elif ('tracking' in self.task):
            pos = pos_traj[0]
        elif self.task == "hovering":
            pos = pos_traj[0]
        else:
            pos = jax.random.uniform(pos_key, shape=(3,), minval=-1.0, maxval=1.0)
        pos_hook = pos + params.hook_offset
        pos_obj = pos + jnp.array([0.0, 0.0, -params.l])
        zeros3 = jnp.zeros(3)
        state = EnvState3D(
            # drone
            pos=pos,
            vel=zeros3,
            omega=zeros3,
            quat=jnp.concatenate([zeros3, jnp.array([1.0])]),
            # object
            pos_obj=pos_obj,
            vel_obj=zeros3,
            # hook
            pos_hook=pos_hook,
            vel_hook=zeros3,
            # rope
            l_rope=params.l,
            zeta=jnp.array([0.0, 0.0, -1.0]),
            zeta_dot=zeros3,
            f_rope=zeros3,
            f_rope_norm=0.0,
            # trajectory
            pos_tar=pos_traj[0],
            vel_tar=vel_traj[0],
            pos_traj=pos_traj,
            vel_traj=vel_traj,
            # debug value
            last_thrust=0.0,
            last_torque=zeros3,
            # step
            time=0,
        )
        return self.get_obs(state, params), state

    @partial(jax.jit, static_argnums=(0,))
    def sample_params(self, key: chex.PRNGKey) -> EnvParams3D:
        """Sample environment parameters."""

        param_key = jax.random.split(key)[0]
        rand_val = jax.random.uniform(param_key, shape=(9,), minval=0.0, maxval=1.0)

        m = 0.025 + 0.015 * rand_val[0]
        I = jnp.array([1.2e-5, 1.2e-5, 2.0e-5]) + 0.5e-5 * rand_val[1:4]
        I = jnp.diag(I)
        mo = 0.01 + 0.01 * rand_val[4]
        l = 0.2 + 0.2 * rand_val[5]
        hook_offset = rand_val[6:9] * 0.04

        return EnvParams3D(m=m, I=I, mo=mo, l=l, hook_offset=hook_offset)

    @partial(jax.jit, static_argnums=(0,))
    def generate_fixed_traj(
        self, key: chex.PRNGKey
    ) -> Tuple[chex.Array, chex.Array, chex.Array, chex.Array]:
        zeros = jnp.zeros((self.default_params.max_steps_in_episode, 3))
        key_pos = jax.random.split(key)[0]
        pos = jax.random.uniform(key_pos, shape=(3,), minval=-1.0, maxval=1.0)
        pos_traj = zeros + pos
        vel_traj = zeros
        return pos_traj, vel_traj

    @partial(jax.jit, static_argnums=(0,))
    def generate_lissa_traj(self, key: chex.PRNGKey) -> chex.Array:
        # get random amplitude and phase
        key_amp, key_phase = jax.random.split(key, 2)
        rand_amp = jax.random.uniform(key_amp, shape=(3, 2), minval=-1.0, maxval=1.0)
        rand_phase = jax.random.uniform(
            key_phase, shape=(3, 2), minval=-jnp.pi, maxval=jnp.pi
        )
        # get trajectory
        scale = 0.8
        ts = jnp.arange(
            0, self.default_params.max_steps_in_episode + 50, self.default_params.dt
        )  # NOTE: do not use params for jax limitation
        w1 = 2 * jnp.pi * 0.25
        w2 = 2 * jnp.pi * 0.50

        pos_traj = scale * jnp.stack(
            [
                rand_amp[i, 0] * jnp.sin(w1 * ts + rand_phase[i, 0])
                + rand_amp[i, 1] * jnp.sin(w2 * ts + rand_phase[i, 1])
                for i in range(3)
            ], 
            axis=1
        )

        vel_traj = scale * jnp.stack(
            [
                rand_amp[i, 0] * w1 * jnp.cos(w1 * ts + rand_phase[i, 0])
                + rand_amp[i, 1] * w2 * jnp.cos(w2 * ts + rand_phase[i, 1])
                for i in range(3)
            ], 
            axis=1
        )

        return pos_traj, vel_traj

    @partial(jax.jit, static_argnums=(0,))
    def generate_zigzag_traj(self, key: chex.PRNGKey) -> chex.Array:
        point_per_seg = 40
        num_seg = self.default_params.max_steps_in_episode // point_per_seg + 1

        key_keypoints = jax.random.split(key, num_seg)
        key_angles = jax.random.split(key, num_seg)

        # sample from 3d -1.5 to 1.5
        prev_point = jax.random.uniform(
            key_keypoints[0], shape=(3,), minval=-1.5, maxval=1.5
        )

        def update_fn(carry, i):
            key_keypoint, key_angle, prev_point = carry

            # Calculate the unit vector pointing to the center
            vec_to_center = -prev_point / jnp.linalg.norm(prev_point)

            # Sample random rotation angles for theta and phi from [-pi/3, pi/3]
            delta_theta, delta_phi = jax.random.uniform(
                key_angle, shape=(2,), minval=-jnp.pi / 3, maxval=jnp.pi / 3
            )

            # Calculate new direction
            theta = jnp.arccos(vec_to_center[2]) + delta_theta
            phi = jnp.arctan2(vec_to_center[1], vec_to_center[0]) + delta_phi
            new_direction = jnp.array(
                [
                    jnp.sin(theta) * jnp.cos(phi),
                    jnp.sin(theta) * jnp.sin(phi),
                    jnp.cos(theta),
                ]
            )

            # Sample the distance from [1.5, 2.5]
            distance = jax.random.uniform(key_keypoint, minval=1.5, maxval=2.5)

            # Calculate the new point
            next_point = prev_point + distance * new_direction

            point_traj_seg = jnp.stack(
                [
                    jnp.linspace(prev, next_p, point_per_seg, endpoint=False)
                    for prev, next_p in zip(prev_point, next_point)
                ],
                axis=-1,
            )
            point_dot_traj_seg = (
                (next_point - prev_point)
                / (point_per_seg + 1)
                * jnp.ones((point_per_seg, 3))
            )

            carry = (key_keypoints[i + 1], key_angles[i + 1], next_point)
            return carry, (point_traj_seg, point_dot_traj_seg)

        initial_carry = (key_keypoints[1], key_angles[1], prev_point)
        _, (point_traj_segs, point_dot_traj_segs) = lax.scan(
            update_fn, initial_carry, jnp.arange(1, num_seg)
        )

        return jnp.concatenate(point_traj_segs, axis=-1), jnp.concatenate(
            point_dot_traj_segs, axis=-1
        )

    def get_obs(self, state: EnvState3D, params: EnvParams3D) -> chex.Array:
        """Return angle in polar coordinates and change."""
        obs_elements = [
            # drone
            state.pos,
            state.vel / 4.0,
            state.quat,
            state.omega / 40.0,  # 3*3+4=13
            # object
            state.pos_obj,
            state.vel_obj / 4.0,  # 3*2=6
            # hook
            state.pos_hook,
            state.vel_hook / 4.0,  # 3*2=6
            # rope
            jnp.expand_dims(state.l_rope, axis=0),
            state.zeta,
            state.zeta_dot / 5.0,  # 3*3=9
            state.f_rope,
            jnp.expand_dims(state.f_rope_norm, axis=0),  # 3+1=4
            # trajectory
            state.pos_tar,
            state.vel_tar / 4.0,  # 3*2=6
        ]  # 13+6+6+9+4+6=44
        # future trajectory observation
        # NOTE: use statis default value rather than params for jax limitation
        traj_obs_len, traj_obs_gap = (
            self.default_params.traj_obs_len,
            self.default_params.traj_obs_gap,
        )
        for i in range(traj_obs_len):  # 6*traj_obs_len
            idx = state.time + 1 + i * traj_obs_gap
            obs_elements.append(state.pos_traj[idx]),
            obs_elements.append(state.vel_traj[idx] / 4.0)

        # parameter observation
        param_elements = [
            jnp.array(
                [
                    (params.m - 0.025) / (0.04 - 0.025) * 2.0 - 1.0,
                    (params.mo - 0.01) / 0.01 * 2.0 - 1.0,
                    (params.l - 0.2) / (0.4 - 0.2) * 2.0 - 1.0,
                ]
            ),  # 3
            ((params.I - 1.2e-5) / 0.5e-5 * 2.0 - 1.0).flatten(),  # 3x3
            (params.hook_offset - 0.0) / 0.04 * 2.0 - 1.0,  # 3
        ]  # 4+3=7
        # DEBUG print all elements in obs_elements and param_elements
        # for i, o in enumerate(obs_elements + param_elements):
        #     ic(i, o.shape)
        # ic(state.pos_tar.shape)
        obs = jnp.concatenate(obs_elements + param_elements).squeeze()
        return obs

    def is_terminal(self, state: EnvState3D, params: EnvParams3D) -> bool:
        """Check whether state is terminal."""
        # Check number of steps in episode termination condition
        done = (
            (state.time >= params.max_steps_in_episode)
            | (jnp.abs(state.pos) > 2.0).any()
            | (jnp.abs(state.pos_obj) > 2.0).any()
            | (jnp.abs(state.omega) > 100.0).any()
            | (state.quat[3] < np.cos(jnp.pi/2 * 0.5))
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
        return spaces.Box(
            -1.0,
            1.0,
            shape=(42 + self.default_params.traj_obs_len * 6 + 15,),
            dtype=jnp.float32,
        )


def test_env(env: Quad3D, policy, render_video=False, num_episodes=3):
    # running environment
    rng = jax.random.PRNGKey(1)
    rng, rng_params = jax.random.split(rng)
    env_params = env.sample_params(rng_params)
    step_jit = jax.jit(env.step)

    state_seq, obs_seq, reward_seq = [], [], []
    rng, rng_reset = jax.random.split(rng)
    obs, env_state = env.reset(rng_reset, env_params)
    n_dones = 0
    n_steps = 0
    final_reward = 0.0
    a_mean = jnp.zeros([10, 4], dtype=jnp.float32)
    a_sigma = jnp.tile(jnp.eye(4, dtype=jnp.float32), [10, 1, 1])
    t0 = time_module.time()
    while True:
        state_seq.append(env_state)
        rng, rng_act, rng_step = jax.random.split(rng, 3)
        action, policy_info = policy(obs, env_state, env_params, rng_act, old_a_mean = a_mean, old_a_sigma = a_sigma)
        if 'a_mean' in policy_info.keys() and 'a_sigma' in policy_info.keys():
            a_mean = policy_info['a_mean']
            a_sigma = policy_info['a_sigma']
        next_obs, next_env_state, reward, done, info = step_jit(
            rng_step, env_state, action, env_params
        )
        if done:
            rng, rng_params = jax.random.split(rng)
            env_params = env.sample_params(rng_params)
        reward_seq.append(reward)
        obs_seq.append(obs)
        if done:
            n_dones += 1
            final_reward += reward
        n_steps += 1
        obs = next_obs
        env_state = next_env_state
        if n_dones >= num_episodes:
            break
    print(f"env running time: {time_module.time()-t0:.2f}s")

    # save state_seq (which is a list of EnvState3D:flax.struct.dataclass)
    with open("../results/state_seq.pkl", "wb") as f:
        pickle.dump(state_seq, f)

    # only keep the last 50 steps to plot
    state_seq = state_seq[-50:]
    obs_seq = obs_seq[-50:]
    reward_seq = reward_seq[-50:]
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
    for i in range(len(obs_seq[0]) // 10 + 1):
        plt.subplot(num_rows, 6, current_fig)
        current_fig += 1
        for j in range(10):
            idx = i * 10 + j
            plt.plot(time, [o[idx] for o in obs_seq], label=f"{idx}")
        plt.ylabel("obs")
        plt.legend(fontsize=6, ncol=2)

    # plot state
    for i, (name, value) in enumerate(state_seq[0].__dict__.items()):
        if name in ["pos_traj", "vel_traj"]:
            continue
        elif (("pos" in name) or ("vel" in name)) and ("tar" not in name):
            xyz = np.array([getattr(s, name) for s in state_seq])
            xyz_tar = np.array([getattr(s, name[:3] + "_tar") for s in state_seq])
            for i, subplot_name in zip(range(3), ["x", "y", "z"]):
                current_fig += 1
                plt.subplot(num_rows, 6, current_fig)
                plt.plot(time, xyz[:, i], label=f"{subplot_name}")
                plt.plot(time, xyz_tar[:, i], "--", label=f"{subplot_name}_tar")
                plt.ylabel(name + "_" + subplot_name)
                plt.legend()
        else:
            current_fig += 1
            plt.subplot(num_rows, 6, current_fig)
            plt.plot(time, [getattr(s, name) for s in state_seq])
            plt.ylabel(name)

    plt.xlabel("time")
    plt.savefig("../results/plot.png")
    print(f"plotting time: {time_module.time()-t0:.2f}s")
    print(f"average steps: {n_steps/n_dones}")
    print(f"average reward: {final_reward/n_dones}")

@pydataclass
class Args:
    task: str = "hovering"
    dynamic_model: str = "hybrid"

def main(args: Args):
    env = Quad3D(task=args.task, dynamic_model=args.dynamic_model)

    def random_policy(obs, state, params, rng):
        return env.action_space(env.default_params).sample(rng)

    def fixed_policy(obs, state, params, rng):
        '''
        policy to keep quadrotor straight up
        '''
        target_vec = jnp.array([0.0, 0.0, 1.0])
        target_vec_local = geom.rotate_with_quat(target_vec, geom.conjugate_quat(state.quat))
        rot_err = jnp.cross(target_vec, target_vec_local)
        w0 = 10.0
        zeta = 0.95
        I_diag = jnp.array([params.I[0,0], params.I[1,1], params.I[2,2]])
        kp = I_diag * (w0**2)
        kd = I_diag * 2.0 * zeta * w0
        torque = kp * rot_err + kd * (-state.omega)
        if state.pos[2]<0:
            thrust = params.g * (params.m + params.mo) / params.max_thrust * 2.0 - 1.0
            thrust = 1.0
        else:
            thrust = -1.5
        return jnp.array(
            [
                thrust,
                *(torque / params.max_torque),
            ]
        )
    
    def make_ppo_policy(path='../results/ppo_params.pkl'):
        from quadjax.train import ActorCritic
        network = ActorCritic(
            env.action_space(env.default_params).shape[0], activation='tanh')
        apply_fn = network.apply
        with open(path, 'rb') as f:
            network_params = pickle.load(f)

        def ppo_policy(obs, state, params, rng):
            return apply_fn(network_params, obs)[0].mean()
        # jit ppo_policy
        ppo_policy_jit = jax.jit(ppo_policy)

        return ppo_policy_jit

    print("starting test...")
    # enable NaN value detection
    # from jax import config
    # config.update("jax_debug_nans", True)
    # with jax.disable_jit():
    from quadjax.controllers.mppi import quad3d_free_mppi_policy
    mppi_jit = jax.jit(partial(quad3d_free_mppi_policy, env=env))
    test_env(env, policy=mppi_jit, num_episodes=1)

if __name__ == "__main__":
    main(tyro.cli(Args))
