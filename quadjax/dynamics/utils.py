import jax
from jax import numpy as jnp
from jax import lax
import chex
from typing import Tuple

from quadjax.dynamics.dataclass import EnvState3D



@jax.jit
def angle_normalize(x: float) -> float:
    """Normalize the angle - radians."""
    return ((x + jnp.pi) % (2 * jnp.pi)) - jnp.pi

@jax.jit
def get_hit_penalty(y: float, z: float) -> float:
    half_width = 0.05
    half_height = 0.3
    within_obs_y_range = jnp.abs(y) < half_width
    outof_obs_z_range = jnp.abs(z) > half_height
    hit_y_bound = within_obs_y_range & outof_obs_z_range
    hit_panelty = -jnp.clip(
        hit_y_bound.astype(jnp.float32)
        * jnp.minimum(half_width - jnp.abs(y), jnp.abs(z) - half_height)
        * 500.0,
        0,
        1,
    )
    return hit_panelty

def rk4(f, x, u, params, dt):
    k1 = f(x, u, params) * dt
    k2 = f(x + k1 / 2, u, params) * dt
    k3 = f(x + k2 / 2, u, params) * dt
    k4 = f(x + k3, u, params) * dt
    x_new = x + (k1 + 2 * k2 + 2 * k3 + k4) / 6
    return x_new


'''
trajectory related
'''
def generate_fixed_traj(
    max_steps: int, dt:float, key: chex.PRNGKey
) -> Tuple[chex.Array, chex.Array, chex.Array, chex.Array]:
    zeros = jnp.zeros((max_steps, 3))
    key_pos = jax.random.split(key)[0]
    pos = jax.random.uniform(key_pos, shape=(3,), minval=-1.0, maxval=1.0)
    pos_traj = zeros + pos
    vel_traj = zeros
    return pos_traj, vel_traj

def generate_lissa_traj(max_steps: int, dt:float, key: chex.PRNGKey) -> chex.Array:
    # get random amplitude and phase
    key_amp, key_phase = jax.random.split(key, 2)
    rand_amp = jax.random.uniform(key_amp, shape=(3, 2), minval=-1.0, maxval=1.0)
    rand_phase = jax.random.uniform(
        key_phase, shape=(3, 2), minval=-jnp.pi, maxval=jnp.pi
    )
    # get trajectory
    scale = 0.8
    ts = jnp.arange(
        0, max_steps + 50, dt
    )  # NOTE: do not use params for jax limitation
    w1 = 2 * jnp.pi * 0.25
    w2 = 2 * jnp.pi * 0.5

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

def generate_zigzag_traj(max_steps: int, dt:float, key: chex.PRNGKey) -> chex.Array:
    point_per_seg = 40
    num_seg = max_steps // point_per_seg + 1

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
            * jnp.ones((point_per_seg, 3)) / dt
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
    

'''
reward function
'''
@jax.jit
def jumping_reward_fn(state: EnvState3D):
    err_pos = jnp.linalg.norm(state.pos_tar - state.pos)
    err_vel = jnp.linalg.norm(state.vel_tar - state.vel)
    drone_panelty = get_hit_penalty(state.y, state.z) * 3.0
    obj_panelty = get_hit_penalty(state.y_obj, state.z_obj) * 3.0
    return 1.0 - 0.6 * err_pos - 0.15 * err_vel + (drone_panelty + obj_panelty)

@jax.jit
def hovering_reward_fn(state: EnvState3D):
    err_pos = jnp.linalg.norm(state.pos_tar - state.pos)
    err_vel = jnp.linalg.norm(state.vel_tar - state.vel)
    return 1.0 - 0.6 * err_pos - 0.1 * err_vel

@jax.jit
def tracking_reward_fn(state: EnvState3D):
    err_pos = jnp.linalg.norm(state.pos_tar - state.pos)
    err_vel = jnp.linalg.norm(state.vel_tar - state.vel)
    return 1.0 - 0.8 * err_pos - 0.05 * err_vel

'''
visualization functions
'''
def plot_states(state_seq, obs_seq, reward_seq, env_params):
    import matplotlib.pyplot as plt
    import numpy as np
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
    plt.savefig("../../results/plot.png")