import jax
from jax import numpy as jnp
from jax import lax
import chex
from typing import Tuple

import quadjax
from quadjax.dynamics.dataclass import EnvState3D, EnvParams3D


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

'''
disturbance related
'''
@jax.jit
def constant_disturbance(x:jnp.ndarray, u:jnp.ndarray, params:EnvParams3D):
    return params.d_offset

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
    return pos_traj, zeros, zeros

def generate_jumping_fixed_traj(
    max_steps: int, dt:float, key: chex.PRNGKey
) -> Tuple[chex.Array, chex.Array, chex.Array, chex.Array]:
    zeros = jnp.zeros((max_steps, 3))
    key_pos = jax.random.split(key)[0]
    pos = jax.random.uniform(key_pos, shape=(3,), minval=-1.0, maxval=1.0)
    # for pos[0]>0 add 0.3 to x, else add -0.3 to x
    # pos = jnp.where(pos[0]>0, pos + jnp.array([0.3, 0.0, 0.0]), pos + jnp.array([-0.3, 0.0, 0.0]))
    pos = pos.at[0].set(jnp.abs(pos[0]) + 0.3)
    pos_traj = zeros + pos
    vel_traj = zeros
    return pos_traj, vel_traj

def generate_given_fixed_traj(
    pos: jnp.ndarray, max_steps: int, dt:float, key: chex.PRNGKey
) -> Tuple[chex.Array, chex.Array, chex.Array, chex.Array]:
    zeros = jnp.zeros((max_steps, 3))
    pos_traj = zeros + pos
    vel_traj = zeros
    return pos_traj, vel_traj

def generate_jumping_fixed_traj(
    max_steps: int, dt:float, key: chex.PRNGKey
) -> Tuple[chex.Array, chex.Array, chex.Array, chex.Array]:
    zeros = jnp.zeros((max_steps, 3))
    key_pos = jax.random.split(key)[0]
    pos = jax.random.uniform(key_pos, shape=(3,), minval=-1.0, maxval=1.0)
    # for pos[0]>0 add 0.3 to x, else add -0.3 to x
    # pos = jnp.where(pos[0]>0, pos + jnp.array([0.3, 0.0, 0.0]), pos + jnp.array([-0.3, 0.0, 0.0]))
    pos = pos.at[0].set(jnp.abs(pos[0]) + 0.3)
    pos_traj = zeros + pos
    vel_traj = zeros
    return pos_traj, vel_traj

def generate_given_fixed_traj(
    pos: jnp.ndarray, max_steps: int, dt:float, key: chex.PRNGKey
) -> Tuple[chex.Array, chex.Array, chex.Array, chex.Array]:
    zeros = jnp.zeros((max_steps, 3))
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
    scale = 1.0
    ts = jnp.arange(0, max_steps + 50)*dt  # NOTE: do not use params for jax limitation
    w1 = 2 * jnp.pi * 0.2
    w2 = 2 * jnp.pi * 0.4

    pos_traj = scale * jnp.stack(
        [
            rand_amp[i, 0] * jnp.sin(w1 * ts + rand_phase[i, 0])
            + rand_amp[i, 1] * jnp.sin(w2 * ts + rand_phase[i, 1])
            for i in range(3)
        ], 
        axis=1
    )
    pos_traj = pos_traj - pos_traj[0]

    vel_traj = scale * jnp.stack(
        [
            rand_amp[i, 0] * w1 * jnp.cos(w1 * ts + rand_phase[i, 0])
            + rand_amp[i, 1] * w2 * jnp.cos(w2 * ts + rand_phase[i, 1])
            for i in range(3)
        ], 
        axis=1
    )

    acc_traj = scale * jnp.stack(
        [
            -rand_amp[i, 0] * w1**2 * jnp.sin(w1 * ts + rand_phase[i, 0])
            - rand_amp[i, 1] * w2**2 * jnp.sin(w2 * ts + rand_phase[i, 1])
            for i in range(3)
        ], 
        axis=1
    )

    return pos_traj, vel_traj, acc_traj

def generate_lissa_traj_2d(max_steps: int, dt: float, key: chex.PRNGKey) -> chex.Array:
    # get random amplitude and phase
    key_amp, key_phase = jax.random.split(key, 2)
    rand_amp = jax.random.uniform(key_amp, shape=(2, 2), minval=-1.0, maxval=1.0)
    rand_phase = jax.random.uniform(
        key_phase, shape=(2, 2), minval=-jnp.pi, maxval=jnp.pi
    )
    
    # get trajectory
    scale = 0.5
    ts = jnp.arange(0, max_steps + 50)*dt  # NOTE: do not use params for jax limitation
    w1 = 2 * jnp.pi * 0.3
    w2 = 2 * jnp.pi * 0.6

    pos_traj = scale * jnp.stack(
        [
            rand_amp[i, 0] * jnp.sin(w1 * ts + rand_phase[i, 0])
            + rand_amp[i, 1] * jnp.sin(w2 * ts + rand_phase[i, 1])
            for i in range(2)
        ],
        axis=1
    )
    pos_traj = pos_traj - pos_traj[0]

    vel_traj = scale * jnp.stack(
        [
            rand_amp[i, 0] * w1 * jnp.cos(w1 * ts + rand_phase[i, 0])
            + rand_amp[i, 1] * w2 * jnp.cos(w2 * ts + rand_phase[i, 1])
            for i in range(2)
        ],
        axis=1
    )
    
    return pos_traj, vel_traj

def generate_zigzag_traj(max_steps: int, dt:float, key: chex.PRNGKey) -> chex.Array:
    point_per_seg = 40
    num_seg = max_steps // point_per_seg + 1

    key_keypoints = jax.random.split(key, num_seg)
    key_angles = jax.random.split(key, num_seg)

    # generate a 3d unit vector
    prev_point = jax.random.uniform(key_keypoints[0], shape=(3,), minval=-1.0, maxval=1.0)
    prev_point = prev_point / jnp.linalg.norm(prev_point) * 0.1

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
        distance = jax.random.uniform(key_keypoint, minval=1.0, maxval=1.5)

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
    point_traj_segs, point_dot_traj_segs = [], []
    _, (point_traj_segs, point_dot_traj_segs) = lax.scan(
        update_fn, initial_carry, jnp.arange(num_seg)
    )

    pos_traj = jnp.concatenate(point_traj_segs, axis=0)
    pos_traj = pos_traj - pos_traj[0]   
    vel_traj = jnp.concatenate(point_dot_traj_segs, axis=0)

    return pos_traj, vel_traj, jnp.zeros_like(pos_traj)



def generate_zigzag_traj_2d(max_steps: int, dt: float, key: chex.PRNGKey) -> chex.Array:
    point_per_seg = 20
    num_seg = max_steps // point_per_seg + 1

    key_keypoints = jax.random.split(key, num_seg)
    key_angles = jax.random.split(key, num_seg)

    # generate a 2d unit vector
    prev_point = jax.random.uniform(key_keypoints[0], shape=(2,), minval=-1.0, maxval=1.0)
    prev_point = prev_point / jnp.linalg.norm(prev_point) * 0.1

    def update_fn(carry, i):
        key_keypoint, key_angle, prev_point = carry

        # Calculate the unit vector pointing to the center
        vec_to_center = -prev_point / jnp.linalg.norm(prev_point)

        # Sample random rotation angle theta from [-pi/3, pi/3]
        delta_theta = jax.random.uniform(key_angle, minval=-jnp.pi / 3, maxval=jnp.pi / 3)

        # Calculate new direction
        theta = jnp.arctan2(vec_to_center[1], vec_to_center[0]) + delta_theta
        new_direction = jnp.array([jnp.cos(theta), jnp.sin(theta)])

        # Sample the distance from [1.5, 2.5]
        distance = jax.random.uniform(key_keypoint, minval=1.0, maxval=1.5)

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
            (next_point - prev_point) / (point_per_seg + 1) * jnp.ones((point_per_seg, 2)) / dt
        )

        carry = (key_keypoints[i + 1], key_angles[i + 1], next_point)
        return carry, (point_traj_seg, point_dot_traj_seg)

    initial_carry = (key_keypoints[1], key_angles[1], prev_point)
    point_traj_segs, point_dot_traj_segs = [], []
    _, (point_traj_segs, point_dot_traj_segs) = lax.scan(
        update_fn, initial_carry, jnp.arange(num_seg)
    )

    pos_traj = jnp.concatenate(point_traj_segs, axis=0)
    pos_traj = pos_traj - pos_traj[0]   
    vel_traj = jnp.concatenate(point_dot_traj_segs, axis=0)

    return pos_traj, vel_traj

    

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
def get_hit_reward(pos, params):
    r = 3.0
    gap_size = (1.0-params.curri_params)*1.0 + 0.1
    a = r - gap_size/2.0
    b = 0.1
    YZ = jnp.sqrt((pos[1])**2 + (pos[2])**2) - r
    l = jnp.sqrt(((pos[0])/b)**2 + (YZ/a)**2)
    return -jnp.clip(jnp.log(1.0+10.0*(1.0-jnp.clip(l, 0.0, 1.0))), 0.0, 2.0)

@jax.jit
def jumping_obj_reward_fn(state: EnvState3D, params: EnvParams3D):
    err_pos = jnp.linalg.norm(state.pos_tar - state.pos_obj)
    small_pos_err = err_pos < 0.2
    err_vel = jnp.linalg.norm(state.vel_tar - state.vel_obj)
    drone_hit_rew = get_hit_reward(state.pos, params)
    obj_hit_rew = get_hit_reward(state.pos_obj, params)
    return 1.5 - 1.0 * log_pos_fn(err_pos) + small_pos_err * 2.0 * (1.0-jnp.clip(0.2*err_vel, 0, 1)) + 3.0*(drone_hit_rew + obj_hit_rew)

@jax.jit
def log_pos_fn(err_pos):
    return err_pos * 0.4 + \
        jnp.clip(jnp.log(err_pos + 1) * 4, 0, 1) * 0.4 + \
        jnp.clip(jnp.log(err_pos + 1) * 8, 0, 1) * 0.2 + \
        jnp.clip(jnp.log(err_pos + 1) * 16, 0, 1) * 0.1 + \
        jnp.clip(jnp.log(err_pos + 1) * 32, 0, 1) * 0.1

@jax.jit
def tracking_reward_fn(state: EnvState3D, params = None):
    err_pos = jnp.linalg.norm(state.pos_tar - state.pos)
    err_vel = jnp.linalg.norm(state.vel_tar - state.vel)
    reward = 1.0 - \
        0.05 * err_vel - \
        log_pos_fn(err_pos)
    return reward
    
@jax.jit
def tracking_penyaw_reward_fn(state: EnvState3D, params = None):
    err_pos = jnp.linalg.norm(state.pos_tar - state.pos)
    err_vel = jnp.linalg.norm(state.vel_tar - state.vel)
    q = state.quat
    yaw = jnp.arctan2(2*(q[3]*q[2]+q[0]*q[1]), 1-2*(q[1]**2+q[2]**2))
    reward = 1.3 - \
        0.05 * err_vel - \
        log_pos_fn(err_pos) - \
        jnp.abs(yaw) * 0.2
        # jnp.abs(state.omega[2]) * 0.02 - \

    return reward

@jax.jit
def tracking_penyaw_obj_reward_fn(state: EnvState3D, params:EnvParams3D):
    err_pos = jnp.linalg.norm(state.pos_tar+jnp.array([0.0, 0.0, params.l])-params.hook_offset-state.pos)
    err_vel = jnp.linalg.norm(state.vel)
    err_pos_obj = jnp.linalg.norm(state.pos_tar - state.pos_obj)
    err_vel_obj = jnp.linalg.norm(state.vel_tar - state.vel_obj)
    reward = 0.0 - \
        0.03 * err_vel - \
        0.6 * err_pos - \
        0.05 * err_vel_obj - \
        log_pos_fn(err_pos_obj) - \
        jnp.linalg.norm(state.omega) * 0.01 - \
        jnp.abs(state.omega[2]) * 0.05
    reward = 1.0 + reward/3.5

    return reward

'''
visualization functions
'''
def plot_states(state_seq, obs_seq, reward_seq, env_params, filename=''):
    import matplotlib.pyplot as plt
    import numpy as np

    # check if quat in state_seq, if true, then add a new item called rpy (roll, pitch, yaw)
    if "quat" in state_seq[0]:
        for i, state in enumerate(state_seq):
            rpy = quadjax.dynamics.qtorpy(state['quat'])
            state_seq[i]["rpy"] = rpy

    # plot results
    num_figs = len(state_seq[0]) + 20
    time = np.arange(len(state_seq)) * env_params.dt

    # calculate number of rows needed
    plot_per_row = 4
    num_rows = int(jnp.ceil(num_figs / plot_per_row))

    # create num_figs subplots
    plt.subplots(num_rows, plot_per_row, figsize=(6*plot_per_row, 2 * num_rows))

    # plot reward
    plt.subplot(num_rows, plot_per_row, 1)
    plt.plot(time, reward_seq)
    plt.ylabel("reward")

    # plot obs
    # plot 10 obs in a subplot
    current_fig = 2
    for i in range(len(obs_seq[0]) // 10 + 1):
        plt.subplot(num_rows, plot_per_row, current_fig)
        current_fig += 1
        for j in range(10):
            idx = i * 10 + j
            plt.plot(time, [o[idx] for o in obs_seq], label=f"{idx}")
        plt.ylabel("obs")
        plt.legend(fontsize=6, ncol=2)

    # plot state
    for i, (name, value) in enumerate(state_seq[0].items()):
        if name in ["pos_traj", "vel_traj", "acc_traj", "control_params", "vel_hist", "omega_hist", "action_hist"]:
            continue
        elif (("pos" in name) or ("vel" in name)) and ("tar" not in name):
            xyz = np.array([s[name] for s in state_seq])
            if 'hat' in name:
                xyz_tar = np.array([s[f"{name[:3]}"] for s in state_seq])
            else:
                xyz_tar = np.array([s[f"{name[:3]}_tar"] for s in state_seq])
            if xyz.shape[1] == 3:
                scan_range = zip(range(3), ["x", "y", "z"])
            elif xyz.shape[1] == 2:
                scan_range = zip(range(2), ["y", "z"])
            else: 
                print(f"[DEBUG] ignore {name} with shape {xyz.shape} while plotting")
            for i, subplot_name in scan_range:
                current_fig += 1
                plt.subplot(num_rows, plot_per_row, current_fig)
                plt.plot(time, xyz[:, i], label=f"{subplot_name}")
                plt.plot(time, xyz_tar[:, i], "--", label=f"{subplot_name}_tar")
                plt.ylabel(f"{name}_{subplot_name}")
                plt.legend()
        elif "d_hat" in name:
            current_fig += 1
            plt.subplot(num_rows, plot_per_row, current_fig)
            plt.plot(time, [s[name] for s in state_seq], label=f"{name}")
            plt.plot(time, [s["f_disturb"]/0.03 for s in state_seq], "--", label="real")
            plt.ylabel(name)
            plt.legend()
        else:
            current_fig += 1
            plt.subplot(num_rows, plot_per_row, current_fig)
            plt.plot(time, [s[name] for s in state_seq])
            plt.ylabel(name)

    plt.xlabel("time")
    plt.savefig(f"{quadjax.get_package_path()}/../results/render_plot_{filename}.png")

def sample_sphere(key: chex.PRNGKey, R, center):
    """Sample a point inside a sphere."""
    theta_key, phi_key, r_key = jax.random.split(key, 3)
    theta = jax.random.uniform(theta_key, shape=(1,), minval=0, maxval=2 * jnp.pi)
    phi = jax.random.uniform(phi_key, shape=(1,), minval=0, maxval=jnp.pi)
    R = jax.random.uniform(r_key, shape=(1,), minval=0, maxval=R)

    x = R * jnp.sin(phi) * jnp.cos(theta) + center[0]
    y = R * jnp.sin(phi) * jnp.sin(theta) + center[1]
    z = R * jnp.cos(phi) + center[2]

    return jnp.concatenate([x, y, z], axis=0)