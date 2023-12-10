from flax import struct
from jax import numpy as jnp
from typing import Optional, Union


def default_array(array):
    return struct.field(default_factory=lambda: jnp.array(array))


@struct.dataclass
class EnvState3D:
    # meta state variable for taut state
    pos: jnp.ndarray  # (x,y,z)
    vel: jnp.ndarray  # (x,y,z)
    quat: jnp.ndarray  # quaternion (x,y,z,w)
    omega: jnp.ndarray  # angular velocity (x,y,z)
    omega_tar: jnp.ndarray  # angular velocity (x,y,z)
    # target trajectory
    pos_traj: jnp.ndarray
    vel_traj: jnp.ndarray
    acc_traj: jnp.ndarray
    pos_tar: jnp.ndarray
    vel_tar: jnp.ndarray
    acc_tar: jnp.ndarray
    # other variables
    last_thrust: float
    last_torque: jnp.ndarray  # torque in the local frame
    time: int
    f_disturb: jnp.ndarray

    # trajectory information for adaptation
    vel_hist: jnp.ndarray
    omega_hist: jnp.ndarray
    action_hist: jnp.ndarray

    # control params is float or dataclass
    control_params: Union[float, struct.dataclass] = 0.0


@struct.dataclass
class EnvParams3D:
    max_speed: float = 8.0
    max_torque: jnp.ndarray = default_array([9e-3, 9e-3, 2e-3])
    max_omega: jnp.ndarray = default_array([10.0, 10.0, 3.0])
    max_thrust: float = 0.8
    dt: float = 0.02
    g: float = 9.81  # gravity

    m: float = 0.027  # mass
    m_mean: float = 0.027  # mass
    m_std: float = 0.003  # mass

    I: jnp.ndarray = default_array(
        [[1.7e-5, 0.0, 0.00], [0.0, 1.7e-5, 0.0], [0.0, 0.0, 3.0e-5]]
    )  # moment of inertia
    I_diag_mean: jnp.ndarray = default_array(
        [1.7e-5, 1.7e-5, 3.0e-5]
    )  # moment of inertia
    I_diag_std: jnp.ndarray = default_array(
        [0.2e-5, 0.2e-5, 0.3e-5]
    )  # moment of inertia

    l: float = 0.3  # length of the rod
    l_mean: float = 0.3
    l_std: float = 0.1

    hook_offset: jnp.ndarray = default_array([0.0, 0.0, -0.01])
    hook_offset_mean: jnp.ndarray = default_array([0.0, 0.0, -0.02])
    hook_offset_std: jnp.ndarray = default_array([0.01, 0.01, 0.01])

    action_scale: float = 1.0
    action_scale_mean: float = 1.0
    action_scale_std: float = 0.1

    # 1st order dynamics
    alpha_bodyrate: float = 0.5
    alpha_thrust: float = 0.6
    alpha_bodyrate_mean: float = 0.5
    alpha_bodyrate_std: float = 0.1

    max_steps_in_episode: int = 300
    rope_taut_therehold: float = 1e-4
    traj_obs_len: int = 5
    traj_obs_gap: int = 5

    # disturbance related parameters
    d_offset: jnp.ndarray = default_array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    disturb_period: int = 50
    disturb_scale: float = 0.2
    disturb_params: jnp.ndarray = default_array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    # curriculum related parameters
    curri_params: float = 1.0

    # RMA related parameters
    adapt_horizon: int = 4

    # noise related parameters
    dyn_noise_scale: float = 0.05
    obs_noise_scale: float = 0.05


@struct.dataclass
class Action3D:
    thrust: float
    torque: jnp.ndarray