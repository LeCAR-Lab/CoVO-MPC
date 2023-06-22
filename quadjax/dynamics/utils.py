from flax import struct
from jax import numpy as jnp
import jax


def default_array(array):
    return struct.field(default_factory=lambda: jnp.array(array))

@struct.dataclass
class EnvState:
    y: float
    z: float
    theta: float
    phi: float
    y_dot: float
    z_dot: float
    theta_dot: float
    phi_dot: float
    last_thrust: float  # Only needed for rendering
    last_tau: float  # Only needed for rendering
    time: int
    y_traj: jnp.ndarray
    z_traj: jnp.ndarray
    y_dot_traj: jnp.ndarray
    z_dot_traj: jnp.ndarray
    y_tar: float
    z_tar: float
    y_dot_tar: float
    z_dot_tar: float
    y_hook: float
    z_hook: float
    y_hook_dot: float
    z_hook_dot: float
    y_obj: float
    z_obj: float
    y_obj_dot: float
    z_obj_dot: float
    f_rope: float
    f_rope_y: float
    f_rope_z: float
    l_rope: float


@struct.dataclass
class EnvParams:
    max_speed: float = 8.0
    max_torque: float = 0.012
    max_thrust: float = 0.8
    dt: float = 0.02
    g: float = 9.81  # gravity
    m: float = 0.03  # mass
    I: float = 2.0e-5  # moment of inertia
    mo: float = 0.005  # mass of the object attached to the rod
    l: float = 0.3  # length of the rod
    delta_yh: float = 0.03  # y displacement of the hook from the quadrotor center
    delta_zh: float = -0.06  # z displacement of the hook from the quadrotor center
    max_steps_in_episode: int = 300
    rope_taut_therehold: float = 1e-4
    traj_obs_len: int = 5
    traj_obs_gap: int = 5


@struct.dataclass
class Action:
    thrust: float
    tau: float


@struct.dataclass
class EnvState3D:
    # meta state variable for taut state
    pos: jnp.ndarray  # (x,y,z)
    vel: jnp.ndarray  # (x,y,z)
    quat: jnp.ndarray  # quaternion (x,y,z,w)
    omega: jnp.ndarray  # angular velocity (x,y,z)
    theta_rope: float  # angle of the rope
    phi_rope: float  # angle of the rope
    theta_rope_dot: float  # angle of the rope
    phi_rope_dot: float  # angle of the rope
    # target trajectory
    pos_traj: jnp.ndarray
    vel_traj: jnp.ndarray
    pos_tar: jnp.ndarray
    vel_tar: jnp.ndarray
    # hook state
    pos_hook: jnp.ndarray
    vel_hook: jnp.ndarray
    # object state
    pos_obj: jnp.ndarray
    vel_obj: jnp.ndarray
    # rope state
    f_rope_norm: float
    f_rope: jnp.ndarray
    l_rope: float
    # other variables
    zeta: jnp.ndarray  # S^2 unit vector (x,y,z)
    zeta_dot: jnp.ndarray  # S^2 (x,y,z)
    last_thrust: float
    last_torque: jnp.ndarray  # torque in the local frame
    time: int


@struct.dataclass
class EnvParams3D:
    max_speed: float = 8.0
    max_torque: jnp.ndarray = default_array([9e-3, 9e-3, 2e-3])
    max_thrust: float = 0.8
    dt: float = 0.02
    g: float = 9.81  # gravity
    m: float = 0.03  # mass
    I: jnp.ndarray = default_array(
        [1.7e-5, 1.7e-5, 3.0e-5])  # moment of inertia
    mo: float = 0.005  # mass of the object attached to the rod
    l: float = 0.3  # length of the rod
    hook_offset: jnp.ndarray = default_array([0.03, 0.02, -0.06])
    max_steps_in_episode: int = 300
    rope_taut_therehold: float = 1e-4
    traj_obs_len: int = 5
    traj_obs_gap: int = 5


@struct.dataclass
class Action3D:
    thrust: float
    torque: jnp.ndarray


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