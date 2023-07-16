import jax
from jax import numpy as jnp
from functools import partial

from quadjax.quad3d import Quad3D
from quadjax.dynamics.dataclass import EnvParams, EnvParams3D, Action, Action3D, EnvState, EnvState3D
from quadjax.dynamics import geom

class PIDController:
    """PID controller for attitude rate control

    Returns:
        _type_: _description_
    """

    def __init__(self, kp, ki, kd, ki_max, integral, last_error):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.ki_max = ki_max
        self.integral = integral
        self.last_error = last_error
        self.reset()

    def reset(self):
        self.integral *= 0.0
        self.last_error *= 0.0

    @partial(jax.jit, static_argnums=(0,))
    def update(self, error, dt):
        self.integral += error * dt
        self.integral = jnp.clip(self.integral, -self.ki_max, self.ki_max)
        derivative = (error - self.last_error) / dt
        self.last_error = error
        return self.kp * error + self.ki * self.integral + self.kd * derivative

def quad3d_trans_pid_policy(
        obs: jnp.ndarray,
        env_state: EnvState3D,
        env_params: EnvParams3D,
        rng: jax.random.PRNGKey,
    ):
    # get object target force
    w0 = 8.0
    zeta = 0.95
    kp = env_params.mo * (w0**2)
    kd = env_params.mo * 2.0 * zeta * w0
    target_force_obj = (
        kp * (env_state.pos_tar - env_state.pos_obj)
        + kd * (env_state.vel_tar - env_state.vel_obj)
        + env_params.mo * jnp.array([0.0, 0.0, env_params.g])
    )
    target_force_obj_norm = jnp.linalg.norm(target_force_obj)
    zeta_target = -target_force_obj / target_force_obj_norm
    pos_tar_quad = env_state.pos_obj - env_params.l * zeta_target
    vel_tar_quad = env_state.vel_tar

    # get drone target force
    w0 = 10.0
    zeta = 0.95
    kp = env_params.m * (w0**2)
    kd = env_params.m * 2.0 * zeta * w0
    target_force = (
        kp * (pos_tar_quad - env_state.pos)
        + kd * (vel_tar_quad - env_state.vel)
        + env_params.m * jnp.array([0.0, 0.0, env_params.g])
        + target_force_obj
    )
    thrust = jnp.linalg.norm(target_force)
    target_unitvec = target_force / thrust
    # target_unitvec = jnp.array([jnp.sin(jnp.pi/6), 0.0, jnp.cos(jnp.pi/6)]) # DEBUG
    target_unitvec_local = geom.rotate_with_quat(
        target_unitvec, geom.conjugate_quat(env_state.quat)
    )

    w0 = 10.0
    zeta = 0.95
    kp = env_params.I[0] * (w0**2)
    kd = env_params.I[0] * 2.0 * zeta * w0
    current_unitvec_local = jnp.array([0.0, 0.0, 1.0])
    rot_axis = jnp.cross(current_unitvec_local, target_unitvec_local)
    rot_angle = jnp.arccos(jnp.dot(current_unitvec_local, target_unitvec_local))
    omega_local = geom.rotate_with_quat(
        env_state.omega, geom.conjugate_quat(env_state.quat)
    )
    torque = kp * rot_angle * rot_axis / jnp.linalg.norm(rot_axis) + kd * (
        -omega_local
    )

    # convert into action space
    thrust_normed = jnp.clip(
        thrust / env.default_params.max_thrust * 2.0 - 1.0, -1.0, 1.0
    )
    tau_normed = jnp.clip(torque / env.default_params.max_torque, -1.0, 1.0)
    return jnp.array([thrust_normed, tau_normed[0], tau_normed[1], 0.0])

def quad3d_free_pid_policy(
        obs: jnp.ndarray,
        env_state: EnvState3D,
        env_params: EnvParams3D,
        rng: jax.random.PRNGKey,
):
    # get drone target force
    kp = 3.0**2
    kd = 2.0 * 1.05 * 3.0
    target_force = env_params.m * (
        kp * (env_state.pos_tar - env_state.pos)
        + kd * (env_state.vel_tar - env_state.vel)
        + jnp.array([0.0, 0.0, env_params.g])
    )
    thrust = jnp.linalg.norm(target_force)
    # target_unitvec = jnp.array([jnp.sin(jnp.pi/6), 0.0, jnp.cos(jnp.pi/6)]) # DEBUG
    # target_unitvec_local = geom.rotate_with_quat(
    #     target_unitvec, geom.conjugate_quat(env_state.quat)
    # )

    target_unitvec = target_force / thrust
    x, y, z = target_unitvec
    angle = jnp.arctan2(jnp.sqrt(x**2 + y**2), z)
    if (x**2 + y**2) > 1e-6:
        axis = 1.0 / jnp.sqrt(x**2+y**2) * jnp.array([-y, x, 0.0])
    else:
        axis = jnp.array([1.0, 0.0, 0.0])
    quat_target_world = geom.rotvec2quat(angle, axis)
    # quat_target_world = geom.euler2quat(roll, pitch, yaw)
    quat_target = geom.multiple_quat(geom.conjugate_quat(env_state.quat), quat_target_world)
    quat_error = quat_target
    qx, qy, qz, qw = quat_error
    quat_red = 1.0 / jnp.sqrt(qw**2+qz**2) * jnp.array([qw*qx-qy*qz, qw*qy+qx*qz, 0.0, qw**2+qz**2])
    quat_yaw = 1.0 / jnp.sqrt(qw**2+qz**2) * jnp.array([0.0, 0.0, qz, qw])
    kpxy = 38.0**2
    kdxy = 2.0 * 1.30 * 38.0
    kpz = 6.5**2
    kdz = 2.0 * 1.0 * 6.5
    alpha = kpxy * quat_red[:3] + jnp.sign(qw) * kpz * quat_yaw[:3] + jnp.array([kdxy, kdxy, kdz]) * (0.0 - env_state.omega)
    torque = env_params.I @ alpha - jnp.cross((env_params.I @ env_state.omega), env_state.omega)
    
    # convert into action space
    thrust_normed = jnp.clip(
        thrust / env_params.max_thrust * 2.0 - 1.0, -1.0, 1.0
    )
    tau_normed = jnp.clip(torque / env_params.max_torque, -1.0, 1.0)
    return jnp.array([thrust_normed, *tau_normed])
