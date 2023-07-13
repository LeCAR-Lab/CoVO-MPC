import jax
from jax import numpy as jnp


@jax.jit
def conjugate_quat(quat: jnp.ndarray) -> jnp.ndarray:
    """Conjugate of quaternion (x, y, z, w)."""
    return jnp.array([-quat[0], -quat[1], -quat[2], quat[3]])


@jax.jit
def integrate_quat(quat: jnp.ndarray, omega: jnp.ndarray, dt: float) -> jnp.ndarray:
    """Integrate quaternion with *local* angular velocity omega."""
    quat_dot = 0.5 * multiple_quat(jnp.concatenate([omega, jnp.zeros(1)]), quat)
    quat = quat + dt * quat_dot
    quat = quat / jnp.linalg.norm(quat)
    return quat


@jax.jit
def multiple_quat(quat1: jnp.ndarray, quat2: jnp.ndarray) -> jnp.ndarray:
    """Multiply two quaternions (x, y, z, w)."""
    quat = jnp.zeros(4)
    w = quat1[3] * quat2[3] - jnp.dot(quat1[:3], quat2[:3])
    xyz = quat1[3] * quat2[:3] + quat2[3] * quat1[:3] + jnp.cross(quat1[:3], quat2[:3])
    quat = quat.at[3].set(w)
    quat = quat.at[:3].set(xyz)
    return quat


@jax.jit
def rotate_with_quat(v: jnp.ndarray, quat: jnp.ndarray) -> jnp.ndarray:
    """Rotate the vector v with quaternion quat (x, y, z, w)."""
    v = jnp.concatenate([v, jnp.zeros(1)])
    v_rot = multiple_quat(multiple_quat(quat, v), conjugate_quat(quat))
    return v_rot[:3]

@jax.jit
def rotvec2quat(angle: float, axis: jnp.ndarray) -> jnp.ndarray:
    """Convert rotation vector (3,) to quaternion (x, y, z, w)."""
    quat = jnp.concatenate([jnp.sin(angle / 2) * axis, jnp.cos(angle / 2) * jnp.ones(1)])
    return quat

@jax.jit
def euler2quat(roll: float, pitch: float, yaw: float) -> jnp.ndarray:
    """Convert Euler xyz angles (in radians) to quaternion (x, y, z, w)."""
    cr = jnp.cos(roll * 0.5)
    sr = jnp.sin(roll * 0.5)
    cp = jnp.cos(pitch * 0.5)
    sp = jnp.sin(pitch * 0.5)
    cy = jnp.cos(yaw * 0.5)
    sy = jnp.sin(yaw * 0.5)
    quat = jnp.zeros(4)
    quat = quat.at[0].set(cy * cp * sr - sy * sp * cr)
    quat = quat.at[1].set(sy * cp * sr + cy * sp * cr)
    quat = quat.at[2].set(sy * cp * cr - cy * sp * sr)
    quat = quat.at[3].set(cy * cp * cr + sy * sp * sr)
    return quat
