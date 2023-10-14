import jax
from jax import numpy as jnp

@jax.jit
def conjugate_quat(quat: jnp.ndarray) -> jnp.ndarray:
    """Conjugate of quaternion (x, y, z, w)."""
    return jnp.array([-quat[0], -quat[1], -quat[2], quat[3]])

@jax.jit
def integrate_quat(quat: jnp.ndarray, omega: jnp.ndarray, dt: float) -> jnp.ndarray:
    """Integrate quaternion with angular velocity omega."""
    quat_dot = 0.5 * multiple_quat(quat, jnp.concatenate([omega, jnp.zeros(1)]))
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

# Quaternion functions
@jax.jit
def hat(v: jnp.ndarray) -> jnp.ndarray:
    return jnp.array([[0, -v[2], v[1]],
                     [v[2], 0, -v[0]],
                     [-v[1], v[0], 0]])

@jax.jit
def L(q: jnp.ndarray) -> jnp.ndarray:
    '''
    L(q) = [sI + hat(v), v; -v^T, s]
    left multiplication matrix of a quaternion
    '''
    s = q[3]
    v = q[:3]
    right = jnp.hstack((v, s)).reshape(-1, 1)
    left_up = s * jnp.eye(3) + hat(v)
    left_down = -v
    left = jnp.vstack((left_up, left_down))
    return jnp.hstack((left, right))

H = jnp.vstack((jnp.eye(3), jnp.zeros((1, 3))))

@jax.jit
def E(q: jnp.ndarray)->jnp.ndarray:
    '''
    reduced matrix for quadrotor state
    '''
    I3 = jnp.eye(3)
    I6 = jnp.eye(6)
    H = jnp.vstack((jnp.eye(3), jnp.zeros((1, 3))))
    G = L(q) @ H
    return jax.scipy.linalg.block_diag(I3, G, I6)

@jax.jit
def qtoQ(q: jnp.ndarray) -> jnp.ndarray:
    '''
    covert a quaternion to a 3x3 rotation matrix
    '''
    T = jnp.diag(jnp.array([-1, -1, -1, 1]))
    H = jnp.vstack((jnp.eye(3), jnp.zeros((1, 3))))
    # H = jnp.vstack((jnp.eye(3), jnp.zeros((1, 3)))) # used to convert a 3d vector to 4d vector
    Lq = L(q)
    return H.T @ T @ Lq @ T @ Lq @ H

@jax.jit
def rptoq(phi: jnp.ndarray) -> jnp.ndarray:
    return 1/jnp.sqrt(1+jnp.dot(phi, phi))*jnp.concatenate((phi, jnp.array([1])))

@jax.jit
def qtorp(q: jnp.ndarray) -> jnp.ndarray:
    return q[:3]/q[3]

def qtorpy(q: jnp.ndarray) -> jnp.ndarray:
    # convert quaternion (x, y, z, w) to roll, pitch, yaw

    roll = jnp.arctan2(2*(q[3]*q[0]+q[1]*q[2]), 1-2*(q[0]**2+q[1]**2))
    pitch = jnp.arcsin(2*(q[3]*q[1]-q[2]*q[0]))
    yaw = jnp.arctan2(2*(q[3]*q[2]+q[0]*q[1]), 1-2*(q[1]**2+q[2]**2))

    return jnp.array([roll, pitch, yaw])

def axisangletoR(axis: jnp.ndarray, angle: float) -> jnp.ndarray:
    # convert axis-angle to quaternion
    # axis: 3d vector
    # angle: radian
    # output: rotation matrix
    axis = axis/jnp.linalg.norm(axis)
    return jnp.eye(3) + jnp.sin(angle)*hat(axis) + (1-jnp.cos(angle))*hat(axis)@hat(axis)

def vee(R: jnp.ndarray):
    # convert skew-symmetric matrix to vector
    # R: 3x3 skew-symmetric matrix
    # output: 3d vector
    return jnp.array([R[2, 1], R[0, 2], R[1, 0]])