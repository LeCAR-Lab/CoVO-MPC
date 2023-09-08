import jax
from jax import numpy as jnp

from quadjax.dynamics.dataclass import EnvParams3D, EnvState3D, Action3D
from quadjax.dynamics import geom, utils

def get_free_dynamics_3d():
    # H = jnp.vstack((jnp.eye(3), jnp.zeros((1, 3))))

    @jax.jit
    def quad_dynamics(x:jnp.ndarray, u:jnp.ndarray, params: EnvParams3D):
        # NOTE: u is normalized thrust and torque [-1, 1]
        # thrust = (u[0] + 1.0) / 2.0 * params.max_thrust
        # torque = u[1:4] * params.max_torque
        thrust = u[0]
        torque = u[1:4]

        r = x[:3] # position in world frame
        q = x[3:7] / jnp.linalg.norm(x[3:7]) # quaternion in world frame
        v = x[7:10] # velocity in body frame
        omega = x[10:13] # angular velocity in body frame
        Q = geom.qtoQ(q) # quaternion to rotation matrix

        # dynamics
        r_dot = Q @ v
        q_dot = 0.5 * geom.L(q) @ geom.H @ omega
        v_dot = Q.T @ jnp.asarray([0, 0, -params.g]) + 1.0 / params.m * jnp.asarray([0, 0, thrust]) - geom.hat(omega) @ v
        omega_dot = jnp.linalg.inv(params.I) @ (torque - geom.hat(omega) @ params.I @ omega)

        # return
        x_dot = jnp.concatenate([r_dot, q_dot, v_dot, omega_dot])
        return x_dot

    @jax.jit
    def quad_dynamics_rk4(x: jnp.ndarray, u: jnp.ndarray, params: EnvParams3D, dt: float) -> jnp.ndarray:    
        f = quad_dynamics
        k1 = f(x, u, params)
        k2 = f(x + k1 * dt / 2, u, params)
        k3 = f(x + k2 * dt / 2, u, params)
        k4 = f(x + k3 * dt, u, params)
        x_new = x + (k1 + 2 * k2 + 2 * k3 + k4) / 6 * dt
        return x_new.at[3:7].set(x_new[3:7] / jnp.linalg.norm(x_new[3:7]))

    @jax.jit
    def free_dynamics_3d(env_params: EnvParams3D, env_state: EnvState3D, env_action: Action3D):
        # dynamics NOTE: u is normalized thrust and torque [-1, 1]
        # thrust_normed = env_action.thrust/env_params.max_thrust * 2.0 - 1.0
        # torque_normed = env_action.torque / env_params.max_torque
        u = jnp.concatenate([jnp.array([env_action.thrust]), env_action.torque])
        x = jnp.concatenate([env_state.pos, env_state.quat, env_state.vel, env_state.omega])

        # rk4
        x_new = quad_dynamics_rk4(x, u, env_params, env_params.dt)
        pos = x_new[:3]
        quat = x_new[3:7] / jnp.linalg.norm(x_new[3:7])
        vel = x_new[7:10]
        omega = x_new[10:13]

        # step
        time = env_state.time + 1

        # trajectory
        pos_tar = env_state.pos_traj[time]
        vel_tar = env_state.vel_traj[time]

        # debug value
        last_thrust = env_action.thrust
        last_torque = env_action.torque

        env_state = env_state.replace(
            # drone
            pos=pos, vel=vel, omega=omega, quat=quat,
            # trajectory
            pos_tar=pos_tar, vel_tar=vel_tar,
            # debug value
            last_thrust=last_thrust, last_torque=last_torque,
            # step
            time=time,  
        )

        return env_state

    return free_dynamics_3d, quad_dynamics_rk4