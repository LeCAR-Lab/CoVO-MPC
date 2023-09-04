from jax import numpy as jnp

from quadjax.dynamics.dataclass import EnvParams3D, EnvState3D, Action3D
from quadjax.dynamics import geom, utils

def get_free_dynamics_3d():
    H = jnp.vstack((jnp.zeros((1, 3)), jnp.eye(3)))

    def quad_dynamics(x:jnp.array, u:jnp.array, params: EnvParams3D):
        thrust = u[0]
        torque = u[1:4]

        r = x[:3] # position in world frame
        q = x[3:7] / jnp.linalg.norm(x[3:7]) # quaternion in world frame
        v = x[7:10] # velocity in body frame
        omega = x[10:13] # angular velocity in body frame
        Q = geom.qtoQ(q) # quaternion to rotation matrix

        # dynamics
        r_dot = Q @ v
        q_dot = 0.5 * geom.L(q) @ H @ omega
        v_dot = Q.T @ jnp.array([0, 0, -params.g]) + thrust / params.m * jnp.array([0, 0, 1]) - jnp.cross(omega, v)
        omega_dot = jnp.linalg.inv(params.I) @ (torque - jnp.cross(omega, params.I @ omega))

        # return
        x_dot = jnp.concatenate([r_dot, q_dot, v_dot, omega_dot])
        return x_dot

    quad_dynamics_rk4 = lambda x, u, params, dt: utils.rk4(quad_dynamics, x, u, params, dt)

    # dynamics (params, states) -> states_dot
    def free_dynamics_3d(env_params: EnvParams3D, env_state: EnvState3D, env_action: Action3D):
        # dynamics
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