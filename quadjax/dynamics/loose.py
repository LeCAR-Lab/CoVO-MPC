from jax import numpy as jnp

from adaptive_control_gym.envs.jax_env.dynamics.utils import angle_normalize, EnvParams, EnvState, Action, EnvParams3D, EnvState3D, Action3D
from adaptive_control_gym.envs.jax_env.dynamics import geom


def get_loose_dynamics():

    # dynamics (params, states) -> states_dot
    def loose_dynamics(env_params: EnvParams, env_state: EnvState, env_action: Action):
        params = [env_params.m, env_params.I, env_params.g, env_params.l,
                  env_params.mo, env_params.delta_yh, env_params.delta_zh]
        states = [env_state.y, env_state.z, env_state.theta, env_state.phi,
                  env_state.y_dot, env_state.z_dot, env_state.theta_dot, env_state.phi_dot]
        action = [env_action.thrust, env_action.tau]

        y_ddot = -env_action.thrust * jnp.sin(env_state.theta) / env_params.m
        z_ddot = env_action.thrust * \
            jnp.cos(env_state.theta) / env_params.m - env_params.g
        theta_ddot = env_action.tau / env_params.I

        new_y_dot = env_state.y_dot + env_params.dt * y_ddot
        new_z_dot = env_state.z_dot + env_params.dt * z_ddot
        new_theta_dot = env_state.theta_dot + env_params.dt * theta_ddot
        new_y = env_state.y + env_params.dt * new_y_dot
        new_z = env_state.z + env_params.dt * new_z_dot
        new_theta = angle_normalize(
            env_state.theta + env_params.dt * new_theta_dot)

        # states = [new_y, new_z, new_theta, env_state.phi, new_y_dot, new_z_dot, new_theta_dot, env_state.phi_dot]

        delta_y_hook = env_params.delta_yh * \
            jnp.cos(new_theta) - env_params.delta_zh * jnp.sin(new_theta)
        delta_z_hook = env_params.delta_yh * \
            jnp.sin(new_theta) + env_params.delta_zh * jnp.cos(new_theta)
        y_hook = new_y + delta_y_hook
        z_hook = new_z + delta_z_hook
        y_hook_dot = new_y_dot - new_theta_dot * delta_z_hook
        z_hook_dot = new_z_dot + new_theta_dot * delta_y_hook

        new_y_obj_dot = env_state.y_obj_dot
        new_z_obj_dot = env_state.z_obj_dot - env_params.g * env_params.dt
        new_y_obj = env_state.y_obj + env_params.dt * new_y_obj_dot
        new_z_obj = env_state.z_obj + env_params.dt * new_z_obj_dot

        phi_th = -jnp.arctan2(y_hook - new_y_obj, z_hook - new_z_obj)
        new_phi = angle_normalize(phi_th - new_theta)

        y_obj2hook_dot = new_y_obj_dot - y_hook_dot
        z_obj2hook_dot = new_z_obj_dot - z_hook_dot
        phi_th_dot = y_obj2hook_dot * \
            jnp.cos(phi_th) + z_obj2hook_dot * jnp.sin(phi_th)
        new_phi_dot = phi_th_dot - new_theta_dot

        new_l_rope = jnp.sqrt((y_hook - new_y_obj) **
                              2 + (z_hook - new_z_obj) ** 2)

        env_state = env_state.replace(
            y=new_y,
            z=new_z,
            theta=new_theta,
            phi=new_phi,
            y_dot=new_y_dot,
            z_dot=new_z_dot,
            theta_dot=new_theta_dot,
            phi_dot=new_phi_dot,
            y_hook=y_hook,
            z_hook=z_hook,
            y_hook_dot=y_hook_dot,
            z_hook_dot=z_hook_dot,
            y_obj=new_y_obj,
            z_obj=new_z_obj,
            y_obj_dot=new_y_obj_dot,
            z_obj_dot=new_z_obj_dot,
            l_rope=new_l_rope,
            f_rope=0.0,
            f_rope_y=0.0,
            f_rope_z=0.0,
            last_thrust=env_action.thrust,
            last_tau=env_action.tau,
            time=env_state.time + 1,
            y_tar=env_state.y_traj[env_state.time],
            z_tar=env_state.z_traj[env_state.time],
            y_dot_tar=env_state.y_dot_traj[env_state.time],
            z_dot_tar=env_state.z_dot_traj[env_state.time],
        )

        return env_state

    return loose_dynamics


def get_loose_dynamics_3d():

    # dynamics (params, states) -> states_dot
    def loose_dynamics_3d(env_params: EnvParams3D, env_state: EnvState3D, env_action: Action3D):
        # dynamics
        thrust_local = jnp.array([0.0, 0.0, env_action.thrust])
        thrust_world = geom.rotate_with_quat(thrust_local, env_state.quat)
        torque_world = geom.rotate_with_quat(env_action.torque, env_state.quat)
        acc = thrust_world / env_params.m - jnp.array([0.0, 0.0, env_params.g])
        alpha = torque_world / env_params.I
        acc_obj = jnp.array([0.0, 0.0, -env_params.g])

        # meta variables
        # quadrotor
        vel = env_state.vel + env_params.dt * acc
        pos = env_state.pos + env_params.dt * vel
        omega = env_state.omega + env_params.dt * alpha
        quat = geom.integrate_quat(env_state.quat, omega, env_params.dt)
        # object
        vel_obj = env_state.vel_obj + env_params.dt * acc_obj
        pos_obj = env_state.pos_obj + env_params.dt * vel_obj
        # step
        time = env_state.time + 1

        # other variables
        # hook
        hook_offset_world = geom.rotate_with_quat(env_params.hook_offset, quat)
        pos_hook = pos + hook_offset_world
        vel_hook = vel + jnp.cross(omega, hook_offset_world)
        # rope
        obj2quad = pos_obj - pos
        l_rope = jnp.linalg.norm(obj2quad)
        zeta = obj2quad / l_rope
        zeta_dot = (vel_obj - vel) / l_rope
        f_rope = jnp.zeros(3)
        f_rope_norm = 0.0
        # trajectory
        pos_tar = env_state.pos_traj[time]
        vel_tar = env_state.vel_traj[time]
        # debug value
        last_thrust = env_action.thrust
        last_torque = env_action.torque

        env_state = env_state.replace(
            # drone
            pos=pos, vel=vel, omega=omega, quat=quat,
            # object
            pos_obj=pos_obj, vel_obj=vel_obj,
            # hook
            pos_hook=pos_hook, vel_hook=vel_hook,
            # rope
            l_rope=l_rope, zeta=zeta, zeta_dot=zeta_dot,
            f_rope=f_rope, f_rope_norm=f_rope_norm,
            # trajectory
            pos_tar=pos_tar, vel_tar=vel_tar,
            # debug value
            last_thrust=last_thrust, last_torque=last_torque,
            # step
            time=time,  
        )

        return env_state

    return loose_dynamics_3d
