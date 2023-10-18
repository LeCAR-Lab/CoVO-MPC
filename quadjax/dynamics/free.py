import jax
import chex
from jax import numpy as jnp
from functools import partial

from quadjax.dynamics.dataclass import EnvParams3D, EnvState3D, Action3D, EnvParams2D, EnvState2D, Action2D
from quadjax.dynamics import geom, utils

def get_free_bodyrate_dynamics_2d():
    @jax.jit
    def quad_dynamics(x:jnp.ndarray, u:jnp.ndarray, params: EnvParams2D):
        thrust = u[0]
        roll_dot = u[1]

        r = x[:2] # position in world frame
        q = x[2] # roll in world frame
        v = x[3:5] # velocity in world frame

        Q = jnp.array([[jnp.cos(q), -jnp.sin(q)], [jnp.sin(q), jnp.cos(q)]]) # quaternion to rotation matrix

        # dynamics
        v_dot = jnp.asarray([0, -params.g]) + 1.0 / params.m * (Q @ jnp.asarray([0, thrust]))

        # return
        x_dot = jnp.asarray([*v, roll_dot, *v_dot])

        return x_dot

    @jax.jit
    def dynamics_fn(x: jnp.ndarray, u: jnp.ndarray, params: EnvParams2D, dt: float) -> jnp.ndarray:    
        f = quad_dynamics
        k1 = f(x, u, params)
        k2 = f(x + k1 * dt / 2, u, params)
        k3 = f(x + k2 * dt / 2, u, params)
        k4 = f(x + k3 * dt, u, params)
        x_new = x + (k1 + 2 * k2 + 2 * k3 + k4) / 6 * dt
        roll = x_new[2]
        roll_normed = utils.angle_normalize(roll)
        return x_new.at[2].set(roll_normed)

    @jax.jit
    def step_fn(env_params: EnvParams2D, env_state: EnvState2D, env_action: Action2D):
        u = jnp.asarray([env_action.thrust, env_action.roll_dot])
        x = jnp.asarray([*env_state.pos, env_state.roll, *env_state.vel])

        # rk4
        x_new = dynamics_fn(x, u, env_params, sim_dt)
        pos = x_new[:2]
        roll = x_new[2]
        vel = x_new[3:5]
        roll_rate = x_new[5]

        # step
        time = env_state.time + 1

        # trajectory
        pos_tar = env_state.pos_traj[time]
        vel_tar = env_state.vel_traj[time]

        # debug value
        last_thrust = env_action.thrust
        last_roll_dot = env_action.roll_dot

        env_state = env_state.replace(
            # drone
            pos=pos, vel=vel, roll=roll, roll_dot=roll_rate,
            # trajectory
            pos_tar=pos_tar, vel_tar=vel_tar,
            # debug value
            last_thrust=last_thrust, last_roll_dot=last_roll_dot,
            # step
            time=time,  
        )

        return env_state

    return step_fn, dynamics_fn

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
        x_new = quad_dynamics_rk4(x, u, env_params, sim_dt)
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

def get_free_dynamics_3d_bodyrate(disturb_type:str='periodic'):
    # H = jnp.vstack((jnp.eye(3), jnp.zeros((1, 3))))

    @jax.jit
    def period_disturb(disturb_key: chex.PRNGKey, params: EnvParams3D, state: EnvState3D):
        time = state.time
        disturb = jnp.where(time % params.disturb_period == 0, jax.random.uniform(disturb_key, shape=(3,), minval=-params.disturb_scale, maxval=params.disturb_scale)
                            , state.f_disturb)
        return disturb
    
    @jax.jit
    def sin_disturb(disturb_key: chex.PRNGKey, params: EnvParams3D, state: EnvState3D):
        time = state.time
        # random_phase = jnp.where(time % params.max_steps_in_episode == 0, jax.random.uniform(disturb_key, shape=(3,), minval=0, maxval=2*jnp.pi)
                            # , state.f_disturb)
        scale = params.disturb_params[:3] * params.disturb_scale
        period = params.disturb_params[:3] * (params.disturb_period/3) + params.disturb_period
        phase = params.disturb_params[3:6] * 2 * jnp.pi
        disturb = scale * jnp.sin(2*jnp.pi/period*time+phase)
        return disturb
    
    @jax.jit
    def drag_disturb(disturb_key: chex.PRNGKey, params: EnvParams3D, state: EnvState3D):
        # time = state.time
        rel_vel = state.vel - params.disturb_params[:3] * 0.5
        disturb = -jnp.abs(params.disturb_scale) * rel_vel * jnp.abs(rel_vel) / (1.5**2)
        return disturb
    
    if disturb_type == 'periodic':
        disturb_func = period_disturb
    elif disturb_type == 'sin':
        disturb_func = sin_disturb
    elif disturb_type == 'drag':
        disturb_func = drag_disturb

    @jax.jit
    def quad_dynamics_bodyrate(x:jnp.ndarray, u:jnp.ndarray, params: EnvParams3D, dt: float):
        # x: state [r, q, v, omega]
        # NOTE: u is normalized thrust and torque [-1, 1]
        # thrust = (u[0] + 1.0) / 2.0 * params.max_thrust
        # torque = u[1:4] * params.max_torque
        u = u * params.action_scale

        thrust = u[0]
        omega_tar = u[1:4]

        r = x[:3] # position in world frame
        q = x[3:7] / jnp.linalg.norm(x[3:7]) # quaternion in world frame
        v = x[7:10] # velocity in world frame
        omega = x[10:13] # angular velocity in body frame
        f_disturb = x[13:16] # disturbance in world frame
        Q = geom.qtoQ(q) # quaternion to rotation matrix

        # dynamics
        r_dot = v
        q_dot = 0.5 * geom.L(q) @ geom.H @ omega
        v_dot = jnp.asarray([0, 0, -params.g]) + 1.0 / params.m * (Q @ jnp.asarray([0, 0, thrust]) + f_disturb)

        # integrate
        r_new = r + r_dot * dt
        q_new = q + q_dot * dt
        v_new = v + v_dot * dt
        omega_new = params.alpha_bodyrate * (omega) + (1-params.alpha_bodyrate) * omega_tar
        f_disturb_new = f_disturb

        # return
        x_new = jnp.concatenate([r_new, q_new, v_new, omega_new, f_disturb_new])
        return x_new

    @jax.jit
    def free_dynamics_3d_bodyrate(env_params: EnvParams3D, env_state: EnvState3D, env_action: Action3D, key:chex.PRNGKey, sim_dt: float):
        # dynamics NOTE: u is normalized thrust and torque [-1, 1]
        # thrust_normed = env_action.thrust/env_params.max_thrust * 2.0 - 1.0
        # torque_normed = env_action.torque / env_params.max_torque
        # NOTE hack here, just convert torque to omega
        omega_tar = env_action.torque / env_params.max_torque * env_params.max_omega
        u = jnp.concatenate([jnp.array([env_action.thrust]), omega_tar])
        x = jnp.concatenate([env_state.pos, env_state.quat, env_state.vel, env_state.omega, env_state.f_disturb])

        x_new = quad_dynamics_bodyrate(x, u, env_params, sim_dt)
        pos = x_new[:3]
        quat = x_new[3:7] / jnp.linalg.norm(x_new[3:7])
        vel = x_new[7:10]
        omega = x_new[10:13]

        # update disturbance
        disturb_key, key = jax.random.split(key)

        # generate period disturbance
        f_disturb = disturb_func(disturb_key, env_params, env_state)

        # step
        time = env_state.time + 1
        
        # generate 3d sinusoidal disturbance with period and scale and random phase
        # random_phase = jax.random.uniform(disturb_key, shape=(3,), minval=0, maxval=2*jnp.pi)
        # f_disturb = env_params.disturb_scale * jnp.sin(2*jnp.pi/env_params.disturb_period*time + random_phase)
        
        # generate disturbance w.r.t. speed
        # f_disturb = env_params.disturb_scale * env_state.vel * jnp.linalg.norm(env_state.vel) / (1.5**2)

        # trajectory
        pos_tar = env_state.pos_traj[time]
        vel_tar = env_state.vel_traj[time]
        acc_tar = env_state.acc_traj[time]

        # debug value
        last_thrust = env_action.thrust
        last_torque = env_action.torque

        # adaptation trajectory history information
        vel_hist = jnp.concatenate([env_state.vel_hist[1:], jnp.expand_dims(env_state.vel, axis=0)])
        omega_hist = jnp.concatenate([env_state.omega_hist[1:], jnp.expand_dims(env_state.omega, axis=0)])
        action = jnp.concatenate([jnp.asarray([env_action.thrust])/env_params.max_thrust*2.0-1.0, env_action.torque/env_params.max_torque])
        action_hist = jnp.concatenate([env_state.action_hist[1:], jnp.expand_dims(action, axis=0)])

        env_state = env_state.replace(
            # drone
            pos=pos, vel=vel, omega=omega, quat=quat,
            # trajectory
            pos_tar=pos_tar, vel_tar=vel_tar, acc_tar=acc_tar, 
            # debug value
            last_thrust=last_thrust, last_torque=last_torque,
            # step
            time=time,  
            # disturbance
            f_disturb=f_disturb,
            # adaptation trajectory history information
            vel_hist=vel_hist, omega_hist=omega_hist, action_hist=action_hist,
        )

        return env_state

    return free_dynamics_3d_bodyrate, quad_dynamics_bodyrate

def get_free_dynamics_3d_disturbance(d_func):
    # H = jnp.vstack((jnp.eye(3), jnp.zeros((1, 3))))

    @partial(jax.jit, static_argnames=['d_func'])
    def quad_dynamics(x:jnp.ndarray, u:jnp.ndarray, params: EnvParams3D, d_func):
        # NOTE: u is normalized thrust and torque [-1, 1]
        # thrust = (u[0] + 1.0) / 2.0 * params.max_thrust
        # torque = u[1:4] * params.max_torque
        thrust = u[0]
        torque = u[1:4]
        
        d = d_func(x, u, params)
        d_f = d[0:3]
        d_t = d[3:6]

        r = x[:3] # position in world frame
        q = x[3:7] / jnp.linalg.norm(x[3:7]) # quaternion in world frame
        v = x[7:10] # velocity in body frame
        omega = x[10:13] # angular velocity in body frame
        Q = geom.qtoQ(q) # quaternion to rotation matrix

        # dynamics
        r_dot = Q @ v
        q_dot = 0.5 * geom.L(q) @ geom.H @ omega
        v_dot = Q.T @ jnp.asarray([0, 0, -params.g]) + 1.0 / params.m * jnp.asarray([0, 0, thrust]) - geom.hat(omega) @ v + d_f
        omega_dot = jnp.linalg.inv(params.I) @ (torque - geom.hat(omega) @ params.I @ omega) + d_t

        # return
        x_dot = jnp.concatenate([r_dot, q_dot, v_dot, omega_dot])
        return x_dot

    @partial(jax.jit, static_argnames=['d_func'])
    def quad_dynamics_rk4(x: jnp.ndarray, u: jnp.ndarray, params: EnvParams3D, dt: float, d_func) -> jnp.ndarray:    
        f = lambda x, u, p: quad_dynamics(x, u, p, d_func)
        k1 = f(x, u, params)
        k2 = f(x + k1 * dt / 2, u, params)
        k3 = f(x + k2 * dt / 2, u, params)
        k4 = f(x + k3 * dt, u, params)
        x_new = x + (k1 + 2 * k2 + 2 * k3 + k4) / 6 * dt
        return x_new.at[3:7].set(x_new[3:7] / jnp.linalg.norm(x_new[3:7]))

    @partial(jax.jit, static_argnames=['d_func'])
    def free_dynamics_3d(env_params: EnvParams3D, env_state: EnvState3D, env_action: Action3D, d_func):
        # dynamics NOTE: u is normalized thrust and torque [-1, 1]
        # thrust_normed = env_action.thrust/env_params.max_thrust * 2.0 - 1.0
        # torque_normed = env_action.torque / env_params.max_torque
        u = jnp.concatenate([jnp.array([env_action.thrust]), env_action.torque])
        x = jnp.concatenate([env_state.pos, env_state.quat, env_state.vel, env_state.omega])

        # rk4
        x_new = quad_dynamics_rk4(x, u, env_params, sim_dt, d_func)
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

    return partial(free_dynamics_3d, d_func=d_func), partial(quad_dynamics_rk4, d_func=d_func)