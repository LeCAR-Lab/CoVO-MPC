import jax
import chex
from jax import numpy as jnp

from quadjax.dynamics.dataclass import EnvParams3D, EnvState3D, Action3D
from quadjax.dynamics import geom, utils


def get_free_dynamics_3d():

    @jax.jit
    def quad_dynamics(x: jnp.ndarray, u: jnp.ndarray, params: EnvParams3D):
        # NOTE: u is normalized thrust and torque [-1, 1]
        # thrust = (u[0] + 1.0) / 2.0 * params.max_thrust
        # torque = u[1:4] * params.max_torque
        thrust = u[0]
        torque = u[1:4]

        r = x[:3]  # position in world frame
        q = x[3:7] / jnp.linalg.norm(x[3:7])  # quaternion in world frame
        v = x[7:10]  # velocity in body frame
        omega = x[10:13]  # angular velocity in body frame
        Q = geom.qtoQ(q)  # quaternion to rotation matrix

        # dynamics
        r_dot = Q @ v
        q_dot = 0.5 * geom.L(q) @ geom.H @ omega
        v_dot = (
            Q.T @ jnp.asarray([0, 0, -params.g])
            + 1.0 / params.m * jnp.asarray([0, 0, thrust])
            - geom.hat(omega) @ v
        )
        omega_dot = jnp.linalg.inv(params.I) @ (
            torque - geom.hat(omega) @ params.I @ omega
        )

        # return
        x_dot = jnp.concatenate([r_dot, q_dot, v_dot, omega_dot])
        return x_dot

    @jax.jit
    def quad_dynamics_rk4(
        x: jnp.ndarray, u: jnp.ndarray, params: EnvParams3D, dt: float
    ) -> jnp.ndarray:
        f = quad_dynamics
        k1 = f(x, u, params)
        k2 = f(x + k1 * dt / 2, u, params)
        k3 = f(x + k2 * dt / 2, u, params)
        k4 = f(x + k3 * dt, u, params)
        x_new = x + (k1 + 2 * k2 + 2 * k3 + k4) / 6 * dt
        return x_new.at[3:7].set(x_new[3:7] / jnp.linalg.norm(x_new[3:7]))

    @jax.jit
    def free_dynamics_3d(
        env_params: EnvParams3D,
        env_state: EnvState3D,
        env_action: Action3D,
        key: chex.PRNGKey,
        sim_dt: float,
    ):
        # dynamics NOTE: u is normalized thrust and torque [-1, 1]
        # thrust_normed = env_action.thrust/env_params.max_thrust * 2.0 - 1.0
        # torque_normed = env_action.torque / env_params.max_torque
        u = jnp.concatenate([jnp.array([env_action.thrust]), env_action.torque])
        x = jnp.concatenate(
            [env_state.pos, env_state.quat, env_state.vel, env_state.omega]
        )

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
            pos=pos,
            vel=vel,
            omega=omega,
            quat=quat,
            # trajectory
            pos_tar=pos_tar,
            vel_tar=vel_tar,
            # debug value
            last_thrust=last_thrust,
            last_torque=last_torque,
            # step
            time=time,
        )

        return env_state

    return free_dynamics_3d, quad_dynamics_rk4


def get_quadrotor_1st_order_dyn(disturb_type: str = "periodic"):
    @jax.jit
    def period_disturb(
        disturb_key: chex.PRNGKey, params: EnvParams3D, state: EnvState3D
    ):
        time = state.time
        disturb = jnp.where(
            time % params.disturb_period == 0,
            jax.random.uniform(
                disturb_key,
                shape=(3,),
                minval=-params.disturb_scale,
                maxval=params.disturb_scale,
            ),
            state.f_disturb,
        )
        return disturb

    @jax.jit
    def sin_disturb(disturb_key: chex.PRNGKey, params: EnvParams3D, state: EnvState3D):
        time = state.time
        # random_phase = jnp.where(time % params.max_steps_in_episode == 0, jax.random.uniform(disturb_key, shape=(3,), minval=0, maxval=2*jnp.pi)
        # , state.f_disturb)
        scale = params.disturb_params[:3] * params.disturb_scale
        period = (
            params.disturb_params[:3] * (params.disturb_period / 3)
            + params.disturb_period
        )
        phase = params.disturb_params[3:6] * 2 * jnp.pi
        disturb = scale * jnp.sin(2 * jnp.pi / period * time + phase)
        return disturb

    @jax.jit
    def drag_disturb(disturb_key: chex.PRNGKey, params: EnvParams3D, state: EnvState3D):
        # time = state.time
        rel_vel = state.vel - params.disturb_params[:3] * 0.5
        disturb = (
            -jnp.abs(params.disturb_scale) * rel_vel * jnp.abs(rel_vel) / (1.5**2)
        )
        return disturb

    @jax.jit
    def mixed_disturb(
        disturb_key: chex.PRNGKey, params: EnvParams3D, state: EnvState3D
    ):
        d_drag = drag_disturb(disturb_key, params, state)
        d_sin = sin_disturb(disturb_key, params, state)
        d_period = period_disturb(disturb_key, params, state)
        return (d_drag + d_sin + d_period) / 3

    if disturb_type == "periodic":
        disturb_func = period_disturb
    elif disturb_type == "sin":
        disturb_func = sin_disturb
    elif disturb_type == "drag":
        disturb_func = drag_disturb
    elif disturb_type == "mixed":
        disturb_func = mixed_disturb
    elif disturb_type == "gaussian":
        disturb_func = (
            lambda disturb_key, params, state: params.dyn_noise_scale
            * jax.random.normal(disturb_key, shape=(3,))
        )
    elif disturb_type == "none":
        disturb_func = lambda disturb_key, params, state: jnp.zeros(3)

    @jax.jit
    def quad_dynamics_bodyrate(
        x: jnp.ndarray,
        u: jnp.ndarray,
        params: EnvParams3D,
        dt: float,
        key: chex.PRNGKey,
    ):
        # x: state [r, q, v, omega]
        # NOTE: u is normalized thrust and torque [-1, 1]
        # thrust = (u[0] + 1.0) / 2.0 * params.max_thrust
        # torque = u[1:4] * params.max_torque
        u = u * params.action_scale

        thrust = u[0]
        omega_tar = u[1:4]

        r = x[:3]  # position in world frame
        q = x[3:7] / jnp.linalg.norm(x[3:7])  # quaternion in world frame
        v = x[7:10]  # velocity in world frame
        omega = x[10:13]  # angular velocity in body frame
        f_disturb = x[13:16]  # disturbance in world frame
        Q = geom.qtoQ(q)  # quaternion to rotation matrix

        # dynamics
        r_dot = v
        q_dot = 0.5 * geom.L(q) @ geom.H @ omega
        v_dot = jnp.asarray([0, 0, -params.g]) + 1.0 / params.m * (
            Q @ jnp.asarray([0, 0, thrust]) + f_disturb
        )

        # integrate
        r_new = r + r_dot * dt
        q_new = q + q_dot * dt
        v_new = v + v_dot * dt
        omega_new = (
            params.alpha_bodyrate * (omega) + (1 - params.alpha_bodyrate) * omega_tar
        )
        f_disturb_new = f_disturb

        # return
        x_new = jnp.concatenate([r_new, q_new, v_new, omega_new, f_disturb_new])
        return x_new

    @jax.jit
    def free_dynamics_3d_bodyrate(
        env_params: EnvParams3D,
        env_state: EnvState3D,
        env_action: Action3D,
        key: chex.PRNGKey,
        sim_dt: float,
    ):
        # NOTE hack here, just convert torque to omega
        omega_tar = env_action.torque / env_params.max_torque * env_params.max_omega
        thrust = env_action.thrust

        u = jnp.concatenate([jnp.array([thrust]), omega_tar])
        x = jnp.concatenate(
            [
                env_state.pos,
                env_state.quat,
                env_state.vel,
                env_state.omega,
                env_state.f_disturb,
            ]
        )

        key, key_dyn = jax.random.split(key)
        x_new = quad_dynamics_bodyrate(x, u, env_params, sim_dt, key_dyn)
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

        # trajectory
        pos_tar = env_state.pos_traj[time]
        vel_tar = env_state.vel_traj[time]
        acc_tar = env_state.acc_traj[time]

        # debug value
        last_thrust = thrust
        last_torque = env_action.torque

        # adaptation trajectory history information
        vel_hist = jnp.concatenate(
            [env_state.vel_hist[1:], jnp.expand_dims(env_state.vel, axis=0)]
        )
        omega_hist = jnp.concatenate(
            [env_state.omega_hist[1:], jnp.expand_dims(env_state.omega, axis=0)]
        )
        action = jnp.concatenate(
            [
                jnp.asarray([env_action.thrust]) / env_params.max_thrust * 2.0 - 1.0,
                env_action.torque / env_params.max_torque,
            ]
        )
        action_hist = jnp.concatenate(
            [env_state.action_hist[1:], jnp.expand_dims(action, axis=0)]
        )

        env_state = env_state.replace(
            # drone
            pos=pos,
            vel=vel,
            omega=omega,
            quat=quat,
            # trajectory
            pos_tar=pos_tar,
            vel_tar=vel_tar,
            acc_tar=acc_tar,
            omega_tar=omega_tar,
            # debug value
            last_thrust=last_thrust,
            last_torque=last_torque,
            # step
            time=time,
            # disturbance
            f_disturb=f_disturb,
            # adaptation trajectory history information
            vel_hist=vel_hist,
            omega_hist=omega_hist,
            action_hist=action_hist,
        )

        return env_state

    return free_dynamics_3d_bodyrate, quad_dynamics_bodyrate
