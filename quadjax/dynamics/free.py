import jax
import chex
from jax import numpy as jnp
from functools import partial

from quadjax.dynamics.dataclass import (
    EnvParams3D,
    EnvState3D,
    Action3D,
    EnvParams2D,
    EnvState2D,
    Action2D,
)
from quadjax.dynamics import geom, utils

# force to rpm
# def force_to_rpm(force):
#     a, b, c = 2.55077341e-08, -4.92422570e-05, -1.51910248e-01
#     force_in_grams = jnp.clip(force * 1000.0 / 9.81, 0.0, 1000)
#     rpm = (-b + jnp.sqrt(b**2 - 4 * a * (c - force_in_grams))) / (2 * a)
#     return rpm
# def rpm_to_force(rpm):
#     a, b, c = 2.55077341e-08, -4.92422570e-05, -1.51910248e-01
#     force_in_grams = a * rpm**2 + b * rpm + c
#     force = force_in_grams * 9.81 / 1000.0
#     return force
# # force to pwm
# def rpm_to_pwm(rpm):
#     a, b = 3.26535711e-01, 3.37495115e03
#     pwm = 1 / a * (rpm - b)
#     return pwm
# def pwm_to_rpm(pwm):
#     a, b = 3.26535711e-01, 3.37495115e03
#     rpm = a * pwm + b
#     return rpm


def get_free_bodyrate_dynamics_2d():
    @jax.jit
    def dynamics_fn(
        x: jnp.ndarray, u: jnp.ndarray, params: EnvParams2D, dt: float, key: chex.PRNGKey
    ):
        thrust = u[0]
        omega_tar = u[1]

        r = x[:2]  # position in world frame
        q = x[2]  # roll in world frame
        v = x[3:5]  # velocity in world frame
        omega = x[5]  # roll rate in world frame

        # make the system unstable
        # omega = omega + 10.0 * jnp.abs(jnp.sin(q))
        omega_new = params.alpha_bodyrate * omega_tar + (1 - params.alpha_bodyrate) * omega

        Q = jnp.array(
            [[jnp.cos(q), -jnp.sin(q)], [jnp.sin(q), jnp.cos(q)]]
        )  # quaternion to rotation matrix

        # dynamics
        v_dot = jnp.asarray([0, -params.g]) + 1.0 / params.m * (
            Q @ jnp.asarray([0, thrust])
        )

        # generate noise
        vel_noise_key, key = jax.random.split(key)
        v = v + params.dyn_noise_scale * 1.0 * jax.random.normal(vel_noise_key, shape=(2,))
        omega_noise_key, key = jax.random.split(key)
        omega_new = omega_new + params.dyn_noise_scale * 8.0 * jax.random.normal(omega_noise_key, shape=(1,))[0]
        v_dot_noise_key, key = jax.random.split(key)
        v_dot = v_dot + params.dyn_noise_scale * 2.0 * jax.random.normal(v_dot_noise_key, shape=(2,))

        # integrate
        r_new = r + v * dt
        q_new = utils.angle_normalize(q + omega_new * dt)
        v_new = v + v_dot * dt

        return jnp.concatenate([r_new, jnp.asarray([q_new]), v_new, jnp.asarray([omega_new])])

    # @jax.jit
    # def dynamics_fn(
    #     x: jnp.ndarray,
    #     u: jnp.ndarray,
    #     params: EnvParams2D,
    #     dt: float,
    #     key: chex.PRNGKey,
    # ) -> jnp.ndarray:
    #     f = quad_dynamics
    #     key1, key2, key3, key4 = jax.random.split(key, 4)
    #     k1 = f(x, u, params, key1, dt/6)
    #     k2 = f(x + k1 * dt / 2, u, params, key2, dt/3)
    #     k3 = f(x + k2 * dt / 2, u, params, key3, dt/3)
    #     k4 = f(x + k3 * dt, u, params, key4)
    #     x_new = x + (k1 + 2 * k2 + 2 * k3 + k4) / 6 * dt
    #     roll = x_new[2]
    #     roll_normed = utils.angle_normalize(roll)
    #     return x_new.at[2].set(roll_normed)

    @jax.jit
    def step_fn(
        env_params: EnvParams2D,
        env_state: EnvState2D,
        env_action: Action2D,
        key: chex.PRNGKey,
    ):
        u = jnp.asarray([env_action.thrust, env_action.omega])
        x = jnp.asarray([*env_state.pos, env_state.roll, *env_state.vel, env_state.omega])

        # rk4
        key, key_dyn = jax.random.split(key)
        x_new = dynamics_fn(x, u, env_params, env_params.dt, key_dyn)
        pos = x_new[:2]
        roll = x_new[2]
        vel = x_new[3:5]
        omega = x_new[5]

        # step
        time = env_state.time + 1

        # trajectory
        pos_tar = env_state.pos_traj[time]
        vel_tar = env_state.vel_traj[time]

        # debug value
        last_thrust = env_action.thrust
        last_omega = env_action.omega

        env_state = env_state.replace(
            # drone
            pos=pos,
            vel=vel,
            roll=roll,
            omega=omega,
            # trajectory
            pos_tar=pos_tar,
            vel_tar=vel_tar,
            # debug value
            last_thrust=last_thrust,
            last_omega=last_omega,
            # step
            time=time,
        )

        return env_state

    return step_fn, dynamics_fn


def get_free_dynamics_2d():
    @jax.jit
    def quad_dynamics(
        x: jnp.ndarray, u: jnp.ndarray, params: EnvParams2D, key: chex.PRNGKey
    ):
        # calculate the motor force of each motor
        arm = 0.707106781 * params.arm_length
        B0 = jnp.array([[1, 1], [-arm, arm]])
        F = jnp.linalg.inv(B0) @ u
        F = jnp.clip(F, 0, 1.0)
        # rpm = force_to_rpm(F)
        # pwm = rpm_to_pwm(rpm)
        # move all pwm down to make sure the max pwm is 2**16
        # jax.debug.print('scale {x}', x=pwm / 2 ** 16)
        reduction = jnp.clip(jnp.max(F) - params.max_motor_force, 0.0, 10.0)
        F = jnp.clip(F - reduction, 0.0, 10.0)
        # calculate the thrust and torque
        # rpm = pwm_to_rpm(pwm)
        # F = rpm_to_force(rpm)

        u_cap = B0 @ F
        # u_cap=u
        thrust = u_cap[0]
        torque = u_cap[1]

        # jax.debug.print('F scale = {a}, \n T scale = {b}', a=u_cap[0]/u[0], b=u_cap[1]/u[1])

        r = x[:2]  # position in world frame
        q = x[2]  # roll in world frame
        v = x[3:5]  # velocity in world frame
        omega = x[5]  # roll rate in world frame

        # make the system unstable
        # omega = omega + 10.0 * jnp.abs(jnp.sin(q))

        Q = jnp.array(
            [[jnp.cos(q), -jnp.sin(q)], [jnp.sin(q), jnp.cos(q)]]
        )  # quaternion to rotation matrix

        # dynamics
        v_dot = jnp.asarray([0, -params.g]) + 1.0 / params.m * (
            Q @ jnp.asarray([0, thrust])
        )
        omega_dot = 1.0 / params.I * torque

        # return
        x_dot = jnp.asarray([*v, omega, *v_dot, omega_dot])

        # generate noise
        noise_key, key = jax.random.split(key)
        x_dot_noise = (
            params.dyn_noise_scale
            * jnp.array([1.0, 1.0, 5.0, 10.0, 10.0, 50.0])
            * jax.random.normal(noise_key, shape=(6,))
        )

        x_dot = x_dot + x_dot_noise

        return x_dot

    @jax.jit
    def dynamics_fn(
        x: jnp.ndarray,
        u: jnp.ndarray,
        params: EnvParams2D,
        dt: float,
        key: chex.PRNGKey,
    ) -> jnp.ndarray:
        f = quad_dynamics
        key1, key2, key3, key4 = jax.random.split(key, 4)
        k1 = f(x, u, params, key1)
        k2 = f(x + k1 * dt / 2, u, params, key2)
        k3 = f(x + k2 * dt / 2, u, params, key3)
        k4 = f(x + k3 * dt, u, params, key4)
        x_new = x + (k1 + 2 * k2 + 2 * k3 + k4) / 6 * dt
        roll = x_new[2]
        roll_normed = utils.angle_normalize(roll)
        return x_new.at[2].set(roll_normed)

    @jax.jit
    def step_fn(
        env_params: EnvParams2D,
        env_state: EnvState2D,
        env_action: Action2D,
        key: chex.PRNGKey,
    ):
        torque = env_action.omega / env_params.max_omega * env_params.max_torque
        u = jnp.asarray([env_action.thrust, torque])
        x = jnp.asarray(
            [*env_state.pos, env_state.roll, *env_state.vel, env_state.omega]
        )

        # rk4
        key, key_dyn = jax.random.split(key)
        x_new = dynamics_fn(x, u, env_params, env_params.dt, key_dyn)
        pos = x_new[:2]
        roll = x_new[2]
        vel = x_new[3:5]
        omega = x_new[5]

        # step
        time = env_state.time + 1

        # trajectory
        pos_tar = env_state.pos_traj[time]
        vel_tar = env_state.vel_traj[time]

        # debug value
        last_thrust = env_action.thrust
        last_omega = env_action.omega

        env_state = env_state.replace(
            # drone
            pos=pos,
            vel=vel,
            roll=roll,
            omega=omega,
            # trajectory
            pos_tar=pos_tar,
            vel_tar=vel_tar,
            # debug value
            last_thrust=last_thrust,
            last_omega=last_omega,
            # step
            time=time,
        )

        return env_state

    return step_fn, dynamics_fn


def get_free_dynamics_3d():
    # H = jnp.vstack((jnp.eye(3), jnp.zeros((1, 3))))

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


def get_free_dynamics_3d_bodyrate(disturb_type: str = "periodic"):
    # H = jnp.vstack((jnp.eye(3), jnp.zeros((1, 3))))

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

        # generate noise
        noise_keys = jax.random.split(key, 4)
        r_dot_noise = (
            params.dyn_noise_scale * 1.0 * jax.random.normal(noise_keys[0], shape=(3,))
        )
        r_dot = r_dot + r_dot_noise
        q_dot_noise = (
            params.dyn_noise_scale * 2.5 * jax.random.normal(noise_keys[1], shape=(4,))
        )
        q_dot = q_dot + q_dot_noise
        v_dot_noise = (
            params.dyn_noise_scale * 10.0 * jax.random.normal(noise_keys[2], shape=(3,))
        )
        v_dot = v_dot + v_dot_noise
        omega_tar_noise = (
            params.dyn_noise_scale * 5.0 * jax.random.normal(noise_keys[3], shape=(3,))
        )
        omega_tar = omega_tar + omega_tar_noise

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
        # dynamics NOTE: u is normalized thrust and torque [-1, 1]
        # thrust_normed = env_action.thrust/env_params.max_thrust * 2.0 - 1.0
        # torque_normed = env_action.torque / env_params.max_torque
        # NOTE hack here, just convert torque to omega
        omega_tar = env_action.torque / env_params.max_torque * env_params.max_omega
        u = jnp.concatenate([jnp.array([env_action.thrust]), omega_tar])
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


def get_free_dynamics_3d_disturbance(d_func):
    # H = jnp.vstack((jnp.eye(3), jnp.zeros((1, 3))))

    @partial(jax.jit, static_argnames=["d_func"])
    def quad_dynamics(x: jnp.ndarray, u: jnp.ndarray, params: EnvParams3D, d_func):
        # NOTE: u is normalized thrust and torque [-1, 1]
        # thrust = (u[0] + 1.0) / 2.0 * params.max_thrust
        # torque = u[1:4] * params.max_torque
        thrust = u[0]
        torque = u[1:4]

        d = d_func(x, u, params)
        d_f = d[0:3]
        d_t = d[3:6]

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
            + d_f
        )
        omega_dot = (
            jnp.linalg.inv(params.I) @ (torque - geom.hat(omega) @ params.I @ omega)
            + d_t
        )

        # return
        x_dot = jnp.concatenate([r_dot, q_dot, v_dot, omega_dot])
        return x_dot

    @partial(jax.jit, static_argnames=["d_func"])
    def quad_dynamics_rk4(
        x: jnp.ndarray, u: jnp.ndarray, params: EnvParams3D, dt: float, d_func
    ) -> jnp.ndarray:
        f = lambda x, u, p: quad_dynamics(x, u, p, d_func)
        k1 = f(x, u, params)
        k2 = f(x + k1 * dt / 2, u, params)
        k3 = f(x + k2 * dt / 2, u, params)
        k4 = f(x + k3 * dt, u, params)
        x_new = x + (k1 + 2 * k2 + 2 * k3 + k4) / 6 * dt
        return x_new.at[3:7].set(x_new[3:7] / jnp.linalg.norm(x_new[3:7]))

    @partial(jax.jit, static_argnames=["d_func"])
    def free_dynamics_3d(
        env_params: EnvParams3D, env_state: EnvState3D, env_action: Action3D, d_func
    ):
        # dynamics NOTE: u is normalized thrust and torque [-1, 1]
        # thrust_normed = env_action.thrust/env_params.max_thrust * 2.0 - 1.0
        # torque_normed = env_action.torque / env_params.max_torque
        u = jnp.concatenate([jnp.array([env_action.thrust]), env_action.torque])
        x = jnp.concatenate(
            [env_state.pos, env_state.quat, env_state.vel, env_state.omega]
        )

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

    return partial(free_dynamics_3d, d_func=d_func), partial(
        quad_dynamics_rk4, d_func=d_func
    )
