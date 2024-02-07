from jax import numpy as jnp
from typing import Tuple, Optional
import chex
from functools import partial
import jax

from quadjax.dynamics import utils, geom
from quadjax.dynamics.dataclass import EnvParams3D, EnvState3D, Action3D

# polyfit using data and scripts from https://github.com/IMRCLab/crazyflie-system-id
# p = jnp.array([1.71479058e-09,  8.80284482e-05, -2.21152097e-01])

@jax.jit
def pwm_to_force(pwm, p):
    force_in_grams = jnp.polyval(p, pwm)
    force_in_newton = force_in_grams * 9.81 / 1000.0
    return jnp.maximum(force_in_newton, 0)

@jax.jit
def force_to_pwm(force, p):
    a, b, c = p
    force_in_grams = force * 1000.0 / 9.81
    pwm = (-b + jnp.sqrt(b**2 - 4*a*(c-force_in_grams))) / (2*a)
    return pwm

arm_length = 0.046  # m
arm = 0.707106781 * arm_length
t2t = 0.006

B0 = jnp.array([
    [1, 1, 1, 1],
    [-arm, -arm, arm, arm],
    [-arm, arm, arm, -arm],
    [-t2t, t2t, -t2t, t2t]
])

def get_free_dynamics_pwm():
    @jax.jit
    def quad_dynamics(x: jnp.ndarray, u: jnp.ndarray, params: EnvParams3D, dt: float) -> jnp.ndarray:
        # NOTE: u is normalized pwm signal from [-1, 1]
        # get state information
        r = x[:3]  # position in world frame
        q = x[3:7] / jnp.linalg.norm(x[3:7])  # quaternion in world frame
        v = x[7:10]  # velocity in body frame
        omega = x[10:13]  # angular velocity in body frame
        Q = geom.qtoQ(q)  # quaternion to rotation matrix
        last_u = x[13:17]  # last pwm signal

        # convert pwm to motor force
        alpha = dt / params.tau_thrust
        # alpha = 1.0
        u_pwm = alpha * u + (1 - alpha) * last_u
        pwm = (u_pwm + 1.0) / 2.0 * 65535
        f = pwm_to_force(pwm, params.pwmf)
        eta = B0 @ f
        thrust = eta[0]  # convert to motor force
        torque = eta[1:]

        # jax.debug.print('torque applied = {x}', x = torque/params.max_torque)

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
        x_dot = jnp.concatenate([r_dot, q_dot, v_dot, omega_dot, jnp.zeros(4)])
        return x_dot

    @jax.jit
    def quad_dynamics_rk4(
        x: jnp.ndarray, u: jnp.ndarray, params: EnvParams3D, dt: float
    ) -> jnp.ndarray:
        f = quad_dynamics
        k1 = f(x, u, params, dt)
        k2 = f(x + k1 * dt / 2, u, params, dt)
        k3 = f(x + k2 * dt / 2, u, params, dt)
        k4 = f(x + k3 * dt, u, params, dt)
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
        thrust_normed = env_action.thrust / env_params.max_thrust * 2.0 - 1.0
        torque_normed = env_action.torque / env_params.max_torque
        last_thrust_normed = env_state.last_thrust / env_params.max_thrust * 2.0 - 1.0
        last_torque_normed = env_state.last_torque / env_params.max_torque
        u = jnp.concatenate([jnp.array([thrust_normed]), torque_normed])
        x = jnp.concatenate(
            [env_state.pos, env_state.quat, env_state.vel, env_state.omega, jnp.array([last_thrust_normed]), last_torque_normed]
        )

        # rk4
        x_new = quad_dynamics_rk4(x, u, env_params, sim_dt)
        pos = x_new[:3]
        quat = x_new[3:7] / jnp.linalg.norm(x_new[3:7])
        vel = x_new[7:10]
        omega = x_new[10:13]

        # step
        # time = env_state.time + 1

        # trajectory
        # pos_tar = env_state.pos_traj[time]
        # vel_tar = env_state.vel_traj[time]

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
            # pos_tar=pos_tar,
            # vel_tar=vel_tar,
            # debug value
            last_thrust=last_thrust,
            last_torque=last_torque,
            # step
            # time=time,
        )

        return env_state
    
    def update_time(env_state: EnvState3D):
        time = env_state.time + 1

        return env_state.replace(
            time=time,
            pos_tar = env_state.pos_traj[time], 
            vel_tar = env_state.vel_traj[time]
        )

    return free_dynamics_3d, update_time
