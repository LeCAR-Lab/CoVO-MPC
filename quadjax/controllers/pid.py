import jax
from jax import numpy as jnp
from functools import partial
from flax import struct

from quadjax import controllers
from quadjax.dynamics import geom
from quadjax.dynamics.dataclass import default_array


@struct.dataclass
class PIDParams:
    Kp: float = 4.0
    Kd: float = 4.0
    Ki: float = 1.0
    Kp_att: float = 4.0
    Ki_att: float = 1.0

    integral: jnp.ndarray = default_array([0.0, 0.0, 0.0])
    quat_desired: jnp.ndarray = default_array([0.0, 0.0, 0.0, 1.0])

    att_integral: jnp.ndarray = default_array([0.0, 0.0, 0.0])

class PIDController(controllers.BaseController):
    """L1 controller for path following

    Returns:
        _type_: _description_
    """

    def __init__(self, env, control_params) -> None:
        super().__init__(env, control_params)
        self.param = self.env.default_params

    def update_params(self, env_param, control_params):
        return control_params

    @partial(jax.jit, static_argnums=(0,))
    def __call__(
        self, obs, state, env_param, rng_act, control_params, info=None
    ) -> jnp.ndarray:
        # position control
        Q = geom.qtoQ(state.quat)
        f_d = self.param.m * (
            jnp.array([0.0, 0.0, self.param.g])
            - control_params.Kp * (state.pos - state.pos_tar)
            - control_params.Kd * (state.vel - state.vel_tar)
            - control_params.Ki * control_params.integral
            + state.acc_tar
        )
        thrust = (Q.T @ f_d)[2]
        thrust = jnp.clip(thrust, 0.0, self.param.max_thrust)

        # attitude control
        # make sure f_d is not zero
        f_d_norm = jnp.linalg.norm(f_d)
        f_d_norm = jnp.where(f_d_norm < 1e-3, 1e-3, f_d_norm)
        z_d = f_d / f_d_norm
        axis_angle = jnp.cross(jnp.array([0.0, 0.0, 1.0]), z_d)
        angle = jnp.linalg.norm(axis_angle)
        angle = jnp.where(angle < 1e-3, 5e-4, angle)
        axis = jnp.where((angle < 1e-3), jnp.array([0.0, 0.0, 1.0]), axis_angle / angle)
        R_d = geom.axisangletoR(axis, angle)
        quat_desired = geom.Qtoq(R_d)
        R_e = R_d.T @ Q
        angle_err = geom.vee(R_e - R_e.T)
        # generate desired angular velocity
        omega_d = -control_params.Kp_att * angle_err

        # generate action
        action = jnp.concatenate(
            [
                jnp.array([(thrust / self.param.max_thrust) * 2.0 - 1.0]),
                (omega_d / self.param.max_omega),
            ]
        )

        # update control_params
        integral = control_params.integral + (state.pos - state.pos_tar) * env_param.dt
        control_params = control_params.replace(
            quat_desired=quat_desired, integral=integral
        )

        return action, control_params, None


class PIDController2D(controllers.BaseController):
    """L1 controller for path following

    Returns:
        _type_: _description_
    """

    def __init__(self, env, control_params) -> None:
        super().__init__(env, control_params)
        self.param = self.env.default_params

    def update_params(self, env_param, control_params):
        return control_params

    @partial(jax.jit, static_argnums=(0,))
    def __call__(
        self, obs, state, env_param, rng_act, control_params, info=None
    ) -> jnp.ndarray:
        # position control
        quat = state.quat
        f_d = self.param.m * (
            jnp.array([0.0, self.param.g])
            - control_params.Kp * (state.pos - state.pos_tar)
            - control_params.Kd * (state.vel - state.vel_tar)
            - control_params.Ki * control_params.integral
        )
        thrust = jnp.dot(jnp.array([-jnp.sin(quat), jnp.cos(quat)]), f_d)
        thrust = jnp.clip(thrust, 0.0, self.param.max_thrust)

        # attitude control
        f_d_norm = jnp.linalg.norm(f_d)
        z_d = f_d / f_d_norm
        z_d = jnp.where(f_d_norm < 1e-3, jnp.array([0.0, 1.0]), z_d)
        roll_tar = jnp.arctan2(-z_d[0], z_d[1])
        angle_err = roll_tar - state.quat
        # generate desired angular velocity
        omega_d = control_params.Kp_att * angle_err

        # attitude rate control
        error = omega_d - state.omega
        torque = 30.0 * error * self.param.I

        # generate action
        action = jnp.array(
            [
                (thrust / self.param.max_thrust) * 2.0 - 1.0,
                torque / self.param.max_torque,
                # omega_d / self.param.max_omega,
            ]
        )

        # update control_params
        integral = control_params.integral + (state.pos - state.pos_tar) * env_param.dt
        control_params = control_params.replace(integral=integral)

        info = {}
        # info["pos_err"]=(control_params.Kp * (state.pos - state.pos_tar))
        # info["vel_err"]=(control_params.Kd * (state.vel - state.vel_tar))
        # info["err_i"]=(control_params.Ki * control_params.integral)
        # info["angle_err"]=(control_params.Kp_att * angle_err)
        # info["err_i_att"]=(jnp.zeros_like(angle_err))
        # info["time"]=state.time
        # info["pos_cur"]=(state.pos)
        # info["vel_cur"]=(state.vel)
        # info["omega_cur"]=(state.omega)
        # info["ang_cur"]=geom.qtorpy(geom.Qtoq(Q))
        # info["pos_tar"]=state.pos_tar
        # info["vel_tar"]=state.vel_tar
        # info["omega_tar"]=(omega_d)
        # info["ang_tar"]=(geom.qtorpy(geom.Qtoq(R_d)))
        # info["f_d"]=(f_d)
        # info["thrust"]=(thrust)



        return action, control_params, info
    
@struct.dataclass
class BodyratePIDParams:
    kp: jnp.ndarray
    ki: jnp.ndarray
    kd: jnp.ndarray
    last_error: jnp.ndarray
    integral: jnp.ndarray


class PIDControllerBodyrate(controllers.BaseController):
    """PID controller for attitude rate control

    Returns:
        _type_: _description_
    """

    def __init__(self, env, control_params) -> None:
        super().__init__(env, control_params)

    def update_params(self, env_params, control_params):
        return control_params.replace(
            last_error=jnp.zeros_like(control_params.last_error),
            integral=jnp.zeros_like(control_params.integral),
        )

    def __call__(self, obs, state, env_params, rng_act, control_params) -> jnp.ndarray:
        error = state.omega_tar - state.omega
        integral = control_params.integral + error * env_params.dt
        derivative = (error - control_params.last_error) / env_params.dt
        control_params = control_params.replace(last_error=error, integral=integral)
        u = (
            control_params.kp * error
            + control_params.ki * integral
            + control_params.kd * derivative
        )
        return u, control_params, None
    
@struct.dataclass
class PIDParamsPWM:
    kp: jnp.ndarray
    ki: jnp.ndarray
    kd: jnp.ndarray
    integral_max: jnp.ndarray
    last_error: jnp.ndarray
    integral: jnp.ndarray
    
class PIDControllerPWM(controllers.BaseController):
    """PID controller for attitude rate control

    Returns:
        _type_: _description_
    """

    def __init__(self, env, control_params) -> None:
        super().__init__(env, control_params)

    def update_params(self, env_params, control_params):
        return control_params.replace(
            last_error=jnp.zeros_like(control_params.last_error),
            integral=jnp.zeros_like(control_params.integral),
        )

    def __call__(self, obs, state, env_params, rng_act, control_params) -> jnp.ndarray:
        dt = env_params.dt * 0.1
        error = state.omega_tar - state.omega
        integral = control_params.integral + error * dt
        integral = jnp.clip(integral, -control_params.integral_max, control_params.integral_max)
        derivative = (error - control_params.last_error) / dt
        control_params = control_params.replace(last_error=error, integral=integral)
        alpha_des = (
            control_params.kp * error
            + control_params.ki * integral
            + control_params.kd * derivative
        )
        alpha_des = jnp.clip(alpha_des, -env_params.max_alpha, env_params.max_alpha)
        torque = env_params.I @ alpha_des + jnp.cross(state.omega, env_params.I @ state.omega)
        return torque, control_params, None