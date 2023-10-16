import jax
from jax import numpy as jnp
from flax import struct
from functools import partial

from quadjax import controllers
from quadjax.dynamics import geom
from quadjax.dynamics.dataclass import default_array


@struct.dataclass
class NLACParams:
    base_dim: int = 12
    a_hat: jnp.ndarray = default_array(jnp.zeros((base_dim,)))
    d_hat: jnp.ndarray = default_array([0.0, 0.0, 0.0])
    vel_hat: jnp.ndarray = default_array([0.0, 0.0, 0.0])
    As: jnp.ndarray = default_array(-5.0 * jnp.eye(3, dtype=jnp.float32))
    P: jnp.ndarray = default_array(5.0 * jnp.eye(base_dim, dtype=jnp.float32))
    R: jnp.ndarray = default_array(1.0 * jnp.eye(3, dtype=jnp.float32))
    alpha: float = 0.9

    Kp: float = 4.0
    Kd: float = 4.0
    Kp_att: float = 4.0


class NLAdaptiveController(controllers.BaseController):
    """Non-linear adaptive controller for path following

    Returns:
        _type_: _description_
    """

    def __init__(self, env, control_params) -> None:
        super().__init__(env, control_params)
        self.param = self.env.default_params

    def update_params(self, env_param, control_params):
        return control_params

    def phi(state, params):
        v = state.vel
        time = state.time
        return jnp.concatenate([jnp.diag(v*jnp.abs(v)), jnp.eye(3)*jnp.sin(2*jnp.pi/params.disturb_period*time), jnp.eye(3)*jnp.cos(2*jnp.pi/params.disturb_period*time), jnp.eye(3)], axis=1)

    @partial(jax.jit, static_argnums=(0,))
    def update_esitimate(self, state, control_params):

        Q = geom.qtoQ(state.quat)
        vel_hat_dot = jnp.array([0.0, 0.0, -self.param.g]) + 1.0 / self.param.m * (Q @ jnp.asarray([0, 0, state.last_thrust])) + \
            NLAdaptiveController.phi(state, self.param) @ control_params.a_hat + \
            control_params.As @ (control_params.vel_hat - state.vel)
        vel_hat = control_params.vel_hat + vel_hat_dot * self.param.dt

        a_hat_new = - control_params.P @ NLAdaptiveController.phi(state, self.param).T @ jnp.linalg.inv(
            control_params.R) @ (state.vel - vel_hat) * self.param.dt + control_params.a_hat

        a_hat = control_params.alpha * control_params.a_hat + \
            (1.0 - control_params.alpha) * a_hat_new

        d_hat = NLAdaptiveController.phi(state, self.param) @ a_hat
        return control_params.replace(vel_hat=vel_hat, d_hat=d_hat, a_hat=a_hat)

    @partial(jax.jit, static_argnums=(0,))
    def __call__(self, obs, state, env_param, rng_act, control_params, info) -> jnp.ndarray:
        control_params = self.update_esitimate(state, control_params)

        # position control
        Q = geom.qtoQ(state.quat)
        f_d = self.param.m * (jnp.array([0.0, 0.0, self.param.g]) - control_params.Kp * (
            state.pos - state.pos_tar) - control_params.Kd * (state.vel - state.vel_tar) - control_params.d_hat + state.acc_tar)
        thrust = (Q.T @ f_d)[2]
        thrust = jnp.clip(thrust, 0.0, self.param.max_thrust)

        # attitude control
        z_d = f_d / jnp.linalg.norm(f_d)
        axis_angle = jnp.cross(jnp.array([0.0, 0.0, 1.0]), z_d)
        angle = jnp.linalg.norm(axis_angle)
        # when the rotation axis is zero, set it to [0.0, 0.0, 1.0] and set angle to 0.0
        small_angle = (jnp.abs(angle) < 1e-4)
        axis = jnp.where(small_angle, jnp.array(
            [0.0, 0.0, 1.0]), axis_angle / angle)
        R_d = geom.axisangletoR(axis, angle)
        R_e = R_d.T @ Q
        angle_err = geom.vee(R_e - R_e.T)
        # generate desired angular velocity
        omega_d = - control_params.Kp_att * angle_err

        # generate action
        action = jnp.concatenate([jnp.array(
            [(thrust/self.param.max_thrust) * 2.0 - 1.0]), (omega_d/self.param.max_omega)])

        return action, control_params, None
