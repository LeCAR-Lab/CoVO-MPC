import jax
from jax import numpy as jnp 
from flax import struct
from functools import partial

from quadjax import controllers
from quadjax.dynamics import geom
from quadjax.dynamics.dataclass import default_array

@struct.dataclass
class L1Params:
    As: float = -0.5
    alpha: float = 0.9

    vel_hat: jnp.ndarray = default_array([0.0, 0.0, 0.0])
    d_hat: jnp.ndarray = default_array([0.0, 0.0, 0.0])

    Kp: float = 4.0
    Kd: float = 4.0
    Kp_att: float = 4.0

class L1Controller(controllers.BaseController):
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
    def update_esitimate(self, state, control_params):

        Q = geom.qtoQ(state.quat)
        vel_hat_dot = jnp.array([0.0, 0.0, -self.param.g]) + 1.0 / self.param.m * (Q @ jnp.asarray([0, 0, state.last_thrust])) + control_params.d_hat + control_params.As * (control_params.vel_hat - state.vel)
        vel_hat = control_params.vel_hat + vel_hat_dot * self.param.dt
        phi = jnp.exp(control_params.As * self.param.dt)
        d_new = - 1.0 / (phi - 1.0) * control_params.As * phi * (vel_hat - state.vel)
        d_hat = control_params.alpha * control_params.d_hat + (1.0 - control_params.alpha) * d_new

        return control_params.replace(vel_hat=vel_hat, d_hat=d_hat)

    @partial(jax.jit, static_argnums=(0,))
    def __call__(self, obs, state, env_param, rng_act, control_params, info) -> jnp.ndarray:
        control_params = self.update_esitimate(state, control_params)

        # position control
        Q = geom.qtoQ(state.quat)
        f_d = self.param.m * (jnp.array([0.0, 0.0, self.param.g]) - control_params.Kp * (state.pos - state.pos_tar) - control_params.Kd * (state.vel - state.vel_tar) - control_params.d_hat + state.acc_tar)
        thrust = (Q.T @ f_d)[2]
        thrust = jnp.clip(thrust, 0.0, self.param.max_thrust)

        # attitude control
        z_d = f_d / jnp.linalg.norm(f_d)
        rotation_axis = jnp.cross(jnp.array([0.0, 0.0, 1.0]), z_d)
        # when the rotation axis is zero, set it to [0.0, 0.0, 1.0] and set angle to 0.0
        small_angle = jnp.linalg.norm(rotation_axis) < 1e-4
        rotation_axis = jnp.where(small_angle, jnp.array([0.0, 0.0, 1.0]), rotation_axis)
        rotation_angle = jnp.where(small_angle, 0.0, jnp.arcsin(jnp.linalg.norm(rotation_axis)))
        R_d = geom.axisangletoR(rotation_axis, rotation_angle)
        R_e = R_d.T @ Q
        angle_err = geom.vee(R_e - R_e.T)
        # generate desired angular velocity
        omega_d = - control_params.Kp_att * angle_err

        # generate action
        action = jnp.concatenate([jnp.array([(thrust/self.param.max_thrust) * 2.0 - 1.0]), (omega_d/self.param.max_omega)])

        return action, control_params, None