import jax
from jax import numpy as jnp 
from flax import struct

from quadjax import controllers
from quadjax.dynamics import geom
from quadjax.dynamics.dataclass import default_array

@struct.dataclass
class L1Params:
    As: float = -0.5
    alpha: float = 0.9

    vel_hat: jnp.ndarray = default_array([0.0, 0.0, 0.0])
    d_hat: jnp.ndarray = default_array([0.0, 0.0, 0.0])

    Kp = 4.0
    Kd = 4.0
    Kp_att = 4.0

class L1Controller(controllers.BaseController):
    """L1 controller for path following

    Returns:
        _type_: _description_
    """

    def __init__(self, env, control_params) -> None:
        super().__init__(env, control_params)

    def update_params(self, envel_params, control_params):
        return control_params
    
    def update_esitimate(self, obs, state, envel_params, rng_act, control_params):
        # overwrite the parameter with the default ones
        envel_params = self.env.default_params

        Q = geom.qtoQ(state.quat)
        vel_hat_dot = jnp.array([0.0, 0.0, -envel_params.g]) + 1.0 / envel_params.m * (Q @ jnp.asarray([0, 0, state.last_thrust])) + control_params.d_hat + control_params.As * (control_params.vel_hat - state.vel)
        vel_hat = control_params.vel_hat + vel_hat_dot * envel_params.dt
        phi = jnp.exp(control_params.As * envel_params.dt)
        d_new = - 1.0 / (phi - 1.0) * control_params.As * phi * (control_params.vel_hat - state.vel)
        d_hat = control_params.alpha * control_params.d_hat + (1.0 - control_params.alpha) * d_new

        return control_params.replace(vel_hat=vel_hat, d_hat=d_hat)

    def __call__(self, obs, state, envel_params, rng_act, control_params, info) -> jnp.ndarray:
        # overwrite the parameter with the default ones
        envel_params = self.env.default_params

        control_params = self.update_esitimate(obs, state, envel_params, rng_act, control_params)

        # position control
        Q = geom.qtoQ(state.quat)
        f_d = envel_params.m * (jnp.array([0.0, 0.0, envel_params.g]) - control_params.Kp * (state.pos - state.pos_tar) - control_params.Kd * (state.vel - state.vel_tar) - control_params.d_hat)
        thrust = (Q.T @ f_d)[2]
        thrust = jnp.clip(thrust, 0.0, envel_params.max_thrust)

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
        action = jnp.array([(thrust/envel_params.max_thrust) * 2.0 - 1.0, *(omega_d/envel_params.max_omega)])

        return action, control_params, None