import jax
from jax import numpy as jnp
from functools import partial
from flax import struct

from quadjax import controllers

@struct.dataclass
class PIDParams:
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
            last_error = jnp.zeros_like(control_params.last_error),
            integral = jnp.zeros_like(control_params.integral)
        )
    
    def __call__(self, obs, state, env_params, rng_act, control_params) -> jnp.ndarray:
        error = state.omega_tar - state.omega
        integral = control_params.integral + error * env_params.dt
        derivative = (error - control_params.last_error) / env_params.dt
        control_params = control_params.replace(last_error=error, integral=integral)
        u = control_params.kp * error + control_params.ki * integral + control_params.kd * derivative
        return u, control_params, None

