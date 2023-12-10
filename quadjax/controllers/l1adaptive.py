import jax
from jax import numpy as jnp 
from flax import struct
from functools import partial

from quadjax import controllers
from quadjax.dynamics import geom
from quadjax.dynamics.dataclass import default_array

@struct.dataclass
class L1ParamsBodyrate:
    As: float = -0.5
    alpha: float = 0.9

    omega_hat: jnp.ndarray = default_array([0.0, 0.0, 0.0])
    d_hat: jnp.ndarray = default_array([0.0, 0.0, 0.0])

    Kp: float = 30.0

class L1ControllerBodyrate(controllers.BaseController):
    """L1 controller for path following

    Returns:
        _type_: _description_
    """

    def __init__(self, env, control_params, sim_dt) -> None:
        super().__init__(env, control_params)
        self.param = self.env.default_params
        self.sim_dt = sim_dt

    def update_params(self, env_param, control_params):
        return control_params
    
    @partial(jax.jit, static_argnums=(0,))
    def update_esitimate(self, state, control_params):
        omega_hat_dot = jnp.linalg.inv(self.param.I) @ state.last_torque + control_params.d_hat + control_params.As * (control_params.omega_hat - state.omega)
        omega_hat = control_params.omega_hat + omega_hat_dot * self.sim_dt
        phi = jnp.exp(control_params.As * self.sim_dt)
        d_new = - 1.0 / (phi - 1.0) * control_params.As * phi * (omega_hat - state.omega)
        d_hat = control_params.alpha * control_params.d_hat + (1.0 - control_params.alpha) * d_new

        return control_params.replace(omega_hat=omega_hat, d_hat=d_hat)

    @partial(jax.jit, static_argnums=(0,))
    def __call__(self, obs, state, env_param, rng_act, control_params, info) -> jnp.ndarray:
        control_params = self.update_esitimate(state, control_params)

        # angular acceleration control with P controller 
        alpha = - control_params.Kp * (state.omega - state.omega_tar) - control_params.d_hat

        return alpha, control_params, None