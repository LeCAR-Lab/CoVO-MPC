import jax
from jax import numpy as jnp
from functools import partial
from flax import struct

from quadjax import controllers

@struct.dataclass
class FixedParams:
    u: jnp.ndarray

class FixedController(controllers.BaseController):
    def __init__(self, env, control_params) -> None:
        super().__init__(env, control_params)

    def update_params(self, env_params, control_params):
        return control_params
    
    def __call__(self, obs, state, env_params, rng_act, control_params, env_info) -> jnp.ndarray:
        return control_params.u, control_params, None