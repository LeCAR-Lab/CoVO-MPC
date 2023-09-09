import jax
from jax import numpy as jnp
from functools import partial
from flax import struct

from quadjax import controllers

@struct.dataclass
class FixedParams:
    u: jnp.ndarray

class FixedController(controllers.BaseController):
    def __init__(self, env) -> None:
        super().__init__(env)

    # @partial(jax.jit, static_argnums=(0,))
    def update_params(self, env_params, control_params):
        pass
    
    @partial(jax.jit, static_argnums=(0,))
    def __call__(self, obs, state, env_params, rng_act, control_params) -> jnp.ndarray:
        return control_params.u