import jax
from jax import numpy as jnp
from functools import partial
from flax import struct

from quadjax.controllers import BaseController

@struct.dataclass
class FeedbackParams:
    K: jnp.ndarray

class FeedbackController(BaseController):
    def __init__(self, env, control_params) -> None:
        self.env = env
        self.init_control_params = control_params

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, env_state=None, env_params=None, control_params=None, key=None):
        return self.init_control_params
    
    @partial(jax.jit, static_argnums=(0,))
    def __call__(self, obs, state, env_params, rng_act, control_params, env_info = None) -> jnp.ndarray:
        return -control_params.K @ obs, control_params, None