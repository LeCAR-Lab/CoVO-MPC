import jax
from jax import numpy as jnp
from functools import partial
from flax import struct

from quadjax import controllers

class RandomController(controllers.BaseController):
    def __init__(self, env, control_params) -> None:
        super().__init__(env, control_params)

    def update_params(self, env_params, control_params):
        return control_params
    
    def __call__(self, obs, state, env_params, rng_act, control_params, env_info=None) -> jnp.ndarray:
        return jax.random.normal(rng_act, (self.env.action_dim,))*0.3, control_params, None