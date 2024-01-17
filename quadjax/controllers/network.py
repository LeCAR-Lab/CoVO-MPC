import jax
from jax import numpy as jnp
from functools import partial

from quadjax import controllers

class NetworkController(controllers.BaseController):
    def __init__(self, apply_fn, env, control_params) -> None:
        super().__init__(env, control_params)
        self.apply_fn = apply_fn

    # @partial(jax.jit, static_argnums=(0,))
    def update_params(self, env_params, control_params):
        return control_params
    
    @partial(jax.jit, static_argnums=(0,))
    def __call__(self, obs, state, env_params, rng_act, control_params, env_info = None) -> jnp.ndarray:
        return self.apply_fn(control_params, obs, env_info)[0].mean(), control_params, {}