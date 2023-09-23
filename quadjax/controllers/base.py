import jax
from jax import numpy as jnp
from functools import partial

class BaseController:
    def __init__(self, env) -> None:
        self.env = env

    # @partial(jax.jit, static_argnums=(0,))
    def update_params(self, env_params, control_params):
        return control_params
    
    @partial(jax.jit, static_argnums=(0,))
    def __call__(self, obs, state, env_params, rng_act, control_params) -> jnp.ndarray:
        raise NotImplementedError