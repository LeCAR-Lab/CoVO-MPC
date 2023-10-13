import jax
from jax import numpy as jnp
from functools import partial

class BaseController:
    def __init__(self, env, control_params) -> None:
        self.env = env
        self.init_control_params = control_params

    # @partial(jax.jit, static_argnums=(0,))
    def update_params(self, env_params, control_params):
        return control_params
    
    def reset(self):
        return self.init_control_params
    
    @partial(jax.jit, static_argnums=(0,))
    def __call__(self, obs, state, env_params, rng_act, control_params, env_info = None) -> jnp.ndarray:
        raise NotImplementedError