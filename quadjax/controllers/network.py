import jax
import chex

from jax import numpy as jnp
from functools import partial


class NetworkController:
    def __init__(self, apply_fn) -> None:
        self.apply_fn = apply_fn

    # @partial(jax.jit, static_argnums=(0,))
    def update_params(self, env_params, control_params):
        return control_params
    
    @partial(jax.jit, static_argnums=(0,))
    def __call__(self, obs, state, env_params, rng_act, control_params) -> jnp.ndarray:
        return self.apply_fn(control_params, obs)[0].mean()