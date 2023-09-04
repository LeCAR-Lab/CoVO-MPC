import jax
from jax import numpy as jnp
from functools import partial

from quadjax.envs import Quad3D
from quadjax.dynamics import EnvParams3D

class LQRController:
    def __init__(self, env:Quad3D) -> None:
        self.env = env
        self.A_func = jax.grad(self.env.dynamics_fn, argnums=0)
        self.B_func = jax.grad(self.env.dynamics_fn, argnums=1)

    @partial(jax.jit, static_argnums=(0,))
    def update_params(self, params: EnvParams3D):
        u = jnp.array([params.m * params.g, 0.0, 0.0, 0.0])
        A = self.A_func(self.env.equib, u, params)
        B = self.B_func(self.env.equib, u, params)
        return A, B