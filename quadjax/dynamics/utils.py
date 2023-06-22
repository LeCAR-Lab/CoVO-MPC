import jax
from jax import numpy as jnp

@jax.jit
def angle_normalize(x: float) -> float:
    """Normalize the angle - radians."""
    return ((x + jnp.pi) % (2 * jnp.pi)) - jnp.pi

@jax.jit
def get_hit_penalty(y: float, z: float) -> float:
    half_width = 0.05
    half_height = 0.3
    within_obs_y_range = jnp.abs(y) < half_width
    outof_obs_z_range = jnp.abs(z) > half_height
    hit_y_bound = within_obs_y_range & outof_obs_z_range
    hit_panelty = -jnp.clip(
        hit_y_bound.astype(jnp.float32)
        * jnp.minimum(half_width - jnp.abs(y), jnp.abs(z) - half_height)
        * 500.0,
        0,
        1,
    )
    return hit_panelty