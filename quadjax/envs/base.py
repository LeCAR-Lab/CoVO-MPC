import jax
import chex
from typing import Tuple, Union, Optional, Any
from functools import partial
from flax import struct
from gymnax.environments.environment import Environment, EnvParams, EnvState 
from gymnax.wrappers.purerl import LogWrapper as PureLogWrapper
from gymnax.wrappers.purerl import LogEnvState as PureLogEnvState


class BaseEnvironment(Environment):
    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return super().__call__(*args, **kwds)

    @partial(jax.jit, static_argnums=(0,))
    def step(
        self,
        key: chex.PRNGKey,
        state: EnvState,
        action: Union[int, float],
        params: Optional[EnvParams] = None,
    ) -> Tuple[chex.Array, EnvState, float, bool, dict]:
        """Performs step transitions in the environment."""
        # Use default env parameters if no others specified
        if params is None:
            params = self.default_params
        key, key_reset = jax.random.split(key)
        obs_st, state_st, reward, done, info = self.step_env(
            key, state, action, params
        )
        obs_re, info_re, state_re = self.reset_env(key_reset, params)
        # Auto-reset environment based on termination
        state = jax.tree_map(
            lambda x, y: jax.lax.select(done, x, y), state_re, state_st
        )
        info = jax.tree_map(
            lambda x, y: jax.lax.select(done, x, y), info_re, info
        )
        obs = jax.lax.select(done, obs_re, obs_st)
        return obs, state, reward, done, info
    
    @partial(jax.jit, static_argnums=(0,))
    def reset(
        self, key: chex.PRNGKey, params: Optional[EnvParams] = None
    ) -> Tuple[chex.Array, EnvState]:
        """Performs resetting of environment."""
        # Use default env parameters if no others specified
        if params is None:
            params = self.default_params
        return self.reset_env(key, params)
    

class LogWrapper(PureLogWrapper):
    """Log the episode returns and lengths."""

    def __init__(self, env: BaseEnvironment):
        super().__init__(env)

    @partial(jax.jit, static_argnums=(0,))
    def reset(
        self, key: chex.PRNGKey, params: Optional[EnvParams] = None
    ) -> Tuple[chex.Array, EnvState]:
        obs, info, env_state = self._env.reset(key, params)
        info["returned_episode_returns"] = 0.0
        info["returned_episode_lengths"] = 0
        info["returned_episode"] = False
        state = PureLogEnvState(env_state, 0, 0, 0, 0)
        return obs, info, state