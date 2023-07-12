from jax import numpy as jnp

from quadjax.dynamics.dataclass import EnvState3D, EnvParams3D, Action3D
from quadjax.dynamics.loose import get_loose_dynamics, get_loose_dynamics_3d
from quadjax.dynamics.taut import get_taut_dynamics, get_taut_dynamics_3d
from quadjax.dynamics.trans import get_dynamic_transfer, get_dynamic_transfer_3d

def make_hybrid_rope_dyn_3d():
    loose_dynamics = get_loose_dynamics_3d()
    taut_dynamics = get_taut_dynamics_3d()
    dynamic_transfer = get_dynamic_transfer_3d()
    def hybrid_rope_dyn_3d(state:EnvState3D, env_action:Action3D, params:EnvParams3D):
        old_loose_state = state.l_rope < (
            params.l - params.rope_taut_therehold)
        taut_state = taut_dynamics(params, state, env_action)
        loose_state = loose_dynamics(params, state, env_action)
        new_state = dynamic_transfer(
            params, loose_state, taut_state, old_loose_state)
        return new_state
    return hybrid_rope_dyn_3d

def make_free_dyn_3d():
    loose_dynamics = get_loose_dynamics_3d()
    def free_dyn_3d(state:EnvState3D, env_action:Action3D, params:EnvParams3D):
        loose_state = loose_dynamics(params, state, env_action)
        loose_state = loose_state.replace(
            zeta = jnp.array([0.0, 0.0, -1.0]),
            zeta_dot = jnp.array([0.0, 0.0, 0.0]), 
            pos_obj = jnp.array([0.0, 0.0, 0.0]),
            vel_obj = jnp.array([0.0, 0.0, 0.0]),
            f_rope_norm = 0.0,
            f_rope = jnp.array([0.0, 0.0, 0.0]),
            l_rope = params.l
        )
        return loose_state
    return free_dyn_3d