import jax
import chex
import numpy as np
from jax import numpy as jnp
from functools import partial
import control
from flax import struct

from quadjax.envs import Quad3D
from quadjax.dynamics import EnvParams3D, EnvState3D
from quadjax.dynamics import geom

@struct.dataclass
class LQRParams:
    Q: jnp.ndarray
    R: jnp.ndarray
    K: jnp.ndarray

class LQRController:
    def __init__(self, env:Quad3D) -> None:
        self.env = env
        self.A_func = jax.jacfwd(self.env.dynamics_fn, argnums=0)
        self.B_func = jax.jacfwd(self.env.dynamics_fn, argnums=1)
        self.E_q0 = geom.E(jnp.array([0.0]*3+[1.0]))

    # @partial(jax.jit, static_argnums=(0,))
    def update_params(self, env_params: EnvParams3D, control_params: LQRParams) -> LQRParams:
        u = jnp.array([env_params.m * env_params.g, 0.0, 0.0, 0.0])
        A = self.A_func(self.env.equib, u, env_params, env_params.dt)
        B = self.B_func(self.env.equib, u, env_params, env_params.dt)
        # verify controllability
        # print(np.linalg.matrix_rank(control.ctrb(np.array(A), np.array(B))))

        A_reduced = self.E_q0.T @ A @ self.E_q0
        B_reduced = self.E_q0.T @ B
        # save A_reduced and B_reduced to csv
        np.savetxt("../../results/A.csv", A_reduced, delimiter=",")
        np.savetxt("../../results/B.csv", B_reduced, delimiter=",")
        # solve discrete time LQR to get K
        dlqr_args = (A_reduced, B_reduced, control_params.Q, control_params.R)
        dlqr_args = [np.asarray(item) for item in dlqr_args]
        # verify controllability
        # print(np.linalg.matrix_rank(control.ctrb(*dlxqr_args[:2])))
        # get control parameters
        K, _, _ = control.dlqr(*dlqr_args)
        # update controller parameters
        control_params = control_params.replace(K=K)
        return control_params
    
    @partial(jax.jit, static_argnums=(0,))
    def __call__(self, obs:jnp.ndarray, state: EnvState3D, env_params: EnvParams3D, rng_act: chex.PRNGKey, control_params: LQRParams) -> jnp.ndarray:
        delta_pos = state.pos - state.pos_tar
        delta_q = geom.L(state.quat).T @ jnp.array([0.0]*3 + [1.0])
        delta_phi = geom.qtorp(delta_q)
        delta_v = state.vel - state.vel_tar
        delta_w = state.omega - jnp.zeros(3)

        delta_x = jnp.concatenate([delta_pos, delta_phi, delta_v, delta_w])
        u = jnp.array([env_params.m * env_params.g, 0.0, 0.0, 0.0]) - \
                control_params.K @ delta_x
        # normalize u
        thrust_normed = ((u[0] + env_params.m * env_params.g) / env_params.max_thrust) * 2.0 - 1.0
        torque_normed = u[1:4] / env_params.max_torque
        return jnp.concatenate([jnp.array([thrust_normed]), torque_normed])