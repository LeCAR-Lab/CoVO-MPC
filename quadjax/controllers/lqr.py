import jax
import chex
import numpy as np
from jax import numpy as jnp
from functools import partial
import control
from flax import struct

# from quadjax.envs import Quad3D
from quadjax.dynamics import EnvParams3D, EnvState3D, EnvParams2D, EnvState2D
from quadjax.dynamics import geom
from quadjax import controllers

@struct.dataclass
class LQRParams:
    Q: jnp.ndarray
    R: jnp.ndarray
    K: jnp.ndarray

class LQRController2D(controllers.BaseController):
    def __init__(self, env, control_params) -> None:
        super().__init__(env, control_params)
        def normed_dynamics_fn(x, u_normed, env_params, dt):
            '''
            dynamics for controller (normalization, esitimation etc. )
            '''
            thrust = (u_normed[0] + 1.0) / 2.0 * self.env.default_params.max_thrust
            roll_rate = u_normed[1] * self.env.default_params.max_omega
            return self.env.dynamics_fn(x, jnp.asarray([thrust, roll_rate]), env_params, dt)
        self.A_func = jax.jacfwd(normed_dynamics_fn, argnums=0)
        self.B_func = jax.jacfwd(normed_dynamics_fn, argnums=1)

    @partial(jax.jit, static_argnums=(0,))
    def update_params(self, env_params: EnvParams2D, control_params: LQRParams) -> LQRParams:
        thrust_hover_normed = (env_params.m * env_params.g / env_params.max_thrust) * 2.0 - 1.0
        u_hover_normed = jnp.array([thrust_hover_normed, 0.0])
        A = self.A_func(self.env.equib, u_hover_normed, env_params, env_params.dt)
        B = self.B_func(self.env.equib, u_hover_normed, env_params, env_params.dt)

        K, _, _ = control.dlqr(A, B, control_params.Q, control_params.R)
        # save K to csv
        control_params = control_params.replace(K=K)
        return control_params

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, env_state, env_params, control_params, key):
        return self.update_params(env_params, control_params)
    
    @partial(jax.jit, static_argnums=(0,))
    def __call__(self, obs:jnp.ndarray, env_state: EnvState2D, env_params: EnvParams2D, rng_act: chex.PRNGKey, control_params: LQRParams) -> jnp.ndarray:
        delta_pos = env_state.pos - env_state.pos_tar
        roll_tar = 0.0
        delta_roll = env_state.roll - roll_tar
        delta_v = env_state.vel - env_state.vel_tar

        delta_x = jnp.asarray([*delta_pos, delta_roll, *delta_v])
        thrust_hover = env_params.m * env_params.g
        thrust_hover_normed = (thrust_hover / env_params.max_thrust) * 2.0 - 1.0
        u = jnp.asarray([thrust_hover_normed, 0.0]) - control_params.K @ delta_x
        return u, control_params, None
    
class LQRController(controllers.BaseController):
    def __init__(self, env, control_params) -> None:
        super().__init__(env, control_params)
        def normed_dynamics_fn(x, u_normed, env_params, dt):
            '''
            dynamics for controller (normalization, esitimation etc. )
            '''
            thrust = (u_normed[0] + 1.0) / 2.0 * self.env.default_params.max_thrust
            torque = u_normed[1:4] * self.env.default_params.max_torque
            return self.env.dynamics_fn(x, jnp.concatenate([jnp.array([thrust]), torque]), env_params, dt)
        self.A_func = jax.jacfwd(normed_dynamics_fn, argnums=0)
        self.B_func = jax.jacfwd(normed_dynamics_fn, argnums=1)

        q = jnp.array([0.0]*3+[1.0])
        # I3 = jnp.eye(3)
        # I9 = jnp.eye(9)
        # H = jnp.vstack((jnp.eye(3), jnp.zeros((1, 3))))
        # G = geom.L(q) @ H
        # self.E_q0 = jax.scipy.linalg.block_diag(I3, G, I9)
        self.E_q0 = geom.E(q)

    # @partial(jax.jit, static_argnums=(0,))
    def update_params(self, env_params: EnvParams3D, control_params: LQRParams) -> LQRParams:
        thrust_hover_normed = (env_params.m * env_params.g / env_params.max_thrust) * 2.0 - 1.0
        u_hover_normed = jnp.array([thrust_hover_normed, 0.0, 0.0, 0.0])
        A = self.A_func(self.env.equib, u_hover_normed, env_params, env_params.dt)[:13, :13]
        B = self.B_func(self.env.equib, u_hover_normed, env_params, env_params.dt)[:13, :4]

        A_reduced = self.E_q0.T @ A @ self.E_q0
        B_reduced = self.E_q0.T @ B
        # save A_reduced and B_reduced to csv save with 3 decimal places
        # np.savetxt('../../results/A.csv', A_reduced, delimiter=',')
        # np.savetxt('../../results/B.csv', B_reduced, delimiter=',')
        # solve discrete time LQR to get K
        K, _, _ = control.dlqr(A_reduced, B_reduced, control_params.Q, control_params.R)
        # save K to csv
        # np.savetxt('../../results/K.csv', K, delimiter=',')
        # update controller parameters
        control_params = control_params.replace(K=K)
        return control_params
    
    def reset(self, env_state, env_params, control_params, key):
        return self.update_params(env_params, control_params)
    
    @partial(jax.jit, static_argnums=(0,))
    def __call__(self, obs:jnp.ndarray, env_state: EnvState3D, env_params: EnvParams3D, rng_act: chex.PRNGKey, control_params: LQRParams, info = None) -> jnp.ndarray:
        delta_pos = env_state.pos - env_state.pos_tar
        quat_tar = jnp.asarray([0.0]*3 + [1.0])
        delta_q = geom.L(quat_tar).T @ env_state.quat
        delta_phi = geom.qtorp(delta_q)
        delta_v = env_state.vel - env_state.vel_tar
        delta_w = env_state.omega - jnp.zeros(3)

        delta_x = jnp.concatenate([delta_pos, delta_phi, delta_v, delta_w])
        thrust_hover = env_params.m * env_params.g
        thrust_hover_normed = (thrust_hover / env_params.max_thrust) * 2.0 - 1.0
        u = jnp.array([thrust_hover_normed, 0.0, 0.0, 0.0]) - control_params.K @ delta_x
        return u, control_params, None