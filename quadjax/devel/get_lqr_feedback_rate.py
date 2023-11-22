import control
import jax
from jax import numpy as jnp

from quadjax.envs.cartpole import CartPole, CartPoleState
from quadjax.envs.acrobot import Acrobot, AcrobotState

def cartpole():
    #cartpole environment
    env = CartPole()
    state = CartPoleState(0.0, 0.0, 0.0, 0.0, 0, 0.0)
    params = env.default_params
    # get A
    def dynamics_fn(x, u):
        state = CartPoleState(x[0], x[1], x[2], x[3], 0, 0.0)
        x_new, _, _, _, _ = env.step_env(None, state, u, params)
        return x_new
    A = jax.jacfwd(dynamics_fn, argnums=0)(jnp.zeros(4), jnp.zeros(1))
    # get B
    B = jax.jacfwd(dynamics_fn, argnums=1)(jnp.zeros(4), jnp.zeros(1))
    # get Q and R
    Q = jnp.diag(jnp.array([0.001, 0.001, 1.0, 0.1]))
    R = jnp.diag(jnp.array([0.1]))
    # get K
    K, _, _ = control.dlqr(A, B, Q, R)
    # print K
    print(f'A = {A}, \n B = {B}, \n Q = {Q}, \n R = {R}, \n K = {K}')
    # run simulation with K
    state = CartPoleState(0.05, 0.05, 0.05, 0.05, 0, 0.0)
    done = False
    obses = []
    while not done:
        u = -K @ jnp.asarray([state.x, state.x_dot, state.theta, state.theta_dot])
        obs, state, _, done, _ = env.step_env(None, state, u, params)
        obses.append(obs)
    # plot all obs
    import matplotlib.pyplot as plt
    obses = jnp.asarray(obses)
    obs_dim = obses.shape[-1]
    fig, axs = plt.subplots(obs_dim, 1)
    for i in range(obs_dim):
        axs[i].plot(obses[:, i])
    plt.savefig('../../results/lqr_obs.png')
    
def acrobot():
    #cartpole environment
    env = Acrobot()
    state = AcrobotState(jnp.pi, 0.0, 0.0, 0.0, 0, 0.0)
    params = env.default_params
    # get A
    def dynamics_fn(x, u):
        state = AcrobotState(x[0], x[1], x[2], x[3], 0, 0.0)
        x_new, _, _, _, _ = env.step_env(None, state, u, params)
        return x_new
    A = jax.jacfwd(dynamics_fn, argnums=0)(jnp.zeros(4), jnp.zeros(1))
    # get B
    B = jax.jacfwd(dynamics_fn, argnums=1)(jnp.zeros(4), jnp.zeros(1))
    # get Q and R
    Q = jnp.diag(jnp.array([1.0, 1.0, 0.1, 0.1]))
    R = jnp.diag(jnp.array([0.1]))
    # check controlability
    C = jnp.hstack([B, A @ B, A @ A @ B, A @ A @ A @ B])
    print(f'rank of C = {jnp.linalg.matrix_rank(C)}')
    # get K
    K, _, _ = control.dlqr(A, B, Q, R)
    # print K
    print(f'A = {A}, \n B = {B}, \n Q = {Q}, \n R = {R}, \n K = {K}')
    # run simulation with K
    state = AcrobotState(0.05+jnp.pi, 0.05, 0.05, 0.05, 0, 0.0)
    state = AcrobotState(jnp.pi, 0.0, 0.0, 0.0, 0, 0.0)
    obs = env.get_obs(state, params)
    done = False
    obses = []
    while not done:
        u = -K @ obs[:4]
        obs, state, _, done, _ = env.step_env(None, state, u, params)
        obses.append(obs)
    # plot all obs
    import matplotlib.pyplot as plt
    obses = jnp.asarray(obses)
    obs_dim = obses.shape[-1]
    fig, axs = plt.subplots(obs_dim, 1)
    for i in range(obs_dim):
        axs[i].plot(obses[:, i])
    plt.savefig('../../results/lqr_obs.png')

if __name__ == "__main__":
    acrobot()