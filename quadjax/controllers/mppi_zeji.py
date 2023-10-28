import jax
import chex
from flax import struct
from functools import partial
from jax import lax
from jax import numpy as jnp
import pickle
from jaxopt import ProjectedGradient
from jaxopt.projection import projection_hyperplane

from quadjax import controllers
from quadjax.dynamics import EnvParams2D, EnvState2D, geom
from quadjax.train import ActorCritic

@struct.dataclass
class MPPIZejiParams:
    gamma_mean: float # mean of gamma
    gamma_sigma: float # std of gamma
    discount: float # discount factor
    sample_sigma: float # std of sampling

    a_mean: jnp.ndarray # mean of action
    a_cov: jnp.ndarray # covariance matrix of action

class MPPIZejiController(controllers.BaseController):
    def __init__(self, env, control_params, N: int, H: int, lam: float, cov_mode:str = 'dynamic') -> None:
        super().__init__(env, control_params)
        self.N = N # NOTE: N is the number of samples, set here as a static number
        self.H = H
        self.lam = lam
        if cov_mode == 'dynamic':
            self.get_sigma = self.get_sigma_zeji
        elif cov_mode == 'static':
            raise NotImplementedError
        # network = ActorCritic(2, activation='tanh')
        # self.apply_fn = network.apply
        # with open('/home/pcy/Research/quadjax/results/ppo_params_quad2d_free_tracking_zigzag_base.pkl', 'rb') as f:
        #     self.network_params = pickle.load(f)

    def get_dJ_du(self, state:EnvState2D, env_params:EnvParams2D, control_params:MPPIZejiParams, a_mean:jnp.ndarray, rng_act: chex.PRNGKey = None):
        def single_rollout_fn(carry, action):
            state, params, reward_before, done_before, key, cumulated_reward = carry
            rng_act, key = jax.random.split(key)
            _, state, reward, done, _ = self.env.step_env_wocontroller(rng_act, state, action, params)
            reward = jnp.where(done_before, reward_before, reward)
            cumulated_reward = cumulated_reward + reward
            return (state, params, reward, done | done_before, key, cumulated_reward), None
        def get_cumulated_cost(a_mean_flattened, carry):
            a_mean = a_mean_flattened.reshape(self.H, -1)
            state, env_params, rng_act = carry
            (_, _, _, _, rng_act, cumulated_reward), _ = lax.scan(single_rollout_fn, (state, env_params, 0.0, False, rng_act, 0.0), a_mean, length=self.H)
            return -cumulated_reward
        # calculate the hessian of get_cumulated_reward
        hessian_fn = jax.jacfwd(jax.jacfwd(get_cumulated_cost, argnums=0), argnums=0)
        return hessian_fn(a_mean.flatten(), (state, env_params, rng_act))
    
    def get_sigma_zeji(self, R: jnp.ndarray, control_params: MPPIZejiParams):
        u, s, vh = jnp.linalg.svd(R)

        # Objective function
        def objective(params):
            return jnp.sum(s/(1+2/self.lam*s*jnp.exp(params))**2)

        # Constraint: x1 + x2 = log(det(Sigma)), represented as a hyperplane
        def projection_fn(params, hyperparams_proj):
            return projection_hyperplane(params, hyperparams=(hyperparams_proj, self.H*2*jnp.log(control_params.sample_sigma)))

        # Initialize 'ProjectedGradient' solver
        solver = ProjectedGradient(fun=objective, projection=projection_fn)

        # Initial parameters
        params_init = jnp.zeros(self.H*2)

        # Define the optimization problem
        sol = solver.run(params_init, hyperparams_proj=jnp.ones(self.H*2))

        sigma_eign = jnp.exp(sol.params)

        return u @ jnp.diag(sigma_eign) @ vh


    @partial(jax.jit, static_argnums=(0,))
    def __call__(self, obs:jnp.ndarray, state, env_params, rng_act: chex.PRNGKey, control_params: MPPIZejiParams, info = None) -> jnp.ndarray:
        # shift operator
        a_mean_old = control_params.a_mean
        a_cov_old = control_params.a_cov

        # DEBUG
        R = self.get_dJ_du(state, env_params, control_params, a_mean_old, rng_act)
        a_cov_zeji = self.get_sigma_zeji(R, control_params)

        control_params = control_params.replace(a_mean=jnp.concatenate([a_mean_old[1:], a_mean_old[-1:]]),
                                                 a_cov=jnp.concatenate([a_cov_old[1:], a_cov_old[-1:]]))
        
        # DEBUG
        control_params = control_params.replace(a_cov=a_cov_zeji)
        # jax.debug.print('a cov zeji {cov}', cov=control_params.a_cov)
         
        # sample action with mean and covariance, repeat for N times to get N samples with shape (N, H, action_dim)
        # a_mean shape (H, action_dim), a_cov shape (H, action_dim, action_dim)
        rng_act, act_key = jax.random.split(rng_act)
        act_keys = jax.random.split(act_key, self.N)

        # DEBUG
        # def single_sample(key, traj_mean, traj_cov):
        #     return jax.vmap(lambda mean, cov: jax.random.multivariate_normal(key, mean, cov))(traj_mean, traj_cov)
        # # repeat single_sample N times to get N samples
        # a_sampled = jax.vmap(single_sample, in_axes=(0, None, None))(act_keys, control_params.a_mean, control_params.a_cov)

        def single_sample(key):
            return jax.random.multivariate_normal(key, control_params.a_mean.flatten(), control_params.a_cov)
        a_sampled_flattened = jax.vmap(single_sample)(act_keys)

        a_sampled = a_sampled_flattened.reshape(self.N, self.H, -1)

        a_sampled = jnp.clip(a_sampled, -1.0, 1.0) # (N, H, action_dim)
        # rollout to get reward with lax.scan
        rng_act, step_key = jax.random.split(rng_act)
        def rollout_fn(carry, action):
            state, params, reward_before, done_before = carry
            obs, state, reward, done, info = jax.vmap(lambda s, a, p: self.env.step_env_wocontroller(step_key, s, a, p))(state, action, params)
            reward = jnp.where(done_before, reward_before, reward)
            return (state, params, reward, done | done_before), (reward, state.pos)
        # repeat state each element to match the sample size N
        state_repeat = jax.tree_map(lambda x: jnp.repeat(jnp.asarray(x)[None, ...], self.N, axis=0), state)
        env_params_repeat = jax.tree_map(lambda x: jnp.repeat(jnp.asarray(x)[None, ...], self.N, axis=0), env_params)
        done_repeat = jnp.full(self.N, False)
        reward_repeat = jnp.full(self.N, 0.0)

        _, (rewards, poses) = lax.scan(rollout_fn, (state_repeat, env_params_repeat, reward_repeat, done_repeat), a_sampled.transpose(1,0,2), length=self.H)
        # get discounted reward sum over horizon (axis=1)
        rewards = rewards.transpose(1,0) # (H, N) -> (N, H)
        discounted_rewards = jnp.sum(rewards * jnp.power(control_params.discount, jnp.arange(self.H)), axis=1, keepdims=False)
        # get cost
        cost = -discounted_rewards

        # get trajectory weight
        cost_exp = jnp.exp(-(cost-jnp.min(cost)) / self.lam)

        weight = cost_exp / jnp.sum(cost_exp)

        # update trajectory mean and covariance with weight
        a_mean = jnp.sum(weight[:, None, None] * a_sampled, axis=0) * control_params.gamma_mean + control_params.a_mean * (1 - control_params.gamma_mean)
        # a_cov = jnp.sum(weight[:, None, None, None] * ((a_sampled - a_mean)[..., None] * (a_sampled - a_mean)[:, :, None, :]), axis=0) * control_params.gamma_sigma + control_params.a_cov * (1 - control_params.gamma_sigma)
        # control_params = control_params.replace(a_mean=a_mean, a_cov=a_cov)
        # DEBUG
        control_params = control_params.replace(a_mean=a_mean)

        # get action
        u = control_params.a_mean[0]

        # debug values
        info = {
            'pos_mean': jnp.mean(poses, axis=1), 
            'pos_std': jnp.std(poses, axis=1)
        }

        return u, control_params, info