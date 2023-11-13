import jax
import chex
from flax import struct
from functools import partial
from jax import lax
from jax import numpy as jnp
import numpy as np
import pickle
from jaxopt import ProjectedGradient
from jaxopt.projection import projection_hyperplane

from quadjax import controllers
from quadjax.dynamics import EnvParams2D, EnvState2D, geom
from quadjax.train import ActorCritic

@struct.dataclass
class MPPIZejiParamsLegacy:
    gamma_mean: float # mean of gamma
    gamma_sigma: float # std of gamma
    discount: float # discount factor
    sample_sigma: float # std of sampling

    a_mean: jnp.ndarray # mean of action
    a_cov: jnp.ndarray # covariance matrix of action
    a_cov_offline: jnp.ndarray # covariance matrix of action

class MPPIZejiControllerLegacy(controllers.BaseController):
    def __init__(self, env, control_params, N: int, H: int, lam: float, expansion_mode:str = 'lqr') -> None:
        super().__init__(env, control_params)
        self.N = N # NOTE: N is the number of saples, set here as a static number
        self.H = H
        self.lam = lam
        if expansion_mode == 'mean':
            # Key method
            def get_sigma_zeji(control_params, env_state, env_params, key):
                R = self.get_dJ_du(env_state, env_params, control_params, control_params.a_mean, key)
                sigma = self.get_sigma_from_R(R, control_params)
                return sigma
            self.get_sigma_zeji = get_sigma_zeji
        elif expansion_mode == 'repeat':
            # NOTE: now for 2D only
            assert env.action_dim == 2, 'repeat only works for 2D'
            def get_AB(env_state, env_params):
                dt = env_params.dt
                th = env_state.roll
                m = env_params.m
                I = env_params.I
                A = jnp.array([
                    [1.0, 0.0, dt, 0.0, 0.0, 0.0], 
                    [0.0, 1.0, 0.0, dt, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 1.0, dt],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                ])
                B = jnp.array([
                    [0.0, 0.0],
                    [0.0, 0.0],
                    [-dt/m*jnp.sin(th), 0.0],
                    [0.0, dt/m*jnp.cos(th)],
                    [0.0, 0.0],
                    [0.0, 1/I*dt],
                ])
                return A, B
            def get_M(A, B):
                # M is in the following form:
                # [0, 0, ..., 0], 
                # [B, 0, ..., 0],
                # [AB, B, 0, ..., 0],
                # [A^2B, AB, B, 0, ..., 0],
                # ...
                # [A^(H-1)B, A^(H-2)B, ..., AB, B]
                state_dim = 6
                M = jnp.zeros((self.H*state_dim, self.H*env.action_dim))
                for i in range(self.H):
                    for j in range(i+1):
                        M = M.at[i*state_dim:(i+1)*state_dim, j*env.action_dim:(j+1)*env.action_dim].set(jnp.linalg.matrix_power(A, i-j) @ B)
                return M
            def get_R(M):
                Q0 = jnp.diag(jnp.array([50.0, 50.0, 0.1, 0.1, 0.0, 0.0]))
                # repeat Q for H times in the diagonal
                state_dim = 6
                Q = jnp.zeros((self.H*state_dim, self.H*state_dim))
                for i in range(self.H):
                    Q = Q.at[i*state_dim:(i+1)*state_dim, i*state_dim:(i+1)*state_dim].set(Q0)
                R0 = jnp.diag(jnp.array([0.1, 0.02]))
                # repeat R for H times in the diagonal
                R = jnp.zeros((self.H*env.action_dim, self.H*env.action_dim))
                for i in range(self.H):
                    R = R.at[i*env.action_dim:(i+1)*env.action_dim, i*env.action_dim:(i+1)*env.action_dim].set(R0)
                return M.T @ Q @ M + R
            def get_sigma_zeji(control_params, env_state, env_params, key):
                A, B = get_AB(env_state, env_params)
                M = get_M(A, B)
                R = get_R(M)
                sigma = self.get_sigma_from_R(R, control_params)
                return sigma
            self.get_sigma_zeji = get_sigma_zeji
        elif expansion_mode in ['lqr', 'zero', 'ppo', 'mppi']:
            if expansion_mode == 'lqr':
                expansion_control_params = controllers.LQRParams(
                    Q = jnp.diag(jnp.ones(5)),
                    R = 0.03 * jnp.diag(jnp.ones(2)),
                    K = jnp.zeros((2, 5)),
                )
                expansion_controller = controllers.LQRController2D(env, expansion_control_params)
            elif expansion_mode == 'zero':
                # m = self.env.default_params.m
                # g = self.env.default_params.g
                # max_thrust = self.env.default_params.max_thrust
                # expansion_control_params = controllers.FixedParams(
                #     u = jnp.array([m*g/max_thrust * 2.0 - 1.0, 0.0]))
                expansion_control_params = controllers.FixedParams(
                    u = control_params.a_mean[0])
                expansion_controller = controllers.FixedController(env, expansion_control_params)
            elif expansion_mode == 'ppo':
                import quadjax
                from quadjax.train import ActorCritic
                network = ActorCritic(env.action_dim, activation='tanh')
                expansion_control_params = pickle.load(open(f"{quadjax.get_package_path()}/../results/ppo_params.pkl", "rb"))
                def apply_fn(train_params, last_obs, env_info):
                    return network.apply(train_params, last_obs)
                expansion_controller = controllers.NetworkController(apply_fn, env, expansion_control_params)
            elif expansion_mode == 'mppi':
                sigma = 0.5
                thrust_hover = env.default_params.m * env.default_params.g
                thrust_hover_normed = (thrust_hover / env.default_params.max_thrust) * 2.0 - 1.0
                a_mean_per_step = jnp.array([thrust_hover_normed, 0.0]) 
                a_mean = jnp.tile(a_mean_per_step, (H, 1))
                a_cov_per_step = jnp.diag(jnp.array([sigma**2, sigma**2]))
                a_cov = jnp.tile(a_cov_per_step, (H, 1, 1))
                expansion_control_params = controllers.MPPIParams(
                    gamma_mean = 1.0,
                    gamma_sigma = 0.0,
                    discount = 1.0,
                    sample_sigma = sigma, 
                    a_mean = a_mean,
                    a_cov = a_cov,
                )
                expansion_controller = controllers.MPPIController(env, expansion_control_params, N=self.N, H=self.H, lam=self.lam)
            else:
                raise NotImplementedError
            def mppi_rollout_fn(carry, unused):
                env_state, env_params, key = carry
                rng_act, key = jax.random.split(key)
                obs = self.env.get_obs(env_state, env_params)
                action, _, _ = expansion_controller(obs, env_state, env_params, rng_act, expansion_control_params)
                action = lax.stop_gradient(action)
                rng_step, key = jax.random.split(key)
                _, env_state, _, _, _ = self.env.step_env_wocontroller_gradient(rng_step, env_state, action, env_params)
                return (env_state, env_params, key), action
            def get_single_a_cov_offline(carry, unused):
                env_state, env_params, key = carry
                _, a_mean = lax.scan(mppi_rollout_fn, (env_state, env_params, key), None, length=self.H)
                R = self.get_dJ_du(env_state, env_params, control_params, a_mean, key)
                a_cov = self.get_sigma_from_R(R, control_params)
                # step forward with lqr
                rng_step, key = jax.random.split(key)
                obs = self.env.get_obs(env_state, env_params)
                action, _, _ = expansion_controller(obs, env_state, env_params, rng_step, expansion_control_params)
                action = lax.stop_gradient(action)
                rng_step, key = jax.random.split(key)
                _, env_state, _, _, _ = self.env.step_env_wocontroller(rng_step, env_state, action, env_params)
                return (env_state, env_params, key), a_cov
            if expansion_mode in ['lqr', 'ppo', 'mppi']:
                def get_a_cov_offline(env_state, env_params, key):
                    _, a_cov_offline = lax.scan(get_single_a_cov_offline, (env_state, env_params, key), None, length=self.H)
                    # a_cov_offline_mean = jnp.mean(a_cov_offline, axis=0)
                    # a_cov_offline = jnp.repeat(a_cov_offline_mean[None, ...], self.H, axis=0)
                    return a_cov_offline
            elif expansion_mode == 'zero':
                def get_a_cov_offline(env_state, env_params, key):
                    _, a_cov = get_single_a_cov_offline((env_state, env_params, key), None)
                    a_cov_offline = jnp.repeat(a_cov[None, ...], self.H, axis=0)
                    return a_cov_offline
            else:
                raise NotImplementedError
            def reset_a_cov_offline(env_state, env_params, control_params, key):
                a_cov_offline = get_a_cov_offline(env_state, env_params, key)
                control_params = control_params.replace(a_cov_offline=a_cov_offline)
                return control_params

            # Key method
            def get_sigma_zeji(control_params, env_state, env_params, key):
                return control_params.a_cov_offline[env_state.time]
            self.get_sigma_zeji = get_sigma_zeji
            # overwrite reset function
            self.reset = reset_a_cov_offline
        else:
            raise NotImplementedError
        # network = ActorCritic(2, activation='tanh')
        # self.apply_fn = network.apply
        # with open('/home/pcy/Research/quadjax/results/ppo_params_quad2d_free_tracking_zigzag_base.pkl', 'rb') as f:
        #     self.network_params = pickle.load(f)

    def get_sigma_from_R(self, R: jnp.ndarray, control_params: MPPIZejiParamsLegacy):
        R = (R + R.T)/2.0
        # print('R eign', np.min(np.linalg.eigvals(R)))
        # exit()
        # jax.debug.print('R eign jax {e}', e=jnp.linalg.eigh(R))
        eigns, u = jnp.linalg.eigh(R)

        eigns = eigns - jnp.min(eigns) + 1e-2

        log_o = jnp.log(eigns)
        log_const = (2 * 4 * self.H * jnp.log(control_params.sample_sigma) + jnp.sum(log_o)) / (2*self.H)
        log_s = 0.5 * log_const - 0.5 * log_o

        # o = eigns
        # s = jnp.exp(log_s)
        # cc = ((o**2)*s)/((1+2/lam*o*s)**3)
        # jax.debug.print('verified solution: log cc {cc}', cc=jnp.log(cc))
        # jax.debug.print('log const {log_const}', log_const=log_const)
        # jax.debug.print('approximated log cc {log_cc}', log_cc=2.0*log_s + 1.0 * log_o)
                        
        a_cov = u @ jnp.diag(jnp.exp(log_s)) @ u.T

        return (a_cov + a_cov.T) / 2.0 # make it symmetric

    def get_dJ_du(self, env_state:EnvState2D, env_params:EnvParams2D, control_params:MPPIZejiParamsLegacy, a_mean:jnp.ndarray, rng_act: chex.PRNGKey = None):
        def single_rollout_fn(carry, action):
            env_state, env_params, reward_before, done_before, key, cumulated_reward = carry
            rng_act, key = jax.random.split(key)
            _, env_state, reward, done, _ = self.env.step_env_wocontroller_gradient(rng_act, env_state, action, env_params)
            # reward = jnp.where(done_before, reward_before, reward)
            cumulated_reward = cumulated_reward + reward
            return (env_state, env_params, reward, done | done_before, key, cumulated_reward), reward
        def get_cumulated_cost(a_mean_flattened, carry):
            a_mean = a_mean_flattened.reshape(self.H, -1)
            env_state, env_params, rng_act = carry
            # (_, _, _, _, rng_act, cumulated_reward), rewards = lax.scan(single_rollout_fn, (env_state, env_params, 0.0, False, rng_act, 0.0), a_mean[:1])#self.H)
            cumulated_reward = 0.0
            carry = (env_state, env_params, 0.0, False, rng_act, 0.0)
            for i in range(self.H):
                carry, reward = single_rollout_fn(carry, a_mean[i])
                cumulated_reward = cumulated_reward + reward
            env_state = carry[0]
            cumulated_reward = cumulated_reward + self.env.reward_fn(env_state)
            # carry = (env_state, env_params, 0.0, False, rng_act, 0.0)
            # (_, _, _, _, rng_act, cumulated_reward), _ = single_rollout_fn(carry, a_mean[0])
            return -cumulated_reward
        # calculate the hessian of get_cumulated_reward
        jabobian_fn = jax.jacfwd(get_cumulated_cost, argnums=0)
        hessian_fn = jax.jacfwd(jabobian_fn, argnums=0)
        # jax.debug.print('H={H}', H=hessian_fn(a_mean.flatten(), (env_state, env_params, rng_act)))
        return hessian_fn(a_mean.flatten(), (env_state, env_params, rng_act))
    
    # def get_sigma_from_R(self, R: jnp.ndarray, control_params: MPPIZejiParams):
    #     u, s, vh = jnp.linalg.svd(R)

    #     # Objective function
    #     def objective(env_params):
    #         return jnp.sum(s/(1+2/self.lam*s*jnp.exp(env_params))**2)

    #     # Constraint: x1 + x2 = log(det(Sigma)), represented as a hyperplane
    #     def projection_fn(env_params, hyperparams_proj):
    #         return projection_hyperplane(env_params, hyperparams=(hyperparams_proj, self.H*2*2*jnp.log(control_params.sample_sigma)))

    #     # Initialize 'ProjectedGradient' solver
    #     solver = ProjectedGradient(fun=objective, projection=projection_fn)

    #     # Initial parameters
    #     params_init = jnp.zeros(self.H*2)

    #     # Define the optimization problem
    #     sol = solver.run(params_init, hyperparams_proj=jnp.ones(self.H*2))

    #     sigma_eign = jnp.exp(sol.params)

    #     # verify the solution
    #     # oo = s
    #     # ss = jnp.exp(sol.params)
    #     # cc = ((oo**2)*ss)/((1+2/self.lam*oo*ss)**3)
    #     # jax.debug.print('R {R}', R=R)
    #     # jax.debug.print('oo {oo}', oo=oo)
    #     # jax.debug.print('ss {ss}', ss=ss)
    #     # jax.debug.print('cc {cc}', cc=cc)

    #     return u @ jnp.diag(sigma_eign) @ vh

    @partial(jax.jit, static_argnums=(0,))
    def __call__(self, obs:jnp.ndarray, env_state, env_params, rng_act: chex.PRNGKey, control_params: MPPIZejiParamsLegacy, info = None) -> jnp.ndarray:
        # shift operator
        a_mean_old = control_params.a_mean
        a_cov_old = control_params.a_cov


        a_mean = jnp.concatenate([a_mean_old[1:], a_mean_old[-1:]])
        # a_cov = jnp.concatenate([a_cov_old[1:], a_cov_old[-1:]])
        control_params = control_params.replace(a_mean=a_mean)
        
        # DEBUG
        a_cov_zeji = self.get_sigma_zeji(control_params, env_state, env_params, rng_act)

        # save a_cov_offline as a heatmap
        # import matplotlib.pyplot as plt
        # import seaborn as sns
        # for i in range(self.H):
        #     plt.figure(figsize=(10, 10))
        #     sns.heatmap(a_cov_zeji)
        #     plt.savefig(f'../../results/a_cov_{env_state.time}.png')
        # if env_state.time == 15:
        #     exit()

        control_params = control_params.replace(a_cov=a_cov_zeji)
        # jax.debug.print('a cov zeji {cov}', cov=a_cov_zeji)
        # jax.debug.print('a cov sym diff {x}', x = a_cov_zeji - a_cov_zeji.T)
        # jax.debug.print('a cov svd {s}', s=jnp.linalg.svd(a_cov_zeji)[1])
        # eigenvalues of a_cov_zeji
        # print(np.linalg.eigvals(a_cov_zeji))
        # print(np.linalg.cholesky(a_cov_zeji))
        # jax.debug.print('a cov cholesky {L}', L=jnp.linalg.cholesky(a_cov_zeji))

        # sample action with mean and covariance, repeat for N times to get N samples with shape (N, H, action_dim)
        # a_mean shape (H, action_dim), a_cov shape (H, action_dim, action_dim)
        rng_act, act_key = jax.random.split(rng_act)
        act_keys = jax.random.split(act_key, self.N)

        def single_sample(key):
            return jax.random.multivariate_normal(key, control_params.a_mean.flatten(), control_params.a_cov)
        a_sampled_flattened = jax.vmap(single_sample)(act_keys)
        a_sampled = a_sampled_flattened.reshape(self.N, self.H, -1)

        a_sampled = jnp.clip(a_sampled, -1.0, 1.0) # (N, H, action_dim)
        # rollout to get reward with lax.scan
        rng_act, step_key = jax.random.split(rng_act)
        def rollout_fn(carry, action):
            env_state, env_params, reward_before, done_before = carry
            obs, env_state, reward, done, info = jax.vmap(lambda s, a, p: self.env.step_env_wocontroller(step_key, s, a, p))(env_state, action, env_params)
            reward = jnp.where(done_before, reward_before, reward)
            return (env_state, env_params, reward, done | done_before), (reward, env_state.pos)
        # repeat env_state each element to match the sample size N
        state_repeat = jax.tree_map(lambda x: jnp.repeat(jnp.asarray(x)[None, ...], self.N, axis=0), env_state)
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