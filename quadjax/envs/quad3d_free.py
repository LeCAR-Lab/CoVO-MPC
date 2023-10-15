import jax
import jax.numpy as jnp
from jax import lax
from gymnax.environments import spaces
from typing import Tuple, Optional
import chex
from functools import partial
from dataclasses import dataclass as pydataclass
import tyro
import pickle
import time as time_module
import numpy as np

import quadjax
from quadjax import controllers
from quadjax.dynamics import utils
from quadjax.dynamics import free
from quadjax.dynamics.dataclass import EnvParams3D, EnvState3D, Action3D
from quadjax.envs.base import BaseEnvironment

# for debug purpose
from icecream import install
install()


class Quad3D(BaseEnvironment):
    """
    JAX Compatible version of Quad3D-v0 OpenAI gym environment. Source:
    github.com/openai/gym/blob/master/gym/envs/classic_control/Quad3D.py
    """

    def __init__(self, task: str = "tracking", dynamics: str = 'free', obs_type: str = 'quad', enable_randomizer: bool = True, lower_controller: str = 'base', disturb_type: str='periodic'):
        super().__init__()
        self.task = task
        # reference trajectory function
        if task == "tracking":
            self.generate_traj = partial(utils.generate_lissa_traj, self.default_params.max_steps_in_episode, self.default_params.dt)
            self.reward_fn = utils.tracking_penyaw_reward_fn
        elif task == "tracking_zigzag":
            self.generate_traj = partial(utils.generate_zigzag_traj, self.default_params.max_steps_in_episode, self.default_params.dt)
            self.reward_fn = utils.tracking_penyaw_reward_fn
        elif task in "jumping":
            self.generate_traj = partial(utils.generate_fixed_traj, self.default_params.max_steps_in_episode, self.default_params.dt)
            self.reward_fn = utils.jumping_reward_fn
        elif task == 'hovering':
            self.generate_traj = partial(utils.generate_fixed_traj, self.default_params.max_steps_in_episode, self.default_params.dt)
            self.reward_fn = utils.tracking_reward_fn
        else:
            raise NotImplementedError
        # dynamics function
        if dynamics == 'free':
            self.step_fn, self.dynamics_fn = free.get_free_dynamics_3d()
        elif dynamics == 'dist_constant':
            self.step_fn, self.dynamics_fn = free.get_free_dynamics_3d_disturbance(utils.constant_disturbance)
        elif dynamics == 'bodyrate':
            self.step_fn, self.dynamics_fn = free.get_free_dynamics_3d_bodyrate(disturb_type=disturb_type)
        else:
            raise NotImplementedError
        # lower-level controller
        if lower_controller == 'base':
            self.default_control_params = 0.0
            def base_controller_fn(obs, state, env_params, rng_act, input_action):
                return input_action, None, state
            self.control_fn = base_controller_fn
        elif lower_controller == 'l1':
            self.default_control_params = controllers.L1Params()
            controller = controllers.L1Controller(self, self.default_control_params)
            def l1_control_fn(obs, state, env_params, rng_act, input_action):
                action_l1, control_params, _ = controller(obs, state, env_params, rng_act, state.control_params, 0.0)
                state = state.replace(control_params=control_params)
                return (action_l1 + input_action), None, state
            self.control_fn = l1_control_fn
        elif lower_controller == 'l1_esitimate_only':
            self.default_control_params = controllers.L1Params()
            controller = controllers.L1Controller(self, self.default_control_params)
            def l1_esitimate_only_control_fn(obs, state, env_params, rng_act, input_action):
                _, control_params, _ = controller(obs, state, env_params, rng_act, state.control_params, 0.0)
                state = state.replace(control_params=control_params)
                return input_action, None, state
            self.control_fn = l1_esitimate_only_control_fn
        else:
            raise NotImplementedError
        # sampling function
        if enable_randomizer:
            def sample_random_params(key: chex.PRNGKey) -> EnvParams3D:
                param_key = jax.random.split(key)[0]
                rand_val = jax.random.uniform(param_key, shape=(11,), minval=-1.0, maxval=1.0) # DEBUG * 0.0

                params = self.default_params
                m = params.m_mean + rand_val[0] * params.m_std
                I_diag = params.I_diag_mean + rand_val[1:4] * params.I_diag_std
                I = jnp.diag(I_diag)
                action_scale = params.action_scale_mean + rand_val[4] * params.action_scale_std
                alpha_bodyrate = params.alpha_bodyrate_mean + rand_val[5] * params.alpha_bodyrate_std

                return EnvParams3D(m=m, I=I, action_scale=action_scale, alpha_bodyrate=alpha_bodyrate)
            self.sample_params = sample_random_params
        else:
            self.sample_params = lambda key: self.default_params
        # observation function
        if obs_type == 'quad_params':
            self.get_obs = self.get_obs_quad_params
            self.obs_dim = 28 + self.default_params.traj_obs_len * 6
        elif obs_type == 'quad':
            self.get_obs = self.get_obs_quadonly
            self.obs_dim = 19 + self.default_params.traj_obs_len * 6
        elif obs_type == 'quad_l1':
            assert 'l1' in lower_controller, "quad_l1 obs_type only works with l1 lower controller"
            self.get_obs = self.get_obs_quad_l1
            self.obs_dim = 25 + self.default_params.traj_obs_len * 6
        else:
            raise NotImplementedError
        # equibrium point
        self.equib = jnp.array([0.0]*6+[1.0]+[0.0]*6)
        # RL parameters
        self.action_dim = 4
        self.adapt_obs_dim = 22 * self.default_params.adapt_horizon
        self.param_obs_dim = 9


    '''
    environment properties
    '''
    @property
    def default_params(self) -> EnvParams3D:
        """Default environment parameters for Quad3D-v0."""
        return EnvParams3D()

    def action_space(self, params: Optional[EnvParams3D] = None) -> spaces.Box:
        """Action3D space of the environment."""
        if params is None:
            params = self.default_params
        return spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(4,),
            dtype=jnp.float32,
        )

    def observation_space(self, params: EnvParams3D) -> spaces.Box:
        """Observation space of the environment."""
        # NOTE: use default params for jax limitation
        return spaces.Box(
            -1.0,
            1.0,
            shape=(19 + self.default_params.traj_obs_len * 6 + 12,),
            dtype=jnp.float32,
        )


    '''
    key methods
    '''
    def step_env(
        self,
        key: chex.PRNGKey,
        state: EnvState3D,
        action: jnp.ndarray,
        params: EnvParams3D,
    ) -> Tuple[chex.Array, EnvState3D, float, bool, dict]:
        action = jnp.clip(action, -1.0, 1.0)
        sub_action, _, state = self.control_fn(None, state, params, key, action)
        next_state = self.raw_step(key, state, sub_action, params)
        return self.get_obs_state_reward_done_info(state, next_state, params)

    def step_env_wocontroller(
        self,
        key: chex.PRNGKey,
        state: EnvState3D,
        sub_action: jnp.ndarray,
        params: EnvParams3D,
    ) -> Tuple[chex.Array, EnvState3D, float, bool, dict]:
        return self.step_env(key, state, sub_action, params)
    
    def raw_step(
        self,
        key: chex.PRNGKey,
        state: EnvState3D,
        sub_action: jnp.ndarray,
        params: EnvParams3D,
    ) -> EnvState3D:
        sub_action = jnp.clip(sub_action, -1.0, 1.0)
        thrust = (sub_action[0] + 1.0) / 2.0 * params.max_thrust
        torque = sub_action[1:] * params.max_torque
        env_action = Action3D(thrust=thrust, torque=torque)
        return self.step_fn(params, state, env_action, key)
    
    def get_obs_state_reward_done_info(
        self,
        state: EnvState3D,
        next_state: EnvState3D, 
        params: EnvParams3D,
    ) -> Tuple[chex.Array, EnvState3D, float, bool, dict]:
        reward = self.reward_fn(state, params)
        done = self.is_terminal(state, params)
        return (
            lax.stop_gradient(self.get_obs(next_state, params)),
            lax.stop_gradient(next_state),
            reward,
            done,
            {
                "discount": self.discount(next_state, params),
                "err_pos": jnp.linalg.norm(state.pos_tar - state.pos),
                "err_vel": jnp.linalg.norm(state.vel_tar - state.vel),
                "obs_param": self.get_obs_paramsonly(state, params),
                "obs_adapt": self.get_obs_adapt_hist(state, params),
            },
        )

    def reset_env(
        self, key: chex.PRNGKey, params: EnvParams3D
    ) -> Tuple[chex.Array, EnvState3D]:
        """Reset environment state by sampling theta, theta_dot."""
        traj_key, disturb_key, key = jax.random.split(key, 3)
        # generate reference trajectory by adding a few sinusoids together
        pos_traj, vel_traj, acc_traj = self.generate_traj(traj_key)
        zeros3 = jnp.zeros(3, dtype=jnp.float32)
        vel_hist = jnp.zeros((self.default_params.adapt_horizon+2, 3), dtype=jnp.float32)
        omega_hist = jnp.zeros((self.default_params.adapt_horizon+2, 3), dtype=jnp.float32)
        action_hist = jnp.zeros((self.default_params.adapt_horizon+2, 4), dtype=jnp.float32)
        state = EnvState3D(
            # drone
            pos=zeros3,
            vel=zeros3,
            omega=zeros3,
            omega_tar=zeros3,
            quat=jnp.concatenate([zeros3, jnp.array([1.0])]),
            # object
            pos_obj=zeros3,vel_obj=zeros3,
            # hook
            pos_hook=zeros3,vel_hook=zeros3,
            # rope
            l_rope=0.0,zeta=zeros3,zeta_dot=zeros3,
            f_rope=zeros3,f_rope_norm=0.0,
            # trajectory
            pos_tar=pos_traj[0],vel_tar=vel_traj[0],acc_tar=acc_traj[0],
            pos_traj=pos_traj,vel_traj=vel_traj,acc_traj=acc_traj, 
            # debug value
            last_thrust=0.0,last_torque=zeros3,
            # step
            time=0,
            # disturbance
            f_disturb=jax.random.uniform(disturb_key, shape=(3,), minval=-params.disturb_scale, maxval=params.disturb_scale),
            # trajectory information for adaptation
            vel_hist=vel_hist,omega_hist=omega_hist,action_hist=action_hist,
            # control parameters
            control_params=self.default_control_params,
        )
        info = {
            "discount": self.discount(state, params),
            "err_pos": jnp.linalg.norm(state.pos_tar - state.pos),
            "err_vel": jnp.linalg.norm(state.vel_tar - state.vel),
            "obs_param": self.get_obs_paramsonly(state, params),
            "obs_adapt": self.get_obs_adapt_hist(state, params),
        }
        return self.get_obs(state, params), info, state
    
    @partial(jax.jit, static_argnums=(0,))
    def get_obs_quadonly(self, state: EnvState3D, params: EnvParams3D) -> chex.Array:
        """Return angle in polar coordinates and change."""
        # future trajectory observation
        traj_obs_len = self.default_params.traj_obs_len
        traj_obs_gap = self.default_params.traj_obs_gap
        # Generate the indices
        indices = state.time + 1 + jnp.arange(traj_obs_len) * traj_obs_gap
        obs_elements = [
            # drone
            state.pos,
            state.vel,
            state.quat,
            state.omega,  # 3*3+4=13
            # trajectory
            state.pos_tar,
            state.vel_tar,  # 3*2=6
            state.pos_traj[indices].flatten(), 
            state.vel_traj[indices].flatten(), 
        ]  # 13+6=19
        obs = jnp.concatenate(obs_elements, axis=-1)

        return obs

    @partial(jax.jit, static_argnums=(0,))
    def get_obs_adapt_hist(self, state: EnvState3D, params: EnvParams3D) -> chex.Array:
        vel_hist = state.vel_hist
        omega_hist = state.omega_hist
        action_hist = state.action_hist

        dvel_hist = jnp.diff(vel_hist, axis=0)
        ddvel_hist = jnp.diff(dvel_hist, axis=0)
        domega_hist = jnp.diff(omega_hist, axis=0)
        ddomega_hist = jnp.diff(domega_hist, axis=0)

        horizon = self.default_params.adapt_horizon
        obs_elements = [
            vel_hist[-horizon:].flatten(),
            omega_hist[-horizon:].flatten(),
            action_hist[-horizon:].flatten(),
            dvel_hist[-horizon:].flatten(),
            ddvel_hist[-horizon:].flatten(),
            domega_hist[-horizon:].flatten(),
            ddomega_hist[-horizon:].flatten(),
        ]

        obs = jnp.concatenate(obs_elements, axis=-1)

        return obs

    
    @partial(jax.jit, static_argnums=(0,))
    def get_obs_paramsonly(self, state: EnvState3D, params: EnvParams3D) -> chex.Array:
        obs_elements = [
            # parameter observation
            # I
            (params.I.diagonal()- params.I_diag_mean)/params.I_diag_std,
            # disturbance
            (state.f_disturb)/params.disturb_scale,
            jnp.array(
                [
                    # mass
                    (params.m - params.m_mean)/params.m_std,
                    # action_scale
                    (params.action_scale - params.action_scale_mean)/params.action_scale_std,
                    # 1st order alpha
                    (params.alpha_bodyrate - params.alpha_bodyrate_mean)/params.alpha_bodyrate_std,
                ]
            )
        ]  # 13+6=19
        obs = jnp.concatenate(obs_elements, axis=-1)
        return obs
    
    @partial(jax.jit, static_argnums=(0,))
    def get_obs_l1only(self, state: EnvState3D, params: EnvParams3D) -> chex.Array:
        obs_elements = [
            # l1 observation
            state.control_params.vel_hat,
            state.control_params.d_hat,
        ]
        obs = jnp.concatenate(obs_elements, axis=-1)
        return obs
    
    @partial(jax.jit, static_argnums=(0,))
    def get_obs_quad_params(self, state: EnvState3D, params: EnvParams3D) -> chex.Array:
        quad_obs = self.get_obs_quadonly(state, params)
        param_obs = self.get_obs_paramsonly(state, params)
        return jnp.concatenate([quad_obs, param_obs], axis=-1)
    
    @partial(jax.jit, static_argnums=(0,))
    def get_obs_quad_l1(self, state: EnvState3D, params: EnvParams3D) -> chex.Array:
        quad_obs = self.get_obs_quadonly(state, params)
        l1_obs = self.get_obs_l1only(state, params)
        return jnp.concatenate([quad_obs, l1_obs], axis=-1)

    @partial(jax.jit, static_argnums=(0,))
    def is_terminal(self, state: EnvState3D, params: EnvParams3D) -> bool:
        """Check whether state is terminal."""
        # Check number of steps in episode termination condition
        done = (state.time >= params.max_steps_in_episode) \
            | (jnp.abs(state.pos) > 3.0).any() \
            | (jnp.abs(state.pos_obj) > 3.0).any() \
            | (jnp.abs(state.omega) > 100.0).any()
        return done

def eval_env(env: Quad3D, controller, control_params, total_steps = 30000, filename = ''):
    # running environment
    rng = jax.random.PRNGKey(1)
    rng, rng_params = jax.random.split(rng)
    env_params = env.sample_params(rng_params)

    rng, rng_reset = jax.random.split(rng)
    obs, info, env_state = env.reset(rng_reset, env_params)

    control_params = controller.update_params(env_params, control_params)

    def run_one_step(carry, _):
        obs, env_state, rng, env_params, control_params, info = carry
        rng, rng_act, rng_step, rng_control = jax.random.split(rng, 4)
        action, control_params, control_info = controller(obs, env_state, env_params, rng_act, control_params, info)
        if control_info is not None:
            if 'a_mean' in control_info:
                action = control_info['a_mean']
        next_obs, next_env_state, reward, done, info = env.step(
            rng_step, env_state, action, env_params)
        # if done, reset controller parameters, aviod use if, use lax.cond instead
        new_control_params = controller.reset()
        control_params = lax.cond(done, lambda x: new_control_params, lambda x: x, control_params)
        return (next_obs, next_env_state, rng, env_params, control_params, info), info['err_pos']
    
    t0 = time_module.time()
    (obs, env_state, rng, env_params, control_params, info), err_pos = lax.scan(
        run_one_step, (obs, env_state, rng, env_params, control_params, info), jnp.arange(total_steps))
    print(f"env running time: {time_module.time()-t0:.2f}s")

    print(f"mean tracking error: {jnp.mean(err_pos):.2f}")

    # save data
    with open(f"{quadjax.get_package_path()}/../results/eval_err_pos_{filename}.pkl", "wb") as f:
        pickle.dump(np.array(err_pos), f)

def render_env(env: Quad3D, controller, control_params, repeat_times = 1, filename = ''):
    # running environment
    rng = jax.random.PRNGKey(1)
    rng, rng_params = jax.random.split(rng)
    env_params = env.sample_params(rng_params)
    # env_params = env.default_params # DEBUG

    state_seq, obs_seq, reward_seq = [], [], []
    control_seq = []
    rng, rng_reset = jax.random.split(rng)
    obs, info, env_state = env.reset(rng_reset, env_params)

    # DEBUG set iniiial state here
    # env_state = env_state.replace(quat = jnp.array([jnp.sin(jnp.pi/4), 0.0, 0.0, jnp.cos(jnp.pi/4)]))
                                  
    control_params = controller.update_params(env_params, control_params)
    n_dones = 0

    t0 = time_module.time()
    while n_dones < repeat_times:
        state_seq.append(env_state)
        rng, rng_act, rng_step = jax.random.split(rng, 3)
        action, control_params, control_info = controller(obs, env_state, env_params, rng_act, control_params, info)
        # manually record certain control parameters into state_seq
        if hasattr(control_params, 'd_hat') and hasattr(control_params, 'vel_hat'):
            control_seq.append({'d_hat': control_params.d_hat, 'vel_hat': control_params.vel_hat})
        next_obs, next_env_state, reward, done, info = env.step(
            rng_step, env_state, action, env_params)
        if done:
            rng, rng_params = jax.random.split(rng)
            env_params = env.sample_params(rng_params)
            control_params = controller.update_params(env_params, control_params)
            n_dones += 1

        reward_seq.append(reward)
        obs_seq.append(obs)
        obs = next_obs
        env_state = next_env_state
    print(f"env running time: {time_module.time()-t0:.2f}s")

    t0 = time_module.time()
    # convert state into dict
    state_seq_dict = [s.__dict__ for s in state_seq]
    if len(control_seq) > 0:
        # merge control_seq into state_seq with dict
        for i in range(len(state_seq)):
            state_seq_dict[i] = {**state_seq_dict[i], **control_seq[i]}
    utils.plot_states(state_seq_dict, obs_seq, reward_seq, env_params, filename)
    print(f"plotting time: {time_module.time()-t0:.2f}s")

    # save state_seq (which is a list of EnvState3D:flax.struct.dataclass)
    # get package quadjax path
    
    with open(f"{quadjax.get_package_path()}/../results/state_seq_{filename}.pkl", "wb") as f:
        pickle.dump(state_seq, f)

'''
reward function here. 
'''

@pydataclass
class Args:
    task: str = "hovering" # tracking, tracking_zigzag, hovering
    dynamics: str = 'free'
    controller: str = 'lqr' # fixed
    controller_params: str = ''
    obs_type: str = 'quad'
    debug: bool = False
    mode: str = 'render' # eval, render
    lower_controller: str = 'base'
    noDR: bool = False
    disturb_type: str = 'periodic' # periodic, sin, drag
    name: str = ''

def main(args: Args):
    env = Quad3D(task=args.task, dynamics=args.dynamics, obs_type=args.obs_type, lower_controller=args.lower_controller, enable_randomizer=not args.noDR, disturb_type=args.disturb_type)

    print("starting test...")
    # enable NaN value detection
    # from jax import config
    # config.update("jax_debug_nans", True)
    # with jax.disable_jit():
    if args.controller == 'lqr':
        control_params = controllers.LQRParams(
            Q = jnp.diag(jnp.ones(12)),
            R = 0.03 * jnp.diag(jnp.ones(4)),
            K = jnp.zeros((4, 12)),
        )
        controller = controllers.LQRController(env, control_params = control_params)
    elif args.controller == 'fixed':
        control_params = controllers.FixedParams(
            u = jnp.asarray([0.0, 0.0, 0.0, 0.0]),
        )
        controller = controllers.FixedController(env, control_params = control_params)
    elif args.controller == 'mppi':
        sigma = 0.1
        if args.controller_params == '':
            N = 8192
            H = 64
            lam = 3e-3
        else:
            # parse in format "N{sample_number}_H{horizon}_sigma{sigma}_lam{lam}"
            N = int(args.controller_params.split('_')[0][1:])
            H = int(args.controller_params.split('_')[1][1:])
            lam = float(args.controller_params.split('_')[2][3:])
            print(f'[DEBUG], set controller parameters to be: N={N}, H={H}, lam={lam}')
        if args.debug:
            N = 8
            print(f'[DEBUG], override controller parameters to be: N={N}')
        thrust_hover = env.default_params.m * env.default_params.g
        thrust_hover_normed = (thrust_hover / env.default_params.max_thrust) * 2.0 - 1.0
        a_mean_per_step = jnp.array([thrust_hover_normed, 0.0, 0.0, 0.0]) 
        a_mean = jnp.tile(a_mean_per_step, (H, 1))
        a_cov_per_step = jnp.diag(jnp.array([sigma**2]*env.action_dim))
        a_cov = jnp.tile(a_cov_per_step, (H, 1, 1))
        control_params = controllers.MPPIParams(
            gamma_mean = 1.0,
            gamma_sigma = 0.01,
            discount = 0.9,
            sample_sigma = sigma,
            a_mean = a_mean,
            a_cov = a_cov,
        )
        controller = controllers.MPPIController(env=env, control_params=control_params, N=N, H=H, lam=lam)
    elif args.controller == 'nn':
        from quadjax.train import ActorCritic
        network = ActorCritic(env.action_dim, activation='tanh')
        if args.controller_params == '':
            file_path = "ppo_params_"
        else:
            file_path = f'{args.controller_params}'
        control_params = pickle.load(open(f"{quadjax.get_package_path()}/../results/{file_path}.pkl", "rb"))
        def apply_fn(train_params, last_obs, env_info):
            return network.apply(train_params, last_obs)
        controller = controllers.NetworkController(apply_fn, env, control_params)
    elif args.controller == 'RMA':
        from quadjax.train import ActorCritic, Compressor, Adaptor
        network = ActorCritic(env.action_dim, activation='tanh')
        compressor = Compressor()
        adaptor = Adaptor()
        if args.controller_params == '':
            file_path = "ppo_params_"
        else:
            file_path = f'{args.controller_params}'
        control_params = pickle.load(open(f"{quadjax.get_package_path()}/../results/{file_path}.pkl", "rb"))
        def apply_fn(train_params, last_obs, env_info):
            adapted_last_obs = adaptor.apply(train_params[2], env_info['obs_adapt'])
            obs = jnp.concatenate([last_obs, adapted_last_obs], axis=-1)
            pi, value = network.apply(train_params[0], obs)
            return pi, value
        controller = controllers.NetworkController(apply_fn, env, control_params)
    elif args.controller == 'RMA-expert':
        from quadjax.train import ActorCritic, Compressor, Adaptor
        network = ActorCritic(env.action_dim, activation='tanh')
        compressor = Compressor()
        adaptor = Adaptor()
        if args.controller_params == '':
            file_path = "ppo_params_"
        else:
            file_path = f'{args.controller_params}'
        control_params = pickle.load(open(f"{quadjax.get_package_path()}/../results/{file_path}.pkl", "rb"))
        def apply_fn(train_params, last_obs, env_info):
            compressed_last_obs = compressor.apply(train_params[1], env_info['obs_param'])
            adapted_last_obs = adaptor.apply(train_params[2], env_info['obs_adapt'])
            jax.debug.print('compressed {com}', com = compressed_last_obs)
            jax.debug.print('adapted {ada}', ada = adapted_last_obs)
            obs = jnp.concatenate([last_obs, compressed_last_obs], axis=-1)
            pi, value = network.apply(train_params[0], obs)
            return pi, value
        controller = controllers.NetworkController(apply_fn, env, control_params)
    elif args.controller == 'l1':
        control_params = controllers.L1Params()
        controller = controllers.L1Controller(env, control_params)
    else:
        raise NotImplementedError
    if args.mode == 'eval':
        eval_env(env, controller=controller, control_params=control_params, total_steps=30000, filename=args.name)
    elif args.mode == 'render':
        render_env(env, controller=controller, control_params=control_params, repeat_times=1, filename=args.name)
    else:
        raise NotImplementedError


if __name__ == "__main__":
    main(tyro.cli(Args))