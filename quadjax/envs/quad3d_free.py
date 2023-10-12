import jax
import jax.numpy as jnp
from jax import lax
from gymnax.environments import environment, spaces
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

# for debug purpose
from icecream import install
install()


class Quad3D(environment.Environment):
    """
    JAX Compatible version of Quad3D-v0 OpenAI gym environment. Source:
    github.com/openai/gym/blob/master/gym/envs/classic_control/Quad3D.py
    """

    def __init__(self, task: str = "tracking", dynamics: str = 'free'):
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
            self.get_obs = self.get_obs_quadonly
        elif dynamics == 'dist_constant':
            self.step_fn, self.dynamics_fn = free.get_free_dynamics_3d_disturbance(utils.constant_disturbance)
            self.get_obs = self.get_obs_quadonly
        elif dynamics == 'bodyrate':
            self.step_fn, self.dynamics_fn = free.get_free_dynamics_3d_bodyrate()
            self.get_obs = self.get_obs_quadonly
        else:
            raise NotImplementedError
        # equibrium point
        self.equib = jnp.array([0.0]*6+[1.0]+[0.0]*6)
        # RL parameters
        self.action_dim = 4
        self.obs_dim = 28 + self.default_params.traj_obs_len * 6


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
        thrust = (action[0] + 1.0) / 2.0 * params.max_thrust
        torque = action[1:4] * params.max_torque
        env_action = Action3D(thrust=thrust, torque=torque)

        reward = self.reward_fn(state)

        next_state = self.step_fn(params, state, env_action, key)

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
            },
        )

    def reset_env(
        self, key: chex.PRNGKey, params: EnvParams3D
    ) -> Tuple[chex.Array, EnvState3D]:
        """Reset environment state by sampling theta, theta_dot."""
        traj_key, disturb_key, key = jax.random.split(key, 3)
        # generate reference trajectory by adding a few sinusoids together
        pos_traj, vel_traj = self.generate_traj(traj_key)
        zeros3 = jnp.zeros(3)
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
            pos_tar=pos_traj[0],vel_tar=vel_traj[0],
            pos_traj=pos_traj,vel_traj=vel_traj,
            # debug value
            last_thrust=0.0,last_torque=zeros3,
            # step
            time=0,
            # disturbance
            f_disturb=jax.random.uniform(disturb_key, shape=(3,), minval=-params.disturb_scale, maxval=params.disturb_scale),
        )
        return self.get_obs(state, params), state

    @partial(jax.jit, static_argnums=(0,))
    def sample_params(self, key: chex.PRNGKey) -> EnvParams3D:
        """Sample environment parameters."""

        param_key = jax.random.split(key)[0]
        rand_val = jax.random.uniform(param_key, shape=(11,), minval=-1.0, maxval=1.0)

        params = self.default_params
        m = params.m_mean + rand_val[0] * params.m_std
        I_diag = params.I_diag_mean + rand_val[1:4] * params.I_diag_std
        I = jnp.diag(I_diag)
        action_scale = params.action_scale_mean + rand_val[4] * params.action_scale_std
        alpha_bodyrate = params.alpha_bodyrate_mean + rand_val[5] * params.alpha_bodyrate_std

        return EnvParams3D(m=m, I=I, action_scale=action_scale, alpha_bodyrate=alpha_bodyrate)
    
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
    env_params = env.default_params

    rng, rng_reset = jax.random.split(rng)
    obs, env_state = env.reset(rng_reset, env_params)

    control_params = controller.update_params(env_params, control_params)

    def run_one_step(carry, _):
        obs, env_state, rng, env_params, control_params = carry
        rng, rng_act, rng_step, rng_control = jax.random.split(rng, 4)
        action, control_params, control_info = controller(obs, env_state, env_params, rng_act, control_params)
        if control_info is not None:
            if 'a_mean' in control_info:
                action = control_info['a_mean']
        next_obs, next_env_state, reward, done, info = env.step(
            rng_step, env_state, action, env_params)
        # if done, reset controller parameters, aviod use if, use lax.cond instead
        new_control_params = controller.reset()
        control_params = lax.cond(done, lambda x: new_control_params, lambda x: x, control_params)
        return (next_obs, next_env_state, rng, env_params, control_params), info['err_pos']
    
    t0 = time_module.time()
    (obs, env_state, rng, env_params, control_params), err_pos = lax.scan(
        run_one_step, (obs, env_state, rng, env_params, control_params), jnp.arange(total_steps))
    print(f"env running time: {time_module.time()-t0:.2f}s")

    # save data
    with open(f"{quadjax.get_package_path()}/../results/eval_err_pos_{filename}.pkl", "wb") as f:
        pickle.dump(np.array(err_pos), f)

def render_env(env: Quad3D, controller, control_params, repeat_times = 1, filename = ''):
    # running environment
    rng = jax.random.PRNGKey(1)
    rng, rng_params = jax.random.split(rng)
    env_params = env.sample_params(rng_params)
    env_params = env.default_params # DEBUG

    state_seq, obs_seq, reward_seq = [], [], []
    rng, rng_reset = jax.random.split(rng)
    obs, env_state = env.reset(rng_reset, env_params)

    # DEBUG set iniiial state here
    # env_state = env_state.replace(quat = jnp.array([jnp.sin(jnp.pi/4), 0.0, 0.0, jnp.cos(jnp.pi/4)]))
                                  
    control_params = controller.update_params(env_params, control_params)
    n_dones = 0

    t0 = time_module.time()
    while n_dones < repeat_times:
        state_seq.append(env_state)
        rng, rng_act, rng_step = jax.random.split(rng, 3)
        action, control_params, control_info = controller(obs, env_state, env_params, rng_act, control_params)
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
    utils.plot_states(state_seq, obs_seq, reward_seq, env_params)
    print(f"plotting time: {time_module.time()-t0:.2f}s")

    # save state_seq (which is a list of EnvState3D:flax.struct.dataclass)
    # get package quadjax path
    
    with open(f"{quadjax.get_package_path()}/../results/state_seq.pkl", "wb") as f:
        pickle.dump(state_seq, f)

'''
reward function here. 
'''

@pydataclass
class Args:
    task: str = "hovering" # tracking, tracking_zigzag, hovering
    dynamics: str = 'free'
    controller: str = 'lqr' # fixed

def main(args: Args):
    env = Quad3D(task=args.task, dynamics=args.dynamics)

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
            u = jnp.asarray([0.8, 0.0, 0.0, 0.0]),
        )
        controller = controllers.FixedController(env)
    else:
        raise NotImplementedError
    render_env(env, controller=controller, control_params=control_params, repeat_times=1)


if __name__ == "__main__":
    main(tyro.cli(Args))