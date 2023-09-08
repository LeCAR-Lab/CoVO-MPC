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

from quadjax import controllers
from quadjax.dynamics import utils
from quadjax.dynamics.free import get_free_dynamics_3d
from quadjax.dynamics.dataclass import EnvParams3D, EnvState3D, Action3D

# for debug purpose
from icecream import install
install()


class Quad3D(environment.Environment):
    """
    JAX Compatible version of Quad3D-v0 OpenAI gym environment. Source:
    github.com/openai/gym/blob/master/gym/envs/classic_control/Quad3D.py
    """

    def __init__(self, task: str = "hovering", dynamics: str = 'free'):
        super().__init__()
        self.task = task
        # reference trajectory function
        if task == "tracking":
            self.generate_traj = partial(utils.generate_lissa_traj, self.default_params.max_steps_in_episode)
            self.reward_fn = utils.tracking_reward_fn
        elif task == "tracking_zigzag":
            self.generate_traj = partial(utils.generate_zigzag_traj, self.default_params.max_steps_in_episode)
            self.reward_fn = utils.tracking_reward_fn
        elif task in "jumping":
            self.generate_traj = partial(utils.generate_fixed_traj, self.default_params.max_steps_in_episode)
            self.reward_fn = utils.jumping_reward_fn
        elif task == 'hovering':
            self.generate_traj = partial(utils.generate_fixed_traj, self.default_params.max_steps_in_episode)
            self.reward_fn = utils.tracking_reward_fn
        else:
            raise NotImplementedError
        # dynamics function
        if dynamics == 'free':
            self.step_fn, self.dynamics_fn = get_free_dynamics_3d()
            self.get_obs = self.get_obs_quadonly
        else:
            raise NotImplementedError
        # equibrium point
        self.equib = jnp.array([0.0]*6+[1.0]+[0.0]*6)
        # RL parameters
        self.action_dim = 4
        self.obs_dim = 19 + self.default_params.traj_obs_len * 6 + 12


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

        next_state = self.step_fn(params, state, env_action)

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
        traj_key, pos_key, key = jax.random.split(key, 3)
        # generate reference trajectory by adding a few sinusoids together
        pos_traj, vel_traj = self.generate_traj(params.dt, traj_key)
        zeros3 = jnp.zeros(3)
        state = EnvState3D(
            # drone
            pos=zeros3,
            vel=zeros3,
            omega=zeros3,
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
        )
        return self.get_obs(state, params), state

    @partial(jax.jit, static_argnums=(0,))
    def sample_params(self, key: chex.PRNGKey) -> EnvParams3D:
        """Sample environment parameters."""

        param_key = jax.random.split(key)[0]
        rand_val = jax.random.uniform(param_key, shape=(9,), minval=0.0, maxval=1.0)

        m = 0.025 + 0.015 * rand_val[0]
        I = jnp.array([1.2e-5, 1.2e-5, 2.0e-5]) + 0.5e-5 * rand_val[1:4]
        I = jnp.diag(I)
        mo = 0.01 + 0.01 * rand_val[4]
        l = 0.2 + 0.2 * rand_val[5]
        hook_offset = rand_val[6:9] * 0.04

        return EnvParams3D(m=m, I=I, mo=mo, l=l, hook_offset=hook_offset)
    
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
            state.vel / 4.0,
            state.quat,
            state.omega / 40.0,  # 3*3+4=13
            # trajectory
            state.pos_tar,
            state.vel_tar / 4.0,  # 3*2=6
            state.pos_traj[indices].flatten(), 
            state.vel_traj[indices].flatten() / 4.0, 
            # parameter observation
            jnp.array(
                [(params.m - 0.025) / (0.04 - 0.025) * 2.0 - 1.0,]), # 3
            ((params.I - 1.2e-5) / 0.5e-5 * 2.0 - 1.0).flatten(),  # 3x3
        ]  # 13+6=19
        obs = jnp.concatenate(obs_elements, axis=-1)

        return obs

    @partial(jax.jit, static_argnums=(0,))
    def is_terminal(self, state: EnvState3D, params: EnvParams3D) -> bool:
        """Check whether state is terminal."""
        # Check number of steps in episode termination condition
        done = (state.time >= params.max_steps_in_episode) \
            | (jnp.abs(state.pos) > 2.0).any() \
            | (jnp.abs(state.pos_obj) > 2.0).any() \
            | (jnp.abs(state.omega) > 100.0).any()
        return done


def test_env(env: Quad3D, controller, control_params, repeat_times = 1):
    # running environment
    rng = jax.random.PRNGKey(1)
    rng, rng_params = jax.random.split(rng)
    env_params = env.sample_params(rng_params)
    env_params = env.default_params # DEBUG

    state_seq, obs_seq, reward_seq = [], [], []
    rng, rng_reset = jax.random.split(rng)
    obs, env_state = env.reset(rng_reset, env_params)
    control_params = controller.update_params(env_params, control_params)
    n_dones = 0

    t0 = time_module.time()
    while n_dones < repeat_times:
        state_seq.append(env_state)
        rng, rng_act, rng_step = jax.random.split(rng, 3)
        action = controller(obs, env_state, env_params, rng_act, control_params)
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
    with open("../../results/state_seq.pkl", "wb") as f:
        pickle.dump(state_seq, f)

'''
reward function here. 
'''

@pydataclass
class Args:
    task: str = "hovering"
    dynamics: str = 'free'


def main(args: Args):
    env = Quad3D(task=args.task, dynamics=args.dynamics)

    print("starting test...")
    # enable NaN value detection
    # from jax import config
    # config.update("jax_debug_nans", True)
    # with jax.disable_jit():
    control_params = controllers.LQRParams(
        Q = jnp.diag(jnp.ones(12)),
        R = 0.03 * jnp.diag(jnp.ones(4)),
        K = jnp.zeros((4, 12)),
    )
    controller = controllers.LQRController(env)
    test_env(env, controller=controller, control_params=control_params)


if __name__ == "__main__":
    main(tyro.cli(Args))