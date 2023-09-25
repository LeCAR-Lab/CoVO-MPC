import jax
import jax.numpy as jnp
from jax import lax
from gymnax.environments import environment, spaces
from typing import Tuple, Optional
import chex
from functools import partial
from flax import struct
from dataclasses import dataclass as pydataclass
import tyro
import pickle
import time as time_module
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import numpy as np

import quadjax
from quadjax import controllers
from quadjax.dynamics import utils
from quadjax.dynamics.free import get_free_bodyrate_dynamics_2d
from quadjax.dynamics.dataclass import EnvParams2D, EnvState2D, Action2D

# for debug purpose
from icecream import install
install()


class Quad2D(environment.Environment):
    """
    JAX Compatible version of Quad2D-v0 OpenAI gym environment. 
    """

    def __init__(self, task: str = "tracking", dynamics: str = 'bodyrate', lower_controller: str = 'base'):
        super().__init__()
        self.task = task
        # reference trajectory function
        if task == "tracking":
            self.generate_traj = partial(utils.generate_lissa_traj_2d, self.default_params.max_steps_in_episode, self.default_params.dt)
            self.reward_fn = utils.tracking_reward_fn
        elif task == "tracking_zigzag":
            self.generate_traj = partial(utils.generate_zigzag_traj_2d, self.default_params.max_steps_in_episode, self.default_params.dt)
            self.reward_fn = utils.tracking_reward_fn
        else:
            raise NotImplementedError
        # dynamics function
        if dynamics == 'bodyrate':
            self.step_fn, self.dynamics_fn = get_free_bodyrate_dynamics_2d()
            self.get_obs = self.get_obs_quadonly
        else:
            raise NotImplementedError
        # controller function
        if lower_controller == 'base':
            def base_controller_fn(obs, state, env_params, rng_act, input_action):
                return input_action, state.control_params, None
            self.control_fn = base_controller_fn
            self.action_dim = 2
            self.init_control_params = None
        elif lower_controller == 'mppi':
            H = 40
            N = 8
            sigma = 0.1
            # setup mppi control parameters
            thrust_hover = self.default_params.m * self.default_params.g
            thrust_hover_normed = (thrust_hover / self.default_params.max_thrust) * 2.0 - 1.0
            a_mean_per_step = jnp.array([thrust_hover_normed, 0.0]) 
            a_mean = jnp.tile(a_mean_per_step, (H, 1))
            a_cov_per_step = jnp.diag(jnp.array([sigma**2, sigma**2]))
            a_cov = jnp.tile(a_cov_per_step, (H, 1, 1))
            self.init_control_params = controllers.MPPIParams(
                gamma_mean = 1.0,
                gamma_sigma = 0.01,
                discount = 0.9,
                sample_sigma = sigma,
                a_mean = a_mean,
                a_cov = a_cov,
            )
            mppi_controller = controllers.MPPIController2D(env=self, N=N, H=H, lam=3e-3)
            def mppi_controller_fn(obs, state, env_params, rng_act, input_action):
                control_params = state.control_params
                # convert action to control parameters
                prior_mean_residue = input_action[:2] * 0.2 # [-0.1, 0.1]
                prior_cov_scale = input_action[2:4] * 0.2 + 1.0 # [0.9, 1.1]
                mppi_mean_residue = input_action[4:6] * 0.1 # [-0.1, 0.1]
                mppi_cov_scale = input_action[6:8] * 0.1 + 1.0 # [0.9, 1.1]
                a_mean = control_params.a_mean + prior_mean_residue
                a_cov = (prior_cov_scale[:, None] @ prior_cov_scale[None, :]) * control_params.a_cov
                control_params = control_params.replace(a_mean=a_mean,a_cov=a_cov)

                # update control parameters
                _, control_params, control_info = mppi_controller(
                    obs, state, env_params, rng_act, control_params)
                
                a_mean_compensated = control_params.a_mean[0] + mppi_mean_residue
                a_cov_scaled = (mppi_cov_scale[:, None] @ mppi_cov_scale[None, :]) * control_params.a_cov[0]
                a_sampled = jax.random.multivariate_normal(rng_act, a_mean_compensated, a_cov_scaled)
                action = a_sampled
                control_info['a_mean'] = a_mean_compensated
                control_info['a_cov'] = a_cov_scaled

                return action, control_params, control_info
            self.control_fn = mppi_controller_fn
            self.action_dim = 8
        else:
            raise NotImplementedError
        # equibrium point
        self.equib = jnp.zeros(5)
        # RL parameters
        self.obs_dim = 10 + self.default_params.traj_obs_len * 4

    '''
    environment properties
    '''
    @property
    def default_params(self) -> EnvParams2D:
        """Default environment parameters for Quad2D-v0."""
        return EnvParams2D()

    '''
    key methods
    '''
    def step_env(
        self,
        key: chex.PRNGKey,
        state: EnvState2D,
        action: jnp.ndarray,
        params: EnvParams2D,
    ) -> Tuple[chex.Array, EnvState2D, float, bool, dict]:
        action = jnp.clip(action, -1.0, 1.0)
        # call controller to get sub_action and new_control_params
        sub_action, new_control_params, control_info = self.control_fn(None, state, params, key, action)
        state = state.replace(control_params = new_control_params)
        # call substep_env to get next_obs, next_state, reward, done, info
        return self.step_env_wocontroller(key, state, sub_action, params)

    def step_env_wocontroller(
        self,
        key: chex.PRNGKey,
        state: EnvState2D,
        sub_action: jnp.ndarray,
        params: EnvParams2D,
    ) -> Tuple[chex.Array, EnvState2D, float, bool, dict]:
        thrust = (sub_action[0] + 1.0) / 2.0 * params.max_thrust
        roll_dot = sub_action[1] * params.max_bodyrate
        env_action = Action2D(thrust=thrust, roll_dot=roll_dot)

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
        self, key: chex.PRNGKey, params: EnvParams2D
    ) -> Tuple[chex.Array, EnvState2D]:
        """Reset environment state by sampling theta, theta_dot."""
        traj_key, pos_key, key = jax.random.split(key, 3)
        # generate reference trajectory by adding a few sinusoids together
        pos_traj, vel_traj = self.generate_traj(traj_key)
        zeros2 = jnp.zeros(2)
        state = EnvState2D(
            # drone
            pos=zeros2, vel=zeros2,
            roll=0.0, roll_dot=0.0, 
            # trajectory
            pos_tar=pos_traj[0],vel_tar=vel_traj[0],
            pos_traj=pos_traj,vel_traj=vel_traj,
            # debug value
            last_thrust=0.0,last_roll_dot=0.0,
            # step
            time=0,
            # control parameters
            control_params=self.init_control_params,
        )
        return self.get_obs(state, params), state

    @partial(jax.jit, static_argnums=(0,))
    def sample_params(self, key: chex.PRNGKey) -> EnvParams2D:
        """Sample environment parameters."""
        # NOTE domain randomization disabled here

        # param_key = jax.random.split(key)[0]
        # rand_val = jax.random.uniform(param_key, shape=(9,), minval=0.0, maxval=1.0)

        m = 0.03
        I = 2.0e-5

        return EnvParams2D(m=m, I=I)
    
    @partial(jax.jit, static_argnums=(0,))
    def get_obs_quadonly(self, state: EnvState2D, params: EnvParams2D) -> chex.Array:
        """Return angle in polar coordinates and change."""
        # future trajectory observation
        traj_obs_len = self.default_params.traj_obs_len
        traj_obs_gap = self.default_params.traj_obs_gap
        # Generate the indices
        indices = state.time + 1 + jnp.arange(traj_obs_len) * traj_obs_gap
        obs_elements = [
            # drone
            *state.pos,
            *(state.vel / 4.0),
            state.roll,
            state.roll_dot / 40.0,  # 3*3+4=13
            # trajectory
            *(state.pos_tar),
            *(state.vel_tar / 4.0),  # 3*2=6
            *state.pos_traj[indices].flatten(), 
            *(state.vel_traj[indices].flatten() / 4.0), 
        ]  # 13+6=19
        obs = jnp.asarray(obs_elements)

        return obs

    @partial(jax.jit, static_argnums=(0,))
    def is_terminal(self, state: EnvState2D, params: EnvParams2D) -> bool:
        """Check whether state is terminal."""
        # Check number of steps in episode termination condition
        done = (state.time >= params.max_steps_in_episode) \
            | (jnp.abs(state.pos) > 3.0).any()
        return done


def test_env(env: Quad2D, controller, control_params, repeat_times = 1):
    # running environment
    rng = jax.random.PRNGKey(1)
    rng, rng_params = jax.random.split(rng)
    env_params = env.sample_params(rng_params)
    env_params = env.default_params # DEBUG

    state_seq, obs_seq, reward_seq, control_info_seq = [], [], [], []
    action_seq = []
    rng, rng_reset = jax.random.split(rng)
    obs, env_state = env.reset(rng_reset, env_params)

    # DEBUG set iniiial state here
    # env_state = env_state.replace(quat = jnp.array([jnp.sin(jnp.pi/4), 0.0, 0.0, jnp.cos(jnp.pi/4)]))
                                  
    control_params = controller.update_params(env_params, control_params)
    n_dones = 0

    t0 = time_module.time()
    while n_dones < repeat_times:
        state_seq.append(env_state.replace(control_params=0.0))
        rng, rng_act, rng_step = jax.random.split(rng, 3)
        action, control_params, control_info = controller(obs, env_state, env_params, rng_act, control_params)
        if control_info is not None:
            if 'a_mean' in control_info:
                action = control_info['a_mean'] # evaluation only
        next_obs, next_env_state, reward, done, info = env.step(
            rng_step, env_state, action, env_params)
        if done:
            rng, rng_params = jax.random.split(rng)
            env_params = env.sample_params(rng_params)
            control_params = controller.update_params(env_params, control_params)
            n_dones += 1

        control_info_seq.append(control_info)
        reward_seq.append(reward)
        obs_seq.append(obs)
        action_seq.append(action)
        obs = next_obs
        env_state = next_env_state
    print(f"env running time: {time_module.time()-t0:.2f}s")

    t0 = time_module.time()
    utils.plot_states(state_seq, obs_seq, reward_seq, env_params)
    print(f"plotting time: {time_module.time()-t0:.2f}s")

    # plot animation
    def update_plot(i):
        plt.gca().clear()
        pos_array = np.asarray([s.pos for s in state_seq[:i+1]])
        tar_array = np.asarray([s.pos_tar for s in state_seq])
        plt.plot(pos_array[:, 0], pos_array[:, 1], "b", alpha=0.5)
        plt.plot(tar_array[:, 0], tar_array[:, 1], "r--", alpha = 0.3)

        # plot action with horizontal line in different colors at the top-left corner of the figure with large width
        action_array = np.asarray(action_seq[i])
        # plot cube with conors at -2.0-> -1.0, 2.0-0.05*num_actions -> 2.0
        plt.plot([-2.0, -1.0, -1.0, -2.0, -2.0], [2.0, 2.0, 2.0 - 0.05*len(action_array), 2.0 - 0.05*len(action_array), 2.0], color = 'k', linewidth = 1)
        # the line start around at [0.5, y] and end at [0.5 + action*0.5, y], where y is different for each action
        for j, a in enumerate(action_array):
            plt.plot([-1.5, -1.5 + a*0.5], [2.0 - j*0.05, 2.0 - j*0.05], color = 'C'+str(j), linewidth = 3)
        # plot vertical line at x=-1.5, y from 2.0 to 2.0-0.05*num_actions
        plt.plot([-1.5, -1.5], [2.0, 2.0 - 0.05*len(action_array)], color = 'k', linewidth = 1)
        # quadrotor 0 with blue arrow
        plt.arrow(
            state_seq[i].pos[0],
            state_seq[i].pos[1],
            -0.1 * jnp.sin(state_seq[i].roll),
            0.1 * jnp.cos(state_seq[i].roll),
            width=0.01,
            color="g",
        )
        # plot the 95% confidence interval of the future control sequence
        if control_info_seq[i] is not None:
            pos_mean = control_info_seq[i]['pos_mean']
            pos_std = control_info_seq[i]['pos_std']
            plt.fill_between(pos_mean[:, 0], pos_mean[:, 1] - pos_std[:, 1], pos_mean[:, 1] + pos_std[:, 1], alpha=0.2, color='g')
        # plot y_tar and z_tar with red dot
        plt.plot(state_seq[i].pos_tar[0], state_seq[i].pos_tar[1], "ro")
        plt.xlabel("y")
        plt.ylabel("z")
        plt.xlim([-2, 2])
        plt.ylim([-2, 2])

    plt.figure(figsize=(4, 4))
    anim = FuncAnimation(plt.gcf(), update_plot, frames=len(state_seq), interval=1)
    anim.save(filename=f"{quadjax.get_package_path()}/../results/anim.gif", writer="imagemagick", fps=int(1.0/env_params.dt))
    
    with open(f"{quadjax.get_package_path()}/../results/state_seq.pkl", "wb") as f:
        pickle.dump(state_seq, f)

'''
reward function here. 
'''

@pydataclass
class Args:
    task: str = "tracking"
    dynamics: str = 'bodyrate'
    controller: str = 'lqr' # mppi
    lower_controller: str = 'base' # mppi
    debug: bool = False

def main(args: Args):
    env = Quad2D(task=args.task, dynamics=args.dynamics, lower_controller=args.lower_controller)

    print("starting test...")
    # enable NaN value detection
    # from jax import config
    # config.update("jax_debug_nans", True)
    # with jax.disable_jit():
    if args.controller == 'lqr':
        control_params = controllers.LQRParams(
            Q = jnp.diag(jnp.ones(5)),
            R = 0.03 * jnp.diag(jnp.ones(2)),
            K = jnp.zeros((2, 5)),
        )
        controller = controllers.LQRController2D(env)
    elif args.controller == 'fixed':
        control_params = controllers.FixedParams(
            u = jnp.zeros(env.action_dim)
        )
        controller = controllers.FixedController(env)
    elif args.controller == 'random':
        control_params = None
        controller = controllers.RandomController(env)
    elif args.controller == 'mppi':
        N = 8192 if not args.debug else 8
        H = 40
        sigma = 0.1
        lam = 3e-3
        if args.lower_controller == 'base':
            thrust_hover = env.default_params.m * env.default_params.g
            thrust_hover_normed = (thrust_hover / env.default_params.max_thrust) * 2.0 - 1.0
            a_mean_per_step = jnp.array([thrust_hover_normed, 0.0]) 
            a_mean = jnp.tile(a_mean_per_step, (H, 1))
            a_cov_per_step = jnp.diag(jnp.array([sigma**2, sigma**2]))
            a_cov = jnp.tile(a_cov_per_step, (H, 1, 1))
        elif args.lower_controller == 'mppi':
            a_mean = jnp.zeros((H, env.action_dim))
            a_cov = jnp.tile(jnp.diag(jnp.ones(env.action_dim)*(sigma**2)), (H, 1, 1))
        control_params = controllers.MPPIParams(
            gamma_mean = 1.0,
            gamma_sigma = 0.01,
            discount = 0.9,
            sample_sigma = sigma,
            a_mean = a_mean,
            a_cov = a_cov,
        )
        controller = controllers.MPPIController2D(env=env, N=N, H=H, lam=lam)
    else:
        raise NotImplementedError
    test_env(env, controller=controller, control_params=control_params, repeat_times=1)


if __name__ == "__main__":
    main(tyro.cli(Args))