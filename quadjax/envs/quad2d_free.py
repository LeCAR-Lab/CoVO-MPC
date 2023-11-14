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
from tqdm import trange

import quadjax
from quadjax.envs.base import BaseEnvironment
from quadjax import controllers
from quadjax.dynamics import utils
from quadjax.dynamics import free
from quadjax.dynamics.dataclass import EnvParams2D, EnvState2D, Action2D


class Quad2D(BaseEnvironment):
    """
    JAX Compatible version of Quad2D-v0 OpenAI gym environment. 
    """

    def __init__(self, task: str = "tracking", dynamics: str = 'bodyrate', lower_controller: str = 'base', reward_type = None):
        super().__init__()
        self.task = task
        # reference trajectory function
        if task == "tracking":
            self.generate_traj = partial(utils.generate_lissa_traj_2d, self.default_params.max_steps_in_episode, self.default_params.dt)
            self.reward_fn = utils.tracking_2d_reward_fn
        elif task == "tracking_zigzag":
            self.generate_traj = partial(utils.generate_zigzag_traj_2d, self.default_params.max_steps_in_episode, self.default_params.dt)
            self.reward_fn = utils.tracking_2d_reward_fn
        else:
            raise NotImplementedError
        if reward_type == 'quadratic':
            self.reward_fn = utils.tracking_2d_quadratic_reward_fn
            print('[DEBUG] reward function set to quadratic')
        # dynamics function
        if dynamics == 'bodyrate':
            self.step_fn, self.dynamics_fn = free.get_free_bodyrate_dynamics_2d()
            self.get_obs = self.get_obs_quadonly
        elif dynamics == 'base':
            self.step_fn, self.dynamics_fn = free.get_free_dynamics_2d()
            self.get_obs = self.get_obs_quadonly
        else:
            raise NotImplementedError
        # controller function
        # TODO: unify the interface of controller function
        if lower_controller == 'base':
            def base_controller_fn(obs, state, env_params, rng_act, input_action):
                return input_action, state.control_params, state
            self.action_dim = 2
            self.control_fn = base_controller_fn
            self.default_control_params = None
        elif lower_controller == 'pid_bodyrate':
            assert dynamics == 'base', 'pid_bodyrate controller only works with base dynamics'
            self.default_control_params = controllers.BodyratePIDParams(
                kp=jnp.array([30.0]),
                ki=jnp.array([3.0]) / self.default_params.dt,
                kd=jnp.array([0.0]),
                last_error=jnp.zeros(1),
                integral=jnp.zeros(1),
            )
            controller = controllers.PIDControllerBodyrate(
                self, self.default_control_params
            )
            def pid_controller_fn(obs, state, env_params, rng_act, input_action):
                thrust_normed = input_action[:1]
                omega_tar = input_action[1:] * self.default_params.max_omega
                state = state.replace(omega_tar=omega_tar)

                u, control_params, _ = controller(
                    obs, state, env_params, rng_act, state.control_params
                )

                state = state.replace(control_params=control_params)
                torque = self.default_params.I * u
                torque_normed = torque / self.default_params.max_torque
                action = jnp.concatenate([thrust_normed, torque_normed])
                return action, control_params, state

            self.action_dim = 2
            self.control_fn = pid_controller_fn
        elif lower_controller == 'mppi':
            H = 32
            N = 32
            sigma = 0.1
            # setup mppi control parameters
            thrust_hover = self.default_params.m * self.default_params.g
            thrust_hover_normed = (thrust_hover / self.default_params.max_thrust) * 2.0 - 1.0
            a_mean_per_step = jnp.array([thrust_hover_normed, 0.0]) 
            a_mean = jnp.tile(a_mean_per_step, (H, 1))
            a_cov_per_step = jnp.diag(jnp.array([sigma**2, sigma**2]))
            a_cov = jnp.tile(a_cov_per_step, (H, 1, 1))
            self.default_control_params = controllers.MPPIParams(
                gamma_mean = 1.0,
                gamma_sigma = 0.01,
                discount = 1.0,
                sample_sigma = sigma,
                a_mean = a_mean,
                a_cov = a_cov,
            )
            mppi_controller = controllers.MPPIController(env=self, control_params=self.default_control_params, N=N, H=H, lam=3e-3)
            def mppi_controller_fn(obs, state, env_params, rng_act, input_action):
                control_params = state.control_params
                # convert action to control parameters
                prior_mean_residue = input_action[:2] * 1.0
                prior_cov_scale = input_action[2:4] * 0.4 + 1.0
                mppi_mean_residue = input_action[4:6] * 1.0 
                mppi_cov_scale = input_action[6:8] * 0.9 + 1.0

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

                return action, control_params, state
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
        deterministic: bool = False,
    ) -> Tuple[chex.Array, EnvState2D, float, bool, dict]:
        action = jnp.clip(action, -1.0, 1.0)
        # call controller to get sub_action and new_control_params
        sub_action, new_control_params, state = self.control_fn(None, state, params, key, action)
        state = state.replace(control_params = new_control_params)
        # call substep_env to get next_obs, next_state, reward, done, info
        return self.step_env_wocontroller(key, state, sub_action, params, deterministic=deterministic)

    def step_env_wocontroller(
        self,
        key: chex.PRNGKey,
        state: EnvState2D,
        sub_action: jnp.ndarray,
        params: EnvParams2D,
        deterministic: bool = True,
    ) -> Tuple[chex.Array, EnvState2D, float, bool, dict]:
        obs, state, reward, done, info = self.step_env_wocontroller_gradient(key, state, sub_action, params, deterministic)
        return obs, lax.stop_gradient(state), reward, done, info
    
    def step_env_wocontroller_gradient(
        self,
        key: chex.PRNGKey,
        state: EnvState2D,
        sub_action: jnp.ndarray,
        params: EnvParams2D,
        deterministic: bool = True,
    ) -> Tuple[chex.Array, EnvState2D, float, bool, dict]:
        # TODO: make all the action in the environment a real action, scale happened in controller.
        sub_action = jnp.clip(sub_action, -1.0, 1.0)
        thrust = (sub_action[0] + 1.0) / 2.0 * params.max_thrust
        omega = sub_action[1] * params.max_omega
        env_action = Action2D(thrust=thrust, omega=omega)

        reward = self.reward_fn(state)

        # disable noise in parameters if deterministic
        dyn_noise_scale = params.dyn_noise_scale * (1.0-deterministic)
        params = params.replace(dyn_noise_scale=dyn_noise_scale)

        key, step_key = jax.random.split(key)
        next_state = self.step_fn(params, state, env_action, step_key)
        done = self.is_terminal(state, params)
        return (
            lax.stop_gradient(self.get_obs(next_state, params)),
            next_state,
            reward,
            done,
            {
                "discount": self.discount(next_state, params),
                "err_pos": jnp.linalg.norm(state.pos_tar - state.pos),
                "err_vel": jnp.linalg.norm(state.vel_tar - state.vel),
                "hit_wall": False, 
                "pass_wall": False
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
            roll=0.0, omega=0.0, 
            # trajectory
            pos_tar=pos_traj[0],vel_tar=vel_traj[0], omega_tar=jnp.zeros(1),
            pos_traj=pos_traj,vel_traj=vel_traj,
            # debug value
            last_thrust=0.0,last_omega=0.0,
            # step
            time=0,
            # control parameters
            control_params=self.default_control_params,
        )
        return self.get_obs(state, params), {"discount": 0.0, "err_pos": 0.0, "err_vel": 0.0, "hit_wall": False, "pass_wall": False}, state

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
            state.omega / 40.0,  # 3*3+4=13
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

def eval_env(env: Quad2D, controller:controllers.BaseController, control_params, total_steps = 30000, filename = '', debug=False):
    # running environment
    rng = jax.random.PRNGKey(1)
    rng, rng_params = jax.random.split(rng)
    env_params = env.default_params
    
    rng, rng_reset = jax.random.split(rng)
    obs, info, env_state = env.reset(rng_reset, env_params)

    rng, rng_control = jax.random.split(rng)                      
    control_params = controller.reset(env_state, env_params, controller.init_control_params, rng_control)

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
        rng, rng_control = jax.random.split(rng)
        # if done:
        #     control_params = controller.reset(env_state, env_params, control_params, rng_control)
        # new_control_params = new_control_params.replace(
        #     a_mean = jax.random.uniform(rng_control, shape=control_params.a_mean.shape, minval=-1.0, maxval=1.0),
        # )
        # control_params = lax.cond(done, lambda x: new_control_params, lambda x: x, control_params)
        return (next_obs, next_env_state, rng, env_params, control_params), (info['err_pos'], done)
    
    # def test_out_controller_once(env_state, env_params, control_params, rng_act):
    #     action, control_params, control_info = controller(obs, env_state, env_params, rng_act, control_params)
    #     return control_params.a_mean[:1]
    # # runing test_out_controller_once with batched input repeatedly
    # batch_size = 1024
    # rng, rng_act = jax.random.split(rng)
    # rng_act_batch = jax.random.split(rng_act, batch_size)
    # a_mean_batch = jax.vmap(test_out_controller_once, in_axes=(None, None, None, 0))(env_state, env_params, control_params, rng_act_batch)
    # a_mean_batch = jnp.reshape(a_mean_batch, (batch_size, -1))*1000
    # print(a_mean_batch[0], a_mean_batch[1], a_mean_batch[3])
    # a_mean_batch_cov = jnp.cov(a_mean_batch, rowvar=False)
    # print(a_mean_batch_cov)
    # print('determinant of sampled action covariance matrix:')
    # print(jnp.linalg.det(a_mean_batch_cov))
    # exit()
    
    t0 = time_module.time()
    def run_one_ep(rng_reset, rng):
        env_params = env.default_params

        obs, info, env_state = env.reset(rng_reset, env_params)

        rng_control, rng = jax.random.split(rng)
        control_params = controller.reset(env_state, env_params, controller.init_control_params, rng_control)

        (obs, env_state, rng, env_params, control_params), (err_pos, dones) = lax.scan(
            run_one_step, (obs, env_state, rng, env_params, control_params), jnp.arange(env.default_params.max_steps_in_episode))
        return rng, err_pos
    run_one_ep_jit = jax.jit(run_one_ep)
    # calculate cumulative err_pos bewteen each done
    num_eps = total_steps // env.default_params.max_steps_in_episode
    err_pos_ep = []
    num_trajs = 4
    rng, rng_reset_meta = jax.random.split(rng)
    rng_reset_list = jax.random.split(rng_reset_meta, num_trajs)
    for i, rng_reset in enumerate(rng_reset_list):
        print(f'[DEBUG] test traj {i+1}')
        for _ in trange(num_eps//num_trajs):
            rng, err_pos = run_one_ep_jit(rng_reset, rng)
            err_pos_ep.append(err_pos.mean())
    # last_ep_end = 0
    # for i in range(len(dones)):
    #     if dones[i]:
    #         err_pos_ep.append(err_pos[last_ep_end:i+1].mean())
    #         last_ep_end = i+1
    err_pos_ep = jnp.array(err_pos_ep)
    # print mean and std of err_pos
    pos_mean, pos_std = jnp.mean(err_pos_ep), jnp.std(err_pos_ep)
    print(f"env running time: {time_module.time()-t0:.2f}s")
    print(f'err_pos mean: {pos_mean:.3f}, std: {pos_std:.3f}')
    print(f'${pos_mean*100:.2f} \pm {pos_std*100:.2f}$')

    # save data
    with open(f"{quadjax.get_package_path()}/../results/eval_err_pos_{filename}.pkl", "wb") as f:
        pickle.dump(np.array(err_pos_ep), f)

def render_env(env: Quad2D, controller:controllers.BaseController, control_params, repeat_times = 1, filename = ''):
    # running environment
    rng = jax.random.PRNGKey(1)
    rng, rng_params = jax.random.split(rng)
    env_params = env.sample_params(rng_params)
    env_params = env.default_params # DEBUG

    state_seq, obs_seq, reward_seq, control_info_seq = [], [], [], []
    action_seq = []
    rng, rng_reset = jax.random.split(rng)
    obs, info, env_state = env.reset(rng_reset, env_params)

    # DEBUG set iniiial state here
    # env_state = env_state.replace(quat = jnp.array([jnp.sin(jnp.pi/4), 0.0, 0.0, jnp.cos(jnp.pi/4)]))

    rng, rng_control = jax.random.split(rng)                      
    control_params = controller.reset(env_state, env_params, controller.init_control_params, rng_control)
    n_dones = 0


    # Profiling algorithms
    # controller_jit = jax.jit(controller)
    # controller_reset_jit = jax.jit(controller.reset)
    # rng, rng_act, rng_step = jax.random.split(rng, 3)
    # controller_jit(obs, env_state, env_params, rng_act, control_params)
    # rng, rng_control = jax.random.split(rng)                      
    # controller_reset_jit(env_state, env_params, controller.default_control_params, rng_control)
    # ts = []
    # for i in range(100):
    #     t0 = time_module.time()
    #     rng, rng_act, rng_step = jax.random.split(rng, 3)
    #     action, control_params, control_info = controller_jit(obs, env_state, env_params, rng_act, control_params)
    #     ts.append((time_module.time()-t0)*1000)
    # print(f'running time: ${np.mean(ts):.2f} \pm {np.std(ts):.2f}$')
    # ts = []
    # for i in range(100):
    #     t0 = time_module.time()
    #     rng, rng_control = jax.random.split(rng)
    #     control_params = controller_reset_jit(env_state, env_params, controller.default_control_params, rng_control)
    #     ts.append((time_module.time()-t0)*1000)
    # print(f'reset time: ${np.mean(ts):.2f} \pm {np.std(ts):.2f}$')
    # exit()

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
            rng, rng_control = jax.random.split(rng)
            control_params = controller.reset(env_state, env_params, control_params, rng_control)
            n_dones += 1

        control_info_seq.append(control_info)
        reward_seq.append(reward)
        obs_seq.append(obs)
        action_seq.append(action)
        obs = next_obs
        env_state = next_env_state
    print(f"env running time: {time_module.time()-t0:.2f}s")

    t0 = time_module.time()
    state_seq_dict = [s.__dict__ for s in state_seq]
    utils.plot_states(state_seq_dict, obs_seq, reward_seq, env_params, filename=filename)
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
    anim.save(filename=f"{quadjax.get_package_path()}/../results/anim_{filename}.gif", writer="imagemagick", fps=int(1.0/env_params.dt))
    
    # convert state into dict
    with open(f"{quadjax.get_package_path()}/../results/state_seq_{filename}.pkl", "wb") as f:
        pickle.dump(state_seq_dict, f)

'''
reward function here. 
'''

@pydataclass
class Args:
    mode: str = "render"
    task: str = "tracking_zigzag"
    dynamics: str = 'bodyrate'
    controller: str = 'lqr' # mppi
    # controller parameter, character + number + character + number + ...
    controller_params: str = ''
    lower_controller: str = 'base' # mppi
    debug: bool = False
    reward_type: str = ''
    eval_repeat_times: int = 10

def main(args: Args):
    env = Quad2D(task=args.task, dynamics=args.dynamics, lower_controller=args.lower_controller, reward_type=args.reward_type)

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
        controller = controllers.LQRController2D(env, control_params)
        control_params = controller.update_params(env.default_params, control_params)
        controller.default_control_params = control_params
        # print control_params.K with , delimiter
        # np.savetxt(f"{quadjax.get_package_path()}/../results/K_{args.controller_params}.csv", control_params.K, delimiter=",")
        # exit()
    elif args.controller == 'pid':
        control_params = controllers.PIDParams(
            Kp=10.0,
            Kd=5.0,
            Ki=0.0,
            Kp_att=10.0,
            integral=jnp.zeros(2),
        )
        controller = controllers.PIDController2D(env, control_params=control_params)
    elif args.controller == 'fixed':
        control_params = controllers.FixedParams(
            u = jnp.zeros(env.action_dim)
        )
        controller = controllers.FixedController(env, control_params)
    elif args.controller == 'ppo':
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
    elif args.controller == 'random':
        control_params = None
        controller = controllers.RandomController(env, control_params)
    elif 'mppi' in args.controller:
        sigma = 0.5
        if args.controller_params == '':
            N = 8192
            H = 16
            lam = 1e-2
        else:
            # parse in format "N{sample_number}_H{horizon}_sigma{sigma}_lam{lam}"
            N = int(args.controller_params.split('_')[0][1:])
            H = int(args.controller_params.split('_')[1][1:])
            lam = float(args.controller_params.split('_')[2][3:])
            print(f'[DEBUG], set controller parameters to be: N={N}, H={H}, lam={lam}')
        if args.debug:
            N = 8
            H = 2
            print('[DEBUG] N = 8')
        elif args.lower_controller == 'mppi':
            a_mean = jnp.zeros((H, env.action_dim))
            a_cov = jnp.tile(jnp.diag(jnp.ones(env.action_dim)*(sigma**2)), (H, 1, 1))
        else:
            thrust_hover = env.default_params.m * env.default_params.g
            thrust_hover_normed = (thrust_hover / env.default_params.max_thrust) * 2.0 - 1.0
            a_mean_per_step = jnp.array([thrust_hover_normed, 0.0]) 
            a_mean = jnp.tile(a_mean_per_step, (H, 1))
            if args.controller == 'mppi':
                a_cov_per_step = jnp.diag(jnp.array([sigma**2, sigma**2]))
                a_cov = jnp.tile(a_cov_per_step, (H, 1, 1))
            elif 'mppi_zeji' in args.controller:
                a_cov = jnp.diag(jnp.ones(H*2)*sigma**2)
        if args.controller == 'mppi':
            control_params = controllers.MPPIParams(
                gamma_mean = 1.0,
                gamma_sigma = 0.0,
                discount = 1.0,
                sample_sigma = sigma,
                a_mean = a_mean,
                a_cov = a_cov,
            )
            controller = controllers.MPPIController(env=env, control_params=control_params, N=N, H=H, lam=lam)
        elif 'mppi_zeji' in args.controller:
            if 'mean' in args.controller:
                expansion_mode = 'mean'
            elif 'repeat' in args.controller:
                expansion_mode = 'repeat'
            elif 'lqr' in args.controller:
                expansion_mode = 'lqr'
            elif 'pid' in args.controller:
                expansion_mode = 'pid'
            elif 'zero' in args.controller:
                expansion_mode = 'zero'
            elif 'ppo' in args.controller:
                expansion_mode = 'ppo'
            elif '_mppizeji' in args.controller:
                expansion_mode = 'mppizeji'
            elif '_mppi' in args.controller:
                expansion_mode = 'mppi'
            else:
                expansion_mode = 'mean'
                print('[DEBUG] unset expansion mode, MPPI(zeji) expansion_mode set to mean')
            control_params = controllers.MPPIZejiParams(
                gamma_mean = 1.0,
                gamma_sigma = 0.0,
                discount = 1.0,
                sample_sigma = sigma,
                a_mean = a_mean,
                a_cov = a_cov,
                a_cov_offline=jnp.zeros((H, env.action_dim, env.action_dim)),
            )
            controller = controllers.MPPIZejiController(env=env, control_params=control_params, N=N, H=H, lam=lam, expansion_mode=expansion_mode)
    else:
        raise NotImplementedError
    
    filename = f'{args.controller}_{args.controller_params}'
    if args.mode == 'render':
        render_env(env, controller=controller, control_params=control_params, repeat_times=1, filename=filename)
    elif args.mode == 'eval':
        eval_env(env, controller=controller, control_params=control_params, total_steps=300*args.eval_repeat_times*4, filename=filename)
    else:
        raise NotImplementedError


if __name__ == "__main__":
    main(tyro.cli(Args))