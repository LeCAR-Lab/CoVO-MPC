import jax
import jax.numpy as jnp
from jax import lax
from gymnax.environments import environment, spaces
from typing import Tuple, Optional
import chex
import matplotlib.pyplot as plt
import numpy as np
from functools import partial
from dataclasses import dataclass as pydataclass
import tyro
import pickle
import time as time_module

import quadjax
from quadjax import controllers
from quadjax.dynamics import utils
from quadjax.dynamics.utils import get_hit_penalty
from quadjax.dynamics.dataclass import EnvParams3D, EnvState3D, Action3D
from quadjax.dynamics.loose import get_loose_dynamics_3d
from quadjax.dynamics.taut import get_taut_dynamics_3d
from quadjax.dynamics.trans import get_dynamic_transfer_3d

# for debug purpose
from icecream import install
install()


class Quad3D(environment.Environment):
    """
    JAX Compatible version of Quad3D-v0 OpenAI gym environment. Source:
    github.com/openai/gym/blob/master/gym/envs/classic_control/Quad3D.py
    """

    def __init__(self, task: str = "tracking"):
        super().__init__()
        self.task = task
        # reference trajectory function
        if task == "tracking":
            self.generate_traj = partial(utils.generate_lissa_traj, self.default_params.max_steps_in_episode, self.default_params.dt)
            self.reward_fn = utils.tracking_penyaw_obj_reward_fn
        elif task == "tracking_zigzag":
            self.generate_traj = partial(utils.generate_zigzag_traj, self.default_params.max_steps_in_episode, self.default_params.dt)
            self.reward_fn = utils.tracking_penyaw_obj_reward_fn
        elif task in "jumping":
            self.generate_traj = partial(utils.generate_fixed_traj, self.default_params.max_steps_in_episode, self.default_params.dt)
            self.reward_fn = utils.jumping_reward_fn
        elif task == 'hovering':
            self.generate_traj = partial(utils.generate_fixed_traj, self.default_params.max_steps_in_episode, self.default_params.dt)
            self.reward_fn = utils.tracking_reward_fn
        else:
            raise NotImplementedError
        # dynamics
        taut_dynamics = get_taut_dynamics_3d()
        loose_dynamics = get_loose_dynamics_3d()
        dynamic_transfer = get_dynamic_transfer_3d()
        def step_fn(params, state, env_action):
            old_loose_state = state.l_rope < (params.l - params.rope_taut_therehold)
            taut_state = taut_dynamics(params, state, env_action)
            loose_state = loose_dynamics(params, state, env_action)
            return dynamic_transfer(
                params, loose_state, taut_state, old_loose_state)
            # return loose_dynamics(params, state, env_action)
        self.step_fn = step_fn
        # lower-level controller
        def base_controller_fn(obs, state, env_params, rng_act, input_action):
            return input_action, state.control_params, None
        self.control_fn = base_controller_fn
        # rl parameters
        self.obs_dim = 42 + self.default_params.traj_obs_len * 6 + 15
        self.action_dim = 4

    @property
    def default_params(self) -> EnvParams3D:
        return EnvParams3D()

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
        # call controller to get sub_action and new_control_params
        sub_action, new_control_params, control_info = self.control_fn(None, state, params, key, action)
        state = state.replace(control_params = new_control_params)
        # call substep_env to get next_obs, next_state, reward, done, info
        return self.step_env_wocontroller(key, state, sub_action, params)

    def step_env_wocontroller(
        self,
        key: chex.PRNGKey,
        state: EnvState3D,
        sub_action: jnp.ndarray,
        params: EnvParams3D,
    ) -> Tuple[chex.Array, EnvState3D, float, bool, dict]:
        sub_action = jnp.clip(sub_action, -1.0, 1.0)
        thrust = (sub_action[0] + 1.0) / 2.0 * params.max_thrust
        torque = sub_action[1:] * params.max_torque
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
        pos_traj, vel_traj = self.generate_traj(traj_key)

        zeros3 = jnp.zeros(3)
        pos_hook = jnp.array([0.0, 0.0, params.l])
        pos = pos_hook - params.hook_offset
        state = EnvState3D(
            # drone
            pos=pos,
            vel=zeros3,
            omega=zeros3,
            quat=jnp.concatenate([zeros3, jnp.array([1.0])]),
            # object
            pos_obj=zeros3,
            vel_obj=zeros3,
            # hook
            pos_hook=pos_hook,
            vel_hook=zeros3,
            # rope
            l_rope=params.l,
            zeta=jnp.array([0.0, 0.0, -1.0]),
            zeta_dot=zeros3,
            f_rope=zeros3,
            f_rope_norm=0.0,
            # trajectory
            pos_tar=pos_traj[0],
            vel_tar=vel_traj[0],
            pos_traj=pos_traj,
            vel_traj=vel_traj,
            # debug value
            last_thrust=0.0,
            last_torque=zeros3,
            # step
            time=0,
        )
        return self.get_obs(state, params), state

    @partial(jax.jit, static_argnums=(0,))
    def sample_params(self, key: chex.PRNGKey) -> EnvParams3D:
        """Sample environment parameters."""

        param_key = jax.random.split(key)[0]
        rand_val = jax.random.uniform(param_key, shape=(9,), minval=0.0, maxval=1.0)

        # m = 0.025 + 0.015 * rand_val[0]
        # I = jnp.array([1.2e-5, 1.2e-5, 2.0e-5]) + 0.5e-5 * rand_val[1:4]
        # I = jnp.diag(I)
        # mo = 0.01 + 0.01 * rand_val[4]
        # l = 0.2 + 0.2 * rand_val[5]
        # hook_offset = rand_val[6:9] * 0.04

        m = 0.025
        I = jnp.array([1.2e-5, 1.2e-5, 2.0e-5])
        I = jnp.diag(I)
        mo = 0.01
        l = 0.2
        hook_offset = jnp.zeros(3)

        return EnvParams3D(m=m, I=I, mo=mo, l=l, hook_offset=hook_offset)

    def get_obs(self, state: EnvState3D, params: EnvParams3D) -> chex.Array:
        """Return angle in polar coordinates and change."""
        indices = state.time + 1 + jnp.arange(self.default_params.traj_obs_len) * self.default_params.traj_obs_gap
        obs_elements = [
            # drone
            state.pos,
            state.vel / 4.0,
            state.quat,
            state.omega / 40.0,  # 3*3+4=13
            # object
            state.pos_obj,
            state.vel_obj / 4.0,  # 3*2=6
            # hook
            state.pos_hook,
            state.vel_hook / 4.0,  # 3*2=6
            # rope
            jnp.expand_dims(state.l_rope, axis=0),
            state.zeta,
            state.zeta_dot,  # 3*3=9
            state.f_rope,
            jnp.expand_dims(state.f_rope_norm, axis=0),  # 3+1=4
            # trajectory
            state.pos_tar,
            state.vel_tar / 4.0,  # 3*2=6
            # future trajectory
            state.pos_traj[indices].flatten(), 
            state.vel_traj[indices].flatten() / 4.0
        ]  # 13+6+6+9+4+6=44

        # parameter observation
        param_elements = [
            jnp.array(
                [
                    (params.m - 0.025) / (0.04 - 0.025) * 2.0 - 1.0,
                    (params.mo - 0.005) / 0.05 * 2.0 - 1.0,
                    (params.l - 0.2) / (0.4 - 0.2) * 2.0 - 1.0,
                ]
            ),  # 3
            ((params.I - 1.2e-5) / 0.5e-5 * 2.0 - 1.0).flatten(),  # 3x3
            (params.hook_offset - 0.0) / 0.04 * 2.0 - 1.0,  # 3
        ]  # 4+3=7

        obs = jnp.concatenate(obs_elements + param_elements).squeeze()
        return obs

    def is_terminal(self, state: EnvState3D, params: EnvParams3D) -> bool:
        """Check whether state is terminal."""
        # Check number of steps in episode termination condition
        done = (
            (state.time >= params.max_steps_in_episode)
            | (jnp.abs(state.pos) > 3.0).any()
            | (jnp.abs(state.omega) > 100.0).any()
            | (state.quat[3] < jnp.cos(jnp.pi / 4.0))
        )
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

    # reset all parameters
    state_seq, obs_seq, reward_seq = [], [], []
    rng, rng_reset = jax.random.split(rng)
    obs, env_state = env.reset(rng_reset, env_params)
    control_params = controller.update_params(env_params, control_params)
    n_dones = 0

    # run environment
    t0 = time_module.time()
    while n_dones < repeat_times:
        state_seq.append(env_state)
        rng, rng_act, rng_step = jax.random.split(rng, 3)
        action, control_params, control_info = controller(obs, env_state, env_params, rng_act, control_params)
        next_obs, next_env_state, reward, done, info = env.step(
            rng_step, env_state, action, env_params
        )
        if done:
            rng, rng_params = jax.random.split(rng)
            env_params = env.sample_params(rng_params)
            control_params = controller.reset()
            control_params = controller.update_params(env_params, control_params)
            n_dones += 1
        if control_info is not None:
            if 'a_mean' in control_info:
                action = control_info['a_mean'] # evaluation only
        reward_seq.append(reward)
        obs_seq.append(obs)
        obs = next_obs
        env_state = next_env_state
    print(f"env running time: {time_module.time()-t0:.2f}s")

    # plot results
    t0 = time_module.time()
    # plot results
    num_figs = len(state_seq[0].__dict__) + 20
    time = [s.time * env_params.dt for s in state_seq]
    # calculate number of rows needed
    num_rows = int(jnp.ceil(num_figs / 6))
    # create num_figs subplots
    plt.subplots(num_rows, 6, figsize=(4 * 6, 2 * num_rows))
    # plot reward
    plt.subplot(num_rows, 6, 1)
    plt.plot(time, reward_seq)
    plt.ylabel("reward")
    # plot obs
    current_fig = 2
    for i in range(len(obs_seq[0]) // 10 + 1):
        plt.subplot(num_rows, 6, current_fig)
        current_fig += 1
        for j in range(10):
            idx = i * 10 + j
            plt.plot(time, [o[idx] for o in obs_seq], label=f"{idx}")
        plt.ylabel("obs")
        plt.legend(fontsize=6, ncol=2)
    # plot state
    for i, (name, value) in enumerate(state_seq[0].__dict__.items()):
        if name in ["pos_traj", "vel_traj"]:
            continue
        elif (("pos" in name) or ("vel" in name)) and ("tar" not in name):
            xyz = np.array([getattr(s, name) for s in state_seq])
            xyz_tar = np.array([getattr(s, name[:3] + "_tar") for s in state_seq])
            for i, subplot_name in zip(range(3), ["x", "y", "z"]):
                current_fig += 1
                plt.subplot(num_rows, 6, current_fig)
                plt.plot(time, xyz[:, i], label=f"{subplot_name}")
                plt.plot(time, xyz_tar[:, i], "--", label=f"{subplot_name}_tar")
                plt.ylabel(name + "_" + subplot_name)
                plt.legend()
        else:
            current_fig += 1
            plt.subplot(num_rows, 6, current_fig)
            plt.plot(time, [getattr(s, name) for s in state_seq])
            plt.ylabel(name)

    plt.xlabel("time")
    plt.savefig(f"{quadjax.get_package_path()}/../results/plot.png")
    print(f"plotting time: {time_module.time()-t0:.2f}s")

    # save state_seq (which is a list of EnvState3D:flax.struct.dataclass)
    with open(f"{quadjax.get_package_path()}/../results/state_seq.pkl", "wb") as f:
        pickle.dump(state_seq, f)

@pydataclass
class Args:
    mode: str = "render"
    task: str = "tracking"
    controller: str = "mppi"
    controller_params: str = ""
    debug: bool = False


def main(args: Args):
    env = Quad3D(task=args.task)

    # setup controllers
    if args.controller == 'mppi':
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
    elif args.controller == 'fixed':
        thrust_hover = env.default_params.m * env.default_params.g
        thrust_hover_normed = (thrust_hover / env.default_params.max_thrust) * 2.0 - 1.0
        control_params = controllers.FixedParams(
            u = jnp.array([thrust_hover_normed, 0.0, 0.0, 0.0]) 
        )
        controller = controllers.FixedController(env, control_params)
    elif args.controller == 'nn':
        from quadjax.train import ActorCritic
        network = ActorCritic(env.action_dim, activation='tanh')
        # initialize network parameter load form file
        control_params = pickle.load(open(f"{quadjax.get_package_path()}/../results/ppo_params.pkl", "rb"))
        controller = controllers.NetworkController(network.apply, env, control_params)
    else:
        raise NotImplementedError
    
    # run env
    filename = f'{args.controller}_{args.controller_params}'
    if args.mode == 'render':
        with jax.disable_jit():
            render_env(env, controller=controller, control_params=control_params, repeat_times=1, filename=filename)
    elif args.mode == 'eval':
        eval_env(env, controller=controller, control_params=control_params, total_steps=30000, filename=filename)
    else:
        raise NotImplementedError

if __name__ == "__main__":
    main(tyro.cli(Args))