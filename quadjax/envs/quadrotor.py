import jax
import jax.numpy as jnp
from jax import lax
from typing import Tuple
import chex
from functools import partial
from dataclasses import dataclass as pydataclass
import tyro
import pickle
import time as time_module
import numpy as np
from tqdm import trange
import os

import quadjax
from quadjax import dynamics as quad_dyn
from quadjax import controllers
from quadjax.dynamics import utils
from quadjax.dynamics.dataclass import EnvParams3D, EnvState3D, Action3D
from quadjax.envs.base import BaseEnvironment


class Quad3D(BaseEnvironment):
    """
    JAX Compatible version of Quad3D-v0 OpenAI gym environment. Source:
    github.com/openai/gym/blob/master/gym/envs/classic_control/Quad3D.py
    """

    def __init__(
        self,
        task: str = "tracking",
        obs_type: str = "quad",
        enable_randomizer: bool = True,
        lower_controller: str = "base",
        disturb_type: str = "periodic",
        disable_rollover_terminate: bool = False,
        generate_noisy_state: bool = False,
    ):
        """Initialize Quad3D-v0 environment."""

        # base class initialization
        super().__init__()

        # task related parameters
        self.task = task
        self.disable_rollover_terminate = disable_rollover_terminate
        self.generate_noisy_state = generate_noisy_state

        # reference trajectory function
        if task == "tracking":
            self.generate_traj = partial(
                utils.generate_lissa_traj,
                self.default_params.max_steps_in_episode,
                self.default_params.dt,
            )
            self.reward_fn = utils.tracking_penyaw_reward_fn
            self.get_init_state = self.get_zero_state
        elif task == "tracking_slow":
            # Note: this is the version with quadratic cost, which is preferred with large model error
            self.generate_traj = partial(
                utils.generate_lissa_traj_slow,
                self.default_params.max_steps_in_episode,
                self.default_params.dt,
            )
            self.reward_fn = utils.tracking_realworld_reward_fn
            self.get_init_state = self.get_zero_state
        elif task == "tracking_zigzag":
            self.generate_traj = partial(
                utils.generate_zigzag_traj,
                self.default_params.max_steps_in_episode,
                self.default_params.dt,
            )
            self.reward_fn = utils.tracking_penyaw_reward_fn
            self.get_init_state = self.get_zero_state
        elif task == "hovering":
            self.generate_traj = partial(
                utils.generate_fixed_traj,
                self.default_params.max_steps_in_episode,
                self.default_params.dt,
            )
            self.reward_fn = utils.tracking_penyaw_reward_fn
            self.get_init_state = self.get_zero_state
        else:
            raise NotImplementedError

        # dynamics function
        self.step_fn, self.dynamics_fn = quad_dyn.get_quadrotor_1st_order_dyn(
            disturb_type=disturb_type
        )
        self.get_err_pos = lambda state: jnp.linalg.norm(state.pos_tar - state.pos)
        self.get_err_vel = lambda state: jnp.linalg.norm(state.vel_tar - state.vel)

        # lower-level controller
        if lower_controller == "base":
            self.default_control_params = 0.0

            def base_controller_fn(obs, state, env_params, rng_act, input_action):
                return input_action, None, state

            self.control_fn = base_controller_fn
        elif lower_controller == "l1":
            self.default_control_params = controllers.L1Params()
            controller = controllers.L1Controller(self, self.default_control_params)

            def l1_control_fn(obs, state, env_params, rng_act, input_action):
                action_l1, control_params, _ = controller(
                    obs, state, env_params, rng_act, state.control_params, 0.0
                )
                state = state.replace(control_params=control_params)
                return (action_l1 + input_action), None, state

            self.control_fn = l1_control_fn
        elif lower_controller == "l1_esitimate_only":
            self.default_control_params = controllers.L1Params()
            controller = controllers.L1Controller(self, self.default_control_params)

            def l1_esitimate_only_control_fn(
                obs, state, env_params, rng_act, input_action
            ):
                _, control_params, _ = controller(
                    obs, state, env_params, rng_act, state.control_params, 0.0
                )
                state = state.replace(control_params=control_params)
                return input_action, None, state

            self.control_fn = l1_esitimate_only_control_fn
        else:
            raise NotImplementedError
        self.sim_dt = self.default_params.dt
        self.substeps = 1  # NOTE: if you want to run lower-level controllers, you can modify it to > 1

        # domain randomization function
        if enable_randomizer:

            def sample_random_params(key: chex.PRNGKey) -> EnvParams3D:
                param_key = jax.random.split(key)[0]
                rand_val = jax.random.uniform(
                    param_key, shape=(17,), minval=-1.0, maxval=1.0
                )

                params = self.default_params
                m = params.m_mean + rand_val[0] * params.m_std
                I_diag = params.I_diag_mean + rand_val[1:4] * params.I_diag_std
                I = jnp.diag(I_diag)
                action_scale = (
                    params.action_scale_mean + rand_val[4] * params.action_scale_std
                )
                alpha_bodyrate = (
                    params.alpha_bodyrate_mean + rand_val[5] * params.alpha_bodyrate_std
                )

                disturb_params = rand_val[6:12] * params.disturb_scale

                return EnvParams3D(
                    m=m,
                    I=I,
                    action_scale=action_scale,
                    alpha_bodyrate=alpha_bodyrate,
                    disturb_params=disturb_params,
                )

            self.sample_params = sample_random_params
        else:

            def sample_default_params(key: chex.PRNGKey) -> EnvParams3D:
                disturb_params = jax.random.uniform(
                    key, shape=(6,), minval=-1.0, maxval=1.0
                )
                return EnvParams3D(disturb_params=disturb_params)

            self.sample_params = sample_default_params
        if enable_randomizer and "params" not in obs_type:
            print("Warning: enable domain randomziation without params in obs_type")

        # observation function
        if obs_type == "quad_params":
            # add parameters to observation
            self.get_obs = self.get_obs_quad_params
            self.obs_dim = 39 + self.default_params.traj_obs_len * 6
        elif obs_type == "quad":
            # only observe quadrotor state
            self.get_obs = self.get_obs_quadonly
            self.obs_dim = 19 + self.default_params.traj_obs_len * 6
        elif obs_type == "quad_l1":
            # add l1 adaptive controller to observation
            assert (
                "l1" in lower_controller
            ), "quad_l1 obs_type only works with l1 lower controller"
            self.get_obs = self.get_obs_quad_l1
            self.obs_dim = 25 + self.default_params.traj_obs_len * 6
        else:
            raise NotImplementedError

        # equibrium point
        self.equib = jnp.array([0.0] * 6 + [1.0] + [0.0] * 9)  # size=16

        # RL parameters
        self.action_dim = 4
        self.adapt_obs_dim = 22 * self.default_params.adapt_horizon
        self.param_obs_dim = 20

    """
    environment properties
    """

    @property
    def default_params(self) -> EnvParams3D:
        """Default environment parameters for Quad3D-v0."""
        return EnvParams3D()

    """
    key methods
    """

    def step_env(
        self,
        key: chex.PRNGKey,
        state: EnvState3D,
        action: jnp.ndarray,
        params: EnvParams3D,
        deterministic: bool = False,
    ) -> Tuple[chex.Array, EnvState3D, float, bool, dict]:
        action = jnp.clip(action, -1.0, 1.0)

        # run one step with lower-level controller
        def step_once(carried, _):
            key, state, action, params = carried
            # call controller to get sub_action and new_control_params
            sub_action, _, state = self.control_fn(None, state, params, key, action)
            next_state = self.raw_step(key, state, sub_action, params)
            return (key, next_state, action, params), None

        # disable noise in parameters if deterministic
        dyn_noise_scale = params.dyn_noise_scale * (1.0 - deterministic)
        params = params.replace(dyn_noise_scale=dyn_noise_scale)

        # call lax.scan to get next_state
        (_, next_state, _, params), _ = lax.scan(
            step_once, (key, state, action, params), jnp.arange(self.substeps)
        )

        # get observation, reward, done, info
        reward = self.reward_fn(state, params)
        done = self.is_terminal(state, params)
        info_key, key = jax.random.split(key)
        info = self.get_info(info_key, state, next_state, params)
        obs = self.get_obs(next_state, params)
        return (obs, next_state, reward, done, info)

    def raw_step(
        self,
        key: chex.PRNGKey,
        state: EnvState3D,
        sub_action: jnp.ndarray,
        params: EnvParams3D,
    ) -> EnvState3D:
        """Run one step of the environment dynamics."""
        sub_action = jnp.clip(sub_action, -1.0, 1.0)
        thrust = (sub_action[0] + 1.0) / 2.0 * params.max_thrust
        torque = sub_action[1:] * params.max_torque
        env_action = Action3D(thrust=thrust, torque=torque)
        key, step_key = jax.random.split(key)
        return self.step_fn(params, state, env_action, step_key, self.sim_dt)

    def get_zero_state(self, key: chex.PRNGKey, params: EnvParams3D) -> EnvState3D:
        """Reset environment state by sampling theta, theta_dot."""
        traj_key, disturb_key, key = jax.random.split(key, 3)
        # generate reference trajectory by adding a few sinusoids together
        pos_traj, vel_traj, acc_traj = self.generate_traj(traj_key)
        zeros3 = jnp.zeros(3, dtype=jnp.float32)
        vel_hist = jnp.zeros(
            (self.default_params.adapt_horizon + 2, 3), dtype=jnp.float32
        )
        omega_hist = jnp.zeros(
            (self.default_params.adapt_horizon + 2, 3), dtype=jnp.float32
        )
        action_hist = jnp.zeros(
            (self.default_params.adapt_horizon + 2, 4), dtype=jnp.float32
        )
        return EnvState3D(
            # drone
            pos=zeros3,
            vel=zeros3,
            omega=zeros3,
            omega_tar=zeros3,
            quat=jnp.concatenate([zeros3, jnp.array([1.0])]),
            # trajectory
            pos_tar=pos_traj[0],
            vel_tar=vel_traj[0],
            acc_tar=acc_traj[0],
            pos_traj=pos_traj,
            vel_traj=vel_traj,
            acc_traj=acc_traj,
            # debug value
            last_thrust=0.0,
            last_torque=zeros3,
            # step
            time=0,
            # disturbanceself
            f_disturb=jax.random.uniform(
                disturb_key,
                shape=(3,),
                minval=-params.disturb_scale,
                maxval=params.disturb_scale,
            ),
            # trajectory information for adaptation
            vel_hist=vel_hist,
            omega_hist=omega_hist,
            action_hist=action_hist,
            # control parameters
            control_params=self.default_control_params,
        )

    def get_info(
        self,
        rng: chex.PRNGKey,
        state: EnvState3D,
        next_state: EnvState3D,
        params: EnvParams3D,
    ) -> dict:
        """Get additional information about the environment."""
        if self.generate_noisy_state:
            rng_pos, rng_vel, rng_quat, rng_omega, rng = jax.random.split(rng, 5)
            obs_noise_scale = self.default_params.obs_noise_scale
            pos_noise = (
                jax.random.normal(rng_pos, shape=next_state.pos.shape)
                * obs_noise_scale
                * 0.25
            )
            vel_noise = (
                jax.random.normal(rng_vel, shape=next_state.vel.shape)
                * obs_noise_scale
                * 0.5
            )
            quat_noise = (
                jax.random.normal(rng_quat, shape=next_state.quat.shape)
                * obs_noise_scale
                * 0.02
            )
            omega_noise = (
                jax.random.normal(rng_omega, shape=next_state.omega.shape)
                * obs_noise_scale
                * 0.5
            )
            noisy_state = next_state.replace(
                pos=next_state.pos + pos_noise,
                vel=next_state.vel + vel_noise,
                quat=next_state.quat + quat_noise,
                omega=next_state.omega + omega_noise,
            )
        else:
            noisy_state = None
        info = {
            "discount": self.discount(state, params),
            "err_pos": self.get_err_pos(state),
            "err_vel": self.get_err_vel(state),
            "obs_param": self.get_obs_paramsonly(state, params),
            "obs_adapt": self.get_obs_adapt_hist(state, params),
            "noisy_state": noisy_state,
        }
        return info

    def reset_env(
        self, key: chex.PRNGKey, params: EnvParams3D
    ) -> Tuple[chex.Array, EnvState3D]:
        """Reset environment state."""
        state = self.get_init_state(key, params)
        info_key, key = jax.random.split(key)
        info = self.get_info(info_key, state, state, params)
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
            state.vel / 3.0,
            state.quat,
            state.omega / 5.0,  # 3*3+4=13
            # trajectory
            state.pos_tar,
            state.vel_tar / 3.0,  # 3*2=6
            state.pos_traj[indices].flatten(),
            state.vel_traj[indices].flatten() / 3.0,
        ]  # 13+6=19
        obs = jnp.concatenate(obs_elements, axis=-1)

        return obs

    @partial(jax.jit, static_argnums=(0,))
    def get_obs_adapt_hist(self, state: EnvState3D, params: EnvParams3D) -> chex.Array:
        """
        Return the history of velocity, acceleration, and jerk
        """
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
        """Return parameter observation by normalizing them."""
        obs_elements = [
            # parameter observation
            # I
            (params.I.diagonal() - params.I_diag_mean) / params.I_diag_std,
            # disturbance
            (state.f_disturb) / params.disturb_scale,
            # hook offset
            (params.hook_offset - params.hook_offset_mean) / params.hook_offset_std,
            # disturbance parameters
            params.disturb_params,
            jnp.array(
                [
                    # mass
                    (params.m - params.m_mean) / params.m_std,
                    # action_scale
                    (params.action_scale - params.action_scale_mean)
                    / params.action_scale_std,
                    # 1st order alpha
                    (params.alpha_bodyrate - params.alpha_bodyrate_mean)
                    / params.alpha_bodyrate_std,
                ]
            ),
        ]  # 13+6=19
        obs = jnp.concatenate(obs_elements, axis=-1)
        return obs

    @partial(jax.jit, static_argnums=(0,))
    def get_obs_l1only(self, state: EnvState3D, params: EnvParams3D) -> chex.Array:
        """Return l1 adaptive controller information."""
        obs_elements = [
            # l1 observation
            state.control_params.vel_hat,
            state.control_params.d_hat,
        ]
        obs = jnp.concatenate(obs_elements, axis=-1)
        return obs

    @partial(jax.jit, static_argnums=(0,))
    def get_obs_quad_params(self, state: EnvState3D, params: EnvParams3D) -> chex.Array:
        """Return angle in polar coordinates and change."""
        quad_obs = self.get_obs_quadonly(state, params)
        param_obs = self.get_obs_paramsonly(state, params)
        return jnp.concatenate([quad_obs, param_obs], axis=-1)

    @partial(jax.jit, static_argnums=(0,))
    def get_obs_quad_l1(self, state: EnvState3D, params: EnvParams3D) -> chex.Array:
        """Return angle in polar coordinates and change."""
        quad_obs = self.get_obs_quadonly(state, params)
        l1_obs = self.get_obs_l1only(state, params)
        return jnp.concatenate([quad_obs, l1_obs], axis=-1)

    @partial(jax.jit, static_argnums=(0,))
    def is_terminal(self, state: EnvState3D, params: EnvParams3D) -> bool:
        """Check whether state is terminal."""
        # Check number of steps in episode termination condition
        done = (state.time >= params.max_steps_in_episode) | (
            jnp.abs(state.pos) > 3.0
        ).any()
        if not self.disable_rollover_terminate:
            rollover = (state.quat[3] < jnp.cos(jnp.pi / 4.0)) | (
                jnp.abs(state.omega) > 100.0
            ).any()
            done = done | rollover
        if self.task == "jumping":
            done = (
                done
                | (
                    (state.pos[0] < 0)
                    & (state.pos_obj[0] < 0)
                    & (jnp.linalg.norm(state.vel) < 1.0)
                    & (jnp.linalg.norm(state.vel_obj) < 1.0)
                )
                | (utils.get_hit_reward(state.pos_obj, params) < -0.5)
                | (utils.get_hit_reward(state.pos, params) < -0.5)
            )
        return done


def eval_env(
    env: Quad3D,
    controller: controllers.BaseController,
    total_steps=30000,
    filename="",
):
    """
    Evaluate the environment with a given controller
    """

    # running environment
    rng = jax.random.PRNGKey(1)

    # run one step
    def run_one_step(carry, _):
        obs, env_state, rng, env_params, control_params, env_infos = carry
        rng, rng_act, rng_step, rng_control = jax.random.split(rng, 4)
        action, control_params, control_info = controller(
            obs, env_state, env_params, rng_act, control_params, env_infos
        )
        # for PPO, use mean action
        if control_info is not None:
            if "a_mean" in control_info:
                action = control_info["a_mean"]
        next_obs, next_env_state, reward, done, info = env.step(
            rng_step, env_state, action, env_params
        )
        # if done, reset controller parameters, aviod use if, use lax.cond instead
        rng, rng_control = jax.random.split(rng)
        return (next_obs, next_env_state, rng, env_params, control_params, info), (
            info["err_pos"],
            done,
        )

    t0 = time_module.time()

    def run_one_ep(rng_reset, rng):
        env_params = env.default_params

        obs, info, env_state = env.reset(rng_reset, env_params)

        rng_control, rng = jax.random.split(rng)
        control_params = controller.reset(
            env_state, env_params, controller.init_control_params, rng_control
        )

        (obs, env_state, rng, env_params, control_params, env_infos), (
            err_pos,
            dones,
        ) = lax.scan(
            run_one_step,
            (obs, env_state, rng, env_params, control_params, info),
            jnp.arange(env.default_params.max_steps_in_episode),
        )
        return rng, err_pos

    # calculate cumulative err_pos bewteen each done
    run_one_ep_jit = jax.jit(run_one_ep)
    num_eps = int(total_steps // env.default_params.max_steps_in_episode)
    err_pos_ep = []
    num_trajs = 4
    rng, rng_reset_meta = jax.random.split(rng)
    rng_reset_list = jax.random.split(rng_reset_meta, num_trajs)
    for i, rng_reset in enumerate(rng_reset_list):
        print(f"[DEBUG] test traj {i+1}")
        for _ in trange(num_eps // num_trajs):
            rng, err_pos = run_one_ep_jit(rng_reset, rng)
            err_pos_ep.append(err_pos.mean())
    err_pos_ep = jnp.array(err_pos_ep)
    # print mean and std of err_pos
    pos_mean, pos_std = jnp.mean(err_pos_ep), jnp.std(err_pos_ep)
    print(f"env running time: {time_module.time()-t0:.2f}s")
    print(f"err_pos mean: {pos_mean:.3f}, std: {pos_std:.3f}")
    print(f"${pos_mean*100:.2f} \pm {pos_std*100:.2f}$")

    # check if f"{quadjax.get_package_path()}/../results" folder exists, if not, create one
    save_path = f"{quadjax.get_package_path()}/../results"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        print("[DEBUG] create folder ", save_path)

    # save data
    with open(
        f"{quadjax.get_package_path()}/../results/eval_err_pos_{filename}.pkl", "wb"
    ) as f:
        pickle.dump(np.array(err_pos_ep), f)


def render_env(env: Quad3D, controller, control_params, repeat_times=1, filename=""):
    """
    Render the environment with a given controller
    """
    # running environment
    rng = jax.random.PRNGKey(1)
    rng, rng_params = jax.random.split(rng)
    env_params = env.sample_params(rng_params)

    state_seq, obs_seq, reward_seq = [], [], []
    control_seq = []
    rng, rng_reset = jax.random.split(rng)
    obs, info, env_state = env.reset(rng_reset, env_params)

    rng, rng_control = jax.random.split(rng)
    control_params = controller.reset(
        env_state, env_params, controller.init_control_params, rng_control
    )
    n_dones = 0

    t0 = time_module.time()
    while n_dones < repeat_times:
        state_seq.append(env_state)
        rng, rng_act, rng_step = jax.random.split(rng, 3)
        action, control_params, control_info = controller(
            obs, env_state, env_params, rng_act, control_params, info
        )
        # manually record certain control parameters into state_seq
        if hasattr(control_params, "d_hat") and hasattr(control_params, "vel_hat"):
            control_seq.append(
                {"d_hat": control_params.d_hat, "vel_hat": control_params.vel_hat}
            )
        if hasattr(control_params, "a_hat"):
            control_seq.append({"a_hat": control_params.a_hat})
        if hasattr(control_params, "quat_desired"):
            control_seq.append({"quat_desired": control_params.quat_desired})
        next_obs, next_env_state, reward, done, info = env.step(
            rng_step, env_state, action, env_params
        )
        if done:
            rng, rng_params = jax.random.split(rng)
            env_params = env.sample_params(rng_params)
            rng, rng_control = jax.random.split(rng)
            control_params = controller.reset(
                env_state, env_params, control_params, rng_control
            )
            n_dones += 1

        reward_seq.append(reward)
        obs_seq.append(obs)
        obs = next_obs
        env_state = next_env_state
    print(f"env running time: {time_module.time()-t0:.2f}s")

    # check if f"{quadjax.get_package_path()}/../results" folder exists, if not, create one
    save_path = f"{quadjax.get_package_path()}/../results"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        print("[DEBUG] create folder ", save_path)

    t0 = time_module.time()
    # convert state into dict
    state_seq_dict = [s.__dict__ for s in state_seq]
    if len(control_seq) > 0:
        # merge control_seq into state_seq with dict
        for i in range(len(state_seq)):
            state_seq_dict[i] = {**state_seq_dict[i], **control_seq[i]}
    utils.plot_states(state_seq_dict, obs_seq, reward_seq, env_params, filename)
    print(f"plotting time: {time_module.time()-t0:.2f}s")

    file_path = f"{quadjax.get_package_path()}/../results/state_seq_{filename}.pkl"
    with open(file_path, "wb") as f:
        pickle.dump(state_seq_dict, f)
    print("[DEBUG] state sequence saved to ", file_path)


def get_controller(env, controller_name, controller_params=None, debug=False):
    def parse_sample_params(param_text: str):
        # parse in format "N{sample_number}_H{horizon}_lam{lam}"
        if param_text == "":
            N = 8192
            H = 32
            lam = 0.01
            sigma = 0.5
        else:
            N = int(param_text.split("_")[0][1:])
            H = int(param_text.split("_")[1][1:])
            lam = float(param_text.split("_")[2][3:])
            sigma = 0.5
        return N, H, lam, sigma

    def get_sample_mean(env):
        thrust_hover = env.default_params.m * env.default_params.g
        thrust_hover_normed = (thrust_hover / env.default_params.max_thrust) * 2.0 - 1.0
        a_mean_per_step = jnp.array([thrust_hover_normed, 0.0, 0.0, 0.0])
        a_mean = jnp.tile(a_mean_per_step, (H, 1))
        return a_mean

    if controller_name == "pid":
        control_params = controllers.PIDParams(
            Kp=10.0,
            Kd=5.0,
            Ki=0.0,
            Kp_att=10.0,
        )
        controller = controllers.PIDController(env, control_params=control_params)
    elif controller_name == "l1":
        control_params = controllers.L1Params()
        controller = controllers.L1Controller(env, control_params)
    elif controller_name == "random":
        control_params = None
        controller = controllers.RandomController(env, control_params)
    elif controller_name == "fixed":
        control_params = controllers.FixedParams(
            u=jnp.asarray([0.0, 0.0, 0.0, 0.0]),
        )
        controller = controllers.FixedController(env, control_params=control_params)
    elif controller_name == "mppi":
        N, H, lam, sigma = parse_sample_params(controller_params)
        if debug:
            N, H = 4, 2
            print(f"[DEBUG], override controller parameters to be: N={N}, H={H}")
        a_mean = get_sample_mean(env)
        if controller_name == "mppi":
            sigmas = jnp.array([sigma] * env.action_dim)
            a_cov_per_step = jnp.diag(sigmas**2)
            # a_cov_per_step = jnp.diag(jnp.array([sigma**2] * env.action_dim))
            a_cov = jnp.tile(a_cov_per_step, (H, 1, 1))
            control_params = controllers.MPPIParams(
                gamma_mean=1.0,
                gamma_sigma=0.0,
                discount=1.0,
                sample_sigma=sigma,
                a_mean=a_mean,
                a_cov=a_cov,
            )
            controller = controllers.MPPIController(
                env=env, control_params=control_params, N=N, H=H, lam=lam
            )
    elif "covo" in controller_name:
        N, H, lam, sigma = parse_sample_params(controller_params)
        if debug:
            N, H = 4, 2
            print(f"[DEBUG], override controller parameters to be: N={N}, H={H}")
        a_mean = get_sample_mean(env)
        a_cov = jnp.diag(jnp.ones(H * env.action_dim) * sigma**2)
        if "online" in controller_name:
            mode = "online"
        elif "offline" in controller_name:
            mode = "offline"
        else:
            mode = "online"
            print("[DEBUG] unset mode, CoVO mode set to online")
        control_params = controllers.CoVOParams(
            gamma_mean=1.0,
            gamma_sigma=0.0,
            discount=1.0,
            sample_sigma=sigma,
            a_mean=a_mean,
            a_cov=a_cov,
            a_cov_offline=jnp.zeros((H, env.action_dim, env.action_dim)),
        )
        controller = controllers.CoVOController(
            env=env, control_params=control_params, N=N, H=H, lam=lam, mode=mode
        )
    elif controller_name == "nn":
        from quadjax.train import ActorCritic

        network = ActorCritic(env.action_dim, activation="tanh")
        if controller_params == "":
            file_path = "ppo_params_"
        else:
            file_path = f"{controller_params}"
        control_params = pickle.load(
            open(f"{quadjax.get_package_path()}/../results/{file_path}.pkl", "rb")
        )

        def apply_fn(train_params, last_obs, env_info):
            return network.apply(train_params, last_obs)

        controller = controllers.NetworkController(apply_fn, env, control_params)
    elif controller_name == "RMA":
        from quadjax.train import ActorCritic, Compressor, Adaptor

        network = ActorCritic(env.action_dim, activation="tanh")
        adaptor = Adaptor()
        if controller_params == "":
            file_path = "ppo_params_"
        else:
            file_path = f"{controller_params}"
        control_params = pickle.load(
            open(f"{quadjax.get_package_path()}/../results/{file_path}.pkl", "rb")
        )

        def apply_fn(train_params, last_obs, env_info):
            adapted_last_obs = adaptor.apply(train_params[2], env_info["obs_adapt"])
            obs = jnp.concatenate([last_obs, adapted_last_obs], axis=-1)
            pi, value = network.apply(train_params[0], obs)
            return pi, value

        controller = controllers.NetworkController(apply_fn, env, control_params)
    else:
        raise NotImplementedError
    return controller, control_params


@pydataclass
class Args:
    task: str = "tracking"  # tracking, tracking_zigzag, hovering
    controller: str = "lqr"  # fixed
    controller_params: str = ""
    obs_type: str = "quad"
    debug: bool = False
    mode: str = "render"  # eval, render
    lower_controller: str = "base"
    noDR: bool = False
    disturb_type: str = "gaussian"  # periodic, sin, drag, gaussian, none
    name: str = ""


def main(args: Args):
    if args.debug:
        jax.config.update("jax_debug_nans", True)

    env = Quad3D(
        task=args.task,
        obs_type=args.obs_type,
        lower_controller=args.lower_controller,
        enable_randomizer=not args.noDR,
        disturb_type=args.disturb_type,
        disable_rollover_terminate=True,
        generate_noisy_state=True,
    )

    print("starting test...")
    controller, control_params = get_controller(
        env, args.controller, args.controller_params
    )
    if args.mode == "eval":
        eval_env(
            env,
            controller=controller,
            total_steps=300 * 4 * 10,
            filename=args.name,
        )
    elif args.mode == "render":
        render_env(
            env,
            controller=controller,
            control_params=control_params,
            repeat_times=1,
            filename=args.name,
        )
    else:
        raise NotImplementedError


if __name__ == "__main__":
    main(tyro.cli(Args))
