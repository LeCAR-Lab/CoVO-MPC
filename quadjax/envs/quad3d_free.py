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
from tqdm import trange

import quadjax
from quadjax import dynamics as quad_dyn
from quadjax import controllers
from quadjax.dynamics import utils, geom
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
        dynamics: str = "free",
        obs_type: str = "quad",
        enable_randomizer: bool = True,
        lower_controller: str = "base",
        disturb_type: str = "periodic",
        disable_rollover_terminate: bool = False,
    ):
        super().__init__()
        self.task = task
        self.disable_rollover_terminate = disable_rollover_terminate
        # reference trajectory function
        self.task = task
        if task == "tracking":
            self.generate_traj = partial(
                utils.generate_lissa_traj,
                self.default_params.max_steps_in_episode,
                self.default_params.dt,
            )
            self.reward_fn = utils.tracking_penyaw_reward_fn
            self.get_init_state = self.fixed_init_state
        elif task == "tracking_slow":
            self.generate_traj = partial(utils.generate_lissa_traj_slow, self.default_params.max_steps_in_episode, self.default_params.dt)
            self.reward_fn = utils.tracking_realworld_reward_fn
            self.get_init_state = self.fixed_init_state
        elif task == "tracking_zigzag":
            self.generate_traj = partial(
                utils.generate_zigzag_traj,
                self.default_params.max_steps_in_episode,
                self.default_params.dt,
            )
            self.reward_fn = utils.tracking_penyaw_reward_fn
            self.get_init_state = self.fixed_init_state
        elif task in "jumping":
            self.generate_traj = partial(
                utils.generate_jumping_fixed_traj,
                self.default_params.max_steps_in_episode,
                self.default_params.dt,
            )
            self.reward_fn = utils.jumping_obj_reward_fn
            self.get_init_state = self.sample_init_state
        elif task == "hovering":
            self.generate_traj = partial(
                utils.generate_fixed_traj,
                self.default_params.max_steps_in_episode,
                self.default_params.dt,
            )
            self.reward_fn = utils.tracking_realworld_reward_fn
            self.get_init_state = self.fixed_init_state
        else:
            raise NotImplementedError
        # dynamics function
        if dynamics == "free":
            self.step_fn, self.dynamics_fn = quad_dyn.get_free_dynamics_3d()
            self.update_time = lambda x: x
            self.get_err_pos = lambda state: jnp.linalg.norm(state.pos_tar - state.pos)
            self.get_err_vel = lambda state: jnp.linalg.norm(state.vel_tar - state.vel)
        elif dynamics == "dist_constant":
            self.step_fn, self.dynamics_fn = quad_dyn.get_free_dynamics_3d_disturbance(
                utils.constant_disturbance
            )
            self.update_time = lambda x: x
            self.get_err_pos = lambda state: jnp.linalg.norm(state.pos_tar - state.pos)
            self.get_err_vel = lambda state: jnp.linalg.norm(state.vel_tar - state.vel)
        elif dynamics == "bodyrate":
            self.step_fn, self.dynamics_fn = quad_dyn.get_free_dynamics_3d_bodyrate(
                disturb_type=disturb_type
            )
            self.update_time = lambda x: x
            self.get_err_pos = lambda state: jnp.linalg.norm(state.pos_tar - state.pos)
            self.get_err_vel = lambda state: jnp.linalg.norm(state.vel_tar - state.vel)
        elif dynamics == "slung":
            taut_dynamics, self.update_time = quad_dyn.get_taut_dynamics_3d()
            loose_dynamics, _update_time = quad_dyn.get_loose_dynamics_3d()
            dynamic_transfer = quad_dyn.get_dynamic_transfer_3d()

            def step_fn(params, state, env_action, key, sim_dt):
                old_loose_state = state.l_rope < (params.l - params.rope_taut_therehold)
                taut_state = taut_dynamics(params, state, env_action, key, sim_dt)
                loose_state = loose_dynamics(params, state, env_action, key, sim_dt)
                new_state = dynamic_transfer(
                    params, loose_state, taut_state, old_loose_state
                )
                return new_state

            self.step_fn = step_fn
            self.dynamics_fn = None
            self.get_err_pos = lambda state: jnp.linalg.norm(
                state.pos_tar - state.pos_obj
            )
            self.get_err_vel = lambda state: jnp.linalg.norm(
                state.vel_tar - state.vel_obj
            )
        else:
            raise NotImplementedError
        # lower-level controller
        if lower_controller == "base":
            self.default_control_params = 0.0

            def base_controller_fn(obs, state, env_params, rng_act, input_action):
                return input_action, None, state

            self.control_fn = base_controller_fn
            self.sub_steps = 1
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
            self.sub_steps = 1
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
            self.sub_steps = 1
        elif lower_controller == "nlac":
            self.default_control_params = controllers.NLACParams()
            controller = controllers.NLAdaptiveController(
                self, self.default_control_params
            )

            def nlac_control_fn(obs, state, env_params, rng_act, input_action):
                action_nlac, control_params, _ = controller(
                    obs, state, env_params, rng_act, state.control_params, 0.0
                )
                state = state.replace(control_params=control_params)
                return (action_nlac + input_action), None, state

            self.control_fn = nlac_control_fn
            self.sub_steps = 1
        elif lower_controller == "nlac_esitimate_only":
            self.default_control_params = controllers.NLACParams()
            controller = controllers.NLAdaptiveController(
                self, self.default_control_params
            )

            def nlac_esitimate_only_control_fn(
                obs, state, env_params, rng_act, input_action
            ):
                _, control_params, _ = controller(
                    obs, state, env_params, rng_act, state.control_params, 0.0
                )
                state = state.replace(control_params=control_params)
                return input_action, None, state

            self.control_fn = nlac_esitimate_only_control_fn
            self.sub_steps = 1
        elif lower_controller == "nlac":
            self.default_control_params = controllers.NLACParams()
            controller = controllers.NLAdaptiveController(
                self, self.default_control_params
            )

            def nlac_control_fn(obs, state, env_params, rng_act, input_action):
                action_nlac, control_params, _ = controller(
                    obs, state, env_params, rng_act, state.control_params, 0.0
                )
                state = state.replace(control_params=control_params)
                return (action_nlac + input_action), None, state

            self.control_fn = nlac_control_fn
            self.sub_steps = 1
        elif lower_controller == "nlac_esitimate_only":
            self.default_control_params = controllers.NLACParams()
            controller = controllers.NLAdaptiveController(
                self, self.default_control_params
            )

            def nlac_esitimate_only_control_fn(
                obs, state, env_params, rng_act, input_action
            ):
                _, control_params, _ = controller(
                    obs, state, env_params, rng_act, state.control_params, 0.0
                )
                state = state.replace(control_params=control_params)
                return input_action, None, state

            self.control_fn = nlac_esitimate_only_control_fn
            self.sub_steps = 1
        elif lower_controller == "pid_bodyrate":
            self.default_control_params = controllers.PIDParams(
                kp=jnp.array([30.0, 30.0, 30.0]),
                ki=jnp.array([3.0, 3.0, 3.0]) / self.default_params.dt,
                kd=jnp.array([0.0, 0.0, 0.0]),
                last_error=jnp.zeros(3),
                integral=jnp.zeros(3),
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
                torque = self.default_params.I @ u + jnp.cross(
                    state.omega, self.default_params.I @ state.omega
                )
                torque_normed = torque / self.default_params.max_torque
                action = jnp.concatenate([thrust_normed, torque_normed])
                return action, None, state

            self.control_fn = pid_controller_fn
            self.sub_steps = 5
        elif lower_controller == "l1_bodyrate":
            self.sub_steps = 5
            self.default_control_params = controllers.L1ParamsBodyrate()
            controller = controllers.L1ControllerBodyrate(
                self,
                self.default_control_params,
                sim_dt=self.default_params.dt / self.sub_steps,
            )

            def l1_controller_fn(obs, state, env_params, rng_act, input_action):
                thrust_normed = input_action[:1]
                omega_tar = input_action[1:] * self.default_params.max_omega
                state = state.replace(omega_tar=omega_tar)

                u, control_params, _ = controller(
                    obs, state, env_params, rng_act, state.control_params, None
                )

                state = state.replace(control_params=control_params)
                torque = self.default_params.I @ u + jnp.cross(
                    state.omega, self.default_params.I @ state.omega
                )
                torque_normed = torque / self.default_params.max_torque
                action = jnp.concatenate([thrust_normed, torque_normed])

                return action, None, state

            self.control_fn = l1_controller_fn
        else:
            raise NotImplementedError
        self.sim_dt = self.default_params.dt / self.sub_steps
        # sampling function
        if enable_randomizer:

            def sample_random_params(key: chex.PRNGKey) -> EnvParams3D:
                param_key = jax.random.split(key)[0]
                rand_val = jax.random.uniform(
                    param_key, shape=(17,), minval=-1.0, maxval=1.0
                )  # DEBUG * 0.0

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

                mo = params.mo_mean + rand_val[12] * params.mo_std
                l = params.l_mean + rand_val[13] * params.l_std
                hook_offset = (
                    params.hook_offset_mean + rand_val[14:17] * params.hook_offset_std
                )

                return EnvParams3D(
                    m=m,
                    I=I,
                    action_scale=action_scale,
                    alpha_bodyrate=alpha_bodyrate,
                    disturb_params=disturb_params,
                    mo=mo,
                    l=l,
                    hook_offset=hook_offset,
                )

            self.sample_params = sample_random_params
        else:

            def sample_default_params(key: chex.PRNGKey) -> EnvParams3D:
                disturb_params = jax.random.uniform(
                    key, shape=(6,), minval=-1.0, maxval=1.0
                )
                return EnvParams3D(disturb_params=disturb_params)

            self.sample_params = sample_default_params
        # observation function
        if dynamics == "slung" and ("obj" not in obs_type):
            obs_type = "quad_obj"
            print("Warning: obs_type is changed to quad_obj for slung dynamics")
        if enable_randomizer and "params" not in obs_type:
            print("Warning: enable domain randomziation without params in obs_type")
        if obs_type == "quad_params":
            self.get_obs = self.get_obs_quad_params
            self.obs_dim = 39 + self.default_params.traj_obs_len * 6
        elif obs_type == "quad":
            self.get_obs = self.get_obs_quadonly
            self.obs_dim = 19 + self.default_params.traj_obs_len * 6
        elif obs_type == "quad_l1":
            assert (
                "l1" in lower_controller
            ), "quad_l1 obs_type only works with l1 lower controller"
            self.get_obs = self.get_obs_quad_l1
            self.obs_dim = 25 + self.default_params.traj_obs_len * 6
        elif obs_type == "quad_nlac":
            assert (
                "nlac" in lower_controller
            ), "quad_nlac obs_type only works with nlac lower controller"
            self.get_obs = self.get_obs_quad_nlac
            self.obs_dim = 37 + self.default_params.traj_obs_len * 6
        elif obs_type == "quad_obj":
            self.get_obs = self.get_obs_quad_obj
            self.obs_dim = 42 + self.default_params.traj_obs_len * 6
        elif obs_type == "quad_obj_params":
            self.get_obs = self.get_obs_quad_obj_params
            self.obs_dim = 62 + self.default_params.traj_obs_len * 6
        else:
            raise NotImplementedError
        # equibrium point
        self.equib = jnp.array([0.0] * 6 + [1.0] + [0.0] * 9) # size=16
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

    """
    key methods
    """

    def step_env(
        self,
        key: chex.PRNGKey,
        state: EnvState3D,
        action: jnp.ndarray,
        params: EnvParams3D,
    ) -> Tuple[chex.Array, EnvState3D, float, bool, dict]:
        action = jnp.clip(action, -1.0, 1.0)

        def step_once(carried, _):
            key, state, action, params = carried
            # call controller to get sub_action and new_control_params
            sub_action, _, state = self.control_fn(None, state, params, key, action)
            next_state = self.raw_step(key, state, sub_action, params)
            return (key, next_state, action, params), None

        # call lax.scan to get next_state
        (_, next_state, _, params), _ = lax.scan(
            step_once, (key, state, action, params), jnp.arange(self.sub_steps)
        )
        next_state = self.update_time(next_state)
        return self.get_obs_state_reward_done_info(state, next_state, params)

    def step_env_wocontroller(
        self,
        key: chex.PRNGKey,
        state: EnvState3D,
        sub_action: jnp.ndarray,
        params: EnvParams3D,
        deterministic: bool = True,
    ) -> Tuple[chex.Array, EnvState3D, float, bool, dict]:
        # TODO: merge into raw_step
        # disable noise in parameters
        dyn_noise_scale = params.dyn_noise_scale * (1.0-deterministic)
        params = params.replace(dyn_noise_scale=dyn_noise_scale)
        return self.step_env(key, state, sub_action, params)

    def step_env_wocontroller_gradient(
        self,
        key: chex.PRNGKey,
        state: EnvState3D,
        action: jnp.ndarray,
        params: EnvParams3D,
        deterministic: bool = True, 
    ) -> Tuple[chex.Array, EnvState3D, float, bool, dict]:
        # TODO: merge into raw_step
        action = jnp.clip(action, -1.0, 1.0)

        def step_once(carried, _):
            key, state, action, params = carried
            # call controller to get sub_action and new_control_params
            sub_action, _, state = self.control_fn(None, state, params, key, action)
            next_state = self.raw_step(key, state, sub_action, params)
            return (key, next_state, action, params), None

        # disable noise in parameters
        dyn_noise_scale = params.dyn_noise_scale * (1.0-deterministic)
        params = params.replace(dyn_noise_scale=dyn_noise_scale)

        # call lax.scan to get next_state
        (_, next_state, _, params), _ = lax.scan(
            step_once, (key, state, action, params), jnp.arange(self.sub_steps)
        )
        next_state = self.update_time(next_state)
        return self.get_obs_state_reward_done_info_gradient(state, next_state, params)

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
        key, step_key = jax.random.split(key)
        return self.step_fn(params, state, env_action, step_key, self.sim_dt)

    def get_obs_state_reward_done_info_gradient(
        self,
        state: EnvState3D,
        next_state: EnvState3D,
        params: EnvParams3D,
    ) -> Tuple[chex.Array, EnvState3D, float, bool, dict]:
        reward = self.reward_fn(state, params)
        done = self.is_terminal(state, params)
        return (
            self.get_obs(next_state, params),
            next_state,
            reward,
            done,
            self.get_info(state, params),
        )

    def get_obs_state_reward_done_info(
        self,
        state: EnvState3D,
        next_state: EnvState3D,
        params: EnvParams3D,
    ) -> Tuple[chex.Array, EnvState3D, float, bool, dict]:
        obs, state, reward, done, info = self.get_obs_state_reward_done_info_gradient(
            state, next_state, params
        )
        obs = lax.stop_gradient(obs)
        state = lax.stop_gradient(state)
        return (
            obs,
            state,
            reward,
            done,
            info,
        )

    def sample_init_state(self, key: chex.PRNGKey, params: EnvParams3D) -> EnvState3D:
        """Reset environment state by sampling theta, theta_dot."""
        traj_key, disturb_key, key = jax.random.split(key, 3)
        # generate reference trajectory by adding a few sinusoids together
        pos_traj, vel_traj, acc_traj = self.generate_traj(traj_key)
        pos_key, key = jax.random.split(key)
        pos_hook = jax.random.uniform(pos_key, shape=(3,), minval=-1.0, maxval=1.0)
        if self.task == "jumping":
            # convert pos_hook to make sure x>0.3
            pos_hook = (pos_hook + jnp.array([1.6, 0.0, 0.0])) / jnp.array(
                [2.0, 1.0, 1.0]
            )
        pos = pos_hook - params.hook_offset
        # randomly sample object position from a sphere with radius params.l and center at hook_pos
        pos_obj = utils.sample_sphere(key, params.l * 0.9, pos_hook)
        l_rope = jnp.linalg.norm(pos_obj - pos_hook)
        # randomly sample object velocity, which is perpendicular to the rope
        vel_key, key = jax.random.split(key)
        vel_obj = jax.random.uniform(vel_key, shape=(3,), minval=-2.0, maxval=2.0)
        zeta = (pos_obj - pos_hook) / l_rope
        zeta_dot = jnp.cross(zeta, vel_obj)
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
            pos=pos,
            vel=zeros3,
            omega=zeros3,
            omega_tar=zeros3,
            quat=jnp.concatenate([zeros3, jnp.array([1.0])]),
            # object
            pos_obj=pos_obj,
            vel_obj=vel_obj,
            # hook
            pos_hook=pos_hook,
            vel_hook=zeros3,
            # rope
            l_rope=l_rope,
            zeta=zeta,
            zeta_dot=zeta_dot,
            f_rope=zeros3,
            f_rope_norm=0.0,
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
            # disturbance
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

    def fixed_init_state(self, key: chex.PRNGKey, params: EnvParams3D) -> EnvState3D:
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
            # object
            pos_obj=zeros3,
            vel_obj=zeros3,
            # hook
            pos_hook=zeros3,
            vel_hook=zeros3,
            # rope
            l_rope=0.0,
            zeta=zeros3,
            zeta_dot=zeros3,
            f_rope=zeros3,
            f_rope_norm=0.0,
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
            # disturbance
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

    def get_info(self, state: EnvState3D, params: EnvParams3D) -> dict:
        info = {
            "discount": self.discount(state, params),
            "err_pos": self.get_err_pos(state),
            "err_vel": self.get_err_vel(state),
            "obs_param": self.get_obs_paramsonly(state, params),
            "obs_adapt": self.get_obs_adapt_hist(state, params),
            "hit_wall": (
                (utils.get_hit_reward(state.pos_obj, params) < -0.5)
                | (utils.get_hit_reward(state.pos, params) < -0.5)
            ),
            "pass_wall": ((state.pos[0] < -0.05) & (state.pos_obj[0] < -0.05)),
        }
        return info

    def reset_env(
        self, key: chex.PRNGKey, params: EnvParams3D
    ) -> Tuple[chex.Array, EnvState3D]:
        state = self.get_init_state(key, params)
        info = self.get_info(state, params)
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
    def get_obs_objonly(self, state: EnvState3D, params: EnvParams3D) -> chex.Array:
        obs_elements = [
            # object
            state.pos_obj,
            state.vel_obj / 3.0,
            # hook
            state.pos_hook,
            state.vel_hook / 3.0,
            # rope
            jnp.expand_dims(state.l_rope, axis=0),
            state.zeta,
            state.zeta_dot / 10.0,  # 3*3=9
            state.f_rope,
            jnp.expand_dims(state.f_rope_norm, axis=0),  # 3+1=4
        ]
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
                    # object mass
                    (params.mo - params.mo_mean) / params.mo_std,
                    # rope length
                    (params.l - params.l_mean) / params.l_std,
                ]
            ),
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
    def get_obs_nlaconly(self, state: EnvState3D, params: EnvParams3D) -> chex.Array:
        obs_elements = [
            # nlac observation
            state.control_params.vel_hat,
            state.control_params.a_hat,
            state.control_params.d_hat,
        ]
        obs = jnp.concatenate(obs_elements, axis=-1)
        return obs

    @partial(jax.jit, static_argnums=(0,))
    def get_obs_nlaconly(self, state: EnvState3D, params: EnvParams3D) -> chex.Array:
        obs_elements = [
            # nlac observation
            state.control_params.vel_hat,
            state.control_params.a_hat,
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
    def get_obs_quad_obj(self, state: EnvState3D, params: EnvParams3D) -> chex.Array:
        quad_obs = self.get_obs_quadonly(state, params)
        obj_obs = self.get_obs_objonly(state, params)
        return jnp.concatenate([quad_obs, obj_obs], axis=-1)

    @partial(jax.jit, static_argnums=(0,))
    def get_obs_quad_obj_params(
        self, state: EnvState3D, params: EnvParams3D
    ) -> chex.Array:
        quad_obs = self.get_obs_quadonly(state, params)
        obj_obs = self.get_obs_objonly(state, params)
        param_obs = self.get_obs_paramsonly(state, params)
        return jnp.concatenate([quad_obs, obj_obs, param_obs], axis=-1)

    @partial(jax.jit, static_argnums=(0,))
    def get_obs_quad_l1(self, state: EnvState3D, params: EnvParams3D) -> chex.Array:
        quad_obs = self.get_obs_quadonly(state, params)
        l1_obs = self.get_obs_l1only(state, params)
        return jnp.concatenate([quad_obs, l1_obs], axis=-1)

    @partial(jax.jit, static_argnums=(0,))
    def get_obs_quad_nlac(self, state: EnvState3D, params: EnvParams3D) -> chex.Array:
        quad_obs = self.get_obs_quadonly(state, params)
        nlac_obs = self.get_obs_nlaconly(state, params)
        return jnp.concatenate([quad_obs, nlac_obs], axis=-1)

    @partial(jax.jit, static_argnums=(0,))
    def get_obs_quad_nlac(self, state: EnvState3D, params: EnvParams3D) -> chex.Array:
        quad_obs = self.get_obs_quadonly(state, params)
        nlac_obs = self.get_obs_nlaconly(state, params)
        return jnp.concatenate([quad_obs, nlac_obs], axis=-1)

    @partial(jax.jit, static_argnums=(0,))
    def is_terminal(self, state: EnvState3D, params: EnvParams3D) -> bool:
        """Check whether state is terminal."""
        # Check number of steps in episode termination condition
        done = (
            (state.time >= params.max_steps_in_episode)
            | (jnp.abs(state.pos) > 3.0).any()
            | (jnp.abs(state.pos_obj) > 3.0).any()
            | (jnp.abs(state.zeta_dot) > 100.0).any()
        )
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


# def eval_env(
#     env: Quad3D, controller, control_params, total_steps=300, filename="", debug=False
# ):
#     if debug:
#         env_num = 1
#     else:
#         env_num = 8192

#     # running environment
#     rng = jax.random.PRNGKey(1)
#     rng, rng_params = jax.random.split(rng)
#     rng_params = jax.random.split(rng_params, env_num)
#     env_params = jax.vmap(env.sample_params)(rng_params)

#     rng, rng_reset = jax.random.split(rng)
#     rng_reset = jax.random.split(rng_reset, env_num)
#     obs, info, env_state = jax.vmap(env.reset)(rng_reset, env_params)

#     rng, rng_control = jax.random.split(rng)
#     control_params = controller.reset(env_state, env_params, controller.init_control_params, rng_control)

#     cumulated_err_pos = jnp.zeros(env_num)

#     def run_one_step(carry, _):
#         obs, env_state, rng, env_params, control_params, info, cumulated_err_pos = carry
#         rng, rng_act, rng_step, rng_control = jax.random.split(rng, 4)
#         rng_act = jax.random.split(rng_act, env_num)
#         action, control_params, control_info = controller(
#             obs, env_state, env_params, rng_act, control_params, info
#         )
#         if control_info is not None:
#             if "a_mean" in control_info:
#                 action = control_info["a_mean"]
#         rng_step = jax.random.split(rng_step, env_num)
#         next_obs, next_env_state, reward, done, info = jax.vmap(env.step)(
#             rng_step, env_state, action, env_params
#         )
#         # if done, reset environment parameters
#         rng, rng_params = jax.random.split(rng)
#         rng_params = jax.random.split(rng_params, env_num)
#         new_env_params = jax.vmap(env.sample_params)(rng_params)

#         def map_fn(done, x, y):
#             # reshaped_done = jnp.broadcast_to(done, x.shape)
#             indexes = (slice(None),) + (None,) * (len(x.shape) - 1)
#             reshaped_done = done[indexes]
#             return reshaped_done * x + (1 - reshaped_done) * y
#             # return jnp.where(reshaped_done, x, y)

#         env_params = jax.tree_map(
#             lambda x, y: map_fn(done, x, y), new_env_params, env_params
#         )
#         cumulated_err_pos = cumulated_err_pos + info["err_pos"]
#         # if done, reset controller parameters, aviod use if, use lax.cond instead
#         # NOTE: controller parameters are not reset here
#         # new_control_params = controller.reset()
#         # control_params = jax.tree_map(
#         #     lambda x, y: map_fn(done, x, y), new_control_params, control_params
#         # )
#         rng, rng_control = jax.random.split(rng)
#         new_control_params = controller.reset(env_state, env_params, control_params, rng_control)
#         # new_control_params = new_control_params.replace(
#         #     a_mean = jax.random.uniform(rng_control, shape=control_params.a_mean.shape, minval=-1.0, maxval=1.0),
#         # )
#         control_params = lax.cond(done, lambda x: new_control_params, lambda x: x, control_params)
#         return (
#             next_obs,
#             next_env_state,
#             rng,
#             env_params,
#             control_params,
#             info,
#             cumulated_err_pos,
#         ), (info['err_pos'], done)

#     t0 = time_module.time()
#     env_rng, rng = jax.random.split(rng)
#     env_rng = jax.random.split(env_rng, env_num)
#     (
#         obs,
#         env_state,
#         env_rng,
#         env_params,
#         control_params,
#         info,
#         cumulated_err_pos,
#     ), (err_pos, dones) = lax.scan(
#         run_one_step,
#         (obs, env_state, rng, env_params, control_params, info, cumulated_err_pos),
#         jnp.arange(total_steps),
#     )
#     print(f"env running time: {time_module.time()-t0:.2f}s")

#     # calculate cumulative err_pos bewteen each done
#     err_pos_ep = []
#     last_ep_end = 0
#     for i in range(len(dones)):
#         if dones[i]:
#             err_pos_ep.append(err_pos[last_ep_end:i+1].mean())
#             last_ep_end = i+1
#     err_pos_ep = jnp.array(err_pos_ep)
#     # print mean and std of err_pos
#     pos_mean, pos_std = jnp.mean(err_pos_ep), jnp.std(err_pos_ep)
#     print(f'err_pos mean: {pos_mean:.3f}, std: {pos_std:.3f}')
#     print(f'${pos_mean*100:.2f} \pm {pos_std*100:.2f}$')

#     # save data
#     with open(
#         f"{quadjax.get_package_path()}/../results/eval_err_pos_{filename}.pkl", "wb"
#     ) as f:
#         pickle.dump(np.array(err_pos_ep), f)


def eval_env(
    env: Quad3D,
    controller: controllers.BaseController,
    control_params,
    total_steps=30000,
    filename="",
    debug=False,
):
    # running environment
    rng = jax.random.PRNGKey(1)
    # rng, rng_params = jax.random.split(rng)
    # env_params = env.default_params
    
    # rng, rng_reset = jax.random.split(rng)
    # obs, info, env_state = env.reset(rng_reset, env_params)

    # rng, rng_control = jax.random.split(rng)                      
    # control_params = controller.reset(env_state, env_params, controller.init_control_params, rng_control)

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
        return (next_obs, next_env_state, rng, env_params, control_params), (info['err_pos'], done)
    
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


def render_env(env: Quad3D, controller, control_params, repeat_times=1, filename=""):
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

    rng, rng_control = jax.random.split(rng)
    control_params = controller.reset(
        env_state, env_params, controller.init_control_params, rng_control
    )
    n_dones = 0

    # Profiling algorithms
    # controller_jit = jax.jit(controller)
    # controller_reset_jit = jax.jit(controller.reset)
    # rng, rng_act, rng_step = jax.random.split(rng, 3)
    # controller_jit(obs, env_state, env_params, rng_act, control_params)
    # rng, rng_control = jax.random.split(rng)                      
    # controller_reset_jit(env_state, env_params, controller.init_control_params, rng_control)
    # ts = []
    # for i in trange(101):
    #     t0 = time_module.time()
    #     rng, rng_act, rng_step = jax.random.split(rng, 3)
    #     action, control_params, control_info = controller_jit(obs, env_state, env_params, rng_act, control_params)
    #     ts.append((time_module.time()-t0)*1000)
    # ts = ts[1:]
    # print(f'running time: ${np.mean(ts):.2f} \pm {np.std(ts):.2f}$')
    # ts = []
    # for i in trange(11):
    #     t0 = time_module.time()
    #     rng, rng_control = jax.random.split(rng)
    #     control_params = controller_reset_jit(env_state, env_params, controller.init_control_params, rng_control)
    #     ts.append((time_module.time()-t0)*1000)
    # ts = ts[1:]
    # print(f'reset time: ${np.mean(ts):.2f} \pm {np.std(ts):.2f}$')
    # exit()

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

    with open(
        f"{quadjax.get_package_path()}/../results/state_seq_{filename}.pkl", "wb"
    ) as f:
        pickle.dump(state_seq_dict, f)


def get_controller(env, controller_name, controller_params=None, debug=False):
    if controller_name == "lqr":
        control_params = controllers.LQRParams(
            Q=jnp.diag(jnp.ones(12)),
            R=0.03 * jnp.diag(jnp.ones(4)),
            K=jnp.zeros((4, 12)),
        )
        controller = controllers.LQRController(env, control_params=control_params)
    elif controller_name == "pid":
        control_params = controllers.PIDParams(
            Kp=8.0, 
            Kd=4.0,  
            Ki=3.0,
            Kp_att=1.0, 
        )
        controller = controllers.PIDController(env, control_params=control_params)
    elif controller_name == "random":
        control_params = None
        controller = controllers.RandomController(env, control_params)
    elif controller_name == "fixed":
        control_params = controllers.FixedParams(
            u=jnp.asarray([0.0, 0.0, 0.0, 0.0]),
        )
        controller = controllers.FixedController(env, control_params=control_params)
    elif "mppi" in controller_name:
        # sigma = 0.5
        sigma = 0.135 #0.5
        if controller_params == "":
            N = 8192
            H = 32
            lam = 5e-2 # 0.01
        else:
            # parse in format "N{sample_number}_H{horizon}_sigma{sigma}_lam{lam}"
            N = int(controller_params.split("_")[0][1:])
            H = int(controller_params.split("_")[1][1:])
            lam = float(controller_params.split("_")[2][3:])
            print(f"[DEBUG], set controller parameters to be: N={N}, H={H}, lam={lam}")
        if debug:
            N = 4
            H = 2
            print(f"[DEBUG], override controller parameters to be: N={N}, H={H}")
        thrust_hover = env.default_params.m * env.default_params.g
        thrust_hover_normed = (thrust_hover / env.default_params.max_thrust) * 2.0 - 1.0
        a_mean_per_step = jnp.array([thrust_hover_normed, 0.0, 0.0, 0.0])
        a_mean = jnp.tile(a_mean_per_step, (H, 1))
        if controller_name == "mppi":
            # DEBUG here
            sigma = jnp.array([0.05, 0.15, 0.15, 0.3])
            a_cov_per_step = jnp.diag(sigma**2)
            # a_cov_per_step = jnp.diag(jnp.array([sigma**2] * env.action_dim))
            a_cov = jnp.tile(a_cov_per_step, (H, 1, 1))
            control_params = controllers.MPPIParams(
                gamma_mean=1.0,
                gamma_sigma=0.0,
                discount= 1.0,
                sample_sigma=sigma,
                a_mean=a_mean,
                a_cov=a_cov,
                obs_noise_scale=0.00,
            )
            controller = controllers.MPPIController(
                env=env, control_params=control_params, N=N, H=H, lam=lam
            )
        elif "mppi_zeji" in controller_name:
            a_cov = jnp.diag(jnp.ones(H*env.action_dim)*sigma**2)
            if "mean" in controller_name:
                expansion_mode = "mean"
            elif "lqr" in controller_name:
                expansion_mode = "lqr"
            elif "zero" in controller_name:
                expansion_mode = "zero"
            elif "ppo" in controller_name:
                expansion_mode = "ppo"
            elif "pid" in controller_name:
                expansion_mode = "pid"
            else:
                expansion_mode = "mean"
                print(
                    "[DEBUG] unset expansion mode, MPPI(zeji) expansion_mode set to mean"
                )
            control_params = controllers.MPPIZejiParams(
                gamma_mean=1.0,
                gamma_sigma=0.0,
                discount=1.0,
                sample_sigma=sigma,
                a_mean=a_mean,
                a_cov=a_cov,
                a_cov_offline=jnp.zeros((H, env.action_dim, env.action_dim)),
                obs_noise_scale = 0.00, 
            )
            controller = controllers.MPPIZejiController(
                env=env,
                control_params=control_params,
                N=N,
                H=H,
                lam=lam,
                expansion_mode=expansion_mode,
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
        compressor = Compressor()
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
    elif controller_name == "RMA-expert":
        from quadjax.train import ActorCritic, Compressor, Adaptor

        network = ActorCritic(env.action_dim, activation="tanh")
        compressor = Compressor()
        adaptor = Adaptor()
        if controller_params == "":
            file_path = "ppo_params_"
        else:
            file_path = f"{controller_params}"
        control_params = pickle.load(
            open(f"{quadjax.get_package_path()}/../results/{file_path}.pkl", "rb")
        )

        def apply_fn(train_params, last_obs, env_info):
            compressed_last_obs = compressor.apply(
                train_params[1], env_info["obs_param"]
            )
            adapted_last_obs = adaptor.apply(train_params[2], env_info["obs_adapt"])
            jax.debug.print("compressed {com}", com=compressed_last_obs)
            jax.debug.print("adapted {ada}", ada=adapted_last_obs)
            obs = jnp.concatenate([last_obs, compressed_last_obs], axis=-1)
            pi, value = network.apply(train_params[0], obs)
            return pi, value

        controller = controllers.NetworkController(apply_fn, env, control_params)
    elif controller_name == "l1":
        control_params = controllers.L1Params()
        controller = controllers.L1Controller(env, control_params)
    elif controller_name == "debug":

        class DebugController(controllers.BaseController):
            def __call__(
                self, obs, state, env_params, rng_act, control_params, env_info
            ) -> jnp.ndarray:
                quat_desired = jnp.array([0.0, 0.0, 0.0, 1.0])
                quat_err = geom.multiple_quat(
                    geom.conjugate_quat(quat_desired), state.quat
                )
                att_err = quat_err[:3]
                omega_tar = -3.0 * att_err
                thrust = (env_params.m + env_params.mo) * env_params.g
                omega_normed = omega_tar / env_params.max_omega
                thrust_normed = thrust / env_params.max_thrust * 2.0 - 1.0
                action = jnp.concatenate([jnp.asarray([thrust_normed]), omega_normed])
                return action, None, None

        control_params = None
        controller = DebugController(env, control_params)
    else:
        raise NotImplementedError
    return controller, control_params


@pydataclass
class Args:
    task: str = "tracking"  # tracking, tracking_zigzag, hovering
    dynamics: str = "bodyrate"  # bodyrate, free, slung
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
    # if args.debug is True, enable NaN detection
    if args.debug:
        jax.config.update("jax_debug_nans", True)

    env = Quad3D(
        task=args.task,
        dynamics=args.dynamics,
        obs_type=args.obs_type,
        lower_controller=args.lower_controller,
        enable_randomizer=not args.noDR,
        disturb_type=args.disturb_type,
        disable_rollover_terminate=True,
    )
    print("starting test...")
    # enable NaN value detection
    # from jax import config
    # config.update("jax_debug_nans", True)
    # with jax.disable_jit():
    controller, control_params = get_controller(
        env, args.controller, args.controller_params
    )
    if args.mode == "eval":
        eval_env(
            env,
            controller=controller,
            control_params=control_params,
            total_steps=300*4*10,
            filename=args.name,
            debug=args.debug,
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
