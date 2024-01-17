import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
import optax
from flax.linen.initializers import constant, orthogonal
from typing import Sequence, NamedTuple
from flax.training.train_state import TrainState
import distrax
import time
from matplotlib import pyplot as plt
from dataclasses import dataclass as pydataclass
import tyro
import pickle
import GPUtil
from functools import partial

import quadjax
from quadjax.controllers import NetworkController
from quadjax.envs.base import LogWrapper

class ActorCritic(nn.Module):
    action_dim: Sequence[int]
    activation: str = "tanh"

    @nn.compact
    def __call__(self, x):
        if self.activation == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh
        actor_mean = nn.Dense(
            256, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(x)
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(
            256, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(actor_mean)
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(
            256, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(actor_mean)
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_mean)
        actor_logtstd = self.param("log_std", nn.initializers.zeros, (self.action_dim,))
        pi = distrax.MultivariateNormalDiag(actor_mean, jnp.exp(actor_logtstd))

        critic = nn.Dense(
            256, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(x)
        critic = activation(critic)
        critic = nn.Dense(
            256, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(critic)
        critic = activation(critic)
        critic = nn.Dense(
            256, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(critic)
        critic = activation(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(
            critic
        )

        return pi, jnp.squeeze(critic, axis=-1)
    
# parameter compressor, 2 layer 128 hidden units with output dim 8 input dim=env.param_obs_dim
class Compressor(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(256, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        x = nn.relu(x)
        x = nn.Dense(256, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        x = nn.relu(x)
        x = nn.Dense(8, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(x)
        return x
    
# parameter adaptor, 2 layer 256 hidden units with output dim 8 input dim=env.adapt_obs_dim
class Adaptor(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(256, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        x = nn.relu(x)
        x = nn.Dense(256, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        x = nn.relu(x)
        x = nn.Dense(8, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(x)
        return x


class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    info: jnp.ndarray


def make_train(env, config):
    config["NUM_PPO_UPDATES"] = int(
        config["PPO_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )
    config["NUM_ADAPT_UPDATES"] = int(
        config["ADAPT_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )

    config["MINIBATCH_SIZE"] = (
        config["NUM_ENVS"] * config["NUM_STEPS"] // config["NUM_MINIBATCHES"]
    )
    env = LogWrapper(env)

    def linear_schedule_ppo(count):
        frac = (
            1.0
            - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"]))
            / config["NUM_PPO_UPDATES"]
        )
        return config["LR"] * frac
    
    def linear_schedule_adapt(count):
        frac = (
            1.0
            - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"]))
            / config["NUM_ADAPT_UPDATES"]
        )
        return config["LR"] * 10.0 * frac

    def train(rng):
        # INIT ENV
        rng, rng2 = jax.random.split(rng)
        param_rng = jax.random.split(rng2, config["NUM_ENVS"])
        env_params = jax.vmap(env.sample_params)(param_rng)
        if config['enable_curri']:
            curri_params = 0.0
            env_params = env_params.replace(curri_params = jnp.ones(config["NUM_ENVS"])*curri_params)

        # INIT NETWORK
        network = ActorCritic(
            env.action_dim, activation=config["ACTIVATION"]
        )

        if config["ANNEAL_LR"]:
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(learning_rate=linear_schedule_ppo, eps=1e-5),
            )
        else:
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(config["LR"], eps=1e-5),
            )

        if config['enable_RMA']:
            compressor = Compressor()
            rng, _rng = jax.random.split(rng)
            compressor_params = compressor.init(_rng, jnp.zeros(env.param_obs_dim))
            adaptor = Adaptor()
            rng, _rng = jax.random.split(rng)
            adaptor_params = adaptor.init(_rng, jnp.zeros(env.adapt_obs_dim))
            def get_pi_value(train_params, last_obs, env_info):
                compressed_params_obs = compressor.apply(train_params[1], env_info['obs_param'])
                obs = jnp.concatenate([last_obs, compressed_params_obs], axis=-1)
                pi, value = network.apply(train_params[0], obs)
                return pi, value
            def get_pi_value_adapt(train_params, last_obs, env_info):
                adapted_last_obs = adaptor.apply(train_params[2], env_info['obs_adapt'])
                obs = jnp.concatenate([last_obs, adapted_last_obs], axis=-1)
                pi, value = network.apply(train_params[0], obs)
                return pi, value
            rng, _rng = jax.random.split(rng)
            network_params = network.init(_rng, jnp.zeros(env.obs_dim+8))

            # load parameters
            # network_params, compressor_params, _ = pickle.load(open(f"{quadjax.get_package_path()}/../results/rma/rma_policy.pkl", "rb"))

            ppo_train_state = TrainState.create(
                apply_fn=get_pi_value_adapt, # for test
                params=[network_params, compressor_params],
                tx=tx,
            )
            # deep copy tx to get a new optimizer
            tx_adapt = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(learning_rate=linear_schedule_adapt, eps=1e-5),
            )
            adapt_train_state = TrainState.create(
                apply_fn=get_pi_value_adapt, # for test
                params=adaptor_params,
                tx=tx_adapt, 
            )
            def get_network_params(ppo_train_state, adapt_train_state):
                return ppo_train_state.params + [adapt_train_state.params]
        else:
            def get_pi_value(train_params, last_obs, env_info):
                pi, value = network.apply(train_params, last_obs)
                return pi, value
            rng, _rng = jax.random.split(rng)
            network_params = network.init(_rng, jnp.zeros(env.obs_dim))
            ppo_train_state = TrainState.create(
                apply_fn=get_pi_value,
                params=network_params,
                tx=tx,
            )
            adapt_train_state = None
            def get_network_params(ppo_train_state, adapt_train_state):
                return ppo_train_state.params

        # COLLECT TRAJECTORIES
        def _env_step(get_pi_value, runner_state, unused):
            ppo_train_state, env_state, last_obs, rng, env_params, env_info, adapt_train_state = runner_state

            # SELECT ACTION
            rng, _rng = jax.random.split(rng)

            pi, value = get_pi_value(get_network_params(ppo_train_state, adapt_train_state), last_obs, env_info)
            action = pi.sample(seed=_rng)
            log_prob = pi.log_prob(action)

            # STEP ENV
            rng, _rng = jax.random.split(rng)
            rng_step = jax.random.split(_rng, config["NUM_ENVS"])
            obsv, env_state, reward, done, info = jax.vmap(env.step)(
                rng_step, env_state, action, env_params
            )
            # resample environment parameters if done TODO wrap this into step function or other env wrapper
            rng_params = jax.random.split(_rng, config["NUM_ENVS"])
            new_env_params = jax.vmap(env.sample_params)(rng_params)
            if config['enable_curri']:
                new_env_params = new_env_params.replace(curri_params = jnp.ones(config["NUM_ENVS"])* curri_params)
            def map_fn(done, x, y):
                # reshaped_done = jnp.broadcast_to(done, x.shape)
                indexes = (slice(None),) + (None,) * (len(x.shape) - 1)
                reshaped_done = done[indexes]
                return reshaped_done * x + (1 - reshaped_done) * y
            env_params = jax.tree_map(
                lambda x, y: map_fn(done, x, y), new_env_params, env_params
            )

            transition = Transition(
                done, action, value, reward, log_prob, last_obs, info
            )
            runner_state = (ppo_train_state, env_state, obsv, rng, env_params, info, adapt_train_state)
            return runner_state, transition
        
        # TRAIN LOOP
        @jax.jit
        def _train_ppo(runner_state, unused):

            _env_step_fn = partial(_env_step, get_pi_value)
            runner_state, traj_batch = jax.lax.scan(
                _env_step_fn, runner_state, None, config["NUM_STEPS"]
            )
            
            # CALCULATE ADVANTAGE
            ppo_train_state, env_state, last_obs, rng, env_params, env_info, adapt_train_state = runner_state
            _, last_val = get_pi_value(ppo_train_state.params, last_obs, env_info)

            def _calculate_gae(traj_batch, last_val):
                def _get_advantages(gae_and_next_value, transition):
                    gae, next_value = gae_and_next_value
                    done, value, reward = (
                        transition.done,
                        transition.value,
                        transition.reward,
                    )
                    delta = reward + config["GAMMA"] * next_value * (1 - done) - value
                    gae = (
                        delta
                        + config["GAMMA"] * config["GAE_LAMBDA"] * (1 - done) * gae
                    )
                    return (gae, value), gae

                _, advantages = jax.lax.scan(
                    _get_advantages,
                    (jnp.zeros_like(last_val), last_val),
                    traj_batch,
                    reverse=True,
                    unroll=16,
                )
                return advantages, advantages + traj_batch.value

            advantages, targets = _calculate_gae(traj_batch, last_val)

            # UPDATE NETWORK
            def _update_epoch(update_state, unused):
                def _update_minbatch(ppo_train_state, batch_info):
                    traj_batch, advantages, targets = batch_info

                    def _loss_fn(params, traj_batch, gae, targets):
                        # RERUN NETWORK
                        pi, value = get_pi_value(params, traj_batch.obs, traj_batch.info)
                        log_prob = pi.log_prob(traj_batch.action)

                        # CALCULATE VALUE LOSS
                        value_pred_clipped = traj_batch.value + (
                            value - traj_batch.value
                        ).clip(-config["CLIP_EPS"], config["CLIP_EPS"])
                        value_losses = jnp.square(value - targets)
                        value_losses_clipped = jnp.square(value_pred_clipped - targets)
                        value_loss = (
                            0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()
                        )

                        # CALCULATE ACTOR LOSS
                        ratio = jnp.exp(log_prob - traj_batch.log_prob)
                        gae = (gae - gae.mean()) / (gae.std() + 1e-8)
                        loss_actor1 = ratio * gae
                        loss_actor2 = (
                            jnp.clip(
                                ratio,
                                1.0 - config["CLIP_EPS"],
                                1.0 + config["CLIP_EPS"],
                            )
                            * gae
                        )
                        loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
                        loss_actor = loss_actor.mean()
                        entropy = pi.entropy().mean()

                        total_loss = (
                            loss_actor
                            + config["VF_COEF"] * value_loss
                            - config["ENT_COEF"] * entropy
                        )
                        return total_loss, (value_loss, loss_actor, entropy)

                    grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                    total_loss, grads = grad_fn(
                        ppo_train_state.params, traj_batch, advantages, targets
                    )
                    ppo_train_state = ppo_train_state.apply_gradients(grads=grads)
                    return ppo_train_state, total_loss

                ppo_train_state, traj_batch, advantages, targets, rng = update_state
                rng, _rng = jax.random.split(rng)
                batch_size = config["MINIBATCH_SIZE"] * config["NUM_MINIBATCHES"]
                assert (
                    batch_size == config["NUM_STEPS"] * config["NUM_ENVS"]
                ), "batch size must be equal to number of steps * number of envs"
                permutation = jax.random.permutation(_rng, batch_size)
                batch = (traj_batch, advantages, targets)
                batch = jax.tree_util.tree_map(
                    lambda x: x.reshape((batch_size,) + x.shape[2:]), batch
                )
                shuffled_batch = jax.tree_util.tree_map(
                    lambda x: jnp.take(x, permutation, axis=0), batch
                )
                minibatches = jax.tree_util.tree_map(
                    lambda x: jnp.reshape(
                        x, [config["NUM_MINIBATCHES"], -1] + list(x.shape[1:])
                    ),
                    shuffled_batch,
                )
                ppo_train_state, total_loss = jax.lax.scan(
                    _update_minbatch, ppo_train_state, minibatches
                )
                update_state = (ppo_train_state, traj_batch, advantages, targets, rng)
                return update_state, total_loss

            update_state = (ppo_train_state, traj_batch, advantages, targets, rng)
            update_state, loss_info = jax.lax.scan(
                _update_epoch, update_state, None, config["UPDATE_EPOCHS"]
            )
            ppo_train_state = update_state[0]
            metric = traj_batch.info
            rng = update_state[-1]

            runner_state = (ppo_train_state, env_state, last_obs, rng, env_params, env_info, adapt_train_state)
            return runner_state, metric
        
        @jax.jit
        def _train_adaptor(runner_state, unused):
            _env_step_fn = partial(_env_step, get_pi_value_adapt)
            runner_state, traj_batch = jax.lax.scan(
                _env_step_fn, runner_state, None, config["NUM_STEPS"]
            )
            ppo_train_state, env_state, last_obs, rng, env_params, env_info, adapt_train_state = runner_state

            # Update adaptor
            def _update_adaptor(update_state, unused):
                def _update_minbatch(adapt_train_state, batch_info):
                    obs_param, obs_adapt = batch_info

                    def _loss_fn(adaptor_params, compressor_params, obs_param, obs_adapt):
                        # RERUN NETWORK
                        compressor_value = compressor.apply(compressor_params, obs_param)
                        adaptor_value = adaptor.apply(adaptor_params, obs_adapt)

                        total_loss = jnp.square(compressor_value - adaptor_value).mean()
                        return total_loss

                    grad_fn = jax.value_and_grad(_loss_fn, argnums=0)
                    total_loss, grads = grad_fn(adapt_train_state.params, ppo_train_state.params[1], obs_param, obs_adapt)
                    adapt_train_state = adapt_train_state.apply_gradients(grads=grads)
                    return adapt_train_state, total_loss
                ppo_train_state, traj_batch, rng, adapt_train_state = update_state
                rng, _rng = jax.random.split(rng)
                batch_size = config["MINIBATCH_SIZE"] * config["NUM_MINIBATCHES"]
                assert (
                    batch_size == config["NUM_STEPS"] * config["NUM_ENVS"]
                ), "batch size must be equal to number of steps * number of envs"
                permutation = jax.random.permutation(_rng, batch_size)
                batch = (traj_batch.info['obs_param'], traj_batch.info['obs_adapt'])
                batch = jax.tree_util.tree_map(
                    lambda x: x.reshape((batch_size,) + x.shape[2:]), batch
                )
                shuffled_batch = jax.tree_util.tree_map(
                    lambda x: jnp.take(x, permutation, axis=0), batch
                )
                minibatches = jax.tree_util.tree_map(
                    lambda x: jnp.reshape(
                        x, [config["NUM_MINIBATCHES"], -1] + list(x.shape[1:])
                    ),
                    shuffled_batch,
                )
                adapt_train_state, total_loss = jax.lax.scan(
                    _update_minbatch, adapt_train_state, minibatches
                )
                update_state = (ppo_train_state, traj_batch, rng, adapt_train_state)
                return update_state, total_loss
        
            update_state = (ppo_train_state, traj_batch, rng, adapt_train_state)
            update_state, loss_info = jax.lax.scan(
                _update_adaptor, update_state, None, config["UPDATE_EPOCHS"]
            )
            adapt_train_state = update_state[-1]
            rng = update_state[-2]

            metric = traj_batch.info
            metric['RMA_loss'] = loss_info

            runner_state = (ppo_train_state, env_state, last_obs, rng, env_params, env_info, adapt_train_state)
            return runner_state, metric

        # train PPO
        print("training PPO...")
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
        obsv, env_info, env_state = jax.vmap(env.reset)(reset_rng, env_params)
        rng, _rng = jax.random.split(rng)
        runner_state = (ppo_train_state, env_state, obsv, _rng, env_params, env_info, adapt_train_state)
        metric = {'step': jnp.array([]), 'returned_episode_returns': jnp.array([]), 'returned_episode_lengths': jnp.array([]), 'mean_episode_returns': jnp.array([]), 'err_pos': jnp.array([]), 'err_vel': jnp.array([]), 'err_pos_last_10': jnp.array([]), 'final_reward': jnp.array([]), 'hit_wall_rate': jnp.array([]), 'pass_wall_rate': jnp.array([])}
        step_per_log = config["NUM_STEPS"] * config["NUM_ENVS"]
        for i in range(config["NUM_PPO_UPDATES"]):
            runner_state, metric_local = jax.block_until_ready(_train_ppo(runner_state, None))

            metric_log = {}
            metric_log['step'] = jnp.array([(i+1)*step_per_log])
            metric_log['mean_episode_returns'] = jnp.mean(metric_local['returned_episode_returns'][-1] / (metric_local['returned_episode_lengths'][-1]+1.0))
            metric_log['returned_episode_returns'] = metric_local['returned_episode_returns'][-1].mean()
            metric_log['returned_episode_lengths'] = metric_local['returned_episode_lengths'][-1].mean()
            metric_log['err_pos'] = metric_local['err_pos'].mean()
            metric_log['err_pos_last_10'] = metric_local['err_pos'][-10:].mean()
            metric_log['err_vel'] = metric_local['err_vel'].mean()
            metric_log['final_reward'] = metric_local['final_reward'][-1].mean()
            # shift metric_local['returned_episode']) one step to the left to get final_step_mask
            metric_log['hit_wall_rate'] = (metric_local['hit_wall'] & metric_local['returned_episode']).sum() / metric_local['returned_episode'].sum()
            final_step_mask = jnp.concatenate([metric_local['returned_episode'][1:], jnp.zeros_like(metric_local['returned_episode'][-1:])], axis=0)
            metric_log['pass_wall_rate'] = (metric_local['pass_wall'] & final_step_mask).sum() / final_step_mask.sum()

            # curriculum learning
            if config['enable_curri']: 
                (ppo_train_state, env_state, obsv, _rng, env_params, env_info, adapt_train_state) = runner_state
                # if (metric_log['pass_wall_rate'] > 0.5):
                if True:
                    curri_params = jnp.clip(curri_params + 0.05, 0.0, 1.0)
                    env_params = env_params.replace(curri_params = jnp.ones(config["NUM_ENVS"])*curri_params)
                    runner_state = (ppo_train_state, env_state, obsv, _rng, env_params, env_info, adapt_train_state)
                print('curri_params: ', curri_params)
                                         
            print('====================')
            print(f'PPO update {i+1}/{config["NUM_PPO_UPDATES"]}')
            for k in metric.keys():
                v_mean = metric_log[k].mean()
                metric[k] = jnp.append(metric[k], v_mean)
                print(f'{k}: {v_mean:.3e}')
            GPUtil.showUtilization()

        # train adaptor
        metric['RMA_loss'] = jnp.array([])
        if config['enable_RMA']:
            print("training adaptor...")
            rng, _rng = jax.random.split(rng)
            reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
            obsv, env_info, env_state = jax.vmap(env.reset)(reset_rng, env_params)
            rng, _rng = jax.random.split(rng)
            for i in range(config["NUM_ADAPT_UPDATES"]):
                runner_state, metric_local = jax.block_until_ready(_train_adaptor(runner_state, None))

                metric_log = {}
                metric_log['step'] = jnp.array([(i+config["NUM_PPO_UPDATES"]+1)*step_per_log])
                metric_log['mean_episode_returns'] = jnp.mean(metric_local['returned_episode_returns'][-1] / (metric_local['returned_episode_lengths'][-1]+1.0))
                metric_log['returned_episode_returns'] = metric_local['returned_episode_returns'][-1].mean()
                metric_log['returned_episode_lengths'] = metric_local['returned_episode_lengths'][-1].mean()
                metric_log['err_pos'] = metric_local['err_pos'].mean()
                metric_log['err_pos_last_10'] = metric_local['err_pos'][-10:].mean()
                metric_log['err_vel'] = metric_local['err_vel'].mean()
                metric_log['final_reward'] = metric_local['final_reward'][-1].mean()
                # shift metric_local['returned_episode']) one step to the left to get final_step_mask
                metric_log['hit_wall_rate'] = (metric_local['hit_wall'] & metric_local['returned_episode']).sum() / metric_local['returned_episode'].sum()
                final_step_mask = jnp.concatenate([metric_local['returned_episode'][1:], jnp.zeros_like(metric_local['returned_episode'][-1:])], axis=0)
                metric_log['pass_wall_rate'] = (metric_local['pass_wall'] & final_step_mask).sum() / final_step_mask.sum()
                metric_log['RMA_loss'] = metric_local['RMA_loss'].mean()

                print('====================')
                print(f'Adaptor update {i+1}/{config["NUM_ADAPT_UPDATES"]}')
                for k in metric.keys():
                    v_mean = metric_log[k].mean()
                    metric[k] = jnp.append(metric[k], v_mean)
                    print(f'{k}: {v_mean:.2e}')
                GPUtil.showUtilization()
    
        return runner_state, metric

    return train

@pydataclass
class Args:
    task: str = "tracking_zigzag" # tracking, tracking_zigzag, 
    env: str = "quad2d_free" # quad2d_free, quad2d, quad3d_free, quad3d
    lower_controller: str = "base" # bodyrate, base, l1, l1_esitimate_only
    obs_type: str = "quad" # quad_params, quad
    debug: bool = False
    curri: bool = False
    dynamics: str = "free"
    curri: bool = False
    RMA: bool = False
    noDR: bool = False # no domain randomization
    disturb_type: str = "periodic" # periodic, sin, drag, mixed
    name: str = "" # experiment name


def main(args: Args):
    if args.debug:
        jax.disable_jit()
    config = {
        "LR": 3e-4,
        "NUM_ENVS": 4096 if not args.debug else 1,
        "NUM_STEPS": 300 if not args.debug else 10,
        "PPO_TIMESTEPS": 1.6e8 if not args.debug else 1e2,
        "ADAPT_TIMESTEPS": 8e6 if not args.debug else 0.3e2,
        "UPDATE_EPOCHS": 2,
        "NUM_MINIBATCHES": 320 if not args.debug else 1,
        "GAMMA": 0.99,
        "GAE_LAMBDA": 0.95,
        "CLIP_EPS": 0.2,
        "ENT_COEF": 0.0,
        "VF_COEF": 0.5,
        "MAX_GRAD_NORM": 0.5,
        "ACTIVATION": "tanh",
        "ANNEAL_LR": False,
        "task": args.task,
        "enable_curri": args.curri,
        "enable_RMA": args.RMA,
    }
    rng = jax.random.PRNGKey(42)
    t0 = time.time()
    if args.env == 'dualquad2d':
        env = quadjax.envs.dualquad2d.DualQuad2D(task=args.task)
        render_fn = quadjax.envs.dualquad2d.render_env
    elif args.env == 'quad2d_free':
        env = quadjax.envs.quad2d_free.Quad2D(task=args.task, lower_controller=args.lower_controller)
        render_fn = quadjax.envs.quad2d_free.render_env
        eval_fn = quadjax.envs.quad2d_free.eval_env
    elif args.env == 'quad3d_free':
        env = quadjax.envs.quad3d_free.Quad3D(task=args.task, dynamics=args.dynamics, obs_type=args.obs_type, enable_randomizer=(not args.noDR), lower_controller=args.lower_controller, disturb_type=args.disturb_type)
        render_fn = quadjax.envs.quad3d_free.render_env
        eval_fn = quadjax.envs.quad3d_free.eval_env
    elif args.env == 'quad3d':
        env = quadjax.envs.quad3d.Quad3D(task=args.task, lower_controller=args.lower_controller)
        render_fn = quadjax.envs.quad3d.render_env
        eval_fn = quadjax.envs.quad3d.eval_env
    elif args.env == 'cartpole':
        env = quadjax.envs.cartpole.CartPole()
        render_fn = quadjax.envs.cartpole.render_env
        eval_fn = quadjax.envs.cartpole.eval_env
    train_fn = make_train(env, config)

    t0 = time.time()
    runner_state, metric = jax.block_until_ready(train_fn(rng))

    training_time = time.time() - t0
    print(f"train time: {training_time:.2f} s")
    # plot in three subplots of returned_episode_returns, err_pos, err_vel
    fig, axs = plt.subplots(5, 1)
    for _, (ax, data, title) in enumerate(
        zip(axs, [metric["returned_episode_returns"], metric["err_pos"], metric["pass_wall_rate"], metric["err_vel"]], ["returns", "err_pos", "pass_wall_rate", "err_vel"])
    ):
        if data.shape[0] <=0: 
            continue
        ax.plot(metric['step'], data)
        ax.set_ylabel(title)
        ax.set_xlabel("steps")
        # label out last value in a box
        last_value = data[-1]
        ax.text(
            metric['step'][-1],
            last_value,
            f"{last_value:.2f}",
            bbox=dict(facecolor="red", alpha=0.8),
        )
    axs[4].plot(metric['RMA_loss'], label='RMA_loss')
    axs[4].set_ylabel('RMA_loss')
    axs[4].set_xlabel("steps")
    # add training time to title
    fig.suptitle(f"training_time: {training_time:.2f} s")
    # save
    if len(args.name) > 0:
        filename = args.name
    else:
        filename = f"{args.env}_{args.task}_{args.lower_controller}"
    plt.savefig(f"{quadjax.get_package_path()}/../results/training_curve_{filename}.png")

    apply_fn = runner_state[0].apply_fn
    if args.RMA:
        params = runner_state[0].params + [runner_state[-1].params]
    else:
        params = runner_state[0].params
    
    with open(f"{quadjax.get_package_path()}/../results/ppo_params_{filename}.pkl", "wb") as f:
        pickle.dump(params, f)

    controller = NetworkController(apply_fn, env, control_params=params)

    # test policy
    eval_fn(env = env, controller = controller, control_params = params, total_steps= 3e3, filename=filename, debug=args.debug)
    render_fn(env = env, controller = controller, control_params = params, repeat_times = 3, filename=filename)

if __name__ == "__main__":
    main(tyro.cli(Args))