import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
import optax
from flax.linen.initializers import constant, orthogonal
from typing import Sequence, NamedTuple
from flax.training.train_state import TrainState
import distrax
from gymnax.wrappers.purerl import LogWrapper
import time
from matplotlib import pyplot as plt
from dataclasses import dataclass as pydataclass
import tyro
import pickle
import GPUtil

import quadjax
from quadjax.controllers import NetworkController

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

class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    info: jnp.ndarray


def make_train(env, config):
    config["NUM_UPDATES"] = int(
        config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )
    config["MINIBATCH_SIZE"] = (
        config["NUM_ENVS"] * config["NUM_STEPS"] // config["NUM_MINIBATCHES"]
    )
    env = LogWrapper(env)

    def linear_schedule(count):
        frac = (
            1.0
            - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"]))
            / config["NUM_UPDATES"]
        )
        return config["LR"] * frac

    def train(rng):
        # INIT ENV
        rng, rng1, rng2 = jax.random.split(rng, 3)
        reset_rng = jax.random.split(rng1, config["NUM_ENVS"])
        param_rng = jax.random.split(rng2, config["NUM_ENVS"])
        env_params = jax.vmap(env.sample_params)(param_rng)
        obsv, env_state = jax.vmap(env.reset)(reset_rng, env_params)

        # INIT NETWORK
        network = ActorCritic(
            env.action_dim, activation=config["ACTIVATION"]
        )
        rng, _rng = jax.random.split(rng)
        init_x = jnp.zeros(env.obs_dim)
        network_params = network.init(_rng, init_x)
        if config["ANNEAL_LR"]:
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(learning_rate=linear_schedule, eps=1e-5),
            )
        else:
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(config["LR"], eps=1e-5),
            )
        train_state = TrainState.create(
            apply_fn=network.apply,
            params=network_params,
            tx=tx,
        )

        # TRAIN LOOP
        @jax.jit
        def _update_step(runner_state, unused):
            # COLLECT TRAJECTORIES
            def _env_step(runner_state, unused):
                train_state, env_state, last_obs, rng, env_params = runner_state

                # SELECT ACTION
                rng, _rng = jax.random.split(rng)
                pi, value = network.apply(train_state.params, last_obs)
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
                runner_state = (train_state, env_state, obsv, rng, env_params)
                return runner_state, transition

            runner_state, traj_batch = jax.lax.scan(
                _env_step, runner_state, None, config["NUM_STEPS"]
            )

            # CALCULATE ADVANTAGE
            train_state, env_state, last_obs, rng, env_params = runner_state
            _, last_val = network.apply(train_state.params, last_obs)

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
                def _update_minbatch(train_state, batch_info):
                    traj_batch, advantages, targets = batch_info

                    def _loss_fn(params, traj_batch, gae, targets):
                        # RERUN NETWORK
                        pi, value = network.apply(params, traj_batch.obs)
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
                        train_state.params, traj_batch, advantages, targets
                    )
                    train_state = train_state.apply_gradients(grads=grads)
                    return train_state, total_loss

                train_state, traj_batch, advantages, targets, rng = update_state
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
                train_state, total_loss = jax.lax.scan(
                    _update_minbatch, train_state, minibatches
                )
                update_state = (train_state, traj_batch, advantages, targets, rng)
                return update_state, total_loss

            update_state = (train_state, traj_batch, advantages, targets, rng)
            update_state, loss_info = jax.lax.scan(
                _update_epoch, update_state, None, config["UPDATE_EPOCHS"]
            )
            train_state = update_state[0]
            metric = traj_batch.info
            rng = update_state[-1]

            runner_state = (train_state, env_state, last_obs, rng, env_params)
            return runner_state, metric

        rng, _rng = jax.random.split(rng)
        runner_state = (train_state, env_state, obsv, _rng, env_params)
        # NOTE not scan here for logging purpose
        # runner_state, metric = jax.lax.scan(
        #     _update_step, runner_state, None, config["NUM_UPDATES"]
        # )
        metric = {'step': jnp.array([]), 'returned_episode_returns': jnp.array([]), 'returned_episode_lengths': jnp.array([]), 'mean_episode_returns': jnp.array([]), 'err_pos': jnp.array([]), 'err_vel': jnp.array([])}
        step_per_log = config["NUM_STEPS"] * config["NUM_ENVS"]
        for i in range(config["NUM_UPDATES"]):
            runner_state, metric_local = _update_step(runner_state, None)
            metric_local['step'] = jnp.array([(i+1)*step_per_log])
            metric_local['mean_episode_returns'] = metric_local['returned_episode_returns'] / metric_local['returned_episode_lengths']
            print('====================')
            print(f'update {i+1}/{config["NUM_UPDATES"]}')
            for k in metric.keys():
                v_mean = metric_local[k].mean()
                metric[k] = jnp.append(metric[k], v_mean)
                print(f'{k}: {v_mean:.2e}')
            GPUtil.showUtilization()
        return runner_state, metric

    return train

@pydataclass
class Args:
    task: str = "tracking_zigzag"
    env: str = "quad2d_free"
    lower_controller: str = "base"
    debug: bool = False


def main(args: Args):
    config = {
        "LR": 3e-4,
        "NUM_ENVS": 4096 if not args.debug else 1,
        "NUM_STEPS": 300 if not args.debug else 100,
        "TOTAL_TIMESTEPS": 1.6e8 if not args.debug else 1e3,
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
    elif args.env == 'quad3d':
        env = quadjax.envs.quad3d.Quad3D(task=args.task)
        render_fn = quadjax.envs.quad3d.render_env
        eval_fn = quadjax.envs.quad3d.eval_env
    train_fn = make_train(env, config)

    t0 = time.time()
    runner_state, metric = jax.block_until_ready(train_fn(rng))
    training_time = time.time() - t0
    print(f"train time: {training_time:.2f} s")
    # plot in three subplots of returned_episode_returns, err_pos, err_vel
    fig, axs = plt.subplots(3, 1)
    for _, (ax, data, title) in enumerate(
        zip(axs, [metric["returned_episode_returns"], metric["err_pos"], metric["err_vel"]], ["returns", "err_pos", "err_vel"])
    ):
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
    # add training time to title
    fig.suptitle(f"training_time: {training_time:.2f} s")
    # save
    filename = f"{args.env}_{args.task}_{args.lower_controller}"
    plt.savefig(f"{quadjax.get_package_path()}/../results/ppo_{filename}.png")

    with open(f"{quadjax.get_package_path()}/../results/ppo_params_{filename}.pkl", "wb") as f:
        pickle.dump(runner_state[0].params, f)

    apply_fn = runner_state[0].apply_fn
    params = runner_state[0].params

    controller = NetworkController(apply_fn, env, control_params=params)

    # test policy
    eval_fn(env = env, controller = controller, control_params = params, total_steps=3e4, filename=filename)
    render_fn(env = env, controller = controller, control_params = params, repeat_times = 3, filename=filename)

if __name__ == "__main__":
    main(tyro.cli(Args))