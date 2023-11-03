import jax
import numpy as np
from jax import numpy as jnp
from jax import random
from matplotlib import pyplot as plt  
import seaborn as sns
from jaxopt import ProjectedGradient
from jaxopt.projection import projection_hyperplane
import pandas as pd
from tqdm import trange

# generate metrix [[0, b, ba, ba^2, ..., ba^H-1], [0,0,b,ba, ..., ba^H-2], ...]
def generate_matrix(a, b, H):
    matrix = jnp.zeros((H+1, H))
    for i in range(1, H+1):
        for j in range(0, i):
            # matrix = jax.ops.index_update(matrix, jax.ops.index[i, j], b * a ** (j - i))
            matrix = matrix.at[i, j].set(b * a ** (i -1 - j))
    return matrix

def get_sigma_eign2(eign_values, lam, H, sig):
    log_o = jnp.log(eign_values)
    log_const =(2 *(H*2*jnp.log(sig)) + jnp.sum(log_o)) / (H)
    log_s = 0.5 * log_const - 0.5 * log_o

    s = jnp.exp(log_s)

    # o = eign_values
    # cc = ((o**2)*s)/((1+2/lam*o*s)**3)
    # # print('verified solution: ', cc, 1/(o*(s**2)))

    return s

def test_once(N=1024, H = 3, sig = 0.2, lam = 0.01, repeat_times=1024, env_steps=30, method='base', a=1.0, b=1.0, mode='render'):

    if method == 'base':
        a_cov = jnp.eye(H)*(sig**2)
    elif method == 'zeji':
        AB_matrix = generate_matrix(a, b, H)
        Q_matrix = jnp.eye(H+1) * 1.0
        R = AB_matrix.T @ Q_matrix @ AB_matrix
        u, s, vh = jnp.linalg.svd(R)
        sigma_eign = get_sigma_eign2(s, lam, H, sig)
        a_cov = vh.T @ jnp.diag(sigma_eign) @ vh

    def dynamic_fn(x, action):
        x = a*x + b*action
        return x
    
    def reward_fn(t, x):
        x_tar = jnp.sin(0.6*t)
        # x_tar = 0.0
        reward = 1.0-1.0 * (x - x_tar)**2
        return reward

    def get_next_a_mean(t, x, a_mean, rng):
        # shift a_mean to the left by 1 and append the last element
        a_mean = jnp.concatenate([a_mean[1:], a_mean[-1:]])

        rng_act, rng = random.split(rng)
        # sample actions with a_mean and a_cov
        a_sampled = jax.vmap(lambda rng: random.multivariate_normal(rng, a_mean, a_cov))(random.split(rng, N)) # (N, H)

        def rollout_fn(carry, action):
            t, x = carry
            x = dynamic_fn(x, action)
            t = t + 1
            reward = reward_fn(t, x)
            return (t, x), (x, reward)

        xx = jnp.repeat(x, N)
        tt = jnp.repeat(t, N)
        _, (x_mppi_sampled, reward) = jax.lax.scan(rollout_fn, (tt, xx), a_sampled.T)

        cost = -jnp.sum(reward, axis=0)
        cost_exp = jnp.exp(-(cost-jnp.min(cost)) / lam)
        weight = cost_exp / jnp.sum(cost_exp)

        a_mean = jnp.sum(a_sampled * weight[:, None], axis=0)

        return a_mean, rng, x_mppi_sampled

    def step_env(carry, unused):
        t, x, a_mean, rng = carry
        a_mean, rng, x_mppi_sampled = get_next_a_mean(t, x, a_mean, rng)
        action = a_mean[0]
        reward = reward_fn(t, x)
        x = dynamic_fn(x, action)
        t = t + 1
        return (t, x, a_mean, rng), (x, reward, x_mppi_sampled)


    def run_exp_once(rng):
        x0 = 0.0
        t0 = 0
        a_mean = jnp.zeros(H)
        carry = (t0, x0, a_mean, rng)
        _, (x, reward, x_mppi_sampled) = jax.lax.scan(step_env, carry, jnp.arange(env_steps))
        return x, reward, x_mppi_sampled

    rng = random.PRNGKey(0)
    sns.set_theme(style="whitegrid")
    if 'eval' in mode:
        # run experiment for repeat_times times
        eval_rng, rng = random.split(rng)
        xs, rewards, _ = jax.vmap(jax.jit(run_exp_once))(random.split(eval_rng, repeat_times))
        # plot results across repeat_times
        x_mean = jnp.mean(xs, axis=0)
        x_upper = x_mean + jnp.std(xs, axis=0)*2
        x_lower = x_mean - jnp.std(xs, axis=0)*2
        if 'plot' in mode: 
            plt.figure()
            plt.plot(x_mean, '-', label='mean')
            for i in range(20):
                plt.plot(xs[i], alpha=0.1, color='blue')
            plt.fill_between(jnp.arange(env_steps), x_lower, x_upper, alpha=0.5)
            # set title
            plt.title(f'{method} \n H={H} lam={lam} a={a} b={b} \n mean={jnp.mean(rewards):.3f} std={jnp.std(rewards):.3f}')
            plt.savefig(f'{method}_eval.png')
        return rewards
    elif mode == 'render':
        # plot environment for debug purpose
        render_rng, rng = random.split(rng)
        x, reward, x_mppi_sampled = run_exp_once(render_rng)
        x_tiled = x[:, None, None].repeat(N, axis=2)
        x_mppi_sampled = jnp.concatenate([x_tiled, x_mppi_sampled], axis=1)
        plt.figure()
        # plot mppi samples
        plt.plot(x, color='red', label='x')
        for t0 in range(0, 30, H):
            # plot 20 samples
            for i in range(20):
                t1 = t0+H
                tt = jnp.arange(t0, t1+1)
                x_sampled = x_mppi_sampled[t0, :, i]
                plt.plot(tt, x_sampled, alpha=0.1, color='green')
            # plot 2 std
            sampled_mean = jnp.mean(x_mppi_sampled[t0], axis=1)
            sampled_std = jnp.std(x_mppi_sampled[t0], axis=1)
            plt.plot(tt, sampled_mean, color='green')
            plt.fill_between(tt, sampled_mean+sampled_std*2, sampled_mean-sampled_std*2, alpha=0.3, color='green')
        plt.title(f'{method} render')
        plt.savefig(f'{method}_render.png')
        # plot a_cov as a heatmap with colorbar
        plt.figure()
        sns.set_theme(style="whitegrid")
        sns.heatmap(a_cov)
        plt.title(f'{method} a_cov \n log det={jnp.log(jnp.linalg.det(a_cov)):.3f}')
        plt.savefig(f'{method}_a_cov.png')


def main():
    N=1024
    H=10
    lam=0.01
    a=1.2
    b=1.0
    sig=0.3
    print('base: ', test_once(N=N, H=H, lam=lam, a=a, b=b, sig=sig, method='base', mode='eval_plot'))
    print('zeji: ', test_once(N=N, H=H, lam=lam, a=a, b=b, sig=sig, method='zeji', mode='eval_plot'))
    # df = pd.DataFrame(columns=['method', 'H', 'lam', 'a', 'b', 'sig', 'cost', 'N'])
    # for i in trange(1, 10):
    #     H = i
    #     for method in ['base', 'zeji']:
    #         rew = test_once(N=N, H=H, lam=lam, a=a, b=b, sig=sig, method=method, mode='eval')
    #         size = rew.shape[0]
    #         new_df = pd.DataFrame(
    #             {
    #                 'method': [method]*size,
    #                 'H': [H]*size,
    #                 'lam': [lam]*size,
    #                 'a': [a]*size,
    #                 'b': [b]*size,
    #                 'sig': [sig]*size,
    #                 'cost': H-rew.sum(axis=1), 
    #                 'N': N
    #             }
    #         )
    #         df = pd.concat([df, new_df])
    #     # test_once(N=N, H=H, lam=lam, a=a, b=b, sig=sig, method='zeji', mode='eval')
    # # plot df with seaborn barplot
    # sns.set_theme(style="whitegrid")
    # # set a figure size (5, 15)
    # plt.figure(figsize=(15, 5))
    # # set y limit from -10 to 50
    # # plt.ylim(-30, 100)
    # sns.barplot(x='H', y='cost', hue='method', data=df)
    # plt.savefig('mppi_devel.png')

if __name__ == '__main__':
    main()