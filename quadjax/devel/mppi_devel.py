import jax
from jax import numpy as jnp
from jax import random
from matplotlib import pyplot as plt  
import seaborn as sns
from jaxopt import ProjectedGradient
from jaxopt.projection import projection_hyperplane


# generate metrix [[0, b, ba, ba^2, ..., ba^H-1], [0,0,b,ba, ..., ba^H-2], ...]
def generate_matrix(a, b, H):
    matrix = jnp.zeros((H+1, H))
    for i in range(1, H+1):
        for j in range(0, i):
            # matrix = jax.ops.index_update(matrix, jax.ops.index[i, j], b * a ** (j - i))
            matrix = matrix.at[i, j].set(b * a ** (i -1 - j))
    return matrix

def get_sigma_eign(eign_values, lam, H, sig):
    # Objective function
    def objective(params):
        return jnp.sum(eign_values/(1+2/lam*eign_values*jnp.exp(params))**2)

    # Constraint: x1 + x2 = 1, represented as a hyperplane
    def projection_fn(params, hyperparams_proj):
        return projection_hyperplane(params, hyperparams=(hyperparams_proj, 2*H*jnp.log(sig)))

    # Initialize 'ProjectedGradient' solver
    solver = ProjectedGradient(fun=objective, projection=projection_fn)

    # Initial parameters
    params_init = jnp.zeros(H)

    # Define the optimization problem
    sol = solver.run(params_init, hyperparams_proj=jnp.ones(H))

    # verify the solution
    o = eign_values
    s = jnp.exp(sol.params)
    cc = ((o**2)*s)/((1+2/lam*o*s)**3)
    print('verified solution: ', cc, 1/(o*(s**2)))

    # Print the optimal solution
    print("Optimal Solution: ", sol.params)

    return jnp.exp(sol.params)

def get_sigma_eign2(eign_values, lam, H, sig):
    log_o = jnp.log(eign_values)
    log_const =(2 *(H*2*jnp.log(sig)) + jnp.sum(log_o)) / (H)
    log_s = 0.5 * log_const - 0.5 * log_o

    o = eign_values
    s = jnp.exp(log_s)
    cc = ((o**2)*s)/((1+2/lam*o*s)**3)
    print('verified solution: ', cc, 1/(o*(s**2)))

    return s

def test_once(N=1024, H = 3, sig = 0.2, lam = 0.01, repeat_times=1024, env_steps=30, mode='base', a=1.0, b=1.0):

    if mode == 'base':
        a_cov = jnp.eye(H)*(sig**2)
    elif mode == 'zeji':
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
        # x_tar = jnp.sin(0.6*t)
        x_tar = 0.0
        reward = 1.0-jnp.abs(x - x_tar)
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
    # run experiment for repeat_times times
    eval_rng, rng = random.split(rng)
    xs, rewards, _ = jax.vmap(jax.jit(run_exp_once))(random.split(eval_rng, repeat_times))
    # plot environment for debug purpose
    render_rng, rng = random.split(rng)
    x, reward, x_mppi_sampled = run_exp_once(render_rng)
    x_tiled = x[:, None, None].repeat(N, axis=2)
    x_mppi_sampled = jnp.concatenate([x_tiled, x_mppi_sampled], axis=1)
    plt.figure()
    sns.set_theme(style="whitegrid")
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
    plt.title(f'{mode} render')
    plt.savefig(f'{mode}_render.png')
    # plot results across repeat_times
    x_mean = jnp.mean(xs, axis=0)
    x_upper = x_mean + jnp.std(xs, axis=0)*2
    x_lower = x_mean - jnp.std(xs, axis=0)*2
    plt.figure()
    plt.plot(x_mean, '-', label='mean')
    plt.fill_between(jnp.arange(env_steps), x_lower, x_upper, alpha=0.5)
    # set title
    plt.title(f'{mode} \n H={H} lam={lam} a={a} b={b} \n mean={jnp.mean(rewards):.3f} std={jnp.std(rewards):.3f}')
    plt.savefig(f'{mode}_eval.png')

    # plot a_cov as a heatmap with colorbar
    plt.figure()
    sns.set_theme(style="whitegrid")
    sns.heatmap(a_cov)
    plt.title(f'{mode} a_cov \n log det={jnp.log(jnp.linalg.det(a_cov)):.3f}')
    plt.savefig(f'{mode}_a_cov.png')

    return rewards.mean(), rewards.std()

def main():
    H=10
    lam=0.01
    a=2.0
    b=1.0
    sig=1.0
    print('base: ', test_once(H=H, lam=lam, a=a, b=b, sig=sig, mode='base'))
    print('zeji: ', test_once(H=H, lam=lam, a=a, b=b, sig=sig, mode='zeji'))

if __name__ == '__main__':
    main()