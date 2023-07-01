import jax
import time

from quadjax.quad2d import Quad2D

def main():
    env = Quad2D()
    # run 8192 environment with random actions for 1000 steps with jax scan
    env_num = 1
    rng = jax.random.PRNGKey(0)
    rng, action_rng = jax.random.split(rng, 2)  # Split PRNG key here
    actions = jax.random.uniform(action_rng, (env_num, 2))
    # jit step function
    def step_fn(state, action):
        obs, state, reward, done, info = env.step(rng, state, action)
        return state, action
    step_fn_jit = jax.jit(step_fn)
    # run 1000 steps and time it
    t = time.time()
    states = jax.lax.scan(step_fn_jit, env.reset(rng), actions)
    print("jax scan time: ", time.time() - t)

if __name__ == '__main__':
    main()