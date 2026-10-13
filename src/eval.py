import numpy as np
import gymnasium as gym

# Half-widths of the uniform start-state box at difficulty 1.0: x, x_dot, theta, theta_dot
BASE_RANGES = np.array([0.5, 0.5, 0.1, 0.5])

# Gym's default start is +-0.05 in every dimension
DIFFICULTY_LEVELS = {"easy": 0.25, "medium": 0.5, "hard": 1.0}


def sample_start_states(n, scale, seed):
    """n start states, uniform in a box of half-widths BASE_RANGES * scale.

    The same seed gives the same underlying samples for every scale (just
    stretched), so difficulty levels are compared on matched states.
    """
    rng = np.random.default_rng(seed)
    unit = rng.uniform(-1.0, 1.0, size=(n, 4))
    return unit * BASE_RANGES * scale


def run_episode(env, q, start_state):
    env.reset()
    env.unwrapped.state = np.asarray(start_state, dtype=np.float64)
    observation = np.asarray(start_state, dtype=np.float32)

    done, steps = False, 0
    while not done:
        action = q.eps_greedy_select_action(observation, eps=0.0)   # pure greedy
        observation, _, terminated, truncated, _ = env.step(action)
        steps += 1
        done = terminated or truncated
    return steps


def evaluate_policy(q, n=100, max_steps=1000, levels=DIFFICULTY_LEVELS, seed=0):
    """Greedy rollouts from n random start states per difficulty level.

    Returns {level: array of episode lengths} and prints a summary.
    """
    env = gym.make("CartPole-v1", max_episode_steps=max_steps, render_mode=None)
    results = {}

    for name, scale in levels.items():
        starts = sample_start_states(n, scale, seed)
        lens = np.array([run_episode(env, q, s) for s in starts])
        results[name] = lens

        n_early = int(np.sum(lens < max_steps))
        print(f"[{name:>6} x{scale:<4}] mean={lens.mean():7.1f}  std={lens.std():7.1f}  "
              f"min={lens.min():4d}  max={lens.max():4d}  "
              f"failed early: {n_early}/{n}")

    env.close()
    return results
