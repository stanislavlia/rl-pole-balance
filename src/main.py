import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from sarsa import SarsaFA
from n_step_sarsa import NStepSarsaFA
from eval import evaluate_policy
# https://gymnasium.farama.org/environments/classic_control/cart_pole/

N_TRAIN_EPISODES = 1000

N_EVAL_EPISODES = 10
MAX_EPISODE_STEPS = 3000
MAX_STEPS_EVAL = 500   # step cap per evaluation episode
N_ROBUST_EVAL = 100   # random start states per difficulty level
EVAL_SEED = 0         # fixed seed -> identical start states across runs
REPORT_WINDOW = 50   # progress bar shows avg length over this many last episodes
ALPHA = 0.001
EPSILON = 0.02

#for n-step SARSA
N=5

# "poly" or "fourier"
REPRESENTATION = "fourier"


def moving_average(x, window):
    """Simple moving average; returns (x_positions, smoothed_values)."""
    x = np.asarray(x, dtype=float)
    window = max(1, min(window, len(x)))
    smoothed = np.convolve(x, np.ones(window) / window, mode="valid")
    return np.arange(window - 1, len(x)), smoothed


def plot_training(episode_lens, update_magnitudes, len_window=50, td_window=1000):
    episode_lens = np.asarray(episode_lens)
    td_abs = np.abs(np.asarray(update_magnitudes))   # TD errors are signed -> use |δ|

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    # Episode lengths (one value per episode)
    ax1.plot(episode_lens, alpha=0.3, label="Episode length")
    xs, ys = moving_average(episode_lens, len_window)
    ax1.plot(xs, ys, linewidth=2, label=f"{len_window}-episode moving avg")
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Steps balanced")
    ax1.set_title("Episode length")
    ax1.legend()
    ax1.grid(alpha=0.3)

    # TD-error magnitudes (one value per step -> much longer series, bigger window)
    xs, ys = moving_average(td_abs, td_window)
    ax2.plot(xs, ys, linewidth=1.5, color="tab:red")
    ax2.set_xlabel("Training step")
    ax2.set_ylabel("|TD error|")
    ax2.set_yscale("log")   # TD errors can span orders of magnitude
    ax2.set_title(f"|TD error| ({td_window}-step moving avg)")
    ax2.grid(alpha=0.3, which="both")

    fig.suptitle("Semi-gradient Sarsa on CartPole")
    fig.tight_layout()
    plt.show()


# ---------------------------------------------------------------
# 1. Train without rendering
# ---------------------------------------------------------------
train_env = gym.make("CartPole-v1", max_episode_steps=MAX_EPISODE_STEPS, render_mode=None)

#sarsa_fa = SarsaFA(alpha=ALPHA, epsilon=EPSILON, discount=1.0, representation=REPRESENTATION)
sarsa_fa = NStepSarsaFA(n=N, alpha=ALPHA, epsilon=EPSILON, discount=1.0, representation=REPRESENTATION)

episode_lens, update_magnitudes = sarsa_fa.control(train_env, n_episodes=N_TRAIN_EPISODES, report_window=REPORT_WINDOW)
train_env.close()

print(f"Average episode length over last 100 training episodes: {np.mean(episode_lens[-100:]):.1f}")

# ---------------------------------------------------------------
# 2a. Robustness eval: greedy rollouts from random start states
# ---------------------------------------------------------------
evaluate_policy(sarsa_fa.linear_fa, n=N_ROBUST_EVAL, max_steps=MAX_STEPS_EVAL, seed=EVAL_SEED)

# ---------------------------------------------------------------
# 2b. Evaluate the learned policy greedily, with rendering
# ---------------------------------------------------------------
eval_env = gym.make("CartPole-v1", max_episode_steps=MAX_STEPS_EVAL, render_mode="human")
q = sarsa_fa.linear_fa   # the trained approximator, not a fresh one

eval_returns = []
for ep in range(N_EVAL_EPISODES):
    observation, info = eval_env.reset()
    done, ep_return = False, 0.0

    while not done:
        action = q.eps_greedy_select_action(observation, eps=0.0)  # pure greedy
        observation, reward, terminated, truncated, info = eval_env.step(action)
        ep_return += reward
        done = terminated or truncated

    eval_returns.append(ep_return)
    print(f"Eval episode {ep + 1:2d}: return = {ep_return:.0f}")

eval_env.close()
print(f"Mean evaluation return: {np.mean(eval_returns):.1f} (max possible {MAX_STEPS_EVAL})")

# ---------------------------------------------------------------
# 3. Plots at the very end
# ---------------------------------------------------------------
plot_training(episode_lens, update_magnitudes)