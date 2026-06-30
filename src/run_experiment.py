import gymnasium as gym
import time
from policy import RandomPolicy, test_policy
from digitize_state import PoleBalanceDigitizer
from n_step_sarsa import NStepSARSA
import pandas as pd
import numpy as np
from scipy import stats
from itertools import product

# Episode budget
N_EPISODES = 10_000
EVAL_EPISODES = 500

# FIXED HYPERPARAMS
EPS = 0.05
DISCOUNT = 0.99

# REPRESENTATION
STATE_N_BINS = 10
MAX_EPISODE_STEPS = 500
ACTION_SPACE = [0, 1]

# To ablate
N_SPACE = [2, 5, 8, 10, 20, 30]
ALPHAS = [0.01, 0.05, 0.1, 0.2]


def mean_confidence_interval(data, confidence=0.95):
    """
    Compute mean and confidence interval using the t-distribution
    (more accurate than normal approx for finite samples).

    Returns:
        mean, ci_low, ci_high, margin
    """
    data = np.asarray(data)
    n = len(data)
    mean = np.mean(data)
    sem = stats.sem(data)                          # standard error of the mean
    margin = sem * stats.t.ppf((1 + confidence) / 2, df=n - 1)
    return mean, mean - margin, mean + margin, margin


def ablate_hyperparams():
    digitzer = PoleBalanceDigitizer(
        cart_position_bins=STATE_N_BINS,
        cart_velocity_bins=STATE_N_BINS,
        pole_angle_bins=STATE_N_BINS,
        pole_angular_velocity_bins=STATE_N_BINS,
    )
    env = gym.make("CartPole-v1", max_episode_steps=MAX_EPISODE_STEPS)

    stats_list = []

    for n, alpha in product(N_SPACE, ALPHAS):
        print(f"Evaluate N-step-SARSA; N={n}, alpha={alpha}")

        algorithm = NStepSARSA(
            env=env, eps=EPS, digitizer=digitzer,
            discount=DISCOUNT, alpha=alpha, n=n,   # ✅ now uses the looped alpha
        )
        Q_learned, policy = algorithm.control(
            n_episodes=N_EPISODES, verbose=True, report_each_n=2_000
        )

        rewards = test_policy(
            env_id="CartPole-v1",
            policy=policy,
            digitizer=digitzer,
            n_episodes=EVAL_EPISODES,
            human=False,
            max_episode_steps=MAX_EPISODE_STEPS,
            display_lag_sec=0.0,
        )

        mean, ci_low, ci_high, margin = mean_confidence_interval(rewards, confidence=0.95)

        stats_list.append({
            "N": n,
            "alpha": alpha,
            "mean_reward": mean,
            "ci_low": ci_low,
            "ci_high": ci_high,
            "ci_margin": margin,   # mean_reward ± ci_margin = 95% CI
        })

    return stats_list


def save_results(stats, filepath="n_step_ablation_results.csv"):
    """
    Sort ablation results by mean_reward (descending) and save to CSV.
    """
    df = pd.DataFrame(stats)
    df = df.sort_values("mean_reward", ascending=False).reset_index(drop=True)
    df.to_csv(filepath, index=False)
    print(f"Results saved to {filepath}")
    return df


if __name__ == "__main__":
    stats = ablate_hyperparams()
    print('=========')
    df = save_results(stats, filepath="n_step_alpha_ablation_results.csv")
    print(df)