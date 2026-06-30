import gymnasium as gym
import time
from policy import RandomPolicy, test_policy
from digitize_state import PoleBalanceDigitizer
from n_step_sarsa import NStepSARSA
import pandas as pd
import numpy as np

#Episode budget
N_EPISODES=10_000
EVAL_EPISODES=500


#FIXED HYPERPARAMS
ALPHA=0.05
EPS=0.05
DISCOUNT=0.99

#REPRESENTATION
STATE_N_BINS=8
MAX_EPISODE_STEPS=500
ACTION_SPACE=[0, 1]

#to ablate
N_SPACE=[4, 10, 15, 20, 25, 30]



def ablate_n_param():

    digitzer = PoleBalanceDigitizer(cart_position_bins=STATE_N_BINS,
                                    cart_velocity_bins=STATE_N_BINS,
                                    pole_angle_bins=STATE_N_BINS,
                                    pole_angular_velocity_bins=STATE_N_BINS)
    env = gym.make("CartPole-v1", max_episode_steps=MAX_EPISODE_STEPS)

    stats = []

    for n in N_SPACE:
        print("Evaluate N-step-SARSA; N =", n)
    
        algorithm = NStepSARSA(env=env, eps=EPS, digitizer=digitzer,
                           discount=DISCOUNT, alpha=ALPHA, n=n)
        
        Q_learned, policy = algorithm.control(n_episodes=N_EPISODES, verbose=True, report_each_n=2_000)

        # Run learned policy, human render mode
        rewards = test_policy(
            env_id="CartPole-v1",
            policy=policy,
            digitizer=digitzer,
            n_episodes=EVAL_EPISODES,
            human=False,
            max_episode_steps=MAX_EPISODE_STEPS,
            display_lag_sec=0.0
        )

        stats.append(
            {"N" : n,
             "mean_reward": np.mean(rewards),
             "rewards_std" : np.std(rewards)}
        )

    return stats

def save_results(stats, filepath="n_step_ablation_results.csv"):
    """
    Sort ablation results by mean_reward (descending) and save to CSV.

    Args:
        stats    : list of dicts, e.g. [{"N": 2, "mean_reward": 10.5, "rewards_std": 2.1}, ...]
        filepath : output CSV path.

    Returns:
        Sorted DataFrame.
    """
    df = pd.DataFrame(stats)
    df = df.sort_values("mean_reward", ascending=False).reset_index(drop=True)
    df.to_csv(filepath, index=False)
    print(f"Results saved to {filepath}")
    return df

if __name__ == "__main__":
    stats = ablate_n_param()

    print('=========')
    df = save_results(stats, filepath="n_step_ablation_results.csv")
    print(df)