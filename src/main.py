import gymnasium as gym
import time
from policy import RandomPolicy, test_policy
from digitize_state import PoleBalanceDigitizer
from n_step_sarsa import NStepSARSA
from n_step_sarsa_offpolicy import NStepSarsaOffPolicy
import numpy as np

#ENV SETUP
HUMAN_MODE=True
DISPLAY_LAG_SEC=0.02
MAX_EPISODE_STEPS=1000

#REPRESENTATION
STATE_N_BINS=10

ACTION_SPACE=[0, 1]

#HYPERPARAMS
N=10
ALPHA=0.05
EPS=0.1
DISCOUNT=0.995
N_EPISODES=20_000


if __name__ == "__main__":
    digitzer = PoleBalanceDigitizer(cart_position_bins=STATE_N_BINS,
                                    cart_velocity_bins=STATE_N_BINS,
                                    pole_angle_bins=STATE_N_BINS,
                                    pole_angular_velocity_bins=STATE_N_BINS)
    env = gym.make("CartPole-v1", max_episode_steps=MAX_EPISODE_STEPS)

    # algorithm = NStepSARSA(env=env, eps=EPS, digitizer=digitzer,
    #                        discount=DISCOUNT, alpha=ALPHA, n=N)

    algorithm = NStepSarsaOffPolicy(env=env, eps=EPS, digitizer=digitzer,
                           discount=DISCOUNT, alpha=ALPHA, n=N)

    Q_learned, policy = algorithm.control(n_episodes=N_EPISODES, verbose=True, report_each_n=1_000)

    # Run learned policy, human render mode
    rewards = test_policy(
        env_id="CartPole-v1",
        policy=policy,
        digitizer=digitzer,
        n_episodes=10,
        human=True,
        max_episode_steps=MAX_EPISODE_STEPS,
        display_lag_sec=DISPLAY_LAG_SEC,
    )

    print(f"\nMean reward: {np.mean(rewards):.1f}  |  Max: {max(rewards):.0f}  |  Min: {min(rewards):.0f}")