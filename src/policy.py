from collections import defaultdict
import numpy as np
from digitize_state import PoleBalanceDigitizer
import time
import gymnasium as gym


class PolicyBase():
    def __init__(self, action_space: list):
        
        self.action_space = action_space
    
    def prob(self, state, action):
        #deterministic policy: always select action_space[0]
        #so prob is either 1.0 or 0

        return 1.0 if action == self.action_space[0] else 0.0


    def sample(self, state):
        #returns action
        return self.action_space[0]

    def __getitem__(self, state):
        return self.sample(state)


class RandomPolicy(PolicyBase):
    def __init__(self, action_space, probs=None):

        self.action_space = action_space
        self.probs = probs
        if probs:
            assert len(action_space) == len(probs)
        else:
            self.probs = {action: 1 / len(action_space)
                          for action in action_space}
        
    
    def sample(self, state):
        return np.random.choice(self.action_space, p=list(self.probs.values()))

    def prob(self, state, action):
        return self.probs[action]


class DeterministicQPolicy(PolicyBase):
    """Greedy policy: always selects the action with the highest Q-value."""

    def __init__(self, Q: dict):
        # Derive action space from Q-table keys
        action_space = list(next(iter(Q.values())).keys())
        super().__init__(action_space)
        self.Q = Q

    def sample(self, state) -> object:
        q_values = self.Q.get(state)
        if q_values is None:
            # Unseen state — fall back to uniform random
            return np.random.choice(self.action_space)
        return max(q_values, key=q_values.get)

    def prob(self, state, action) -> float:
        return 1.0 if action == self.sample(state) else 0.0


def make_deterministic_policy(Q: dict) -> PolicyBase:
    """
    Construct a greedy deterministic policy from a Q-table.

    Args:
        Q: Nested dict of shape {state: {action: value}}.

    Returns:
        A PolicyBase-compatible policy that always picks argmax_a Q[s][a].
    """
    if not Q:
        raise ValueError("Q-table is empty — cannot derive a policy.")
    return DeterministicQPolicy(Q)


def make_epsilon_soft_policy(
    action_space: list,
    Q: dict,
    epsilon: float = 0.1,
    digitizer: PoleBalanceDigitizer = None,
    seed: int = None,
) -> PolicyBase:
    rng = np.random.default_rng(seed)

    class _EpsilonSoftPolicy(PolicyBase):
        def __init__(self):
            super().__init__(action_space)

        def sample(self, state):
            key = digitizer(state) if digitizer else state
            if rng.random() < epsilon:
                return rng.choice(self.action_space)
            return max(self.action_space, key=lambda a: Q[key][a])

        def prob(self, state, action) -> float:
            key = digitizer(state) if digitizer else state
            greedy = max(self.action_space, key=lambda a: Q[key][a])
            p = epsilon / len(self.action_space)
            if action == greedy:
                p += (1.0 - epsilon)
            return p

    return _EpsilonSoftPolicy()


def test_policy(env_id, policy, digitizer, n_episodes=10, human=True,
                 max_episode_steps=500, display_lag_sec=0.02):
    """
    Run a policy on CartPole and report results.

    Args:
        env_id           : gym environment id, e.g. "CartPole-v1".
        policy            : PolicyBase-compatible policy (uses policy[state]).
        digitizer         : PoleBalanceDigitizer to discretize raw observations.
        n_episodes        : number of episodes to run.
        human             : if True, renders with render_mode="human" + adds delay.
        max_episode_steps : episode step cap.
        display_lag_sec   : sleep between steps when human=True (for visibility).

    Returns:
        List of total rewards, one per episode.
    """
    env = gym.make(
        env_id,
        render_mode="human" if human else None,
        max_episode_steps=max_episode_steps,
    )

    rewards = []

    for episode in range(n_episodes):
        obs, _ = env.reset()
        total_reward = 0
        step = 0

        while True:
            state = digitizer(obs)
            action = policy[state]
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            step += 1

            if human:
                time.sleep(display_lag_sec)

            if terminated or truncated:
                if human:
                    print(f"Episode {episode + 1:2d} | Steps: {step:4d} | Total reward: {total_reward:.0f}")
                break

        rewards.append(total_reward)

    env.close()
    return rewards