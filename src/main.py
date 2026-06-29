import gymnasium as gym
import time
from policy import RandomPolicy


#ENV SETUP
HUMAN_MODE=True
DISPLAY_LAG_SEC=0.05
MAX_EPISODE_STEPS=500
N_EPISODES=3

ACTION_SPACE=[0, 1]

if __name__ == "__main__":

    random_policy = RandomPolicy(
        action_space=ACTION_SPACE,
        probs={
            0: 0.5,
            1: 0.5
        }
    )

    policy = random_policy
    env = gym.make("CartPole-v1",
                   render_mode="human" if HUMAN_MODE else None,
                   max_episode_steps=MAX_EPISODE_STEPS)
    obs, _ = env.reset()

    current_episode=0

    while current_episode < N_EPISODES:
        pole_angle = obs[2]            # positive = leaning right

        state = (pole_angle, )
        action = policy[state]

        
        if HUMAN_MODE:
            #small delay
            time.sleep(DISPLAY_LAG_SEC)

        obs, reward, terminated, truncated, _ = env.step(action)
        print(f"State = {state}; Action = {action}; Reward={reward}")
        if terminated or truncated:
            current_episode += 1
            print()
            print("=" * 20 + "EPISODE DONE" + "=" * 20)
            print()

        if terminated or truncated:
            obs, _ = env.reset()

    env.close()