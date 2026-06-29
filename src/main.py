import gymnasium as gym
import time
from policy import RandomPolicy
from digitize_state import PoleBalanceDigitizer
from n_step_sarsa import NStepSARSA


#ENV SETUP
HUMAN_MODE=True
DISPLAY_LAG_SEC=0.05
MAX_EPISODE_STEPS=500

#REPRESENTATION
STATE_N_BINS=14

ACTION_SPACE=[0, 1]

#HYPERPARAMS
N=8
ALPHA=0.05
EPS=0.05
DISCOUNT=0.99

N_EPISODES=30_000


def test_random_policy():

    digitzer = PoleBalanceDigitizer(cart_position_bins=STATE_N_BINS,
                                    cart_velocity_bins=STATE_N_BINS,
                                    pole_angle_bins=STATE_N_BINS,
                                    pole_angular_velocity_bins=STATE_N_BINS)


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
        print(f"State = {state}; Digitized State: {digitzer.transform(obs)}; Action = {action}; Reward={reward}")
        
        if terminated or truncated:
            current_episode += 1
            print()
            print("=" * 20 + "EPISODE DONE" + "=" * 20)
            print()

        if terminated or truncated:
            obs, _ = env.reset()

    env.close()

if __name__ == "__main__":

    digitzer = PoleBalanceDigitizer(cart_position_bins=STATE_N_BINS,
                                    cart_velocity_bins=STATE_N_BINS,
                                    pole_angle_bins=STATE_N_BINS,
                                    pole_angular_velocity_bins=STATE_N_BINS)
    
    
    env = gym.make("CartPole-v1",
                   max_episode_steps=MAX_EPISODE_STEPS)
    
    algorithm = NStepSARSA(env=env,
                           eps=EPS,
                           digitizer=digitzer,
                           discount=DISCOUNT,
                           alpha=ALPHA,
                           n=N)
    
    Q_learned, policy = algorithm.control(n_episodes=N_EPISODES, verbose=True, report_each_n=1_000)

    env_human = gym.make("CartPole-v1",
                         render_mode="human",
                         max_episode_steps=MAX_EPISODE_STEPS)

    for episode in range(10):
        obs, _ = env_human.reset()
        total_reward = 0
        step = 0

        while True:
            state = digitzer(obs)
            action = policy[state]
            obs, reward, terminated, truncated, _ = env_human.step(action)
            total_reward += reward
            step += 1
            time.sleep(DISPLAY_LAG_SEC)

            if terminated or truncated:
                print(f"Episode {episode + 1:2d} | Steps: {step:4d} | Total reward: {total_reward:.0f}")
                break

    env_human.close()
