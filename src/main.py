import gymnasium as gym
import time
from linear_func_approximation import LinearApproximatedValueFunc
#https://gymnasium.farama.org/environments/classic_control/cart_pole/

HUMAN_MODE=True
MAX_EPISODE_STEPS=1000


env = gym.make("CartPole-v1", max_episode_steps=MAX_EPISODE_STEPS,
               render_mode="human" if HUMAN_MODE else None)
observation, info = env.reset()


lfa =  LinearApproximatedValueFunc(n_actions=2)

for _ in range(500):

    action = env.action_space.sample()  # Take a random action

    state_rep = LinearApproximatedValueFunc.get_state_representation(observation)

    print(f"Observation: {observation} | representation: {state_rep} | action: {action}")

    estimate_value = lfa.estimate_value(observation, action=action)
    print("Estimated value =  ", estimate_value)


    observation, reward, terminated, truncated, info = env.step(action)
    
    if terminated or truncated:
        observation, info = env.reset()

env.close()