import numpy as np
from linear_func_approximation import LinearApproximatedValueFunc
from tqdm import tqdm

class SarsaFA():
    def __init__(self, alpha, epsilon, discount=1.0):
        
        self.alpha = alpha
        self.epsilon = epsilon
        self.discount = discount
        
        self.linear_fa = LinearApproximatedValueFunc(n_actions=2)


    def control(self, env, n_episodes):
        
        update_magnitudes = []
        episode_lens = []

        for i in tqdm(range(n_episodes), desc="SARSA with FA Learning..."):

            done = False
            state, info = env.reset()
            action = self.linear_fa.eps_greedy_select_action(state=state, eps=self.epsilon)

            step = 0

            while not done:
    
                next_state, reward, terminated, truncated, info = env.step(action)
                next_action = self.linear_fa.eps_greedy_select_action(state=next_state, eps=self.epsilon)

                step += 1
                
                #update
                next_state_value = self.linear_fa.estimate_value(next_state, next_action)
                if terminated or truncated:
                    done = True
                    next_state_value = 0

                    #track len of episode
                    episode_lens.append(step)

                update_target = (reward + self.discount * next_state_value) - \
                                    self.linear_fa.estimate_value(state, action)
                
                update_magnitudes.append(update_target)
                
                #update weights
                self.linear_fa.update_weights(update_target, self.alpha, action, state)

                state = next_state
                action = next_action

        return episode_lens, update_magnitudes

                

            
            


    