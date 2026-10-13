import numpy as np
from linear_func_approximation import LinearApproximatedValueFunc
from tqdm import tqdm



class NStepSarsaFA():
    def __init__(self, n, alpha, epsilon, discount=1.0, representation="poly"):
        
        self.alpha = alpha
        self.epsilon = epsilon
        self.discount = discount

        self.n = n

        
        self.linear_fa = LinearApproximatedValueFunc(n_actions=2, representation=representation)


    def control(self, env, n_episodes, report_window=50):
        update_magnitudes = []
        episode_lens = []

        pbar = tqdm(range(n_episodes), desc="n-step SARSA with FA Learning...")
        for i in pbar:
            if episode_lens:
                pbar.set_postfix(avg_len=f"{np.mean(episode_lens[-report_window:]):.1f}")

            state, info = env.reset()
            action = self.linear_fa.eps_greedy_select_action(state=state, eps=self.epsilon)

            #store
            states = [state]
            actions = [action]
            rewards = []

            T = 10**10

            for t in range(T):
                if t < T:
                    #Act and collect experience/observations
                    next_state, reward, terminated, truncated, _ = env.step(action)

                    #store next_R and next_S
                    states.append(next_state)
                    rewards.append(reward)

                    done = terminated or truncated
                    
                    if done: #next_S is terminal
                        T = t + 1
                        episode_lens.append(len(states))

                    else:
                        #select next action
                        action = self.linear_fa.eps_greedy_select_action(next_state, self.epsilon)
                        actions.append(action)

                tau = t - self.n + 1 #tau - time whose estimate is being updates
                if tau >= 0:
                    #Update Q estimates
                    G_return = sum([
                        (self.discount ** (k - tau)) * rewards[k]
                        for k in range(tau, min(tau + self.n, T))
                    ])
                    if tau + self.n < T:
                        G_return += (self.discount ** self.n) * self.linear_fa.estimate_value(states[tau+self.n], actions[tau+self.n])

                    #update
                    update_target = (G_return - self.linear_fa.estimate_value(states[tau],actions[tau]))
                    update_magnitudes.append(update_target)
                    self.linear_fa.update_weights(update_target, self.alpha, actions[tau], states[tau] )

                            
                if tau == T - 1:
                    break
                    
        return episode_lens, update_magnitudes