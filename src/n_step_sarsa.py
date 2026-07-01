from policy import PolicyBase, DeterministicQPolicy, make_deterministic_policy, make_epsilon_soft_policy
import numpy as np
from tqdm import tqdm
from digitize_state import PoleBalanceDigitizer
import random

class NStepSARSA():
    def __init__(self,
                 env,
                 digitizer: PoleBalanceDigitizer,
                 discount=1.0,
                 eps=0.1,
                 alpha=0.1,
                 n=2,
                 ):
        
        self.env = env
        self.digitizer = digitizer
        
        self.action_space = list(range(env.action_space.n)) 
        self.state_space = digitizer.get_all_states()

        
        #hyperparams
        self.discount = discount
        self.eps = eps
        self.alpha = alpha
        self.n = n
    
    def _initialize_Q(self):


        Q = dict()

        for s in self.state_space:
            
            Q[s] = {a: random.uniform(0, 0.2) for a in self.action_space}

        return Q

    def control(self, n_episodes=1000, verbose=False, report_each_n=1000):
        
        Q = self._initialize_Q()

        #initialize policy
        policy : PolicyBase = make_epsilon_soft_policy(action_space=self.action_space,
                                                       Q=Q,
                                                       epsilon=self.eps,
                                                       digitizer=None,)
        
        #track total reward and ep length
        cumulative_rewards = []
        episode_lens = []

        #For each episode...
        for i in tqdm(range(n_episodes), desc=f"n-step-Sarsa(n={self.n})"):
            
            #init state
            state, _ = self.env.reset()
            state = self.digitizer(state)

            #get action
            action = policy[state]

            #store
            states = [state]
            actions = [action]
            rewards = []

            T = 10 ** 10

            for t in range(T):
                if t < T:
                    #Act and collect experience/observations
                    next_state, reward, terminated, truncated, _ = self.env.step(action)
                    next_state = self.digitizer(next_state)

                    #store next_R and next_S
                    states.append(next_state)
                    rewards.append(reward)

                    done = terminated or truncated
                    
                    if done: #next_S is terminal
                        
                        if (i+1) % report_each_n == 0 or i == 100:
                            print()
                            print("CUM REWARD AVERAGE: ", np.mean(cumulative_rewards[-100:]),
                                   " AVG EP LENGTH: ", np.mean(episode_lens[-100:]))

                        T = t + 1

                        #track cumulative rewards and ep length
                        cum_reward = sum(rewards)
                        cumulative_rewards.append(cum_reward)

                        episode_lens.append(len(states))


                    else:
                        #select next action
                        action = policy[next_state]
                        actions.append(action)

                tau = t - self.n + 1 #tau - time whose estimate is being updates
                if tau >= 0:
                    #Update Q estimates
                    G_return = sum([
                        (self.discount ** (k - tau)) * rewards[k]
                        for k in range(tau, min(tau + self.n, T))
                    ])
                    if tau + self.n < T:
                        G_return += (self.discount ** self.n) * Q[states[tau+self.n]][actions[tau+self.n]]

                    #update rule
                    Q[states[tau]][actions[tau]] += self.alpha * (G_return - Q[states[tau]][actions[tau]])
                            
                if tau == T - 1:
                    break

        #Derive target policy
        target_policy = make_deterministic_policy(Q)
        return Q, target_policy
    
