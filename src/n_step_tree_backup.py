from policy import PolicyBase, DeterministicQPolicy, make_deterministic_policy, make_epsilon_soft_policy
import numpy as np
from tqdm import tqdm
from digitize_state import PoleBalanceDigitizer
import random

class NStepTreeBackup():
    def __init__(self,
                 env,
                 digitizer: PoleBalanceDigitizer,
                 discount=1.0,
                 eps=0.1,
                 alpha=0.1,
                 n=2,
                 behavior_policy: PolicyBase = None
                 ):
        

        self.env = env
        self.digitizer = digitizer
        
        self.action_space = list(range(env.action_space.n)) 
        self.state_space = digitizer.get_all_states()

        self.behavior_policy = behavior_policy


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

    
    
    def control(self, n_episodes=True, verbose=False, report_each_n=1000):
        
        Q = self._initialize_Q()

        #initialize target policy: Greedy wrt to Q
        self.target_policy =  make_deterministic_policy(Q)
        
        #behavior policy: intitialize if not specified
        if self.behavior_policy is None:
            self.behavior_policy = make_epsilon_soft_policy(action_space=self.action_space,
                                                            Q=Q,
                                                            epsilon=self.eps,
                                                            digitizer=None)#avoid digitizing twice

        #track total reward and ep length
        cumulative_rewards = []
        episode_lens = []

        #For each episode...
        for i in tqdm(range(n_episodes), desc=f"n-step-Tree-Backup(n={self.n})"):
            
            #init state
            state, _ = self.env.reset()
            state = self.digitizer(state)

            # we can select action arbitrary (algorithm does not need behavior policy). 
            # use behavior policy for convenience
            action = self.behavior_policy[state]
            
            #store
            states = [state]
            actions = [action]
            rewards = []

            T = 10 ** 10


            for t in range(T):
                #Act and collect experience/observations
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
                        #Action selection can be arbitary: here we use B policy
                        action = self.behavior_policy[next_state]
                        actions.append(action)
                        
                
                #index: time whose estimates is being updated
                tau = t - self.n + 1
                if tau >= 0:
                    #Update
                    if t+1 >= T:
                        G = rewards[T -1]
                    else:

                        #Implement recursion computation through loop
                        G = rewards[t] + self.discount *  sum([
                                                                self.target_policy.prob(state=states[t + 1], action=a) * Q[states[t + 1]][a]
                                                                for a in self.action_space
                                                            ])
                    #incremental computation of recursion starts:
                    for k in range(min(t, T-1), tau, -1): #decrementing sequence

                        G = rewards[k - 1]  + self.discount *  sum([
                                                                self.target_policy.prob(state=states[k], action=a) * Q[states[k]][a]
                                                                if a != actions[k] else 0
                                                                for a in self.action_space
                                                            ])  \
                        + self.discount * self.target_policy.prob(state=states[k], action=actions[k]) * G

                    #Update of Q
                    Q[states[tau]][actions[tau]] += self.alpha * (G - Q[states[tau]][actions[tau]])

                if tau == T - 1:
                    break

        return Q, self.target_policy


