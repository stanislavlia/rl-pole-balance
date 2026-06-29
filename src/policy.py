from collections import defaultdict
import numpy as np

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
    
    
