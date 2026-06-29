from policy import PolicyBase, DeterministicQPolicy, make_deterministic_policy
import numpy as np
from tqdm import tqdm
from digitize_state import PoleBalanceDigitizer

class NStepSARSA():
    def __init__(self,
                 action_space,
                 state_space,
                 digitizer: PoleBalanceDigitizer,
                 eps=0.1,
                 alpha=0.1,
                 n=2,
                 ):
        
        self.action_space = action_space
        self.state_space = state_space
        self.digitizer = digitizer

        self.eps = eps
        self.alpha = alpha
        self.n = n
    

