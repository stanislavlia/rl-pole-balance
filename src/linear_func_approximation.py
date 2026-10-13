import numpy as np
import itertools
import random

SCALES = np.array([2.4, 3.0, 0.21, 3.5])   # x, x_dot, theta, theta_dot
#SCALES = np.array([1, 1, 1, 1]) #don't normalize

class LinearApproximatedValueFunc():
    def __init__(self, n_actions, representation="poly", fourier_order=2):

        self.n_actions = n_actions
        self.representation = representation

        if representation == "poly":
            self.n_features = 15
            self._features_fn = self._rep_poly
        elif representation == "fourier":
            # full grid of integer frequency vectors, built once
            self._coeffs = np.array(
                list(itertools.product(range(fourier_order + 1), repeat=4)), dtype=float
            )
            self.n_features = len(self._coeffs)
            self._features_fn = self._rep_fourier
        else:
            raise ValueError(f"Unknown representation: {representation}")

        self.weights_vectors = self._initialize_weights()

    def _initialize_weights(self):
        # separate weights vector per action, initialized to zero
        return [np.zeros(self.n_features) for _ in range(self.n_actions)]

    def get_state_representation(self, state):
        return self._features_fn(state)

    @staticmethod
    def _rep_poly(state):
        x, v, th, w = np.clip(np.asarray(state) / SCALES, -1, 1)
        return np.array([
            1.0,                       # bias
            x, v, th, w,               # linear
            x*v, x*th, x*w,            # interactions (all 6 pairs)
            v*th, v*w, th*w,
            x**2, v**2, th**2, w**2,   # squares
        ])

    def _rep_fourier(self, state):
        # normalize to [0, 1]
        state = (np.clip(np.asarray(state) / SCALES, -1, 1) + 1) / 2
        return np.cos(np.pi * self._coeffs @ state)


    def estimate_value(self, state: list, action : int):

        #retrieve needed weights for corresponding action
        weights = self.weights_vectors[action]
        feature_vector = self.get_state_representation(state)

        return np.dot(weights, feature_vector)
    
    def update_weights(self, upd_target, alpha, action, state):

        feature_vector = self.get_state_representation(state)

        #Semi-gradient update rule
        self.weights_vectors[action] += alpha * upd_target * feature_vector
    
    def eps_greedy_select_action(self, state: list, eps: float) -> int:
        

        values = np.array([self.estimate_value(state, action) for action in list(range(self.n_actions))])
        if np.random.random() < eps:
            #randomly select action
            action = np.random.choice(list(range(self.n_actions)))
            return action
        
        #argmax with ties break even
        return int(np.random.choice(np.flatnonzero(values == values.max())))




