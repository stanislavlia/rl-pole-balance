import numpy as np


class LinearApproximatedValueFunc():
    def __init__(self, n_actions):
        
        self.n_actions = n_actions
        self.n_features = 11 #Hardcoded state representation function

        self.weights_vectors = self._initialize_weights()

    def _initialize_weights(self):
        
        weights_vectors = []


        for i in  range(self.n_actions):
            #store separate weights vector for each action
            #initialize to zero
            weights_vectors.append(
                np.zeros(self.n_features)
            )
        return weights_vectors
    
    @staticmethod
    def get_state_representation(state: list):
        """Fixed state representation"""

        position, velocity, angle, angular_velocity = state

        return np.array([
          1, #bias
          position,
          velocity,
          angle,
          angular_velocity,

          #interactions
          position * velocity,
          angle * angular_velocity,
          velocity * angle,
          angular_velocity * position,

          #squares
          velocity ** 2,
          angular_velocity ** 2
        ])

    def estimate_value(self, state: list, action : int):

        #retrieve needed weights for corresponding action
        weights = self.weights_vectors[action]
        feature_vector = self.get_state_representation(state)

        return np.dot(weights, feature_vector)