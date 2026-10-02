import numpy as np


class LinearApproximatedValueFunc():
    def __init__(self, n_actions):
        
        self.n_actions = n_actions
        self.n_features = 13 #Hardcoded state representation function

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
          position / 5,
          velocity / 10,
          angle,
          angular_velocity,

          #interactions
          position * velocity,
          angle * angular_velocity,
          velocity * angle,
          angular_velocity * position,

          #squares
          position ** 2 / 25,
          angle ** 2,
          velocity ** 2 / 100,
          angular_velocity ** 2
        ])

    def estimate_value(self, state: list, action : int):

        #retrieve needed weights for corresponding action
        weights = self.weights_vectors[action]
        feature_vector = self.get_state_representation(state)

        return np.dot(weights, feature_vector)
    
    def update_weights(self, upd_target, alpha, action, state):

        feature_vector = self.get_state_representation(state)

        #SGD update rule
        self.weights_vectors[action] += alpha * upd_target * feature_vector
    
    def eps_greedy_select_action(self, state: list, eps: float) -> int:
        

        values = [self.estimate_value(state, action) for action in list(range(self.n_actions))]
        print("Values: ", values)
        if np.random.random() < eps:
            #randomly select action
            action = np.random.choice(list(range(self.n_actions)))
            print("Random action: ", action)
            return action
        print("Greedy action: ", np.argmax(values))
        return np.argmax(values)
