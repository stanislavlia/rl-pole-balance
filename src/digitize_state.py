import numpy as np
from itertools import product

class PoleBalanceDigitizer():
    # CartPole observation bounds
    # Infinite ranges (velocity) are clipped to a practical window
    BOUNDS = [
        (-4.8,  4.8),   # cart position      — env hard limit
        (-3.0,  3.0),   # cart velocity       — practical clip
        (-0.418, 0.418),# pole angle (rad)    — env hard limit (~±24°)
        (-3.0,  3.0),   # pole angular vel    — practical clip
    ]

    def __init__(self,
                 cart_position_bins=10,
                 cart_velocity_bins=10,
                 pole_angle_bins=10,
                 pole_angular_velocity_bins=10):

        n_bins = [
            cart_position_bins,
            cart_velocity_bins,
            pole_angle_bins,
            pole_angular_velocity_bins,
        ]

        # n bin edges  →  n+1 regions, but we want exactly k bins
        # so we use k-1 edges; digitize then returns values in [0, k-1]
        self.bins = [
            np.linspace(low, high, k - 1)
            for (low, high), k in zip(self.BOUNDS, n_bins)
        ]

        self.n_bins = n_bins

    def transform(self, state) -> tuple:
        """Map a continuous CartPole observation to a discrete bin tuple."""
        return tuple(
            int(np.digitize(state[i], self.bins[i]))
            for i in range(4)
        )

    def __call__(self, state) -> tuple:
        return self.transform(state)

    @property
    def n_states(self) -> int:
        """Total number of discrete states (product of all bin counts)."""
        return int(np.prod(self.n_bins))

    def get_all_states(self) -> list:
        """Return all possible discrete states as a list of tuples."""
        return list(product(*[range(k) for k in self.n_bins]))