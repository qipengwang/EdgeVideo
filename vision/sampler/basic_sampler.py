import os
import numpy as np
import random


class BasicSampler:
    def __init__(self, sample_rate: float=0, seed: int=None):
        assert 0 <= sample_rate <= 1
        self.sample_rate = sample_rate
        if seed:
            np.random.seed(seed)
    
    def sample(self, arr):
        return arr

