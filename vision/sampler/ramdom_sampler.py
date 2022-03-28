import numpy as np
from .basic_sampler import BasicSampler


class RandomSampler(BasicSampler):
    def __init__(self, sample_rate: float=0.5, seed: int=None):
        super(RandomSampler, self).__init__(sample_rate, seed)
    
    def sample(self, arr):
        sample_cnt = int(self.sample_rate * len(arr))
        return np.random.choice(arr, sample_cnt, replace=False)