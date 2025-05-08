import random
import numpy as np

def apply_symmetries(states, pis, vs):
    # placeholder for board symmetries
    return states, pis, vs

class ReplayBuffer:
    def __init__(self, size):
        self.size = size
        self.buffer = []

    def add(self, samples):
        self.buffer.extend(samples)
        if len(self.buffer) > self.size:
            self.buffer = self.buffer[-self.size:]

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)