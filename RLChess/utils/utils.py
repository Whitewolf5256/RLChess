import pickle
import random
import numpy as np

def apply_symmetries(states, pis, vs):
    return states, pis, vs  # Placeholder

class ReplayBuffer:
    def __init__(self, size):
        self.size = size
        self.max_per_category = size // 3
        self.win = []    # stores samples where z == +1
        self.lose = []   # stores samples where z == -1
        self.tie = []    # stores samples where z == 0

    def add(self, samples):
        if not samples:
            return
        # Split samples by their value (z)
        for sample in samples:
            outcome = sample[2]
            if outcome == 1:
                self._add_category(self.win, [sample])
            elif outcome == -1:
                self._add_category(self.lose, [sample])
            else:
                self._add_category(self.tie, [sample])

    def _add_category(self, buffer_list, samples):
        buffer_list.extend(samples)
        # FIFO: remove oldest if overflow
        while len(buffer_list) > self.max_per_category:
            del buffer_list[0:(len(buffer_list) - self.max_per_category)]

    def sample_balanced(self, batch_size):
        per_class = batch_size // 3

        # Determine how much we *can* sample per class
        print(f"[Buffer Sampling] win: {len(self.win)}, lose: {len(self.lose)}, tie: {len(self.tie)}")

        actual_per_class = min(len(self.win), len(self.lose), len(self.tie), per_class)

        win_samples = random.sample(self.win, actual_per_class)
        lose_samples = random.sample(self.lose, actual_per_class)
        tie_samples = random.sample(self.tie, actual_per_class)

        print(f"[Buffer Sampling] win: {len(win_samples)}, lose: {len(lose_samples)}, tie: {len(tie_samples)}")
        return win_samples + lose_samples + tie_samples

    def __len__(self):
        return len(self.win) + len(self.lose) + len(self.tie)

    def __iter__(self):
        # Allows iteration over all samples in the buffer
        return iter(self.win + self.lose + self.tie)
    
    def save(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filename):
        with open(filename, 'rb') as f:
            return pickle.load(f)