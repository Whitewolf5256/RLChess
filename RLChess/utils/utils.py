import random
import numpy as np

def apply_symmetries(states, pis, vs):
    return states, pis, vs  # Placeholder

class ReplayBuffer:
    def __init__(self, size):
        self.size = size
        self.max_per_category = size // 3
        self.white_wins = []
        self.black_wins = []
        self.draws = []

    def add(self, samples):
        if not samples:
            return
        outcome = samples[0][2]  # z value: +1 = white win, -1 = black win, 0 = draw

        if outcome == 1:
            self._add_category(self.white_wins, samples)
        elif outcome == -1:
            self._add_category(self.black_wins, samples)
        else:
            self._add_category(self.draws, samples)

    def _add_category(self, buffer_list, samples):
        buffer_list.extend(samples)
        # FIFO: remove oldest if overflow
        while len(buffer_list) > self.max_per_category:
            del buffer_list[0:(len(buffer_list) - self.max_per_category)]

    def sample_balanced(self, batch_size):
        per_class = batch_size // 3

        # Determine how much we *can* sample per class
        actual_per_class = min(len(self.white_wins), len(self.black_wins), len(self.draws), per_class)

        white = random.sample(self.white_wins, actual_per_class)
        black = random.sample(self.black_wins, actual_per_class)
        draw = random.sample(self.draws, actual_per_class)

        print(f"[Buffer Sampling] white: {len(white)}, black: {len(black)}, draws: {len(draw)}")
        return white + black + draw

    def __len__(self):
        return len(self.white_wins) + len(self.black_wins) + len(self.draws)
