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
        while len(buffer_list) > self.max_per_category:
            excess = len(buffer_list) - self.max_per_category
            del buffer_list[:excess]

    def sample_balanced(self, batch_size):
        per_class = batch_size // 3
        white = random.sample(self.white_wins, min(per_class, len(self.white_wins)))
        black = random.sample(self.black_wins, min(per_class, len(self.black_wins)))
        draw_needed = batch_size - len(white) - len(black)
        draw = random.sample(self.draws, min(draw_needed, len(self.draws)))

        print(f"[Buffer Sampling] white: {len(self.white_wins)}, black: {len(self.black_wins)}, draws: {len(self.draws)}")

        return white + black + draw

    def __len__(self):
        return len(self.white_wins) + len(self.black_wins) + len(self.draws)
