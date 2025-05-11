import random
import numpy as np

def apply_symmetries(states, pis, vs):
    # placeholder for board symmetries
    return states, pis, vs

class ReplayBuffer:
    def __init__(self, size):
        self.size = size
        self.white_wins = []
        self.black_wins = []
        self.draws = []

    def add(self, samples):
        # Assume all samples in the list come from a single game
        if not samples:
            return
        outcome = samples[0][2]  # z value: +1 = white win, -1 = black win, 0 = draw
        if outcome == 1:
            self.white_wins.extend(samples)
        elif outcome == -1:
            self.black_wins.extend(samples)
        else:
            self.draws.extend(samples)

        # Truncate all lists if total size exceeds buffer size
        total = len(self)
        if total > self.size:
            # Proportionally trim each
            over = total - self.size
            self._trim_buffer(over)

    def _trim_buffer(self, over):
        for buf in [self.white_wins, self.black_wins, self.draws]:
            if len(buf) > 0:
                trim = min(over, len(buf))
                del buf[:trim]
                over -= trim
                if over <= 0:
                    break

    def sample_balanced(self, batch_size):
        # Step 1: Get white win steps (up to 1/3 of batch)
        target_per_class = batch_size // 3
        white_steps = random.sample(self.white_wins, min(target_per_class, len(self.white_wins)))

        # Step 2: Match that number of black win steps
        black_steps = random.sample(self.black_wins, min(len(white_steps), len(self.black_wins)))

        # Step 3: Fill the rest with draw steps
        remaining = batch_size - len(white_steps) - len(black_steps)
        draw_steps = random.sample(self.draws, min(remaining, len(self.draws)))

        return white_steps + black_steps + draw_steps

    def __len__(self):
        return len(self.white_wins) + len(self.black_wins) + len(self.draws)
