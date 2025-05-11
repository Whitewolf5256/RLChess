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
        # Step 1: Pick as many white wins as available, up to batch_size
        white_count = min(len(self.white_wins), batch_size)
        white = random.sample(self.white_wins, white_count)

        # Step 2: Match that many black wins
        black_count = min(len(self.black_wins), white_count)
        black = random.sample(self.black_wins, black_count)

        # Step 3: Fill the rest with draws
        remaining = batch_size - len(white) - len(black)
        remaining = max(batch_size - len(white) - len(black), 0)
        draw = random.sample(self.draws, min(remaining, len(self.draws)))
        return white + black + draw


    def __len__(self):
        return len(self.white_wins) + len(self.black_wins) + len(self.draws)
