# chess_alpha/utils/config.py
import torch

class SelfPlayParams:
    def __init__(self):
        self.num_iters = 500                  # Total training iterations
        self.num_selfplay_games = 80          # Self-play games per iteration
        self.num_mcts_sims = 100              # MCTS sims per move
        self.exploration_temp = 1.5           # Temperature for move selection
        self.cpuct = 1.5                      # Exploration-exploitation balance
        self.temperature_cutoff = 20
        self.device = "cuda" if torch.cuda.is_available() else "cpu"


class LearningParams:
    def __init__(self):
        self.use_symmetries = True
        self.mem_buffer_size = 20000          # How many past games to remember
        self.batch_size = 128                 # Training batch size
        self.loss_comp_batch_size = 256       # Optional: smaller for memory
        self.num_checkpoints = 10
        self.max_batches_per_checkpoint = 1000
        self.min_checkpoints_per_epoch = 5
        self.nonvalidity_penalty = 2.0
        self.l2_regularization = 1e-4
        self.learning_rate = 1e-3             # Slightly more stable learning rate
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'


class ArenaParams:
    def __init__(self):
        self.num_games = 80                   # Arena games for model comparison
        self.replace_threshold = 0.55         # % new model must win to replace old
        self.num_mcts_sims = 100              # Match self-play value
        self.cpuct = 1.5                      # Match self-play value
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
