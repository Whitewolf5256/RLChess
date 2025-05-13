# chess_alpha/utils/config.py
import torch

class SelfPlayParams:
    def __init__(self):
        self.num_iters = 100                  # Total training iterations
        self.num_selfplay_games = 100          # Self-play games per iteration
        self.num_mcts_sims = 200              # MCTS sims per move
        self.exploration_temp = 2.0           # Temperature for move selection
        self.cpuct = 1.5                      # Exploration-exploitation balance
        self.temperature_cutoff = 20
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.max_game_length = 80
        self.add_dirichlet_noise = True
        self.dirichlet_alpha = 0.3      # AlphaZero used 0.3 for chess
        self.dirichlet_epsilon = 0.3 
        self.opponent_pool_size = 5
        self.opponent_selection_prob = [0.4, 0.3, 0.2, 0.08, 0.02]
        self.freeze_opponents = True
        self.entropy_coeff = 0.01  # New entropy coefficient

class LearningParams:
    def __init__(self):
        self.use_symmetries = True
        self.mem_buffer_size = 30000          # How many past games to remember
        self.batch_size = 256                 # Training batch size
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
        self.num_games = 100                   # Arena games for model comparison
        self.replace_threshold = 0.55         # % new model must win to replace old
        self.num_mcts_sims = 100              # Match self-play value
        self.cpuct = 1.5                   # Match self-play value
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
