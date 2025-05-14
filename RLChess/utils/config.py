# chess_alpha/utils/config.py
import torch
import platform

class SelfPlayParams:
    def __init__(self):
        self.num_iters = 100
        self.num_selfplay_games = 100
        if platform.system() == "Windows":
            self.num_mcts_sims = 100
        elif platform.system() == "Darwin":
            self.num_mcts_sims = 100
        else:
            self.num_mcts_sims = 200  # Default or for Linux

        self.exploration_temp = 2.0
        self.cpuct = 1.5
        self.temperature_cutoff = 20
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.max_game_length = 250
        self.add_dirichlet_noise = True
        self.dirichlet_alpha = 0.3
        self.dirichlet_epsilon = 0.3
        self.opponent_pool_size = 5
        self.opponent_selection_prob = [0.4, 0.3, 0.2, 0.08, 0.02]
        self.freeze_opponents = True
        self.entropy_coeff = 0.1  # Increased entropy coefficient for more exploration

class LearningParams:
    def __init__(self):
        self.use_symmetries = True
        self.mem_buffer_size = 30000          # How many past games to remember
        self.batch_size = 300                 # Training batch size
        self.loss_comp_batch_size = 300       # Optional: smaller for memory
        self.num_checkpoints = 10
        self.max_batches_per_checkpoint = 1000
        self.min_checkpoints_per_epoch = 5
        self.nonvalidity_penalty = 2.0
        self.l2_regularization = 1e-4
        self.learning_rate = 5e-4             # Reduced learning rate for more stable training
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.entropy_coeff = 0.05             # Increased entropy coefficient
        # New parameters
        self.annealing_factor = 0.95          # Learning rate annealing factor
        self.min_learning_rate = 1e-5         # Minimum learning rate
        self.temperature = 1.0                # Temperature for policy sampling

class ArenaParams:
    def __init__(self):
        self.num_games = 100                  # Arena games for model comparison
        self.replace_threshold = 0.55         # % new model must win to replace old
        self.num_mcts_sims = 100              # Match self-play value
        self.cpuct = 1.5                      # Match self-play value
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.temperature = 0.5                # Add temperature for evaluation
