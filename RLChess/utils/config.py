# chess_alpha/utils/config.py
import torch

class SelfPlayParams:
    def __init__(self):
        # how many full AlphaZero iterations (self‑play + training)
        self.num_iters = 1000
        # self‑play games per iteration 250
        self.num_selfplay_games = 1
        # MCTS simulations per move 400
        self.num_mcts_sims = 4
        # exploration temperature
        self.exploration_temp = 1.0
        self.cpuct = 1.0
        # GPU vs CPU
        import torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

class LearningParams:
    def __init__(self):
        self.use_symmetries = True
        self.mem_buffer_size = 200000
        self.batch_size = 512
        self.loss_comp_batch_size = 1024
        self.num_checkpoints = 10
        self.max_batches_per_checkpoint = 2000
        self.min_checkpoints_per_epoch = 5
        self.nonvalidity_penalty = 2.0
        self.l2_regularization = 1e-4
        self.learning_rate = 5e-4

        # ← Add this line so train.py can read cfg.device
        import torch
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'


class ArenaParams:
    def __init__(self):
        self.num_games = 100
        self.replace_threshold = 0.6
        self.num_mcts_sims = 400
        self.cpuct = 1.0
        import torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
