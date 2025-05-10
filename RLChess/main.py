# chess_alpha/mcts/main.py

import math
import os
import numpy as np
import time
import torch
import matplotlib.pyplot as plt
from model.model import ChessNet
from training.optimiser import get_optimizer
from training.train import train
from utils.utils import ReplayBuffer
from utils.config import SelfPlayParams, LearningParams, ArenaParams
from selfplay.selfplay import self_play
from selfplay.arena import evaluate_new_model
from chess_env.chess_game import ChessGame

if __name__ == "__main__":
    # 1) Load configs
    sp_cfg = SelfPlayParams()
    lr_cfg = LearningParams()
    arena_cfg = ArenaParams()

    # 2) Build network & optimizer
    device = sp_cfg.device
    nnet = ChessNet().to(device)
    optimizer = get_optimizer(nnet)

    # 3) Create replay buffer
    replay_buffer = ReplayBuffer(lr_cfg.mem_buffer_size)

    # 4) No best_model until after first training
    best_model = None
    win_rates = []

    # 5) Main AlphaZero loop
    for iteration in range(sp_cfg.num_iters):
        iter_start = time.time()
        print(f"\n[Iteration {iteration+1}/{sp_cfg.num_iters}] Starting at {time.strftime('%Y-%m-%d %H:%M:%S')}")

        # — Self-play phase
        sp_start = time.time()
        games_played = self_play(nnet, replay_buffer)
        sp_duration = time.time() - sp_start
        print(f"[Iteration {iteration+1}] Self-play: {games_played} games, buffer size={len(replay_buffer)} (took {sp_duration:.1f}s)")

        # — Training phase
        tr_start = time.time()
        loss_avg = train(nnet, replay_buffer, optimizer, lr_cfg)
        tr_duration = time.time() - tr_start
        print(f"[Iteration {iteration+1}] Training: avg loss={loss_avg:.4f}, buffer size={len(replay_buffer)} (took {tr_duration:.1f}s)")

        # — Evaluation phase (skip Arena on first iteration)
        if iteration == 0:
            # Save trained model as best model
            best_model = ChessNet().to(device)
            best_model.load_state_dict(nnet.state_dict())
            win_rates.append(0.0)
            print("[Iteration 1] No arena run — using trained model as initial best model.")
        else:
            model = ChessNet().to(device)
            model.load_state_dict(nnet.state_dict())  # Load state of the trained model into the new instance

            # Evaluate against best model
            game = ChessGame()
            new_wins, best_wins = evaluate_new_model(game, best_model, best_model, arena_cfg)  # Pass `model` instead of `nnet`
            winrate = new_wins / max((new_wins + best_wins), 1)
            win_rates.append(winrate)
            print(f"[Iteration {iteration+1}] Arena win rate: {winrate:.2%}")

            # Replace best model if performance exceeds threshold
            if winrate > arena_cfg.replace_threshold:
                best_model.load_state_dict(nnet.state_dict())
                print(f"[Iteration {iteration+1}] New model accepted as best model.")

        # — Save checkpoint
        ckpt_dir = f"C:/Users/timcw/source/repos/RLChess/RLChess/checkpoints/"
        os.makedirs(ckpt_dir, exist_ok=True)  # Create the directory if it doesn't exist
        ckpt_path = os.path.join(ckpt_dir, f"model_iter{iteration+1}.pt")

        torch.save(nnet.state_dict(), ckpt_path)
        iter_duration = time.time() - iter_start
        print(f"[Iteration {iteration+1}] Checkpoint saved to {ckpt_path} (iteration took {iter_duration:.1f}s)")

    # 6) Plot win rate chart
    plt.plot(range(1, len(win_rates)+1), win_rates)
    plt.xlabel("Iteration")
    plt.ylabel("Arena Win Rate")
    plt.title("Model Improvement Over Time")
    plt.grid(True)
    plt.savefig("arena_winrates.png")
    print("Arena win rate chart saved to 'arena_winrates.png'")
