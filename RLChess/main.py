# chess_alpha/mcts/main.py

import math
import os
import numpy as np
import time
import torch
import matplotlib.pyplot as plt
import multiprocessing as mp
import platform

from model.model import ChessNet
from training.optimiser import get_optimizer
from training.train import train
from utils.utils import ReplayBuffer
from utils.config import SelfPlayParams, LearningParams, ArenaParams
from selfplay.selfplay import self_play
from selfplay.arena import evaluate_new_model
from chess_env.chess_game import ChessGame
from selfplay.parallel_selfplay import parallel_self_play
from selfplay.parallel_arena import parallel_arena
if __name__ == "__main__":
    mp.set_start_method('spawn')
    
    torch.backends.cudnn.benchmark = True
    if os.system == "Windows" and torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(0.5)
    elif os.system == "Darwin":  # macOS
        print("CUDA not supported on macOS. Skipping CUDA memory config.")

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

    # 4) Try loading existing global best model
    best_model = ChessNet().to(device)
    best_model_path = "./checkpoints/best_model.pt"
    if os.path.exists(best_model_path):
        print(f"[Startup] Loading existing best model from {best_model_path}")
        best_model.load_state_dict(torch.load(best_model_path, map_location=device))
        use_existing = True
    else:
        print("[Startup] No best model found, starting from scratch.")
        best_model = None
        use_existing = False

    win_rates = []

    # 5) Create a unique folder for this run's iteration checkpoints
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    run_ckpt_dir = os.path.join(
        "./checkpoints", f"run_{timestamp}")
    os.makedirs(run_ckpt_dir, exist_ok=True)
    print(f"[Init] Saving iteration checkpoints to: {run_ckpt_dir}")

    # 6) Main AlphaZero loop
    prev_loss_avg = float('inf')

    for iteration in range(sp_cfg.num_iters):
        iter_start = time.time()
        print(f"\n[Iteration {iteration+1}/{sp_cfg.num_iters}] Starting at {time.strftime('%Y-%m-%d %H:%M:%S')}")

        # — Self-play phase
        sp_start = time.time()
        #games_played = self_play(nnet, replay_buffer)
        games_played = parallel_self_play(nnet, replay_buffer)
        sp_duration = time.time() - sp_start
        print(f"[Iteration {iteration+1}] Self-play: {games_played} games, buffer size={len(replay_buffer)} (took {sp_duration:.1f}s)")

        # — Training phase
        tr_start = time.time()
        loss_avg = train(nnet, replay_buffer, optimizer, lr_cfg)
        tr_duration = time.time() - tr_start
        print(f"[Iteration {iteration+1}] Training: avg loss={loss_avg:.4f}, buffer size={len(replay_buffer)} (took {tr_duration:.1f}s)")

        # — Evaluation phase
        if iteration == 0 and best_model is None:
            # first iteration, no prior best
            best_model = ChessNet().to(device)
            best_model.load_state_dict(nnet.state_dict())
            win_rates.append(0.0)
            prev_loss_avg = loss_avg
            print("[Iteration 1] No arena run — using trained model as initial best model.")
        else:
            # evaluate new candidate vs best
            model = ChessNet().to(device)
            model.load_state_dict(nnet.state_dict())
            # game = ChessGame()
            # (
            #     new_wins, best_wins, draws,
            #     tiebreak_new_better,
            #     total_new_cp_loss, total_best_cp_loss,
            #     top_match_counts
            # ) = evaluate_new_model(game, model, best_model, arena_cfg)
            game = ChessGame()
            (
                new_wins, best_wins, draws,
                tiebreak_new_better,
                total_new_cp_loss, total_best_cp_loss,
                top_match_counts
            ) = parallel_arena(game, model, best_model, arena_cfg)

            winrate = new_wins / max((new_wins + best_wins), 1)
            win_rates.append(winrate)
            print(f"[Iteration {iteration+1}] Arena win rate: {winrate:.2%}")

            # replacement criterion: better wins or tie-break by lower cp-loss
            if (new_wins > best_wins) or (new_wins == best_wins and total_new_cp_loss < total_best_cp_loss):
                best_model = ChessNet().to(device)
                best_model.load_state_dict(nnet.state_dict())
                print(f"[Iteration {iteration+1}] New model accepted as best (win/tie-break).")
            else:
                print(f"[Iteration {iteration+1}] Best model retained.")

            prev_loss_avg = loss_avg

        # — Save iteration checkpoint
        ckpt_path = os.path.join(run_ckpt_dir, f"model_iter{iteration+1}.pt")
        torch.save(nnet.state_dict(), ckpt_path)
        iter_duration = time.time() - iter_start
        print(f"[Iteration {iteration+1}] Checkpoint saved to {ckpt_path} (iteration took {iter_duration:.1f}s)")

        # — Save/overwrite global best model file if we have one
        if best_model is not None:
            torch.save(best_model.state_dict(), best_model_path)
            print(f"[Iteration {iteration+1}] Global best model saved to: {best_model_path}")

    # 7) Plot win rate chart
    chart_path = os.path.join(run_ckpt_dir, "arena_winrates.png")
    plt.plot(range(1, len(win_rates)+1), win_rates)
    plt.xlabel("Iteration")
    plt.ylabel("Arena Win Rate")
    plt.title("Model Improvement Over Time")
    plt.grid(True)
    plt.savefig(chart_path)
    print(f"Arena win rate chart saved to '{chart_path}'")
