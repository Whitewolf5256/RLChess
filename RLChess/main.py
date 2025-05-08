# chess_alpha/mcts/mcts.py
import math
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
    sp_cfg = SelfPlayParams()      # self‑play + iteration params
    lr_cfg = LearningParams()      # learning params
    arena_cfg = ArenaParams()

    # 2) Build network & optimizer on the right device
    device = sp_cfg.device
    nnet = ChessNet().to(device)
    optimizer = get_optimizer(nnet)

    # 3) Create replay buffer
    replay_buffer = ReplayBuffer(lr_cfg.mem_buffer_size)

    # 4) Init best model as current
    best_model = ChessNet().to(device)
    best_model.load_state_dict(nnet.state_dict())

    win_rates = []

    # 5) Main AlphaZero loop
    for iteration in range(sp_cfg.num_iters):
        iter_start = time.time()
        print(f"[Iteration {iteration+1}/{sp_cfg.num_iters}] Starting self-play and training at {time.strftime('%Y-%m-%d %H:%M:%S')}")

        # — Self‑play phase: generate games & fill buffer
        sp_start = time.time()
        games_played = self_play(nnet, replay_buffer)
        sp_duration = time.time() - sp_start
        print(f"[Iteration {iteration+1}] Self-play: {games_played} games, buffer size={len(replay_buffer)} (took {sp_duration:.1f}s)")

        # — Learning phase: sample from buffer & update network
        tr_start = time.time()
        loss_avg = train(nnet, replay_buffer, optimizer, lr_cfg)
        tr_duration = time.time() - tr_start
        print(f"[Iteration {iteration+1}] Training: avg loss={loss_avg:.4f}, buffer size={len(replay_buffer)} (took {tr_duration:.1f}s)")

        # — Evaluation phase (Arena)
        game = ChessGame()
        new_wins, best_wins = evaluate_new_model(game, nnet, best_model, arena_cfg)
        winrate = new_wins / max((new_wins + best_wins), 1)
        win_rates.append(winrate)
        print(f"[Iteration {iteration+1}] Arena win rate: {winrate:.2%}")

        # — Replace best model if threshold met
        if winrate > arena_cfg.replace_threshold:
            best_model.load_state_dict(nnet.state_dict())
            print(f"[Iteration {iteration+1}] New model accepted as best model.")

        # — Checkpoint the network
        ckpt_path = f"checkpoints/model_iter{iteration+1}.pt"
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