# chess_alpha/selfplay/parallel_selfplay.py
import torch
import numpy as np
import multiprocessing as mp
from chess_env.chess_game import ChessGame
from mcts.mcts import MCTS
from utils.config import SelfPlayParams
import chess

def self_play_worker(worker_id, nnet_weights, cfg, result_queue, device):
    torch.cuda.set_device(device)
    game = ChessGame()
    nnet = cfg.model_class().to(device)
    nnet.load_state_dict(nnet_weights)
    nnet.eval()

    samples = []
    white_wins, black_wins, draws = 0, 0, 0

    for g in range(cfg.games_per_worker):
        board = game.reset()
        data = []
        z = 0

        for t in range(500):
            valid_moves = game.get_valid_moves(board)
            pi = np.zeros_like(valid_moves, dtype=np.float32)

            if board.turn == chess.WHITE and t == 0:
                move = chess.Move.from_uci("d2d4")
                action = game.move_to_index.get(move)
                if action is None or valid_moves[action] == 0:
                    break
                pi[action] = 1.0
            else:
                mcts = MCTS(game, nnet, cfg)
                pi = mcts.get_action_probs(board, cfg.exploration_temp)
                pi = pi / np.sum(pi) if np.sum(pi) > 0 else np.ones_like(pi) / len(pi)

                action = np.random.choice(len(pi), p=pi)

            state_tensor = game.encode_board(board)
            data.append((state_tensor, pi, None, 1))
            board = game.get_next_state(board, action)

            z = game.get_game_ended(board)
            if z != 0:
                if z == 1:
                    white_wins += 1
                elif z == -1:
                    black_wins += 1
                else:
                    draws += 1
                break

        if z == 0:
            draws += 1
            z = 0

        for i, (s_tensor, p, _, _) in enumerate(data):
            t_rem = len(data) - i
            samples.append((s_tensor, p, z, t_rem))

    result_queue.put((samples, white_wins, black_wins, draws))

def parallel_self_play(nnet, buffer, device):
    cfg = SelfPlayParams()
    num_procs = cfg.num_parallel_games
    games_per_worker = cfg.num_selfplay_games // num_procs

    ctx = mp.get_context("spawn")
    result_queue = ctx.Queue()

    nnet_weights = nnet.state_dict()
    nnet_shareable = {k: v.clone().detach().cpu() for k, v in nnet_weights.items()}

    procs = []
    for i in range(num_procs):
        p = ctx.Process(
            target=self_play_worker,
            args=(i, nnet_shareable, cfg, result_queue, device.index if device.type == "cuda" else None)
        )
        p.start()
        procs.append(p)

    all_samples = []
    total_white, total_black, total_draws = 0, 0, 0
    for _ in range(num_procs):
        samples, white, black, draws = result_queue.get()
        all_samples.extend(samples)
        total_white += white
        total_black += black
        total_draws += draws

    for p in procs:
        p.join()

    buffer.add(all_samples)
    print(f"[Parallel Self-Play] Completed. White: {total_white}, Black: {total_black}, Draws: {total_draws}")
