import time
import numpy as np
import chess
from mcts.mcts import MCTS
from chess_env.chess_game import ChessGame
from utils.config import SelfPlayParams
import random

def self_play(nnet, buffer):
    """
    Run a batch of self-play games, store (state_tensor, pi, z, t_rem, is_white) in buffer,
    and print per-game and aggregate stats.
    Returns (white_wins, black_wins, draws).
    """
    game = ChessGame()
    cfg = SelfPlayParams()
    # Ensure action_size for MCTS tree reuse
    cfg.action_size = len(game.move_to_index)

    white_wins = 0
    black_wins = 0
    draws = 0

    for g in range(cfg.num_selfplay_games):
        board = game.reset()
        # reuse one MCTS per game to preserve search tree
        mcts = MCTS(game, nnet, cfg)
        data = []  # entries: (state_tensor, pi, flipped, t_rem, is_white)
        z = 0

        for t in range(getattr(cfg, 'max_game_length', 80)):
            valid_moves = game.get_valid_moves(board)
            pi = np.zeros_like(valid_moves, dtype=np.float32)

            # Force 1.d4 on White's first move
            if board.turn == chess.WHITE and t == 0:
                move = chess.Move.from_uci("d2d4")
                action = game.move_to_index.get(move)
                if action is None or valid_moves[action] == 0:
                    print(f"[Self-Play] Game {g}: forced d2d4 illegal, skipping")
                    break
                pi[action] = 1.0
            else:
                # get MCTS policy with reuse
                temp = cfg.exploration_temp if t < getattr(cfg, 'temperature_cutoff', t+1) else 0
                # encourage exploration for White
                if board.turn == chess.WHITE and hasattr(cfg, 'white_temp_min'):
                    temp = max(temp, cfg.white_temp_min)
                pi = mcts.get_action_probs(board, temp)
                # normalize
                s = pi.sum()
                if s <= 0 or np.isnan(s):
                    idxs = np.nonzero(valid_moves)[0]
                    if len(idxs):
                        pi = np.zeros_like(pi)
                        pi[idxs] = 1.0 / len(idxs)
                    else:
                        pi = np.ones_like(pi) / pi.size
                else:
                    pi = pi / s
                action = np.random.choice(len(pi), p=pi)

            # record state with perspective flip for Black
            flipped = (board.turn == chess.BLACK)
            flipped_board = board.mirror() if flipped else board
            state_tensor = game.encode_board(flipped_board)
            is_white = not flipped
            data.append((state_tensor, pi, flipped, 1, is_white))

            # play and update MCTS root
            board = game.get_next_state(board, action)
            mcts.update_root(action)

            # check terminal
            z = game.get_game_ended(board)
            if z != 0:
                winner = "White" if z == 1 else "Black" if z == -1 else "Draw"
                print(f"[Self-Play] Game {g} ended: {winner} (z={z}) in {t+1} plies")
                if z == 1:
                    white_wins += 1
                elif z == -1:
                    black_wins += 1
                else:
                    draws += 1
                print(f"[Self-Play] Cumulative → W:{white_wins}, B:{black_wins}, D:{draws}")
                break

        if z == 0:
            print(f"[Self-Play] Game {g} reached max plies. Declaring draw.")
            draws += 1
            print(f"[Self-Play] Cumulative → W:{white_wins}, B:{black_wins}, D:{draws}")

        # only include games where White did not lose
        if z >= 0:
            samples = []
            for i, (s_tensor, p, flipped, _, is_white) in enumerate(data):
                t_rem = len(data) - i
                # flip reward back if stored from Black's view
                z_final = -z if flipped else z
                samples.append((s_tensor, p, z_final, t_rem, is_white))
            # balance white/black samples
            white_s = [s for s in samples if s[4]]
            black_s = [s for s in samples if not s[4]]
            if white_s and black_s and len(white_s) != len(black_s):
                if len(white_s) < len(black_s):
                    white_s *= (len(black_s)//len(white_s)+1); white_s = white_s[:len(black_s)]
                else:
                    black_s *= (len(white_s)//len(black_s)+1); black_s = black_s[:len(white_s)]
            balanced = white_s + black_s
            random.shuffle(balanced)
            buffer.add(balanced)
        else:
            print(f"[Self-Play] Game {g} skipped (Black win)")

    print(f"[Self-Play Summary] White: {white_wins}, Black: {black_wins}, Draws: {draws}")
    return white_wins, black_wins, draws
