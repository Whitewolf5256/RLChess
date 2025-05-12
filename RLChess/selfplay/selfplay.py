import time
import numpy as np
import chess
import torch
from mcts.mcts import MCTS
from chess_env.chess_game import ChessGame
from utils.config import SelfPlayParams
from utils.logging import log_self_play_results

def self_play(nnet, buffer):
    """
    Run a batch of self-play games, store (state_tensor, pi, z, t_rem, is_white) in buffer,
    and print per-game and aggregate stats.
    Returns (white_wins, black_wins, draws).
    """
    game = ChessGame()
    cfg = SelfPlayParams()

    device = cfg.device
    white_wins, black_wins, draws = 0, 0, 0
    total_games = 0

    def outcomes_met():
        return white_wins > 0 and black_wins > 0 and draws > 0

    while total_games < cfg.num_selfplay_games or not outcomes_met():
    # for g in range(cfg.num_selfplay_games):
        board = game.reset()
        mcts = MCTS(game, nnet, cfg)
        mcts.root_state = board
        data = []  # (state_tensor, pi, flipped, is_white)
        z = 0

        for t in range(getattr(cfg, 'max_game_length', 80)):
            valid = game.get_valid_moves(board)
            pi = np.zeros_like(valid, dtype=np.float32)

            # ✅ Force 1.d4 on first move
            if board.turn == chess.WHITE and t == 0:
                move = chess.Move.from_uci("d2d4")
                action = game.move_to_index.get(move)
                if action is None or valid[action] == 0:
                    print(f"[Self-Play] Game {total_games}: forced d2d4 illegal, skipping")
                    break
                pi[action] = 1.0
            else:
                temp = cfg.exploration_temp
                if board.turn == chess.WHITE:
                    temp *= 1.5
                elif t >= cfg.temperature_cutoff:
                    temp = 0

                pi = mcts.get_action_probs(board, temp)
                s = pi.sum()
                if s <= 0 or np.isnan(s):
                    idxs = np.nonzero(valid)[0]
                    if len(idxs):
                        pi = np.zeros_like(pi)
                        pi[idxs] = 1.0 / len(idxs)
                    else:
                        pi = np.ones_like(pi) / pi.size
                else:
                    pi = pi / s
                action = np.random.choice(len(pi), p=pi)

            # ✅ Encode board state and move it to the correct side and device
            flipped = (board.turn == chess.BLACK)
            state = board.mirror() if flipped else board
            state_tensor = torch.tensor(
                game.encode_board(state),
                dtype=torch.float32,
                device=device
            )
            is_white = not flipped

            data.append((state_tensor, pi, flipped, is_white))

            board = game.get_next_state(board, action)
            mcts.update_root(action)

            z = game.get_game_ended(board)
            if z != 0:
                if z == 1:
                    white_wins += 1
                elif z == -1:
                    black_wins += 1
                else:
                    draws += 1
                print(f"[Self-Play] Game {total_games} ended: {'White' if z==1 else 'Black' if z==-1 else 'Draw'} (z={z}) in {t+1} plies")
                print(f"[Self-Play] Cumulative → White:{white_wins}, Black:{black_wins}, Draw:{draws}")
                break

        if z == 0:
            draws += 1
            print(f"[Self-Play] Game {total_games} reached max plies. Declaring draw.")
            print(f"[Self-Play] Cumulative → White:{white_wins}, Black:{black_wins}, Draw:{draws}")

        # ✅ Add both sides of data with correct reward + time-to-end
        samples = []
        for i, (s_tensor, p, flipped, is_white) in enumerate(data):
            t_rem = len(data) - i
            z_final = -z if flipped else z
            samples.append((s_tensor, torch.tensor(p, dtype=torch.float32, device=device), z_final, t_rem, is_white))

        buffer.add(samples)
        total_games += 1

    print(f"[Self-Play Summary] White: {white_wins}, Black: {black_wins}, Draws: {draws}")
    log_self_play_results(white_wins, black_wins, draws)
    return white_wins, black_wins, draws