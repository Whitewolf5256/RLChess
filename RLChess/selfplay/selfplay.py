import time
import numpy as np
import chess
from mcts.mcts import MCTS
from chess_env.chess_game import ChessGame
from utils.config import SelfPlayParams

def self_play(nnet, buffer):
    """
    Run a batch of self-play games, store (state_tensor, pi, z, t_rem) in buffer,
    and print per-game and aggregate stats.
    Returns (white_wins, black_wins, draws).
    """
    game = ChessGame()
    cfg = SelfPlayParams()

    white_wins = 0
    black_wins = 0
    draws = 0

    for g in range(cfg.num_selfplay_games):
        board = game.reset()
        data = []
        move_history = []

        start = time.time()
        z = 0  # Initialize game result to "not ended"

        for t in range(500):
            valid_moves = game.get_valid_moves(board)
            pi = np.zeros_like(valid_moves, dtype=np.float32)

            # Force 1.d4 on White’s first move
            if board.turn == chess.WHITE and t == 0:
                move = chess.Move.from_uci("d2d4")
                action = game.move_to_index.get(move)
                if action is None or valid_moves[action] == 0:
                    print(f"[Self-Play] Game {g}: forced d2d4 illegal, skipping")
                    break
                pi[action] = 1.0
            else:
                # MCTS-based policy
                mcts = MCTS(game, nnet, cfg)
                pi = mcts.get_action_probs(board, cfg.exploration_temp)

                # Re-normalize to exactly 1
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

            # Save current state as tensor
            state_tensor = game.encode_board(board)  # shape: (1, C, 8, 8)
            data.append((state_tensor, pi, None, 1))

            # # Get move before playing it
            # move = game.index_to_move.get(action)
            # if move and move in board.legal_moves:
            #     move_san = board.san(move)
            # else:
            #     move_san = "???"
            # move_history.append(move_san)
            # print(f"[Game {g}] Ply {t+1}: {'White' if board.turn else 'Black'} plays {move_san}")

            # Play move
            board = game.get_next_state(board, action)

            # # Print board after Black's move (every second ply)
            # if board.turn == chess.WHITE:
            #     print(board)
            #     print("-" * 40)

            # Check game end
            z = game.get_game_ended(board)
            if z != 0:
                winner = "White" if z == 1 else "Black" if z == -1 else "Draw"
                print(f"[Self-Play] Game {g} ended: {winner} (z={z}) in {t+1} plies")
                # print("Final board:")
                # print(board)
                # print("Move history:", " ".join(move_history))
                if z == 1:
                    white_wins += 1
                elif z == -1:
                    black_wins += 1
                else:
                    draws += 1

                print(f"[Self-Play] Cumulative Results → White: {white_wins}, Black: {black_wins}, Draws: {draws}")
                break

        # If game hit 500 plies without ending
        if z == 0:
            print(f"[Self-Play] Game {g} reached 500 plies. Declaring draw.")
            z = 0
            draws += 1
            # print("Final board:")
            # print(board)
            # print("Move history:", " ".join(move_history))
            print(f"[Self-Play] Cumulative Results → White: {white_wins}, Black: {black_wins}, Draws: {draws}")

        # Convert to training samples
        samples = []
        for i, (s_tensor, p, _, _) in enumerate(data):
            t_rem = len(data) - i
            samples.append((s_tensor, p, z, t_rem))

        buffer.add(samples)
        # print(f"[Self-Play] Added {len(samples)} samples from game {g} (took {time.time()-start:.1f}s)")

    # Batch summary
    print(f"[Self-Play Summary] White: {white_wins}, Black: {black_wins}, Draws: {draws}")
    return white_wins, black_wins, draws
