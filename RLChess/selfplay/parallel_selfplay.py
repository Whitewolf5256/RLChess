import multiprocessing as mp
import numpy as np
import chess
from mcts.mcts import MCTS
from chess_env.chess_game import ChessGame
from utils.config import SelfPlayParams
from utils.logging import log_self_play_results
import os
import platform

def run_self_play_game(nnet, cfg, game_num):
    game = ChessGame()
    board = game.reset()
    mcts = MCTS(game, nnet, cfg)
    mcts.root_state = board
    data = []

    for t in range(cfg.max_game_length):
        valid = game.get_valid_moves(board)
        pi = np.zeros_like(valid, dtype=np.float32)

        if board.turn == chess.WHITE and t == 0:
            move = board.parse_uci("d2d4")
            action = game.move_to_index.get(move)
            if action is None or valid[action] == 0:
                print(f"[WARN] Invalid forced opening in game {game_num}. Drawing game.")
                return [], 0, 0, 1
            pi[action] = 1.0
        else:
            temp = cfg.exploration_temp
            if board.turn == chess.WHITE:
                temp *= 1.5
            if t >= cfg.temperature_cutoff:
                temp = 0

            pi = mcts.get_action_probs(board, temp)

            s = pi.sum()
            if s <= 0 or np.isnan(s):
                print(f"[WARN] Bad pi vector at step {t} in game {game_num}, fixing...")
                idxs = np.nonzero(valid)[0]
                if len(idxs):
                    pi = np.zeros_like(pi)
                    pi[idxs] = 1.0 / len(idxs)
                else:
                    pi = np.ones_like(pi) / pi.size
            else:
                pi = pi / s

            action = np.random.choice(len(pi), p=pi)

        flipped = (board.turn == chess.BLACK)
        state = board.mirror() if flipped else board
        state_tensor = game.encode_board(state)
        is_white = not flipped
        data.append((state_tensor, pi, flipped, is_white))

        board = game.get_next_state(board, action)
        mcts.update_root(action)

        z = game.get_game_ended(board)
        if z != 0:
            print(f"[INFO] Game {game_num} ended after {t+1} moves. Result: {z}")
            break

    if z == 0:
        print(f"[INFO] Game {game_num} ended by tie.")
        z = 0

    samples = []
    for i, (s_tensor, p, flipped, is_white) in enumerate(data):
        t_rem = len(data) - i
        z_final = -z if flipped else z
        samples.append((s_tensor, p, z_final, t_rem, is_white))

    return samples, int(z == 1), int(z == -1), int(z == 0)

def _worker(args):
    nnet, cfg, game_num = args
    return run_self_play_game(nnet, cfg, game_num)


def parallel_self_play(nnet, buffer):
    """
    Launch parallel self-play using multiprocessing.
    Saves (state_tensor, pi, z, t_rem, is_white) to buffer.
    Returns (white_win_count, black_win_count, draw_count)
    """
    cfg = SelfPlayParams()
    num_games = cfg.num_selfplay_games

    # Detect the operating system (macOS or Windows)
    system = platform.system()

    if system == "Windows":
        num_cpus = os.cpu_count()  # Windows should be fine with this as it uses fork
    elif system == "Darwin":  # macOS
        num_cpus = os.cpu_count()  # This should be 8 for MacBook M1, but we can set a custom value if needed.
    else:
        num_cpus = os.cpu_count()

    # Adjust the number of processes (ensure it's an integer)
    num_cpus = int(min(num_cpus / 5, num_games))  # Ensure num_cpus is an integer

    print(f"[INFO] Using {num_cpus} CPUs for parallel self-play.")

    white_wins, black_wins, draws = 0, 0, 0
    total_games_played = 0

    def play_batch(batch_size):
        nonlocal white_wins, black_wins, draws, total_games_played
        print(f"[INFO] Launching {batch_size} self-play games using {num_cpus} processes...")
        with mp.Pool(processes=num_cpus) as pool:
            results = pool.map(_worker, [(nnet, cfg, i) for i in range(batch_size)])

        # Unpacking 4 values: (samples, white_wins, black_wins, draws)
        for samples, white_w, black_w, draw in results:
            if samples:
                buffer.add(samples)
            white_wins += white_w
            black_wins += black_w
            draws += draw
        total_games_played += batch_size

    # Run initial batch of self-play
    play_batch(num_games)

    # Keep playing until all outcome types are present
    while white_wins == 0 or black_wins == 0 or draws == 0:
        play_batch(num_cpus)  # play smaller batches (1 per CPU)

    print(f"[Self-Play Complete] Total Games: {total_games_played} | White Wins: {white_wins}, Black Wins: {black_wins}, Draws: {draws}")
    log_self_play_results(white_wins, black_wins, draws, filename="logs/self_play/self_play_results.csv")
    return white_wins, black_wins, draws