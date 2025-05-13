import multiprocessing as mp
import numpy as np
import chess
from mcts.mcts import MCTS
from chess_env.chess_game import ChessGame
from utils.config import SelfPlayParams
from utils.logging import log_self_play_results
import os
import platform
import torch

def run_self_play_game(nnet, cfg, game_num, opponent_model=None):
    """
    If opponent_model is provided, alternate moves between nnet and opponent_model.
    Otherwise, use nnet for both sides (standard self-play).
    """
    game = ChessGame()
    board = game.reset()
    mcts_nnet = MCTS(game, nnet, cfg)
    if opponent_model is not None:
        mcts_opp = MCTS(game, opponent_model, cfg)
    else:
        mcts_opp = mcts_nnet
    mcts_nnet.root_state = board
    mcts_opp.root_state = board
    data = []

    for t in range(cfg.max_game_length):
        valid = game.get_valid_moves(board)
        pi = np.zeros_like(valid, dtype=np.float32)

        # Decide which model to use for this move
        if opponent_model is not None:
            # True for white, False for black
            model_to_use = nnet if board.turn else opponent_model
            mcts_to_use = mcts_nnet if board.turn else mcts_opp
        else:
            model_to_use = nnet
            mcts_to_use = mcts_nnet

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

            pi = mcts_to_use.get_action_probs(board, temp, selfplay=True)
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

        current_player = board.turn  # True for White, False for Black

        state_tensor = torch.tensor(
            game.encode_board(board),
            dtype=torch.float32,
            device=cfg.device
        )
        data.append((state_tensor, pi, current_player))

        # Make the move
        board = game.get_next_state(board, action)
        mcts_nnet.update_root(action)
        if opponent_model is not None:
            mcts_opp.update_root(action)

        z = game.get_game_ended(board)
        if z != 0:
            print(f"[INFO] Game {game_num} ended after {t+1} moves. Result: {z}")
            break

    if z == 0:
        print(f"[INFO] Game {game_num} ended by tie.")

    samples = []
    for i, (s_tensor, p, player) in enumerate(data):
        if z == 0:
            value = 0
        elif (player and z == 1) or (not player and z == -1):
            value = 1
        else:
            value = -1
        t_rem = len(data) - i
        samples.append((s_tensor, p, value, t_rem))

    if z == 1:
        win, lose, draw = 1, 0, 0
    elif z == -1:
        win, lose, draw = 0, 1, 0
    else:
        win, lose, draw = 0, 0, 1

    return samples, win, lose, draw

def _worker(args):
    nnet, cfg, game_num = args
    return run_self_play_game(nnet, cfg, game_num)

def parallel_self_play(nnet, buffer):
    """
    Launch parallel self-play using multiprocessing.
    Saves (state_tensor, pi, value, t_rem) to buffer.
    Returns (win_count, lose_count, draw_count)
    """
    cfg = SelfPlayParams()
    num_games = cfg.num_selfplay_games
    nnet.to("cpu")

    system = platform.system()
    num_cpus = os.cpu_count()
    num_cpus = int(min(num_cpus / 2, num_games))

    print(f"[INFO] Using {num_cpus} CPUs for parallel self-play.")

    win_count, lose_count, draw_count = 0, 0, 0
    added_win, added_lose, added_draw = 0, 0, 0
    total_games_played = 0

    def play_batch(batch_size):
        batch_win, batch_lose, batch_draw = 0, 0, 0
        print(f"[INFO] Launching {batch_size} self-play games using {num_cpus} processes...")
        with mp.Pool(processes=num_cpus) as pool:
            results = pool.map(_worker, [(nnet, cfg, i) for i in range(batch_size)])

        for samples, win, lose, draw in results:
            if samples:
                buffer.add(samples)
            batch_win += win
            batch_lose += lose
            batch_draw += draw
        return batch_win, batch_lose, batch_draw

    # Record starting sizes
    start_win = len(buffer.win)
    start_lose = len(buffer.lose)
    start_draw = len(buffer.tie)

    while True:
        batch_win, batch_lose, batch_draw = play_batch(num_cpus)
        win_count += batch_win
        lose_count += batch_lose
        draw_count += batch_draw
        total_games_played += num_cpus

        buffer.save('./replay_buffer.pkl')

        print(f"[Batch Status] Added: Win: {batch_win}, Lose: {batch_lose}, Tie: {batch_draw}")
        print(f"[Buffer Status] win: {len(buffer.win)}, lose: {len(buffer.lose)}, tie: {len(buffer.tie)}")

        # Check how many new samples of each type have been added since loop start
        new_win = len(buffer.win) - start_win
        new_lose = len(buffer.lose) - start_lose
        new_draw = len(buffer.tie) - start_draw

        print(f"[Run Status] Added since start: Win: {new_win}, Lose: {new_lose}, Tie: {new_draw}")

        if new_win >= 500 and new_lose >= 500 and new_draw >= 500:
            break


    print(f"[Self-Play Complete] Total Games: {total_games_played} | White Wins: {win_count}, Black Win: {lose_count}, Draws: {draw_count}")
    log_self_play_results(win_count, lose_count, draw_count, filename="logs/self_play/self_play_results.csv")
    return win_count, lose_count, draw_count