import multiprocessing as mp
import numpy as np
import chess
from mcts.mcts import MCTS
from chess_env.chess_game import ChessGame
from utils.config import SelfPlayParams
from utils.logging import log_self_play_results
from utils.logging import log_mcts_moves
import os
import platform
import torch
import copy
import random

def run_self_play_game(nnet, cfg, game_num, opponent_weights=None, log_folder="mcts_moves"):
    """
    If opponent_weights is provided, alternate moves between nnet and opponent_model.
    Otherwise, use nnet for both sides (standard self-play).
    Logs both White and Black moves in order.
    """
    game = ChessGame()
    board = game.reset()
    mcts_nnet = MCTS(game, nnet, cfg)
    mcts_nnet.reset_log()
    played_moves_uci = []
    if opponent_weights is not None:
        opponent_model = copy.deepcopy(nnet)
        opponent_model.load_state_dict(opponent_weights)
        opponent_model.to(cfg.device)
        opponent_model.eval()
        mcts_opp = MCTS(game, opponent_model, cfg)
        mcts_opp.reset_log()
    else:
        mcts_opp = mcts_nnet

    mcts_nnet.root_state = board
    mcts_opp.root_state = board
    data = []

    for t in range(cfg.max_game_length):
        valid = game.get_valid_moves(board)
        pi = np.zeros_like(valid, dtype=np.float32)

        # Decide which model to use for this move
        if opponent_weights is not None:
            mcts_to_use = mcts_nnet if board.turn else mcts_opp
        else:
            mcts_to_use = mcts_nnet

        temp = cfg.exploration_temp
        if board.turn == chess.WHITE:
            temp *= 1.5
        if t >= cfg.temperature_cutoff:
            temp = 0

        player_str = "White" if board.turn else "Black"
        pi = mcts_to_use.get_action_probs(
            board, temp, selfplay=True, board=board, player=player_str, move_number=t
        )
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

        valid_idxs = np.nonzero(pi)[0]        # Indices of valid moves (pi > 0)
        pi_valid = pi[valid_idxs]             # Probabilities for valid moves
        pi_valid = pi_valid / pi_valid.sum()  # Normalize to sum to 1

        action_idx = np.random.choice(len(valid_idxs), p=pi_valid)
        action = valid_idxs[action_idx]
        uci_move = game.index_to_uci_move(action)
        played_moves_uci.append(uci_move)
        
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
        if opponent_weights is not None:
            mcts_opp.update_root(action)

        z = game.get_game_ended(board)
        if z != 0:
            break

    samples = []
    for i, (s_tensor, p, player) in enumerate(data):
        if z == 0:
            value = 0
        elif (player and z == 1) or (not player and z == -1):
            value = 1
        else:
            value = -1
        samples.append((s_tensor, p, value))

    if z == 1:
        win, lose, draw = 1, 0, 0
    elif z == -1:
        win, lose, draw = 0, 1, 0
    else:
        win, lose, draw = 0, 0, 1

    # --- Merge move logs from both MCTS (for correct move order) ---
    if opponent_weights is not None:
        # Sort by move_number to preserve move order
        backprop_info = {
            "white": mcts_nnet.get_backpropagation_info(),
            "black": mcts_opp.get_backpropagation_info()
        }
        log_mcts_moves(game_num, played_moves_uci, backprop_info, game, folder=log_folder)
    else:
        backprop_info = mcts_nnet.get_backpropagation_info()
        log_mcts_moves(game_num, played_moves_uci, backprop_info, game, folder=log_folder)

    return samples, win, lose, draw

def _worker(args):
    nnet, cfg, game_num, opponent_weights = args
    return run_self_play_game(nnet, cfg, game_num, opponent_weights=opponent_weights)

def parallel_self_play(nnet, buffer):
    """
    Launch parallel self-play using multiprocessing and an opponent pool.
    Each game samples an opponent from the pool.
    """
    import random  # Make sure random is imported
    import copy

    cfg = SelfPlayParams()
    # Opponent pool settings
    cfg.opponent_pool_size = 5
    cfg.opponent_selection_prob = [0.4, 0.3, 0.2, 0.08, 0.02]
    cfg.freeze_opponents = True
    cfg.entropy_coeff = 0.01

    num_games = cfg.num_selfplay_games
    nnet.to("cpu")

    system = platform.system()
    num_cpus = os.cpu_count()
    num_cpus = int(min(num_cpus / 2, num_games))

    print(f"[INFO] Using {num_cpus} CPUs for parallel self-play.")

    win_count, lose_count, draw_count = 0, 0, 0
    added_win, added_lose, added_draw = 0, 0, 0
    total_games_played = 0

    # Initialize opponent pool with current model
    opponent_pool = [copy.deepcopy(nnet.state_dict())]

    # Record starting sizes
    start_win = len(buffer.win)
    start_lose = len(buffer.lose)
    start_draw = len(buffer.tie)

    def play_batch(batch_size):
        batch_win, batch_lose, batch_draw = 0, 0, 0

        # Prepare opponent pool probabilities
        if len(opponent_pool) < cfg.opponent_pool_size:
            pool_probs = cfg.opponent_selection_prob[:len(opponent_pool)]
            pool_probs = [p / sum(pool_probs) for p in pool_probs]
        else:
            pool_probs = cfg.opponent_selection_prob

        # For each game, pick an opponent
        opponent_indices = random.choices(
            range(len(opponent_pool)), weights=pool_probs, k=batch_size
        )
        opponent_weights_list = [opponent_pool[idx] for idx in opponent_indices]

        print(f"[INFO] Launching {batch_size} self-play games using {num_cpus} processes...")
        with mp.Pool(processes=num_cpus) as pool:
            results = pool.map(
                _worker,
                [(copy.deepcopy(nnet), cfg, i, opponent_weights_list[i]) for i in range(batch_size)]
            )

        for samples, win, lose, draw in results:
            if samples:
                buffer.add(samples)
            batch_win += win
            batch_lose += lose
            batch_draw += draw
        return batch_win, batch_lose, batch_draw

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

        if new_win >= 5000 and new_lose >= 5000 and new_draw >= 5000:
            # Fill up the pool if not frozen
            if not cfg.freeze_opponents:
                opponent_pool.append(copy.deepcopy(nnet.state_dict()))
                if len(opponent_pool) > cfg.opponent_pool_size:
                    opponent_pool.pop(0)
            elif len(opponent_pool) < cfg.opponent_pool_size:
                # Only fill up the pool once, then freeze
                opponent_pool.append(copy.deepcopy(nnet.state_dict()))
            
            # === Trim the buffer so each category has the same length ===
            min_len = min(len(buffer.win), len(buffer.lose), len(buffer.tie))
            if min_len > 0:
                buffer.win = random.sample(buffer.win, min_len)
                buffer.lose = random.sample(buffer.lose, min_len)
                buffer.tie = random.sample(buffer.tie, min_len)
                print(f"[Buffer Trimmed] win: {len(buffer.win)}, lose: {len(buffer.lose)}, tie: {len(buffer.tie)}")
                buffer.save('./replay_buffer.pkl')
            break

    print(f"[Self-Play Complete] Total Games: {total_games_played} | White Wins: {win_count}, Black Win: {lose_count}, Draws: {draw_count}")
    log_self_play_results(win_count, lose_count, draw_count, filename="logs/self_play/self_play_results.csv")
    return win_count, lose_count, draw_count
