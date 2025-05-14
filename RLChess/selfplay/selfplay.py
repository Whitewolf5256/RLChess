import torch
import copy
import random
from utils.config import SelfPlayParams
from utils.logging import log_self_play_results
from selfplay.parallel_selfplay import run_self_play_game

def self_play_gpu_single_process(nnet, buffer):
    """
    Self-play with an opponent pool.
    Keeps a pool of the last 5 models, samples opponents with specified probabilities,
    and uses entropy regularization.
    """
    cfg = SelfPlayParams()
    # Set new attributes
    cfg.opponent_pool_size = 5
    cfg.opponent_selection_prob = [0.4, 0.3, 0.2, 0.08, 0.02]
    cfg.freeze_opponents = True
    cfg.entropy_coeff = 0.01

    nnet.to(cfg.device)
    nnet.eval()

    # Opponent pool: list of model weights (state_dicts)
    opponent_pool = [copy.deepcopy(nnet.state_dict())]

    win_count, lose_count, draw_count = 0, 0, 0
    total_games_played = 0

    added_win, added_lose, added_draw = 0, 0, 0
    game_num = 0

    while True:
        print(f"[INFO] Starting self-play game {game_num + 1}...")

        # Select opponent from pool
        if len(opponent_pool) < cfg.opponent_pool_size:
            pool_probs = cfg.opponent_selection_prob[:len(opponent_pool)]
            pool_probs = [p/sum(pool_probs) for p in pool_probs]
        else:
            pool_probs = cfg.opponent_selection_prob

        # Choose opponent index
        opponent_idx = torch.multinomial(
            torch.tensor(pool_probs), 1
        ).item()
        # Pass opponent weights, not model
        opponent_weights = opponent_pool[opponent_idx]

        # Self-play: pass both current model and opponent weights
        samples, win, lose, draw = run_self_play_game(
            nnet, cfg, game_num, opponent_weights=opponent_weights
        )

        # Count samples before adding to buffer
        win_samples = sum(1 for s in samples if s[2] == 1)
        lose_samples = sum(1 for s in samples if s[2] == -1)
        draw_samples = sum(1 for s in samples if s[2] == 0)

        if samples:
            buffer.add(samples)
            win_count += win
            lose_count += lose
            draw_count += draw
            added_win += win_samples
            added_lose += lose_samples
            added_draw += draw_samples
            total_games_played += 1

        buffer.save('./replay_buffer.pkl')

        print(f"[Progress] Added Samples - Win: {added_win}, Lose: {added_lose}, Tie: {added_draw}")

        # Keeps earliest games 
        if added_win >= 5000 and added_lose >= 5000 and added_draw >= 5000:
            print(f"[INFO] Stopping criteria met.")
            min_len = min(len(buffer.win), len(buffer.lose), len(buffer.tie))
            if min_len > 0:
                buffer.win = buffer.win[:min_len]
                buffer.lose = buffer.lose[:min_len]
                buffer.tie = buffer.tie[:min_len]
                print(f"[Buffer Trimmed] win: {len(buffer.win)}, lose: {len(buffer.lose)}, tie: {len(buffer.tie)}")
                buffer.save('./replay_buffer.pkl')
            break

        # After each game, update opponent pool (if not freezing, but here we freeze)
        if not cfg.freeze_opponents:
            opponent_pool.append(copy.deepcopy(nnet.state_dict()))
            if len(opponent_pool) > cfg.opponent_pool_size:
                opponent_pool.pop(0)
        elif len(opponent_pool) < cfg.opponent_pool_size:
            opponent_pool.append(copy.deepcopy(nnet.state_dict()))

        game_num += 1

    print(f"[Self-Play Complete] Total Games: {total_games_played} | White Wins: {win_count}, Black Wins: {lose_count}, Draws: {draw_count}")
    log_self_play_results(win_count, lose_count, draw_count, filename="logs/self_play/self_play_results.csv")
    return win_count, lose_count, draw_count
