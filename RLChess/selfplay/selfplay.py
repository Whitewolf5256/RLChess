from utils.config import SelfPlayParams
from utils.logging import log_self_play_results
from selfplay.parallel_selfplay import run_self_play_game

def self_play_gpu_single_process(nnet, buffer):
    """
    Single-process self-play using GPU for inference.
    Plays games until at least 5 of win, lose, draw are added.
    """
    cfg = SelfPlayParams()

    # Ensure model is on GPU
    nnet.to(cfg.device)
    nnet.eval()

    win_count, lose_count, draw_count = 0, 0, 0
    total_games_played = 0

    added_win, added_lose, added_draw = 0, 0, 0
    game_num = 0

    while True:
        print(f"[INFO] Starting self-play game {game_num + 1}...")
        samples, win, lose, draw = run_self_play_game(nnet, cfg, game_num)

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

        if added_win >= 500 and added_lose >= 500 and added_draw >= 500:
            print(f"[INFO] Stopping criteria met.")
            break

        game_num += 1

    print(f"[Self-Play Complete] Total Games: {total_games_played} | White Wins: {win_count}, Black Wins: {lose_count}, Draws: {draw_count}")
    log_self_play_results(win_count, lose_count, draw_count, filename="logs/self_play/self_play_results.csv")
    return win_count, lose_count, draw_count
