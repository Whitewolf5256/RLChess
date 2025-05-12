def self_play_gpu_single_process(nnet, buffer):
    """
    Single-process self-play using GPU for inference.
    Plays games until at least 500 of win, lose, draw are added.
    """
    from utils.config import SelfPlayParams
    from utils.logging import log_self_play_results
    from selfplay.parallel_selfplay import run_self_play_game

    cfg = SelfPlayParams()

    # Ensure model is on GPU
    nnet.to(cfg.device)
    nnet.eval()

    win_count, lose_count, draw_count = 0, 0, 0
    total_games_played = 0

    # Accumulators for samples added (regardless of buffer maxlen)
    new_win = 0
    new_lose = 0
    new_draw = 0

    game_num = 0
    while True:
        print(f"[INFO] Starting self-play game {game_num + 1}...")
        samples, win, lose, draw = run_self_play_game(nnet, cfg, game_num)

        if samples:
            buffer.add(samples)
            win_count += win
            lose_count += lose
            draw_count += draw
            new_win += win
            new_lose += lose
            new_draw += draw
            total_games_played += 1

        buffer.save('./replay_buffer.pkl')

        print(f"[Progress] Added - Win: {new_win}, Lose: {new_lose}, Tie: {new_draw}")

        if new_win >= 500 and new_lose >= 500 and new_draw >= 500:
            print(f"[INFO] Stopping criteria met.")
            break

        game_num += 1

    print(f"[Self-Play Complete] Total Games: {total_games_played} | White Wins: {win_count}, Black Wins: {lose_count}, Draws: {draw_count}")
    log_self_play_results(win_count, lose_count, draw_count, filename="logs/self_play/self_play_results.csv")
    return win_count, lose_count, draw_count
