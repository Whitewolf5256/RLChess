import multiprocessing as mp
import numpy as np
import chess
import chess.engine
from mcts.mcts import MCTS
from utils.logging import log_arena_results

STOCKFISH_PATH = r"C:\Users\timcw\Downloads\stockfish-windows-x86-64-avx2\stockfish\stockfish-windows-x86-64-avx2.exe"
#STOCKFISH_PATH = "/opt/homebrew/bin/stockfish"

def run_arena_game(args):
    game, model, best_model, cfg, game_index = args
    engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)

    state = game.reset()
    players = [model, best_model] if game_index % 2 == 0 else [best_model, model]
    mcts_players = [MCTS(game, p, cfg) for p in players]
    board = chess.Board()

    # Force 1.d4
    d4_move = chess.Move.from_uci("d2d4")
    action = game.move_to_index[d4_move]
    state = game.get_next_state(state, action)
    board.push(d4_move)

    current = 1
    move_count = 2
    result = 0
    new_cp_loss = 0
    best_cp_loss = 0
    top_matches = {"new": 0, "best": 0}
    top_total = 0

    while True:
        pi = mcts_players[current].get_action_probs(state, temp=0, selfplay=False)
        action = np.argmax(pi)
        move = game.index_to_move[action]

        if move not in board.legal_moves:
            break
        board.push(move)
        state = game.get_next_state(state, action)
        result = game.get_game_ended(state)
        move_count += 1

        if result != 0 or move_count >= 80:
            if result == 0:
                replay_board = chess.Board()
                for idx, move in enumerate(board.move_stack):
                    info = engine.analyse(replay_board, chess.engine.Limit(depth=12))
                    top_move = info["pv"][0]
                    score = info["score"].white().score(mate_score=10000)

                    if score is not None:
                        is_new_turn = (game_index % 2 == 0 and idx % 2 == 0) or (game_index % 2 == 1 and idx % 2 == 1)
                        if is_new_turn:
                            new_cp_loss += abs(score)
                        else:
                            best_cp_loss += abs(score)

                        if move == top_move:
                            if is_new_turn:
                                top_matches["new"] += 1
                            else:
                                top_matches["best"] += 1
                        top_total += 1

                    replay_board.push(move)

            break

        current = 1 - current

    engine.quit()

    winner = None
    if result == 1:
        winner = 0
    elif result == -1:
        winner = 1

    outcome = {
        "winner": winner,
        "draw": result == 0,
        "game_index": game_index,
        "new_white_win": winner == 0 and players[0] == model and game_index % 2 == 0,
        "new_black_win": winner == 0 and players[0] == model and game_index % 2 == 1,
        "best_white_win": winner == 0 and players[0] == best_model and game_index % 2 == 0,
        "best_black_win": winner == 0 and players[0] == best_model and game_index % 2 == 1,
        "tiebreak_new_better": new_cp_loss < best_cp_loss if result == 0 else False,
        "new_cp_loss": new_cp_loss,
        "best_cp_loss": best_cp_loss,
        "top_matches": top_matches,
        "top_total": top_total,
        "new_win": winner is not None and players[winner] == model,
        "best_win": winner is not None and players[winner] == best_model,
    }
    return outcome

def _worker(args):
    return run_arena_game(args)

def parallel_arena(game, model, best_model, cfg):
    num_games = cfg.num_games
    num_cpus = min(4, mp.cpu_count(), num_games)
    print(f"[INFO] Running {num_games} arena games using {num_cpus} CPUs...")
    
    model.to("cpu")
    best_model.to("cpu")

    args = [(game, model, best_model, cfg, i) for i in range(num_games)]

    with mp.Pool(processes=num_cpus) as pool:
        results = pool.map(_worker, args)

    new_wins = sum(r["new_win"] for r in results)
    best_wins = sum(r["best_win"] for r in results)
    draws = sum(r["draw"] for r in results)
    tiebreak_new_better = sum(r["tiebreak_new_better"] for r in results)
    new_white_wins = sum(r["new_white_win"] for r in results)
    new_black_wins = sum(r["new_black_win"] for r in results)
    best_white_wins = sum(r["best_white_win"] for r in results)
    best_black_wins = sum(r["best_black_win"] for r in results)
    total_new_cp_loss = sum(r["new_cp_loss"] for r in results)
    total_best_cp_loss = sum(r["best_cp_loss"] for r in results)
    total_top_choices = sum(r["top_total"] for r in results)
    top_match_counts = {
        "new": sum(r["top_matches"]["new"] for r in results),
        "best": sum(r["top_matches"]["best"] for r in results)
    }

    print("\n--- Final Arena Results (Parallel) ---")
    print(f"New Wins: {new_wins}")
    print(f"Best Wins: {best_wins}")
    print(f"Draws: {draws}")
    print(f"Tiebreak Wins (New model better): {tiebreak_new_better}")
    print(f"Avg CP Loss — New: {total_new_cp_loss / max(draws, 1):.1f}, Best: {total_best_cp_loss / max(draws, 1):.1f}")
    print(f"Top Stockfish Move Matches — New: {top_match_counts['new']}, Best: {top_match_counts['best']} / {total_top_choices} total")
    print(f"--- Win Breakdown by Color ---")
    print(f"New Model — White Wins: {new_white_wins}, Black Wins: {new_black_wins}")
    print(f"Best Model — White Wins: {best_white_wins}, Black Wins: {best_black_wins}")

    log_arena_results(
        new_wins=new_wins,
        best_wins=best_wins,
        draws=draws,
        tiebreak_new_better=tiebreak_new_better,
        total_new_cp_loss=total_new_cp_loss,
        total_best_cp_loss=total_best_cp_loss,
        total_top_choices=total_top_choices,
        top_match_counts=top_match_counts,
        new_white_wins=new_white_wins,
        new_black_wins=new_black_wins,
        best_white_wins=best_white_wins,
        best_black_wins=best_black_wins,
        filename="logs/arena/arena_results.csv"
        )

    return new_wins, best_wins, draws, tiebreak_new_better, total_new_cp_loss, total_best_cp_loss, top_match_counts
