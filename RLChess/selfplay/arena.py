from mcts.mcts import MCTS
import numpy as np
import chess
import chess.engine
import platform
from utils.logging import log_arena_results

if platform.system() == "Windows":
    STOCKFISH_PATH = r"C:\Users\timcw\Downloads\stockfish-windows-x86-64-avx2\stockfish\stockfish-windows-x86-64-avx2.exe"
elif platform.system() == "Darwin":
    STOCKFISH_PATH = "/opt/homebrew/bin/stockfish"
else:
    raise EnvironmentError("Unsupported OS for Stockfish path auto-detection")

def evaluate_new_model(game, model, best_model, cfg):
    import collections

    new_wins = 0
    best_wins = 0
    draws = 0
    tiebreak_new_better = 0
    new_white_wins = 0
    new_black_wins = 0
    best_white_wins = 0
    best_black_wins = 0

    engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)

    total_new_cp_loss = 0
    total_best_cp_loss = 0
    top_match_counts = {"new": 0, "best": 0}
    total_top_choices = 0

    for i in range(cfg.num_games):
        state = game.reset()
        players = [model, best_model] if i % 2 == 0 else [best_model, model]
        mcts_players = [MCTS(game, p, cfg) for p in players]
        board = chess.Board()

        print(f"\n[Arena Game {i+1}] Starting — {'New' if players[0] == model else 'Best'} model is White")

        # Force 1.d4
        d4_move = chess.Move.from_uci("d2d4")
        action = game.move_to_index[d4_move]
        state = game.get_next_state(state, action)
        board.push(d4_move)
        move_count = 1
        current = 0  # White's turn
        model_type = "New" if players[current] == model else "Best"
        color = "White" if board.turn == chess.WHITE else "Black"
        color = "White" if board.turn == chess.WHITE else "Black"
        print(f"Move {move_count}: {model_type} model ({color}) plays d2d4")

        move_count += 1
        current = 1  # Black's turn

        while True:
            pi = mcts_players[current].get_action_probs(state, temp=0, selfplay=False)
            action = np.argmax(pi)
            move = game.index_to_move[action]

            model_type = "New" if players[current] == model else "Best"
            color = "White" if board.turn == chess.WHITE else "Black"
            print(f"Move {move_count}: {model_type} model ({color}) plays {move.uci()}")

            if move not in board.legal_moves:
                print(f"Illegal move attempted: {move}, skipping")
                break
            board.push(move)

            state = game.get_next_state(state, action)
            result = game.get_game_ended(state)
            move_count += 1

            if result != 0 or move_count >= 80:
                if result == 0:
                    draws += 1
                    # Stockfish analysis for tiebreak
                    print("[Tiebreak] Game ended in draw. Running Stockfish analysis...")
                    replay_board = chess.Board()
                    analysis = []
                    new_cp_loss = 0
                    best_cp_loss = 0

                    for idx, move in enumerate(board.move_stack):
                        info = engine.analyse(replay_board, chess.engine.Limit(depth=12))
                        top_move = info["pv"][0]
                        score = info["score"].white().score(mate_score=10000)

                        if score is not None:
                            is_new_turn = (i % 2 == 0 and idx % 2 == 0) or (i % 2 == 1 and idx % 2 == 1)
                            if is_new_turn:
                                new_cp_loss += abs(score)
                            else:
                                best_cp_loss += abs(score)

                            # Count top-move matches
                            if move == top_move:
                                if is_new_turn:
                                    top_match_counts["new"] += 1
                                else:
                                    top_match_counts["best"] += 1
                            total_top_choices += 1

                        replay_board.push(move)

                    total_new_cp_loss += new_cp_loss
                    total_best_cp_loss += best_cp_loss

                    if new_cp_loss < best_cp_loss:
                        tiebreak_new_better += 1
                        print("[Tiebreak] New model had lower centipawn loss.")
                    else:
                        print("[Tiebreak] Best model had lower centipawn loss.")
                else:
                    winner = 0 if result == 1 else 1
                    winning_model = "New" if players[winner] == model else "Best"
                    winning_color = "White" if (i % 2 == 0 and winner == 0) or (i % 2 == 1 and winner == 1) else "Black"

                    print(f"[Result] {winning_model} model won as {winning_color}")
    
                    if winning_model == "New":
                        new_wins += 1
                        if winning_color == "White":
                            new_white_wins += 1
                        else:
                            new_black_wins += 1
                    else:
                        best_wins += 1
                        if winning_color == "White":
                            best_white_wins += 1
                        else:
                            best_black_wins += 1
                break

            current = 1 - current

    engine.quit()

    print("\n--- Final Arena Results ---")
    print(f"New Wins: {new_wins}")
    print(f"Best Wins: {best_wins}")
    print(f"Draws: {draws}")
    print(f"Tiebreak Wins (New model better): {tiebreak_new_better}")
    print(f"Avg CP Loss — New: {total_new_cp_loss / max(draws, 1):.1f}, Best: {total_best_cp_loss / max(draws, 1):.1f}")
    print(f"Top Stockfish Move Matches — New: {top_match_counts['new']}, Best: {top_match_counts['best']} / {total_top_choices} total")
    print(f"--- Win Breakdown by Color ---")
    print(f"New Model — White Wins: {new_white_wins}, Black Wins: {new_black_wins}")
    print(f"Best Model — White Wins: {best_white_wins}, Black Wins: {best_black_wins}\n")

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
        best_black_wins=best_black_wins
    )
    return new_wins, best_wins, draws, tiebreak_new_better, total_new_cp_loss, total_best_cp_loss, top_match_counts
