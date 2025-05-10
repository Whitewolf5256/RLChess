from mcts.mcts import MCTS
import numpy as np
import chess

def evaluate_new_model(game, model, best_model, cfg):
    new_wins = 0
    best_wins = 0
    draws = 0

    for i in range(cfg.num_games):
        state = game.reset()
        players = [model, best_model] if i % 2 == 0 else [best_model, model]
        mcts_players = [MCTS(game, p, cfg) for p in players]
        current = 0
        move_count = 0
        max_moves = 80

        # Always enforce 1.d4 for the White player (player 0)
        d4_move = chess.Move.from_uci("d2d4")
        action = game.move_to_index[d4_move]
        print(f"[Arena Game {i+1}] Forcing 1.d4 by White (Player 0) as action {action}")
        state = game.get_next_state(state, action)
        result = game.get_game_ended(state)
        move_count += 1

        if result != 0:
            # Game already ended from forced move (very rare)
            winner = 0 if result == 1 else 1 if result == -1 else -1
            if winner == 0:
                if players[0] == model:
                    new_wins += 1
                else:
                    best_wins += 1
            else:
                draws += 1
            continue

        current = 1  # After White's move, it's Black's turn

        while True:
            pi = mcts_players[current].get_action_probs(state, temp=0)
            action = np.argmax(pi)
            print(f"Move {move_count}, Player {current}, Action {action}")

            state = game.get_next_state(state, action)
            result = game.get_game_ended(state)
            print(f"Result: {result}")

            move_count += 1

            if result != 0 or move_count >= max_moves:
                print(f"Breaking loop: result={result}, move_count={move_count}")

                if move_count >= max_moves:
                    draws += 1
                elif result == 1:
                    if current == 0:
                        if i % 2 == 0:
                            new_wins += 1 
                        else:
                            best_wins += 1
                    else:
                        if i % 2 == 0:
                            best_wins += 1 
                        else:
                            new_wins += 1
                else:
                    draws += 1
                break

            current = 1 - current

        print(f"[Game {i+1}] New: {new_wins}, Best: {best_wins}, Draws: {draws}")

    print(f"Arena Results — New: {new_wins}, Best: {best_wins}, Draws: {draws}")

    if new_wins > best_wins:
        print("New model outperformed best model — replacing best model.")
        return new_wins, best_wins
    else:
        print("Best model remains — new model did not outperform.")
        return new_wins, best_wins
