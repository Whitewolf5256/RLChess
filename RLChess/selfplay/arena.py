from mcts.mcts import MCTS

def evaluate_new_model(game, model, best_model, cfg):
    new_wins = 0
    best_wins = 0
    draws = 0

    for i in range(cfg.num_games):
        state = game.reset()
        players = [model, best_model] if i % 2 == 0 else [best_model, model]
        mcts_players = [MCTS(game, p, cfg) for p in players]
        current = 0
        while True:
            pi = mcts_players[current].get_action_probs(state, temp=0)
            action = pi.index(max(pi))
            state = game.get_next_state(state, action)
            result = game.get_game_ended(state)
            if result != 0:
                if result == 1:
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

    total = new_wins + best_wins + draws
    print(f"Arena Results — New: {new_wins}, Best: {best_wins}, Draws: {draws}")

    if new_wins > best_wins:
        print("New model outperformed best model — replacing best model.")
        return new_wins, best_wins
    else:
        print("Best model remains — new model did not outperform.")
        return new_wins, best_wins