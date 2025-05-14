import csv
import os
from datetime import datetime
import chess 
import chess.pgn

def ensure_log_dir(path):
    os.makedirs(path, exist_ok=True)

def log_self_play_results(white_wins, black_wins, draws, filename="logs/self_play/self_play_results.csv"):
    # Ensure directory exists
    ensure_log_dir(os.path.dirname(filename))

    # --- CSV Logging (for programmatic use) ---
    file_exists = os.path.isfile(filename)
    with open(filename, mode="a", newline="") as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(["timestamp", "white_wins", "black_wins", "draws"])
        writer.writerow([datetime.now().isoformat(), white_wins, black_wins, draws])

    # --- Pretty Text Logging (for human inspection) ---
    pretty_filename = filename.replace(".csv", "_pretty.txt")
    with open(pretty_filename, mode="a") as f:
        f.write("\n=== Self-Play Summary ===\n")
        f.write(f"{'Timestamp':<15}: {datetime.now().isoformat()}\n")
        f.write(f"{'White Wins':<15}: {white_wins}\n")
        f.write(f"{'Black Wins':<15}: {black_wins}\n")
        f.write(f"{'Draws':<15}: {draws}\n")
        f.write("===========================\n")

def log_arena_results(
    new_wins, best_wins, draws, tiebreak_new_better,
    total_new_cp_loss, total_best_cp_loss, total_top_choices,
    top_match_counts, new_white_wins, new_black_wins,
    best_white_wins, best_black_wins,
    filename="logs/arena/arena_results.csv"
):
    # Ensure the directory exists for the logs
    ensure_log_dir(os.path.dirname(filename))
    file_exists = os.path.isfile(filename)

    # --- CSV Logging (for programmatic use) ---
    with open(filename, mode="a", newline="") as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow([
                "timestamp", "new_wins", "best_wins", "draws", "tiebreak_new_better",
                "avg_cp_loss_new", "avg_cp_loss_best",
                "top_move_matches_new", "top_move_matches_best", "total_top_choices",
                "new_white_wins", "new_black_wins",
                "best_white_wins", "best_black_wins"
            ])
        writer.writerow([
            datetime.now().isoformat(), new_wins, best_wins, draws, tiebreak_new_better,
            total_new_cp_loss / max(draws, 1), total_best_cp_loss / max(draws, 1),
            top_match_counts["new"], top_match_counts["best"], total_top_choices,
            new_white_wins, new_black_wins, best_white_wins, best_black_wins
        ])

    # --- Pretty Text Logging (for human inspection) ---
    pretty_filename = filename.replace(".csv", "_pretty.txt")
    with open(pretty_filename, mode="a") as f:
        f.write("\n=== Arena Results Summary ===\n")
        f.write(f"{'Timestamp':<30}: {datetime.now().isoformat()}\n")
        f.write(f"{'New Wins':<30}: {new_wins}\n")
        f.write(f"{'Best Wins':<30}: {best_wins}\n")
        f.write(f"{'Draws':<30}: {draws}\n")
        f.write(f"{'Tiebreak New Better':<30}: {tiebreak_new_better}\n")
        f.write(f"{'Avg CP Loss New':<30}: {total_new_cp_loss / max(draws, 1):.4f}\n")
        f.write(f"{'Avg CP Loss Best':<30}: {total_best_cp_loss / max(draws, 1):.4f}\n")
        f.write(f"{'Top Move Matches (New)':<30}: {top_match_counts['new']}\n")
        f.write(f"{'Top Move Matches (Best)':<30}: {top_match_counts['best']}\n")
        f.write(f"{'Total Top Choices':<30}: {total_top_choices}\n")
        f.write(f"{'New White Wins':<30}: {new_white_wins}\n")
        f.write(f"{'New Black Wins':<30}: {new_black_wins}\n")
        f.write(f"{'Best White Wins':<30}: {best_white_wins}\n")
        f.write(f"{'Best Black Wins':<30}: {best_black_wins}\n")
        f.write("===========================\n")

def ascii_board(fen):
    """Return a pretty ASCII board from FEN."""
    board = chess.Board(fen)
    return board.unicode(invert_color=False)

def moves_to_pgn(start_fen, move_uci_list):
    """
    Convert a list of UCI move strings to a PGN string.
    """
    board = chess.Board(start_fen)
    game = chess.pgn.Game()
    game.setup(board)
    node = game

    for uci in move_uci_list:
        move = chess.Move.from_uci(uci)
        if move not in board.legal_moves:
            raise ValueError(f"Move {uci} is not legal in position {board.fen()}")
        node = node.add_main_variation(move)
        board.push(move)

    return str(game)

def log_mcts_moves(game_num, move_uci_list, backprop_info, game, folder="mcts_moves"):
    """
    Logs played UCI moves to a pretty TXT with Chess.com/PGN info.
    """
    os.makedirs(folder, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    pretty_path = os.path.join(folder, f"game_{game_num:04d}_{timestamp}_pretty.txt")

    # Generate PGN and Chess.com instructions
    start_fen = game.STARTING_FEN if hasattr(game, 'STARTING_FEN') else chess.STARTING_FEN
    pgn_str = moves_to_pgn(start_fen, move_uci_list)

    with open(pretty_path, "w", encoding="utf-8") as f:
        f.write(f"Game {game_num}\n" + "="*40 + "\n")
        f.write("Chess.com analysis link:\n")
        f.write("1. Copy the PGN below.\n")
        f.write("2. Go to https://www.chess.com/analysis?tab=analysis\n")
        f.write("3. Click the PGN tab and paste the PGN.\n\n")
        f.write(pgn_str + "\n\n")
        f.write("Move list (UCI):\n")
        f.write(" ".join(move_uci_list) + "\n\n")
        f.write("\nBackpropagation info:\n")
        f.write(str(backprop_info))
        f.write("\n")