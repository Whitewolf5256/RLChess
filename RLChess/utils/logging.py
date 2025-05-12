import csv
import os
from datetime import datetime

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
