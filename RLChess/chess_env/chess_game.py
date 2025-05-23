﻿import chess
import numpy as np

class ChessGame:
    def __init__(self):
        self.move_to_index, self.index_to_move = self._build_move_lookup()

    def reset(self):
        return chess.Board()

    def get_next_state(self, board, action_idx):
        move = self.index_to_move[action_idx]
        new_board = board.copy()
        if move in new_board.legal_moves:
            new_board.push(move)
        else:
            raise ValueError(f"Attempted illegal move: {move.uci()}")
        return new_board

    def get_valid_moves(self, board):
        valid = np.zeros(len(self.move_to_index), dtype=np.uint8)
        for m in board.legal_moves:
            idx = self.move_to_index.get(m)
            if idx is not None:
                valid[idx] = 1
        return valid

    def get_game_ended(self, board):
        if board.is_game_over():
            res = board.result()
            if res == '1-0': return 1
            if res == '0-1': return -1
            return 0
        return 0

    def string_representation(self, board):
        return board.fen()

    def encode_board(self, board):
        planes = np.zeros((18,8,8), dtype=np.float32)
        piece_map = board.piece_map()
        plane_map = {'P':0,'N':1,'B':2,'R':3,'Q':4,'K':5,'p':6,'n':7,'b':8,'r':9,'q':10,'k':11}
        for sq,p in piece_map.items():
            r = 7 - chess.square_rank(sq) if board.turn == chess.WHITE else chess.square_rank(sq)
            c = chess.square_file(sq)
            planes[plane_map[p.symbol()]][r][c] = 1
        planes[12,:,:]=board.turn
        planes[13,:,:]=board.has_kingside_castling_rights(chess.WHITE)
        planes[14,:,:]=board.has_queenside_castling_rights(chess.WHITE)
        planes[15,:,:]=board.has_kingside_castling_rights(chess.BLACK)
        planes[16,:,:]=board.has_queenside_castling_rights(chess.BLACK)
        planes[17,:,:]=board.fullmove_number/100.0
        return planes

    def _build_move_lookup(self):
        move_to_index = {}
        index_to_move = {}
        idx = 0
        for from_sq in chess.SQUARES:
            for to_sq in chess.SQUARES:
                for promo in [None, chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT]:
                    m = chess.Move(from_sq, to_sq, promotion=promo)
                    move_to_index[m] = idx
                    index_to_move[idx] = m
                    idx += 1
        return move_to_index, index_to_move

    def index_to_uci_move(self, idx):
        move = self.index_to_move.get(idx)
        if move is not None:
            return move.uci()
        return "invalid"
