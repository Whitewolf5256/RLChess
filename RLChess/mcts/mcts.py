import math
import numpy as np
import torch
import copy
import chess


def run_self_play_game(nnet, cfg, game_num, opponent_weights=None, log_folder="mcts_moves"):
    """
    If opponent_weights is provided, alternate moves between nnet and opponent_model.
    Otherwise, use nnet for both sides (standard self-play).
    Logs both White and Black moves in order.
    """
    from your_game_module import ChessGame, log_mcts_moves  # adjust import as needed

    game = ChessGame()
    board = game.reset()

    mcts_nnet = MCTS(game, nnet, cfg)
    mcts_nnet.reset_log()
    played_moves_uci = []

    if opponent_weights is not None:
        opponent_model = copy.deepcopy(nnet)
        opponent_model.load_state_dict(opponent_weights)
        opponent_model.to(cfg.device)
        opponent_model.eval()
        mcts_opp = MCTS(game, opponent_model, cfg)
        mcts_opp.reset_log()
    else:
        mcts_opp = mcts_nnet

    mcts_nnet.root_state = board
    mcts_opp.root_state = board
    data = []

    for t in range(cfg.max_game_length):
        # get valid moves and assert legality
        valid = np.array(game.get_valid_moves(board), dtype=np.float32)
        assert valid.sum() > 0, f"No valid moves at step {t} in game {game_num}!"

        # pick MCTS instance based on side
        mcts_to_use = mcts_nnet if (opponent_weights is None or board.turn) else mcts_opp

        temp = cfg.exploration_temp * (1.5 if board.turn == chess.WHITE else 1.0)
        if t >= cfg.temperature_cutoff:
            temp = 0

        player_str = "White" if board.turn else "Black"

        # get raw pi
        pi = mcts_to_use.get_action_probs(
            board, temp, selfplay=True,
            board=board, player=player_str, move_number=t
        )

        # ensure numeric stability
        if np.any(np.isnan(pi)) or pi.sum() <= 0:
            print(f"[WARN] Bad pi vector at step {t} in game {game_num}, fixing... (NaN or sum<=0)")
            idxs = np.nonzero(valid)[0]
            if len(idxs) > 0:
                pi = np.zeros_like(pi)
                pi[idxs] = 1.0 / len(idxs)
            else:
                pi = np.ones_like(pi) / pi.size
        else:
            pi = pi / pi.sum()

        valid_idxs = np.nonzero(pi)[0]
        pi_valid = pi[valid_idxs]
        pi_valid = pi_valid / pi_valid.sum()

        action_idx = np.random.choice(len(valid_idxs), p=pi_valid)
        action = valid_idxs[action_idx]
        uci_move = game.index_to_uci_move(action)
        played_moves_uci.append(uci_move)

        current_player = board.turn

        state_tensor = torch.tensor(
            game.encode_board(board),
            dtype=torch.float32,
            device=cfg.device
        )
        data.append((state_tensor, pi, current_player))

        # apply move
        board = game.get_next_state(board, action)
        mcts_nnet.update_root(action)
        if opponent_weights is not None:
            mcts_opp.update_root(action)

        z = game.get_game_ended(board)
        if z != 0:
            break

    # prepare samples
    samples = []
    for s_tensor, p, player in data:
        if z == 0:
            value = 0
        elif (player and z == 1) or (not player and z == -1):
            value = 1
        else:
            value = -1
        samples.append((s_tensor, p, value))

    if z == 1:
        win, lose, draw = 1, 0, 0
    elif z == -1:
        win, lose, draw = 0, 1, 0
    else:
        win, lose, draw = 0, 0, 1

    # log
    if opponent_weights is not None:
        backprop_info = {
            "white": mcts_nnet.get_backpropagation_info(),
            "black": mcts_opp.get_backpropagation_info()
        }
    else:
        backprop_info = mcts_nnet.get_backpropagation_info()

    log_mcts_moves(game_num, played_moves_uci, backprop_info, game, folder=log_folder)
    return samples, win, lose, draw


class MCTS:
    def __init__(self, game, nnet, cfg):
        self.game = game
        self.nnet = nnet
        self.cfg = cfg
        self.virtual_loss = 3.0
        self.reset()

    def reset(self):
        self.Qsa = {}
        self.Nsa = {}
        self.Ns = {}
        self.Ps = {}
        self.Es = {}
        self.Vs = {}
        self.root_state = None
        self.nodes_expanded = 0
        self.max_depth = 0
        self.move_log = []

    def search(self, state, is_root=False, depth=0):
        self.max_depth = max(self.max_depth, depth)
        if state.is_game_over():
            z = self.game.get_game_ended(state)
            return -z

        s = self.game.string_representation(state)

        if s not in self.Ps:
            self.nodes_expanded += 1
            board_np = self.game.encode_board(state)
            board_t = torch.tensor(board_np, dtype=torch.float32, device=self.cfg.device).unsqueeze(0)

            with torch.no_grad():
                policy_logits, value = self.nnet(board_t)
                # clamp NaNs/Infs
                policy_logits = torch.nan_to_num(policy_logits, nan=-1e9, posinf=1e9, neginf=-1e9)

            policy = policy_logits[0].softmax(dim=0)
            valid = torch.tensor(self.game.get_valid_moves(state), device=self.cfg.device, dtype=torch.float32)
            assert valid.sum() > 0, f"No valid moves when expanding at depth {depth}!"
            self.Vs[s] = valid.cpu().numpy()
            policy = policy * valid
            total_policy = policy.sum().item()
            total_valid = valid.sum().item()
            if total_policy > 0:
                policy = policy / total_policy
            elif total_valid > 0:
                policy = valid / total_valid
            else:
                policy = torch.ones_like(valid) / len(valid)

            if is_root and self.cfg.add_dirichlet_noise:
                alpha, epsilon = self.cfg.dirichlet_alpha, self.cfg.dirichlet_epsilon
                valid_indices = torch.nonzero(valid).squeeze(1)
                noise = torch.zeros_like(policy)
                if len(valid_indices) > 0:
                    noise_values = torch.from_numpy(
                        np.random.dirichlet([alpha] * len(valid_indices))
                    ).to(self.cfg.device).float()
                    noise[valid_indices] = noise_values
                    policy = (1 - epsilon) * policy + epsilon * noise

            self.Ps[s] = policy.cpu().numpy()
            self.Ns[s] = 0
            return -value.item()

        valid = self.Vs[s]
        policy = self.Ps[s]

        best_act, best_score = None, -float('inf')
        visit_count = self.Ns.get(s, 0)

        for a in np.nonzero(valid)[0]:
            q = self.Qsa.get((s, a), 0)
            n = self.Nsa.get((s, a), 0)
            if visit_count > 0:
                bonus = self.cfg.cpuct * policy[a] * math.sqrt(math.log(visit_count + 1)) / (1 + n)
            else:
                bonus = self.cfg.cpuct * policy[a] * 0.5 / (1 + n)
            u = q + bonus + 1e-5 * np.random.random()
            if u > best_score:
                best_score, best_act = u, a

        if best_act is None:
            return 0

        # virtual loss
        sa = (s, best_act)
        old_q, old_n = self.Qsa.get(sa, 0), self.Nsa.get(sa, 0)
        vl = self.virtual_loss
        self.Qsa[sa] = (old_n * old_q - vl) / (old_n + vl)
        self.Nsa[sa] = old_n + vl

        next_state = self.game.get_next_state(state, best_act)
        v = self.search(next_state, depth=depth+1)

        # backprop update
        self.Qsa[sa] = (old_n * old_q + v) / (old_n + 1)
        self.Nsa[sa] = old_n + 1
        self.Ns[s] = visit_count + 1
        return -v

    def get_action_probs(self, state, temp=1, selfplay=False, board=None, player=None, move_number=None):
        if self.root_state is None:
            self.root_state = state

        self.nodes_expanded = 0
        self.max_depth = 0

        # run simulations
        for i in range(self.cfg.num_mcts_sims):
            self.search(state, is_root=(i == 0 and selfplay))

        s = self.game.string_representation(state)
        valid = np.array(self.game.get_valid_moves(state), dtype=np.float32)
        assert valid.sum() > 0, "No valid moves in get_action_probs!"

        counts = np.array([self.Nsa.get((s, a), 0) for a in range(len(valid))], dtype=np.float32)
        counts = np.nan_to_num(counts, nan=0.0, posinf=0.0, neginf=0.0)

        # deterministic if temp==0
        if temp == 0:
            bests = np.argwhere(counts == counts.max()).flatten()
            probs = np.zeros_like(counts)
            probs[np.random.choice(bests)] = 1.0
            # log
            if selfplay:
                self._log_move(state, probs, board, player, move_number, counts)
            return probs

        # soft probabilities
        counts = counts ** (1.0 / temp)
        total = counts.sum()
        if total > 0:
            probs = counts / total
        else:
            idxs = np.nonzero(valid)[0]
            probs = np.zeros_like(counts)
            if len(idxs) > 0:
                probs[idxs] = 1.0 / len(idxs)

        if selfplay:
            self._log_move(state, probs, board, player, move_number, counts)
        return probs

    def _log_move(self, state, probs, board, player, move_number, counts):
        action = int(np.argmax(probs))
        uci = self.game.index_to_uci_move(action)
        policy_output = self.nnet(
            torch.tensor(self.game.encode_board(state), dtype=torch.float32, device=self.cfg.device).unsqueeze(0)
        )[0].cpu().detach().numpy().flatten().tolist()
        self.move_log.append({
            "move_idx": action,
            "uci": uci,
            "policy": probs.tolist(),
            "policy_output": policy_output,
            "value": float(self.Qsa.get((self.game.string_representation(state), action), 0)),
            "player": player,
            "fen": state.fen() if hasattr(state, "fen") else "N/A",
            "move_number": move_number,
            "visits": int(counts[action])
        })

    def update_root(self, action):
        if self.root_state is None:
            return
        self.root_state = self.game.get_next_state(self.root_state, action)
        if getattr(self.cfg, 'prune_tree', True):
            s_new = self.game.string_representation(self.root_state)
            self.Qsa = {k: v for k, v in self.Qsa.items() if k[0] == s_new}
            self.Nsa = {k: v for k, v in self.Nsa.items() if k[0] == s_new}
            self.Ns = {s: v for s, v in self.Ns.items() if s == s_new}

    def reset_log(self):
        self.move_log = []

    def get_backpropagation_info(self):
        return {
            "Qsa": {str(k): v for k, v in self.Qsa.items()},
            "Nsa": {str(k): v for k, v in self.Nsa.items()},
            "Ns": {str(k): v for k, v in self.Ns.items()},
            "nodes": self.nodes_expanded,
            "max_depth": self.max_depth,
        }
