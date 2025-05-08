import math
import numpy as np
import torch

class MCTS:
    def __init__(self, game, nnet, cfg):
        self.game = game
        self.nnet = nnet
        self.cfg = cfg
        self.Qsa = {}   # Q values for (s,a)
        self.Nsa = {}   # visit count for (s,a)
        self.Ns  = {}   # visit count for s
        self.Ps  = {}   # initial policy returned by neural net for s

    def search(self, state):
        # If terminal state, return negative reward (from current player's perspective)
        if state.is_game_over():
            z = self.game.get_game_ended(state)
            return -z

        s = self.game.string_representation(state)

        # Expand leaf node
        if s not in self.Ps:
            # Get policy and value from neural network
            board_t = torch.tensor(self.game.encode_board(state), dtype=torch.float32)
            board_t = board_t.unsqueeze(0).to(self.cfg.device)
            policy_logits, value = self.nnet(board_t)
            policy = policy_logits.detach().cpu().numpy().flatten()

            # Mask invalid moves
            valid = self.game.get_valid_moves(state)
            policy = policy * valid

            # Normalize or fallback
            total_policy = policy.sum()
            total_valid = valid.sum()
            if total_policy > 0:
                policy = policy / total_policy
            elif total_valid > 0:
                policy = valid / total_valid
            else:
                # No valid moves: uniform over full action space
                policy = np.ones_like(valid, dtype=np.float32) / len(valid)

            self.Ps[s] = policy
            self.Ns[s] = 0
            return -value.item()

        # Select action with highest UCT score
        valid = self.game.get_valid_moves(state)
        policy = self.Ps[s]
        best_act = None
        best_score = -float('inf')

        for a in np.nonzero(valid)[0]:
            q = self.Qsa.get((s, a), 0)
            n = self.Nsa.get((s, a), 0)
            u = q + self.cfg.cpuct * policy[a] * math.sqrt(self.Ns[s] + 1e-8) / (1 + n)
            if u > best_score:
                best_score = u
                best_act = a

        # If no valid action found, treat as terminal draw
        if best_act is None:
            return 0

        # Recurse
        next_state = self.game.get_next_state(state, best_act)
        v = self.search(next_state)

        # Update Qsa, Nsa, Ns
        old_q = self.Qsa.get((s, best_act), 0)
        old_n = self.Nsa.get((s, best_act), 0)
        self.Qsa[(s, best_act)] = (old_n * old_q + v) / (old_n + 1)
        self.Nsa[(s, best_act)] = old_n + 1
        self.Ns[s] += 1
        return -v

    def get_action_probs(self, state, temp=1):
        # Run simulations
        for _ in range(self.cfg.num_mcts_sims):
            self.search(state)

        s = self.game.string_representation(state)
        valid = self.game.get_valid_moves(state)
        counts = np.array([self.Nsa.get((s, a), 0) for a in range(len(valid))], dtype=np.float32)

        # If no visits, uniform over valid
        if counts.sum() == 0:
            idxs = np.nonzero(valid)[0]
            probs = np.zeros_like(counts)
            if len(idxs) > 0:
                probs[idxs] = 1.0 / len(idxs)
            return probs

        # Temperature
        if temp == 0:
            bests = np.argwhere(counts == counts.max()).flatten()
            probs = np.zeros_like(counts)
            probs[np.random.choice(bests)] = 1.0
            return probs

        counts = counts ** (1.0 / temp)
        total = counts.sum()
        if total > 0:
            return counts / total
        else:
            # fallback uniform
            idxs = np.nonzero(valid)[0]
            probs = np.zeros_like(counts)
            if len(idxs) > 0:
                probs[idxs] = 1.0 / len(idxs)
            return probs