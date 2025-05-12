import math
import numpy as np
import torch

# ✅ Enable cuDNN benchmarking
torch.backends.cudnn.benchmark = True

class MCTS:
    def __init__(self, game, nnet, cfg):
        self.game = game
        self.nnet = nnet
        self.cfg = cfg
        self.Qsa = {}  # Q values for (s,a)
        self.Nsa = {}  # visit count for (s,a)
        self.Ns = {}   # visit count for s
        self.Ps = {}   # initial policy returned by neural net for s
        self.root_state = None

    def search(self, state, is_root=False):
        if state.is_game_over():
            z = self.game.get_game_ended(state)
            return -z

        s = self.game.string_representation(state)

        if s not in self.Ps:
            board_np = self.game.encode_board(state)
            board_t = torch.tensor(board_np, dtype=torch.float32, device=self.cfg.device).unsqueeze(0)

            with torch.no_grad():
                policy_logits, value = self.nnet(board_t)

            policy = policy_logits[0].softmax(dim=0)
            valid = torch.tensor(self.game.get_valid_moves(state), device=self.cfg.device, dtype=torch.float32)

            policy = policy * valid

            total_policy = policy.sum()
            total_valid = valid.sum()

            if total_policy > 0:
                policy = policy / total_policy
            elif total_valid > 0:
                policy = valid / total_valid
            else:
                policy = torch.ones_like(valid) / len(valid)

            # ✅ Add Dirichlet noise at root
            if is_root and self.cfg.add_dirichlet_noise:
                alpha = self.cfg.dirichlet_alpha
                epsilon = self.cfg.dirichlet_epsilon
                noise = torch.from_numpy(np.random.dirichlet([alpha] * len(valid))).to(self.cfg.device)
                policy = (1 - epsilon) * policy + epsilon * noise

            self.Ps[s] = policy.cpu().numpy()
            self.Ns[s] = 0
            return -value.item()

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

        if best_act is None:
            return 0

        next_state = self.game.get_next_state(state, best_act)
        v = self.search(next_state)

        old_q = self.Qsa.get((s, best_act), 0)
        old_n = self.Nsa.get((s, best_act), 0)
        self.Qsa[(s, best_act)] = (old_n * old_q + v) / (old_n + 1)
        self.Nsa[(s, best_act)] = old_n + 1
        self.Ns[s] += 1
        return -v

    def get_action_probs(self, state, temp=1):
        if self.root_state is None:
            self.root_state = state

        for i in range(self.cfg.num_mcts_sims):
            self.search(state, is_root=(i == 0))

        s = self.game.string_representation(state)
        valid = self.game.get_valid_moves(state)
        counts = np.array([self.Nsa.get((s, a), 0) for a in range(len(valid))], dtype=np.float32)

        if counts.sum() == 0:
            idxs = np.nonzero(valid)[0]
            probs = np.zeros_like(counts)
            if len(idxs) > 0:
                probs[idxs] = 1.0 / len(idxs)
            return probs

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
            idxs = np.nonzero(valid)[0]
            probs = np.zeros_like(counts)
            if len(idxs) > 0:
                probs[idxs] = 1.0 / len(idxs)
            return probs

    def update_root(self, action):
        if self.root_state is None:
            return
        self.root_state = self.game.get_next_state(self.root_state, action)
