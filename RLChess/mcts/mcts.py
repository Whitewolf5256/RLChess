import math
import numpy as np
import torch
import os
import csv

class MCTS:
    def __init__(self, game, nnet, cfg):
        self.game = game
        self.nnet = nnet
        self.cfg = cfg
        self.Qsa = {}  # Q values for (s,a)
        self.Nsa = {}  # visit count for (s,a)
        self.Ns = {}   # visit count for s
        self.Ps = {}   # initial policy returned by neural net for s
        self.Es = {}   # game ended for s
        self.Vs = {}   # Valid moves for state s
        self.root_state = None
        
        # New virtual loss parameter to discourage simultaneous exploration of same paths
        self.virtual_loss = 3.0
        
        # Track nodes expanded this search
        self.nodes_expanded = 0
        
        # Track maximum depth reached during search
        self.max_depth = 0

        # For logging
        self.move_log = []

    def reset(self):
        """Reset the search tree between games"""
        self.Qsa = {}
        self.Nsa = {}
        self.Ns = {}
        self.Ps = {}
        self.Es = {}
        self.Vs = {}
        self.root_state = None
        self.nodes_expanded = 0
        self.max_depth = 0
        self.move_log = [] # Clear move log for each new game

    def search(self, state, is_root=False, depth=0):
        """
        Search the game tree starting from state
        is_root: whether this is the root node of the search
        depth: current depth in the search tree
        """
        # Update max depth tracking
        self.max_depth = max(self.max_depth, depth)
        
        # Check for game termination
        if state.is_game_over():
            z = self.game.get_game_ended(state)
            return -z

        s = self.game.string_representation(state)

        # Expand node if not visited before
        if s not in self.Ps:
            self.nodes_expanded += 1
            
            # Encode board and get policy and value from neural network
            board_np = self.game.encode_board(state)
            board_t = torch.tensor(board_np, dtype=torch.float32, device=self.cfg.device).unsqueeze(0)

            with torch.no_grad():
                policy_logits, value = self.nnet(board_t)

            # Convert to probabilities and filter valid moves
            policy = policy_logits[0].softmax(dim=0)
            valid = torch.tensor(self.game.get_valid_moves(state), device=self.cfg.device, dtype=torch.float32)
            
            # Cache valid moves
            self.Vs[s] = valid.cpu().numpy()

            # Multiply policy by valid moves mask
            policy = policy * valid
            
            # Safety checks for policy
            total_policy = policy.sum().item()
            total_valid = valid.sum().item()

            if total_policy > 0:
                policy = policy / total_policy
            elif total_valid > 0:
                # Uniform random policy if no valid moves with non-zero probability
                policy = valid / total_valid
            else:
                # Uniform random over all moves as fallback
                policy = torch.ones_like(valid) / len(valid)

            # Add Dirichlet noise at root for exploration
            if is_root and self.cfg.add_dirichlet_noise:
                alpha = self.cfg.dirichlet_alpha
                epsilon = self.cfg.dirichlet_epsilon
                # Generate noise only for valid moves
                valid_indices = torch.nonzero(valid).squeeze(1)
                
                # Create noise array filled with zeros
                noise = torch.zeros_like(policy)
                
                # Only generate Dirichlet noise if there are valid moves
                if len(valid_indices) > 0:
                    # Generate noise only for valid indices
                    noise_values = torch.from_numpy(
                        np.random.dirichlet([alpha] * len(valid_indices))
                    ).to(self.cfg.device).float()
                    # Place noise values at valid indices
                    for i, idx in enumerate(valid_indices):
                        noise[idx] = noise_values[i]
                    
                    # Apply noise to policy
                    policy = (1 - epsilon) * policy + epsilon * noise

            # Store data for this state
            self.Ps[s] = policy.cpu().numpy()
            self.Ns[s] = 0
            return -value.item()

        # If state is already expanded, use cached valid moves
        valid = self.Vs.get(s, np.array(self.game.get_valid_moves(state)))
        policy = self.Ps[s]

        # UCB formula for action selection
        best_act = None
        best_score = -float('inf')
        
        # Add some small noise to break ties consistently instead of arbitrarily
        noise_factor = 1e-5
        
        # Make sure s exists in self.Ns
        if s not in self.Ns:
            self.Ns[s] = 0
            
        for a in np.nonzero(valid)[0]:
            q = self.Qsa.get((s, a), 0)
            n = self.Nsa.get((s, a), 0)
            
            # Calculate exploration bonus with a small noise factor to break ties
            # Use a safer approach that checks for dictionary existence
            visit_count = self.Ns[s]
            
            # Use a simpler PUCT formula that's less prone to errors
            if visit_count > 0:
                exploration_term = self.cfg.cpuct * policy[a] * math.sqrt(math.log(visit_count + 1)) / (1 + n)
            else:
                exploration_term = self.cfg.cpuct * policy[a] * 0.5 / (1 + n)  # Initial case with no visits
            
            # UCB formula with small noise for consistent tie-breaking
            u = q + exploration_term + noise_factor * np.random.random()
            
            if u > best_score:
                best_score = u
                best_act = a

        # If no valid moves, return draw evaluation
        if best_act is None:
            return 0

        # Apply virtual loss (temporarily reduces appeal of this move)
        sa_key = (s, best_act)
        old_q = self.Qsa.get(sa_key, 0)
        old_n = self.Nsa.get(sa_key, 0)
        
        # Temporarily add virtual loss
        virtual_n = old_n + self.virtual_loss
        virtual_q = (old_n * old_q - self.virtual_loss) / virtual_n if virtual_n > 0 else -self.virtual_loss
        self.Qsa[sa_key] = virtual_q
        self.Nsa[sa_key] = virtual_n
        
        # Recursive search
        next_state = self.game.get_next_state(state, best_act)
        v = self.search(next_state, depth=depth+1)
        
        # Remove virtual loss and update with real result
        self.Qsa[sa_key] = (old_n * old_q + v) / (old_n + 1)
        self.Nsa[sa_key] = old_n + 1
        self.Ns[s] += 1
        
        return -v

    def get_action_probs(self, state, temp=1, selfplay=False, board=None, player=None, move_number=None):
        """
        Returns action probabilities based on MCTS simulation.
        temp: temperature parameter controlling exploration
        selfplay: whether this is being used for self-play (enables dirichlet noise)
        move_number: the ply number (for logging and sorting)
        """
        # Store root state for potential tree reuse
        if self.root_state is None:
            self.root_state = state

        # Reset node stats for new search
        self.nodes_expanded = 0
        self.max_depth = 0

        # Run MCTS simulations
        for i in range(self.cfg.num_mcts_sims):
            self.search(state, is_root=(i == 0 and selfplay))

        # Get state representation and valid moves
        s = self.game.string_representation(state)
        valid = self.game.get_valid_moves(state)

        # Get visit counts for each action
        counts = np.array([self.Nsa.get((s, a), 0) for a in range(len(valid))], dtype=np.float32)

        # Handle edge case: no visits
        if counts.sum() == 0:
            idxs = np.nonzero(valid)[0]
            probs = np.zeros_like(counts)
            if len(idxs) > 0:
                probs[idxs] = 1.0 / len(idxs)
            return probs

        # Temperature-adjusted probabilities
        if temp == 0:  # Select best move deterministically
            bests = np.argwhere(counts == counts.max()).flatten()
            probs = np.zeros_like(counts)
            probs[np.random.choice(bests)] = 1.0

            if selfplay and board is not None and player is not None:
                action = np.argmax(probs)
                uci = self.game.index_to_uci_move(action)
                policy_output = self.nnet(
                    torch.tensor(self.game.encode_board(state), dtype=torch.float32, device=self.cfg.device).unsqueeze(0)
                )[0].cpu().detach().numpy().flatten()
                move_row = {
                    "move_idx": int(action),
                    "uci": uci,
                    "policy": probs.tolist(),
                    "policy_output": policy_output.tolist(),
                    "value": float(self.Qsa.get((s, action), 0)),
                    "player": player,
                    "fen": state.fen() if hasattr(state, "fen") else "N/A",
                    "move_number": move_number if move_number is not None else 0,
                    "visits": int(counts[action])  # Actual visit count for this move
                }
                self.move_log.append(move_row)

            return probs

        # Calculate temperature-based probabilities
        counts = counts ** (1.0 / max(temp, 1e-10))  # Avoid division by zero
        total = counts.sum()
        if total > 0:
            probs = counts / total
        else:
            idxs = np.nonzero(valid)[0]
            probs = np.zeros_like(counts)
            if len(idxs) > 0:
                probs[idxs] = 1.0 / len(idxs)

        # LOGGING
        if selfplay and board is not None and player is not None:
            action = np.argmax(probs)
            uci = self.game.index_to_uci_move(action)
            policy_output = self.nnet(
                torch.tensor(self.game.encode_board(state), dtype=torch.float32, device=self.cfg.device).unsqueeze(0)
            )[0].cpu().detach().numpy().flatten()
            move_row = {
                "move_idx": int(action),
                "uci": uci,
                "policy": probs.tolist(),
                "policy_output": policy_output.tolist(),
                "value": float(self.Qsa.get((s, action), 0)),
                "player": player,
                "fen": state.fen() if hasattr(state, "fen") else "N/A",
                "move_number": move_number if move_number is not None else 0,
                "visits": int(counts[action])  # Actual visit count for this move
            }
            self.move_log.append(move_row)

        return probs

    def update_root(self, action):
        """Update root state after a move is played"""
        if self.root_state is None:
            return
            
        # Update root state
        self.root_state = self.game.get_next_state(self.root_state, action)
        
        # Prune the tree to save memory and keep relevant parts
        if getattr(self.cfg, 'prune_tree', True):
            new_Qsa = {}
            new_Nsa = {}
            new_Ns = {}
            
            # Get new root state string
            s_new = self.game.string_representation(self.root_state)
            
            # Keep only children of new root
            for (s, a), v in self.Qsa.items():
                if s == s_new:
                    new_Qsa[(s, a)] = v
            for (s, a), v in self.Nsa.items():
                if s == s_new:
                    new_Nsa[(s, a)] = v
                    
            for s, v in self.Ns.items():
                if s == s_new:
                    new_Ns[s] = v
                    
            # Update dictionaries
            self.Qsa = new_Qsa
            self.Nsa = new_Nsa
            self.Ns = new_Ns

    def reset_log(self):
        self.move_log = []

    def get_backpropagation_info(self):
        info = {
            "Qsa": {str(k): v for k, v in self.Qsa.items()},
            "Nsa": {str(k): v for k, v in self.Nsa.items()},
            "Ns": {str(k): v for k, v in self.Ns.items()},
            "nodes": getattr(self, "nodes_expanded", "N/A"),
            "max_depth": getattr(self, "max_depth", "N/A"),
        }
        return info