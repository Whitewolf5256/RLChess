# Updated training function with learning rate annealing and better diagnostics
import torch
import time
import random
import numpy as np
from utils.utils import apply_symmetries
from model.loss import compute_loss

def train(model, buffer, optimizer, cfg):
    """
    Train the model using data from the buffer.
    Supports GPU acceleration, batch training with augmentation, and entropy diagnostics.
    Returns average loss over all updates.
    """
    device = cfg.device
    model.train()

    # Compute number of batches to run
    ntotal = len(buffer) // cfg.batch_size
    n_updates = ntotal

    if n_updates == 0:
        print("[Training] Not enough samples for training.")
        return 0.0

    total_loss = 0.0
    entropy_history = []
    value_ranges = []
    policy_distributions = []

    # Track policy distribution statistics for a subset of batches
    track_distribution = max(1, n_updates // 10)

    for i in range(n_updates):
        # Sample a balanced batch from the buffer
        raw_batch = buffer.sample_balanced(cfg.batch_size)

        # Pad or trim if needed
        if len(raw_batch) > cfg.batch_size:
            batch = random.sample(raw_batch, cfg.batch_size)
        elif len(raw_batch) < cfg.batch_size:
            batch = raw_batch
        else:
            batch = raw_batch

        # Unpack 5-tuple: state, pi, z
        states, pis, zs = zip(*batch)

        # Data augmentation (e.g., symmetries)
        if cfg.use_symmetries:
            states, pis, zs = apply_symmetries(states, pis, zs)

        # Convert to torch tensors and move to GPU
        states = torch.stack([
            s if isinstance(s, torch.Tensor) else torch.tensor(s, dtype=torch.float32)
            for s in states
        ]).to(device)

        pis = torch.stack([
            p if isinstance(p, torch.Tensor) else torch.tensor(p, dtype=torch.float32)
            for p in pis
        ]).to(device)

        zs = torch.tensor(zs, dtype=torch.float32, device=device).view(-1, 1)

        # Forward pass
        pred_p, pred_v = model(states)

        # Mask invalid entries (0 in target π) to stabilize training
        invalid_mask = (pis != 0).float()

        # Compute loss with entropy diagnostics
        loss, policy_loss, value_loss, entropy, v_range = compute_loss(
            pred_p, pred_v, pis, zs, invalid_mask, cfg
        )

        # Store diagnostics
        entropy_history.append(entropy)
        value_ranges.append(v_range)

        # Track policy distribution characteristics
        if i % track_distribution == 0:
            with torch.no_grad():
                # Apply temperature to logits
                tempered_logits = pred_p / cfg.temperature
                probs = torch.softmax(tempered_logits, dim=1)
                # Compute statistics for top actions
                top_probs, _ = torch.topk(probs, k=5, dim=1)
                policy_distributions.append({
                    'max_prob': top_probs[:, 0].mean().item(),
                    'top5_diff': (top_probs[:, 0] - top_probs[:, -1]).mean().item()
                })

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # Reporting
        if i % 100 == 0 or i == n_updates - 1:
            avg_entropy = sum(entropy_history[-100:]) / len(entropy_history[-100:])
            avg_vrange = sum(value_ranges[-100:]) / len(value_ranges[-100:])
            print(f"[{time.strftime('%H:%M:%S')}] Batch {i+1}/{n_updates} → "
                  f"Total: {loss.item():.4f}, Policy loss: {policy_loss:.4f}, Value loss: {value_loss:.4f}, "
                  f"Entropy: {avg_entropy:.3f} (curr {entropy:.3f}), Value Range: {avg_vrange:.3f}")

    # Policy distribution analysis
    if policy_distributions:
        avg_max_prob = sum(d['max_prob'] for d in policy_distributions) / len(policy_distributions)
        avg_top5_diff = sum(d['top5_diff'] for d in policy_distributions) / len(policy_distributions)
        print(f"Policy distribution: Avg max prob: {avg_max_prob:.3f}, Avg top5 diff: {avg_top5_diff:.3f}")
        
        # Alert if policy is becoming too deterministic
        if avg_max_prob > 0.8:
            print("WARNING: Policy becoming too deterministic. Consider increasing entropy coefficient or temperature.")
        elif avg_max_prob < 0.3:
            print("NOTE: Policy is very exploratory. Consider reducing entropy coefficient slightly.")

    # Final diagnostics
    avg_entropy = sum(entropy_history) / len(entropy_history)
    print(f"Policy entropy: {avg_entropy:.3f} (target >1.0)")
    if avg_entropy < 0.8:
        print("Warning: Low policy entropy - consider increasing entropy coefficient")
    elif avg_entropy > 3.0:
        print("Note: Very high policy entropy - consider decreasing entropy coefficient slightly")

    # Learning rate annealing
    for param_group in optimizer.param_groups:
        old_lr = param_group['lr']
        new_lr = max(cfg.min_learning_rate, old_lr * cfg.annealing_factor)
        param_group['lr'] = new_lr
        print(f"Learning rate annealed: {old_lr:.6f} → {new_lr:.6f}")

    avg_loss = total_loss / n_updates
    print(f"[{time.strftime('%H:%M:%S')}] Training complete. AvgLoss = {avg_loss:.4f}")
    return avg_loss


# Updated loss computation with better entropy handling
import torch.nn.functional as F
import torch

def compute_loss(pred_p, pred_v, target_p, target_v, invalid_mask, cfg):
    """
    Compute loss with entropy regularization and policy diagnostics
    """
    # Shape validation
    if invalid_mask.shape != pred_p.shape:
        raise ValueError(f"Shape mismatch: invalid_mask {invalid_mask.shape}, pred_p {pred_p.shape}")

    # Apply temperature scaling to logits for better exploration
    pred_p = pred_p / getattr(cfg, 'temperature', 1.0)
    
    # Mask invalid moves before softmax
    pred_p = pred_p.masked_fill(invalid_mask == 0, -1e9)

    # Policy calculations
    probs = F.softmax(pred_p, dim=1)
    log_probs = F.log_softmax(pred_p, dim=1)
    
    # Policy loss (cross-entropy)
    loss_p = -(target_p * log_probs).sum(dim=1).mean()
    
    # Value loss (MSE)
    loss_v = F.mse_loss(pred_v, target_v)
    
    # Entropy regularization - using a more stable formula
    entropy = -(probs * log_probs).sum(dim=1).mean()
    
    # Invalid move penalty
    penalty = cfg.nonvalidity_penalty * (invalid_mask.numel() - invalid_mask.sum()) / invalid_mask.numel()
    
    # Total loss with entropy component
    # Higher coefficient for entropy regularization to prevent premature convergence
    entropy_term = cfg.entropy_coeff * entropy
    total_loss = loss_p + loss_v + penalty - entropy_term
    
    # Diagnostic metrics
    with torch.no_grad():
        policy_entropy = entropy.item()
        value_range = pred_v.max().item() - pred_v.min().item()
    
    return total_loss, loss_p.item(), loss_v.item(), policy_entropy, value_range
