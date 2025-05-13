import torch
import time
import random
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

    # Final diagnostics
    avg_entropy = sum(entropy_history) / len(entropy_history)
    print(f"Policy entropy: {avg_entropy:.3f} (target >1.0)")
    if avg_entropy < 0.5:
        print("Warning: Low policy entropy - consider increasing entropy coefficient")

    avg_loss = total_loss / n_updates
    print(f"[{time.strftime('%H:%M:%S')}] Training complete. AvgLoss = {avg_loss:.4f}")
    return avg_loss
