import torch
import time
import random
import torch.nn.functional as F
from utils.utils import apply_symmetries
from model.loss import compute_loss

def train(model, buffer, optimizer, cfg):
    """
    Train the model using data from the buffer.
    Supports GPU acceleration and batch training with augmentation.
    Returns average loss over all updates.
    """
    device = cfg.device
    model.train()

    # Compute number of batches to run
    ntotal = len(buffer) // cfg.batch_size
    n_updates = min(cfg.max_batches_per_checkpoint,
                    ntotal // cfg.min_checkpoints_per_epoch)
    if n_updates == 0:
        print("[Training] Not enough samples for training.")
        return 0.0

    total_loss = 0.0

    for i in range(n_updates):
        raw_batch = buffer.sample_balanced(cfg.batch_size)

        # Pad or trim if needed
        if len(raw_batch) > cfg.batch_size:
            batch = random.sample(raw_batch, cfg.batch_size)
        elif len(raw_batch) < cfg.batch_size:
            batch = raw_batch
        else:
            batch = raw_batch

        # Unpack 5-tuple: state, pi, z, t_rem, is_white
        states, pis, zs, ts, is_white = zip(*batch)

        # Debug info
        # white_win_count = sum(1 for z, is_w in zip(zs, is_white) if (z == 1 and is_w) or (z == -1 and not is_w))
        # black_win_count = sum(1 for z, is_w in zip(zs, is_white) if (z == 1 and not is_w) or (z == -1 and is_w))
        # tie_count = sum(1 for z in zs if z == 0)
        # print(f"Batch {i+1}/{n_updates} → {white_win_count}W / {black_win_count}B / {tie_count}D")

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

        # Compute loss
        loss, policy_loss, value_loss = compute_loss(pred_p, pred_v, pis, zs, invalid_mask, cfg)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if i % 100 == 0 or i == n_updates - 1:
            print(f"[{time.strftime('%H:%M:%S')}] Batch {i+1}/{n_updates} → "
                  f"Total: {loss.item():.4f}, Policy: {policy_loss:.4f}, Value: {value_loss:.4f}")

    avg_loss = total_loss / n_updates
    print(f"[{time.strftime('%H:%M:%S')}] Training complete. AvgLoss = {avg_loss:.4f}")
    return avg_loss
