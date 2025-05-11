import torch
import time
import random
from utils.utils import apply_symmetries
from model.loss import compute_loss
import torch.nn.functional as F


def train(model, buffer, optimizer, cfg):
    """
    Perform training on `model` using samples from `buffer` according to `cfg`.
    Returns average loss over all updates.
    """
    # Compute how many update steps to run
    ntotal = len(buffer) // cfg.batch_size
    n_updates = min(cfg.max_batches_per_checkpoint,
                    ntotal // cfg.min_checkpoints_per_epoch)
    total_loss = 0.0

    for i in range(n_updates):
        batch = buffer.sample_balanced(cfg.batch_size)

        # If too many, downsample; if too few, pad with draws or positive
        if len(batch) > cfg.batch_size:
            batch = random.sample(batch, cfg.batch_size)
        elif len(batch) < cfg.batch_size:
            # pad from raw_batch
            pad = [ex for ex in raw_batch if ex not in batch]
            needed = cfg.batch_size - len(batch)
            batch += pad[:needed]

        # Unpack 5-tuple: state, pi, z, t_rem, is_white
        states, pis, zs, ts, is_white = zip(*batch)

        # Count outcomes
        white_win_count = sum(1 for z, is_w in zip(zs, is_white) if z == 1 and is_w)
        black_win_count = sum(1 for z, is_w in zip(zs, is_white) if z == 1 and not is_w)
        tie_count = sum(1 for z in zs if z == 0)

        print(f"Batch composition: {white_win_count} white wins, {black_win_count} black wins, {tie_count} draws")

        # Data augmentation (symmetries)
        if cfg.use_symmetries:
            states, pis, zs = apply_symmetries(states, pis, zs)

        # Move to device
        device = cfg.device
        states = torch.stack([
            torch.tensor(s) if not isinstance(s, torch.Tensor) else s
            for s in states
        ]).to(device)
        pis = torch.stack([
            torch.tensor(p, dtype=torch.float32) if not isinstance(p, torch.Tensor) else p
            for p in pis
        ]).to(device)
        zs = torch.tensor(zs, dtype=torch.float32).view(-1, 1).to(device)

        # Forward pass
        pred_p, pred_v = model(states)

        # Compute loss
        invalid_mask = (pis != 0).float()
        loss, policy_loss, value_loss = compute_loss(
            pred_p, pred_v, pis, zs, invalid_mask, cfg
        )

        # Backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if i % 100 == 0:
            print(f"[{time.strftime('%H:%M:%S')}] Batch {i+1}/{n_updates} - "
                  f"TotalLoss: {loss.item():.4f}, PolicyLoss: {policy_loss:.4f}, ValueLoss: {value_loss:.4f}")

    avg_loss = total_loss / max(n_updates, 1)
    print(f"[{time.strftime('%H:%M:%S')}] [Training] Updates: {n_updates}, AvgLoss: {avg_loss:.4f}")
    return avg_loss
