import torch
import time
from utils.utils import apply_symmetries
from model.loss import compute_loss
import torch.nn.functional as F

def train(model, buffer, optimizer, cfg):
    """
    Perform training on `model` using samples from `buffer` according to `cfg`.
    Returns average loss over all updates.
    """
    ntotal = len(buffer) // cfg.batch_size
    n_updates = min(cfg.max_batches_per_checkpoint,
                    ntotal // cfg.min_checkpoints_per_epoch)
    total_loss = 0.0

    for i in range(n_updates):
        batch = buffer.sample(cfg.batch_size)
        states, pis, zs, ts = zip(*batch)

        # Data augmentation
        if cfg.use_symmetries:
            states, pis, zs = apply_symmetries(states, pis, zs)

        # Move to device
        device = cfg.device
        states = torch.stack([torch.tensor(s) if not isinstance(s, torch.Tensor) else s for s in states]).to(device)
        pis = torch.stack([torch.tensor(p, dtype=torch.float32) if not isinstance(p, torch.Tensor) else p for p in pis]).to(device)
        zs = torch.tensor(zs, dtype=torch.float32).view(-1, 1).to(device)

        # Compute loss
        pred_p, pred_v = model(states)
        invalid_mask = (pis != 0).float()
        loss, policy_loss, value_loss = compute_loss(pred_p, pred_v, pis, zs, invalid_mask, cfg)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if i % 100 == 0:
            print(f"[{time.strftime('%H:%M:%S')}] Batch {i+1}/{n_updates} - TotalLoss: {loss.item():.4f}, PolicyLoss: {policy_loss:.4f}, ValueLoss: {value_loss:.4f}")

    avg_loss = total_loss / max(n_updates, 1)
    print(f"[{time.strftime('%H:%M:%S')}] [Training] Updates: {n_updates}, AvgLoss: {avg_loss:.4f}")
    return avg_loss
