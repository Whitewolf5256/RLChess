
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
