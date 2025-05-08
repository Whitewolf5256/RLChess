def compute_loss(pred_p, pred_v, target_p, target_v, invalid_mask, cfg):
    # Ensure invalid_mask matches pred_p shape
    if invalid_mask.shape != pred_p.shape:
        raise ValueError(f"Shape mismatch: invalid_mask {invalid_mask.shape}, pred_p {pred_p.shape}")

    # Mask invalid moves before softmax
    pred_p = pred_p.masked_fill(invalid_mask == 0, -1e9)

    # Policy loss: cross-entropy between predicted log-probs and target distribution
    log_probs = F.log_softmax(pred_p, dim=1)
    loss_p = -(target_p * log_probs).sum(dim=1).mean()

    # Value loss: MSE between predicted and target values
    loss_v = F.mse_loss(pred_v, target_v)

    # Optional penalty for invalid masks
    penalty = cfg.nonvalidity_penalty * (invalid_mask.numel() - invalid_mask.sum()) / invalid_mask.numel()
    total = loss_p + loss_v + penalty
    return total, loss_p.item(), loss_v.item()
