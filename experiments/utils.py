import numpy as np
import torch
import torch.nn as nn

from model.detector import AnomalyDetecter

@torch.no_grad()
def collect_anomaly_scores(
    detector: AnomalyDetecter,
    loader,
    device: torch.device
) -> np.ndarray:
    anomaly_scores = []
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        s = detector.score(x, y).detach().cpu().numpy()
        anomaly_scores.append(s)
    return np.concatenate(anomaly_scores, axis=0)

@torch.no_grad()
def compute_val_loss(
    predictor: nn.Module,
    loader,
    device: torch.device,
    criterion: nn.Module,
) -> float:
    total = 0
    n_batches = 0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        y_hat = predictor(x)
        total += float(criterion(y_hat, y).item())
        n_batches += 1
    return total / max(n_batches, 1)

@torch.no_grad()
def predict_segment(
    predictor,
    segment_vals: np.ndarray,
    start_offset: int,
    w: int,
    device,
    pred_series: np.ndarray,
):
    for i in range(0, len(segment_vals) - w):
        x = segment_vals[i : i + w]
        x_t = torch.tensor(x, dtype=torch.float32).view(1, 1, w).to(device)
        y_hat = predictor(x_t)  # (1,1,1) when p_w=1
        pred_series[start_offset + i + w] = float(y_hat[0, 0, 0].cpu().numpy())
