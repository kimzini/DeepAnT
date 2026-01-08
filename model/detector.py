import numpy as np
import torch
import torch.nn as nn

class AnomalyDetecter:
    def __init__(self, predictor: nn.Module):
        self.predictor = predictor

    def score(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        y_hat = self.predictor(x)
        diff = y_hat - y
        return torch.sqrt(torch.sum(diff * diff, dim=(1, 2)))

    def all_scores(self, loader, device):
        device = torch.device(device)
        self.predictor.to(device)

        scores = []
        with torch.no_grad():
            for x, y in loader:
                x = x.to(device)
                y = y.to(device)
                s = self.score(x, y)
                scores.append(s.cpu())
        return torch.cat(scores, dim=0).numpy()

    def detect(self, loader, device, threshold):
        scores = self.all_scores(loader, device)
        is_anomaly = scores >= threshold
        return scores, is_anomaly