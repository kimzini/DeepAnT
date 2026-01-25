import numpy as np
import pandas as pd
import torch
import json
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Dict

class Windows(Dataset):
    def __init__(self, values: np.ndarray, w: int, p_w: int):
        self.v = values
        self.w = w
        self.p_w = p_w

    def __len__(self):
        return len(self.v) - self.w  - self.p_w + 1

    # Sliding Window
    def __getitem__(self, idx):
        x = self.v[idx : idx + self.w]  # 과거 시점 w개
        y = self.v[idx + self.w : idx + self.w + self.p_w]  # 실제값 (정답)
        x = torch.tensor(x, dtype=torch.float32).unsqueeze(0)   # (1, w)
        y = torch.tensor(y, dtype=torch.float32).unsqueeze(0)
        return x, y


def load(csv_path: str):
    df = pd.read_csv(csv_path)

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    timestamp = df["timestamp"].to_numpy()
    values = df["value"].to_numpy().astype(np.float32)

    return timestamp, values


def fit(values: np.ndarray, end: int):
    min_val = float(np.min(values[:end]))
    max_val = float(np.max(values[:end]))
    denom = (max_val - min_val)
    scaled = ((values - min_val) / denom).astype(np.float32)
    return scaled, min_val, max_val


def split(values: np.ndarray, train_ratio, val_ratio):
    n = len(values)
    n_train_val = int(train_ratio * n)
    n_val = int(val_ratio * n_train_val)
    n_train = n_train_val - n_val

    scaled, min_val, max_val = fit(values, end=n_train_val)

    train = scaled[:n_train]
    val = scaled[n_train:n_train_val]
    test = scaled[n_train_val:]

    return train, val, test, n_train, min_val, max_val


def loaders(train, val, w, p_w, batch_size):
    train_loader = DataLoader(Windows(train, w, p_w), batch_size=batch_size, shuffle=True) # (B, 1, w)
    val_loader = DataLoader(Windows(val, w, p_w), batch_size=batch_size, shuffle=False)
    return train_loader, val_loader


# 과거 w개 시점으로 예측하고, 다음 p_w 실제값이 들어오면 score 계산
def slide(test: np.ndarray, w: int, p_w: int):
    for i in range(0, len(test) - w - p_w + 1):
        x = test[i : i + w]
        y = test[i + w : i + w + p_w]
        yield i, x, y


def load_anomaly_windows(csv_path: str, json_path: str) -> List[Tuple[str, str]]:
    with open(json_path, 'r') as f:
        anomaly_windows_dict = json.load(f)
    
    csv_filename = csv_path.split('/')[-1]
    for key, windows in anomaly_windows_dict.items():
        if key.endswith(csv_filename):
            return windows
    
    return []


def create_anomaly_labels(
    timestamps: np.ndarray,
    csv_path: str,
    json_path: str,
    start_idx: int = 0
) -> np.ndarray:
    anomaly_windows = load_anomaly_windows(csv_path, json_path)
    
    labels = np.zeros(len(timestamps), dtype=bool)
    ts_series = pd.to_datetime(timestamps)
    
    for start_str, end_str in anomaly_windows:
        start_time = pd.to_datetime(start_str)
        end_time = pd.to_datetime(end_str)
        
        mask = (ts_series >= start_time) & (ts_series <= end_time)
        labels[mask] = True
    
    return labels