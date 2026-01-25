import os
import random
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, precision_score, recall_score

from model.predictor import TimeSeriesPredictor, PredictorParams
from model.detector import AnomalyDetecter

from .data import load, split, loaders, create_anomaly_labels, load_anomaly_windows
from .train import train
from .plot import plot

def set_seed(seed: int = 42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def collect_anomaly_scores(detector: AnomalyDetecter, loader, device) -> np.ndarray:
    anomaly_scores = []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            s = detector.score(x, y).detach().cpu().numpy()
            anomaly_scores.append(s)
    return np.concatenate(anomaly_scores, axis=0)


def predict_segment(
    predictor,
    segment_vals: np.ndarray,
    start_offset: int,
    w: int,
    device,
    pred_series: np.ndarray,
):
    with torch.no_grad():
        for i in range(0, len(segment_vals) - w):
            x = segment_vals[i : i + w]
            x_t = torch.tensor(x, dtype=torch.float32).view(1, 1, w).to(device)
            y_hat = predictor(x_t)  # (1,1,1) when p_w=1
            pred_series[start_offset + i + w] = float(y_hat[0, 0, 0].cpu().numpy())



def run(
    csv_path="data/TravelTime_387.csv",
    seed=42,
    p_w=1,
    conv_kernel_size=5,
    epochs=50,
    lr=1e-3,
    batch_size=16,
    train_ratio=0.4,
    val_ratio=0.1,
    w_candidates=(5, 7, 10, 12, 15, 30),
    out_path=None,
    anomaly_windows_json="data/label/combined_windows.json",
):
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if out_path is None:
        csv_filename = csv_path.split('/')[-1].replace('.csv', '')
        out_path = f"experiments/images/{csv_filename}_plot.png"

    ts, values = load(csv_path)
    train_vals, val_vals, test_vals, _, _, _ = split(
        values, train_ratio=train_ratio, val_ratio=val_ratio
    )
    
    # 이상 라벨 생성
    full_labels = create_anomaly_labels(ts, csv_path, json_path=anomaly_windows_json, start_idx=0)
    anomaly_windows = load_anomaly_windows(csv_path, anomaly_windows_json)
    
    # train/val/test로 분할
    split_idx = len(train_vals) + len(val_vals)
    test_labels = full_labels[split_idx:]

    full_series = np.concatenate([train_vals, val_vals, test_vals])
    pred_series = np.full(len(full_series), np.nan, dtype=np.float32)

    best = {
        "val_loss": float("inf"),
        "w": None,
        "predictor_state": None,
        "thr": None,
    }

    criterion = nn.L1Loss()

    for w in w_candidates:
        train_loader, val_loader = loaders(train_vals, val_vals, w=w, p_w=p_w, batch_size=batch_size)

        cfg = PredictorParams(
            w=w,
            conv_kernel_size=conv_kernel_size,
            p_w=p_w,
            feature_dim=1,
            conv_channels=32,
            pool_kernel_size=2,
        )

        predictor = TimeSeriesPredictor(cfg)
        predictor = train(predictor, train_loader, val_loader, device, epochs=epochs, lr=lr)

        detector = AnomalyDetecter(predictor)

        val_scores = collect_anomaly_scores(detector, val_loader, device)
        mu = np.mean(val_scores)
        sigma = np.std(val_scores)
        k = 4
        thr = mu + k * sigma
        
        val_loss = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)
                y = y.to(device)
                y_hat = predictor(x)
                val_loss += criterion(y_hat, y).item()
        val_loss /= len(val_loader)
        
        print(
            f"[w={w}], val_loss={val_loss:.6f}, thr={thr:.6f}, mu={mu:.6f}, sigma={sigma:.6f}"
        )

        if val_loss < best["val_loss"]:
            best.update({
                "val_loss": val_loss,
                "w": w,
                "thr": thr,
                "predictor_state": {k: v.cpu().clone() for k, v in predictor.state_dict().items()},
            })

    w = best["w"]
    thr = best["thr"]

    cfg = PredictorParams(
        w=w,
        conv_kernel_size=conv_kernel_size,
        p_w=p_w,
        feature_dim=1,
        conv_channels=32,
        pool_kernel_size=2,
    )
    predictor = TimeSeriesPredictor(cfg)
    predictor.load_state_dict(best["predictor_state"])
    predictor.to(device)
    predictor.eval()

    trainval = np.concatenate([train_vals, val_vals])
    predict_segment(predictor, trainval, 0, w, device, pred_series)
    predict_segment(predictor, test_vals, split_idx, w, device, pred_series)

    err = np.abs(pred_series - full_series)
    valid = ~np.isnan(pred_series)
    test_mask = valid.copy()
    test_mask[:split_idx] = False

    test_idx = np.where(test_mask)[0]
    detected_idx = test_idx[err[test_mask] >= thr].tolist()

    test_pred = np.zeros(len(test_labels), dtype=bool)
    for idx in detected_idx:
        test_pred[idx - split_idx] = True

    eval_mask = np.ones(len(test_labels), dtype=bool)
    eval_mask[:w] = False

    test_true = test_labels[eval_mask]
    test_pred_eval = test_pred[eval_mask]
    
    test_precision = precision_score(test_true, test_pred_eval, zero_division=0)
    test_recall = recall_score(test_true, test_pred_eval, zero_division=0)
    test_f1 = f1_score(test_true, test_pred_eval, zero_division=0)

    split_time = ts[split_idx]

    test_anomaly_windows = []
    for (start, end) in anomaly_windows:
        if np.datetime64(end) >= np.datetime64(split_time):
            test_anomaly_windows.append((start, end))

    print(f"best_w={w}, thr={thr:.6f}, val_loss={best['val_loss']:.6f}")
    print(f"test_precision={test_precision:.6f}, test_recall={test_recall:.6f}, test_f1={test_f1:.6f}")

    plot(
        timestamps=ts,
        series=full_series,
        w=w,
        split_idx=split_idx,
        pred_series=pred_series,
        detected_idx=np.array(detected_idx),
        out_path=out_path,
        anomaly_windows=test_anomaly_windows,
    )