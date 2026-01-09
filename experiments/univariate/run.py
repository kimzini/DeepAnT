import numpy as np
import torch

from model.predictor import TimeSeriesPredictor, PredictorParams
from model.detector import AnomalyDetecter

from .data import load, split, loaders
from .train import train
from .plot import plot


def collect_scores(detector: AnomalyDetecter, loader, device) -> np.ndarray:
    scores = []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            s = detector.score(x, y).detach().cpu().numpy()
            scores.append(s)
    return np.concatenate(scores, axis=0)


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
            y_hat_scalar = float(y_hat[0, 0, 0].detach().cpu().numpy())

            pred_at = start_offset + i + w
            if 0 <= pred_at < len(pred_series):
                pred_series[pred_at] = y_hat_scalar


def run(
    csv_path="data/TravelTime_451.csv",
    p_w=1,
    conv_kernel_size=5,
    epochs=50,
    lr=1e-3,
    batch_size=16,
    train_ratio=0.4,
    val_ratio=0.1,
    w_candidates=(5, 10, 15, 20, 25, 30),
    k=2.0,  #k-sigma
    out_path="experiments/univariate/images/result_plot.png",
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ts, values = load(csv_path)

    train_vals, val_vals, test_vals, n_train, min_val, max_val = split(
        values, train_ratio=train_ratio, val_ratio=val_ratio
    )

    split_idx = len(train_vals) + len(val_vals)

    full_scaled = np.concatenate([train_vals, val_vals, test_vals], axis=0)
    pred_series = np.full(len(full_scaled), np.nan, dtype=np.float32)

    best = {
        "val_score_mean": float("inf"),
        "w": None,
        "predictor_state": None,
        "thr": None,
    }

    # w 선택
    w_list = w_candidates if isinstance(w_candidates, (list, tuple)) else [w_candidates]

    for w in w_list:
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

        val_scores = collect_scores(detector, val_loader, device)

        val_mean = float(np.mean(val_scores))
        sigma = float(np.std(val_scores))
        thr = val_mean + float(k) * sigma

        val_score_mean = val_mean

        if val_score_mean < best["val_score_mean"]:
            best["val_score_mean"] = val_score_mean
            best["w"] = int(w)
            best["thr"] = float(thr)
            best["mean"] = float(val_mean)
            best["sigma"] = float(sigma)
            best["predictor_state"] = {k: v.detach().cpu().clone() for k, v in predictor.state_dict().items()}

    w = best["w"]
    thr = best["thr"]

    # best predictor 복원
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

    pred_series[split_idx: split_idx + w] = np.nan

    trainval_segment = np.concatenate([train_vals, val_vals], axis=0)
    predict_segment(
        predictor=predictor,
        segment_vals=trainval_segment,
        start_offset=0,
        w=w,
        device=device,
        pred_series=pred_series,
    )

    predict_segment(
        predictor=predictor,
        segment_vals=test_vals,
        start_offset=split_idx,
        w=w,
        device=device,
        pred_series=pred_series,
    )

    err_series = np.full(len(full_scaled), np.nan, dtype=np.float32)
    valid = ~np.isnan(pred_series)
    err_series[valid] = np.abs(pred_series[valid] - full_scaled[valid])

    err_series[split_idx: split_idx + w] = np.nan

    detected_idx = []
    test_valid_mask = valid.copy()
    test_valid_mask[:split_idx] = False
    test_err = err_series[test_valid_mask]
    test_idx = np.where(test_valid_mask)[0]

    detected_idx = test_idx[np.where(test_err >= thr)[0]].tolist()

    print(f"best_w={w}, thr={thr:.6f}")

    plot(
        timestamps=ts,
        series=full_scaled,
        w=w,
        split_idx=split_idx,
        pred_series=pred_series,
        detected_idx=np.asarray(detected_idx, dtype=np.int64),
        out_path=out_path,
    )