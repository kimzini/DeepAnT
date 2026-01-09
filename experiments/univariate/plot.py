import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot(
    timestamps,
    series,
    w,
    split_idx: int,
    pred_series,
    detected_idx,
    out_path,
):
    ts = pd.to_datetime(timestamps)
    y = np.asarray(series, dtype=np.float32)
    y_hat = np.asarray(pred_series, dtype=np.float32)

    detected_idx = np.asarray(detected_idx if detected_idx is not None else [], dtype=np.int64)

    fig, ax = plt.subplots(figsize=(16, 5))

    # Actual
    ax.plot(ts[:split_idx], y[:split_idx], color="gold", linewidth=1.2, label="Actual (Train+Val)")
    ax.plot(ts[split_idx:], y[split_idx:], color="crimson", linewidth=1.2, label="Actual (Test)")

    # Prediction (train+val)
    ax.plot(ts[:split_idx], y_hat[:split_idx], color="royalblue", linewidth=1.1, label="Prediction")

    # Prediction (test)
    ax.plot(ts[split_idx + w:], y_hat[split_idx + w:], color="royalblue", linewidth=1.1, label=None)

    # Detected
    if detected_idx.size > 0:
        detected_idx = detected_idx[(detected_idx >= 0) & (detected_idx < len(ts))]
        first = True
        for i in detected_idx:
            ax.axvline(
                ts[i],
                color="gray",
                linestyle="--",
                linewidth=0.9,
                label="Detected" if first else None,
            )
            first = False

    ax.grid(True, alpha=0.25)
    ax.set_xlabel("Time")
    ax.set_ylabel("Value")
    ax.legend(loc="best")

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)