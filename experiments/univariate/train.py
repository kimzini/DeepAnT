import numpy as np
import torch
import torch.nn as nn

def train(predictor, train_loader, val_loader, device, epochs, lr):
    predictor.to(device)

    # Loss Function
    criterion = nn.L1Loss()

    # 가중치 업데이트
    optim = torch.optim.Adam(predictor.parameters(), lr=lr)

    best_val = None
    best_state = None

    for ep in range(epochs):
        # train
        tr_losses = []

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            y_hat = predictor(x)
            loss = criterion(y_hat, y)

            optim.zero_grad()
            loss.backward()
            optim.step()

            tr_losses.append(loss.item())

        # validation
        va_losses = []
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)
                y = y.to(device)
                va_losses.append(criterion(predictor(x), y).item())

        tr_mean = float(np.mean(tr_losses))
        va_mean = float(np.mean(va_losses))

        print(f"[Epoch {ep}] train_loss={tr_mean:.6f}  val_loss={va_mean:.6f}")
        
        if best_val is None or va_mean < best_val:
            best_val = va_mean
            best_state = {k: v.detach().cpu().clone() for k, v in predictor.state_dict().items()}
        
    if best_state is not None:
        predictor.load_state_dict(best_state)

    return predictor