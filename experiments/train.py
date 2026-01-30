import numpy as np
import torch
import torch.nn as nn

def train(predictor, train_loader, val_loader, device, epochs, lr):
    predictor.to(device)

    # Loss Function
    criterion = nn.L1Loss()

    # 가중치 업데이트
    optimizer = torch.optim.Adam(predictor.parameters(), lr=lr)

    best_val_loss = float("inf")
    best_state = None

    for epoch in range(epochs):
        # train
        tr_losses = []

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            y_hat = predictor(x)
            loss = criterion(y_hat, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            tr_losses.append(loss.item())

        # validation
        val_losses = []
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)
                y = y.to(device)
                val_losses.append(criterion(predictor(x), y).item())

        train_loss = float(np.mean(tr_losses)) if tr_losses else float("nan")
        val_loss = float(np.mean(val_losses)) if val_losses else float("nan")

        print(f"[Epoch {epoch}] train_loss={train_loss:.6f}  val_loss={val_loss:.6f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in predictor.state_dict().items()}

    if best_state is not None:
        predictor.load_state_dict(best_state)

    return predictor