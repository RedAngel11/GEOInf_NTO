import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
import sys
import os
import joblib


class LSTMNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(1, 64, 2, dropout=0.2, batch_first=True)
        self.fc = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.2), nn.Linear(32, 1))

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


def forecast_future(model, seq, hours, scaler):
    model.eval()
    seq = seq.copy()
    preds = []
    with torch.no_grad():
        for _ in range(hours):
            inp = torch.FloatTensor(seq).unsqueeze(0)
            next_val = model(inp).numpy()[0, 0]
            preds.append(next_val)
            seq = np.roll(seq, -1, axis=0)
            seq[-1] = next_val
    return scaler.inverse_transform(np.array(preds).reshape(-1, 1))


def predict_mode():
    if not os.path.exists('best_model.pth'):
        print("Модель не найдена. Запустите python ai.py --train")
        return

    df = pd.read_csv("kiyazevo_iot_realistic.csv", parse_dates=["timestamp"]).set_index("timestamp")
    data = df[["ec_microsiemens"]].values.astype(np.float32)

    scaler = joblib.load('scaler.save')
    data_norm = scaler.transform(data)

    model = LSTMNet()
    model.load_state_dict(torch.load('best_model.pth'))
    model.eval()

    last_seq = data_norm[-24:].copy()
    forecast = forecast_future(model, last_seq, 72, scaler)

    last_ts = df.index[-1]
    forecast_ts = pd.date_range(start=last_ts + pd.Timedelta(hours=1), periods=72, freq='h')

    forecast_df = pd.DataFrame({'timestamp': forecast_ts, 'ec_forecast': forecast.flatten()})
    forecast_df.to_csv('forecast_72h.csv', index=False)

    plt.figure(figsize=(12, 6))
    plt.plot(df.index[-168:], data[-168:], label='История', linewidth=1.5)
    plt.plot(forecast_ts, forecast, 'r-', label='Прогноз 72ч', linewidth=2)
    plt.axvline(x=last_ts, color='g', linestyle='--', alpha=0.7)
    plt.title('Прогноз электропроводности воды')
    plt.ylabel('EC (мкСм/см)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('forecast.png', dpi=120)
    plt.show()

    print(f"Прогноз сохранён в forecast_72h.csv")


def train_mode():
    df = pd.read_csv("kiyazevo_iot_realistic.csv", parse_dates=["timestamp"]).set_index("timestamp")
    data = df[["ec_microsiemens"]].values.astype(np.float32)

    scaler = StandardScaler()
    data_norm = scaler.fit_transform(data)
    joblib.dump(scaler, 'scaler.save')

    SEQ = 24
    X, y = [], []
    for i in range(len(data_norm) - SEQ):
        X.append(data_norm[i:i + SEQ])
        y.append(data_norm[i + SEQ])
    X, y = np.array(X), np.array(y)

    split = int(0.8 * len(X))

    train_loader = DataLoader(TensorDataset(torch.FloatTensor(X[:split]), torch.FloatTensor(y[:split])),
                              batch_size=32, shuffle=True)
    val_loader = DataLoader(TensorDataset(torch.FloatTensor(X[split:]), torch.FloatTensor(y[split:])),
                            batch_size=32)

    model = LSTMNet()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    best_val = float('inf')
    patience = 0

    print("Обучение...")
    for epoch in range(200):
        model.train()
        train_loss = 0
        for Xb, yb in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(Xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for Xb, yb in val_loader:
                val_loss += criterion(model(Xb), yb).item()
        val_loss /= len(val_loader)

        if val_loss < best_val:
            best_val = val_loss
            patience = 0
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            patience += 1
            if patience >= 30:
                break

        if (epoch + 1) % 50 == 0:
            print(f"Epoch {epoch + 1}: train={train_loss:.6f}, val={val_loss:.6f}")

    model.eval()
    preds, actuals = [], []
    with torch.no_grad():
        for Xb, yb in val_loader:
            preds.extend(model(Xb).numpy().flatten())
            actuals.extend(yb.numpy().flatten())

    preds = scaler.inverse_transform(np.array(preds).reshape(-1, 1))
    actuals = scaler.inverse_transform(np.array(actuals).reshape(-1, 1))

    mae = mean_absolute_error(actuals, preds)
    r2 = r2_score(actuals, preds)
    mape = np.mean(np.abs((actuals - preds) / actuals)) * 100

    print(f"\nMAE: {mae:.1f} мкСм/см")
    print(f"R²: {r2:.4f}")
    print(f"MAPE: {mape:.2f}%")
    print(f"\nГотово! Запустите: python ai.py --predict")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == '--predict':
        predict_mode()
    else:
        train_mode()