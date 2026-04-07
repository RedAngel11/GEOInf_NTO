"""
Модель LSTM для прогнозирования электропроводности воды (EC) на 72 часа.
Данные с IoT-датчика (р. Яйва), частота измерений – 1 час.
Архитектура: двухслойная LSTM + полносвязная голова.
"""

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

# Определение архитектуры нейросети
class LSTMNet(nn.Module):
    """
    Простая, но эффективная LSTM для временных рядов.
    Вход: (batch, seq_len, 1) – нормализованная EC.
    Выход: одно значение (EC на следующий час).
    """
    def __init__(self):
        super().__init__()
        # Двухслойная LSTM: 1 входной признак, 64 скрытых нейрона, 2 слоя.
        # Dropout 0.2 между слоями – чтобы не переобучаться на маленьком датасете (всего 168 точек).
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=64,
            num_layers=2,
            dropout=0.2,
            batch_first=True
        )
        # Голова-регрессор: сжимаем выход LSTM до одного числа.
        self.fc = nn.Sequential(
            nn.Linear(64, 32),        # первый полносвязный слой
            nn.ReLU(),                # нелинейность
            nn.Dropout(0.2),          # ещё дропаут – регуляризация
            nn.Linear(32, 1)          # итоговый прогноз
        )

    def forward(self, x):
        # x: (batch, seq_len, 1)
        out, _ = self.lstm(x)
        # Берём только последний выход последовательности (самый свежий момент)
        last_out = out[:, -1, :]
        return self.fc(last_out)

# Функция итеративного прогноза на 72 часа
def forecast_future(model, seq, hours, scaler):
    """
    Рекурсивная экстраполяция: модель предсказывает следующий час,
    добавляет его в конец окна, сдвигает окно и повторяет.
    """
    model.eval()
    seq = seq.copy()                  # не портим исходный массив
    preds = []
    with torch.no_grad():             # отключаем градиенты – только forward
        for _ in range(hours):
            # Превращаем окно в батч из одного примера
            inp = torch.FloatTensor(seq).unsqueeze(0)   # (1, seq_len, 1)
            next_val = model(inp).numpy()[0, 0]         # скаляр
            preds.append(next_val)
            # Сдвигаем окно: удаляем самый старый элемент, вставляем новый
            seq = np.roll(seq, -1, axis=0)   # циклический сдвиг влево
            seq[-1] = next_val               # последний элемент заменяем прогнозом
    # Возвращаем прогноз в исходных единицах (мкСм/см) через обратное масштабирование
    return scaler.inverse_transform(np.array(preds).reshape(-1, 1))

# Режим прогноза (загружаем обученную модель и делаем предсказание)

def predict_mode():
    # Проверяем, есть ли сохранённая модель
    if not os.path.exists('best_model.pth'):
        print("Модель не найдена. Запустите python ai.py --train")
        return

    # Загружаем и подготавливаем данные
    df = pd.read_csv("kiyazevo_iot_realistic.csv", parse_dates=["timestamp"]).set_index("timestamp")
    data = df[["ec_microsiemens"]].values.astype(np.float32)

    # Загружаем скейлер (обучен вместе с моделью)
    scaler = joblib.load('scaler.save')
    data_norm = scaler.transform(data)

    # Создаём экземпляр модели и загружаем веса
    model = LSTMNet()
    model.load_state_dict(torch.load('best_model.pth'))
    model.eval()

    # Берём последние 24 часа как начальное окно для прогноза
    last_seq = data_norm[-24:].copy()
    forecast = forecast_future(model, last_seq, 72, scaler)

    # Готовим временные метки для прогноза
    last_ts = df.index[-1]
    forecast_ts = pd.date_range(start=last_ts + pd.Timedelta(hours=1), periods=72, freq='h')

    # Сохраняем результат в CSV (удобно для дальнейшего анализа)
    forecast_df = pd.DataFrame({'timestamp': forecast_ts, 'ec_forecast': forecast.flatten()})
    forecast_df.to_csv('forecast_72h.csv', index=False)

    # Строим график: последние 168 часов истории + прогноз
    plt.figure(figsize=(12, 6))
    plt.plot(df.index[-168:], data[-168:], label='История', linewidth=1.5)
    plt.plot(forecast_ts, forecast, 'r-', label='Прогноз 72ч', linewidth=2)
    plt.axvline(x=last_ts, color='g', linestyle='--', alpha=0.7)   # разделитель прошлое/будущее
    plt.title('Прогноз электропроводности воды (LSTM)')
    plt.ylabel('EC (мкСм/см)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('forecast.png', dpi=120)   # сохраняем картинку
    plt.show()

    print(f"Прогноз сохранён в forecast_72h.csv")

# Режим обучения
def train_mode():
    # --- Загрузка и предобработка данных ---
    df = pd.read_csv("kiyazevo_iot_realistic.csv", parse_dates=["timestamp"]).set_index("timestamp")
    data = df[["ec_microsiemens"]].values.astype(np.float32)

    # Нормализация: StandardScaler (среднее 0, дисперсия 1) – стандарт для LSTM
    scaler = StandardScaler()
    data_norm = scaler.fit_transform(data)
    joblib.dump(scaler, 'scaler.save')   # сохраняем, чтобы при прогнозе сделать обратное преобразование

    # Формируем обучающие пары (окно, цель)
    SEQ = 24                     # окно в 24 часа – охватываем суточный цикл
    X, y = [], []
    for i in range(len(data_norm) - SEQ):
        X.append(data_norm[i:i + SEQ])   # последовательность из SEQ значений
        y.append(data_norm[i + SEQ])     # следующее значение
    X, y = np.array(X), np.array(y)

    # Разделение на train (80%) и validation (20%)
    split = int(0.8 * len(X))

    train_loader = DataLoader(
        TensorDataset(torch.FloatTensor(X[:split]), torch.FloatTensor(y[:split])),
        batch_size=32, shuffle=True       # перемешивание для устойчивости обучения
    )
    val_loader = DataLoader(
        TensorDataset(torch.FloatTensor(X[split:]), torch.FloatTensor(y[split:])),
        batch_size=32
    )

    model = LSTMNet()
    criterion = nn.MSELoss()               # среднеквадратичная ошибка – классика для регрессии
    optimizer = optim.Adam(model.parameters(), lr=0.001)   # Адам с дефолтным lr хорошо стартует

    best_val = float('inf')
    patience = 0                           # счётчик для early stopping

    print("Обучение LSTM модели...")
    for epoch in range(200):               # максимум 200 эпох, но остановимся раньше, если валидация не улучшается
        # --- Тренировочная фаза ---
        model.train()
        train_loss = 0
        for Xb, yb in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(Xb), yb)
            loss.backward()
            # Клиппируем градиенты – защита от взрыва градиента (особенно для LSTM)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        # --- Валидационная фаза ---
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for Xb, yb in val_loader:
                val_loss += criterion(model(Xb), yb).item()
        val_loss /= len(val_loader)

        # Early stopping: сохраняем лучшую модель, если loss уменьшился
        if val_loss < best_val:
            best_val = val_loss
            patience = 0
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            patience += 1
            if patience >= 30:            # если 30 эпох подряд нет улучшения – стоп
                print(f"Ранняя остановка на эпохе {epoch+1}")
                break

        if (epoch + 1) % 50 == 0:
            print(f"Epoch {epoch + 1}: train_loss={train_loss:.6f}, val_loss={val_loss:.6f}")

    # --- Оценка на валидационной выборке в исходных единицах ---
    model.eval()
    preds, actuals = [], []
    with torch.no_grad():
        for Xb, yb in val_loader:
            preds.extend(model(Xb).numpy().flatten())
            actuals.extend(yb.numpy().flatten())

    # Обратное масштабирование
    preds = scaler.inverse_transform(np.array(preds).reshape(-1, 1))
    actuals = scaler.inverse_transform(np.array(actuals).reshape(-1, 1))

    mae = mean_absolute_error(actuals, preds)
    r2 = r2_score(actuals, preds)
    mape = np.mean(np.abs((actuals - preds) / actuals)) * 100

    print(f"\nРезультаты на валидации:")
    print(f"MAE:  {mae:.1f} мкСм/см")
    print(f"R²:   {r2:.4f}")
    print(f"MAPE: {mape:.2f}%")
    print(f"Запустить: python ai.py --predict")

# Точка входа – выбор режима по аргументу командной строки
if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == '--predict':
        predict_mode()
    else:
        train_mode()


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == '--predict':
        predict_mode()
    else:
        train_mode()
