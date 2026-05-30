import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

class TimeSeriesDataset(Dataset):
    def __init__(self, series, sequence_length):
        self.series = series
        self.sequence_length = sequence_length

    def __len__(self):
        return len(self.series) - self.sequence_length

    def __getitem__(self, index):
        x = self.series[index:index + self.sequence_length]
        y = self.series[index + self.sequence_length]
        return x.unsqueeze(-1), y

class RNNModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn = nn.RNN(input_size=1, hidden_size=32, batch_first=True)
        self.fc = nn.Linear(32, 1)
    def forward(self, x):
        output, _ = self.rnn(x)
        last_step = output[:, -1, :]
        return self.fc(last_step)

class LSTMModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=32, batch_first=True)
        self.fc = nn.Linear(32, 1)

    def forward(self, x):
        output, _ = self.lstm(x)
        last_step = output[:, -1, :]
        return self.fc(last_step)

def build_series():
    values = []
    for i in range(500):
        x = i * 0.05
        values.append(math.sin(x) + 0.1 * math.sin(3 * x))
    return torch.tensor(values, dtype=torch.float32)

def train_model(model, train_loader, test_loader):
    device = torch.device("cpu")
    model = model.to(device)
    print("device:", device)

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(1, 21):
        model.train()
        train_loss = 0.0
        train_total = 0

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device).unsqueeze(-1)

            pred = model(x)
            loss = loss_fn(pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_size = x.size(0)
            train_loss += loss.item() * batch_size
            train_total += batch_size

        model.eval()
        test_loss = 0.0
        test_total = 0

        with torch.no_grad():
            for x, y in test_loader:
                x = x.to(device)
                y = y.to(device).unsqueeze(-1)

                pred = model(x)
                loss = loss_fn(pred, y)

                batch_size = x.size(0)
                test_loss += loss.item() * batch_size
                test_total += batch_size

        train_loss = train_loss / train_total
        test_loss = test_loss / test_total
        print(f"Epoch {epoch}: train loss {train_loss:.6f}, test loss {test_loss:.6f}")

    predictions = []
    targets = []

    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            pred = model(x).squeeze(-1).cpu()
            predictions.extend(pred.tolist())
            targets.extend(y.tolist())

    return predictions, targets


if __name__ == "__main__":
    series = build_series()
    sequence_length = 20

    train_size = 400
    train_series = series[:train_size]
    test_series = series[train_size - sequence_length:]

    plt.figure(figsize=(10, 4))
    plt.plot(series.numpy(), label="series")
    plt.axvline(train_size, color="red", linestyle="--", label="train/test split")
    plt.title("Generated Time Series")
    plt.xlabel("time step")
    plt.ylabel("value")
    plt.legend()
    os.makedirs("outputs", exist_ok=True)
    out0 = os.path.join("outputs", "series.png")
    plt.savefig(out0)
    print("Saved series plot to", out0)
    plt.close()
    #write by llm
    train_dataset = TimeSeriesDataset(train_series, sequence_length)
    test_dataset = TimeSeriesDataset(test_series, sequence_length)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    print("RNN:")
    rnn_predictions, true_values = train_model(RNNModel(), train_loader, test_loader)

    print("LSTM:")
    lstm_predictions, _ = train_model(LSTMModel(), train_loader, test_loader)

    plt.figure(figsize=(12, 5))
    plt.plot(true_values, label="true", linewidth=2)
    plt.plot(rnn_predictions, label="RNN pred", linestyle="--")
    plt.plot(lstm_predictions, label="LSTM pred", linestyle=":")
    plt.title("Prediction vs True Value")
    plt.xlabel("test step")
    plt.ylabel("value")
    plt.legend()
    os.makedirs("outputs", exist_ok=True)
    out1 = os.path.join("outputs", "predictions.png")
    plt.savefig(out1)
    print("Saved predictions plot to", out1)
    plt.close()
    # write by llm
