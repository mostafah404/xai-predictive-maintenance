"""
Shared definitions for the XAI analysis scripts: the three RUL architectures (copied verbatim
from data_preprocessing.ipynb), the FD001 preprocessing, and a loader for the saved weights.
Importing this module does not train anything.
"""
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
WINDOW_SIZE = 50


class LSTMAttentionModel(nn.Module):
    def __init__(self, input_size, hidden_size=96):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=2, batch_first=True, dropout=0.2)
        self.attention = nn.Linear(hidden_size, 1)
        self.regressor = nn.Linear(hidden_size, 1)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        attention_weights = torch.softmax(self.attention(lstm_out), dim=1)
        context_vector = torch.sum(attention_weights * lstm_out, dim=1)
        rul_output = self.regressor(context_vector).squeeze(-1)
        return rul_output, attention_weights


class CNNModel(nn.Module):
    def __init__(self, input_size, seq_len=50):
        super().__init__()
        self.conv1 = nn.Conv1d(input_size, 32, kernel_size=5, padding=2)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv1d(32, 16, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.fc1 = nn.Linear(16 * seq_len, 64)
        self.relu3 = nn.ReLU()
        self.drop = nn.Dropout(0.2)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = x.flatten(1)
        x = self.relu3(self.fc1(x))
        x = self.drop(x)
        return self.fc2(x).squeeze(-1)


class HybridCNNLSTMBiLSTM(nn.Module):
    def __init__(self, input_size=24, seq_len=50, cnn_channels=None, lstm_hidden=96, lstm_layers=3,
                 bilstm_hidden=64, bilstm_layers=2, dropout=0.2):
        super().__init__()
        if cnn_channels is None:
            cnn_channels = [32, 64, 128]
        self.cnn = nn.Sequential(
            nn.Conv1d(input_size, cnn_channels[0], kernel_size=3, padding=1), nn.ReLU(), nn.Dropout(dropout),
            nn.Conv1d(cnn_channels[0], cnn_channels[1], kernel_size=3, padding=1), nn.ReLU(), nn.Dropout(dropout),
            nn.Conv1d(cnn_channels[1], cnn_channels[2], kernel_size=3, padding=1), nn.ReLU(), nn.Dropout(dropout),
        )
        self.lstm = nn.LSTM(cnn_channels[-1], lstm_hidden, num_layers=lstm_layers, batch_first=True,
                            dropout=dropout)
        self.bilstm = nn.LSTM(lstm_hidden, bilstm_hidden, num_layers=bilstm_layers, batch_first=True,
                              bidirectional=True, dropout=dropout)
        self.regressor = nn.Sequential(nn.Linear(bilstm_hidden * 2, 64), nn.ReLU(), nn.Dropout(dropout),
                                       nn.Linear(64, 1))

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.cnn(x)
        x = x.transpose(1, 2)
        x, _ = self.lstm(x)
        x, _ = self.bilstm(x)
        x = x[:, -1, :]
        return self.regressor(x).squeeze(-1)


def lstm_rul(model):
    """wrapper returning only the RUL of LSTMAttentionModel"""
    def f(x):
        return model(x)[0]
    return f


def load_fd001():
    """Replicates the notebook preprocessing; returns dict with X_val, y_val, feature names, scaler, ids."""
    columns = ["machine_id", "cycle"] + [f"operating_setting_{i}" for i in range(1, 4)] + \
              [f"sensor_{i}" for i in range(1, 22)]
    df = pd.read_csv(os.path.join(DATA, "train_FD001.txt"), sep=" ", header=None).dropna(axis=1)
    df.columns = columns
    max_cycles = df.groupby("machine_id")["cycle"].max()
    df["max_cycle"] = df["machine_id"].map(max_cycles)
    df["RUL"] = (df["max_cycle"] - df["cycle"]).clip(upper=125)
    df = df.drop(columns=["max_cycle"])
    feature_cols = df.columns[2:-1]
    scaler = StandardScaler()
    df[feature_cols] = scaler.fit_transform(df[feature_cols])
    machine_ids = df["machine_id"].unique()
    train_ids, val_ids = train_test_split(machine_ids, test_size=0.2, random_state=42)

    def windows(ids):
        X, y = [], []
        for mid in ids:
            md = df[df["machine_id"] == mid]
            vals = md[feature_cols].values
            rul = md["RUL"].values
            for i in range(len(md) - WINDOW_SIZE):
                X.append(vals[i:i + WINDOW_SIZE])
                y.append(rul[i + WINDOW_SIZE])
        return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

    X_val, y_val = windows(val_ids)
    X_train, y_train = windows(train_ids)
    return dict(X_val=torch.tensor(X_val), y_val=y_val, X_train=X_train, X_train_shape=X_train.shape,
                feature_names=list(feature_cols), scaler=scaler, val_ids=val_ids, train_ids=train_ids)


def load_models():
    """returns (lstm, cnn, hybrid) in eval mode with the saved FD001 weights"""
    n_feat = 24
    lstm = LSTMAttentionModel(n_feat, hidden_size=96)
    lstm.load_state_dict(torch.load(os.path.join(DATA, "model.pth"), map_location="cpu"))
    cnn = CNNModel(n_feat, seq_len=WINDOW_SIZE)
    cnn.load_state_dict(torch.load(os.path.join(DATA, "cnn_model.pth"), map_location="cpu"))
    hyb = HybridCNNLSTMBiLSTM(input_size=n_feat)
    hyb.load_state_dict(torch.load(os.path.join(DATA, "hybrid_model.pth"), map_location="cpu"))
    for m in (lstm, cnn, hyb):
        m.eval()
    return lstm, cnn, hyb
