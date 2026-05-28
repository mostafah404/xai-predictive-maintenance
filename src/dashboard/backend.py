#!/usr/bin/env python
# coding: utf-8
"""
backend.py — Data loading, model training/loading, inference, and XAI.
Loaded once at dashboard startup; all results returned as Python dicts/DataFrames.
"""

# ─────────────────────────────────────────────────────────────────────────────
# Imports & Seeds
# ─────────────────────────────────────────────────────────────────────────────
import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report
)
from scipy.stats import spearmanr
from captum.attr import IntegratedGradients, Occlusion
import streamlit as st

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

print("[backend] Torch:", torch.__version__)

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────
_BASE = os.path.dirname(os.path.abspath(__file__))
_LSTM_PATH = os.path.join(_BASE, "../../data/model.pth")
_CNN_PATH  = os.path.join(_BASE, "../../data/cnn_model.pth")

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
WINDOW_SIZE = 50

columns = (
    ["machine_id", "cycle"]
    + [f"operating_setting_{i}" for i in range(1, 4)]
    + [f"sensor_{i}" for i in range(1, 22)]
)

# ─────────────────────────────────────────────────────────────────────────────
# Data Loading & Preprocessing
# ─────────────────────────────────────────────────────────────────────────────
file_path = os.path.join(_BASE, "../../data/train_FD001.txt")
df = pd.read_csv(file_path, sep=" ", header=None).dropna(axis=1)
df.columns = columns

# RUL computation + cap at 125 (standard CMAPSS protocol)
max_cycles    = df.groupby("machine_id")["cycle"].max()
df["max_cycle"] = df["machine_id"].map(max_cycles)
df["RUL"]       = df["max_cycle"] - df["cycle"]
df["RUL"]       = df["RUL"].clip(upper=125)
df              = df.drop(columns=["max_cycle"])

feature_cols = df.columns[2:-1]          # 24 sensor/setting columns
feature_names = list(feature_cols)

# Keep raw copy for inverse-transform and healthy baseline
raw_df = df.copy()

# Healthy baseline (windows where RUL >= 120)
healthy_raw        = raw_df[raw_df["RUL"] >= 120]
healthy_mean_raw   = healthy_raw[feature_cols].mean()
healthy_std_raw    = healthy_raw[feature_cols].std()
healthy_mean_raw_np = np.array(healthy_mean_raw)
healthy_std_raw_np  = np.array(healthy_std_raw)

# StandardScaler fitted on training data only
scaler = StandardScaler()
df[feature_cols] = scaler.fit_transform(df[feature_cols])

MAX_CYCLE = df["cycle"].max()

# ─────────────────────────────────────────────────────────────────────────────
# Train / Validation Split + Sequence Creation
# ─────────────────────────────────────────────────────────────────────────────
machine_ids = df["machine_id"].unique()
train_ids, val_ids = train_test_split(machine_ids, test_size=0.2, random_state=42)

print(f"[backend] Train engines: {len(train_ids)}  |  Val engines: {len(val_ids)}")


def create_sequences_by_ids(dataframe, ids, window_size):
    sequences, targets, seq_ids = [], [], []
    for mid in ids:
        mdata = dataframe[dataframe["machine_id"] == mid]
        for i in range(len(mdata) - window_size):
            sequences.append(mdata.iloc[i : i + window_size][feature_cols].values)
            targets.append(mdata.iloc[i + window_size]["RUL"])
            seq_ids.append(mid)
    return np.array(sequences), np.array(targets), np.array(seq_ids)


# Val sequences only — training happens in the notebook
X_val_np, y_val_np, val_machine_ids = create_sequences_by_ids(df, val_ids, WINDOW_SIZE)

X_val = torch.tensor(X_val_np, dtype=torch.float32)
y_val = torch.tensor(y_val_np, dtype=torch.float32)

val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=64)

# ─────────────────────────────────────────────────────────────────────────────
# Risk Label Helpers
# ─────────────────────────────────────────────────────────────────────────────
def create_risk_labels(rul_tensor):
    risk = torch.zeros_like(rul_tensor, dtype=torch.long)
    risk[rul_tensor < 40] = 2
    risk[(rul_tensor >= 40) & (rul_tensor < 80)] = 1
    return risk


def rul_to_risk(rul_value):
    if rul_value < 40:
        return "HIGH"
    elif rul_value < 80:
        return "MEDIUM"
    return "LOW"


def get_true_risk(rul):
    if rul < 40:
        return "HIGH"
    elif rul < 100:
        return "MEDIUM"
    return "LOW"


# ─────────────────────────────────────────────────────────────────────────────
# Model Definitions
# ─────────────────────────────────────────────────────────────────────────────
class LSTMAttentionModel(nn.Module):
    def __init__(self, input_size, hidden_size=96):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers=2,
            batch_first=True, dropout=0.2
        )
        self.attention  = nn.Linear(hidden_size, 1)
        self.regressor  = nn.Linear(hidden_size, 1)

    def forward(self, x):
        lstm_out, _      = self.lstm(x)
        attn_weights     = torch.softmax(self.attention(lstm_out), dim=1)
        context_vector   = torch.sum(attn_weights * lstm_out, dim=1)
        rul_output       = self.regressor(context_vector).squeeze(-1)
        return rul_output, attn_weights


class CNNModel(nn.Module):
    """1-D CNN for RUL regression. Separate ReLU instances for LRP compatibility."""
    def __init__(self, input_size, seq_len=50):
        super().__init__()
        self.conv1 = nn.Conv1d(input_size, 32, kernel_size=5, padding=2)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv1d(32, 16, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.fc1   = nn.Linear(16 * seq_len, 64)
        self.relu3 = nn.ReLU()
        self.drop  = nn.Dropout(0.2)
        self.fc2   = nn.Linear(64, 1)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = x.flatten(1)
        x = self.relu3(self.fc1(x))
        x = self.drop(x)
        return self.fc2(x).squeeze(-1)


# ─────────────────────────────────────────────────────────────────────────────
# LSTM — Load pre-trained weights (run notebook first to generate model.pth)
# ─────────────────────────────────────────────────────────────────────────────
input_size = len(feature_cols)
model = LSTMAttentionModel(input_size, hidden_size=96)

if not os.path.exists(_LSTM_PATH):
    raise FileNotFoundError(
        f"[backend] LSTM weights not found at {_LSTM_PATH}.\n"
        "Run the notebook (data_preprocessing.ipynb) to train and save the model first."
    )

model.load_state_dict(torch.load(_LSTM_PATH, map_location="cpu"))
model.eval()
print(f"[backend] LSTM loaded from {_LSTM_PATH}")

# ─────────────────────────────────────────────────────────────────────────────
# LSTM — Validation Evaluation (for dashboard performance tab)
# ─────────────────────────────────────────────────────────────────────────────
model.eval()
all_risk_preds, all_risk_true = [], []
all_rul_preds,  all_rul_true  = [], []

with torch.no_grad():
    for X_batch, y_batch in val_loader:
        rul_preds, _ = model(X_batch)
        all_risk_preds.extend(create_risk_labels(rul_preds).numpy())
        all_risk_true.extend(create_risk_labels(y_batch).numpy())
        all_rul_preds.extend(rul_preds.numpy())
        all_rul_true.extend(y_batch.numpy())

all_risk_preds = np.array(all_risk_preds)
all_risk_true  = np.array(all_risk_true)
all_rul_preds  = np.array(all_rul_preds)
all_rul_true   = np.array(all_rul_true)

# ─────────────────────────────────────────────────────────────────────────────
# Test Set Loading
# ─────────────────────────────────────────────────────────────────────────────
test_path = os.path.join(_BASE, "../../data/test_FD001.txt")
test_df = pd.read_csv(test_path, sep=" ", header=None).dropna(axis=1)
test_df.columns = columns
test_df[feature_cols] = scaler.transform(test_df[feature_cols])

test_sequences, test_machine_ids = [], []
for mid in test_df["machine_id"].unique():
    mdata = test_df[test_df["machine_id"] == mid]
    if len(mdata) >= WINDOW_SIZE:
        window = mdata.iloc[-WINDOW_SIZE:]
    else:
        padding = pd.DataFrame([mdata.iloc[0]] * (WINDOW_SIZE - len(mdata)))
        window  = pd.concat([padding, mdata], ignore_index=True)
    test_sequences.append(window[feature_cols].values)
    test_machine_ids.append(mid)

X_test_np = np.array(test_sequences)
X_test    = torch.tensor(X_test_np, dtype=torch.float32)

rul_file      = os.path.join(_BASE, "../../data/RUL_FD001.txt")
true_rul_test = pd.read_csv(rul_file, header=None)[0]

# LSTM test predictions
model.eval()
with torch.no_grad():
    rul_preds_test, _ = model(X_test)
    pred_rul_test     = rul_preds_test.numpy().flatten()

risk_labels   = ["LOW", "MEDIUM", "HIGH"]
predicted_risk_test = [risk_labels[c] for c in create_risk_labels(rul_preds_test).numpy()]

comparison_df = pd.DataFrame({
    "Machine_ID":     test_machine_ids,
    "True_RUL":       true_rul_test.values,
    "True_Risk":      true_rul_test.apply(get_true_risk).values,
    "Predicted_RUL":  pred_rul_test,
    "Predicted_Risk": predicted_risk_test,
})

# ─────────────────────────────────────────────────────────────────────────────
# Integrated Gradients Setup
# ─────────────────────────────────────────────────────────────────────────────
def forward_rul_only(x):
    rul_pred, _ = model(x)
    return rul_pred

ig = IntegratedGradients(forward_rul_only)

# ─────────────────────────────────────────────────────────────────────────────
# General Helpers
# ─────────────────────────────────────────────────────────────────────────────
def get_max_cycle():
    return MAX_CYCLE



def get_comparison_df():
    return comparison_df.copy()


# ─────────────────────────────────────────────────────────────────────────────
# LSTM Fleet Ranking
# ─────────────────────────────────────────────────────────────────────────────
def build_fleet_ranking(cycle_number):
    fleet_records = []
    model.eval()
    for machine_id in val_ids:
        machine_df = df[df["machine_id"] == machine_id]
        history    = st.session_state.get(f"maintenance_history_{machine_id}", [])
        reset_cycle = history[-1] if history else None
        eff = max(WINDOW_SIZE, cycle_number - reset_cycle) if reset_cycle else cycle_number
        machine_df = machine_df[machine_df["cycle"] <= eff]
        if len(machine_df) < WINDOW_SIZE:
            continue
        window       = machine_df.iloc[-WINDOW_SIZE:]
        sample_input = torch.tensor(window[feature_cols].values, dtype=torch.float32).unsqueeze(0)
        rul_pred, _  = model(sample_input)
        predicted_rul = max(0, rul_pred.item())
        fleet_records.append({
            "Machine_ID":    machine_id,
            "Predicted_RUL": predicted_rul,
            "Risk_Label":    rul_to_risk(predicted_rul),
            "Cycle":         eff,
        })
    return pd.DataFrame(fleet_records).sort_values("Predicted_RUL").reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# LSTM Explain Machine (IG + Attention)
# ─────────────────────────────────────────────────────────────────────────────
def explain_machine(machine_id, cycle_number, fleet_df, top_k=3, show_plot=False):
    row = fleet_df[fleet_df["Machine_ID"] == machine_id]
    if row.empty:
        return None, None

    cycle_number = int(row.iloc[0]["Cycle"])
    machine_df   = df[df["machine_id"] == machine_id]
    history      = st.session_state.get(f"maintenance_history_{machine_id}", [])
    reset_cycle  = history[-1] if history else None
    eff = max(WINDOW_SIZE, cycle_number - reset_cycle) if reset_cycle else cycle_number
    machine_df   = machine_df[machine_df["cycle"] <= eff]
    if len(machine_df) < WINDOW_SIZE:
        return None, None

    window       = machine_df.iloc[-WINDOW_SIZE:]
    window_array = window[feature_cols].values
    sample_input = torch.tensor(window_array, dtype=torch.float32).unsqueeze(0)

    model.eval()
    rul_pred, attention_weights = model(sample_input)
    predicted_rul = max(0, rul_pred.item())
    risk_label    = rul_to_risk(predicted_rul)

    baseline     = torch.zeros_like(sample_input)
    attributions = ig.attribute(sample_input, baselines=baseline, n_steps=50)
    attr         = attributions.squeeze().detach().numpy()      # (50, 24) signed
    att          = attention_weights.squeeze().detach().numpy() # (50,)

    feature_importance = np.mean(np.abs(attr), axis=0)
    top_features       = np.argsort(feature_importance)[::-1][:top_k]
    total_importance   = np.sum(feature_importance)

    original_window = scaler.inverse_transform(sample_input.squeeze().detach().numpy())
    cycle_labels    = np.arange(cycle_number - WINDOW_SIZE + 1, cycle_number + 1)
    cycle_range     = np.arange(cycle_number - WINDOW_SIZE + 1, cycle_number + 1)

    report_rows = []
    for f in top_features:
        sensor_name  = feature_names[f]
        sensor_series = original_window[:, f]
        mean_value    = np.mean(sensor_series)
        baseline_mean = healthy_mean_raw.iloc[f]
        baseline_std  = healthy_std_raw.iloc[f]
        pct_denom     = max(abs(baseline_mean), baseline_std, 1e-6)
        mean_deviation     = mean_value - baseline_mean
        mean_percent_dev   = (mean_deviation / pct_denom) * 100
        direction_mean     = "above" if mean_deviation > 0 else "below"

        peak_index    = np.argmax(np.abs(attr[:, f]))
        peak_cycle    = cycle_number - WINDOW_SIZE + 1 + peak_index
        peak_value    = original_window[peak_index, f]
        peak_deviation     = peak_value - baseline_mean
        peak_percent_dev   = (peak_deviation / pct_denom) * 100
        direction_peak     = "above" if peak_deviation > 0 else "below"
        importance_pct     = (feature_importance[f] / (total_importance + 1e-8)) * 100

        def _fmt(v): return f"{v:.6g}"
        report_rows.append({
            "Sensor":                   sensor_name,
            "Window Mean Value":        _fmt(mean_value),
            "Healthy Mean":             _fmt(baseline_mean),
            "Mean Deviation":           f"{_fmt(abs(mean_deviation))} ({direction_mean})",
            "Mean Percentage Deviation": f"{abs(mean_percent_dev):.2f}%",
            "Peak Cycle":               int(peak_cycle),
            "Peak Value":               _fmt(peak_value),
            "Peak Deviation":           f"{_fmt(abs(peak_deviation))} ({direction_peak})",
            "Peak Percentage Deviation ": f"{abs(peak_percent_dev):.2f}%",
            "IG Contribution %":        round(importance_pct, 2),
        })

    result = {
        "machine_id":    machine_id,
        "cycle":         cycle_labels,
        "risk":          risk_label,
        "rul":           predicted_rul,
        "attention":     att,
        "ig":            attr,
        "cycles":        cycle_range,
        "feature_names": feature_names,
        "sensor_values": original_window,
        "healthy_values": healthy_mean_raw_np,
        "healthy_std":   healthy_std_raw_np,
    }
    return result, pd.DataFrame(report_rows)


# ─────────────────────────────────────────────────────────────────────────────
# IG–Occlusion Agreement
# ─────────────────────────────────────────────────────────────────────────────
def get_ig_occlusion_data(machine_id: int, cycle_number: int) -> dict:
    try:
        machine_id = int(machine_id)
        machine_df = df[df["machine_id"] == machine_id]
        history    = st.session_state.get(f"maintenance_history_{machine_id}", [])
        reset_cycle = history[-1] if history else None
        eff = max(WINDOW_SIZE, cycle_number - reset_cycle) if reset_cycle else cycle_number
        machine_df = machine_df[machine_df["cycle"] <= eff]
        if len(machine_df) < WINDOW_SIZE:
            return {"ig": [], "occ": [], "corr": 0}

        window       = machine_df.iloc[-WINDOW_SIZE:]
        window_array = window[feature_cols].values
        sample_input = torch.tensor(window_array, dtype=torch.float32).unsqueeze(0)

        model.eval()
        baseline  = torch.zeros_like(sample_input)
        ig_attr   = ig.attribute(sample_input, baselines=baseline, n_steps=50)
        ig_np     = ig_attr.squeeze().detach().numpy()
        ig_importance = np.mean(np.abs(ig_np), axis=0)

        occlusion    = Occlusion(forward_rul_only)
        occ_attr     = occlusion.attribute(
            sample_input, baselines=0,
            sliding_window_shapes=(3, 1), strides=(1, 1)
        )
        occ_np        = occ_attr.squeeze().detach().numpy()
        occ_importance = np.mean(np.abs(occ_np), axis=0)

        corr = (
            float(spearmanr(ig_importance, occ_importance)[0])
            if np.std(ig_importance) > 0 and np.std(occ_importance) > 0
            else 0.0
        )
        return {"ig": ig_importance.tolist(), "occ": occ_importance.tolist(), "corr": corr}
    except Exception as e:
        print("ERROR in get_ig_occlusion_data:", e)
        return {"ig": [], "occ": [], "corr": 0}


# ─────────────────────────────────────────────────────────────────────────────
# LSTM Performance Helpers
# ─────────────────────────────────────────────────────────────────────────────
def get_model_performance():
    accuracy = accuracy_score(all_risk_true, all_risk_preds)
    cm       = confusion_matrix(all_risk_true, all_risk_preds)
    report   = classification_report(
        all_risk_true, all_risk_preds,
        target_names=["Low", "Medium", "High"], output_dict=True
    )
    class_df = pd.DataFrame(report).transpose().reset_index()

    rmse_all  = np.sqrt(np.mean((all_rul_preds - all_rul_true) ** 2))
    mae_all   = np.mean(np.abs(all_rul_preds - all_rul_true))
    mask_high = all_risk_true == 2
    rmse_high = (
        np.sqrt(np.mean((all_rul_preds[mask_high] - all_rul_true[mask_high]) ** 2))
        if mask_high.sum() > 0 else float("nan")
    )
    mae_high = (
        np.mean(np.abs(all_rul_preds[mask_high] - all_rul_true[mask_high]))
        if mask_high.sum() > 0 else float("nan")
    )
    return dict(
        accuracy=accuracy, confusion_matrix=cm, classification_df=class_df,
        rmse_all=rmse_all, mae_all=mae_all, rmse_high=rmse_high, mae_high=mae_high,
    )


def get_val_scatter_data():
    return {"true": all_rul_true, "pred": all_rul_preds}


def get_engine_rul_plot(machine_id):
    idxs = np.where(val_machine_ids == machine_id)[0]
    tr, pr, cs = [], [], []
    model.eval()
    with torch.no_grad():
        for idx in idxs:
            rul_pred, _ = model(X_val[idx].unsqueeze(0))
            tr.append(y_val_np[idx])
            pr.append(rul_pred.item())
            cs.append(len(cs))
    tr, pr, cs = np.array(tr), np.array(pr), np.array(cs)
    mask = tr < 100
    return {"cycles": cs[mask], "true": tr[mask], "pred": pr[mask]}


def get_test_scatter_data():
    return {"true": comparison_df["True_RUL"].values, "pred": comparison_df["Predicted_RUL"].values}


def get_test_performance():
    risk_map = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}
    true_int = comparison_df["True_Risk"].map(risk_map).values
    pred_int = comparison_df["Predicted_Risk"].map(risk_map).values
    accuracy = accuracy_score(true_int, pred_int)
    cm       = confusion_matrix(true_int, pred_int)
    report   = classification_report(
        true_int, pred_int, target_names=["Low", "Medium", "High"], output_dict=True
    )
    class_df = pd.DataFrame(report).transpose().reset_index()
    rmse     = np.sqrt(np.mean((comparison_df["Predicted_RUL"] - comparison_df["True_RUL"]) ** 2))
    mae      = np.mean(np.abs(comparison_df["Predicted_RUL"] - comparison_df["True_RUL"]))
    mask_high = comparison_df["True_Risk"] == "HIGH"
    rmse_high = (
        np.sqrt(np.mean((comparison_df.loc[mask_high, "Predicted_RUL"]
                         - comparison_df.loc[mask_high, "True_RUL"]) ** 2))
        if mask_high.sum() > 0 else float("nan")
    )
    mae_high = (
        np.mean(np.abs(comparison_df.loc[mask_high, "Predicted_RUL"]
                       - comparison_df.loc[mask_high, "True_RUL"]))
        if mask_high.sum() > 0 else float("nan")
    )
    return dict(
        accuracy=accuracy, confusion_matrix=cm, classification_df=class_df,
        rmse=rmse, mae=mae, rmse_high=rmse_high, mae_high=mae_high,
    )


# ─────────────────────────────────────────────────────────────────────────────
# CNN — Load from disk
# ─────────────────────────────────────────────────────────────────────────────
CNN_AVAILABLE = False
cnn_model     = None

if os.path.exists(_CNN_PATH):
    try:
        cnn_model = CNNModel(input_size=len(feature_cols), seq_len=WINDOW_SIZE)
        cnn_model.load_state_dict(torch.load(_CNN_PATH, map_location="cpu"))
        cnn_model.eval()
        CNN_AVAILABLE = True
        print(f"[backend] CNN loaded from {_CNN_PATH}")
    except Exception as _e:
        print(f"[backend] CNN load failed: {_e}")
else:
    print(f"[backend] CNN not found at {_CNN_PATH} — train in notebook first.")

# ─────────────────────────────────────────────────────────────────────────────
# CNN — Validation & Test Evaluation
# ─────────────────────────────────────────────────────────────────────────────
if CNN_AVAILABLE:
    _cnn_vp, _cnn_vt, _cnn_vrp, _cnn_vrt = [], [], [], []
    cnn_model.eval()
    with torch.no_grad():
        for _Xb, _yb in val_loader:
            _p = cnn_model(_Xb)
            _cnn_vp.extend(_p.numpy().flatten())
            _cnn_vt.extend(_yb.numpy().flatten())
            _cnn_vrp.extend(create_risk_labels(_p).numpy())
            _cnn_vrt.extend(create_risk_labels(_yb).numpy())
    _cnn_val_rul_preds  = np.array(_cnn_vp)
    _cnn_val_rul_true   = np.array(_cnn_vt)
    _cnn_val_risk_preds = np.array(_cnn_vrp)
    _cnn_val_risk_true  = np.array(_cnn_vrt)

    cnn_model.eval()
    with torch.no_grad():
        _cnn_test_preds_arr = cnn_model(X_test).numpy().flatten()
    _cnn_pred_risk_test = [
        risk_labels[c]
        for c in create_risk_labels(torch.tensor(_cnn_test_preds_arr)).numpy()
    ]
    comparison_df_cnn = pd.DataFrame({
        "Machine_ID":     test_machine_ids,
        "True_RUL":       true_rul_test.values,
        "True_Risk":      true_rul_test.apply(get_true_risk).values,
        "Predicted_RUL":  _cnn_test_preds_arr,
        "Predicted_Risk": _cnn_pred_risk_test,
    })

# ─────────────────────────────────────────────────────────────────────────────
# CNN — Activation Maximization (cached)
# ─────────────────────────────────────────────────────────────────────────────
_cnn_act_max_cache = None


def get_cnn_activation_max():
    global _cnn_act_max_cache
    if _cnn_act_max_cache is not None:
        return _cnn_act_max_cache
    if not CNN_AVAILABLE:
        return np.zeros((WINDOW_SIZE, len(feature_cols)))
    _tmp = CNNModel(input_size=len(feature_cols), seq_len=WINDOW_SIZE)
    _tmp.load_state_dict(cnn_model.state_dict())
    _tmp.train()
    synthetic = torch.zeros(1, WINDOW_SIZE, len(feature_cols), requires_grad=True)
    opt = torch.optim.Adam([synthetic], lr=0.05)
    for _ in range(300):
        opt.zero_grad()
        (-_tmp(synthetic)).backward()
        opt.step()
    _cnn_act_max_cache = synthetic.squeeze().detach().numpy()
    return _cnn_act_max_cache


# ─────────────────────────────────────────────────────────────────────────────
# CNN — Fleet Ranking
# ─────────────────────────────────────────────────────────────────────────────
def build_fleet_ranking_cnn(cycle_number):
    if not CNN_AVAILABLE:
        return pd.DataFrame()
    records = []
    cnn_model.eval()
    for machine_id in val_ids:
        mdf  = df[df["machine_id"] == machine_id]
        hist = st.session_state.get(f"maintenance_history_{machine_id}", [])
        rc   = hist[-1] if hist else None
        eff  = max(WINDOW_SIZE, cycle_number - rc) if rc else cycle_number
        mdf  = mdf[mdf["cycle"] <= eff]
        if len(mdf) < WINDOW_SIZE:
            continue
        win = mdf.iloc[-WINDOW_SIZE:]
        si  = torch.tensor(win[feature_cols].values, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            rul = max(0, cnn_model(si).item())
        records.append({
            "Machine_ID":    machine_id,
            "Predicted_RUL": rul,
            "Risk_Label":    rul_to_risk(rul),
            "Cycle":         eff,
        })
    return pd.DataFrame(records).sort_values("Predicted_RUL").reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# CNN — Explain Machine (LRP)
# ─────────────────────────────────────────────────────────────────────────────
def explain_machine_cnn(machine_id, cycle_number, fleet_df_cnn, top_k=3):
    if not CNN_AVAILABLE:
        return None, None
    row = fleet_df_cnn[fleet_df_cnn["Machine_ID"] == machine_id]
    if row.empty:
        return None, None

    cycle_number = int(row.iloc[0]["Cycle"])
    mdf  = df[df["machine_id"] == machine_id]
    hist = st.session_state.get(f"maintenance_history_{machine_id}", [])
    rc   = hist[-1] if hist else None
    eff  = max(WINDOW_SIZE, cycle_number - rc) if rc else cycle_number
    mdf  = mdf[mdf["cycle"] <= eff]
    if len(mdf) < WINDOW_SIZE:
        return None, None

    win = mdf.iloc[-WINDOW_SIZE:]
    si  = torch.tensor(win[feature_cols].values, dtype=torch.float32).unsqueeze(0)
    cnn_model.eval()
    with torch.no_grad():
        rul = max(0, cnn_model(si).item())
    risk = rul_to_risk(rul)

    # LRP (ε-rule)
    from captum.attr import LRP as _LRP
    from captum.attr._utils.lrp_rules import EpsilonRule as _EpsilonRule
    for _mod in cnn_model.modules():
        if isinstance(_mod, (nn.Conv1d, nn.Linear, nn.ReLU, nn.Dropout)):
            _mod.rule = _EpsilonRule()
    lrp_map = _LRP(cnn_model).attribute(si).squeeze().detach().numpy()  # (50, 24) signed

    orig_win = scaler.inverse_transform(si.squeeze().detach().numpy())
    fi_vals  = np.mean(np.abs(lrp_map), axis=0)
    top_f    = np.argsort(fi_vals)[::-1][:top_k]
    total_fi = fi_vals.sum()
    start_c  = cycle_number - WINDOW_SIZE + 1
    cyc_rng  = np.arange(start_c, cycle_number + 1)

    rows = []
    for f in top_f:
        sn  = feature_names[f]
        mv  = np.mean(orig_win[:, f])
        bm  = healthy_mean_raw.iloc[f]
        bs  = healthy_std_raw.iloc[f]
        den = max(abs(bm), bs, 1e-6)
        md  = mv - bm
        d   = "above" if md > 0 else "below"
        pi  = np.argmax(np.abs(lrp_map[:, f]))
        pv  = orig_win[pi, f]
        pd2 = pv - bm
        dp  = "above" if pd2 > 0 else "below"
        imp = fi_vals[f] / (total_fi + 1e-8) * 100
        def _f(v): return f"{v:.6g}"
        rows.append({
            "Sensor":             sn,
            "Window Mean":        _f(mv),
            "Healthy Mean":       _f(bm),
            "Mean Deviation":     f"{_f(abs(md))} ({d})",
            "Mean % Dev":         f"{abs(md / den) * 100:.2f}%",
            "Peak Cycle":         int(start_c + pi),
            "Peak Value":         _f(pv),
            "Peak Deviation":     f"{_f(abs(pd2))} ({dp})",
            "Peak % Dev":         f"{abs(pd2 / den) * 100:.2f}%",
            "LRP Contribution %": round(imp, 2),
        })

    result = {
        "machine_id":    machine_id,
        "risk":          risk,
        "rul":           rul,
        "lrp":           lrp_map,
        "cycles":        cyc_rng,
        "feature_names": list(feature_names),
        "sensor_values": orig_win,
        "healthy_values": healthy_mean_raw_np,
        "healthy_std":   healthy_std_raw_np,
    }
    return result, pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# CNN — Performance Helpers
# ─────────────────────────────────────────────────────────────────────────────
def get_model_performance_cnn():
    if not CNN_AVAILABLE:
        return None
    acc = accuracy_score(_cnn_val_risk_true, _cnn_val_risk_preds)
    cm  = confusion_matrix(_cnn_val_risk_true, _cnn_val_risk_preds)
    rep = classification_report(
        _cnn_val_risk_true, _cnn_val_risk_preds,
        target_names=["Low", "Medium", "High"], output_dict=True
    )
    cdf = pd.DataFrame(rep).transpose().reset_index()
    ra  = np.sqrt(np.mean((_cnn_val_rul_preds - _cnn_val_rul_true) ** 2))
    ma  = np.mean(np.abs(_cnn_val_rul_preds - _cnn_val_rul_true))
    mmh = _cnn_val_risk_true != 0
    rmh = np.sqrt(np.mean((_cnn_val_rul_preds[mmh] - _cnn_val_rul_true[mmh]) ** 2))
    mmh2 = np.mean(np.abs(_cnn_val_rul_preds[mmh] - _cnn_val_rul_true[mmh]))
    mh  = _cnn_val_risk_true == 2
    rh  = (np.sqrt(np.mean((_cnn_val_rul_preds[mh] - _cnn_val_rul_true[mh]) ** 2))
           if mh.sum() > 0 else float("nan"))
    mah = (np.mean(np.abs(_cnn_val_rul_preds[mh] - _cnn_val_rul_true[mh]))
           if mh.sum() > 0 else float("nan"))
    return dict(
        accuracy=acc, confusion_matrix=cm, classification_df=cdf,
        rmse_all=ra, mae_all=ma, rmse=rmh, mae=mmh2,
        rmse_high=rh, mae_high=mah,
    )


def get_test_performance_cnn():
    if not CNN_AVAILABLE:
        return None
    rm = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}
    ti = comparison_df_cnn["True_Risk"].map(rm).values
    pi = comparison_df_cnn["Predicted_Risk"].map(rm).values
    acc = accuracy_score(ti, pi)
    cm  = confusion_matrix(ti, pi)
    rep = classification_report(ti, pi, target_names=["Low", "Medium", "High"], output_dict=True)
    cdf = pd.DataFrame(rep).transpose().reset_index()
    r   = np.sqrt(np.mean((comparison_df_cnn["Predicted_RUL"] - comparison_df_cnn["True_RUL"]) ** 2))
    m   = np.mean(np.abs(comparison_df_cnn["Predicted_RUL"] - comparison_df_cnn["True_RUL"]))
    mh  = comparison_df_cnn["True_Risk"] == "HIGH"
    rh  = (np.sqrt(np.mean((comparison_df_cnn.loc[mh, "Predicted_RUL"]
                             - comparison_df_cnn.loc[mh, "True_RUL"]) ** 2))
           if mh.sum() > 0 else float("nan"))
    mah = (np.mean(np.abs(comparison_df_cnn.loc[mh, "Predicted_RUL"]
                           - comparison_df_cnn.loc[mh, "True_RUL"]))
           if mh.sum() > 0 else float("nan"))
    return dict(
        accuracy=acc, confusion_matrix=cm, classification_df=cdf,
        rmse=r, mae=m, rmse_high=rh, mae_high=mah,
    )


def get_cnn_val_scatter_data():
    if not CNN_AVAILABLE:
        return {"true": np.array([]), "pred": np.array([])}
    return {"true": _cnn_val_rul_true, "pred": _cnn_val_rul_preds}


def get_cnn_test_scatter_data():
    if not CNN_AVAILABLE:
        return {"true": np.array([]), "pred": np.array([])}
    return {"true": comparison_df_cnn["True_RUL"].values,
            "pred": comparison_df_cnn["Predicted_RUL"].values}


def get_cnn_engine_rul_plot(machine_id):
    if not CNN_AVAILABLE:
        return {"cycles": [], "true": [], "pred": []}
    idxs = np.where(val_machine_ids == machine_id)[0]
    tr, pr, cs = [], [], []
    cnn_model.eval()
    with torch.no_grad():
        for i in idxs:
            pr.append(cnn_model(X_val[i].unsqueeze(0)).item())
            tr.append(y_val_np[i])
            cs.append(len(cs))
    t, p, c = np.array(tr), np.array(pr), np.array(cs)
    mask = t < 100
    return {"cycles": c[mask], "true": t[mask], "pred": p[mask]}


# ─────────────────────────────────────────────────────────────────────────────
# Export Helpers
# ─────────────────────────────────────────────────────────────────────────────
_feature_names_export = list(feature_cols)
_healthy_mean_export  = healthy_mean_raw_np.copy()
_healthy_std_export   = healthy_std_raw_np.copy()


def get_engine_peak_snapshot(cycle_number=None):
    """Raw (unscaled) sensor values at the highest observed cycle per engine."""
    records = []
    for machine_id in val_ids:
        mdf = raw_df[raw_df["machine_id"] == machine_id]
        if cycle_number is not None:
            mdf = mdf[mdf["cycle"] <= cycle_number]
        if len(mdf) == 0:
            continue
        peak_row = mdf.loc[mdf["cycle"].idxmax()]
        rec = {"Machine_ID": machine_id, "Peak_Cycle": int(peak_row["cycle"])}
        for col in feature_cols:
            rec[col] = round(float(peak_row[col]), 4)
        records.append(rec)
    return pd.DataFrame(records).sort_values("Machine_ID").reset_index(drop=True)


def get_lrp_ig_data_cnn(machine_id: int, cycle_number: int) -> dict:
    """Spearman agreement between LRP and IG feature importance for the CNN."""
    if not CNN_AVAILABLE:
        return {"lrp": [], "ig": [], "corr": 0}
    try:
        machine_id = int(machine_id)
        mdf = df[df["machine_id"] == machine_id]
        history     = st.session_state.get(f"maintenance_history_{machine_id}", [])
        reset_cycle = history[-1] if history else None
        eff = max(WINDOW_SIZE, cycle_number - reset_cycle) if reset_cycle else cycle_number
        mdf = mdf[mdf["cycle"] <= eff]
        if len(mdf) < WINDOW_SIZE:
            return {"lrp": [], "ig": [], "corr": 0}

        si = torch.tensor(
            mdf.iloc[-WINDOW_SIZE:][feature_cols].values, dtype=torch.float32
        ).unsqueeze(0)

        # ── LRP ──────────────────────────────────────────────────────────────
        from captum.attr import LRP as _LRP
        from captum.attr._utils.lrp_rules import EpsilonRule as _EpsilonRule
        for _mod in cnn_model.modules():
            if isinstance(_mod, (nn.Conv1d, nn.Linear, nn.ReLU, nn.Dropout)):
                _mod.rule = _EpsilonRule()
        lrp_np = _LRP(cnn_model).attribute(si).squeeze().detach().numpy()
        lrp_importance = np.mean(np.abs(lrp_np), axis=0)   # (24,)

        # ── IG on CNN ────────────────────────────────────────────────────────
        ig_cnn   = IntegratedGradients(cnn_model)
        baseline = torch.zeros_like(si)
        ig_np    = ig_cnn.attribute(si, baselines=baseline, n_steps=50)
        ig_np    = ig_np.squeeze().detach().numpy()
        ig_importance = np.mean(np.abs(ig_np), axis=0)      # (24,)

        corr = (
            float(spearmanr(lrp_importance, ig_importance)[0])
            if np.std(lrp_importance) > 0 and np.std(ig_importance) > 0
            else 0.0
        )
        return {"lrp": lrp_importance.tolist(), "ig": ig_importance.tolist(), "corr": corr}
    except Exception as e:
        print("ERROR in get_lrp_ig_data_cnn:", e)
        return {"lrp": [], "ig": [], "corr": 0}


def get_fleet_sensor_summary(cycle_number, use_cnn=False):
    """Mean raw sensor values over the last WINDOW_SIZE cycles, with RUL prediction."""
    records = []
    for machine_id in val_ids:
        mdf_raw = raw_df[raw_df["machine_id"] == machine_id]
        mdf_raw = mdf_raw[mdf_raw["cycle"] <= cycle_number]
        if len(mdf_raw) < WINDOW_SIZE:
            continue
        sensor_means = mdf_raw.iloc[-WINDOW_SIZE:][feature_cols].mean()

        mdf = df[df["machine_id"] == machine_id]
        mdf = mdf[mdf["cycle"] <= cycle_number]
        if len(mdf) < WINDOW_SIZE:
            continue
        si = torch.tensor(mdf.iloc[-WINDOW_SIZE:][feature_cols].values,
                          dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            if use_cnn and CNN_AVAILABLE:
                rul = max(0.0, cnn_model(si).item())
            else:
                out = model(si)
                rul = max(0.0, (out[0] if isinstance(out, tuple) else out).item())

        rec = {
            "Machine_ID":    machine_id,
            "Predicted_RUL": round(rul, 1),
            "Risk_Label":    rul_to_risk(rul),
            "Cycle":         cycle_number,
        }
        for col_name, val in sensor_means.items():
            rec[f"mean_{col_name}"] = round(float(val), 4)
        records.append(rec)
    return pd.DataFrame(records).sort_values("Predicted_RUL").reset_index(drop=True)
