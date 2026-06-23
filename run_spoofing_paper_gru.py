from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from split import group_train_val_test_split


ROOT = Path(__file__).resolve().parent
RUN_ROOT = ROOT / "Outputs" / "spoofing_paper_gru01_six_attack_seed42"
GENERATED = RUN_ROOT / "generated_internal"
PREPARED = RUN_ROOT / "data_internal_seq10"
NPZ_PATH = PREPARED / "processed_spoofing.npz"
MODEL_DIR = RUN_ROOT / "model_gru_seed42"
MODEL_PATH = MODEL_DIR / "model.pt"
SEED = 42
ATTACKS = (
    "gradual_drift",
    "location_jump",
    "replay",
    "meaconing",
    "ghost",
    "mirroring",
)


class PaperGRU(nn.Module):
    """GRU comparator adapted from Agrebi et al. (IEEE Access, 2025)."""

    def __init__(self, input_size: int):
        super().__init__()
        self.gru = nn.GRU(
            input_size=int(input_size),
            hidden_size=64,
            num_layers=1,
            batch_first=True,
            bidirectional=False,
        )
        self.batch_norm = nn.BatchNorm1d(64)
        self.dropout_gru = nn.Dropout(0.30)
        self.dense = nn.Linear(64, 32)
        self.dropout_dense = nn.Dropout(0.20)
        self.output = nn.Linear(32, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sequence, _ = self.gru(x)
        features = sequence[:, -1, :]
        features = self.batch_norm(features)
        features = self.dropout_gru(features)
        features = torch.relu(self.dense(features))
        features = self.dropout_dense(features)
        return self.output(features)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except TypeError:
        torch.use_deterministic_algorithms(True)


def run_command(command: list[str], label: str) -> None:
    print(f"[paper-gru] {label}")
    print(" ".join(command))
    subprocess.run(command, cwd=ROOT, check=True)


def prepare() -> None:
    """Generate all six attacks and preprocess sequence length 10."""
    if not list(GENERATED.glob("spoofed_*.csv")):
        run_command(
            [
                sys.executable,
                "main.py",
                "make_spoofing",
                "--input_path", "Dataset",
                "--out_dir", str(GENERATED),
                "--attacks", *ATTACKS,
                "--seed", str(SEED),
                "--limit_rows", "300000",
                "--normal_keep_frac", "0.5",
                "--max_vessels_per_file", "10",
                "--min_points_per_vessel", "160",
                "--points_per_attack", "120",
                "--scenarios_per_attack", "1",
                "--max_attack_gap_seconds", "10800",
                "--drift_rate_kmh", "0.033",
                "--drift_rate_jitter_frac", "0.50",
                "--jump_lat_deg", "0.50",
                "--jump_lon_deg", "0.50",
                "--reported_motion_mode", "recompute",
                "--include_matched_normal_controls",
            ],
            "generate six-attack internal dataset",
        )
    else:
        print("[paper-gru] generated data already exists; skip generation")

    if not NPZ_PATH.is_file():
        run_command(
            [
                sys.executable,
                "main.py",
                "preprocess",
                "--data_dir", str(GENERATED),
                "--out_dir", str(PREPARED),
                "--task", "spoofing",
                "--seq_len", "10",
                "--stride", "1",
                "--gap_seconds", "10800",
                "--min_points_per_vessel", "10",
                "--max_windows_per_vessel", "1200",
                "--max_windows_per_file", "30000",
                "--spoofing_window_threshold", "0.20",
                "--exclude_location_features",
            ],
            "preprocess paper-style sequence length 10",
        )
    else:
        print("[paper-gru] processed NPZ already exists; skip preprocessing")


def load_data() -> dict[str, np.ndarray | dict[int, str]]:
    if not NPZ_PATH.is_file():
        raise FileNotFoundError(
            f"Missing {NPZ_PATH}. Run: python run_spoofing_paper_gru.py prepare"
        )
    data = np.load(NPZ_PATH, allow_pickle=True)
    label_map = {
        int(key): str(value) for key, value in data["label_map"].tolist()
    }
    if {key: value.lower() for key, value in label_map.items()} != {
        0: "normal", 1: "spoofing"
    }:
        raise ValueError(f"Unexpected spoofing label map: {label_map}")
    n = int(data["y"].shape[0])
    return {
        "X": data["X"].astype(np.float32),
        "y": data["y"].astype(np.int64),
        "groups": data["groups"].astype(str),
        "event_ids": (
            data["window_event_ids"].astype(str)
            if "window_event_ids" in data.files
            else np.array([""] * n, dtype=str)
        ),
        "attack_types": (
            data["window_kinds"].astype(str)
            if "window_kinds" in data.files
            else np.array(["unknown"] * n, dtype=str)
        ),
        "feature_cols": (
            data["feature_cols"].astype(str)
            if "feature_cols" in data.files
            else np.array([], dtype=str)
        ),
        "label_map": label_map,
    }


def scaled_arrays(
    X: np.ndarray,
    split,
) -> tuple[np.ndarray, StandardScaler]:
    scaler = StandardScaler()
    feature_count = int(X.shape[-1])
    scaler.fit(X[split.train_idx].reshape(-1, feature_count))
    transformed = scaler.transform(X.reshape(-1, feature_count)).reshape(X.shape)
    return transformed.astype(np.float32), scaler


def make_train_loader(X: np.ndarray, y: np.ndarray) -> DataLoader:
    counts = np.bincount(y, minlength=2).astype(np.float64)
    weights = counts.sum() / np.clip(2.0 * counts, 1.0, None)
    sample_weights = weights[y]
    generator = torch.Generator().manual_seed(SEED)
    sampler = WeightedRandomSampler(
        torch.as_tensor(sample_weights, dtype=torch.double),
        num_samples=int(len(y)),
        replacement=True,
        generator=generator,
    )
    dataset = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    return DataLoader(
        dataset,
        batch_size=32,
        sampler=sampler,
        num_workers=0,
        drop_last=True,
    )


def make_loader(X: np.ndarray, y: np.ndarray) -> DataLoader:
    dataset = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    return DataLoader(dataset, batch_size=256, shuffle=False, num_workers=0)


def predict(model: nn.Module, loader: DataLoader, device: torch.device) -> np.ndarray:
    output = []
    model.eval()
    with torch.inference_mode():
        for xb, _ in loader:
            probabilities = torch.softmax(model(xb.to(device)), dim=1)[:, 1]
            output.append(probabilities.cpu().numpy())
    return np.concatenate(output).astype(float)


def binary_metrics(y_true: np.ndarray, scores: np.ndarray) -> dict[str, float | int]:
    predictions = (scores >= 0.50).astype(np.int64)
    return {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, predictions)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "average_precision": float(average_precision_score(y_true, scores)),
        "roc_auc": float(roc_auc_score(y_true, scores)),
        "tn": int(((y_true == 0) & (predictions == 0)).sum()),
        "fp": int(((y_true == 0) & (predictions == 1)).sum()),
        "fn": int(((y_true == 1) & (predictions == 0)).sum()),
        "tp": int(((y_true == 1) & (predictions == 1)).sum()),
        "threshold": 0.50,
    }


def evaluate_split(
    model: nn.Module,
    device: torch.device,
    X: np.ndarray,
    y: np.ndarray,
    indices: np.ndarray,
    attack_types: np.ndarray,
    event_ids: np.ndarray,
    split_name: str,
) -> dict:
    y_split = y[indices]
    scores = predict(model, make_loader(X[indices], y_split), device)
    predictions = (scores >= 0.50).astype(np.int64)
    frame = pd.DataFrame(
        {
            "true_id": y_split,
            "pred_id": predictions,
            "spoofing_probability": scores,
            "attack_type": attack_types[indices],
            "scenario_id": event_ids[indices],
        }
    )
    frame.to_csv(MODEL_DIR / f"{split_name}_sequence_predictions.csv", index=False)

    normal_mask = frame["attack_type"].astype(str).str.lower().isin(
        ["normal", "normal_random"]
    )
    per_attack = []
    for attack in ATTACKS:
        subset = frame.loc[
            normal_mask | frame["attack_type"].astype(str).str.lower().eq(attack)
        ]
        if not bool((subset["true_id"] == 1).any()):
            continue
        metrics = binary_metrics(
            subset["true_id"].to_numpy(dtype=np.int64),
            subset["spoofing_probability"].to_numpy(dtype=float),
        )
        per_attack.append({"attack_type": attack, **metrics})
    pd.DataFrame(per_attack).to_csv(
        MODEL_DIR / f"{split_name}_per_attack_metrics.csv", index=False
    )
    return {
        "split": split_name,
        "sequence_metrics": binary_metrics(y_split, scores),
        "per_attack_metrics": per_attack,
        "windows": int(len(frame)),
    }


def train() -> None:
    """Train only when explicitly invoked by the user."""
    set_seed(SEED)
    arrays = load_data()
    X = arrays["X"]
    y = arrays["y"]
    groups = arrays["groups"]
    assert isinstance(X, np.ndarray)
    assert isinstance(y, np.ndarray)
    assert isinstance(groups, np.ndarray)

    split = group_train_val_test_split(
        X,
        y,
        groups,
        val_size=0.15,
        test_size=0.15,
        random_state=SEED,
        stratify_groups=True,
        max_tries=400,
        mixed_label_groups=True,
    )
    X_scaled, scaler = scaled_arrays(X, split)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, MODEL_DIR / "scaler.joblib")
    np.savez_compressed(
        MODEL_DIR / "split_indices.npz",
        train_idx=split.train_idx,
        val_idx=split.val_idx,
        test_idx=split.test_idx,
    )

    train_groups = set(groups[split.train_idx].tolist())
    val_groups = set(groups[split.val_idx].tolist())
    test_groups = set(groups[split.test_idx].tolist())
    overlap = {
        "train_validation": sorted(train_groups & val_groups),
        "train_test": sorted(train_groups & test_groups),
        "validation_test": sorted(val_groups & test_groups),
    }
    if any(overlap.values()):
        raise RuntimeError(f"Source-group leakage detected: {overlap}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PaperGRU(input_size=int(X.shape[-1])).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    train_counts = np.bincount(y[split.train_idx], minlength=2).astype(np.float32)
    class_weights = train_counts.sum() / np.clip(2.0 * train_counts, 1.0, None)
    criterion = nn.CrossEntropyLoss(
        weight=torch.as_tensor(class_weights, dtype=torch.float32, device=device)
    )
    train_loader = make_train_loader(X_scaled[split.train_idx], y[split.train_idx])
    val_loader = make_loader(X_scaled[split.val_idx], y[split.val_idx])

    history = []
    best_loss = float("inf")
    best_epoch = 0
    no_improve = 0
    for epoch in range(1, 51):
        model.train()
        total_loss = 0.0
        seen = 0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * int(len(yb))
            seen += int(len(yb))

        model.eval()
        val_loss = 0.0
        val_seen = 0
        with torch.inference_mode():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                loss = criterion(model(xb), yb)
                val_loss += float(loss.item()) * int(len(yb))
                val_seen += int(len(yb))
        train_loss = total_loss / max(seen, 1)
        val_loss = val_loss / max(val_seen, 1)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        print(
            f"[paper-gru] epoch={epoch:02d} train_loss={train_loss:.6f} "
            f"val_loss={val_loss:.6f}"
        )

        if val_loss < best_loss - 1e-6:
            best_loss = val_loss
            best_epoch = epoch
            no_improve = 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "model_type": "paper_gru",
                    "input_size": int(X.shape[-1]),
                    "sequence_length": int(X.shape[1]),
                    "attacks": list(ATTACKS),
                    "seed": SEED,
                    "best_epoch": best_epoch,
                    "best_validation_loss": best_loss,
                    "paper_parameters": {
                        "gru_units": 64,
                        "gru_dropout": 0.30,
                        "dense_units": 32,
                        "dense_dropout": 0.20,
                        "learning_rate": 0.001,
                        "batch_size": 32,
                        "epochs": 50,
                        "early_stopping_patience": 10,
                    },
                },
                MODEL_PATH,
            )
        else:
            no_improve += 1
            if no_improve >= 10:
                print(f"[paper-gru] early stopping at epoch {epoch}")
                break

    (MODEL_DIR / "history.json").write_text(
        json.dumps(history, indent=2), encoding="utf-8"
    )
    checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    event_ids = arrays["event_ids"]
    attack_types = arrays["attack_types"]
    assert isinstance(event_ids, np.ndarray)
    assert isinstance(attack_types, np.ndarray)
    validation = evaluate_split(
        model, device, X_scaled, y, split.val_idx, attack_types, event_ids, "validation"
    )
    test = evaluate_split(
        model, device, X_scaled, y, split.test_idx, attack_types, event_ids, "test"
    )
    report = {
        "experiment": "paper_style_gru_six_attacks_with_mirroring",
        "comparison_only": True,
        "paper_difference": (
            "Adds mirroring and uses original_mmsi-disjoint train/validation/test groups. "
            "Uses this repository's leakage-safe kinematic feature set rather than absolute LAT/LON."
        ),
        "training_data": str(NPZ_PATH.resolve()),
        "feature_columns": arrays["feature_cols"].tolist(),
        "group_overlap": overlap,
        "best_epoch": int(best_epoch),
        "validation": validation,
        "test": test,
    }
    (MODEL_DIR / "evaluation_summary.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(f"[paper-gru] model -> {MODEL_PATH}")
    print(f"[paper-gru] report -> {MODEL_DIR / 'evaluation_summary.json'}")


def status() -> None:
    print(f"run_root={RUN_ROOT}")
    print(f"generated={bool(list(GENERATED.glob('spoofed_*.csv')))}")
    print(f"preprocessed={NPZ_PATH.is_file()}")
    print(f"trained={MODEL_PATH.is_file()}")
    print(f"evaluated={(MODEL_DIR / 'evaluation_summary.json').is_file()}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Paper-style GRU comparator for six spoofing attacks."
    )
    parser.add_argument("action", choices=["prepare", "train", "status"])
    args = parser.parse_args()
    if args.action == "prepare":
        prepare()
    elif args.action == "train":
        train()
    else:
        status()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
