from __future__ import annotations

import csv
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from data_preparation import timestamp_to_epoch_seconds


ROOT = Path(__file__).resolve().parent
TUNING_ROOT = ROOT / "Outputs" / "gear_tuning06_internal_hparam_gap12h_opfilter"
BASELINE_ROOT = (
    ROOT / "Outputs" / "gear_tuning04_gap12h_opfilter_1to12_geo0_multiseed"
)
FINAL_EVAL_ROOT = TUNING_ROOT / "stage3_external_final" / "baseline_geo0"
ANALYSIS_ROOT = TUNING_ROOT / "final_analysis"
EXTERNAL_DATA = ROOT / "Dataset_Test_Enriched"
SEEDS = (42, 43, 44, 45, 46)
TARGET_MMSI = {"525600095", "525600097", "412510000"}
CLASS_LABELS = [
    "drifting_longlines",
    "fixed_gear",
    "purse_seines",
    "trawlers",
]


def read_json(path: Path) -> dict | list:
    return json.loads(path.read_text(encoding="utf-8"))


def load_external() -> pd.DataFrame:
    frames = []
    for path in sorted(EXTERNAL_DATA.glob("*.csv")):
        frame = pd.read_csv(path)
        frame["file_label"] = path.stem
        frames.append(frame)
    data = pd.concat(frames, ignore_index=True)
    data["mmsi"] = (
        pd.to_numeric(data["mmsi"], errors="coerce")
        .astype("Int64")
        .astype(str)
    )
    data["timestamp_seconds"] = timestamp_to_epoch_seconds(data["timestamp"])
    for col in ("speed", "course", "lat", "lon", "is_fishing"):
        data[col] = pd.to_numeric(data[col], errors="coerce")
    return data


def prediction_audit() -> pd.DataFrame:
    vessels: dict[str, dict[str, object]] = defaultdict(
        lambda: {
            "true_label": "",
            "predictions": [],
            "confidences": [],
            "n_sequences": 0,
        }
    )
    for seed in SEEDS:
        path = (
            FINAL_EVAL_ROOT
            / f"seed_{seed}"
            / "external_test_eval"
            / "per_vessel_predictions.csv"
        )
        for row in csv.DictReader(path.open(encoding="utf-8-sig")):
            item = vessels[row["mmsi"]]
            item["true_label"] = row["true_label"]
            item["predictions"].append(row["pred_label"])
            item["confidences"].append(float(row["confidence"]))
            item["n_sequences"] = int(row["n_sequences"])

    rows = []
    for mmsi, item in vessels.items():
        predictions = list(item["predictions"])
        true_label = str(item["true_label"])
        votes = Counter(predictions)
        rows.append(
            {
                "mmsi": mmsi,
                "true_label": true_label,
                "correct_seeds": sum(p == true_label for p in predictions),
                "total_seeds": len(predictions),
                "prediction_votes": json.dumps(dict(votes), sort_keys=True),
                "mean_confidence": float(np.mean(item["confidences"])),
                "max_confidence": float(np.max(item["confidences"])),
                "n_sequences": int(item["n_sequences"]),
                "persistent_error": sum(p == true_label for p in predictions) == 0,
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["correct_seeds", "true_label", "mmsi"]
    )


def plot_confusion_matrix(
    matrix: np.ndarray,
    labels: list[str],
    title: str,
    path: Path,
    *,
    normalized: bool,
) -> None:
    values = matrix.astype(np.float64)
    if normalized:
        row_sums = values.sum(axis=1, keepdims=True)
        values = np.divide(
            values,
            row_sums,
            out=np.zeros_like(values),
            where=row_sums > 0,
        )

    fig, ax = plt.subplots(figsize=(8.5, 7))
    image = ax.imshow(
        values,
        interpolation="nearest",
        cmap="Blues",
        vmin=0.0,
        vmax=1.0 if normalized else None,
    )
    fig.colorbar(image, ax=ax)
    ax.set(
        xticks=np.arange(len(labels)),
        yticks=np.arange(len(labels)),
        xticklabels=labels,
        yticklabels=labels,
        xlabel="Predicted label",
        ylabel="True label",
        title=title,
    )
    plt.setp(ax.get_xticklabels(), rotation=35, ha="right")

    threshold = float(values.max()) / 2.0 if values.size else 0.0
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            text = f"{values[i, j]:.2f}" if normalized else str(int(matrix[i, j]))
            ax.text(
                j,
                i,
                text,
                ha="center",
                va="center",
                color="white" if values[i, j] > threshold else "black",
            )
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def create_final_confusion_matrices() -> None:
    out_dir = ANALYSIS_ROOT / "final_confusion_matrices"
    out_dir.mkdir(parents=True, exist_ok=True)

    seed46_eval = FINAL_EVAL_ROOT / "seed_46" / "external_test_eval"
    shutil.copy2(
        seed46_eval / "confusion_matrix.png",
        out_dir / "FINAL_TEST_confusion_matrix_seed46.png",
    )
    shutil.copy2(
        seed46_eval / "confusion_matrix_normalized.png",
        out_dir / "FINAL_TEST_confusion_matrix_seed46_normalized.png",
    )

    votes: dict[str, dict[str, object]] = defaultdict(
        lambda: {"true_label": "", "predictions": []}
    )
    for seed in SEEDS:
        path = (
            FINAL_EVAL_ROOT
            / f"seed_{seed}"
            / "external_test_eval"
            / "per_vessel_predictions.csv"
        )
        for row in csv.DictReader(path.open(encoding="utf-8-sig")):
            item = votes[row["mmsi"]]
            item["true_label"] = row["true_label"]
            item["predictions"].append(row["pred_label"])

    label_to_id = {label: idx for idx, label in enumerate(CLASS_LABELS)}
    matrix = np.zeros((len(CLASS_LABELS), len(CLASS_LABELS)), dtype=np.int64)
    consensus_rows = []
    for mmsi, item in sorted(votes.items()):
        counts = Counter(item["predictions"])
        # Deterministic tie-break by the fixed class order.
        prediction = max(
            CLASS_LABELS,
            key=lambda label: (counts[label], -label_to_id[label]),
        )
        true_label = str(item["true_label"])
        matrix[label_to_id[true_label], label_to_id[prediction]] += 1
        consensus_rows.append(
            {
                "mmsi": mmsi,
                "true_label": true_label,
                "consensus_prediction": prediction,
                "prediction_votes": json.dumps(dict(counts), sort_keys=True),
                "correct": prediction == true_label,
            }
        )

    pd.DataFrame(consensus_rows).to_csv(
        out_dir / "final_test_consensus_predictions.csv",
        index=False,
    )
    plot_confusion_matrix(
        matrix,
        CLASS_LABELS,
        "External Test Consensus Confusion Matrix (5 seeds, 21 vessels)",
        out_dir / "FINAL_TEST_confusion_matrix_consensus_5seeds.png",
        normalized=False,
    )
    plot_confusion_matrix(
        matrix,
        CLASS_LABELS,
        "External Test Consensus Confusion Matrix, Row Normalized",
        out_dir / "FINAL_TEST_confusion_matrix_consensus_5seeds_normalized.png",
        normalized=True,
    )
    (out_dir / "README.md").write_text(
        """# Final Test Confusion Matrices

- Use `FINAL_TEST_confusion_matrix_seed46_normalized.png` as the main
  confusion matrix for the representative final checkpoint.
- Use `FINAL_TEST_confusion_matrix_consensus_5seeds_normalized.png` as
  supporting evidence of prediction stability across seeds.
- The consensus matrix contains 21 unique vessels. It does not count the same
  vessel five times.
""",
        encoding="utf-8",
    )


def create_final_validation_confusion_matrices() -> None:
    out_dir = ANALYSIS_ROOT / "final_validation_confusion_matrices"
    out_dir.mkdir(parents=True, exist_ok=True)

    seed46_eval = BASELINE_ROOT / "seed_46" / "validation_eval"
    shutil.copy2(
        seed46_eval / "confusion_matrix.png",
        out_dir / "FINAL_VALIDATION_confusion_matrix_seed46.png",
    )
    shutil.copy2(
        seed46_eval / "confusion_matrix_normalized.png",
        out_dir / "FINAL_VALIDATION_confusion_matrix_seed46_normalized.png",
    )

    label_to_id = {label: idx for idx, label in enumerate(CLASS_LABELS)}
    matrix = np.zeros((len(CLASS_LABELS), len(CLASS_LABELS)), dtype=np.int64)
    pooled_rows = []
    for seed in SEEDS:
        path = (
            BASELINE_ROOT
            / f"seed_{seed}"
            / "validation_eval"
            / "per_vessel_predictions.csv"
        )
        for row in csv.DictReader(path.open(encoding="utf-8-sig")):
            true_label = row["true_label"]
            pred_label = row["pred_label"]
            matrix[label_to_id[true_label], label_to_id[pred_label]] += 1
            pooled_rows.append(
                {
                    "seed": seed,
                    "mmsi": row["mmsi"],
                    "true_label": true_label,
                    "pred_label": pred_label,
                    "correct": true_label == pred_label,
                }
            )

    pd.DataFrame(pooled_rows).to_csv(
        out_dir / "validation_pooled_5splits_predictions.csv",
        index=False,
    )
    plot_confusion_matrix(
        matrix,
        CLASS_LABELS,
        "Validation Confusion Matrix (5 splits, 115 vessel assignments)",
        out_dir / "FINAL_VALIDATION_confusion_matrix_pooled_5splits.png",
        normalized=False,
    )
    plot_confusion_matrix(
        matrix,
        CLASS_LABELS,
        "Validation Confusion Matrix, Row Normalized (5 splits)",
        out_dir
        / "FINAL_VALIDATION_confusion_matrix_pooled_5splits_normalized.png",
        normalized=True,
    )
    (out_dir / "README.md").write_text(
        """# Final Validation Confusion Matrices

- Use `FINAL_VALIDATION_confusion_matrix_seed46_normalized.png` as the main
  validation confusion matrix for the representative checkpoint in Chapter 4.
- The pooled five-split matrix is supporting evidence only.
- Validation membership changes with the split seed. Therefore, the pooled
  matrix contains 115 vessel assignments (23 per split), not 115 unique
  vessels, and it is not a majority-vote consensus matrix.
""",
        encoding="utf-8",
    )


def vessel_data_audit(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (label, mmsi), group in data.groupby(["file_label", "mmsi"]):
        group = group.sort_values("timestamp_seconds")
        operational = group[group["speed"].between(1.0, 12.0)].copy()
        dt_hours = group["timestamp_seconds"].diff() / 3600.0
        metadata = {}
        for col in (
            "vessel_name",
            "gear_label",
            "gear_raw_gfw",
            "gear_inferred",
            "gear_registry",
            "flag_gfw",
            "source_zip",
        ):
            values = group[col].dropna().astype(str).unique().tolist()
            metadata[col] = " | ".join(values)
        rows.append(
            {
                "mmsi": mmsi,
                "file_label": label,
                **metadata,
                "raw_points": len(group),
                "operational_points": len(operational),
                "operational_ratio": len(operational) / max(len(group), 1),
                "operational_speed_median": operational["speed"].median(),
                "operational_speed_p90": operational["speed"].quantile(0.90),
                "operational_fishing_ratio": operational["is_fishing"].mean(),
                "segments": group["seg_id"].nunique(),
                "gaps_over_12h": int((dt_hours > 12.0).sum()),
                "duration_days": (
                    group["timestamp_seconds"].max()
                    - group["timestamp_seconds"].min()
                )
                / 86400.0,
                "lat_min": group["lat"].min(),
                "lat_max": group["lat"].max(),
                "lon_min": group["lon"].min(),
                "lon_max": group["lon"].max(),
                "registry_support": bool(metadata["gear_registry"]),
                "needs_manual_label_review": mmsi == "412510000",
            }
        )
    return pd.DataFrame(rows).sort_values(["file_label", "mmsi"])


def plot_vessel(data: pd.DataFrame, mmsi: str, out_dir: Path) -> Path:
    group = data[data["mmsi"] == mmsi].sort_values("timestamp_seconds")
    label = str(group["file_label"].iloc[0])
    name = str(group["vessel_name"].dropna().iloc[0])
    operational = group[group["speed"].between(1.0, 12.0)]

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    scatter = axes[0].scatter(
        group["lon"],
        group["lat"],
        c=group["speed"],
        s=7,
        cmap="viridis",
        alpha=0.85,
    )
    axes[0].plot(group["lon"], group["lat"], linewidth=0.5, alpha=0.25)
    axes[0].set_title("Full raw trajectory")
    axes[0].set_xlabel("Longitude")
    axes[0].set_ylabel("Latitude")
    axes[0].grid(alpha=0.2)
    fig.colorbar(scatter, ax=axes[0], label="Speed (knots)")

    axes[1].scatter(
        operational["lon"],
        operational["lat"],
        c=operational["speed"],
        s=7,
        cmap="viridis",
        alpha=0.85,
    )
    axes[1].plot(
        operational["lon"],
        operational["lat"],
        linewidth=0.5,
        alpha=0.25,
    )
    axes[1].set_title("Points retained by 1-12 knot filter")
    axes[1].set_xlabel("Longitude")
    axes[1].set_ylabel("Latitude")
    axes[1].grid(alpha=0.2)

    fig.suptitle(f"{mmsi} - {name} - source label: {label}")
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{mmsi}_{label}_trajectory_audit.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def write_manifest() -> dict:
    winner = read_json(TUNING_ROOT / "stage2_winner.json")
    validation = list(winner["per_seed"])
    representative = max(validation, key=lambda row: float(row["macro_f1"]))
    models = []
    for seed in SEEDS:
        model_dir = BASELINE_ROOT / f"seed_{seed}" / "model_gear"
        models.append(
            {
                "seed": seed,
                "model_path": str(model_dir / "model.pt"),
                "scaler_path": str(model_dir / "scaler.joblib"),
                "best_epoch_path": str(model_dir / "best_epoch.json"),
                "validation_eval": str(
                    BASELINE_ROOT / f"seed_{seed}" / "validation_eval"
                ),
            }
        )
    manifest = {
        "status": "frozen",
        "task": "gear",
        "selection_rule": "highest mean vessel-level validation Macro-F1 across seeds 42-46",
        "external_used_for_selection": False,
        "configuration": {
            "gap_seconds": 43200,
            "operational_speed_min": 1.0,
            "operational_speed_max": 12.0,
            "use_location_features": True,
            "lr": winner["lr"],
            "hidden_size": winner["hidden_size"],
            "num_layers": 2,
            "input_proj_dim": 256,
            "embed_dim": 512,
            "dropout": winner["dropout"],
            "weight_decay": winner["weight_decay"],
            "class_weight_power": winner["class_weight_power"],
            "focal_gamma": winner["focal_gamma"],
            "geo_aux_weight": winner["geo_aux_weight"],
        },
        "validation_multiseed": {
            "mean_macro_f1": winner["mean_macro_f1"],
            "std_macro_f1": winner["std_macro_f1"],
            "mean_balanced_acc": winner["mean_balanced_acc"],
            "std_balanced_acc": winner["std_balanced_acc"],
            "mean_accuracy": winner["mean_accuracy"],
        },
        "representative_checkpoint": {
            "selection_source": "internal validation only",
            "seed": representative["seed"],
            "model_path": str(
                BASELINE_ROOT
                / f"seed_{representative['seed']}"
                / "model_gear"
                / "model.pt"
            ),
            "validation_macro_f1": representative["macro_f1"],
        },
        "all_seed_models": models,
        "reporting_policy": "Report mean and standard deviation across all five seeds; do not report only the representative checkpoint.",
    }
    path = ANALYSIS_ROOT / "final_model_manifest.json"
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def write_report(
    manifest: dict,
    prediction_df: pd.DataFrame,
    vessel_df: pd.DataFrame,
) -> None:
    external = read_json(
        TUNING_ROOT
        / "stage3_external_final"
        / "final_external_summary.json"
    )
    persistent = prediction_df[prediction_df["persistent_error"]]
    audit = vessel_df[vessel_df["mmsi"].isin(TARGET_MMSI)].set_index("mmsi")
    table_columns = [
        "mmsi",
        "true_label",
        "prediction_votes",
        "mean_confidence",
        "n_sequences",
    ]
    table_lines = [
        "| MMSI | True label | Prediction votes | Mean confidence | Sequences |",
        "|---|---|---|---:|---:|",
    ]
    for _, row in persistent[table_columns].iterrows():
        table_lines.append(
            f"| {row['mmsi']} | {row['true_label']} | "
            f"`{row['prediction_votes']}` | {float(row['mean_confidence']):.3f} | "
            f"{int(row['n_sequences'])} |"
        )
    persistent_table = "\n".join(table_lines)
    report = f"""# Final Gear Model Analysis

## Frozen Configuration

- Winner: `baseline_geo0`
- Gap: 12 hours
- Operational filter: 1-12 knots
- Location features: enabled
- Learning rate: {manifest["configuration"]["lr"]}
- Hidden size: {manifest["configuration"]["hidden_size"]}
- Dropout: {manifest["configuration"]["dropout"]}
- Focal gamma: {manifest["configuration"]["focal_gamma"]}
- Geo auxiliary weight: {manifest["configuration"]["geo_aux_weight"]}

The configuration was selected using internal validation only.

## Internal Validation, Five Seeds

- Vessel Macro-F1: {manifest["validation_multiseed"]["mean_macro_f1"]:.4f} +/- {manifest["validation_multiseed"]["std_macro_f1"]:.4f}
- Balanced accuracy: {manifest["validation_multiseed"]["mean_balanced_acc"]:.4f} +/- {manifest["validation_multiseed"]["std_balanced_acc"]:.4f}
- Accuracy: {manifest["validation_multiseed"]["mean_accuracy"]:.4f}

## External Evaluation, Five Seeds

- Vessel Macro-F1: {external["mean_metrics_vessel"]["macro_f1"]:.4f}
- Balanced accuracy: {external["mean_metrics_vessel"]["balanced_acc"]:.4f}
- Accuracy: {external["mean_metrics_vessel"]["accuracy"]:.4f}
- Weighted F1: {external["mean_metrics_vessel"]["weighted_f1"]:.4f}
- External vessels: 21
- External sequences: 3378

## Persistent Errors

{persistent_table}

## Label Audit

- `525600095` and `525600097` are supported by GFW raw, inferred, and registry labels as drifting longlines. Do not relabel them solely because the model disagrees.
- `412510000` is labeled fixed gear by GFW raw/inferred but has no registry support in this dataset. Its retained operational median speed is {audit.loc["412510000", "operational_speed_median"]:.1f} knots and its trajectory spans a very large geographic range. This vessel requires manual source verification.
- No MMSI appears in more than one external class file.

## Decision

Hyperparameter tuning is complete for the current research scope. Additional random hyperparameter trials are not recommended. The next improvement should come from label verification and adding independent drifting-longline vessels, especially examples behaviorally similar to `525600095` and `525600097`.

The representative checkpoint is seed {manifest["representative_checkpoint"]["seed"]}, selected only from internal validation. Scientific reporting must still use the five-seed mean and standard deviation.
"""
    (ANALYSIS_ROOT / "FINAL_GEAR_ANALYSIS.md").write_text(
        report,
        encoding="utf-8",
    )


def write_indonesian_summary() -> None:
    winner = read_json(TUNING_ROOT / "stage2_winner.json")
    external = read_json(
        TUNING_ROOT
        / "stage3_external_final"
        / "final_external_summary.json"
    )
    summary = f"""# Ringkasan Hasil Final Klasifikasi Gear

## Bab 4 - Pelatihan dan Validasi Internal

Konfigurasi final dipilih hanya menggunakan validation internal pada seed
42-46. Data external tidak digunakan untuk memilih hyperparameter.

Konfigurasi final:

- gap trajectory: 12 jam;
- filter kecepatan operasional: 1-12 knot;
- fitur jarak pantai dan pelabuhan: digunakan;
- learning rate: {winner["lr"]};
- hidden size: {winner["hidden_size"]};
- dropout: {winner["dropout"]};
- focal gamma: {winner["focal_gamma"]};
- class-weight power: {winner["class_weight_power"]};
- geo auxiliary weight: {winner["geo_aux_weight"]}.

Hasil validation vessel-level lima seed:

| Metrik | Rata-rata | Standar deviasi |
|---|---:|---:|
| Macro-F1 | {winner["mean_macro_f1"]:.4f} | {winner["std_macro_f1"]:.4f} |
| Balanced accuracy | {winner["mean_balanced_acc"]:.4f} | {winner["std_balanced_acc"]:.4f} |
| Accuracy | {winner["mean_accuracy"]:.4f} | - |

Baseline Geo0 tetap menjadi pemenang. Pengurangan hidden size menjadi 256 dan
dropout menjadi 0,20 tidak meningkatkan rata-rata validation Macro-F1.

## Bab 5 - Pengujian External

Dataset external terdiri atas 21 kapal dan 3.378 sequence. Hasil vessel-level
lima seed:

| Metrik | Rata-rata |
|---|---:|
| Accuracy | {external["mean_metrics_vessel"]["accuracy"]:.4f} |
| Macro-F1 | {external["mean_metrics_vessel"]["macro_f1"]:.4f} |
| Balanced accuracy | {external["mean_metrics_vessel"]["balanced_acc"]:.4f} |
| Weighted F1 | {external["mean_metrics_vessel"]["weighted_f1"]:.4f} |

Confusion matrix utama untuk Bab 5 adalah confusion matrix normalized external
test seed 46 karena seed tersebut digunakan sebagai checkpoint representatif.
Confusion matrix konsensus lima seed dapat ditampilkan sebagai analisis
pendukung kestabilan; matrix konsensus tetap berisi 21 kapal unik.

Confusion matrix utama untuk Bab 4 menggunakan validation seed 46 agar sesuai
dengan checkpoint representatif. Matrix pooled lima split hanya digunakan
sebagai pendukung karena anggota validation berubah pada setiap seed.

Performa per kelas menunjukkan bahwa purse seine dan trawler paling konsisten.
Kelemahan utama terdapat pada drifting longlines dan fixed gear. Tiga kapal
selalu salah pada kelima seed:

- `525600095`, drifting longlines, diprediksi fixed gear/trawler;
- `525600097`, drifting longlines, selalu diprediksi fixed gear;
- `412510000`, fixed gear, diprediksi trawler/drifting longlines.

## Audit Data

Label `525600095` dan `525600097` didukung oleh label GFW raw, inferred, dan
registry. Keduanya tidak boleh diganti label hanya karena model salah. Plot
menunjukkan perjalanan lurus yang panjang sebelum atau sesudah area operasi.
Filter 1-12 knot masih mempertahankan sebagian besar perjalanan tersebut.

`412510000` tidak memiliki dukungan registry pada dataset ini. Kapal tersebut
memiliki median kecepatan operasional 10,5 knot dan lintasan lintas wilayah yang
sangat luas selama sekitar 111 hari. Label dan penyusunan track kapal ini perlu
diperiksa kembali ke sumber sebelum dipakai sebagai dasar kesimpulan kelas
fixed gear.

Tidak ditemukan MMSI yang muncul pada lebih dari satu file kelas external.

## Keputusan Penelitian

Tuning hyperparameter gear dinyatakan selesai untuk lingkup penelitian ini.
Percobaan hyperparameter tambahan tidak direkomendasikan karena kegagalan
tersisa konsisten pada kapal yang sama dan tidak hilang melalui perubahan
learning rate, hidden size, dropout, focal gamma, atau class weighting.

Langkah berikutnya:

1. verifikasi manual label dan track `412510000`;
2. pertahankan label dua kapal longline yang didukung registry;
3. tambahkan kapal longline independen dengan pola perjalanan serupa, bukan
   menambah window dari kapal yang sudah sama;
4. gunakan rata-rata dan standar deviasi lima seed dalam laporan;
5. gunakan seed 46 hanya sebagai checkpoint representatif, bukan sebagai
   satu-satunya angka hasil penelitian.
"""
    (ANALYSIS_ROOT / "RINGKASAN_BAB4_BAB5_GEAR.md").write_text(
        summary,
        encoding="utf-8",
    )


def main() -> None:
    ANALYSIS_ROOT.mkdir(parents=True, exist_ok=True)
    data = load_external()
    predictions = prediction_audit()
    vessel_audit = vessel_data_audit(data)
    combined = predictions.merge(
        vessel_audit,
        on="mmsi",
        how="left",
    )
    predictions.to_csv(
        ANALYSIS_ROOT / "external_prediction_stability.csv",
        index=False,
    )
    vessel_audit.to_csv(
        ANALYSIS_ROOT / "external_vessel_data_audit.csv",
        index=False,
    )
    combined[combined["mmsi"].isin(TARGET_MMSI)].to_csv(
        ANALYSIS_ROOT / "problematic_vessels_audit.csv",
        index=False,
    )
    plot_dir = ANALYSIS_ROOT / "trajectory_plots"
    for mmsi in sorted(TARGET_MMSI):
        plot_vessel(data, mmsi, plot_dir)
    manifest = write_manifest()
    write_report(manifest, predictions, vessel_audit)
    write_indonesian_summary()
    create_final_confusion_matrices()
    create_final_validation_confusion_matrices()
    print(f"[gear-final-analysis] saved: {ANALYSIS_ROOT}")


if __name__ == "__main__":
    main()
