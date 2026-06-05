# AIS/GFW Gear Classification, Spoofing, and Go-Dark

Project ini untuk skripsi: membaca dataset AIS/GFW, membuat sequence trajectory, lalu menjalankan tiga pipeline terpisah:

1. **Gear classification**: deteksi jenis fishing gear kapal.
2. **Spoofing detection**: generate dan deteksi trajectory GPS/AIS yang dimanipulasi.
3. **Go-dark detection**: generate dan deteksi kapal yang menghilang dari AIS/GPS.

Versi ini sudah **menghapus command `predict`**. Untuk kebutuhan skripsi, alur berhenti di `eval`, karena confusion matrix, metrics, dan tabel prediksi evaluasi sudah keluar dari `eval`.

## Struktur utama

```text
main.py
data_preparation.py
train.py
eval.py
suspected.py
spoofing_simulator.py
plot_spoofing.py
go_dark_simulator.py
plot_go_dark.py
plot_trajectory.py
model.py
metrics.py
requirements.txt
```

## Install

```bash
pip install -r requirements.txt
```

## Catatan split train/val/test

Split train/validation/test dibuat saat tahap `train`, bukan saat generator CSV dijalankan. Split ini berbasis `vessel/MMSI`, jadi kapal yang masuk train tidak ikut masuk validation/test. Validation dipakai untuk memilih best epoch, `tau`, dan agregasi; test hanya dipakai sekali saat `eval`.

Scaler di-fit hanya dari train split, lalu dipakai untuk validation/test. Ini mengurangi leakage distribusi test ke training.

Untuk metric validasi:

- `gear` tetap memakai metric utama vessel-level.
- `spoofing` dan `go-dark` memakai metric utama sequence-level saat training/eval, karena satu vessel bisa mengandung sangat banyak window normal dan hanya sedikit window anomali.

---

# 1. Gear classification

Dataset GFW biasanya sudah dipisah per file gear, misalnya:

```text
trawlers.csv
trollers.csv
fixed_gear.csv
purse_seines.csv
drifting_longlines.csv
pole_and_line.csv
unknown.csv
```

## Preprocess gear

```bash
python3 main.py preprocess \
  --data_dir "Dataset" \
  --out_dir outputs_gear \
  --task gear \
  --exclude_labels unknown
```

Output:

```text
outputs_gear/processed_gear.npz
```

`processed_gear.npz` menyimpan fitur mentah. Scaler dibuat saat `train` di folder model.

## Train gear

```bash
python3 main.py train \
  --data_npz outputs_gear/processed_gear.npz \
  --out_dir outputs_gear/model_gear \
  --device auto \
  --random_state 42
```

Default training sekarang memakai dropout lebih kuat, 1 blok self-attention setelah LSTM, dan auxiliary Haversine loss kecil dari koordinat window:

```bash
python3 main.py train \
  --data_npz outputs_gear/processed_gear.npz \
  --out_dir outputs_gear/model_gear \
  --dropout 0.40 \
  --attention_heads 4 \
  --attention_layers 1 \
  --geo_aux_weight 0.03
```

Jika Haversine auxiliary loss membuat validasi turun, coba `--geo_aux_weight 0.01` atau matikan dengan `--geo_aux_weight 0`.

## Evaluate gear

```bash
python3 main.py eval \
  --data_npz outputs_gear/processed_gear.npz \
  --model_path outputs_gear/model_gear/model.pt \
  --out_dir outputs_gear/model_gear \
  --device auto
```

Output evaluasi utama:

```text
outputs_gear/model_gear/confusion_matrix.png
outputs_gear/model_gear/confusion_matrix_normalized.png
outputs_gear/model_gear/per_vessel_predictions.csv
outputs_gear/model_gear/eval_summary.json
outputs_gear/model_gear/suspected_model.csv
outputs_gear/model_gear/scaler.joblib
outputs_gear/model_gear/split_indices.npz
```

Untuk skripsi, cukup berhenti di `eval` karena `eval` sudah membandingkan **true label vs predicted label**.

---

# 2. Spoofing pipeline

Spoofing berarti posisi/trajectory kapal dimanipulasi. Modul ini membuat data spoofing sintetis dari AIS/GFW normal.

Jenis spoofing yang tersedia:

```text
gradual_drift
location_jump
replay
meaconing
ghost
mirroring
```

## Generate spoofing

```bash
python3 main.py make_spoofing \
  --input_path "Dataset" \
  --out_dir outputs_spoofing \
  --attacks gradual_drift location_jump replay meaconing ghost mirroring \
  --limit_rows 300000 \
  --normal_keep_frac 0.25 \
  --max_vessels_per_file 20 \
  --points_per_attack 120 \
  --combine_outputs \
  --seed 42
```

Output:

```text
outputs_spoofing/spoofed_all.csv
outputs_spoofing/summaries/summary_<nama_file>.csv
```

## Plot spoofing

```bash
python3 main.py plot_spoofing \
  --csv_path outputs_spoofing/spoofed_all.csv \
  --out_dir outputs_spoofing/plots
```

## Heatmap spoofing

```bash
python3 main.py heatmap_spoofing \
  --csv_path outputs_spoofing/spoofed_all.csv \
  --out_dir outputs_spoofing/heatmaps \
  --log_scale
```

## Preprocess spoofing

```bash
python3 main.py preprocess \
  --data_dir outputs_spoofing \
  --out_dir outputs_spoofing \
  --task spoofing
```

## Plot trajectory spoofing setelah preprocess

```bash
python3 main.py plot_preprocessed \
  --npz_path outputs_spoofing/processed_spoofing.npz \
  --out_dir outputs_spoofing/plots/preprocessed \
  --task spoofing
```

## Train spoofing

```bash
python3 main.py train \
  --data_npz outputs_spoofing/processed_spoofing.npz \
  --out_dir outputs_spoofing/model_spoofing \
  --device auto \
  --random_state 42
```

## Evaluate spoofing

```bash
python3 main.py eval \
  --data_npz outputs_spoofing/processed_spoofing.npz \
  --model_path outputs_spoofing/model_spoofing/model.pt \
  --out_dir outputs_spoofing/model_spoofing \
  --device auto
```

Output penting:

```text
outputs_spoofing/model_spoofing/confusion_matrix.png
outputs_spoofing/model_spoofing/confusion_matrix_normalized.png
outputs_spoofing/model_spoofing/per_vessel_predictions.csv
outputs_spoofing/model_spoofing/eval_summary.json
```

---

# 3. Go-dark pipeline

Go-dark berarti kapal menghilang dari AIS/GPS selama periode tertentu, lalu muncul lagi. Modul ini membuat gap sintetis dengan menghapus segmen trajectory dan menyimpan segmen tersembunyi sebagai ground truth.

## Generate go-dark

```bash
python3 main.py make_godark \
  --input_path "Dataset" \
  --out_dir outputs_godark \
  --limit_rows 300000 \
  --max_vessels_per_file 20 \
  --min_points_per_vessel 120 \
  --events_per_vessel 1 \
  --min_hidden_points 20 \
  --max_hidden_points 120 \
  --min_dark_seconds 3600 \
  --max_dark_seconds 604800 \
  --min_hidden_distance_km 0.5 \
  --label_after_points 30 \
  --combine_outputs \
  --seed 42
```

Output:

```text
outputs_godark/godark_all.csv
outputs_godark/events/events_godark_<nama_file>.csv
outputs_godark/hidden_truth/hidden_truth_godark_<nama_file>.csv
outputs_godark/summaries/summary_godark_<nama_file>.csv
```

## Plot go-dark

```bash
python3 main.py plot_godark \
  --csv_path outputs_godark/godark_all.csv \
  --out_dir outputs_godark/plots
```

## Heatmap go-dark

```bash
python3 main.py heatmap_godark \
  --csv_path outputs_godark/godark_all.csv \
  --out_dir outputs_godark/heatmaps \
  --log_scale
```

## Preprocess go-dark

```bash
python3 main.py preprocess \
  --data_dir outputs_godark \
  --out_dir outputs_godark \
  --task godark \
  --seq_len 120 \
  --stride 6 \
  --gap_seconds 86400 \
  --max_implied_knots 1000 \
  --min_points_per_vessel 80 \
  --spoofing_window_threshold 0.05
```

## Plot trajectory go-dark setelah preprocess

```bash
python3 main.py plot_preprocessed \
  --npz_path outputs_godark/processed_godark.npz \
  --out_dir outputs_godark/plots/preprocessed \
  --task godark
```

## Train go-dark

```bash
python3 main.py train \
  --data_npz outputs_godark/processed_godark.npz \
  --out_dir outputs_godark/model_godark \
  --device auto \
  --random_state 42
```

## Evaluate go-dark

```bash
python3 main.py eval \
  --data_npz outputs_godark/processed_godark.npz \
  --model_path outputs_godark/model_godark/model.pt \
  --out_dir outputs_godark/model_godark \
  --device auto
```

Output penting:

```text
outputs_godark/model_godark/confusion_matrix.png
outputs_godark/model_godark/confusion_matrix_normalized.png
outputs_godark/model_godark/per_vessel_predictions.csv
outputs_godark/model_godark/eval_summary.json
```

---

# Urutan skripsi yang rapi

```text
Dataset AIS/GFW asli
        ↓
Preprocess gear
        ↓
Train gear
        ↓
Eval gear
        ↓
Generate spoofing
        ↓
Preprocess spoofing
        ↓
Train spoofing
        ↓
Eval spoofing
        ↓
Generate go-dark
        ↓
Preprocess go-dark
        ↓
Train go-dark
        ↓
Eval go-dark
```

Jadi tidak ada lagi tahap `predict`. Semua hasil laporan utama diambil dari `eval`.

---

# 4. One-command pipeline

Kalau mau jalanin semua pipeline sekaligus dalam satu command, pakai script:

```bash
bash run_all_pipeline.sh
```

Default script ini akan:

- membaca data dari `Dataset`
- menyimpan hasil ke `output/run01`
- menjalankan `gear`, `spoofing`, dan `go-dark` sampai `eval`

Kalau mau ganti folder data dan output:

```bash
bash run_all_pipeline.sh Dataset output/run02
```

Kalau mau ganti device atau seed:

```bash
DEVICE=cuda SEED=43 bash run_all_pipeline.sh
```

---

# 5. Hyperparameter tuning

Kalau mau tuning hyperparameter tanpa edit kode berulang, command `train` sekarang sudah menerima argumen:

```bash
python3 main.py train \
  --data_npz output/run02/godark/processed_godark.npz \
  --out_dir output/experiments/godark_try01 \
  --device cuda \
  --epochs 120 \
  --batch_size 64 \
  --lr 0.0005 \
  --hidden_size 256 \
  --num_layers 2 \
  --dropout 0.2 \
  --optimizer adam
```

Optimizer yang tersedia:

- `adamw`
- `adam`
- `sgd`

Setiap run akan menyimpan:

- `model.pt`
- `best_epoch.json`
- `history.json`
- `train_config.json`
- `eval_summary.json` jika `eval` dijalankan

Kalau mau sweep beberapa kombinasi sekaligus dan menghitung waktu komputasi tiap percobaan:

```bash
DEVICE=cuda bash run_hparam_sweep.sh \
  output/run02/godark/processed_godark.npz \
  output/tuning/godark
```

Script ini akan membuat banyak folder percobaan dan satu file rekap:

```text
output/tuning/godark/summary.csv
```

Di `summary.csv` akan ada:

- optimizer
- learning rate
- batch size
- dropout
- hidden size
- num layers
- waktu train
- waktu eval
- macro F1
- balanced accuracy
- accuracy
