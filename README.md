# AIS/GFW Gear Classification, Spoofing, Go-Dark, and Transshipment Candidates

Project ini untuk skripsi: membaca dataset AIS/GFW, membuat sequence trajectory, lalu menjalankan empat pipeline terpisah:

1. **Gear classification**: deteksi jenis fishing gear kapal.
2. **Spoofing detection**: generate dan deteksi trajectory GPS/AIS yang dimanipulasi.
3. **Go-dark detection**: generate dan deteksi kapal yang menghilang dari AIS/GPS.
4. **Transshipment candidate detection**: deteksi kandidat encounter dua kapal dan loitering offshore berbasis AIS.

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
transshipment_detector.py
plot_transshipment.py
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
- `spoofing`, `go-dark`, dan `transshipment` memakai metric utama sequence-level/event-level saat training/eval, karena satu vessel/pair/event bisa mengandung banyak window normal dan sedikit window anomali.

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

Secara default pipeline gear final memakai 4 kelas training: `drifting_longlines`,
`fixed_gear`, `purse_seines`, dan `trawlers`. Label `unknown`, `pole_and_line`,
dan `trollers` dikeluarkan dari preprocess training.

```bash
python3 main.py preprocess \
  --data_dir "Dataset" \
  --out_dir outputs_gear \
  --task gear \
  --exclude_labels unknown pole_and_line trollers
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

Eksperimen utama hanya memakai `gradual_drift` dan `location_jump`, karena
keduanya dapat menghasilkan perubahan kinematik di dalam satu window.
`replay`, `meaconing`, `ghost`, dan `mirroring` memerlukan konteks historis,
identitas silang, atau referensi geospasial eksternal dan harus dilaporkan
sebagai eksperimen keterbatasan terpisah.

## Generate spoofing

Generator spoofing default-nya tidak memakai sumber `pole_and_line` dan `trollers`.

```bash
python3 main.py make_spoofing \
  --input_path "Dataset" \
  --out_dir outputs_spoofing \
  --attacks gradual_drift location_jump \
  --limit_rows 300000 \
  --exclude_labels pole_and_line trollers \
  --normal_keep_frac 0.50 \
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
outputs_spoofing/model_spoofing/per_vessel_predictions.csv
outputs_spoofing/model_spoofing/spoofing_sequence_predictions.csv
outputs_spoofing/model_spoofing/spoofing_attack_metrics.csv
outputs_spoofing/model_spoofing/eval_summary.json
```

Untuk eksperimen multiseed yang memisahkan internal train/validation dan
external test:

```bash
python3 run_spoofing_multiseed.py run
```

Sebelum multiseed final, jalankan sensitivity study magnitude jump yang hanya
memakai validation internal:

```bash
python3 run_spoofing_jump_sensitivity.py run
```

Pipeline spoofing otomatis menonaktifkan fitur jarak pantai/pelabuhan dan
geographic auxiliary loss karena koordinat sintetis tidak memiliki distance
raster yang dihitung ulang. Pemisahan split menggunakan `original_mmsi`, bukan
ID trajectory sintetis.

---

# 3. Go-dark pipeline

Pipeline ini mendeteksi **suspected AIS transmission gap**, bukan membuktikan
niat ilegal. Rule-based stage mencari gap yang memenuhi konteks penerimaan,
jarak pantai, dan ping sebelum gap. BiLSTM kemudian mengklasifikasikan satu
event dari konteks lintasan sebelum dan sesudah gap. Desain candidate-gap
mengikuti prinsip Welch et al. (2022), DOI `10.1126/sciadv.abq2109`.

## Protokol permanen Go-Dark

Go-Dark tidak memakai internal test. Domain dibentuk sebagai berikut:

```text
Dataset                     -> generate/preprocess -> train + validation
Dataset_Test_Enriched       -> generate/preprocess -> pure external test
```

Berdasarkan hasil EDA, kedua domain dikunci dengan allowlist yang sama dan
hanya memakai `drifting_longlines`, `fixed_gear`, `purse_seines`, dan
`trawlers`. `unknown`, `pole_and_line`, `trollers`, serta kelas baru lain tidak
akan ikut kecuali allowlist diubah secara eksplisit.

Filter pantai pipeline ditetapkan sama pada kedua domain sebesar 5 nautical
miles. Ambang 50 nm tidak dipakai karena seluruh purse seiner external berada
di bawah 50 nm dan akan terhapus total. Karena itu ruang lingkup klaim adalah
deteksi `suspected AIS gap`, bukan khusus high-seas AIS disabling.

Model, scaler, hard-negative classifier, dan threshold hanya dipelajari dari
internal train/validation. Seluruh external NPZ dievaluasi memakai
`--eval_split all`. Pipeline akan gagal jika MMSI overlap, skema fitur berbeda,
internal test tidak kosong, atau external evaluation melakukan tuning.

Persiapan data saja, tanpa training:

```bash
bash run_godark_external_test_pipeline.sh prepare
```

Training internal train/validation selama tepat 50 epoch tanpa early stopping.
Setelah training selesai, validation dan pure external evaluation otomatis
langsung dijalankan:

```bash
DEVICE=cuda bash run_godark_external_test_pipeline.sh train
```

Untuk mengulang evaluation saja tanpa training:

```bash
DEVICE=cuda bash run_godark_external_test_pipeline.sh evaluate
```

Jalankan seluruh tahap sekaligus hanya jika memang ingin training langsung:

```bash
DEVICE=cuda bash run_godark_external_test_pipeline.sh all
```

Output final:

```text
Outputs/godark_external01/data_internal_trainval/processed_godark.npz
Outputs/godark_external01/data_external_test/processed_godark.npz
Outputs/godark_external01/model_godark/model.pt
Outputs/godark_external01/validation_eval/eval_summary.json
Outputs/godark_external01/external_test_eval/eval_summary.json
Outputs/godark_external01/external_test_eval/confusion_matrix.png
Outputs/godark_external01/external_protocol_audit.json
```

Hanya satu confusion matrix Go-Dark dibuat per evaluasi, yaitu matrix event
anomaly-first `confusion_matrix.png`. Untuk laporan external gunakan file dari
folder `external_test_eval`.

Setiap kandidat gap memiliki ID yang dibentuk dari MMSI serta timestamp dua
boundary yang terlihat. Ground-truth `go_dark_event_id` hanya dipakai untuk
menentukan target saat membuat dataset, bukan untuk membentuk grup evaluasi.
Karena satu gap adalah satu context window, evaluasi event memakai:

```text
event positif jika:
go_dark_probability >= godark_event_prob_threshold
```

Threshold dipilih hanya dari validation lalu dibekukan pada test. Grid sengaja
dibatasi menjadi tiga probability threshold agar runtime lebih ringan dan
mengurangi multiple-comparison overfitting. Untuk diagnosis, baca
`per_godark_event_predictions.csv` dan `godark_event_error_breakdown.csv`,
terutama false positive pada `hard_negative_gap`.

---

# 4. Transshipment candidate pipeline

Transshipment di AIS tidak bisa membuktikan transfer ikan/barang secara langsung. Modul ini membuat **weak-label candidate events** berdasarkan pola literatur:

- **Encounter**: dua kapal dalam jarak <= 0.5 km, durasi >= 2 jam, median speed < 2 knot, dan jauh dari port/anchorage proxy.
- **Loitering**: satu kapal bergerak lambat < 2 knot selama >= 8 jam, minimal 20 nautical miles dari shore.
- Untuk akurasi yang lebih stabil, pipeline juga menyimpan `encounter_rule_score`, `loitering_rule_score`, dan `risk_score`, lalu evaluasi transshipment memakai pembanding LSTM, tabular Random Forest, dan hybrid rule+ML.
- Target default yang disarankan untuk eksperimen awal adalah binary `--transshipment_target any` = `normal` vs `potential_transshipment`. Gunakan `encounter`, `loitering`, atau `multiclass` kalau data positifnya sudah cukup.

## Protokol training/evaluasi transshipment

Pipeline resmi memisahkan domain sebelum event dibuat:

```text
Dataset / internal train      -> real + synthetic encounter
Dataset / internal validation -> real event saja, MMSI disjoint
Dataset_Test_Enriched          -> pure external real event, tanpa synthetic
```

Keempat sumber dikunci ke `drifting_longlines`, `fixed_gear`, `purse_seines`,
dan `trawlers`. Split internal source-stratified dilakukan pada MMSI sebelum
generator berjalan. Checkpoint dipilih dengan metrik event-level; synthetic
dilarang masuk validation maupun external test.

Persiapan seluruh data tanpa training:

```bash
bash run_transshipment_external_test_pipeline.sh prepare
```

Smoke test lima epoch, hanya membuka internal validation:

```bash
DEVICE=cuda bash run_transshipment_external_test_pipeline.sh smoke
```

Training 50 epoch tanpa early stopping, kemudian validation dan pure external
evaluation otomatis:

```bash
DEVICE=cuda bash run_transshipment_external_test_pipeline.sh train
```

Evaluasi ulang tanpa training:

```bash
DEVICE=cuda bash run_transshipment_external_test_pipeline.sh evaluate
```

Setelah baseline, tuning resmi dilakukan hanya dari internal real-event OOF.
Generator synthetic train mencakup variasi jarak `0.05-0.48 km`, durasi
`2-6 jam`, kecepatan `0.2-1.9 knot`, course jitter `+-30 derajat`, dan
penyeimbangan pasangan empat gear. Jumlah synthetic dibandingkan pada level
50, 125, dan seluruh event aman yang tersedia per fold.

```bash
# Setelah perubahan generator, siapkan ulang data lebih dahulu.
bash run_transshipment_external_test_pipeline.sh prepare

# Hanya membuat 3 fold vessel/pair-disjoint; tidak training.
python run_transshipment_oof_tuning.py prepare

# 3 synthetic-count variants x seeds 42/43/44 x 3 folds, 50 epoch.
DEVICE=cuda python run_transshipment_oof_tuning.py search
```

Mode `search` tidak membaca external. Ia melakukan cross-fitted Platt
calibration, memilih threshold dari pooled OOF dengan precision/recall floor,
dan menilai macro performance per event-kind serta source-pair. External yang
sudah pernah dilihat hanya boleh dibuka sebagai development benchmark:

```bash
DEVICE=cuda python run_transshipment_oof_tuning.py external
```

Output internal utama:

```text
Outputs/transshipment_tuning01_internal_oof/fold_audit.csv
Outputs/transshipment_tuning01_internal_oof/internal_oof_variant_ranking.csv
Outputs/transshipment_tuning01_internal_oof/winner_internal_only.json
Outputs/transshipment_tuning01_internal_oof/calibrators/
```

Checkpoint, history, scaler, dan validation output setiap training trial sengaja
dipisahkan dari data/laporan dan disimpan di:

```text
Outputs/transshipment_training01_internal_oof/runs/<variant>/seed_<seed>/fold_<fold>/
```

Lokasi ini dapat diganti tanpa mengubah folder data melalui environment
variable `TRANS_TRAIN_ROOT`.

Audit utama tersimpan di:

```text
Outputs/transshipment_external01/manifests/mmsi_split_audit.json
Outputs/transshipment_external01/data_internal_trainval/transshipment_protocol_audit.json
Outputs/transshipment_external01/validation_eval/eval_summary.json
Outputs/transshipment_external01/external_test_eval/eval_summary.json
```

Perintah manual di bawah hanya untuk eksplorasi generator. Jangan memakai satu
NPZ campuran untuk klaim validation/final karena synthetic dapat tercampur.

## Generate transshipment candidates

Generator transshipment memakai allowlist permanen hasil EDA: hanya
`drifting_longlines`, `fixed_gear`, `purse_seines`, dan `trawlers`.
`unknown`, `pole_and_line`, `trollers`, maupun file kelas baru lain otomatis
ditolak. Audit sumber tersimpan di
`outputs_transshipment/summaries/source_input_audit.csv`.

```bash
python3 main.py make_transshipment \
  --input_path "Dataset" \
  --out_dir outputs_transshipment \
  --mode both \
  --limit_rows 300000 \
  --include_labels drifting_longlines fixed_gear purse_seines trawlers \
  --exclude_labels pole_and_line trollers \
  --max_vessels_per_file 60 \
  --grid_minutes 10 \
  --encounter_distance_km 0.5 \
  --encounter_min_hours 2 \
  --encounter_max_speed_knots 2 \
  --loitering_min_hours 8 \
  --loitering_max_speed_knots 2 \
  --loitering_min_shore_nm 20 \
  --synthetic_encounters_per_file 250 \
  --combine_outputs \
  --seed 42
```

Output:

```text
outputs_transshipment/transshipment_all.csv
outputs_transshipment/events/events_transshipment_all.csv
outputs_transshipment/summaries/summary_transshipment.csv
```

## Plot transshipment candidates

```bash
python3 main.py plot_transshipment_examples \
  --csv_path outputs_transshipment/transshipment_all.csv \
  --out_dir outputs_transshipment/plots/events
```

## Preprocess transshipment

```bash
python3 main.py preprocess \
  --data_dir outputs_transshipment \
  --out_dir outputs_transshipment \
  --task transshipment \
  --transshipment_target any \
  --transshipment_feature_mode fair \
  --seq_len 24 \
  --stride 3 \
  --min_points_per_vessel 3
```

## Train transshipment

```bash
python3 main.py train \
  --data_npz outputs_transshipment/processed_transshipment.npz \
  --out_dir outputs_transshipment/model_transshipment \
  --device auto \
  --random_state 42 \
  --geo_aux_weight 0
```

## Evaluate transshipment

```bash
python3 main.py eval \
  --data_npz outputs_transshipment/processed_transshipment.npz \
  --model_path outputs_transshipment/model_transshipment/model.pt \
  --out_dir outputs_transshipment/model_transshipment \
  --device auto
```

Output penting:

```text
outputs_transshipment/model_transshipment/confusion_matrix.png
outputs_transshipment/model_transshipment/confusion_matrix_normalized.png
outputs_transshipment/model_transshipment/per_event_predictions.csv
outputs_transshipment/model_transshipment/per_event_predictions_hybrid.csv
outputs_transshipment/model_transshipment/transshipment_tabular.joblib
outputs_transshipment/model_transshipment/eval_summary.json
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
        ↓
Generate transshipment candidates
        ↓
Preprocess transshipment
        ↓
Train transshipment
        ↓
Eval transshipment
```

Jadi tidak ada lagi tahap `predict`. Semua hasil laporan utama diambil dari `eval`.

---

# 5. One-command pipeline

Kalau mau jalanin semua pipeline sekaligus dalam satu command, pakai script:

```bash
bash run_all_pipeline.sh
```

Default script ini akan:

- membaca data dari `Dataset`
- menyimpan hasil ke `output/run01`
- memproses gear dari full CSV (`GEAR_LIMIT_ROWS=0`) supaya split vessel-level stabil
- membatasi generator spoofing/go-dark/transshipment dengan `SOURCE_LIMIT_ROWS=300000`
- menjalankan `gear`, `spoofing`, `go-dark`, dan `transshipment` sampai `eval`

Kalau mau ganti folder data dan output:

```bash
bash run_all_pipeline.sh Dataset output/run02
```

Kalau mau ganti device atau seed:

```bash
DEVICE=cuda SEED=43 bash run_all_pipeline.sh
```

Kalau ingin mengubah batas baris generator tanpa memotong gear:

```bash
DEVICE=cuda SOURCE_LIMIT_ROWS=300000 bash run_all_pipeline.sh Dataset Outputs/run01
```

---

# 6. Hyperparameter tuning

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
- Go-Dark event precision/recall/F1 jika task yang dievaluasi adalah Go-Dark

---

# 7. Go-Dark: tuning internal dan external test terkunci

Workflow Go-Dark terbaru memakai empat sumber kapal hasil EDA saja:
`drifting_longlines`, `fixed_gear`, `purse_seines`, dan `trawlers`.
Setiap window menyimpan metadata sumber kapal, lalu training menyeimbangkan
gabungan `source_class x label`. Metadata sumber tidak dipakai sebagai fitur
model.

Karena format NPZ bertambah, siapkan ulang data terlebih dahulu:

```bash
bash run_godark_external_test_pipeline.sh prepare
```

Jalankan pencarian hyperparameter hanya pada internal train/validation:

```bash
DEVICE=cuda python run_godark_hparam_tuning.py search
```

Tahap ini menjalankan model compact selama 50 epoch tanpa early stopping pada
seed 42, 43, dan 44. Ketiga validation fold bersifat disjoint, vessel-level,
source-stratified, dan bersama-sama mencakup seluruh data internal tepat satu
kali. Probabilitas dikalibrasi dengan Platt scaling dari pooled OOF internal.
Threshold serta konfigurasi dipilih memakai macro-F1 antar-source dengan
minimum source recall sebagai constraint, bukan overall F1 yang dapat
didominasi satu jenis kapal. Hasil utamanya:

```text
Outputs/godark_tuning01_internal_oof/internal_multiseed_ranking.csv
Outputs/godark_tuning01_internal_oof/pooled_oof_threshold_sweep.csv
Outputs/godark_tuning01_internal_oof/winner_internal_only.json
Outputs/godark_tuning01_internal_oof/source_stratified_folds/fold_source_audit.csv
Outputs/godark_tuning01_internal_oof/oof_calibrators/
```

Data external tidak dibaca oleh mode `search`. Setelah winner internal sudah
terkunci, buka external test tepat satu tahap dengan:

```bash
DEVICE=cuda python run_godark_hparam_tuning.py external
```

Untuk final holdout baru yang belum pernah diperiksa:

```bash
DEVICE=cuda \
GODARK_EXTERNAL_NPZ=Outputs/godark_final_holdout/data_external_test/processed_godark.npz \
python run_godark_hparam_tuning.py external
```

Threshold pada external test selalu berasal dari pooled internal validation;
external test tidak melakukan threshold tuning atau pemilihan model. Ringkasan
akhir disimpan di:

```text
Outputs/godark_tuning01_internal_oof/final_external_summary.json
Outputs/godark_tuning01_internal_oof/final_external_winner_results.csv
Outputs/godark_tuning01_internal_oof/final_external_ensemble/ensemble_summary.json
Outputs/godark_tuning01_internal_oof/final_external_ensemble/confusion_matrix.png
```

Ensemble memakai mean calibrated probability seluruh fold model dan kebijakan
ini dikunci sebelum external dibuka. Laporan ensemble menyertakan 95% bootstrap
confidence interval dari 2.000 resampling cluster kapal yang distratifikasi
per source.

## Go-Dark manipulation-point sensitivity

Eksperimen ini menjawab pengaruh jumlah dan posisi titik manipulasi tanpa
memakai external test. Satu master NPZ yang sama disubset menjadi enam varian:

```text
event count: 1, 2, 3 per vessel
trajectory position: early, middle, late
```

Posisi didefinisikan dari fraksi temporal relatif di trajectory setiap vessel:
`[0,1/3)`, `[1/3,2/3)`, dan `[2/3,1]`. Semua varian memakai model pemenang
`compact_h128`, tiga source-stratified OOF folds, dan hard-negative pool yang
sama agar hanya faktor manipulasi yang berubah.

```bash
bash run_godark_external_test_pipeline.sh prepare
python run_godark_manipulation_tuning.py prepare
DEVICE=cuda python run_godark_manipulation_tuning.py search
```

Rekap untuk laporan dosen:

```text
Outputs/godark_manipulation_tuning01/variant_data_audit.csv
Outputs/godark_manipulation_tuning01/manipulation_sensitivity_summary.csv
Outputs/godark_manipulation_tuning01/manipulation_sensitivity_best_internal.json
```

Eksperimen ini adalah sensitivity/ablation internal, bukan alasan membuka atau
menyetel ulang external test.
