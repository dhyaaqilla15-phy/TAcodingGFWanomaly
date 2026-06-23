# Protokol tuning Go-Dark

## Tujuan

Menaikkan generalisasi Go-Dark tanpa memakai external test untuk memilih model,
hyperparameter, epoch, atau threshold.

## Data dan pembagian

- Internal hanya dibagi menjadi train dan validation secara vessel/group-aware.
- Internal test harus kosong.
- External enriched dipakai seluruhnya sebagai final test.
- Internal dan external hanya memakai `drifting_longlines`, `fixed_gear`,
  `purse_seines`, dan `trawlers`.
- MMSI internal dan external harus tidak overlap.

## Perubahan model dan training

- Metadata `window_source_labels` ikut disimpan di NPZ untuk audit dan sampling.
- Source tidak menjadi input feature, sehingga model tidak bisa mengambil jalan
  pintas dengan membaca identitas dataset.
- Training sampler memakai inverse frequency untuk setiap pasangan
  `source_class x label`.
- Context/hard-negative model memakai sample weight pasangan yang sama.
- Class weight dibuat netral saat sampler source-aware aktif agar imbalance tidak
  dikoreksi dua kali.
- Kandidat model dibuat lebih kecil: hidden size 64-256, satu recurrent layer,
  projection 64-128, dan embedding 128-256.
- Semua trial berjalan 50 epoch tanpa early stopping.
- Generator mengambil seluruh vessel eligible dan membagi kandidat sintetis ke
  strata durasi blackout, cadence ping, dan hidden distance. Pemilihan kandidat
  mengutamakan strata yang masih kurang terwakili.

## Seleksi internal

1. Bentuk minimal tiga fold vessel yang disjoint dan source-stratified.
2. Setiap sample internal menjadi validation tepat satu kali sehingga pooled
   predictions benar-benar out-of-fold, bukan repeated random holdout.
3. Fit Platt calibration hanya pada pooled internal OOF probabilities.
4. Cari satu threshold bersama setelah kalibrasi.
5. Pilih konfigurasi berdasarkan macro event F1 antar-source dengan minimum
   source recall 0.50 sebagai constraint; overall event F1 menjadi tie-breaker.
6. Tulis winner ke `winner_internal_only.json` sebelum external dibuka.

## Final external test

- Mode `external` menolak berjalan jika manifest winner internal belum ada.
- Hanya konfigurasi pemenang dan threshold internal yang digunakan.
- External test dievaluasi pada semua seed pemenang untuk melaporkan mean dan
  standard deviation, bukan untuk memilih seed terbaik.
- Confusion matrix dan metrik per sumber kapal dibuat oleh evaluator untuk setiap
  hasil final.
- Kebijakan deployment adalah mean calibrated probability dari seluruh fold
  model; seed tidak dipilih menggunakan external.
- Laporan ensemble menyimpan 95% confidence interval dari 2.000 bootstrap
  resampling pada cluster vessel, distratifikasi per source.
- Dataset external yang sudah pernah dibuka untuk merancang iterasi berikutnya
  harus disebut external-development. Klaim final unbiased membutuhkan holdout
  external baru (kapal/waktu/region berbeda) yang belum pernah diperiksa.

## Command

```bash
bash run_godark_external_test_pipeline.sh prepare
DEVICE=cuda python run_godark_hparam_tuning.py search
DEVICE=cuda python run_godark_hparam_tuning.py external
```

## Manipulation-point ablation

- Jumlah event dibandingkan pada level 1, 2, dan 3 per vessel.
- Posisi dibandingkan pada sepertiga awal, tengah, dan akhir trajectory vessel.
- Semua level dibentuk dari master candidate bank yang sama.
- Hard-negative pool, model `compact_h128`, seed, dan fold dibuat tetap.
- Ranking memakai macro-source F1 dan minimum source recall dari internal OOF.
- External tidak dibaca oleh eksperimen ini.

```bash
python run_godark_manipulation_tuning.py prepare
DEVICE=cuda python run_godark_manipulation_tuning.py search
```
