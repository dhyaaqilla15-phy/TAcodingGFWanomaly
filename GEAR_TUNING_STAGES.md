# Gear Hyperparameter Tuning

Alur ini menjaga external test agar tidak dipakai memilih hyperparameter.

## Tahap 1: Internal Search

Menjalankan 13 konfigurasi pada split internal seed 42. Kandidat mencakup
baseline Geo0 (`geo_aux_weight=0`) dan baseline ori (`geo_aux_weight=0.03`)
dengan preprocessing serta split yang sama. Data external tidak diproses atau
dievaluasi.

Hasil baseline Geo0 seed 42-46 dari `gear_tuning04` dipakai ulang sehingga
baseline tersebut tidak dilatih ulang.

```powershell
python run_gear_hparam_tuning.py search
```

Hasil utama:

- `stage1_search_summary.csv`
- `stage1_top3_candidates.json`

## Tahap 2: Multiseed Confirmation

Menjalankan tiga kandidat terbaik pada seed 42-46 dan memilih pemenang
berdasarkan rata-rata vessel-level validation Macro-F1.

```powershell
python run_gear_hparam_tuning.py confirm
```

Hasil utama:

- `stage2_multiseed_summary.csv`
- `stage2_winner.json`

## Tahap 3: Final External Evaluation

Hanya pemenang tahap internal yang dievaluasi pada `Dataset_Test_Enriched`.

```powershell
python run_gear_hparam_tuning.py external
```

Hasil utama:

- `stage3_external_final/final_external_summary.json`
- confusion matrix setiap seed

## Status

```powershell
python run_gear_hparam_tuning.py status
```

Semua output disimpan di:

`Outputs/gear_tuning06_internal_hparam_gap12h_opfilter`

## Jalankan Semua Tahap

Tahap search, confirm, dan external dijalankan berurutan. Runner melewati
output yang sudah lengkap jika command dijalankan ulang.

```powershell
python run_gear_hparam_tuning.py all
```
