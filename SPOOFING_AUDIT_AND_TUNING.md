# Spoofing Pipeline Audit and Tuning

## Validity Fixes

- Synthetic scenarios use unique trajectory IDs, so normal and spoofed rows no
  longer overwrite one another at identical MMSI/timestamps.
- Train/validation/external grouping uses `original_mmsi`, preventing source
  trajectory leakage through ghost or copied scenarios.
- Normal downsampling keeps contiguous segments per vessel.
- Generator seeds are stable across Python processes.
- Location-distance features and geo auxiliary loss are disabled for spoofing.
- Positive windows are preserved when the preprocessing cap is applied.
- Evaluation writes PR-AUC, ROC-AUC, sequence/scenario predictions, and
  metrics per attack type.

## Main Experiment

The primary experiment uses only attacks identifiable from a single kinematic
window:

- `gradual_drift`
- `location_jump`

`replay`, `meaconing`, `ghost`, and `mirroring` require historical,
cross-vessel, identity, or external geospatial context. They may be generated
for a limitation study but must not be mixed into the main performance claim.

## Run

```powershell
python run_spoofing_multiseed.py run
```

Status:

```powershell
python run_spoofing_multiseed.py status
```

Output:

`Outputs/spoofing_baseline01_identifiable_multiseed`

The runner:

1. generates internal synthetic spoofing from `Dataset`;
2. generates external synthetic spoofing from `Dataset_Test_Enriched` using a
   different simulation seed;
3. preprocesses both without location-distance features;
4. trains seeds 42-46 using internal train/validation only;
5. evaluates validation and external data;
6. creates confusion matrices, per-attack metrics, and scenario predictions
   per seed.

## Metrics To Report

Primary:

- spoofing-class precision;
- spoofing-class recall;
- spoofing-class F1;
- PR-AUC / average precision;
- balanced accuracy;
- per-attack recall and F1;
- mean and standard deviation across seeds.

Accuracy is secondary because normal and spoofing windows may be imbalanced.

Scenario aggregation uses the mean of the highest 10% sequence probabilities
and a temporary threshold of 0.5. Treat it as baseline diagnostics. Select the
final scenario threshold using validation data only, then freeze it before the
external evaluation.

## Tuning Order

Do not sweep everything at once. Use internal validation only.

1. Data/window parameters:
   - `seq_len`: 60, 120;
   - `points_per_attack`: 180, 240;
   - `spoofing_window_threshold`: 0.10, 0.20, 0.40;
   - drift magnitude: 0.02, 0.05, 0.08 degrees;
   - jump magnitude: 0.10, 0.30, 0.70 degrees.
2. Model capacity:
   - hidden size: 128, 256;
   - LSTM layers: 1, 2;
   - attention layers: 0, 1.
3. Optimization:
   - learning rate: 0.0001, 0.00025, 0.0005;
   - dropout: 0.20, 0.30, 0.40;
   - focal gamma: 0.0, 1.2, 2.0.
4. Confirm the best two or three configurations on seeds 42-46.
5. Freeze the winner, then evaluate `Dataset_Test_Enriched` once.

## Required Ablations

- easy versus mild attack magnitude;
- `gradual_drift` versus `location_jump`;
- sequence length 60 versus 120;
- focal loss versus weighted cross-entropy;
- context-required attacks as a separately labeled limitation experiment.
