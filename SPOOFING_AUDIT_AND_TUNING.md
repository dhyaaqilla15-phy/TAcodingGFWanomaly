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
- Normal trajectory retention is 50%, so a 300-point eligible source still
  has enough contiguous normal points for a 120-point window.
- Location-jump state and detectable jump-boundary labels are stored
  separately; motion-only training labels only windows that contain the jump.
- Each synthetic scenario records nominal/applied coordinate offsets and the
  resulting displacement in kilometres.
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

### Stage 1: Jump-magnitude sensitivity

Run this before the multiseed baseline:

```powershell
python run_spoofing_jump_sensitivity.py run
```

It compares nominal jump magnitudes `0.10`, `0.30`, `0.50`, and `0.80`
degrees while keeping the generated source segments, drift magnitude, split,
model, and seed fixed. It uses internal validation only and writes:

`Outputs/spoofing_tuning01_jump_magnitude_seed42/jump_sensitivity_summary.csv`

The degree value is a simulation severity parameter, not a model decision
threshold. It must be reported as a sensitivity curve rather than selected
merely because the largest value gives the highest F1. Since latitude and
longitude are both shifted and the simulator varies each component by 65-135%
of the nominal value, a nominal `0.80` degree jump can exceed 100 km. The
generated magnitude audit records the actual displacement in kilometres.

Check progress with:

```powershell
python run_spoofing_jump_sensitivity.py status
```

### Final multiseed baseline

Do not start this until the Stage 1 sensitivity result has been reviewed and
the final training severity policy has been frozen.

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

1. Attack-severity sensitivity, seed 42:
   - jump magnitude: 0.10, 0.30, 0.50, 0.80 degrees;
   - report actual displacement in kilometres and per-attack metrics;
   - do not use the external set and do not call the largest/easiest attack
     the best hyperparameter.
2. Data/window parameters after Stage 1:
   - `seq_len`: 60, 120;
   - `points_per_attack`: 180, 240;
   - gradual-drift window threshold: 0.10, 0.20, 0.40;
   - drift magnitude: 0.01, 0.03, 0.05, 0.08 degrees.
   `location_jump` is event-labelled at the jump boundary, so the fraction
   threshold is not applied to it.
3. Model capacity:
   - hidden size: 128, 256;
   - LSTM layers: 1, 2;
   - attention layers: 0, 1.
4. Optimization:
   - learning rate: 0.0001, 0.00025, 0.0005;
   - dropout: 0.20, 0.30, 0.40;
   - focal gamma: 0.0, 1.2, 2.0.
5. Confirm the best two or three model configurations on seeds 42-46.
6. Freeze generation, preprocessing, model, and decision threshold, then
   evaluate `Dataset_Test_Enriched` once.

## Required Ablations

- easy versus mild attack magnitude;
- `gradual_drift` versus `location_jump`;
- sequence length 60 versus 120;
- focal loss versus weighted cross-entropy;
- context-required attacks as a separately labeled limitation experiment.
