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
- Natural validation prevalence is retained. Training handles imbalance with
  class weights, focal loss, and a weighted sampler; validation is not
  artificially balanced, so precision and PR-AUC remain interpretable.
- Preprocess, training, and evaluation assert the fixed binary mapping
  `0=normal`, `1=spoofing`; positive probabilities and true positives always
  refer to the anomaly class.
- Evaluation writes only one report-ready spoofing confusion matrix,
  `confusion_matrix.png`. It is anomaly-first, row-normalized, and also shows
  sample counts in every TP/FN/FP/TN cell. Per-attack results are kept in
  `spoofing_attack_metrics.csv` instead of repeated confusion-matrix images.
- Train/validation splitting is group-disjoint by `original_mmsi` and uses a
  mixed-label group strategy, because each source vessel contains both normal
  and spoofing windows. Both window classes are required in each split.
- Normal trajectory retention is 50%, so a 300-point eligible source still
  has enough contiguous normal points for a 120-point window.
- Location-jump state and detectable jump-boundary labels are stored
  separately; motion-only training labels only windows that contain the jump.
- Each synthetic scenario records nominal/applied coordinate offsets and the
  resulting displacement in kilometres, duration, and gradual-drift rate.
- Attack segments cannot cross a timestamp gap above three hours, preventing
  natural AIS outages from becoming a synthetic spoofing shortcut.
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

### Attack coverage and status

| Attack | Current status | Detector input required |
|---|---|---|
| `gradual_drift` | Main model, trained and evaluated | One kinematic window |
| `location_jump` | Main model, trained and evaluated | One kinematic window |
| `replay` | Context-detector phase | Historical trajectory matching |
| `meaconing` | Context-detector phase | Historical timing and delayed-signal comparison |
| `ghost` | Context-detector phase | Cross-vessel and vessel-identity comparison |
| `mirroring` | Context-detector phase | Simultaneous cross-vessel trajectory comparison |

The final spoofing system should therefore be a two-branch system: the current
kinematic-window model handles gradual drift and location jump, while a
separate context detector handles replay, meaconing, ghost, and mirroring.
Both branches may produce the same final binary decision (`normal` or
`spoofing`), but their performance must be evaluated separately before a
combined system-level result is reported. The four context attacks remain
available in the simulator; they are not discarded or silently counted as
training coverage.

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

The first frozen run exposed a duration-induced domain shift: a fixed 0.01
degree total drift produced a median rate of 0.033 km/h internally but only
0.014 km/h externally because external scenarios lasted much longer. The
corrected baseline therefore controls gradual drift by physical rate:
`0.033 km/h` with `25%` per-scenario jitter. Location jump remains fixed at
`0.50` degrees. Corrected runs are written to
`Outputs/spoofing_baseline02_rate_matched_multiseed`; the original baseline01
is retained as audit evidence and must not be overwritten.

The mixed-label group splitter also uses non-overlapping candidate streams per
split seed. Baseline01 seeds 42-46 all selected the same six validation source
groups despite different requested seeds; baseline02 must verify five distinct
validation group sets before its results are called multiseed robustness.

Baseline02 then exposed a second simulator artifact: attack trajectories always
overwrote reported AIS `speed` and `course` with position-derived values, while
normal trajectories retained reported AIS values. Internally, median reported
speed was approximately 1.70 knots for normal rows but only 0.04-0.05 knots for
attack rows, creating an artificial classification shortcut. Baseline03 writes
to `Outputs/spoofing_baseline03_motion_semantics_multiseed` and uses two
scenarios per attack/source vessel with a 50/50 mixture of position-only
(`speed/course` preserved) and full-message-consistent (`speed/course`
recomputed) scenarios. The selected mode is recorded in every magnitude audit.
Drift-rate jitter is broadened to 50% for internal domain diversity.

A real-data preparation audit then showed that preserve-only attack segments
could still have a different baseline speed distribution from the broad normal
pool because the selected source segments differed. Baseline04 therefore uses
paired normal controls: every manipulated kinematic segment is accompanied by
the exact unmodified segment from the same source vessel and timestamps.
Reported AIS `speed/course` are preserved for the primary position-only threat
model, while position-derived features still respond to the altered
coordinates. Baseline04 writes to
`Outputs/spoofing_baseline04_paired_position_only_multiseed`. Baseline03 is an
untrained preparation audit and must not be used as a final model result.

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
6. creates one primary confusion matrix, per-attack metrics, and scenario predictions
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
   Because AIS reporting cadence varies, always report `attack_duration_hours`.
   For gradual drift, compare `attack_drift_rate_kmh`, not only total degrees.
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

## Fixed, Not Tuned

- split grouping by `original_mmsi`;
- maximum continuity gap of three hours;
- exclusion of absolute shore/port-distance features;
- geographic auxiliary weight of zero;
- external-test membership;
- random seed (used for robustness measurement, not model selection).

## Required Ablations

- easy versus mild attack magnitude;
- `gradual_drift` versus `location_jump`;
- sequence length 60 versus 120;
- focal loss versus weighted cross-entropy;
- context-required attacks as a separately labeled limitation experiment.

## Current Reporting Baseline

The reporting baseline is
`Outputs/spoofing_baseline04_paired_position_only_multiseed` with sequence
length 120. A time-constrained sequence-length 240 diagnostic was rejected:
on seed 42 it retained only 24 positive validation windows and produced
spoofing precision 0.024, recall 0.125, F1 0.041, and PR-AUC 0.009. Its high
raw accuracy was caused by the normal-class majority and is not evidence of a
better detector.

Run the final reporting analysis with:

```powershell
python analyze_spoofing_final.py
```

The analysis selects thresholds from validation predictions only, using the
median of the five per-seed positive-class F1 optima. The frozen reporting
thresholds are 0.75 for sequence predictions and 0.45 for scenario
predictions. It writes five-seed mean/standard-deviation tables and per-attack
tables to `Outputs/spoofing_baseline04_paired_position_only_multiseed/final_analysis`.
Vessel-level metrics are excluded because the validation source-vessel labels
do not define meaningful positive spoofing vessels. The existing external set
is reported only as a diagnostic set, not as a pristine final holdout.

## Context-Detector Baseline

The four context-required attacks are evaluated by a separate trusted-history
baseline in `analyze_spoofing_context.py`; they are not added to the
single-window LSTM labels.

- `replay`: a previously observed trajectory reappears at a future time;
- `meaconing`: historical positions are reported with a measurable lag;
- `ghost`: the claimed MMSI is absent from the trusted identity registry;
- `mirroring`: a simultaneous trajectory is copied with a near-constant
  coordinate translation.

The simulator keeps its collision-free synthetic `mmsi` trajectory key and
now writes `claimed_mmsi`, which represents the transmitted identity available
to the context detector. Generate and analyze the internal context baseline:

```powershell
python main.py make_spoofing --input_path Dataset --out_dir Outputs/spoofing_context01_trusted_history/generated_internal --attacks replay meaconing ghost mirroring --seed 42 --limit_rows 300000 --normal_keep_frac 0.5 --max_vessels_per_file 10 --min_points_per_vessel 160 --points_per_attack 120 --scenarios_per_attack 1 --max_attack_gap_seconds 10800 --reported_motion_mode preserve
python analyze_spoofing_context.py --generated_dir Outputs/spoofing_context01_trusted_history/generated_internal --reference_dir Dataset --out_dir Outputs/spoofing_context01_trusted_history/internal_analysis --limit_rows 300000 --seq_len 120
```

This branch does not require neural-network training. The current synthetic
sanity check contains 132 scenarios (28 normal and 104 attacks) and detects all
four attack types without a false positive. This perfect score must not be
presented as real-world generalization: the transformations are exact and the
detector is given a trusted registry plus trusted trajectory history. Its
purpose is to verify that the four attacks are detectable when their required
context actually exists. Robustness to coordinate noise, incomplete history,
identity-registry errors, and approximate trajectory copies remains future
work. External generated results are diagnostics only.

The context implementation now exposes every decision threshold explicitly.
These thresholds are engineering guard bands for the simulated threat model;
they are not attributed to Agrebi et al. The detector also emits
`context_score`, supports approximate translation-invariant history matching,
and can be audited with coordinate noise and incomplete trusted history.

Internal robustness diagnostics (132 scenarios) currently give:

| Condition | Binary F1 | Interpretation |
|---|---:|---|
| Exact simulator output, complete history | 1.000 | Sanity check only |
| Coordinate noise SD 0.00001 degrees | 0.949 | Small-noise diagnostic |
| Coordinate noise SD 0.00005 degrees | 0.944 | Stronger-noise diagnostic |
| 75% contiguous trusted history retained | 0.924 | Incomplete-history diagnostic |

`analyze_spoofing_hybrid.py` implements logical-OR fusion of BiLSTM and
context alerts, but deliberately refuses to run unless both branches contain
exactly the same `scenario_id` set and identical true labels. The existing
Baseline04 and context outputs are not aligned, so they must not be combined
into a hybrid metric. A unified six-attack evaluation dataset must first be
scored independently by both frozen branches.

### Unified hybrid diagnostic

The aligned diagnostic is available under
`Outputs/spoofing_hybrid01_unified_six_attack`. It uses simulation seed 2042,
all six attacks, position-only reported-motion semantics, the frozen
Baseline04 seed-45 checkpoint, the frozen validation-selected BiLSTM scenario
threshold (0.45), and the explicit context rules. No model was retrained.

Both branches scored exactly the same 242 scenarios: 74 normal scenarios and
28 scenarios for each attack. Logical-OR fusion produced precision 1.000,
recall 0.845, F1 0.916, balanced accuracy 0.923, zero false positives, and 26
false negatives. Per-attack recall was 0.321 for gradual drift, 0.750 for
location jump, and 1.000 for replay, meaconing, ghost, and mirroring.

With coordinate-noise SD 0.00001 degrees, the hybrid result remained F1 0.916
on this diagnostic. With only 75% contiguous trusted history retained, recall
fell to 0.750 and F1 to 0.857; meaconing recall was 0.714, while replay and
mirroring recall were both 0.857. These remain synthetic diagnostic results,
not real-world performance claims. AP and ROC-AUC are intentionally omitted
for hybrid logical-OR output because its score is a binary decision, not a
calibrated ranking probability.

## Paper-Style GRU Comparator

`run_spoofing_paper_gru.py` implements a separate comparison experiment based
on Agrebi et al. (IEEE Access, 2025). It uses a one-layer GRU with 64 units,
batch normalization, dropout 0.30, a 32-unit dense layer, dropout 0.20, Adam
with learning rate 0.001, batch size 32, at most 50 epochs, and validation-loss
early stopping with patience 10. Input sequences contain 10 AIS readings.

The experiment includes all five attacks in the paper plus `mirroring`:
`gradual_drift`, `location_jump`, `replay`, `meaconing`, `ghost`, and
`mirroring`. It is intentionally isolated from Baseline04 and the trusted-
history detector. Unlike the paper's insufficiently specified split, this
implementation enforces disjoint `original_mmsi` groups across training,
validation, and internal test. It also retains this repository's
leakage-safer kinematic feature policy instead of feeding absolute latitude and
longitude to the classifier. Consequently, it is a paper-style comparator,
not an exact numerical reproduction.

Preparation does not train a model:

```powershell
python run_spoofing_paper_gru.py prepare
```

Training must be started explicitly:

```powershell
python run_spoofing_paper_gru.py train
```

Check artifact status without changing data or training:

```powershell
python run_spoofing_paper_gru.py status
```

Outputs are reserved under
`Outputs/spoofing_paper_gru01_six_attack_seed42`. Results from this comparator
must be reported separately from the two-attack Baseline04 and separately from
the four-attack trusted-history rules.
