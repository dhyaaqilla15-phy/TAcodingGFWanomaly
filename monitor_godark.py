#!/usr/bin/env python3
"""Audit output candidate-event GoDark tanpa menjalankan training."""
import argparse
from pathlib import Path
import pandas as pd

def analyze_godark(csv_path):
    print(f"\n{'='*70}")
    print(f"Analyzing: {csv_path}")
    print(f"{'='*70}\n")
    
    df = pd.read_csv(csv_path)
    
    # Boundary labels are metadata only; one event sample is created later.
    normal_count = (df['is_go_dark'] == 0).sum()
    godark_count = (df['is_go_dark'] == 1).sum()
    total = len(df)
    
    print(f"Total rows: {total:,}")
    print(f"Normal samples: {normal_count:,} ({100*normal_count/total:.2f}%)")
    print(f"Go_dark samples: {godark_count:,} ({100*godark_count/total:.2f}%)")
    print(f"Boundary row ratio: {normal_count/max(1, godark_count):.1f}:1 (bukan class balance model)")

    event_mask = df.get("go_dark_event_id", pd.Series("normal", index=df.index)).astype(str).ne("normal")
    print(f"Synthetic candidate events: {df.loc[event_mask, 'go_dark_event_id'].nunique():,}")
    print(f"Source vessels: {df['mmsi'].nunique():,}")
    
    # By label
    print(f"\nBy 'label' column:")
    label_dist = df['label'].value_counts()
    for label, count in label_dist.items():
        print(f"  {label}: {count:,} ({100*count/total:.2f}%)")
    
    # By event_phase
    print(f"\nBy 'event_phase' column:")
    phase_dist = df['event_phase'].value_counts()
    for phase, count in phase_dist.items():
        print(f"  {phase}: {count:,} ({100*count/total:.2f}%)")
    
    # Behavioral stats
    if 'behavior_anomaly_score' in df.columns:
        print(f"\nBehavioral anomaly scores (go_dark samples):")
        godark_df = df[df['is_go_dark'] == 1]
        if len(godark_df) > 0:
            scores = godark_df['behavior_anomaly_score']
            print(f"  Mean: {scores.mean():.2f}")
            print(f"  Median: {scores.median():.2f}")
            print(f"  Min: {scores.min():.2f}, Max: {scores.max():.2f}")
    
    print("\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "path",
        nargs="?",
        default=str(Path(__file__).resolve().parent / "Outputs"),
        help="godark_all.csv, folder godark, atau root output berisi run*/godark",
    )
    args = parser.parse_args()
    target = Path(args.path)
    if target.is_file():
        analyze_godark(target)
    else:
        candidates = sorted(target.glob("godark_all.csv"))
        candidates += sorted(target.glob("*/godark/godark_all.csv"))
        if not candidates:
            raise FileNotFoundError(f"Tidak menemukan godark_all.csv di {target}")
        for godark_csv in candidates:
            analyze_godark(godark_csv)
