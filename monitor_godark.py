#!/usr/bin/env python3
"""
Monitor go_dark generation dan check class balance improvements
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np

def analyze_godark(csv_path):
    print(f"\n{'='*70}")
    print(f"Analyzing: {csv_path}")
    print(f"{'='*70}\n")
    
    df = pd.read_csv(csv_path)
    
    # Class distribution
    normal_count = (df['is_go_dark'] == 0).sum()
    godark_count = (df['is_go_dark'] == 1).sum()
    total = len(df)
    
    print(f"Total rows: {total:,}")
    print(f"Normal samples: {normal_count:,} ({100*normal_count/total:.2f}%)")
    print(f"Go_dark samples: {godark_count:,} ({100*godark_count/total:.2f}%)")
    print(f"Imbalance ratio: {normal_count/max(1, godark_count):.1f}:1")
    
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
    base_path = Path("/home/aqila/newcodinggfw/output")
    
    for run_folder in sorted(base_path.glob("run*")):
        godark_csv = run_folder / "godark" / "godark_all.csv"
        if godark_csv.exists():
            analyze_godark(godark_csv)
