from __future__ import annotations

import argparse
from pathlib import Path

from data_preparation import build_sequences_to_npz
from train import train_from_npz
from eval import evaluate
from plot_trajectory import (
    plot_trajectory_from_csv,
    plot_all_from_csv,
    heatmap_from_csv,
)


def main():
    parser = argparse.ArgumentParser("AIS LSTM Gear-Type Classification + Suspected Ship")
    sub = parser.add_subparsers(dest="cmd", required=True)

    # ===== preprocess =====
    p = sub.add_parser("preprocess")
    p.add_argument("--data_dir", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--task", default="gear", choices=["gear", "fishing"])
    p.add_argument("--exclude_labels", nargs="*", default=[])
    p.add_argument("--limit_rows", type=int, default=0)
    p.add_argument("--chunksize", type=int, default=0)

    # ===== train =====
    t = sub.add_parser("train")
    t.add_argument("--data_npz", required=True)
    t.add_argument("--out_dir", required=True)
    t.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    # NEW: seed untuk split + sampler
    t.add_argument("--random_state", type=int, default=42)
    # optional: override test_size kalau mau
    t.add_argument("--test_size", type=float, default=0.2)

    # ===== eval =====
    e = sub.add_parser("eval")
    e.add_argument("--data_npz", required=True)
    e.add_argument("--model_path", required=True)
    e.add_argument("--out_dir", required=True)
    e.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    # NEW: seed kalau eval perlu bikin split baru (kalau split_indices.npz belum ada)
    e.add_argument("--random_state", type=int, default=42)
    e.add_argument("--test_size", type=float, default=0.2)
    e.add_argument("--batch_size", type=int, default=64)

    # ===== plot =====
    pl = sub.add_parser("plot")
    pl.add_argument("--csv_path", required=True)
    pl.add_argument("--out_dir", default="outputs/plots")
    pl.add_argument("--sample_vessel", default="")
    pl.add_argument("--max_points", type=int, default=6000)
    pl.add_argument("--color_by", default="speed", choices=["time", "speed", "none"])
    pl.add_argument("--cmap", default="viridis")

    # ===== plot_all =====
    pa = sub.add_parser("plot_all")
    pa.add_argument("--csv_path", required=True)
    pa.add_argument("--out_dir", default="outputs")
    pa.add_argument("--chunksize", type=int, default=300_000)
    pa.add_argument("--max_points_per_vessel", type=int, default=6000)
    pa.add_argument("--color_by", default="speed", choices=["time", "speed", "none"])
    pa.add_argument("--cmap", default="viridis")

    # ===== heatmap =====
    hm = sub.add_parser("heatmap")
    hm.add_argument("--csv_path", required=True)
    hm.add_argument("--out_dir", default="outputs/heatmaps")
    hm.add_argument("--bins", type=int, default=350)
    hm.add_argument("--chunksize", type=int, default=500_000)
    hm.add_argument("--log_scale", action="store_true")

    args = parser.parse_args()

    if args.cmd == "preprocess":
        build_sequences_to_npz(
            data_dir=Path(args.data_dir),
            out_dir=Path(args.out_dir),
            task=args.task,
            exclude_labels=args.exclude_labels,
            limit_rows=args.limit_rows,
            chunksize=args.chunksize,
        )

    elif args.cmd == "train":
        train_from_npz(
            data_npz=Path(args.data_npz),
            out_dir=Path(args.out_dir),
            device=args.device,
            random_state=int(args.random_state),
            test_size=float(args.test_size),
        )

    elif args.cmd == "eval":
        mp = Path(args.model_path)
        if not mp.exists():
            raise FileNotFoundError(f"Model not found: {mp}. Jalankan train sampai selesai dulu.")
        evaluate(
            data_npz=Path(args.data_npz),
            model_path=mp,
            out_dir=Path(args.out_dir),
            device=args.device,
            batch_size=int(args.batch_size),
            test_size=float(args.test_size),
            random_state=int(args.random_state),
        )

    elif args.cmd == "plot":
        plot_trajectory_from_csv(
            csv_path=Path(args.csv_path),
            out_dir=Path(args.out_dir),
            sample_vessel=args.sample_vessel,
            max_points=args.max_points,
            color_by=args.color_by,
            cmap=args.cmap,  
        )

    elif args.cmd == "plot_all":
        plot_all_from_csv(
            csv_path=Path(args.csv_path),
            out_dir=Path(args.out_dir),
            chunksize=args.chunksize,
            max_points_per_vessel=args.max_points_per_vessel,
            color_by=args.color_by,
            cmap=args.cmap,
        )

    elif args.cmd == "heatmap":
        heatmap_from_csv(
            csv_path=Path(args.csv_path),
            out_dir=Path(args.out_dir),
            bins=args.bins,
            chunksize=args.chunksize,
            log_scale=args.log_scale,
        )

if __name__ == "__main__":
    main()
