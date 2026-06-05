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
from spoofing_simulator import SpoofingSimCfg, generate_spoofing_dataset
from plot_spoofing import (
    plot_spoofing_overlay_from_csv,
    heatmap_spoofing_from_csv,
    plot_spoofing_examples_from_csv,
)
from plot_preprocessed import plot_preprocessed_trajectory_from_npz
from go_dark_simulator import GoDarkSimCfg, generate_go_dark_dataset
from plot_go_dark import plot_go_dark_overlay_from_csv, heatmap_go_dark_from_csv, plot_go_dark_examples_from_csv


def main():
    parser = argparse.ArgumentParser("AIS LSTM Gear-Type Classification + Suspected Ship")
    sub = parser.add_subparsers(dest="cmd", required=True)

    # ===== preprocess =====
    p = sub.add_parser("preprocess")
    p.add_argument("--data_dir", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--task", default="gear", choices=["gear", "fishing", "spoofing", "godark"])
    p.add_argument("--exclude_labels", nargs="*", default=[])
    p.add_argument("--limit_rows", type=int, default=0)
    p.add_argument("--chunksize", type=int, default=0)
    p.add_argument("--seq_len", type=int, default=120)
    p.add_argument("--stride", type=int, default=6)
    p.add_argument("--gap_seconds", type=int, default=10800)
    p.add_argument("--max_implied_knots", type=float, default=42.0)
    p.add_argument("--min_points_per_vessel", type=int, default=80)
    p.add_argument(
        "--min_windows_per_vessel",
        type=int,
        default=0,
        help="Buang vessel yang menghasilkan window lebih sedikit dari nilai ini. Berguna untuk gear vessel-level agar val/test tidak diisi trajectory terlalu pendek.",
    )
    p.add_argument("--max_windows_per_vessel", type=int, default=1200)
    p.add_argument("--max_windows_per_file", type=int, default=20000)
    p.add_argument(
        "--balance_gear_classes",
        action="store_true",
        help="Opsional: downsample window gear agar setiap kelas punya jumlah sama. Default off karena training sudah memakai class/vessel balancing.",
    )
    p.add_argument("--spoofing_window_threshold", type=float, default=0.20)
    p.add_argument(
        "--apply_jump_filter",
        action="store_true",
        help="Default task=spoofing dan task=godark tidak membuang jump. Flag ini memaksa jump filter aktif.",
    )

    # ===== train =====
    t = sub.add_parser("train")
    t.add_argument("--data_npz", required=True)
    t.add_argument("--out_dir", required=True)
    t.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    t.add_argument("--random_state", type=int, default=42)
    t.add_argument("--test_size", type=float, default=0.2)
    t.add_argument("--val_size", type=float, default=0.15)
    t.add_argument("--epochs", type=int, default=320)
    t.add_argument("--batch_size", type=int, default=128)
    t.add_argument("--lr", type=float, default=2.5e-4)
    t.add_argument("--hidden_size", type=int, default=384)
    t.add_argument("--num_layers", type=int, default=2)
    t.add_argument("--input_proj_dim", type=int, default=256, help="0 = legacy auto bottleneck.")
    t.add_argument("--embed_dim", type=int, default=512, help="0 = legacy auto embedding.")
    t.add_argument("--dropout", type=float, default=0.40)
    t.add_argument("--attention_heads", type=int, default=4, help="0 mematikan self-attention block.")
    t.add_argument("--attention_layers", type=int, default=1, help="Jumlah residual self-attention block setelah LSTM.")
    t.add_argument(
        "--geo_aux_weight",
        type=float,
        default=0.03,
        help="Bobot auxiliary Haversine loss dari prediksi koordinat akhir window. Isi 0 untuk mematikan.",
    )
    t.add_argument(
        "--geo_aux_scale_km",
        type=float,
        default=1000.0,
        help="Skala normalisasi Haversine loss agar tidak mendominasi classification loss.",
    )
    t.add_argument("--optimizer", type=str, default="adamw", choices=["adamw", "adam", "sgd"])
    t.add_argument("--weight_decay", type=float, default=1.3e-3)
    t.add_argument("--sgd_momentum", type=float, default=0.9)
    t.add_argument("--early_stop_patience", type=int, default=90)

    # ===== eval =====
    e = sub.add_parser("eval")
    e.add_argument("--data_npz", required=True)
    e.add_argument("--model_path", required=True)
    e.add_argument("--out_dir", required=True)
    e.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
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

    # ===== make_spoofing =====
    sp = sub.add_parser("make_spoofing")
    sp.add_argument("--input_path", required=True, help="CSV file atau folder Dataset berisi CSV")
    sp.add_argument("--out_dir", required=True)
    sp.add_argument(
        "--attacks",
        nargs="*",
        default=["gradual_drift", "location_jump", "replay", "meaconing", "ghost", "mirroring"],
        choices=["gradual_drift", "location_jump", "replay", "meaconing", "ghost", "mirroring"],
    )
    sp.add_argument("--seed", type=int, default=42)
    sp.add_argument("--limit_rows", type=int, default=0)
    sp.add_argument("--chunksize", type=int, default=0)
    sp.add_argument("--sample_frac", type=float, default=0.0)
    sp.add_argument("--normal_keep_frac", type=float, default=1.0)
    sp.add_argument("--max_vessels_per_file", type=int, default=20)
    sp.add_argument("--min_points_per_vessel", type=int, default=80)
    sp.add_argument("--points_per_attack", type=int, default=120)
    sp.add_argument("--drift_lat_deg", type=float, default=0.08)
    sp.add_argument("--drift_lon_deg", type=float, default=0.08)
    sp.add_argument("--jump_lat_deg", type=float, default=0.70)
    sp.add_argument("--jump_lon_deg", type=float, default=0.70)
    sp.add_argument("--replay_delay_seconds", type=int, default=21600)
    sp.add_argument("--meacon_lag_steps", type=int, default=8)
    sp.add_argument("--ghost_offset_min_deg", type=float, default=1.5)
    sp.add_argument("--ghost_offset_max_deg", type=float, default=8.0)
    sp.add_argument("--mirror_offset_min_deg", type=float, default=1.5)
    sp.add_argument("--mirror_offset_max_deg", type=float, default=8.0)
    sp.add_argument("--combine_outputs", action="store_true")

    # ===== plot_spoofing =====
    ps = sub.add_parser("plot_spoofing")
    ps.add_argument("--csv_path", required=True)
    ps.add_argument("--out_dir", default="outputs_spoofing/plots")
    ps.add_argument("--sample_vessel", default="")
    ps.add_argument("--attack_type", default="all")
    ps.add_argument("--max_points", type=int, default=8000)

    # ===== plot_spoofing_examples =====
    pse = sub.add_parser("plot_spoofing_examples")
    pse.add_argument("--csv_path", required=True)
    pse.add_argument("--out_dir", default="outputs_spoofing/plots/examples")
    pse.add_argument(
        "--attacks",
        nargs="*",
        default=[],
        help="Daftar attack yang ingin dibuatkan contoh plot. Kosong = semua attack spoofing.",
    )
    pse.add_argument("--max_points", type=int, default=8000)

    # ===== heatmap_spoofing =====
    hs = sub.add_parser("heatmap_spoofing")
    hs.add_argument("--csv_path", required=True)
    hs.add_argument("--out_dir", default="outputs_spoofing/heatmaps")
    hs.add_argument("--bins", type=int, default=300)
    hs.add_argument("--log_scale", action="store_true")

    # ===== make_godark =====
    gd = sub.add_parser("make_godark")
    gd.add_argument("--input_path", required=True, help="CSV file atau folder Dataset berisi CSV")
    gd.add_argument("--out_dir", required=True)
    gd.add_argument("--seed", type=int, default=42)
    gd.add_argument("--limit_rows", type=int, default=0)
    gd.add_argument("--chunksize", type=int, default=0)
    gd.add_argument("--sample_frac", type=float, default=0.0)
    gd.add_argument("--max_vessels_per_file", type=int, default=20)
    gd.add_argument("--min_points_per_vessel", type=int, default=120)
    gd.add_argument("--events_per_vessel", type=int, default=1)
    gd.add_argument("--min_hidden_points", type=int, default=20)
    gd.add_argument("--max_hidden_points", type=int, default=120)

    gd.add_argument(
        "--min_dark_seconds",
        type=int,
        default=43200,
        help="Minimum durasi AIS blackout. Default 12 jam.",
    )
    gd.add_argument("--max_dark_seconds", type=int, default=604800)
    gd.add_argument("--min_hidden_distance_km", type=float, default=0.5)

    gd.add_argument(
        "--min_distance_from_shore_nm",
        type=float,
        default=50.0,
        help="Filter jarak minimal dari pantai dalam nautical miles. Isi 0 untuk mematikan filter.",
    )
    gd.add_argument(
        "--ping_window_seconds",
        type=int,
        default=43200,
        help="Jendela hitung ping sebelum blackout. Default 12 jam.",
    )
    gd.add_argument(
        "--min_ping_count_prev_window",
        type=int,
        default=0,
        help="Opsional. Isi 14 kalau ingin lebih strict.",
    )

    gd.add_argument("--label_before_points", type=int, default=2)
    gd.add_argument("--label_after_points", type=int, default=30)
    gd.add_argument("--combine_outputs", action="store_true")

    # ===== plot_godark =====
    pgd = sub.add_parser("plot_godark")
    pgd.add_argument("--csv_path", required=True)
    pgd.add_argument("--out_dir", default="outputs_godark/plots")
    pgd.add_argument("--sample_vessel", default="")
    pgd.add_argument("--event_id", default="all")
    pgd.add_argument("--max_points", type=int, default=9000)

    # ===== heatmap_godark =====
    hgd = sub.add_parser("heatmap_godark")
    hgd.add_argument("--csv_path", required=True)
    hgd.add_argument("--out_dir", default="outputs_godark/heatmaps")
    hgd.add_argument("--bins", type=int, default=300)
    hgd.add_argument("--log_scale", action="store_true")

    # ===== plot_go_dark_examples =====
    pgde = sub.add_parser("plot_go_dark_examples")
    pgde.add_argument("--csv_path", required=True)
    pgde.add_argument("--out_dir", default="outputs_godark/plots/examples")
    pgde.add_argument("--num_examples", type=int, default=6)
    pgde.add_argument("--max_points", type=int, default=9000)

    # ===== plot_preprocessed =====
    ppre = sub.add_parser("plot_preprocessed")
    ppre.add_argument("--npz_path", required=True)
    ppre.add_argument("--out_dir", default="outputs_preprocessed/plots")
    ppre.add_argument("--task", default="auto", choices=["auto", "spoofing", "godark"])
    ppre.add_argument("--sample_vessel", default="")
    ppre.add_argument("--max_windows", type=int, default=12)
    ppre.add_argument(
        "--include_normal_windows",
        action="store_true",
        help="Default hanya plot window anomaly. Flag ini ikut menampilkan window normal.",
    )

    args = parser.parse_args()

    if args.cmd == "preprocess":
        build_sequences_to_npz(
            data_dir=Path(args.data_dir),
            out_dir=Path(args.out_dir),
            task=args.task,
            exclude_labels=args.exclude_labels,
            limit_rows=args.limit_rows,
            chunksize=args.chunksize,
            seq_len=int(args.seq_len),
            stride=int(args.stride),
            gap_seconds=int(args.gap_seconds),
            max_implied_knots=float(args.max_implied_knots),
            min_points_per_vessel=int(args.min_points_per_vessel),
            min_windows_per_vessel=int(args.min_windows_per_vessel),
            max_windows_per_vessel=int(args.max_windows_per_vessel),
            max_windows_per_file=int(args.max_windows_per_file),
            balance_gear_classes=bool(args.balance_gear_classes),
            spoofing_window_threshold=float(args.spoofing_window_threshold),
            apply_jump_filter=(True if bool(args.apply_jump_filter) else None),
        )

    elif args.cmd == "train":
        train_from_npz(
            data_npz=Path(args.data_npz),
            out_dir=Path(args.out_dir),
            epochs=int(args.epochs),
            batch_size=int(args.batch_size),
            lr=float(args.lr),
            hidden_size=int(args.hidden_size),
            num_layers=int(args.num_layers),
            input_proj_dim=(None if int(args.input_proj_dim) <= 0 else int(args.input_proj_dim)),
            embed_dim=(None if int(args.embed_dim) <= 0 else int(args.embed_dim)),
            dropout=float(args.dropout),
            attention_heads=int(args.attention_heads),
            attention_layers=int(args.attention_layers),
            geo_aux_weight=float(args.geo_aux_weight),
            geo_aux_scale_km=float(args.geo_aux_scale_km),
            optimizer_name=str(args.optimizer),
            weight_decay=float(args.weight_decay),
            sgd_momentum=float(args.sgd_momentum),
            device=args.device,
            random_state=int(args.random_state),
            test_size=float(args.test_size),
            val_size=float(args.val_size),
            early_stop_patience=int(args.early_stop_patience),
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

    elif args.cmd == "make_spoofing":
        cfg = SpoofingSimCfg(
            attacks=list(args.attacks),
            seed=int(args.seed),
            limit_rows=int(args.limit_rows),
            chunksize=int(args.chunksize),
            sample_frac=float(args.sample_frac),
            normal_keep_frac=float(args.normal_keep_frac),
            max_vessels_per_file=int(args.max_vessels_per_file),
            min_points_per_vessel=int(args.min_points_per_vessel),
            points_per_attack=int(args.points_per_attack),
            drift_lat_deg=float(args.drift_lat_deg),
            drift_lon_deg=float(args.drift_lon_deg),
            jump_lat_deg=float(args.jump_lat_deg),
            jump_lon_deg=float(args.jump_lon_deg),
            replay_delay_seconds=int(args.replay_delay_seconds),
            meacon_lag_steps=int(args.meacon_lag_steps),
            ghost_offset_min_deg=float(args.ghost_offset_min_deg),
            ghost_offset_max_deg=float(args.ghost_offset_max_deg),
            mirror_offset_min_deg=float(args.mirror_offset_min_deg),
            mirror_offset_max_deg=float(args.mirror_offset_max_deg),
            combine_outputs=bool(args.combine_outputs),
        )
        generate_spoofing_dataset(
            input_path=Path(args.input_path),
            out_dir=Path(args.out_dir),
            cfg=cfg,
        )

    elif args.cmd == "plot_spoofing":
        plot_spoofing_overlay_from_csv(
            csv_path=Path(args.csv_path),
            out_dir=Path(args.out_dir),
            sample_vessel=args.sample_vessel,
            attack_type=args.attack_type,
            max_points=int(args.max_points),
        )

    elif args.cmd == "plot_spoofing_examples":
        plot_spoofing_examples_from_csv(
            csv_path=Path(args.csv_path),
            out_dir=Path(args.out_dir),
            attacks=list(args.attacks),
            max_points=int(args.max_points),
        )

    elif args.cmd == "heatmap_spoofing":
        heatmap_spoofing_from_csv(
            csv_path=Path(args.csv_path),
            out_dir=Path(args.out_dir),
            bins=int(args.bins),
            log_scale=bool(args.log_scale),
        )

    elif args.cmd == "make_godark":
        cfg = GoDarkSimCfg(
            seed=int(args.seed),
            limit_rows=int(args.limit_rows),
            chunksize=int(args.chunksize),
            sample_frac=float(args.sample_frac),
            max_vessels_per_file=int(args.max_vessels_per_file),
            min_points_per_vessel=int(args.min_points_per_vessel),
            events_per_vessel=int(args.events_per_vessel),
            min_hidden_points=int(args.min_hidden_points),
            max_hidden_points=int(args.max_hidden_points),
            min_dark_seconds=int(args.min_dark_seconds),
            max_dark_seconds=int(args.max_dark_seconds),
            min_hidden_distance_km=float(args.min_hidden_distance_km),
            min_distance_from_shore_nm=float(args.min_distance_from_shore_nm),
            ping_window_seconds=int(args.ping_window_seconds),
            min_ping_count_prev_window=int(args.min_ping_count_prev_window),
            label_before_points=int(args.label_before_points),
            label_after_points=int(args.label_after_points),
            combine_outputs=bool(args.combine_outputs),
        )
        generate_go_dark_dataset(
            input_path=Path(args.input_path),
            out_dir=Path(args.out_dir),
            cfg=cfg,
        )

    elif args.cmd == "plot_godark":
        plot_go_dark_overlay_from_csv(
            csv_path=Path(args.csv_path),
            out_dir=Path(args.out_dir),
            sample_vessel=args.sample_vessel,
            event_id=args.event_id,
            max_points=int(args.max_points),
        )

    elif args.cmd == "heatmap_godark":
        heatmap_go_dark_from_csv(
            csv_path=Path(args.csv_path),
            out_dir=Path(args.out_dir),
            bins=int(args.bins),
            log_scale=bool(args.log_scale),
        )

    elif args.cmd == "plot_go_dark_examples":
        plot_go_dark_examples_from_csv(
            csv_path=Path(args.csv_path),
            out_dir=Path(args.out_dir),
            num_examples=int(args.num_examples),
            max_points=int(args.max_points),
        )

    elif args.cmd == "plot_preprocessed":
        plot_preprocessed_trajectory_from_npz(
            npz_path=Path(args.npz_path),
            out_dir=Path(args.out_dir),
            task=args.task,
            sample_vessel=args.sample_vessel,
            max_windows=int(args.max_windows),
            only_anomaly=not bool(args.include_normal_windows),
        )


if __name__ == "__main__":
    main()
