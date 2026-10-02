"""Command line interface: ``python -m quantumretail <command>``."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _fetch(args) -> None:
    from .data import fetch

    train, ev = fetch(args.cache)
    print(f"train: {train}\neval : {ev}")


def _validate_recovery(args) -> None:
    from . import recovery as rec
    from .data import build_panel, fetch

    tp, ep = fetch(args.cache)
    panel = build_panel(tp, ep).first_days(90)
    res = rec.validate_recovery(panel)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k != "by_stockout_onset"}, indent=1))


def _benchmark(args) -> None:
    from .backtest import BenchmarkConfig, run_benchmark
    from .data import fetch

    tp, ep = fetch(args.cache)
    cfg = BenchmarkConfig(n_train_series=args.train_series, n_calib_series=args.calib_series)
    run_benchmark(tp, ep, args.out, cfg)


def _eda(args) -> None:
    from . import eda
    from . import recovery as rec
    from .data import build_panel, fetch

    tp, ep = fetch(args.cache)
    panel = build_panel(tp, ep).first_days(90)
    prof = rec.fit_profile(panel, slice(0, 61))
    demand, _ = rec.recover_panel(panel, prof, 0.05)
    res = eda.effects(panel, demand, panel.sales)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


def _build_demo(args) -> None:
    from .data import fetch
    from .demo import build_demo_bundle

    tp, ep = fetch(args.cache)
    build_demo_bundle(tp, ep, args.models, args.panel)
    print(f"wrote {args.models} and {args.panel}")


def _forecast(args) -> None:
    import numpy as np

    from .bundle import load_panel
    from .service import DemandService

    svc = DemandService(args.models, load_panel(args.panel))
    s = svc.panel.static
    hit = np.where((s["store_id"] == args.store) & (s["product_id"] == args.product))[0]
    if len(hit) == 0:
        raise SystemExit("series not in the demo panel; run with --list to see available series")
    fc = svc.forecast(int(hit[0]))
    plan = svc.order_plan(fc, args.underage_cost, args.overage_cost)
    show = fc[["date", "point", "lo80", "hi80", "observed_sales"]].assign(date=fc["date"].dt.strftime("%Y-%m-%d"))
    print(show.round(2).to_string(index=False))
    print(f"\nnewsvendor order plan (critical fractile {plan.critical_fractile:.2f}):")
    pf = plan.frame[["date", "order_qty", "expected_fill_rate", "expected_waste_units"]]
    print(pf.assign(date=pf["date"].dt.strftime("%Y-%m-%d")).round(2).to_string(index=False))


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="quantumretail")
    ap.add_argument("--cache", default="data/raw", help="where the dataset is cached")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("fetch", help="download the dataset").set_defaults(fn=_fetch)

    p = sub.add_parser("validate-recovery", help="controlled-censoring validation of demand recovery")
    p.add_argument("--out", default="benchmarks/results/recovery_validation.json")
    p.set_defaults(fn=_validate_recovery)

    p = sub.add_parser("benchmark", help="full forecasting and inventory benchmark")
    p.add_argument("--out", default="benchmarks/results")
    p.add_argument("--train-series", type=int, default=10_000)
    p.add_argument("--calib-series", type=int, default=5_000)
    p.set_defaults(fn=_benchmark)

    p = sub.add_parser("eda", help="within-series demand effects")
    p.add_argument("--out", default="benchmarks/results/eda_effects.json")
    p.set_defaults(fn=_eda)

    p = sub.add_parser("build-demo", help="train the compact demo model and write the demo panel")
    p.add_argument("--models", default="models/qr_v2")
    p.add_argument("--panel", default="data/demo/demo_panel.npz")
    p.set_defaults(fn=_build_demo)

    p = sub.add_parser("forecast", help="print a forecast and order plan for a demo series")
    p.add_argument("--models", default="models/qr_v2")
    p.add_argument("--panel", default="data/demo/demo_panel.npz")
    p.add_argument("--store", type=int, required=True)
    p.add_argument("--product", type=int, required=True)
    p.add_argument("--underage-cost", type=float, default=3.0)
    p.add_argument("--overage-cost", type=float, default=1.0)
    p.set_defaults(fn=_forecast)

    args = ap.parse_args(argv)
    args.fn(args)
