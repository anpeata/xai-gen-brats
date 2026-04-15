from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Metrics:
    dice_mean: float
    hd95_mean: float
    ece: float


def _read_metrics(path: Path) -> Metrics:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    return Metrics(
        dice_mean=float(payload["dice_mean"]),
        hd95_mean=float(payload["hd95_mean"]),
        ece=float(payload["ece"]),
    )


def _parse_seeds(raw: str) -> list[int]:
    seeds: list[int] = []
    for part in raw.replace(" ", "").split(","):
        if not part:
            continue
        seeds.append(int(part))
    if not seeds:
        raise ValueError("No seeds provided")
    return seeds


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Aggregate per-seed evaluation JSONs into a summary CSV.")
    p.add_argument(
        "--seeds",
        type=str,
        default="42,43,44",
        help="Comma-separated list of seeds (e.g. 42,43,44)",
    )
    p.add_argument(
        "--baseline-pattern",
        type=str,
        required=True,
        help="Path pattern with {seed} placeholder for baseline metrics JSON.",
    )
    p.add_argument(
        "--variant-pattern",
        type=str,
        required=True,
        help="Path pattern with {seed} placeholder for variant metrics JSON.",
    )
    p.add_argument(
        "--out",
        type=str,
        required=True,
        help="Output CSV path.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    seeds = _parse_seeds(args.seeds)

    rows: list[dict[str, float | str]] = []
    for seed in seeds:
        baseline_path = Path(args.baseline_pattern.format(seed=seed))
        variant_path = Path(args.variant_pattern.format(seed=seed))
        if not baseline_path.exists():
            raise FileNotFoundError(f"Missing baseline metrics: {baseline_path}")
        if not variant_path.exists():
            raise FileNotFoundError(f"Missing variant metrics: {variant_path}")

        base = _read_metrics(baseline_path)
        var = _read_metrics(variant_path)

        rows.append(
            {
                "seed": str(seed),
                "baseline_dice": base.dice_mean,
                "variant_dice": var.dice_mean,
                "delta_dice": var.dice_mean - base.dice_mean,
                "baseline_hd95": base.hd95_mean,
                "variant_hd95": var.hd95_mean,
                "delta_hd95": var.hd95_mean - base.hd95_mean,
                "baseline_ece": base.ece,
                "variant_ece": var.ece,
                "delta_ece": var.ece - base.ece,
            }
        )

    mean = {
        "seed": "mean",
        "baseline_dice": sum(float(r["baseline_dice"]) for r in rows) / len(rows),
        "variant_dice": sum(float(r["variant_dice"]) for r in rows) / len(rows),
        "delta_dice": sum(float(r["delta_dice"]) for r in rows) / len(rows),
        "baseline_hd95": sum(float(r["baseline_hd95"]) for r in rows) / len(rows),
        "variant_hd95": sum(float(r["variant_hd95"]) for r in rows) / len(rows),
        "delta_hd95": sum(float(r["delta_hd95"]) for r in rows) / len(rows),
        "baseline_ece": sum(float(r["baseline_ece"]) for r in rows) / len(rows),
        "variant_ece": sum(float(r["variant_ece"]) for r in rows) / len(rows),
        "delta_ece": sum(float(r["delta_ece"]) for r in rows) / len(rows),
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "seed",
        "baseline_dice",
        "variant_dice",
        "delta_dice",
        "baseline_hd95",
        "variant_hd95",
        "delta_hd95",
        "baseline_ece",
        "variant_ece",
        "delta_ece",
    ]

    with out_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)
        w.writerow(mean)

    print(f"Wrote summary CSV: {out_path}")


if __name__ == "__main__":
    main()
