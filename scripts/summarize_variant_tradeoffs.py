from __future__ import annotations

import argparse
import csv
from pathlib import Path


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _find_mean_row(rows: list[dict[str, str]]) -> dict[str, str]:
    for row in rows:
        if row.get("seed", "").strip().lower() == "mean":
            return row
    raise ValueError("Could not find mean row")


def _to_float_any(row: dict[str, str], *keys: str) -> float:
    for key in keys:
        raw = row.get(key, "")
        if raw is None:
            continue
        raw = raw.strip()
        if raw != "":
            return float(raw)
    available = ", ".join(sorted(row.keys()))
    wanted = ", ".join(keys)
    raise KeyError(f"Missing numeric column (tried: {wanted}). Available: {available}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build a unified medium multiseed variant comparison table.")
    p.add_argument("--untuned-summary", type=str, required=True)
    p.add_argument("--tuned-summary", type=str, required=True)
    p.add_argument("--tuned64-summary", type=str, required=True)
    p.add_argument("--out-csv", type=str, required=True)
    return p.parse_args()


def _score(delta_dice: float, delta_hd95: float, delta_ece: float) -> float:
    # Higher is better; Dice gain rewarded, HD95 increase penalized, ECE gain rewarded mildly.
    return delta_dice - (0.0005 * delta_hd95) - (0.25 * delta_ece)


def main() -> None:
    args = parse_args()

    untuned_rows = _read_rows(Path(args.untuned_summary))
    tuned_rows = _read_rows(Path(args.tuned_summary))
    tuned64_rows = _read_rows(Path(args.tuned64_summary))

    untuned_mean = _find_mean_row(untuned_rows)
    tuned_mean = _find_mean_row(tuned_rows)
    tuned64_mean = _find_mean_row(tuned64_rows)

    baseline_dice = _to_float_any(untuned_mean, "baseline_dice")
    baseline_hd95 = _to_float_any(untuned_mean, "baseline_hd95")
    baseline_ece = _to_float_any(untuned_mean, "baseline_ece")

    variants = [
        {
            "variant": "baseline_medium",
            "mean_dice": baseline_dice,
            "delta_dice": 0.0,
            "mean_hd95": baseline_hd95,
            "delta_hd95": 0.0,
            "mean_ece": baseline_ece,
            "delta_ece": 0.0,
        },
        {
            "variant": "untuned_synth_16",
            "mean_dice": _to_float_any(untuned_mean, "synth_dice", "variant_dice"),
            "delta_dice": _to_float_any(untuned_mean, "delta_dice"),
            "mean_hd95": _to_float_any(untuned_mean, "synth_hd95", "variant_hd95"),
            "delta_hd95": _to_float_any(untuned_mean, "delta_hd95"),
            "mean_ece": _to_float_any(untuned_mean, "synth_ece", "variant_ece"),
            "delta_ece": _to_float_any(untuned_mean, "delta_ece"),
        },
        {
            "variant": "tuned_synth_16",
            "mean_dice": _to_float_any(tuned_mean, "tuned_synth_dice", "variant_dice", "synth_dice"),
            "delta_dice": _to_float_any(tuned_mean, "delta_tuned", "delta_dice"),
            "mean_hd95": _to_float_any(tuned_mean, "tuned_synth_hd95", "variant_hd95", "synth_hd95"),
            "delta_hd95": _to_float_any(tuned_mean, "delta_hd95_tuned", "delta_hd95"),
            "mean_ece": _to_float_any(tuned_mean, "tuned_synth_ece", "variant_ece", "synth_ece"),
            "delta_ece": _to_float_any(tuned_mean, "delta_ece_tuned", "delta_ece"),
        },
        {
            "variant": "tuned_synth_64",
            "mean_dice": _to_float_any(tuned64_mean, "variant_dice", "tuned_synth_dice", "synth_dice"),
            "delta_dice": _to_float_any(tuned64_mean, "delta_dice", "delta_tuned"),
            "mean_hd95": _to_float_any(tuned64_mean, "variant_hd95", "tuned_synth_hd95", "synth_hd95"),
            "delta_hd95": _to_float_any(tuned64_mean, "delta_hd95", "delta_hd95_tuned"),
            "mean_ece": _to_float_any(tuned64_mean, "variant_ece", "tuned_synth_ece", "synth_ece"),
            "delta_ece": _to_float_any(tuned64_mean, "delta_ece", "delta_ece_tuned"),
        },
    ]

    for row in variants:
        row["tradeoff_score"] = _score(
            delta_dice=float(row["delta_dice"]),
            delta_hd95=float(row["delta_hd95"]),
            delta_ece=float(row["delta_ece"]),
        )

    ranked = sorted(variants, key=lambda r: float(r["tradeoff_score"]), reverse=True)
    rank_map = {row["variant"]: i + 1 for i, row in enumerate(ranked)}
    for row in variants:
        row["rank"] = rank_map[row["variant"]]

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "variant",
        "mean_dice",
        "delta_dice",
        "mean_hd95",
        "delta_hd95",
        "mean_ece",
        "delta_ece",
        "tradeoff_score",
        "rank",
    ]

    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in variants:
            writer.writerow(row)

    print(f"Wrote unified comparison: {out_path}")


if __name__ == "__main__":
    main()
