from __future__ import annotations

import argparse
import csv
from pathlib import Path


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _find_mean(rows: list[dict[str, str]]) -> dict[str, str]:
    for row in rows:
        if row.get("seed", "").strip().lower() == "mean":
            return row
    raise ValueError(f"No mean row in {rows}")


def _score(delta_dice: float, delta_hd95: float, delta_ece: float) -> float:
    return delta_dice - (0.0005 * delta_hd95) - (0.25 * delta_ece)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Summarize multi-dose sweep from per-dose summary CSV files.")
    p.add_argument("--doses", type=str, default="8,16,24,32")
    p.add_argument(
        "--summary-pattern",
        type=str,
        default="results/tables/seed_ablation_medium_tuned_dose{dose}_summary.csv",
        help="Path pattern containing {dose} placeholder",
    )
    p.add_argument("--out", type=str, default="results/tables/dose_sweep_medium_summary.csv")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    doses = [int(x.strip()) for x in args.doses.split(",") if x.strip()]

    rows_out: list[dict[str, float | int]] = []
    for dose in doses:
        path = Path(args.summary_pattern.format(dose=dose))
        if not path.exists():
            raise FileNotFoundError(f"Missing summary for dose {dose}: {path}")

        mean = _find_mean(_read_rows(path))
        delta_dice = float(mean["delta_dice"])
        delta_hd95 = float(mean["delta_hd95"])
        delta_ece = float(mean["delta_ece"])

        row = {
            "dose": dose,
            "baseline_dice": float(mean["baseline_dice"]),
            "dose_dice": float(mean["variant_dice"] if "variant_dice" in mean else mean["synth_dice"]),
            "delta_dice": delta_dice,
            "baseline_hd95": float(mean["baseline_hd95"]),
            "dose_hd95": float(mean["variant_hd95"] if "variant_hd95" in mean else mean["synth_hd95"]),
            "delta_hd95": delta_hd95,
            "baseline_ece": float(mean["baseline_ece"]),
            "dose_ece": float(mean["variant_ece"] if "variant_ece" in mean else mean["synth_ece"]),
            "delta_ece": delta_ece,
            "tradeoff_score": _score(delta_dice, delta_hd95, delta_ece),
        }
        rows_out.append(row)

    ranked = sorted(rows_out, key=lambda r: float(r["tradeoff_score"]), reverse=True)
    rank_map = {int(r["dose"]): i + 1 for i, r in enumerate(ranked)}
    for r in rows_out:
        r["rank"] = rank_map[int(r["dose"])]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "dose",
        "baseline_dice",
        "dose_dice",
        "delta_dice",
        "baseline_hd95",
        "dose_hd95",
        "delta_hd95",
        "baseline_ece",
        "dose_ece",
        "delta_ece",
        "tradeoff_score",
        "rank",
    ]
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in sorted(rows_out, key=lambda x: int(x["dose"])):
            writer.writerow(r)

    best = min(rows_out, key=lambda r: int(r["rank"]))
    print(f"Wrote dose summary: {out_path}")
    print(f"Best dose by tradeoff_score: {int(best['dose'])}")


if __name__ == "__main__":
    main()
