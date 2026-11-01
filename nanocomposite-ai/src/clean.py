"""Data cleaning + merging (Phase 2).

Merges data/raw/seed_data.csv (seed_example) with data/processed/synthetic.csv
(synthetic), drops duplicates, validates every row via config.validate_formulation,
logs (does not silently impute) removed rows, and writes:

  - data/processed/dataset.csv        (merged, validated)
  - reports/data_report.md            (row counts per source, missing values, describe())
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from config import (  # noqa: E402
    INPUT_FEATURES,
    TARGET_PROPERTIES,
    Formulation,
    validate_formulation,
)

SEED_PATH = Path("data/raw/seed_data.csv")
SYNTHETIC_PATH = Path("data/processed/synthetic.csv")
OUT_PATH = Path("data/processed/dataset.csv")
REPORT_PATH = Path("reports/data_report.md")

ALL_COLUMNS = INPUT_FEATURES + list(TARGET_PROPERTIES.keys()) + ["source", "reference"]


def load_seed() -> pd.DataFrame:
    if not SEED_PATH.exists():
        raise FileNotFoundError(f"Missing seed file: {SEED_PATH}")
    df = pd.read_csv(SEED_PATH)
    missing = [c for c in ALL_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"seed_data.csv missing columns: {missing}")
    return df[ALL_COLUMNS]


def load_synthetic() -> pd.DataFrame:
    if not SYNTHETIC_PATH.exists():
        raise FileNotFoundError(f"Missing synthetic file: {SYNTHETIC_PATH} (run src.data_gen first)")
    df = pd.read_csv(SYNTHETIC_PATH)
    missing = [c for c in ALL_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"synthetic.csv missing columns: {missing}")
    return df[ALL_COLUMNS]


def validate_dataframe(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split df into valid + removed (invalid) rows using the Formulation validator."""
    valid_rows: list[pd.Series] = []
    removed_rows: list[pd.Series] = []
    removed_reasons: list[str] = []

    for _, row in df.iterrows():
        row_dict = {k: row[k] for k in Formulation.__dataclass_fields__ if k in row}
        errs = validate_formulation(Formulation(**row_dict))
        if errs:
            removed_rows.append(row)
            removed_reasons.append("; ".join(errs))
        else:
            valid_rows.append(row)

    valid = pd.DataFrame(valid_rows, columns=df.columns) if valid_rows else df.iloc[:0].copy()
    removed = pd.DataFrame(removed_rows, columns=df.columns) if removed_rows else df.iloc[:0].copy()
    if not removed.empty:
        removed["_removal_reason"] = removed_reasons
    return valid, removed


def build_report(df: pd.DataFrame, removed: pd.DataFrame) -> str:
    lines: list[str] = []
    lines.append("# Data Cleaning Report")
    lines.append("")
    lines.append(f"Total rows after merge: {len(df) + len(removed)}")
    lines.append(f"Valid rows kept: {len(df)}")
    lines.append(f"Rows removed (invalid): {len(removed)}")
    lines.append("")
    lines.append("## Row counts per source")
    src_counts = df["source"].value_counts().to_dict() if not df.empty else {}
    for src, cnt in sorted(src_counts.items()):
        lines.append(f"- {src}: {cnt}")
    lines.append("")
    lines.append("## Missing values (valid dataset)")
    if not df.empty:
        missing = df.isna().sum()
        for col, cnt in missing.items():
            lines.append(f"- {col}: {int(cnt)}")
    else:
        lines.append("- (no valid rows)")
    lines.append("")
    lines.append("## Numeric summary (describe())")
    if not df.empty:
        num = df.select_dtypes(include="number")
        lines.append("```")
        lines.append(num.describe().to_string())
        lines.append("```")
    else:
        lines.append("- (no valid rows)")
    lines.append("")
    if not removed.empty:
        lines.append("## Removed rows (first 20)")
        lines.append("```")
        lines.append(removed.drop(columns=["_removal_reason"]).head(20).to_string(index=False))
        lines.append("```")
        lines.append("")
        lines.append("## Removal reasons (first 20)")
        lines.append("```")
        for reason in removed["_removal_reason"].head(20):
            lines.append(f"- {reason}")
        lines.append("```")
    return "\n".join(lines)


def main() -> None:
    """CLI entry: python -m src.clean"""
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    Path("reports").mkdir(parents=True, exist_ok=True)

    seed_df = load_seed()
    synth_df = load_synthetic()

    merged = pd.concat([seed_df, synth_df], ignore_index=True)
    before = len(merged)

    # Drop exact duplicates on all feature+target columns (keep source/reference for audit)
    dedup_cols = INPUT_FEATURES + list(TARGET_PROPERTIES.keys())
    merged = merged.drop_duplicates(subset=dedup_cols, keep="first")
    after_dedup = len(merged)
    print(f"Merged {before} rows -> {after_dedup} after duplicate removal")

    valid, removed = validate_dataframe(merged)
    print(f"Valid: {len(valid)}, Removed (invalid): {len(removed)}")

    if not removed.empty:
        removed_path = Path("data/processed/removed_rows.csv")
        removed.to_csv(removed_path, index=False)
        print(f"Removed rows logged to {removed_path}")

    # Reorder columns
    valid = valid[ALL_COLUMNS]
    valid.to_csv(OUT_PATH, index=False)
    print(f"Valid dataset saved -> {OUT_PATH} ({len(valid)} rows)")

    report = build_report(valid, removed)
    REPORT_PATH.write_text(report)
    print(f"Report saved -> {REPORT_PATH}")


if __name__ == "__main__":
    main()
