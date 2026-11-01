"""Feedback loop (Phase 10).

- submit_sample(formulation): validate a measured sample, append to
  data/raw/user_data.csv with source="user".
- retrain(): back up old models to models/archive/<timestamp>/, then re-run
  clean -> kg_build -> features -> train, returning before/after metrics.

Run CLI: python -m src.feedback submit  (reads a sample JSON from stdin)
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from config import (  # noqa: E402
    INPUT_FEATURES,
    TARGET_PROPERTIES,
    Formulation,
    validate_formulation,
)

USER_DATA_PATH = Path("data/raw/user_data.csv")
MODELS_DIR = Path("models")
ARCHIVE_DIR = Path("models/archive")

ALL_COLUMNS = INPUT_FEATURES + list(TARGET_PROPERTIES.keys()) + ["source", "reference"]


def _ensure_user_data() -> None:
    USER_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not USER_DATA_PATH.exists():
        pd.DataFrame(columns=ALL_COLUMNS).to_csv(USER_DATA_PATH, index=False)


def submit_sample(formulation: dict[str, Any]) -> tuple[bool, str | list[str]]:
    """Validate and save a user-submitted measured sample.

    Returns (ok, message_or_errors). On success appends to
    data/raw/user_data.csv with source="user".
    """
    errs = validate_formulation(
        Formulation(
            **{k: formulation[k] for k in Formulation.__dataclass_fields__ if k in formulation},
            source="user",
        )
    )
    if errs:
        return False, errs

    _ensure_user_data()
    row = {k: formulation.get(k) for k in ALL_COLUMNS}
    row["source"] = "user"
    df = pd.read_csv(USER_DATA_PATH)
    if len(df):
        df = pd.concat([df, pd.DataFrame([row], columns=ALL_COLUMNS)], ignore_index=True)
    else:
        df = pd.DataFrame([row], columns=ALL_COLUMNS)
    df.to_csv(USER_DATA_PATH, index=False)
    return True, f"Sample saved to {USER_DATA_PATH} (source=user)"


def list_user_samples() -> pd.DataFrame:
    """Return all user-submitted samples."""
    _ensure_user_data()
    return pd.read_csv(USER_DATA_PATH)


def _run(module: str) -> None:
    """Run `python -m src.<module>`, raising on failure."""
    result = subprocess.run(
        [sys.executable, "-m", f"src.{module}"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        sys.stdout.write(result.stdout)
        sys.stderr.write(result.stderr)
        raise RuntimeError(f"python -m src.{module} failed")


def _archive_models() -> Path:
    """Move existing models into a timestamped archive folder. Returns archive dir."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    archive = ARCHIVE_DIR / ts
    archive.mkdir(parents=True, exist_ok=True)
    moved = False
    for p in MODELS_DIR.glob("*.joblib"):
        shutil.move(str(p), str(archive / p.name))
        moved = True
    if not moved:
        # no models to archive; remove empty dir
        try:
            archive.rmdir()
        except OSError:
            pass
        return archive  # may not exist
    return archive


def retrain() -> dict[str, Any]:
    """Re-run clean -> kg_build -> features -> train with archived old models.

    Returns dict with before/after metrics and archive path.
    """
    from pathlib import Path as _P

    METRICS_JSON = Path("reports/metrics.json")
    before = None
    if METRICS_JSON.exists():
        before = json.loads(METRICS_JSON.read_text())

    archive = _archive_models()

    # Re-run the pipeline
    _run("clean")
    _run("kg_build")
    _run("features")
    _run("train")

    after = None
    if METRICS_JSON.exists():
        after = json.loads(METRICS_JSON.read_text())

    return {
        "archive": str(archive) if archive.exists() else None,
        "before": before,
        "after": after,
        "user_samples": int(len(list_user_samples())),
    }


def main() -> None:
    """CLI entry:
      python -m src.feedback submit <sample.json>
      python -m src.feedback retrain
    """
    if len(sys.argv) < 2:
        print("Usage: python -m src.feedback {submit|retrain} [sample.json]")
        raise SystemExit(1)
    cmd = sys.argv[1]
    if cmd == "submit":
        path = sys.argv[2] if len(sys.argv) > 2 else "-"
        if path == "-":
            formulation = json.load(sys.stdin)
        else:
            formulation = json.loads(Path(path).read_text())
        ok, msg = submit_sample(formulation)
        print(msg)
        if not ok:
            raise SystemExit(1)
    elif cmd == "retrain":
        out = retrain()
        print(json.dumps(out, indent=2))
    else:
        print(f"Unknown command: {cmd}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
