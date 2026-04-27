"""
utils.py
--------
Shared helpers for the Superstore analytics pipeline.

Centralising the loader/cleaner means every downstream script reads the data
the exact same way and we don't end up with subtle date-parsing or dtype
mismatches between scripts.
"""

from __future__ import annotations

from pathlib import Path
import pandas as pd

# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------
# scripts/utils.py -> project root is the parent of `scripts/`
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_CSV = PROJECT_ROOT / "Sample - Superstore.csv"

OUTPUT_DIR = PROJECT_ROOT / "outputs"
DATA_DIR = OUTPUT_DIR / "data"
FIG_DIR = OUTPUT_DIR / "figures"

# Make sure the output folders exist no matter which script the user runs first.
for _p in (DATA_DIR, FIG_DIR):
    _p.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Load + clean
# ---------------------------------------------------------------------------
def load_superstore(path: Path | str = RAW_CSV) -> pd.DataFrame:
    """Load the raw Superstore CSV and return a cleaned DataFrame.

    Cleaning steps:
      * read with latin-1 (the file ships with a few non-UTF8 product names)
      * parse Order Date / Ship Date as real datetimes (M/D/YYYY format)
      * strip whitespace on object columns
      * drop exact duplicate rows (defensive - none in the sample, but cheap)
      * sort by Order Date so time-series ops behave predictably
      * add a few convenience date columns used everywhere downstream
    """
    df = pd.read_csv(path, encoding="latin1")

    # Parse dates explicitly; the file uses US M/D/YYYY style.
    for col in ("Order Date", "Ship Date"):
        df[col] = pd.to_datetime(df[col], format="%m/%d/%Y", errors="coerce")

    # Trim whitespace on string columns.
    obj_cols = df.select_dtypes(include="object").columns
    df[obj_cols] = df[obj_cols].apply(lambda s: s.str.strip())

    # Defensive de-dup, then sort.
    df = df.drop_duplicates().sort_values("Order Date").reset_index(drop=True)

    # Convenience date features used by EDA, FE, RFM and the dashboard export.
    df["Order Year"] = df["Order Date"].dt.year
    df["Order Month"] = df["Order Date"].dt.to_period("M").dt.to_timestamp()
    df["Order YearMonth"] = df["Order Date"].dt.strftime("%Y-%m")
    df["Ship Days"] = (df["Ship Date"] - df["Order Date"]).dt.days

    return df


def save_fig(fig, name: str):
    """Save a Matplotlib figure to outputs/figures/<name>.png and return the path."""
    out = FIG_DIR / f"{name}.png"
    fig.tight_layout()
    fig.savefig(out, dpi=120, bbox_inches="tight")
    return out
