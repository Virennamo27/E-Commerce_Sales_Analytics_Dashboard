"""
04_export_dashboard_data.py
---------------------------
Step 4 of the Superstore analytics pipeline.

Joins everything we've built so far into a single denormalised, Tableau-ready
file:

  - row-level orders (cleaned + enriched in 02_feature_engineering)
  - customer-level RFM + segment labels (from 03_rfm_segmentation)
  - cohort fields (Cohort_Month, Cohort_Index)

Output:
  outputs/data/dashboard_data.csv

Why one big flat file?
Tableau is happiest with a single, denormalised extract: it lets the user
build slice-and-dice views without configuring relationships. Disk size is
fine here (~10k rows).

Run with:
    python scripts/04_export_dashboard_data.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Reuse what the previous scripts already implemented - no duplication.
# Note: filenames start with a digit, so we load them via importlib.util.
from utils import DATA_DIR, load_superstore


def _import_numbered(module_name: str):
    """Helper to import scripts whose filenames start with a digit."""
    import importlib.util
    from pathlib import Path
    path = Path(__file__).resolve().parent / f"{module_name}.py"
    spec = importlib.util.spec_from_file_location(module_name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> None:
    df = load_superstore()

    # --- enriched order rows (row-level features + monthly trend features) ---
    fe = _import_numbered("02_feature_engineering")
    rows = fe.add_row_features(df)
    monthly = fe.build_monthly_features(df)
    rows = fe.attach_monthly_to_rows(rows, monthly)

    # --- RFM table -----------------------------------------------------------
    rfm_mod = _import_numbered("03_rfm_segmentation")
    rfm = rfm_mod.build_rfm(df)

    rfm_join = rfm[[
        "Customer ID", "Recency", "Frequency", "Monetary",
        "R_Score", "F_Score", "M_Score", "RFM_Score", "RFM_Sum",
        "Segment_RFM", "First_Purchase",
    ]].rename(columns={"First_Purchase": "Customer_First_Purchase"})

    # --- Cohort fields per row ----------------------------------------------
    cohort = rows[["Customer ID", "Order Month"]].copy()
    first_month = (
        rows.groupby("Customer ID")["Order Month"].min().rename("Cohort_Month")
    )
    cohort = cohort.merge(first_month, on="Customer ID", how="left")
    cohort["Cohort_Index"] = (
        (cohort["Order Month"].dt.year - cohort["Cohort_Month"].dt.year) * 12
        + (cohort["Order Month"].dt.month - cohort["Cohort_Month"].dt.month)
    )
    rows = rows.assign(
        Cohort_Month=cohort["Cohort_Month"].values,
        Cohort_Index=cohort["Cohort_Index"].values,
    )

    # --- Final join ----------------------------------------------------------
    dash = rows.merge(rfm_join, on="Customer ID", how="left")

    # Tidy column ordering for Tableau.
    ordered = [
        # Identifiers
        "Row ID", "Order ID", "Order Date", "Ship Date", "Ship Mode", "Ship Days",
        "Customer ID", "Customer Name", "Segment",
        # Geography
        "Country", "Region", "State", "City", "Postal Code",
        # Product
        "Category", "Sub-Category", "Product ID", "Product Name",
        # Money
        "Sales", "Quantity", "Discount", "Discounted",
        "Profit", "Profit_Margin",
        "Order_Total_Sales", "Order_Total_Profit", "Order_Line_Items",
        # Date features
        "Order Year", "Order Month", "Order YearMonth", "Months_Since_Start",
        # Trend features
        "Sales_3M_RollAvg", "Profit_3M_RollAvg",
        "Sales_YoY_Growth", "Profit_YoY_Growth",
        # RFM
        "Recency", "Frequency", "Monetary",
        "R_Score", "F_Score", "M_Score", "RFM_Score", "RFM_Sum", "Segment_RFM",
        "Customer_First_Purchase",
        # Cohort
        "Cohort_Month", "Cohort_Index",
    ]
    # Keep only columns we know about (in case schema drifts), preserve order.
    ordered = [c for c in ordered if c in dash.columns]
    dash = dash[ordered]

    out = DATA_DIR / "dashboard_data.csv"
    dash.to_csv(out, index=False)

    # Summary printout
    print(f"Wrote {out}")
    print(f"Rows: {len(dash):,}    Cols: {dash.shape[1]}")
    print("\nColumns:")
    for c in dash.columns:
        print(f"  - {c}  ({dash[c].dtype})")
    print("\nSample (first 3 rows, key cols):")
    preview = dash[["Order Date", "Region", "Category", "Sales", "Profit",
                    "Profit_Margin", "Segment_RFM", "Cohort_Month",
                    "Cohort_Index"]].head(3)
    print(preview.to_string(index=False))


if __name__ == "__main__":
    main()
