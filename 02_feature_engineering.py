"""
02_feature_engineering.py
-------------------------
Step 2 of the Superstore analytics pipeline.

Adds derived columns we'll need for both modelling and the Tableau dashboard:

  * Profit_Margin       - row-level profit / sales (NaN-safe)
  * Discounted          - boolean flag for any discount > 0
  * Order_Total_Sales   - total sales of the parent order (Order ID)
  * Order_Total_Profit  - same for profit
  * YoY_Sales_Growth    - year-over-year % growth at month grain, broadcast
                          back onto each row
  * Sales_3M_RollAvg    - 3-month rolling avg sales at month grain
  * Profit_3M_RollAvg   - same for profit
  * Months_Since_Start  - tenure-style integer feature

Persists two artefacts:
  outputs/data/orders_enriched.csv   - row-level enriched dataset
  outputs/data/monthly_features.csv  - month-grain trend table

Run with:
    python scripts/02_feature_engineering.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from utils import DATA_DIR, load_superstore


# ---------------------------------------------------------------------------
# Row-level features
# ---------------------------------------------------------------------------
def add_row_features(df: pd.DataFrame) -> pd.DataFrame:
    """Per-row derived columns."""
    df = df.copy()

    # Profit margin - guard against zero sales (rare, but possible).
    df["Profit_Margin"] = np.where(
        df["Sales"] != 0, df["Profit"] / df["Sales"], np.nan
    )

    # Did this line have any discount applied?
    df["Discounted"] = df["Discount"] > 0

    # Order-level rollups, then broadcast back so each row carries the totals
    # for the order it belongs to. Useful for Tableau order-level views.
    order_totals = (
        df.groupby("Order ID")
          .agg(Order_Total_Sales=("Sales", "sum"),
               Order_Total_Profit=("Profit", "sum"),
               Order_Line_Items=("Row ID", "count"))
    )
    df = df.merge(order_totals, on="Order ID", how="left")

    return df


# ---------------------------------------------------------------------------
# Month-grain trend features (YoY growth + rolling averages)
# ---------------------------------------------------------------------------
def build_monthly_features(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate to month grain and compute trend features."""
    monthly = (
        df.groupby("Order Month", as_index=False)
          .agg(Sales=("Sales", "sum"),
               Profit=("Profit", "sum"),
               Orders=("Order ID", "nunique"),
               Customers=("Customer ID", "nunique"))
          .sort_values("Order Month")
          .reset_index(drop=True)
    )

    # 3-month rolling averages (current month + 2 prior).
    monthly["Sales_3M_RollAvg"] = monthly["Sales"].rolling(3, min_periods=1).mean()
    monthly["Profit_3M_RollAvg"] = monthly["Profit"].rolling(3, min_periods=1).mean()

    # YoY growth: compare each month to the same calendar month one year prior.
    # Trick: shift by 12 *positions* works because we have one row per month with
    # no gaps. We assert that's true to avoid silent bugs if data ever changes.
    expected = pd.date_range(monthly["Order Month"].min(),
                             monthly["Order Month"].max(), freq="MS")
    assert (monthly["Order Month"].values == expected.values).all(), (
        "Monthly series has gaps - YoY shift-by-12 would be wrong."
    )
    monthly["Sales_YoY_Growth"] = monthly["Sales"].pct_change(12)
    monthly["Profit_YoY_Growth"] = monthly["Profit"].pct_change(12)

    # Tenure feature relative to the first observed month.
    start = monthly["Order Month"].min()
    monthly["Months_Since_Start"] = (
        (monthly["Order Month"].dt.year - start.year) * 12
        + (monthly["Order Month"].dt.month - start.month)
    )

    return monthly


# ---------------------------------------------------------------------------
# Merge monthly trend features back onto each order row
# ---------------------------------------------------------------------------
def attach_monthly_to_rows(df_rows: pd.DataFrame,
                           monthly: pd.DataFrame) -> pd.DataFrame:
    """Broadcast month-grain features onto each row via Order Month."""
    cols = ["Order Month", "Sales_3M_RollAvg", "Profit_3M_RollAvg",
            "Sales_YoY_Growth", "Profit_YoY_Growth", "Months_Since_Start"]
    return df_rows.merge(monthly[cols], on="Order Month", how="left")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    df = load_superstore()

    print(f"Loaded {len(df):,} rows.")
    df_rows = add_row_features(df)
    monthly = build_monthly_features(df)
    df_rows = attach_monthly_to_rows(df_rows, monthly)

    rows_path = DATA_DIR / "orders_enriched.csv"
    monthly_path = DATA_DIR / "monthly_features.csv"
    df_rows.to_csv(rows_path, index=False)
    monthly.to_csv(monthly_path, index=False)

    print("\nMonthly features (head):")
    print(monthly.head(6).to_string(index=False))
    print("\nMonthly features (tail - YoY now populated):")
    print(monthly.tail(6).to_string(index=False))
    print(f"\nWrote: {rows_path}  ({len(df_rows):,} rows, {df_rows.shape[1]} cols)")
    print(f"Wrote: {monthly_path}  ({len(monthly):,} rows)")


if __name__ == "__main__":
    main()
