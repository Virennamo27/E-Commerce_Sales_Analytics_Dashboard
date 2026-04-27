"""
03_rfm_segmentation.py
----------------------
Step 3 of the Superstore analytics pipeline.

Builds two customer-level views:

  1. RFM segmentation
     - Recency  : days since the customer's last order (lower is better)
     - Frequency: number of distinct orders
     - Monetary : total sales (USD)
     Each component is bucketed into quintiles 1-5 (5 = best). The three
     digits are concatenated into an RFM_Score (e.g. '545') and mapped to
     human-readable segments (Champions, Loyal Customers, At Risk, ...).

  2. Cohort analysis
     - Each customer is assigned a cohort = month of their first purchase.
     - We then count active customers per (Cohort_Month, Cohort_Index)
       where Cohort_Index is the # of months since first purchase.
     - From that we derive a retention matrix (% of cohort still active).

Persists:
  outputs/data/customer_rfm.csv     - one row per customer
  outputs/data/cohort_counts.csv    - long-form active-customer counts
  outputs/data/cohort_retention.csv - retention % matrix (cohort x months)
  outputs/figures/06_rfm_segments.png
  outputs/figures/07_cohort_retention.png

Run with:
    python scripts/03_rfm_segmentation.py
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from utils import DATA_DIR, load_superstore, save_fig

sns.set_theme(style="whitegrid", context="talk")


# ---------------------------------------------------------------------------
# 1. RFM
# ---------------------------------------------------------------------------
def build_rfm(df: pd.DataFrame) -> pd.DataFrame:
    """Compute Recency, Frequency, Monetary per Customer ID + scoring + segment."""
    # The "snapshot" date is one day after the last observed order, so the
    # most recent buyer has Recency = 1 day rather than 0.
    snapshot = df["Order Date"].max() + pd.Timedelta(days=1)

    rfm = (
        df.groupby(["Customer ID", "Customer Name", "Segment"])
          .agg(Recency=("Order Date", lambda s: (snapshot - s.max()).days),
               Frequency=("Order ID", "nunique"),
               Monetary=("Sales", "sum"),
               First_Purchase=("Order Date", "min"),
               Last_Purchase=("Order Date", "max"),
               Total_Profit=("Profit", "sum"))
          .reset_index()
    )

    # Quintile scoring (1..5).
    # - Recency: lower is better, so labels are reversed.
    # - Frequency: ties on the boundaries are common; rank first to spread them
    #   out before binning, otherwise qcut will complain about duplicate edges.
    rfm["R_Score"] = pd.qcut(rfm["Recency"], 5, labels=[5, 4, 3, 2, 1]).astype(int)
    rfm["F_Score"] = pd.qcut(rfm["Frequency"].rank(method="first"),
                             5, labels=[1, 2, 3, 4, 5]).astype(int)
    rfm["M_Score"] = pd.qcut(rfm["Monetary"], 5, labels=[1, 2, 3, 4, 5]).astype(int)

    rfm["RFM_Score"] = (
        rfm["R_Score"].astype(str)
        + rfm["F_Score"].astype(str)
        + rfm["M_Score"].astype(str)
    )
    rfm["RFM_Sum"] = rfm[["R_Score", "F_Score", "M_Score"]].sum(axis=1)

    # Map to readable segments. Standard RFM segmentation rules.
    rfm["Segment_RFM"] = rfm.apply(_label_segment, axis=1)
    return rfm


def _label_segment(row: pd.Series) -> str:
    """Standard RFM segment labels based on R/F/M quintiles."""
    r, f, m = row["R_Score"], row["F_Score"], row["M_Score"]

    if r >= 4 and f >= 4 and m >= 4:
        return "Champions"
    if r >= 3 and f >= 4:
        return "Loyal Customers"
    if r >= 4 and f <= 2:
        return "New Customers"
    if r >= 3 and f >= 3 and m >= 3:
        return "Potential Loyalists"
    if r <= 2 and f >= 4:
        return "At Risk"
    if r <= 2 and f >= 2 and m >= 3:
        return "Cant Lose Them"
    if r <= 2 and f <= 2 and m <= 2:
        return "Lost"
    if r >= 3 and f <= 2 and m <= 2:
        return "Promising"
    return "Needs Attention"


def plot_rfm_segments(rfm: pd.DataFrame) -> None:
    """Bar chart of customer counts per segment + revenue per segment."""
    seg = (
        rfm.groupby("Segment_RFM")
           .agg(Customers=("Customer ID", "nunique"),
                Revenue=("Monetary", "sum"),
                Avg_Recency=("Recency", "mean"),
                Avg_Frequency=("Frequency", "mean"))
           .sort_values("Revenue", ascending=False)
    )

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    sns.barplot(x=seg.index, y=seg["Customers"], hue=seg.index, ax=axes[0], palette="viridis", legend=False)
    axes[0].set_title("Customers by RFM Segment")
    axes[0].set_ylabel("# Customers")
    axes[0].tick_params(axis="x", rotation=30)

    sns.barplot(x=seg.index, y=seg["Revenue"], hue=seg.index, ax=axes[1], palette="magma", legend=False)
    axes[1].set_title("Revenue by RFM Segment")
    axes[1].set_ylabel("Revenue (USD)")
    axes[1].tick_params(axis="x", rotation=30)

    save_fig(fig, "06_rfm_segments")
    plt.close(fig)

    seg.round(2).to_csv(DATA_DIR / "rfm_segment_summary.csv")


# ---------------------------------------------------------------------------
# 2. Cohort analysis
# ---------------------------------------------------------------------------
def build_cohort(df: pd.DataFrame):
    """Cohort = month of customer's first purchase. Returns counts + retention."""
    df = df.copy()
    df["Cohort_Month"] = (
        df.groupby("Customer ID")["Order Month"].transform("min")
    )

    # Cohort_Index = number of whole months between this order and the
    # customer's first-ever order month (0 = first month).
    def months_between(later: pd.Series, earlier: pd.Series) -> pd.Series:
        return (later.dt.year - earlier.dt.year) * 12 + (
            later.dt.month - earlier.dt.month
        )

    df["Cohort_Index"] = months_between(df["Order Month"], df["Cohort_Month"])

    cohort_counts = (
        df.groupby(["Cohort_Month", "Cohort_Index"])["Customer ID"]
          .nunique()
          .reset_index(name="Active_Customers")
    )

    cohort_pivot = cohort_counts.pivot(index="Cohort_Month",
                                       columns="Cohort_Index",
                                       values="Active_Customers")
    cohort_size = cohort_pivot.iloc[:, 0]
    retention = cohort_pivot.divide(cohort_size, axis=0)

    return cohort_counts, cohort_pivot, retention


def plot_cohort_retention(retention: pd.DataFrame) -> None:
    """Heatmap of retention % over time for each cohort."""
    # Pre-format the index/columns so seaborn picks them up directly and
    # we don't have to mess with set_yticklabels (which trips over the auto
    # locator when the matrix is tall).
    pretty = retention.copy() * 100
    pretty.index = [d.strftime("%Y-%m") for d in pretty.index]

    fig, ax = plt.subplots(figsize=(16, 12))
    sns.heatmap(pretty, annot=True, fmt=".0f", cmap="YlGnBu",
                vmin=0, vmax=100, cbar_kws={"label": "Retention %"}, ax=ax,
                annot_kws={"size": 7})
    ax.set_title("Cohort Retention % (rows = first-purchase month, "
                 "cols = months since first purchase)")
    ax.set_xlabel("Months since first purchase")
    ax.set_ylabel("Cohort (first purchase month)")
    plt.setp(ax.get_yticklabels(), rotation=0)
    save_fig(fig, "07_cohort_retention")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    df = load_superstore()

    rfm = build_rfm(df)
    rfm_path = DATA_DIR / "customer_rfm.csv"
    rfm.to_csv(rfm_path, index=False)
    plot_rfm_segments(rfm)

    cohort_counts, cohort_pivot, retention = build_cohort(df)
    cohort_counts.to_csv(DATA_DIR / "cohort_counts.csv", index=False)
    retention.round(4).to_csv(DATA_DIR / "cohort_retention.csv")
    plot_cohort_retention(retention)

    print(f"RFM table     -> {rfm_path}  ({len(rfm):,} customers)")
    print("\nSegment distribution:")
    print(rfm["Segment_RFM"].value_counts().to_string())
    print(f"\nCohorts       : {retention.shape[0]} cohorts x "
          f"{retention.shape[1]} month buckets")
    print(f"Avg 1-month retention: "
          f"{retention.iloc[:, 1].dropna().mean() * 100:.1f}%")


if __name__ == "__main__":
    main()
