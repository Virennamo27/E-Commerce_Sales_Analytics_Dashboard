"""
01_eda.py
---------
Step 1 of the Superstore analytics pipeline.

What this script does
=====================
1. Loads and cleans the raw `Sample - Superstore.csv` via `utils.load_superstore`.
2. Prints a quick data-quality report (shape, dtypes, null counts, date range).
3. Generates the three EDA views the brief asked for:
     * monthly sales trend (with a 3-month rolling smoother)
     * profit margin distribution (overall + by category)
     * regional performance (sales / profit / margin per Region)
4. Saves every figure to outputs/figures/ and dumps a couple of summary
   tables to outputs/data/ so downstream scripts (and humans) can reuse them.

Run with:
    python scripts/01_eda.py
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from utils import DATA_DIR, FIG_DIR, load_superstore, save_fig

# Consistent visual style across all EDA charts.
sns.set_theme(style="whitegrid", context="talk")
plt.rcParams["figure.figsize"] = (12, 6)


# ---------------------------------------------------------------------------
# 1. Load + quick data-quality report
# ---------------------------------------------------------------------------
def data_quality_report(df: pd.DataFrame) -> None:
    """Print a compact data-quality summary to stdout."""
    print("=" * 70)
    print("DATA QUALITY REPORT")
    print("=" * 70)
    print(f"Rows x Cols          : {df.shape[0]:,} x {df.shape[1]}")
    print(f"Order date range     : {df['Order Date'].min().date()} -> "
          f"{df['Order Date'].max().date()}")
    print(f"Unique customers     : {df['Customer ID'].nunique():,}")
    print(f"Unique orders        : {df['Order ID'].nunique():,}")
    print(f"Unique products      : {df['Product ID'].nunique():,}")
    print(f"Duplicate rows       : {df.duplicated().sum()}")
    print("\nNull counts (non-zero only):")
    nulls = df.isna().sum()
    nulls = nulls[nulls > 0]
    print(nulls.to_string() if len(nulls) else "  (none)")
    print("\nNumeric summary:")
    print(df[["Sales", "Quantity", "Discount", "Profit"]].describe().round(2))
    print("=" * 70, "\n")


# ---------------------------------------------------------------------------
# 2. Sales trends over time
# ---------------------------------------------------------------------------
def plot_sales_trend(df: pd.DataFrame) -> pd.DataFrame:
    """Monthly sales line chart + 3-month rolling average overlay."""
    monthly = (
        df.groupby("Order Month", as_index=False)
          .agg(Sales=("Sales", "sum"), Profit=("Profit", "sum"),
               Orders=("Order ID", "nunique"))
          .sort_values("Order Month")
    )
    monthly["Sales_3M_Avg"] = monthly["Sales"].rolling(3, min_periods=1).mean()

    fig, ax = plt.subplots()
    ax.plot(monthly["Order Month"], monthly["Sales"],
            marker="o", label="Monthly sales", color="#1f77b4")
    ax.plot(monthly["Order Month"], monthly["Sales_3M_Avg"],
            linestyle="--", label="3-month rolling avg", color="#ff7f0e")
    ax.set_title("Monthly Sales Trend")
    ax.set_xlabel("Month")
    ax.set_ylabel("Sales (USD)")
    ax.legend()
    save_fig(fig, "01_sales_trend_monthly")
    plt.close(fig)

    # Also persist the table - feature engineering uses it.
    monthly.to_csv(DATA_DIR / "monthly_sales.csv", index=False)
    return monthly


# ---------------------------------------------------------------------------
# 3. Profit margin analysis
# ---------------------------------------------------------------------------
def plot_profit_margins(df: pd.DataFrame) -> pd.DataFrame:
    """Distribution of order-level profit margin and by-category boxplot."""
    # Margin per row: Profit / Sales. Guard against the rare zero-sales row.
    margin = df.assign(
        Margin=np.where(df["Sales"] != 0, df["Profit"] / df["Sales"], np.nan)
    )

    # ---- Histogram of order-level margin -------------------------------------
    fig, ax = plt.subplots()
    sns.histplot(margin["Margin"].clip(-1, 1), bins=60, kde=True, ax=ax,
                 color="#2ca02c")
    ax.axvline(0, color="red", linestyle="--", linewidth=1)
    ax.set_title("Order-Level Profit Margin Distribution (clipped to +/-100%)")
    ax.set_xlabel("Profit margin (Profit / Sales)")
    save_fig(fig, "02_profit_margin_hist")
    plt.close(fig)

    # ---- Boxplot of margin by category --------------------------------------
    fig, ax = plt.subplots()
    sns.boxplot(data=margin, x="Category", y="Margin", hue="Category",
                ax=ax, showfliers=False, palette="Set2", legend=False)
    ax.axhline(0, color="red", linestyle="--", linewidth=1)
    ax.set_title("Profit Margin by Category")
    ax.set_ylabel("Profit margin")
    save_fig(fig, "03_profit_margin_by_category")
    plt.close(fig)

    # Category-level summary table.
    cat_summary = (
        margin.groupby("Category")
              .agg(Sales=("Sales", "sum"), Profit=("Profit", "sum"),
                   Orders=("Order ID", "nunique"),
                   Avg_Margin=("Margin", "mean"))
              .assign(Margin_Pct=lambda d: d["Profit"] / d["Sales"])
              .round(4)
    )
    cat_summary.to_csv(DATA_DIR / "category_summary.csv")
    return cat_summary


# ---------------------------------------------------------------------------
# 4. Regional performance
# ---------------------------------------------------------------------------
def plot_regional_performance(df: pd.DataFrame) -> pd.DataFrame:
    """Sales, profit, and margin% by Region - one figure with 3 panels."""
    region = (
        df.groupby("Region")
          .agg(Sales=("Sales", "sum"), Profit=("Profit", "sum"),
               Orders=("Order ID", "nunique"),
               Customers=("Customer ID", "nunique"))
          .assign(Margin_Pct=lambda d: d["Profit"] / d["Sales"] * 100)
          .sort_values("Sales", ascending=False)
    )

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    sns.barplot(x=region.index, y=region["Sales"], hue=region.index, ax=axes[0], palette="Blues_d", legend=False)
    axes[0].set_title("Total Sales by Region")
    axes[0].set_ylabel("Sales (USD)")

    sns.barplot(x=region.index, y=region["Profit"], hue=region.index, ax=axes[1], palette="Greens_d", legend=False)
    axes[1].set_title("Total Profit by Region")
    axes[1].set_ylabel("Profit (USD)")

    sns.barplot(x=region.index, y=region["Margin_Pct"], hue=region.index, ax=axes[2], palette="Oranges_d", legend=False)
    axes[2].set_title("Profit Margin % by Region")
    axes[2].set_ylabel("Margin %")

    for ax in axes:
        ax.set_xlabel("")
    save_fig(fig, "04_regional_performance")
    plt.close(fig)

    region.round(2).to_csv(DATA_DIR / "region_summary.csv")
    return region


# ---------------------------------------------------------------------------
# 5. Bonus: top sub-categories heatmap (Region x Sub-Category profit)
# ---------------------------------------------------------------------------
def plot_subcategory_heatmap(df: pd.DataFrame) -> None:
    pivot = df.pivot_table(index="Sub-Category", columns="Region",
                           values="Profit", aggfunc="sum").fillna(0)
    pivot = pivot.loc[pivot.sum(axis=1).sort_values(ascending=False).index]

    fig, ax = plt.subplots(figsize=(10, 10))
    sns.heatmap(pivot, annot=True, fmt=".0f", cmap="RdYlGn", center=0,
                linewidths=0.4, ax=ax, cbar_kws={"label": "Profit (USD)"})
    ax.set_title("Profit by Sub-Category x Region")
    save_fig(fig, "05_subcategory_region_heatmap")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    df = load_superstore()
    data_quality_report(df)

    monthly = plot_sales_trend(df)
    cat = plot_profit_margins(df)
    region = plot_regional_performance(df)
    plot_subcategory_heatmap(df)

    print("Monthly sales (first/last):")
    print(monthly.head(3).to_string(index=False))
    print("...")
    print(monthly.tail(3).to_string(index=False))
    print("\nCategory summary:\n", cat)
    print("\nRegion summary:\n", region.round(2))
    print(f"\nFigures saved to: {FIG_DIR}")
    print(f"Data tables saved to: {DATA_DIR}")


if __name__ == "__main__":
    main()
