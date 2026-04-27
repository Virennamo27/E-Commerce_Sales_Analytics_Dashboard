# E-Commerce_Sales_Analytics_Dashboard

End-to-end sales analytics pipeline built on the classic **Sample - Superstore** dataset. The project loads and cleans ~10K order line-items, runs exploratory analysis, engineers trend and customer-level features (YoY growth, rolling averages, RFM, cohorts), and exports a single Tableau-ready extract.

## Dataset

`Sample - Superstore.csv` — 9,994 rows, 21 columns, covering 4 years of US retail orders (Jan 2014 – Dec 2017) across 4 regions, 3 product categories, and 793 customers.

| Metric | Value |
|---|---|
| Rows | 9,994 |
| Date range | 2014-01-03 → 2017-12-30 |
| Customers | 793 |
| Orders | 5,009 |
| Products | 1,862 |
| Total sales | $2,297,200.86 |
| Total profit | $286,397.02 |

## Project structure
E-Commerce Sales Analytics Dashboard/
├── Sample - Superstore.csv          # raw dataset
├── README.md
├── .gitignore
├── scripts/
│   ├── utils.py                     # shared loader + path constants
│   ├── 01_eda.py                    # load, clean, EDA charts
│   ├── 02_feature_engineering.py    # YoY growth, rolling avgs, margins
│   ├── 03_rfm_segmentation.py       # RFM scoring + cohort analysis
│   └── 04_export_dashboard_data.py  # Tableau-ready extract
└── outputs/
├── data/                        # 10 derived CSVs
└── figures/                     # 7 PNG charts

## Pipeline overview

The four scripts run in order. Each is independently runnable and writes its own artefacts; `04_export_dashboard_data.py` reuses the functions from scripts 02 and 03 so there's no logic duplication.

### `01_eda.py` — Load, clean, explore
- Reads the raw CSV (latin-1 encoded), parses `Order Date` / `Ship Date` as proper datetimes (M/D/YYYY), strips whitespace, drops duplicates.
- Adds convenience date columns (`Order Year`, `Order Month`, `Order YearMonth`, `Ship Days`).
- Prints a data-quality report.
- Generates EDA charts: monthly sales trend with 3-month smoother, profit-margin distribution, profit margin by category, regional sales/profit/margin %, sub-category × region profit heatmap.

### `02_feature_engineering.py` — Trend & order-level features
- **Row-level**: `Profit_Margin` (Profit / Sales, NaN-safe), `Discounted` flag, order-level rollups (`Order_Total_Sales`, `Order_Total_Profit`, `Order_Line_Items`).
- **Month-grain**: `Sales_3M_RollAvg`, `Profit_3M_RollAvg`, `Sales_YoY_Growth`, `Profit_YoY_Growth`, `Months_Since_Start`. Asserts the monthly series has no gaps before the YoY shift.
- Broadcasts month-grain features back onto every row so the downstream extract is self-contained.

### `03_rfm_segmentation.py` — Customers & cohorts
- **RFM**: per-customer Recency, Frequency, Monetary. Each component bucketed into quintiles (1–5, 5 = best). Concatenated `RFM_Score` (e.g. `545`) plus a readable `Segment_RFM` label using standard rules.
- Segments: Champions, Loyal Customers, Potential Loyalists, New Customers, Promising, Needs Attention, At Risk, Cant Lose Them, Lost.
- **Cohort analysis**: each customer's cohort = month of first purchase. Builds an active-customer count matrix and a retention % matrix (cohort × months since first purchase), saved as a heatmap.

### `04_export_dashboard_data.py` — Tableau extract
- Joins enriched orders + RFM + cohort fields into a single denormalised file.
- Writes `outputs/data/dashboard_data.csv` (9,994 rows × 47 columns) with a tidy column ordering grouped by: identifiers → geography → product → money → date features → trend features → RFM → cohort.

## How to run

Requires Python 3.10+ and the packages below.

```bash
pip install pandas numpy matplotlib seaborn

# from the project root
python scripts/01_eda.py
python scripts/02_feature_engineering.py
python scripts/03_rfm_segmentation.py
python scripts/04_export_dashboard_data.py
```

Each script is idempotent — it can be re-run any time and will overwrite its outputs.

## Outputs

### Data (`outputs/data/`)

| File | Description |
|---|---|
| `monthly_sales.csv` | Monthly sales / profit / orders + 3-mo rolling avg |
| `category_summary.csv` | Sales, profit, margin by category |
| `region_summary.csv` | Sales, profit, margin %, customer count by region |
| `monthly_features.csv` | Month-grain trend table (rolling avgs, YoY growth) |
| `orders_enriched.csv` | Row-level orders with all engineered features |
| `customer_rfm.csv` | One row per customer with RFM scores + segment |
| `rfm_segment_summary.csv` | Customer / revenue counts per segment |
| `cohort_counts.csv` | Long-form active-customer counts by cohort |
| `cohort_retention.csv` | Retention % matrix |
| **`dashboard_data.csv`** | **Final Tableau-ready flat file (47 cols)** |

### Figures (`outputs/figures/`)

| File | Chart |
|---|---|
| `01_sales_trend_monthly.png` | Monthly sales + 3-mo rolling avg |
| `02_profit_margin_hist.png` | Order-level margin distribution |
| `03_profit_margin_by_category.png` | Margin boxplot by category |
| `04_regional_performance.png` | Sales / profit / margin % by region |
| `05_subcategory_region_heatmap.png` | Profit by sub-category × region |
| `06_rfm_segments.png` | Customers and revenue per RFM segment |
| `07_cohort_retention.png` | Cohort retention heatmap |

## Key findings

- **Technology** is the most profitable category (17.4% margin) despite Furniture having comparable revenue (Furniture: 2.5% margin — barely breakeven).
- **West** and **East** regions dominate both sales and profit; **Central** has the weakest margin (~7.9%) driven by heavy losses in Tables and Bookcases.
- **Loyal Customers** and **Champions** together account for ~30% of customers but contribute the majority of revenue — the classic Pareto pattern.
- ~12% of customers return in the month following their first purchase; long-term cohort retention drops off quickly, suggesting room for retention-focused campaigns.

## Tech stack

- **Python**: pandas, numpy, matplotlib, seaborn
- **Tableau** (optional, downstream): point Tableau at `outputs/data/dashboard_data.csv` to build interactive dashboards on the engineered fields.

## Extending the project

- Add a forecasting script (e.g. Prophet / SARIMA on `monthly_features.csv`).
- Push the cleaned extract to a database (Postgres / BigQuery) instead of CSV.
- Wrap the pipeline in a Makefile or Airflow DAG for scheduled refreshes.
- Build an interactive web dashboard (Streamlit / Dash) over `dashboard_data.csv`.
