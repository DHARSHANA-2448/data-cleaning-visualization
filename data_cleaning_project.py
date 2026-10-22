"""
Data Cleaning & Visualization Project
-------------------------------------
Usage:
    python data_cleaning_project.py                 # runs on a built-in messy demo dataset
    python data_cleaning_project.py mydata.csv      # runs on YOUR dataset

Outputs (in ./output):
    cleaned_data.csv, cleaning_log.txt, charts/*.png, report.html
"""
import sys, os, base64
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid", palette="deep")
OUT = "output"
CH = os.path.join(OUT, "charts")
os.makedirs(CH, exist_ok=True)
log = []


def say(msg):
    print(msg)
    log.append(msg)


# ---------------------------------------------------------------- 1. LOAD
def make_demo_data(n=600, seed=42):
    """Messy sales-style dataset so the project runs without any download."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "order_id": range(1, n + 1),
        "region": rng.choice(["North", "South", "East", "West", " north", "SOUTH"], n),
        "category": rng.choice(["Electronics", "Clothing", "Grocery", "Furniture"], n),
        "age": rng.integers(18, 70, n).astype(float),
        "quantity": rng.integers(1, 10, n).astype(float),
        "unit_price": rng.normal(500, 150, n).round(2),
        "rating": rng.choice([1, 2, 3, 4, 5], n, p=[.05, .1, .2, .35, .3]).astype(float),
        "order_date": pd.date_range("2025-01-01", periods=n, freq="14h").astype(str),
    })
    df["revenue"] = (df["quantity"] * df["unit_price"]).round(2)
    # inject problems
    for col in ["age", "rating", "unit_price", "region"]:
        df.loc[rng.choice(n, 40, replace=False), col] = np.nan
    df.loc[rng.choice(n, 8, replace=False), "unit_price"] = 20000   # outliers
    df.loc[rng.choice(n, 5, replace=False), "age"] = 250             # impossible ages
    df = pd.concat([df, df.sample(30, random_state=1)], ignore_index=True)  # duplicates
    return df


path = sys.argv[1] if len(sys.argv) > 1 else None
df = pd.read_csv(path) if path else make_demo_data()
say(f"Loaded {'file: ' + path if path else 'demo dataset'} -> {df.shape[0]} rows x {df.shape[1]} columns")

# ---------------------------------------------------------------- 2. INSPECT
raw = df.copy()
say("\n--- RAW DATA QUALITY ---")
say(f"Duplicate rows : {df.duplicated().sum()}")
say("Missing values per column:\n" + df.isna().sum()[df.isna().sum() > 0].to_string())

# ---------------------------------------------------------------- 3. CLEAN
# 3a. tidy column names + text
df.columns = df.columns.str.strip().str.lower().str.replace(r"\s+", "_", regex=True)
for c in df.select_dtypes(include="object"):
    df[c] = df[c].astype("string").str.strip()

# 3b. dates
for c in df.columns:
    if "date" in c or "time" in c:
        df[c] = pd.to_datetime(df[c], errors="coerce")

# 3c. standardise categories (e.g. ' north' / 'SOUTH' -> 'North' / 'South')
cat_cols = [c for c in df.select_dtypes(include=["object", "string"]) if df[c].nunique() < 30]
for c in cat_cols:
    df[c] = df[c].str.title()

# 3d. duplicates
before = len(df)
df = df.drop_duplicates()
say(f"\n[Duplicates] removed {before - len(df)} rows")

# 3e. missing values: median for numeric, mode for categorical
num_cols = df.select_dtypes(include="number").columns.tolist()
id_like = [c for c in num_cols if c.endswith("id") or df[c].is_monotonic_increasing and df[c].nunique() == len(df)]
num_cols = [c for c in num_cols if c not in id_like]
for c in num_cols:
    n = df[c].isna().sum()
    if n:
        df[c] = df[c].fillna(df[c].median())
        say(f"[Missing] {c}: filled {n} values with median")
for c in cat_cols:
    n = df[c].isna().sum()
    if n:
        df[c] = df[c].fillna(df[c].mode()[0])
        say(f"[Missing] {c}: filled {n} values with mode")

# 3f. outliers: IQR rule, capped (winsorised) instead of dropped
for c in num_cols:
    q1, q3 = df[c].quantile([.25, .75])
    iqr = q3 - q1
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    n = ((df[c] < lo) | (df[c] > hi)).sum()
    if n:
        df[c] = df[c].clip(lo, hi)
        say(f"[Outliers] {c}: capped {n} values to [{lo:.1f}, {hi:.1f}]")

# 3g. recompute derived columns if they exist
if {"quantity", "unit_price", "revenue"} <= set(df.columns):
    df["revenue"] = (df["quantity"] * df["unit_price"]).round(2)

df.to_csv(os.path.join(OUT, "cleaned_data.csv"), index=False)
say(f"\nCleaned data: {df.shape[0]} rows x {df.shape[1]} columns (saved to output/cleaned_data.csv)")

# ---------------------------------------------------------------- 4. VISUALISE
charts = []


def save(fig, name, title, insight):
    p = os.path.join(CH, name)
    fig.tight_layout()
    fig.savefig(p, dpi=130)
    plt.close(fig)
    charts.append((p, title, insight))


# 4a. before vs after missing values
miss_before = raw.isna().sum()
miss_before = miss_before[miss_before > 0]
if len(miss_before):
    fig, ax = plt.subplots(figsize=(8, 4))
    miss_before.sort_values().plot.barh(ax=ax, color="#e4572e")
    ax.set_title("Missing values found in raw data")
    ax.set_xlabel("Count")
    save(fig, "01_missing.png", "Missing values (raw data)",
         f"{int(miss_before.sum())} missing cells across {len(miss_before)} columns were imputed.")

# 4b. outliers before vs after
if num_cols:
    show = [c for c in num_cols if c in raw.columns][:4]
    fig, axes = plt.subplots(1, len(show), figsize=(4 * len(show), 4))
    axes = np.atleast_1d(axes)
    for ax, c in zip(axes, show):
        sns.boxplot(data=pd.DataFrame({"raw": raw[c], "cleaned": df[c]}), ax=ax)
        ax.set_title(c)
    save(fig, "02_outliers.png", "Outliers: before vs after cleaning",
         "Extreme values were capped using the IQR rule, so distributions are no longer distorted.")

# 4c. distributions
if num_cols:
    fig = plt.figure(figsize=(12, 7))
    for i, c in enumerate(num_cols[:4], 1):
        ax = fig.add_subplot(2, 2, i)
        sns.histplot(df[c], kde=True, ax=ax)
        ax.set_title(f"Distribution of {c}")
    save(fig, "03_distributions.png", "Distributions of numeric columns",
         "Histograms with density curves show the shape and spread of each numeric feature.")

# 4d. category counts
for c in cat_cols[:2]:
    fig, ax = plt.subplots(figsize=(7, 4))
    order = df[c].value_counts().index
    sns.countplot(data=df, y=c, order=order, ax=ax)
    ax.set_title(f"Records per {c}")
    top = order[0]
    save(fig, f"04_{c}.png", f"Records per {c}", f"'{top}' is the most common {c} ({df[c].value_counts().iloc[0]} records).")

# 4e. average of target numeric by category
target = "revenue" if "revenue" in df.columns else (num_cols[0] if num_cols else None)
if target and cat_cols:
    c = cat_cols[0]
    fig, ax = plt.subplots(figsize=(7, 4))
    g = df.groupby(c)[target].mean().sort_values()
    g.plot.barh(ax=ax, color="#2e86ab")
    ax.set_title(f"Average {target} by {c}")
    save(fig, "05_group.png", f"Average {target} by {c}",
         f"{g.index[-1]} has the highest average {target} ({g.iloc[-1]:,.0f}); {g.index[0]} the lowest ({g.iloc[0]:,.0f}).")

# 4f. time trend
date_cols = df.select_dtypes(include="datetime").columns
if len(date_cols) and target:
    d = date_cols[0]
    ts = df.set_index(d)[target].resample("ME").sum()
    fig, ax = plt.subplots(figsize=(9, 4))
    ts.plot(ax=ax, marker="o", color="#3b7a57")
    ax.set_title(f"Monthly total {target}")
    save(fig, "06_trend.png", f"Monthly {target} trend",
         f"Peak month: {ts.idxmax():%b %Y} ({ts.max():,.0f}).")

# 4g. correlation heatmap
if len(num_cols) > 1:
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(df[num_cols].corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Correlation heatmap")
    corr = df[num_cols].corr().abs().where(~np.eye(len(num_cols), dtype=bool)).stack()
    a, b = corr.idxmax()
    save(fig, "07_corr.png", "Correlation heatmap", f"Strongest relationship: {a} and {b} (|r| = {corr.max():.2f}).")

# ---------------------------------------------------------------- 5. REPORT
def b64(p):
    return base64.b64encode(open(p, "rb").read()).decode()


kpis = [("Rows (raw)", len(raw)), ("Rows (clean)", len(df)),
        ("Duplicates removed", int(raw.duplicated().sum())),
        ("Missing cells fixed", int(raw.isna().sum().sum()))]
cards = "".join(f"<div class='k'><b>{v:,}</b><span>{k}</span></div>" for k, v in kpis)
imgs = "".join(
    f"<section><h2>{t}</h2><img src='data:image/png;base64,{b64(p)}'><p>💡 {i}</p></section>"
    for p, t, i in charts)
html = f"""<!doctype html><html><head><meta charset='utf-8'><title>Data Report</title><style>
body{{font-family:Segoe UI,Arial;background:#f4f6fa;margin:0;padding:30px;color:#222}}
h1{{text-align:center}} .kp{{display:flex;gap:16px;justify-content:center;flex-wrap:wrap;margin:20px}}
.k{{background:#fff;padding:18px 28px;border-radius:12px;box-shadow:0 2px 8px #0001;text-align:center}}
.k b{{display:block;font-size:28px;color:#2e86ab}} section{{background:#fff;max-width:900px;margin:20px auto;padding:20px;border-radius:12px;box-shadow:0 2px 8px #0001}}
img{{max-width:100%}} pre{{background:#fff;max-width:900px;margin:auto;padding:16px;border-radius:12px;white-space:pre-wrap}}
</style></head><body><h1>📊 Data Cleaning & Visualization Report</h1><div class='kp'>{cards}</div>
{imgs}<h2 style='text-align:center'>Cleaning log</h2><pre>{chr(10).join(log)}</pre></body></html>"""
open(os.path.join(OUT, "report.html"), "w", encoding="utf-8").write(html)
open(os.path.join(OUT, "cleaning_log.txt"), "w", encoding="utf-8").write("\n".join(log))
print("\nDone! Open output/report.html in your browser.")
