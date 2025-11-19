# Storytelling mit Daten – Ammar Taha ata4447, Benjamin Hartmann bha9465
# AIS Data Analysis – (Jan–Mar 2022), Single month not available since timestamps are not available
#
# disclaimer:
# This dataset does NOT contain all AIS fields described in the official AIS specification, Thus, this analysis only covers the columns available in the provided CSV file
#
# plots are saved as PDF in:
#     results/plots/
# data tables are saved as CSV in:
#     results/data/

import shutil # high-level file operations
from pathlib import Path # file path management
import numpy as np # numerical computing library
import pandas as pd # data manipulation library
import seaborn as sns # standard visualization library
import matplotlib.pyplot as plt # plotting library


from sklearn.preprocessing import MinMaxScaler # data scaling
from sklearn.decomposition import PCA # principal component analysis
import umap.umap_ as umap   # pip install umap-learn, dimensionality reduction


DATA_FILE = "ais_data.csv"

PLOTS_DIR = Path("results/plots")
DATA_DIR = Path("results/data")

if PLOTS_DIR.exists():
    shutil.rmtree(PLOTS_DIR)
if DATA_DIR.exists():
    shutil.rmtree(DATA_DIR)

PLOTS_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="darkgrid")


# helper functions

def save_plot(fig, filename: str):
    out_path = PLOTS_DIR / f"{filename}.pdf"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[PLOT] Saved: {out_path}")


def save_data(df: pd.DataFrame, filename: str):
    out_path = DATA_DIR / f"{filename}.csv"
    df.to_csv(out_path, index=False)
    print(f"[DATA] Saved: {out_path}")


# load and clean data

def load_and_clean():
    print("\nLoading AIS dataset...")
    df = pd.read_csv(DATA_FILE)
    print("Initial shape:", df.shape)

    # drop index column if present
    for col in df.columns:
        if col.lower().startswith("unnamed"):
            df = df.drop(columns=[col])
            break

    print("Columns:", list(df.columns))

    # add area, but missing length or width results in NaN
    if {"length", "width"}.issubset(df.columns):
        df["area"] = df["width"] * df["length"]

    # remove extreme outliers that likely indicate faulty data
    if "sog" in df.columns:
        df = df[(df["sog"] >= 0) & (df["sog"] <= 100)]

    if "length" in df.columns:
        df = df[(df["length"] > 0) & (df["length"] <= 1000)]

    if "width" in df.columns:
        df = df[(df["width"] > 0) & (df["width"] <= 100)]

    if "draught" in df.columns:
        df = df[(df["draught"].isna()) | ((df["draught"] >= 0) & (df["draught"] <= 50))]

    # Impute static vessel attributes per MMSI
    static_cols = ["length", "width", "draught", "area"]
    existing_static = [c for c in static_cols if c in df.columns]

    for col in existing_static:
        df[col] = df.groupby("mmsi")[col].transform(
            lambda x: x.fillna(x.median())
        )

    # drop rows missing key info
    before = len(df)
    df = df.dropna(subset=["shiptype"])
    print(f"Dropped {before - len(df)} rows missing shiptype.")

    return df


# analysis, ship type distribution
def analyze_shiptype_distribution(df):
    counts = (
        df.groupby("shiptype")
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )
    counts["percent"] = 100 * counts["count"] / counts["count"].sum()

    save_data(counts, "shiptype_distribution")

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(counts["shiptype"], counts["count"])
    ax.set_title("Ship Type Distribution")
    ax.set_xlabel("AIS Messages")
    ax.invert_yaxis()
    save_plot(fig, "shiptype_distribution")


# navigational status by shiptype

def analyze_navstatus_by_shiptype(df):
    if "navigationalstatus" not in df.columns:
        return

    table = (
        df.groupby(["shiptype", "navigationalstatus"])
        .size()
        .reset_index(name="count")
    )
    save_data(table, "navstatus_by_shiptype")

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(data=table, x="shiptype", y="count",
                hue="navigationalstatus", ax=ax)
    ax.set_title("Navigational Status by Ship Type")
    ax.set_xlabel("Ship Type")
    ax.tick_params(axis="x", rotation=30)
    save_plot(fig, "navstatus_by_shiptype")


# speed distributions

def analyze_speed_by_shiptype(df):
    if "sog" not in df.columns:
        return

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.boxplot(data=df, x="shiptype", y="sog", ax=ax)
    ax.set_title("Speed over Ground by Ship Type")
    ax.set_ylabel("Speed (knots)")
    ax.tick_params(axis="x", rotation=30)
    save_plot(fig, "speed_boxplot_by_shiptype")

    # Summary CSV
    stats = (
        df.groupby("shiptype")["sog"]
        .agg(["count", "mean", "median", "std", "min", "max"])
        .reset_index()
    )
    save_data(stats, "speed_stats_by_shiptype")
    

# ship dimensions

def analyze_dimensions(df):
    cols = [c for c in ["length", "width", "draught", "area"] if c in df.columns]
    if not cols:
        return

    # summary stats
    stats = (
        df.groupby("shiptype")[cols]
        .agg(["count", "mean", "median", "std", "min", "max"])
    )
    save_data(stats.reset_index(), "dimension_stats_by_shiptype")

    # boxplots
    for col in cols:
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.boxplot(data=df, x="shiptype", y=col, ax=ax)
        ax.set_title(f"{col.capitalize()} by Ship Type")
        ax.tick_params(axis="x", rotation=30)
        save_plot(fig, f"{col}_boxplot_by_shiptype")

    # length vs width scatter
    if {"length", "width"}.issubset(df.columns):
        sample = df.sample(min(5000, len(df)), random_state=42)
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.scatter(sample["length"], sample["width"], alpha=0.3)
        ax.set_title("Length vs Width (Sample)")
        ax.set_xlabel("Length (m)")
        ax.set_ylabel("Width (m)")
        save_plot(fig, "length_vs_width_scatter")


# correlation analysis

def analyze_correlation(df):
    numeric = df.select_dtypes(include=[np.number])
    corr = numeric.corr()

    save_data(corr.reset_index(), "correlation_matrix")

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, cmap="Spectral", ax=ax)
    ax.set_title("Correlation Matrix")
    save_plot(fig, "correlation_matrix")


# top 20 most active vessels

def analyze_top_vessels(df):
    top = (
        df.groupby(["mmsi", "shiptype"])
        .size()
        .reset_index(name="messages")
        .sort_values("messages", ascending=False)
        .head(20)
    )
    save_data(top, "top_20_vessels")

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.barh(top["mmsi"].astype(str), top["messages"])
    ax.set_title("Top 20 Most Active Vessels")
    ax.set_xlabel("AIS Messages")
    ax.invert_yaxis()
    save_plot(fig, "top_20_vessels")


# pca, umap cluster visualization

def analyze_pca_umap(df):
    # use only numeric data for pca/umap
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric = df[numeric_cols].fillna(0)

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(numeric)

    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(scaled)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(pca_result[:, 0], pca_result[:, 1],
               c=pd.factorize(df["shiptype"])[0],
               cmap="tab20", alpha=0.5, s=5)
    ax.set_title("PCA – Ship Type Clusters")
    save_plot(fig, "pca_clusters")

    reducer = umap.UMAP(n_neighbors=30, min_dist=0.1, random_state=42)
    umap_result = reducer.fit_transform(scaled)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(umap_result[:, 0], umap_result[:, 1],
               c=pd.factorize(df["shiptype"])[0],
               cmap="tab20", alpha=0.5, s=5)
    ax.set_title("UMAP – Ship Type Clusters")
    save_plot(fig, "umap_clusters")


# main

def main():
    df = load_and_clean()
    print(f"Cleaned dataset shape: {df.shape}")
    print(f"Unique vessels: {df['mmsi'].nunique()}")

    analyze_shiptype_distribution(df)
    analyze_navstatus_by_shiptype(df)
    analyze_speed_by_shiptype(df)
    analyze_dimensions(df)
    analyze_correlation(df)
    analyze_top_vessels(df)
    analyze_pca_umap(df)

    print("\nAll analyses completed. Files saved in `results/`.")


if __name__ == "__main__":
    main()
