"""
Download and analyze NOAA AIS data for a specific year and month.

- Downloads daily AIS_YYYY_MM_DD.zip files from:
  https://coast.noaa.gov/htdata/CMSP/AISDataHandler/{year}/

- For each day:
    * Download ZIP (if exists)
    * Extract CSV(s) to AIS-YYYY-MM/
    * Sample up to N rows per file (default 10k) for analysis
    * Compute daily summary + contribute to monthly aggregates
    * Delete that day's CSV files after processing to save disk space

- Outputs (in AIS-YYYY-MM/results/):
    * daily_summary_YYYY_MM.csv
    * monthly_summary_YYYY_MM.csv
    * vesseltype_counts_month_YYYY_MM.csv
    * status_counts_month_YYYY_MM.csv

- Plots (in AIS-YYYY-MM/plots/):
    * vessel_type_distribution_YYYY_MM.(png/pdf)
    * status_distribution_YYYY_MM.(png/pdf)
    * sog_distribution_YYYY_MM.(png/pdf)

Requires:
    pip install requests pandas numpy matplotlib seaborn
"""

import sys
import calendar
import shutil
from pathlib import Path
from zipfile import ZipFile
from collections import Counter

import requests
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# -----------------------------
# CONFIG
# -----------------------------
DEFAULT_YEAR = 2024
DEFAULT_MONTH = 1

BASE_URL = "https://coast.noaa.gov/htdata/CMSP/AISDataHandler"

SAMPLE_ROWS_PER_FILE = 10_000  # max rows to sample per file for analysis

# -----------------------------
# AIS code dictionaries
# -----------------------------

AIS_VESSEL_TYPE = {
    0: "Not available",
    20: "Wing in ground (WIG)",
    30: "Fishing",
    31: "Towing",
    32: "Towing: length>200m or breadth>25m",
    33: "Dredging or underwater ops",
    34: "Diving ops",
    35: "Military ops",
    36: "Sailing",
    37: "Pleasure craft",
    40: "High speed craft (HSC)",
    50: "Pilot vessel",
    51: "Search and rescue vessel",
    52: "Tug",
    53: "Port tender",
    54: "Anti-pollution",
    55: "Law enforcement",
    58: "Medical transport",
    59: "Noncombatant",
    60: "Passenger",
    70: "Cargo",
    80: "Tanker",
    90: "Other / special",
}

AIS_NAV_STATUS = {
    0: "Under way using engine",
    1: "At anchor",
    2: "Not under command",
    3: "Restricted manoeuverability",
    4: "Constrained by her draught",
    5: "Moored",
    6: "Aground",
    7: "Engaged in fishing",
    8: "Under way sailing",
    9: "Reserved (HSC)",
    10: "Reserved (WIG)",
    11: "Reserved",
    12: "Reserved",
    13: "Reserved",
    14: "AIS-SART active",
    15: "Not defined",
}

AIS_TRANSCEIVER_CLASS = {
    "A": "Class A",
    "B": "Class B",
    "": "Unknown",
    None: "Unknown",
}

# -----------------------------
# Directory helpers
# -----------------------------


def prepare_output_dirs(year: int, month: int):
    base_dir = Path.cwd() / f"AIS-{year}-{month:02d}"
    results_dir = base_dir / "results"
    plots_dir = base_dir / "plots"

    # Start fresh: remove previous results folder if it exists
    if base_dir.exists():
        print(f"Cleaning existing directory {base_dir} ...")
        shutil.rmtree(base_dir)

    base_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    return base_dir, results_dir, plots_dir


# -----------------------------
# Download + extract
# -----------------------------


def download_and_extract_day(year: int, month: int, day: int, out_dir: Path):
    """
    Download AIS_YYYY_MM_DD.zip for given day (if available),
    extract into out_dir, return list of CSV files for that day.
    """
    zip_name = f"AIS_{year}_{month:02d}_{day:02d}.zip"
    url = f"{BASE_URL}/{year}/{zip_name}"
    print(f"  -> {url} ...", end="", flush=True)

    try:
        resp = requests.get(url, stream=True, timeout=60)
    except Exception as e:
        print(f" ERROR (request failed: {e})")
        return []

    if resp.status_code != 200:
        print(f" not found (HTTP {resp.status_code}), skipping.")
        return []

    # Save zip
    tmp_zip_path = out_dir / zip_name
    with open(tmp_zip_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=1_048_576):
            if chunk:
                f.write(chunk)

    # Extract
    try:
        with ZipFile(tmp_zip_path, "r") as zf:
            zf.extractall(out_dir)
        print(" downloaded and extracted.")
    except Exception as e:
        print(f" ERROR during unzip: {e}")
        tmp_zip_path.unlink(missing_ok=True)
        return []
    finally:
        # remove zip to save space
        tmp_zip_path.unlink(missing_ok=True)

    # Find CSVs for that day
    day_csv_files = sorted(out_dir.glob(f"AIS_{year}_{month:02d}_{day:02d}*.csv"))
    if not day_csv_files:
        print("  (no CSVs found after extraction)")
    return day_csv_files


# -----------------------------
# Analysis for one day
# -----------------------------


def analyze_day(
    year: int,
    month: int,
    day: int,
    csv_files,
    results_accum,
    monthly_accum,
):
    """
    Analyze all CSV files for a given day, using row sampling per file.
    Updates:
      - results_accum["daily_summaries"]: list of dicts with daily metrics
      - monthly_accum: counters and aggregates for the whole month
    """
    if not csv_files:
        return

    print(f"\nAnalyzing day {year}-{month:02d}-{day:02d} ({len(csv_files)} file(s))")

    # Daily accumulators
    day_total_messages = 0
    day_mmsi_set = set()
    day_vesseltype_counter = Counter()
    day_status_counter = Counter()
    day_sog_samples = []

    # Columns to load (if present)
    cols_of_interest = [
        "MMSI",
        "BaseDateTime",
        "LAT",
        "LON",
        "SOG",
        "COG",
        "Heading",
        "VesselName",
        "IMO",
        "CallSign",
        "VesselType",
        "Status",
        "Length",
        "Width",
        "Draft",
        "Cargo",
        "TransceiverClass",
    ]

    for csv_path in csv_files:
        print(f"    File: {csv_path.name}")
        try:
            preview = pd.read_csv(csv_path, nrows=5)
            usecols = [c for c in cols_of_interest if c in preview.columns]
            df = pd.read_csv(csv_path, usecols=usecols)
        except Exception as e:
            print(f"      ERROR reading {csv_path.name}: {e}")
            continue

        # Stats on full file (not sampled)
        day_total_messages += len(df)
        if "MMSI" in df.columns:
            df["MMSI"] = pd.to_numeric(df["MMSI"], errors="coerce").astype("Int64")
            day_mmsi_set.update(df["MMSI"].dropna().astype(int).tolist())

        # Convert numeric columns
        for num_col in ["SOG", "COG", "Heading", "Length", "Width", "Draft", "VesselType", "Status"]:
            if num_col in df.columns:
                df[num_col] = pd.to_numeric(df[num_col], errors="coerce")

        # Row sampling for analysis
        if len(df) > SAMPLE_ROWS_PER_FILE:
            df_sample = df.sample(SAMPLE_ROWS_PER_FILE, random_state=42)
        else:
            df_sample = df

        # Vessel type counts (sample-based)
        if "VesselType" in df_sample.columns:
            vt_counts = df_sample["VesselType"].value_counts(dropna=False)
            day_vesseltype_counter.update(vt_counts.to_dict())

        # Status counts (sample-based)
        if "Status" in df_sample.columns:
            st_counts = df_sample["Status"].value_counts(dropna=False)
            day_status_counter.update(st_counts.to_dict())

        # SOG samples
        if "SOG" in df_sample.columns:
            sog = df_sample["SOG"].dropna()
            if not sog.empty:
                day_sog_samples.append(sog.values)

    # Build daily summary
    if day_total_messages == 0:
        print("    No messages found for this day (after reading).")
        return

    day_sog_all = np.concatenate(day_sog_samples) if day_sog_samples else np.array([])
    sog_mean = float(day_sog_all.mean()) if day_sog_all.size > 0 else np.nan
    sog_median = float(np.median(day_sog_all)) if day_sog_all.size > 0 else np.nan
    sog_max = float(day_sog_all.max()) if day_sog_all.size > 0 else np.nan

    day_summary = {
        "date": f"{year}-{month:02d}-{day:02d}",
        "n_files": len(csv_files),
        "total_messages": day_total_messages,
        "unique_mmsi": len(day_mmsi_set),
        "sog_mean_sampled": sog_mean,
        "sog_median_sampled": sog_median,
        "sog_max_sampled": sog_max,
    }
    results_accum["daily_summaries"].append(day_summary)

    # Update monthly accumulators
    monthly_accum["total_messages_month"] += day_total_messages
    monthly_accum["mmsi_month"].update(day_mmsi_set)
    monthly_accum["vesseltype_counter_month"].update(day_vesseltype_counter)
    monthly_accum["status_counter_month"].update(day_status_counter)
    if day_sog_all.size > 0:
        monthly_accum["sog_samples_month"].append(day_sog_all)


# -----------------------------
# Monthly aggregate + plotting
# -----------------------------


def finalize_monthly_results(year: int, month: int, results_dir: Path, plots_dir: Path,
                             results_accum, monthly_accum):
    sns.set_theme(style="darkgrid")

    # ---- Daily summary CSV ----
    daily_df = pd.DataFrame(results_accum["daily_summaries"])
    daily_df.sort_values("date", inplace=True)
    daily_summary_path = results_dir / f"daily_summary_{year}_{month:02d}.csv"
    daily_df.to_csv(daily_summary_path, index=False)
    print(f"\nSaved daily summary to {daily_summary_path}")

    # ---- Monthly summary CSV ----
    sog_month_all = (
        np.concatenate(monthly_accum["sog_samples_month"])
        if monthly_accum["sog_samples_month"]
        else np.array([])
    )
    monthly_rows = [
        ("year", year),
        ("month", month),
        ("total_messages", monthly_accum["total_messages_month"]),
        ("total_unique_mmsi", len(monthly_accum["mmsi_month"])),
    ]
    if sog_month_all.size > 0:
        monthly_rows.extend([
            ("sog_mean_sampled", float(sog_month_all.mean())),
            ("sog_median_sampled", float(np.median(sog_month_all))),
            ("sog_max_sampled", float(sog_month_all.max())),
        ])

    monthly_df = pd.DataFrame(monthly_rows, columns=["metric", "value"])
    monthly_summary_path = results_dir / f"monthly_summary_{year}_{month:02d}.csv"
    monthly_df.to_csv(monthly_summary_path, index=False)
    print(f"Saved monthly summary to {monthly_summary_path}")

    # ---- Vessel type distribution (monthly) ----
    vt_counter = monthly_accum["vesseltype_counter_month"]
    if vt_counter:
        vt_month_df = pd.Series(vt_counter).rename("count").to_frame()
        vt_month_df.index.name = "VesselTypeCode"
        vt_month_df["VesselTypeLabel"] = vt_month_df.index.map(
            lambda code: AIS_VESSEL_TYPE.get(int(code), f"Other / code {int(code)}")
            if pd.notna(code)
            else "NaN"
        )
        vt_month_df.sort_values("count", ascending=False, inplace=True)

        vt_out = results_dir / f"vesseltype_counts_month_{year}_{month:02d}.csv"
        vt_month_df.to_csv(vt_out)
        print(f"Saved vessel type counts to {vt_out}")

        # Plot top 20 vessel types by message count (sample-based)
        top_vt = vt_month_df.head(20).copy()
        plt.figure(figsize=(10, 6))
        sns.barplot(
            x="count",
            y="VesselTypeLabel",
            data=top_vt,
        )
        plt.xlabel("Message count (sample-based)")
        plt.ylabel("Vessel type")
        plt.title(f"Vessel type distribution – {year}-{month:02d}")
        plt.tight_layout()

        png_path = plots_dir / f"vessel_type_distribution_{year}_{month:02d}.png"
        pdf_path = plots_dir / f"vessel_type_distribution_{year}_{month:02d}.pdf"
        plt.savefig(png_path)
        plt.savefig(pdf_path)
        plt.close()
        print(f"Saved vessel type plots to {png_path} and {pdf_path}")

    # ---- Status distribution (monthly) ----
    st_counter = monthly_accum["status_counter_month"]
    if st_counter:
        st_month_df = pd.Series(st_counter).rename("count").to_frame()
        st_month_df.index.name = "StatusCode"
        st_month_df["StatusLabel"] = st_month_df.index.map(
            lambda code: AIS_NAV_STATUS.get(int(code), f"Other / code {int(code)}")
            if pd.notna(code)
            else "NaN"
        )
        st_month_df.sort_values("count", ascending=False, inplace=True)

        st_out = results_dir / f"status_counts_month_{year}_{month:02d}.csv"
        st_month_df.to_csv(st_out)
        print(f"Saved status counts to {st_out}")

        plt.figure(figsize=(8, 6))
        sns.barplot(
            x="count",
            y="StatusLabel",
            data=st_month_df,
        )
        plt.xlabel("Message count (sample-based)")
        plt.ylabel("Navigation status")
        plt.title(f"Navigation status distribution – {year}-{month:02d}")
        plt.tight_layout()

        png_path = plots_dir / f"status_distribution_{year}_{month:02d}.png"
        pdf_path = plots_dir / f"status_distribution_{year}_{month:02d}.pdf"
        plt.savefig(png_path)
        plt.savefig(pdf_path)
        plt.close()
        print(f"Saved status plots to {png_path} and {pdf_path}")

    # ---- SOG histogram (monthly) ----
    if sog_month_all.size > 0:
        plt.figure(figsize=(10, 6))
        plt.hist(sog_month_all, bins=50)
        plt.xlabel("Speed over ground (knots)")
        plt.ylabel("Message count (sample-based)")
        plt.title(f"SOG distribution – {year}-{month:02d}")
        plt.tight_layout()

        png_path = plots_dir / f"sog_distribution_{year}_{month:02d}.png"
        pdf_path = plots_dir / f"sog_distribution_{year}_{month:02d}.pdf"
        plt.savefig(png_path)
        plt.savefig(pdf_path)
        plt.close()
        print(f"Saved SOG plots to {png_path} and {pdf_path}")


# -----------------------------
# CLI and main
# -----------------------------


def parse_args():
    """
    Usage:
        python download_and_analyze_ais.py [YEAR] [MONTH]

    If not provided, defaults from CONFIG are used.
    """
    if len(sys.argv) >= 3:
        year = int(sys.argv[1])
        month = int(sys.argv[2])
    else:
        year = DEFAULT_YEAR
        month = DEFAULT_MONTH
        print(f"No YEAR and MONTH provided, using defaults: {year}-{month:02d}")

    if not (2000 <= year <= 2100):
        raise ValueError("Year must be between 2000 and 2100.")
    if not (1 <= month <= 12):
        raise ValueError("Month must be between 1 and 12.")

    return year, month


def main():
    year, month = parse_args()

    base_dir, results_dir, plots_dir = prepare_output_dirs(year, month)

    # Accumulators
    results_accum = {
        "daily_summaries": [],
    }
    monthly_accum = {
        "total_messages_month": 0,
        "mmsi_month": set(),
        "vesseltype_counter_month": Counter(),
        "status_counter_month": Counter(),
        "sog_samples_month": [],
    }

    num_days = calendar.monthrange(year, month)[1]
    print(f"\nProcessing AIS data for {year}-{month:02d} ({num_days} days)...")

    for day in range(1, num_days + 1):
        # 1) Download + extract that day's data
        day_csv_files = download_and_extract_day(year, month, day, base_dir)
        if not day_csv_files:
            continue

        # 2) Analyze that day (sampling + daily summary)
        analyze_day(year, month, day, day_csv_files, results_accum, monthly_accum)

        # 3) Delete this day's CSV files to save disk space
        for f in day_csv_files:
            try:
                f.unlink()
            except Exception as e:
                print(f"  WARNING: could not delete {f}: {e}")

    if not results_accum["daily_summaries"]:
        print("\nNo data processed for this month.")
        return

    # 4) Final monthly results + plots
    finalize_monthly_results(year, month, results_dir, plots_dir, results_accum, monthly_accum)

    print("\nAll done.")


if __name__ == "__main__":
    main()
