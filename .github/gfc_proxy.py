# -*- coding: utf-8 -*-
"""
gfc_proxy.py

This script discovers national stock market indices from Stooq, downloads their
price history, and constructs daily and monthly proxies for the Global Financial
Cycle (GFC) using Principal Component Analysis (PCA).

It then compares these proxies against an official GFC factor, calculates
goodness-of-fit statistics, and generates a comparison chart. The outputs (data
and chart) are saved locally.

This script is designed to be run automatically, for instance, via a GitHub Action.
"""

import os
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from pandas_datareader import data as pdr

# --- Configuration ---
CONFIG = {
    "START_DATE": "2000-01-01",
    "MIN_START_YEAR": 2005,
    "MIN_COUNTRIES": 15,
    "ROLLING_WINDOW_DAILY": 21,  # Business days in a month for smoothing
    "OFFICIAL_GFC_URL": "https://www.dropbox.com/scl/fi/a1iqkelgv9yzlfdmwyqcn/GFC-Factor-Updates-2024.xlsx?rlkey=4hrbknfue1q1l8y7hntffj008&st=o1scqels&dl=1",
    "CHART_OUTPUT_PATH": "chart/gfc_comparison_plot.png",
    "DAILY_DATA_OUTPUT_PATH": "data/gfc_proxy_daily.csv",
    "MONTHLY_DATA_OUTPUT_PATH": "data/gfc_proxies_monthly.csv",
}

def discover_indices() -> pd.DataFrame:
    """Scrapes Stooq to find candidate country indices."""
    print("Discovering country indices from Stooq...")
    urls = ["https://stooq.com/t/?i=510", "https://stooq.com/t/?f=1&i=510&u=1&v=6", "https://stooq.com/t/?f=1&i=510&n=1&u=1&v=0"]
    tables = [tbl for url in urls for tbl in pd.read_html(url, header=0, flavor="lxml")]

    candidates_df = (
        pd.concat([t.iloc[:, :2].rename(columns={t.columns[0]: "symbol", t.columns[1]: "name"}) for t in tables if t.shape[1] >= 2])
        .dropna().astype(str)
        .drop_duplicates(subset="symbol")
    )

    bad_patterns = r"FUTURE|MSCI|STOXX|SECTOR|VOLATILITY|COMMOD|ETF|WORLD|EUROPE|ASIA|GLOBAL|EMERGING"
    candidates_df = candidates_df[~candidates_df['name'].str.contains(bad_patterns, case=False)]
    candidates_df = candidates_df[~candidates_df['symbol'].str.contains(r"^\.|\.F$", case=False)]

    candidates_df['country'] = candidates_df['name'].str.split(' - ').str[-1].str.replace(r'\(.*\)', '', regex=True).str.strip()
    candidates_df = candidates_df.dropna(subset=['country'])
    print(f"Found {len(candidates_df)} potential candidates.")
    return candidates_df

def download_data(candidates: pd.DataFrame) -> pd.DataFrame:
    """Downloads time series for the best index per country."""
    print("\nDownloading data using pandas-datareader...")
    selected_series = {}
    for _, row in candidates.sample(frac=1).iterrows():
        country, symbol = row['country'], row['symbol']
        if country in selected_series:
            continue
        try:
            series = pdr.get_data_stooq(symbol, start=CONFIG["START_DATE"])['Close']
            if not series.empty and series.index.min().year <= CONFIG["MIN_START_YEAR"]:
                selected_series[country] = series.sort_index()
                print(f"  + Selected '{symbol}' for {country}")
        except Exception:
            continue

    if not selected_series:
        raise SystemExit("Could not download any valid time series.")
    
    return pd.concat(selected_series, axis=1).ffill()

def calculate_gfc_proxy(returns_panel: pd.DataFrame) -> (pd.Series, float):
    """Calculates the GFC proxy (PC1) from a panel of returns."""
    pca = PCA(n_components=1)
    gfc_raw = pca.fit_transform(returns_panel)
    gfc_z = pd.Series(gfc_raw.flatten(), index=returns_panel.index)
    gfc_z = (gfc_z - gfc_z.mean()) / gfc_z.std()
    variance_explained = pca.explained_variance_ratio_[0]
    return gfc_z, variance_explained

def analyze_plot_and_save(gfc_z_daily: pd.Series, gfc_z_monthly: pd.Series):
    """Fetches official GFC, analyzes fit, plots, and saves all outputs."""
    print("\nComparing proxies with the official GFC factor...")
    
    # 1. Fetch and process official GFC data
    gfc_official = pd.read_excel(CONFIG["OFFICIAL_GFC_URL"], sheet_name="Standardized", skiprows=2, index_col=1, engine='openpyxl')
    gfc_official = gfc_official.iloc[:, [2]].dropna()
    gfc_official.columns = ["GFC Official (Monthly)"]
    gfc_official.index = pd.to_datetime(gfc_official.index, format="%Y-%m") + pd.offsets.MonthEnd(0)

    # 2. Create cumulative, standardized versions for comparison
    gfc_proxy_monthly = pd.DataFrame(
        StandardScaler().fit_transform(gfc_z_monthly.cumsum().values.reshape(-1, 1)),
        index=gfc_z_monthly.index, columns=['GFC Proxy (Monthly)']
    )
    gfc_proxy_daily = pd.DataFrame(
        StandardScaler().fit_transform(gfc_z_daily.cumsum().values.reshape(-1, 1)),
        index=gfc_z_daily.index, columns=['GFC Proxy (Daily)']
    )

    # 3. Merge monthly series for statistical analysis
    monthly_comparison = gfc_proxy_monthly.merge(gfc_official, left_index=True, right_index=True)

    # 4. STATISTICAL ANALYSIS: Calculate goodness-of-fit
    correlation = monthly_comparison['GFC Proxy (Monthly)'].corr(monthly_comparison['GFC Official (Monthly)'])
    mse = mean_squared_error(monthly_comparison['GFC Official (Monthly)'], monthly_comparison['GFC Proxy (Monthly)'])
    print(f"\n--- Goodness-of-Fit (Monthly Proxy vs. Official) ---")
    print(f"Correlation: {correlation:.4f}")
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print("----------------------------------------------------")

    # 5. Create plot
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.plot(gfc_proxy_daily.index, gfc_proxy_daily['GFC Proxy (Daily)'], label=f'GFC Proxy (Daily)', color='lightblue', linewidth=1.5, zorder=1)
    ax.plot(monthly_comparison.index, monthly_comparison['GFC Proxy (Monthly)'], label=f'GFC Proxy (Monthly) | Corr: {correlation:.2f}', color='navy', marker='o', markersize=4, zorder=2)
    ax.plot(monthly_comparison.index, monthly_comparison['GFC Official (Monthly)'], label='GFC Official (Monthly)', color='red', marker='x', markersize=5, zorder=3)
    
    ax.set_title(f"Daily & Monthly GFC Proxies vs. Official Factor (Updated: {datetime.now().strftime('%Y-%m-%d')})")
    ax.set_ylabel("Standardized Value (Cumulative)")
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax.legend()
    plt.tight_layout()

    # 6. Save outputs
    os.makedirs(os.path.dirname(CONFIG["CHART_OUTPUT_PATH"]), exist_ok=True)
    os.makedirs(os.path.dirname(CONFIG["DAILY_DATA_OUTPUT_PATH"]), exist_ok=True)
    plt.savefig(CONFIG["CHART_OUTPUT_PATH"])
    gfc_proxy_daily.to_csv(CONFIG["DAILY_DATA_OUTPUT_PATH"])
    monthly_comparison.to_csv(CONFIG["MONTHLY_DATA_OUTPUT_PATH"])
    print(f"\nChart saved to: {CONFIG['CHART_OUTPUT_PATH']}")
    print(f"Daily data saved to: {CONFIG['DAILY_DATA_OUTPUT_PATH']}")
    print(f"Monthly data saved to: {CONFIG['MONTHLY_DATA_OUTPUT_PATH']}")


def main():
    """Main execution pipeline."""
    candidates = discover_indices()
    price_panel = download_data(candidates)

    # Calculate MONTHLY proxy
    print("\nCalculating MONTHLY GFC proxy...")
    returns_monthly = price_panel.resample('ME').last().pipe(lambda df: np.log(df / df.shift(1))).dropna()
    gfc_z_monthly, var_exp_monthly = calculate_gfc_proxy(returns_monthly)
    print(f"Monthly proxy calculated. Variance Explained: {var_exp_monthly:.2%}")

    # Calculate DAILY proxy
    print("\nCalculating DAILY GFC proxy...")
    daily_returns = np.log(price_panel / price_panel.shift(1))
    smoothed_daily_returns = daily_returns.rolling(window=CONFIG["ROLLING_WINDOW_DAILY"], min_periods=10).mean().dropna()
    gfc_z_daily, var_exp_daily = calculate_gfc_proxy(smoothed_daily_returns)
    print(f"Daily proxy calculated. Variance Explained: {var_exp_daily:.2%}")

    # Final analysis, plotting, and saving
    analyze_plot_and_save(gfc_z_daily, gfc_z_monthly)

if __name__ == "__main__":
    main()