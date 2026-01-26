# -*- coding: utf-8 -*-
"""
gfc_proxy.py

This script reads a predefined list of stock market indices, downloads their
price history, and constructs daily and monthly proxies for the Global Financial
Cycle (GFC) using Principal Component Analysis (PCA) based on  Habib & Venditti (2019)

It then compares these proxies against the  GFC factor from Miranda-Agrippino & Rey (2020), calculates
goodness-of-fit statistics, and generates a comparison chart. The outputs (data
and chart) are saved locally.
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
import matplotlib.dates as mdates


# --- Configuration ---
CONFIG = {
    "START_DATE": "2000-01-01",
    "MIN_START_YEAR": 2005,
    "ROLLING_WINDOW_DAILY": 21,  
    "OFFICIAL_GFC_URL": "https://www.dropbox.com/scl/fi/a1iqkelgv9yzlfdmwyqcn/GFC-Factor-Updates-2024.xlsx?rlkey=4hrbknfue1q1l8y7hntffj008&st=o1scqels&dl=1",
    "CHART_OUTPUT_PATH": "chart/gfc_comparison_plot.png",
    "DAILY_DATA_OUTPUT_PATH": "data/gfc_proxy_daily.csv",
    "MONTHLY_DATA_OUTPUT_PATH": "data/gfc_proxies_monthly.csv",
    "INDICES_FILE_PATH": "indices_list.txt",
}

def download_data(candidates: pd.DataFrame) -> pd.DataFrame:
    """Downloads time series for each index in the provided DataFrame."""
    print("\nDownloading data using pandas-datareader...")
    selected_series = {}
    for _, row in candidates.iterrows():
        country, symbol = row['country'], row['symbol']
        try:
            series = pdr.get_data_stooq(symbol, start=CONFIG["START_DATE"])['Close']
            if not series.empty and series.index.min().year <= CONFIG["MIN_START_YEAR"]:
                selected_series[country] = series.sort_index()
                print(f"  + Downloaded '{symbol}' for {country}")
            else:
                print(f"  - Skipped '{symbol}' (insufficient history)")
        except Exception:
            print(f"  - Failed to download '{symbol}'")
            continue

    if not selected_series:
        raise SystemExit("Could not download any valid time series from the list.")
    
    final_panel = pd.concat(selected_series, axis=1).ffill()
    print(f"\nSuccessfully created a panel with {final_panel.shape[1]} country indices.")
    return final_panel

def calculate_gfc_proxy(returns_panel: pd.DataFrame) -> (pd.Series, float):
    """Calculates the GFC proxy (PC1) from a panel of returns."""
    pca = PCA(n_components=1)
    gfc_raw = pca.fit_transform(returns_panel)
    gfc_z = pd.Series(gfc_raw.flatten(), index=returns_panel.index)
    gfc_z = (gfc_z - gfc_z.mean()) / gfc_z.std()
    variance_explained = pca.explained_variance_ratio_[0]
    return gfc_z, variance_explained, pca 

def analyze_plot_and_save(gfc_z_daily: pd.Series, gfc_z_monthly: pd.Series) -> (float, float):
    """Fetches the GFC, analyzes fit, plots, and saves all outputs."""
    print("\nComparing proxies with the Miranda-Agrippino & Rey (2020) GFC factor...")
    
    # 1. Fetch and process the benchmark GFC data
    gfc_mar = pd.read_excel(CONFIG["OFFICIAL_GFC_URL"], sheet_name="Standardized", skiprows=2, index_col=1, engine='openpyxl')
    gfc_mar = gfc_mar.iloc[:, [2]].dropna()
    gfc_mar.columns = ["GFC M&R (2020)-Monthly"]
    gfc_mar.index = pd.to_datetime(gfc_mar.index, format="%Y-%m") + pd.offsets.MonthEnd(0)

    # 2. Create cumulative, standardized versions for comparison
    gfc_proxy_monthly = pd.DataFrame(
        StandardScaler().fit_transform(gfc_z_monthly.cumsum().values.reshape(-1, 1)),
        index=gfc_z_monthly.index, columns=['GFC Proxy-Monthly']
    )
    gfc_proxy_daily = pd.DataFrame(
        StandardScaler().fit_transform(gfc_z_daily.cumsum().values.reshape(-1, 1)),
        index=gfc_z_daily.index, columns=['GFC Proxy-Daily']
    )

    # 3. Merge monthly series for statistical analysis
    monthly_comparison_unaligned = gfc_proxy_monthly.merge(gfc_mar, left_index=True, right_index=True, how="left")

    # 4. Remove any rows with NaN values to align the series perfectly
    monthly_comparison = monthly_comparison_unaligned.dropna()
    
    # 5. STATISTICAL ANALYSIS: Calculate goodness-of-fit
    correlation = monthly_comparison['GFC Proxy-Monthly'].corr(monthly_comparison['GFC M&R (2020)-Monthly'])
    mse = mean_squared_error(monthly_comparison['GFC M&R (2020)-Monthly'], monthly_comparison['GFC Proxy-Monthly'])
    print(f"\n--- Goodness-of-Fit (Monthly Proxy vs. Miranda-Agrippino & Rey (2020)) ---")
    print(f"Correlation: {correlation:.4f}")
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print("----------------------------------------------------")

    # 6. Create plot with a professional style
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(14, 8)) 
    ax.plot(gfc_proxy_daily.index, gfc_proxy_daily['GFC Proxy-Daily'],
            label='GFC Proxy (Daily)', color='#80B1D3', linewidth=1.2, zorder=1) 

    ax.plot(monthly_comparison_unaligned.index, monthly_comparison_unaligned['GFC Proxy-Monthly'],
            label=f'GFC Proxy (Monthly) | Corr: {correlation:.2f}', color='#1F78B4', 
            linewidth=2.0, marker='o', markersize=5, zorder=3)
    
    ax.plot(monthly_comparison.index, monthly_comparison['GFC M&R (2020)-Monthly'],
            label='GFC M&R (2020)', color='#E31A1C',
            linewidth=2.0, marker='x', markersize=6, zorder=2)
    
    # Formatting the plot
    ax.set_title(f"Global Financial Cycle (GFC) Proxies vs. Miranda-Agrippino & Rey (2020) Factor", fontsize=16, pad=20)
    ax.set_ylabel("Standardized Value (Cumulative)", fontsize=12)
    ax.set_xlabel(None)
    ax.legend(loc='upper left', frameon=True, fontsize=10)
    
    ax.xaxis.set_major_locator(mdates.YearLocator(2)) 
    ax.xaxis.set_minor_locator(mdates.YearLocator())  
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    plt.setp(ax.get_xticklabels(), rotation=0, ha='center')
    ax.tick_params(axis='x', which='minor', bottom=True)
    ax.axhline(0, color='grey', linestyle='--', linewidth=0.8)

    note = (
        "Source: Author's calculations using data from Stooq.com. GFC Factor from Miranda-Agrippino & Rey (2020).\n"
        "Methodology based on Habib & Venditti (2019).\n"
        "Note: Proxies are the first principal component (PC1) of returns from a panel of global stock indices. "
        "The cumulative series are standardized for comparison.\n"
        "Data available at https://github.com/iweigandi/daily-global-financial-cycle-proxy"
    )

    fig.text(0.08, 0.05, note, ha='left', va='bottom', fontsize=9, color='gray', wrap=True)
    
    plt.subplots_adjust(left=0.08, right=0.95, top=0.92, bottom=0.18) 

    # 7. Save outputs
    os.makedirs(os.path.dirname(CONFIG["CHART_OUTPUT_PATH"]), exist_ok=True)
    os.makedirs(os.path.dirname(CONFIG["DAILY_DATA_OUTPUT_PATH"]), exist_ok=True)
    plt.savefig(CONFIG["CHART_OUTPUT_PATH"])
    gfc_proxy_daily.to_csv(CONFIG["DAILY_DATA_OUTPUT_PATH"])
    gfc_proxy_monthly.to_csv(CONFIG["MONTHLY_DATA_OUTPUT_PATH"])
    print(f"\nChart saved to: {CONFIG['CHART_OUTPUT_PATH']}")
    print(f"Daily data saved to: {CONFIG['DAILY_DATA_OUTPUT_PATH']}")
    print(f"Monthly data saved to: {CONFIG['MONTHLY_DATA_OUTPUT_PATH']}")

    


def main():
    """Main execution pipeline."""
    # Read the static list of indices from the text file
    print(f"Reading index list from {CONFIG['INDICES_FILE_PATH']}...")
    try:
        candidates = pd.read_csv(CONFIG['INDICES_FILE_PATH'], names=['symbol', 'country'])
    except FileNotFoundError:
        print(f"Error: {CONFIG['INDICES_FILE_PATH']} not found. Please create it.")
        return
        
    price_panel = download_data(candidates)

    # Calculate MONTHLY proxy
    print("\nCalculating MONTHLY GFC proxy...")
    returns_monthly = price_panel.resample('M').last().pipe(lambda df: np.log(df / df.shift(1))).dropna()
    gfc_z_monthly, var_exp_monthly, pca_monthly = calculate_gfc_proxy(returns_monthly)
    print(f"Monthly proxy calculated. Variance Explained: {var_exp_monthly:.2%}")

    # Calculate DAILY proxy
    print("\nCalculating DAILY GFC proxy...")
    daily_returns = np.log(price_panel / price_panel.shift(1))
    smoothed_daily_returns = daily_returns.rolling(window=CONFIG["ROLLING_WINDOW_DAILY"], min_periods=10).mean().dropna()
    gfc_z_daily, var_exp_daily, _ = calculate_gfc_proxy(smoothed_daily_returns)
    print(f"Daily proxy calculated. Variance Explained: {var_exp_daily:.2%}")

       
    # Final analysis, plotting, and saving
    analyze_plot_and_save(gfc_z_daily, gfc_z_monthly)

if __name__ == "__main__":
    main()
