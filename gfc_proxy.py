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
import time
from urllib.parse import quote
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import matplotlib.dates as mdates


# --- Configuration ---
CONFIG = {
    "START_DATE": "2000-01-01",
    "MIN_START_YEAR": 2005,
    "OFFICIAL_GFC_URL": "https://www.dropbox.com/scl/fi/a1iqkelgv9yzlfdmwyqcn/GFC-Factor-Updates-2024.xlsx?rlkey=4hrbknfue1q1l8y7hntffj008&st=o1scqels&dl=1",
    "CHART_OUTPUT_PATH": "chart/gfc_comparison_plot.png",
    "DAILY_DATA_OUTPUT_PATH": "data/gfc_proxy_daily.csv",
    "MONTHLY_DATA_OUTPUT_PATH": "data/gfc_proxies_monthly.csv",
    "HISTORICAL_PRICE_PANEL_PATH": "data/stooq_price_panel.csv",
    "PRICE_OVERLAP_DIAGNOSTICS_PATH": "data/price_overlap_diagnostics.csv",
    "INDICES_FILE_PATH": "indices_list.txt",
    "DATA_SOURCE_NAME": "historical input panel, Yahoo Finance, and public exchange APIs",
    "MIN_PRICE_OBSERVATIONS": 250,
    "MIN_RETURN_OBSERVATIONS": 36,
    "MIN_PRICE_OVERLAP_OBSERVATIONS": 60,
    "MAX_PRICE_OVERLAP_MEDIAN_ABS_PCT_DIFF": 0.02,
    "DOWNLOAD_TIMEOUT_SECONDS": 30,
    "DOWNLOAD_RETRIES": 3,
    "DOWNLOAD_RETRY_BACKOFF_SECONDS": 2,
    "DOWNLOAD_SLEEP_SECONDS": 0.1,
}


# The index file keeps the original Stooq-style symbols. Stooq's single-symbol
# CSV endpoint is now behind browser verification, so use Yahoo's chart API and
# translate symbols where the naming differs.
YAHOO_SYMBOL_OVERRIDES = {
    "^TASI": "^TASI.SR",
    "^XU100": "XU100.IS",
    "^BEL20": "^BFX",
    "^DJC": "^DJA",
    "^CAC": "^FCHI",
    "^NKX": "^N225",
    "^BVP": "^BVSP",
    "^MRV": "^MERV",
    "^OSEAX": "OSEBX.OL",
    "^SOFIX": "SOFIX.BD",
    "^SMI": "^SSMI",
    "^PSEI": "PSEI.PS",
    "^SET": "^SET.BK",
    "^JCI": "^JKSE",
    "^TDXP": "^TECDAX",
    "^SPX": "^GSPC",
    "^NDQ": "^IXIC",
    "^SNX": "^BSESN",
    "^BET": "^BETI",
    "^TOP40": "^J200.JO",
    "^MOEX": "MOEX.ME",
    "^FMIB": "FTSEMIB.MI",
    "^CRY": "^TRCCRB",
    "^MDAX": "^MDAXI",
    "^OMXS": "^OMX",
    "^IPC": "^MXX",
    "^PSI20": "PSI20.LS",
    "^DAX": "^GDAXI",
    "^KOSPI": "^KS11",
    "^ATH": "GD.AT",
    "^TSX": "^GSPTSE",
    "^HEX": "^OMXHPI",
    "^SHC": "000001.SS",
    "^TWSE": "^TWII",
    "^UKX": "^FTSE",
    "^SDXP": "^SDAXI",
    "^FTM": "^FTMC",
    "^KLCI": "^KLSE",
}


PUBLIC_API_OVERRIDES = {
    "^SAX": ("Bratislava Stock Exchange", "SAX"),
    "^PX": ("Prague Stock Exchange", "PX"),
    "^ICEX": ("Nasdaq Global Indexes", "OMXIPI"),
    "^OMXR": ("Nasdaq Global Indexes", "OMXRGI"),
    "^OMXT": ("Nasdaq Global Indexes", "OMXTGI"),
    "^OMXV": ("Nasdaq Global Indexes", "OMXVGI"),
    "^OMXS": ("Nasdaq Global Indexes", "OMXSGI"),
}




def set_custom_style():
    """Applies the custom plotting palette and Matplotlib style."""
    palette = [
        "#00466F",
        "#F38C10",
        "#3297DB",
        "#037E73",
        "#C62828",
        "#FEBD00",
        "#41B01E",
        "#E84C3D",
        "#3D3D3D",
    ]

    plt.style.use("default")
    plt.rcParams.update(
        {
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "axes.edgecolor": "black",
            "axes.linewidth": 1,
            "axes.grid": False,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "legend.frameon": False,
            "font.size": 10,
            "text.usetex": True,
            "lines.linewidth": 1.5,
            "lines.color": "black",
            "figure.figsize": (6, 4),
            "figure.dpi": 300,
            "axes.prop_cycle": plt.cycler(color=palette),
        }
    )
    return palette

class DataDownloadError(Exception):
    """Raised when a market data response cannot be used."""


def _unix_timestamp(date_like) -> int:
    """Converts a date-like value to UTC Unix seconds for Yahoo's chart API."""
    timestamp = pd.Timestamp(date_like)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    else:
        timestamp = timestamp.tz_convert("UTC")
    return int(timestamp.timestamp())


def _yahoo_symbol(symbol: str) -> str:
    """Returns the Yahoo Finance symbol corresponding to an index-list symbol."""
    return YAHOO_SYMBOL_OVERRIDES.get(symbol, symbol)


def _parse_json_response(response: requests.Response):
    """Parses endpoints that sometimes return JSON and sometimes a JSON string."""
    payload = response.json()
    if isinstance(payload, str):
        import json

        payload = json.loads(payload)
    return payload


def _download_yahoo_series(symbol: str, session: requests.Session) -> pd.Series:
    """Downloads a daily close-price series from Yahoo Finance's chart API."""
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{quote(symbol, safe='')}"
    params = {
        "period1": _unix_timestamp(CONFIG["START_DATE"]),
        "period2": _unix_timestamp(pd.Timestamp.utcnow() + pd.Timedelta(days=1)),
        "interval": "1d",
        "events": "history",
        "includeAdjustedClose": "true",
    }
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept": "application/json",
    }

    last_error = None
    for attempt in range(1, CONFIG["DOWNLOAD_RETRIES"] + 1):
        try:
            response = session.get(
                url,
                params=params,
                headers=headers,
                timeout=CONFIG["DOWNLOAD_TIMEOUT_SECONDS"],
            )
            if response.status_code in {429, 500, 502, 503, 504}:
                raise DataDownloadError(f"HTTP {response.status_code}")
            response.raise_for_status()

            payload = response.json()
            chart = payload.get("chart", {})
            error = chart.get("error")
            if error:
                code = error.get("code", "error")
                description = error.get("description", "unknown error")
                raise DataDownloadError(f"{code}: {description}")

            results = chart.get("result") or []
            if not results:
                raise DataDownloadError("empty chart result")

            result = results[0]
            timestamps = result.get("timestamp") or []
            quotes = result.get("indicators", {}).get("quote") or []
            closes = quotes[0].get("close") if quotes else []
            if not timestamps or not closes:
                raise DataDownloadError("missing timestamps or close prices")

            index = pd.to_datetime(timestamps, unit="s", utc=True)
            timezone_name = result.get("meta", {}).get("exchangeTimezoneName", "UTC")
            try:
                index = index.tz_convert(timezone_name)
            except Exception:
                index = index.tz_convert("UTC")
            index = index.tz_localize(None).normalize()

            series = pd.Series(closes, index=index, name=symbol, dtype="float64").dropna()
            series = series[~series.index.duplicated(keep="last")]
            if series.empty:
                raise DataDownloadError("no non-missing close prices")
            return series.sort_index()
        except (DataDownloadError, requests.RequestException, ValueError) as exc:
            last_error = exc
            if attempt < CONFIG["DOWNLOAD_RETRIES"]:
                time.sleep(CONFIG["DOWNLOAD_RETRY_BACKOFF_SECONDS"] * attempt)

    raise DataDownloadError(str(last_error))


def _download_nasdaq_index_series(symbol: str, session: requests.Session) -> pd.Series:
    """Downloads daily index values from Nasdaq Global Indexes."""
    start = pd.Timestamp(CONFIG["START_DATE"]).strftime("%Y-%m-%dT00:00:00.000")
    end = pd.Timestamp.utcnow().strftime("%Y-%m-%dT00:00:00.000")
    response = session.post(
        "https://indexes.nasdaqomx.com/Index/HistoryChartData",
        data={"id": symbol, "startDate": start, "endDate": end},
        headers={
            "User-Agent": "Mozilla/5.0",
            "Accept": "application/json",
            "Referer": f"https://indexes.nasdaqomx.com/Index/History/{symbol}",
            "X-Requested-With": "XMLHttpRequest",
        },
        timeout=CONFIG["DOWNLOAD_TIMEOUT_SECONDS"],
    )
    response.raise_for_status()
    payload = _parse_json_response(response)
    if not isinstance(payload, list) or not payload:
        raise DataDownloadError("empty Nasdaq history")

    index = pd.to_datetime([row["x"] for row in payload], unit="ms", utc=True)
    index = index.tz_localize(None).normalize()
    series = pd.Series([row.get("y") for row in payload], index=index, name=symbol, dtype="float64")
    series = series.dropna()
    if series.empty:
        raise DataDownloadError("no non-missing Nasdaq index values")
    return series[series.index >= pd.Timestamp(CONFIG["START_DATE"])].sort_index()


def _download_prague_index_series(symbol: str, session: requests.Session) -> pd.Series:
    """Downloads daily index values from the Prague Stock Exchange chart API."""
    response = session.get(
        f"https://www.pse.cz/api/index-chart?indexName={quote(symbol, safe='')}&range=max",
        headers={
            "User-Agent": "Mozilla/5.0",
            "Accept": "application/json",
            "Content-Language": "en",
            "X-API-Key": "PSE",
        },
        timeout=CONFIG["DOWNLOAD_TIMEOUT_SECONDS"],
    )
    response.raise_for_status()
    payload = _parse_json_response(response).get("data", {})
    values = payload.get("value") or []
    if not values:
        raise DataDownloadError("empty Prague index history")

    index = pd.to_datetime([row[0] for row in values], unit="ms", utc=True)
    index = index.tz_localize(None).normalize()
    series = pd.Series([row[1] for row in values], index=index, name=symbol, dtype="float64")
    series = series.dropna()
    return series[series.index >= pd.Timestamp(CONFIG["START_DATE"])].sort_index()


def _download_bratislava_sax_series(symbol: str, session: requests.Session) -> pd.Series:
    """Downloads daily SAX values from the Bratislava Stock Exchange API."""
    end = pd.Timestamp.utcnow().strftime("%Y-%m-%d")
    response = session.get(
        "https://www.bsse.sk/BCPB_WEB_API/api/SAX/Get",
        params={
            "choice": "S",
            "from": CONFIG["START_DATE"],
            "to": end,
            "page": "1",
            "pagesize": "10000",
            "rowcount": "10000",
            "lang": "EN",
        },
        headers={
            "User-Agent": "Mozilla/5.0",
            "Accept": "application/json",
            "Referer": "https://www.bsse.sk/bcpb/en/indices/sax-index/",
        },
        timeout=CONFIG["DOWNLOAD_TIMEOUT_SECONDS"],
    )
    response.raise_for_status()
    payload = _parse_json_response(response)
    tables = payload.get("Tables") or []
    rows = tables[0].get("Rows") if tables else []
    if not rows:
        raise DataDownloadError("empty SAX index history")

    dates = [row["Cells"][0] for row in rows]
    values = [str(row["Cells"][2]).replace(",", ".") for row in rows]
    series = pd.Series(
        pd.to_numeric(values, errors="coerce"),
        index=pd.to_datetime(dates).normalize(),
        name=symbol,
    ).dropna()
    return series[series.index >= pd.Timestamp(CONFIG["START_DATE"])].sort_index()


def _download_series(symbol: str, session: requests.Session) -> tuple[pd.Series, str, str]:
    """Downloads an index series using the configured source for its symbol."""
    if symbol in PUBLIC_API_OVERRIDES:
        source, provider_symbol = PUBLIC_API_OVERRIDES[symbol]
        if source == "Nasdaq Global Indexes":
            return _download_nasdaq_index_series(provider_symbol, session), provider_symbol, source
        if source == "Prague Stock Exchange":
            return _download_prague_index_series(provider_symbol, session), provider_symbol, source
        if source == "Bratislava Stock Exchange":
            return _download_bratislava_sax_series(provider_symbol, session), provider_symbol, source
        raise DataDownloadError(f"unsupported source: {source}")

    provider_symbol = _yahoo_symbol(symbol)
    return _download_yahoo_series(provider_symbol, session), provider_symbol, "Yahoo Finance"


def download_data(candidates: pd.DataFrame) -> pd.DataFrame:
    """Downloads time series for each index in the provided DataFrame."""
    print(f"\nDownloading data from {CONFIG['DATA_SOURCE_NAME']}...")
    selected_series = {}
    session = requests.Session()

    for _, row in candidates.iterrows():
        country, symbol = row['country'], row['symbol']
        try:
            series, provider_symbol, source = _download_series(symbol, session)
            series = series[~series.index.duplicated(keep="last")].sort_index()
            if len(series) >= CONFIG["MIN_PRICE_OBSERVATIONS"]:
                selected_series[country] = series.sort_index()
                symbol_note = f"{symbol} -> {provider_symbol}" if symbol != provider_symbol else symbol
                first_date = series.index.min().date()
                print(f"  + Downloaded '{symbol_note}' for {country} ({source}; first date: {first_date})")
            else:
                first_date = series.index.min().date() if not series.empty else "n/a"
                print(
                    f"  - Skipped '{symbol}' "
                    f"(only {len(series)} observations; first date: {first_date})"
                )
        except Exception as exc:
            provider_symbol = PUBLIC_API_OVERRIDES.get(symbol, (None, _yahoo_symbol(symbol)))[1]
            symbol_note = f"{symbol} -> {provider_symbol}" if symbol != provider_symbol else symbol
            print(f"  - Failed to download '{symbol_note}': {exc}")
            continue
        finally:
            time.sleep(CONFIG["DOWNLOAD_SLEEP_SECONDS"])

    if not selected_series:
        raise SystemExit("Could not download any valid time series from the list.")
    
    final_panel = pd.concat(selected_series, axis=1).ffill()
    print(f"\nSuccessfully created a panel with {final_panel.shape[1]} country indices.")
    return final_panel


def load_csv_timeseries(path: str) -> pd.DataFrame | None:
    """Loads an indexed time-series CSV if it is available."""
    if not os.path.exists(path):
        return None
    frame = pd.read_csv(path, index_col=0)
    frame.index = pd.to_datetime(frame.index).normalize()
    frame = frame.apply(pd.to_numeric, errors="coerce")
    frame = frame[~frame.index.duplicated(keep="last")].sort_index()
    return frame.dropna(how="all")


def load_historical_price_panel() -> pd.DataFrame | None:
    """Loads the historical input price panel required for the canonical proxy."""
    panel = load_csv_timeseries(CONFIG["HISTORICAL_PRICE_PANEL_PATH"])
    if panel is None:
        raise SystemExit(
            "Missing historical input price panel: "
            f"{CONFIG['HISTORICAL_PRICE_PANEL_PATH']}. "
            "This script now requires historical country/index price data, "
            "then splices replacement-source prices and recomputes the GFC proxy from scratch."
        )

    print(
        "\nLoaded historical input price panel "
        f"({panel.shape[1]} indices, {panel.index.min().date()} to {panel.index.max().date()})."
    )
    return panel


def compare_price_overlap(
    historical: pd.Series,
    replacement: pd.Series,
    name: str,
) -> tuple[pd.Series, pd.Series]:
    """Checks whether two price series agree over their overlapping dates."""
    common_index = historical.dropna().index.intersection(replacement.dropna().index)
    if len(common_index) == 0:
        diagnostics = pd.Series(
            {
                "index": name,
                "overlap_observations": 0,
                "overlap_start": pd.NaT,
                "overlap_end": pd.NaT,
                "level_scale": np.nan,
                "level_correlation": np.nan,
                "return_correlation": np.nan,
                "median_abs_pct_diff_after_scaling": np.nan,
                "max_abs_pct_diff_after_scaling": np.nan,
                "status": "fail_no_overlap",
            }
        )
        return replacement, diagnostics

    hist_overlap = historical.loc[common_index].astype(float)
    new_overlap = replacement.loc[common_index].astype(float)
    ratio = (hist_overlap / new_overlap).replace([np.inf, -np.inf], np.nan).dropna()
    scale = ratio.median() if not ratio.empty else 1.0
    scaled_replacement = replacement * scale
    scaled_overlap = scaled_replacement.loc[common_index]

    rel_diff = ((scaled_overlap - hist_overlap) / hist_overlap).replace([np.inf, -np.inf], np.nan)
    hist_returns = np.log(hist_overlap / hist_overlap.shift(1)).replace([np.inf, -np.inf], np.nan)
    new_returns = np.log(new_overlap / new_overlap.shift(1)).replace([np.inf, -np.inf], np.nan)

    median_abs_pct_diff = rel_diff.abs().median()
    return_correlation = hist_returns.corr(new_returns)
    level_correlation = hist_overlap.corr(new_overlap)
    has_enough_overlap = len(common_index) >= CONFIG["MIN_PRICE_OVERLAP_OBSERVATIONS"]
    has_close_levels = median_abs_pct_diff <= CONFIG["MAX_PRICE_OVERLAP_MEDIAN_ABS_PCT_DIFF"]
    status = "pass" if has_enough_overlap and has_close_levels else "review"

    diagnostics = pd.Series(
        {
            "index": name,
            "overlap_observations": len(common_index),
            "overlap_start": common_index.min().date(),
            "overlap_end": common_index.max().date(),
            "level_scale": scale,
            "level_correlation": level_correlation,
            "return_correlation": return_correlation,
            "median_abs_pct_diff_after_scaling": median_abs_pct_diff,
            "max_abs_pct_diff_after_scaling": rel_diff.abs().max(),
            "status": status,
        }
    )
    return scaled_replacement, diagnostics


def merge_price_panels(
    historical_panel: pd.DataFrame,
    replacement_panel: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Splices historical input prices with replacement-source prices."""
    merged_series = {}
    diagnostics = []
    all_columns = historical_panel.columns

    for column in all_columns:
        historical = historical_panel[column].dropna()
        replacement = replacement_panel[column].dropna() if column in replacement_panel else pd.Series(dtype=float)

        if replacement.empty:
            merged_series[column] = historical
            diagnostics.append(
                pd.Series(
                    {
                        "index": column,
                        "overlap_observations": 0,
                        "status": "historical_only",
                    }
                )
            )
            continue

        scaled_replacement, diagnostic = compare_price_overlap(historical, replacement, column)
        append_part = scaled_replacement[scaled_replacement.index > historical.index.max()]
        merged_series[column] = pd.concat([historical, append_part]).sort_index()
        diagnostics.append(diagnostic)

    merged_panel = pd.concat(merged_series, axis=1).ffill()
    diagnostics_frame = pd.DataFrame(diagnostics).set_index("index").sort_index()
    return merged_panel, diagnostics_frame


def prepare_returns_for_pca(returns_panel: pd.DataFrame) -> pd.DataFrame:
    """Prepares a balanced return panel while preserving raw return scaling."""
    panel = returns_panel.replace([np.inf, -np.inf], np.nan).dropna()
    if panel.empty:
        raise SystemExit("No complete return panel is available for PCA.")
    return panel


def calculate_gfc_proxy(returns_panel: pd.DataFrame) -> tuple[pd.Series, float, PCA]:
    """Calculates the GFC proxy (PC1) from a balanced panel of returns."""
    returns_panel = prepare_returns_for_pca(returns_panel)
    pca = PCA(n_components=1)
    gfc_raw = pca.fit_transform(returns_panel)
    gfc_z = pd.Series(gfc_raw.flatten(), index=returns_panel.index)
    gfc_z = (gfc_z - gfc_z.mean()) / gfc_z.std()
    variance_explained = pca.explained_variance_ratio_[0]
    return gfc_z, variance_explained, pca 


def calculate_proxy_outputs(price_panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, float, float]:
    """Calculates standardized daily and monthly cumulative GFC proxy outputs."""
    print("\nCalculating MONTHLY GFC proxy...")
    monthly_prices = price_panel.resample('ME').last()
    current_month_end = pd.Timestamp.today().normalize() + pd.offsets.MonthEnd(0)
    if pd.Timestamp.today().normalize() < current_month_end:
        monthly_prices = monthly_prices[monthly_prices.index < current_month_end]

    returns_monthly = (
        monthly_prices
        .pipe(lambda df: np.log(df / df.shift(1)))
        .dropna()
    )
    gfc_z_monthly, var_exp_monthly, _ = calculate_gfc_proxy(returns_monthly)
    print(f"Monthly proxy calculated. Variance Explained: {var_exp_monthly:.2%}")

    print("\nCalculating DAILY GFC proxy...")
    daily_returns = np.log(price_panel / price_panel.shift(1)).dropna()
    gfc_z_daily, var_exp_daily, _ = calculate_gfc_proxy(daily_returns)
    print(f"Daily proxy calculated. Variance Explained: {var_exp_daily:.2%}")

    monthly_proxy = pd.DataFrame(
        StandardScaler().fit_transform(gfc_z_monthly.cumsum().values.reshape(-1, 1)),
        index=gfc_z_monthly.index,
        columns=["GFC Proxy-Monthly"],
    )
    daily_proxy = pd.DataFrame(
        StandardScaler().fit_transform(gfc_z_daily.cumsum().values.reshape(-1, 1)),
        index=gfc_z_daily.index,
        columns=["GFC Proxy-Daily"],
    )
    return daily_proxy, monthly_proxy, var_exp_daily, var_exp_monthly


def analyze_plot_and_save(
    gfc_proxy_daily: pd.DataFrame,
    gfc_proxy_monthly: pd.DataFrame,
    source_note: str,
) -> None:
    """Fetches the GFC, analyzes fit, plots, and saves all outputs."""
    print("\nComparing proxies with the Miranda-Agrippino & Rey (2020) GFC factor...")
    
    # 1. Fetch and process the benchmark GFC data
    gfc_mar = pd.read_excel(CONFIG["OFFICIAL_GFC_URL"], sheet_name="Standardized", skiprows=2, index_col=1, engine='openpyxl')
    gfc_mar = gfc_mar.iloc[:, [2]].dropna()
    gfc_mar.columns = ["GFC M&R (2020)-Monthly"]
    gfc_mar.index = pd.to_datetime(gfc_mar.index, format="%Y-%m") + pd.offsets.MonthEnd(0)

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

    # 6. Create plot with the custom publication style
    palette = set_custom_style()
    fig, ax = plt.subplots() 
    ax.plot(gfc_proxy_daily.index, gfc_proxy_daily['GFC Proxy-Daily'],
            label='GFC Proxy (Daily)', color=palette[2], linewidth=1.0, marker=None, zorder=1) 

    ax.plot(monthly_comparison_unaligned.index, monthly_comparison_unaligned['GFC Proxy-Monthly'],
            label=f'GFC Proxy (Monthly) | Corr: {correlation:.2f}', color=palette[0], 
            linewidth=1.8, marker=None, zorder=3)
    
    ax.plot(monthly_comparison.index, monthly_comparison['GFC M&R (2020)-Monthly'],
            label='GFC M\\&R (2020)', color=palette[1],
            linewidth=1.8, marker=None, zorder=2)
    
    # Formatting the plot
    ax.set_title("Global Financial Cycle Proxies vs. Miranda-Agrippino \\& Rey Factor", pad=10)
    ax.set_ylabel("Standardized value (cumulative)")
    ax.set_xlabel(None)
    ax.legend(loc='upper right')
    
    ax.xaxis.set_major_locator(mdates.YearLocator(2)) 
    ax.xaxis.set_minor_locator(mdates.YearLocator())  
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    plt.setp(ax.get_xticklabels(), rotation=0, ha='center')
    ax.tick_params(axis='x', which='minor', bottom=True)
    ax.axhline(0, color=palette[8], linestyle='--', linewidth=0.8)

    note = (
        f"Source: Author's calculations using {source_note}. "
        "GFC Factor from Miranda-Agrippino \\& Rey (2020).\n"
        "Methodology based on Habib \\& Venditti (2019).\n"
        "Note: Proxies are the first principal component (PC1) of returns from a panel of global stock indices. "
        "The cumulative series are standardized for comparison.\n"
        "Data available at https://github.com/iweigandi/daily-global-financial-cycle-proxy"
        )
    fig.text(0.08, 0.015, note, ha='left', va='bottom', fontsize=5, color=palette[8], wrap=True)
    plt.subplots_adjust(left=0.13, right=0.97, top=0.9, bottom=0.28)

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
        
    historical_price_panel = load_historical_price_panel()

    historical_indices = set(historical_price_panel.columns)
    missing_from_archive = candidates.loc[
        ~candidates["country"].isin(historical_indices),
        ["symbol", "country"],
    ]
    if not missing_from_archive.empty:
        print(
            "\nSkipping indices absent from the historical input price panel: "
            + ", ".join(missing_from_archive["country"].tolist())
        )
    candidates = candidates[candidates["country"].isin(historical_indices)].copy()
        
    replacement_panel = download_data(candidates)

    print("\nMerging historical input prices with replacement-source prices...")
    price_panel, price_diagnostics = merge_price_panels(
        historical_price_panel,
        replacement_panel,
    )
    price_diagnostics.to_csv(CONFIG["PRICE_OVERLAP_DIAGNOSTICS_PATH"])
    print(f"Price overlap diagnostics saved to: {CONFIG['PRICE_OVERLAP_DIAGNOSTICS_PATH']}")
    print(
        "Price overlap status counts: "
        f"{price_diagnostics['status'].value_counts().to_dict()}"
    )

    gfc_proxy_daily, gfc_proxy_monthly, _, _ = calculate_proxy_outputs(price_panel)
    source_note = (
        "a historical input panel (archived Stooq prices plus documented Yahoo backfills) "
        "spliced with Yahoo Finance and public exchange APIs"
    )
    analyze_plot_and_save(gfc_proxy_daily, gfc_proxy_monthly, source_note)

if __name__ == "__main__":
    main()


