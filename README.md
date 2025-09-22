# Real-Time Global Financial Cycle (GFC) Proxy

![Update GFC Proxy Data and Chart](https://github.com/YOUR_USERNAME/YOUR_REPO_NAME/actions/workflows/run_gfc_proxy.yml/badge.svg)

This repository contains an open-source project to automatically calculate and update daily and monthly proxies for the Global Financial Cycle. The script uses public stock market data from Stooq and is updated weekly via a GitHub Action.

---

### Live Chart

This chart is automatically regenerated every week. It compares the daily and monthly proxies against a widely-used official GFC factor.

![Live GFC Chart](https://github.com/YOUR_USERNAME/YOUR_REPO_NAME/blob/main/chart/gfc_comparison_plot.png?raw=true)

---

### Live Data

The generated data is also updated weekly and can be accessed directly:
* **Daily GFC Proxy:** [`data/gfc_proxy_daily.csv`](data/gfc_proxy_daily.csv)
* **Monthly Proxies:** [`data/gfc_proxies_monthly.csv`](data/gfc_proxies_monthly.csv)

---

### Methodology

1.  **Index Discovery:** The script scrapes Stooq's "Main Indices" pages to dynamically discover a broad set of national stock market indices.
2.  **Data Download:** It downloads daily historical price data for one representative index per country using the `pandas-datareader` library.
3.  **Proxy Calculation:**
    * **Monthly Proxy:** Calculated from the first principal component (PC1) of monthly log-returns of the indices.
    * **Daily Proxy:** Calculated from the PC1 of a 21-day rolling average of daily log-returns to filter for the cyclical component.
4.  **Validation:** The monthly proxy is statistically compared against an official GFC factor by calculating the correlation and Mean Squared Error.
5.  **Automation:** A GitHub Action runs the entire pipeline every Monday, committing the updated data and chart back to this repository.