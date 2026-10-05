# Real-Time Global Financial Cycle (GFC) Proxy

![Update GFC Proxy Data and Chart](https://github.com/iweigandi/daily-global-financial-cycle-proxy/actions/workflows/run_gfc_proxy.yml/badge.svg)

This repository provides daily and monthly proxies for the Global Financial Cycle (GFC), constructed from a panel of global equity market indices following the approach of Habib and Venditti (2019). The proxies are updated regularly using Stooq and Yahoo Finance price data and are compared with the GFC factor of Miranda-Agrippino and Rey (2020).

---

### Live Chart

The chart compares the daily and monthly GFC proxies with the Miranda-Agrippino and Rey (2020) factor.

![Live GFC Chart](https://github.com/iweigandi/daily-global-financial-cycle-proxy/blob/main/chart/gfc_comparison_plot.png?raw=true)

---

### Live Data

The generated data can be accessed directly:
* **Daily GFC Proxy:** [`data/gfc_proxy_daily.csv`](data/gfc_proxy_daily.csv)
* **Monthly GFC Proxies:** [`data/gfc_proxies_monthly.csv`](data/gfc_proxies_monthly.csv)
* **Price Overlap Diagnostics:** [`data/price_overlap_diagnostics.csv`](data/price_overlap_diagnostics.csv)

---

### Methodology

1. **Data:** The proxy is based on a 57-index panel of daily equity index prices covering advanced and emerging markets.
2. **Returns:** Daily and monthly log returns are computed from closing prices. PCA is estimated on balanced return matrices.
3. **Proxy Calculation:** The daily and monthly proxies are defined as the first principal component (PC1) of the corresponding return panels. The cumulative factors are standardized for comparison.
4. **Validation:** The monthly proxy is compared with the Miranda-Agrippino and Rey (2020) GFC factor.
5. **Details:** For a fuller explanation of the data and methods, see the [Methodological Note](Methodology.pdf).

---

### Replication Note

The public repository contains the derived proxy series, chart, methodology note, and replication code. Raw historical price inputs are managed separately so that the public repository distributes only derived outputs.