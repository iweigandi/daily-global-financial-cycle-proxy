# Real-Time Global Financial Cycle (GFC) Proxy

![Update GFC Proxy Data and Chart](https://github.com/iweigandi/daily-global-financial-cycle-proxy/actions/workflows/run_gfc_proxy.yml/badge.svg)

This repository contains an open-source project to automatically calculate and update daily and monthly proxies for the Global Financial Cycle. The script starts from a historical input price panel, built from archived Stooq data plus documented Yahoo Finance backfills for the three Dow Jones indices absent from the archive, splices it with replacement data from Yahoo Finance and public exchange APIs, and recomputes the proxies from the merged country/index price panel. It is updated weekly via a GitHub Action.

---

### Live Chart

This chart is automatically regenerated every week. It compares the daily and monthly proxies against a the GFC factor of Miranda-Agrippino
& Rey (2020).

![Live GFC Chart](https://github.com/iweigandi/daily-global-financial-cycle-proxy/blob/main/chart/gfc_comparison_plot.png?raw=true)

---

### Live Data

The generated data is also updated weekly and can be accessed directly:
* **Daily GFC Proxy:** [`data/gfc_proxy_daily.csv`](data/gfc_proxy_daily.csv)
* **Monthly GFC Proxies:** [`data/gfc_proxies_monthly.csv`](data/gfc_proxies_monthly.csv)

---

### Archived Price Panel

The replication script requires `data/stooq_price_panel.csv`, built from a locally archived Stooq historical database download plus documented Yahoo Finance backfills for `DOW JONES COMP`, `DOW JONES UTIL`, and `DOW JONES TRANS`. That raw input panel is treated as private data and is ignored by git. Do not commit it to a public repository unless the data source licenses/terms explicitly allow redistribution.

---

### Methodology


1.  **Data Update:** It requires the historical input price panel at `data/stooq_price_panel.csv`, downloads replacement daily price data for one representative index per country using Yahoo Finance's chart API and public exchange APIs, checks overlap quality by index, splices the price series, and recomputes the proxies from the merged 57-index price panel using balanced return matrices for PCA.
2.  **Proxy Calculation:**
    * **Monthly Proxy:** Calculated from the first principal component (PC1) of a balanced panel of monthly log-returns of the indices.
    * **Daily Proxy:** Calculated from the PC1 of a balanced panel of unsmoothed daily log-returns of the indices.
3.  **Validation:** The monthly proxy is statistically compared against the GFC factor of Miranda-Agrippino
& Rey (2020).
4.  **Diagnostics:** Price overlap diagnostics are written to `data/price_overlap_diagnostics.csv`.
5.  **Automation:** A GitHub Action runs the entire pipeline every Monday, committing the updated data and chart back to this repository.
6.  **Details:** For a detailed explanation of the data and methods, please see the [Methodological Note](Methodology.pdf).
