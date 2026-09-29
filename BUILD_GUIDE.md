# Build guide (for you, not for recruiters)

Work through these in order. Each step ends with something you can check.

## Week 1: set up and get clean data

**1. Install tools**
- Python 3.11 or 3.12 (python.org). Prophet can be fussy on the newest Python versions.
- VS Code, Git, and a free GitHub account.
- PostgreSQL 16 plus pgAdmin (postgresql.org/download). Remember the password you set.

**2. Set up the project**
```bash
cd flight-delay-intelligence
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements-pipeline.txt
```

**3. Test everything with fake data first (10 minutes)**
```bash
python scripts/make_sample_data.py
python src/clean_data.py
python src/build_aggregates.py
python src/forecast.py
python src/train_model.py --rows 100000
streamlit run app/streamlit_app.py
```
The dashboard should open in your browser. Click every tab. If something breaks, fix it now,
while the data is small. Then delete the fake data:
```bash
# Mac/Linux
rm data/raw/*.csv data/processed/*.parquet
# Windows PowerShell
Remove-Item data\raw\*.csv, data\processed\*.parquet
```

**4. Download the real data**
```bash
python src/download_data.py
```
This takes a while (about 24 files). If the BTS server refuses the download, get the files by
hand: search "BTS Reporting Carrier On-Time Performance 1987-present download", choose year and
month, tick "Prezipped File", download, unzip, and rename each CSV to `ontime_2024_01.csv` etc.
inside `data/raw/`.

**5. Clean and validate**
```bash
python src/clean_data.py
```
Open `reports/data_quality.csv`. You should see ~500k to 650k flights per month, delay rates
roughly 17% to 25%, and "delay_causes_sum_to_arr_delay_pct" near 100. These checks are
interview material: "I validated that delay cause minutes reconcile to arrival delay."

## Week 2: SQL and dashboards

**6. Load PostgreSQL.** In pgAdmin, create a database called `flights`. Then:
```bash
# Mac/Linux
export DATABASE_URL="postgresql://postgres:YOURPASSWORD@localhost:5432/flights"
# Windows PowerShell
$env:DATABASE_URL="postgresql://postgres:YOURPASSWORD@localhost:5432/flights"
python src/load_to_postgres.py
```

**7. Run the 10 queries** in `sql/03_analysis.sql` one at a time in pgAdmin. For every query,
be able to explain the question, the SQL technique, and the answer. Write down the numbers.
They go into the README "Key findings".

**8. Build summary tables for the app**
```bash
python src/build_aggregates.py
```

**9. Tableau Public version (recommended).** Install Tableau Public (free). Export the summary
table to CSV, since Tableau Public reads CSV and Excel:
```bash
python -c "import pandas as pd; pd.read_parquet('app/data/monthly.parquet').to_csv('reports/monthly_for_tableau.csv', index=False)"
```
Connect Tableau to that CSV and build a one-page
dashboard: KPI tiles, on-time trend, airline bars, cause breakdown. Publish and copy the link.
Recruiters who use Tableau often click this first.

## Week 3: forecasting, model, deploy

**10. Forecast and model**
```bash
python src/forecast.py
python src/train_model.py            # use --rows 500000 if your laptop struggles
```
Expect ROC-AUC around 0.65 to 0.72. That is correct for booking-time-only features. Be ready
to explain why that is more honest than 0.97 (no leakage, time-based test).

**11. Run the app locally** and check every tab with real data.
```bash
streamlit run app/streamlit_app.py
```

**12. Put it on GitHub**
```bash
git init
git add .
git commit -m "Flight delay intelligence: pipeline, SQL, dashboard, forecast, model"
git branch -M main
git remote add origin https://github.com/YOURUSERNAME/flight-delay-intelligence.git
git push -u origin main
```
The `.gitignore` keeps the multi-GB raw data off GitHub; the small `app/data` tables and the
model are included, which is what the live app needs. If a file is over 100 MB, GitHub will
refuse it, so check `app/data` sizes first.

**13. Deploy free on Streamlit Community Cloud.** Sign in at share.streamlit.io with GitHub,
click "Create app", choose your repo, set the main file path to `app/streamlit_app.py`, deploy.
You'll get a public link.

## Week 4: make it visible

**14. Finish the README.** Replace every [placeholder], add a screenshot at
`reports/dashboard.png`, and paste in your live links. Add `reports/shap_importance.png` too.

**15. Record a 60 to 90 second walkthrough** (Loom is free). Lead with one surprising finding,
show two tabs, end with the predictor.

**16. Post on LinkedIn.** Open with the finding, not the tools. Add the live link and video.
Pin it in your Featured section.

**17. Update your resume.** Put this project first under Projects with the live link, for example:

> **U.S. Flight Delay Intelligence** | Python, PostgreSQL, Tableau, Streamlit, Prophet, XGBoost | [live link]
> - Built an end-to-end pipeline processing 13M+ BTS flight records, with automated data-quality checks
> - Wrote 10 PostgreSQL analyses (CTEs, window functions, percentiles) on airline, route, and hourly reliability
> - Deployed an interactive dashboard with a Prophet forecast (X% better than baseline) and an
>   XGBoost delay-risk model (ROC-AUC X) explained with SHAP

Only use numbers you actually got.

## Stretch ideas (after it's live)
- Join NOAA daily weather to quantify weather's real impact.
- Add a U.S. map of airport delay rates (you'll need an airport-coordinates file).
- Automate monthly refresh with GitHub Actions.
