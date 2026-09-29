"""Which direction of travel is most delayed? Groups airports into the four
U.S. Census regions (from the state in each airport's city name) and compares
delays for flights leaving, arriving in, and travelling between regions.

Usage:
    python src/region_analysis.py
"""
import pandas as pd

REGIONS = {
    "Northeast": "CT ME MA NH RI VT NJ NY PA",
    "Midwest": "IL IN MI OH WI IA KS MN MO NE ND SD",
    "South": "DE DC FL GA MD NC SC VA WV AL KY MS TN AR LA OK TX",
    "West": "AZ CO ID MT NV NM UT WY AK CA HI OR WA",
}
STATE_TO_REGION = {st: region for region, states in REGIONS.items() for st in states.split()}

airports = pd.read_parquet("app/data/airports.parquet")
airports["state"] = airports["city"].str.split(", ").str[-1]
airports["region"] = airports["state"].map(STATE_TO_REGION).fillna("Territories")
region_of = dict(zip(airports["airport"], airports["region"]))

r = pd.read_parquet("app/data/routes.parquet")
r["from_region"] = r["origin"].map(region_of).fillna("Territories")
r["to_region"] = r["dest"].map(region_of).fillna("Territories")
cols = ["flights", "completed", "delayed", "cancelled", "late_minutes"]
order = ["Northeast", "Midwest", "South", "West", "Territories"]

def rates(g):
    g = g[cols].sum().astype(float)
    return pd.DataFrame({
        "flights": g["flights"].astype(int),
        "delayed_%": (100 * g["delayed"] / g["completed"]).round(1),
        "cancelled_%": (100 * g["cancelled"] / g["flights"]).round(2),
        "avg_min_late": (g["late_minutes"] / g["delayed"]).round(0),
    })

print("=== Flights LEAVING each region ===")
print(rates(r.groupby("from_region")).reindex([o for o in order if o in set(r["from_region"])]).to_string())
print("\n=== Flights ARRIVING in each region ===")
print(rates(r.groupby("to_region")).reindex([o for o in order if o in set(r["to_region"])]).to_string())

pair = r.groupby(["from_region", "to_region"])[cols].sum().astype(float)
pair["delayed_%"] = (100 * pair["delayed"] / pair["completed"]).round(1)
matrix = pair["delayed_%"].unstack()
matrix = matrix.reindex(index=[o for o in order if o in matrix.index],
                        columns=[o for o in order if o in matrix.columns])
print("\n=== Delayed % by direction (rows = FROM, columns = TO) ===")
print(matrix.to_string())

big = pair[pair["flights"] >= 10000].sort_values("delayed_%", ascending=False)
print("\n=== Region pairs ranked, worst first (10,000+ flights) ===")
print(big[["flights", "delayed_%"]].astype({"flights": int}).to_string())
