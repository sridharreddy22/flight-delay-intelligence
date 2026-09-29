"""U.S. Flight Delay Intelligence - live dashboard.

Run locally:   streamlit run app/streamlit_app.py
"""
import json
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

APP_DIR = Path(__file__).parent
DATA_DIR = APP_DIR / "data"
MODEL_DIR = APP_DIR / "models"
HOME_AIRPORT = "STL"

# Palette taken from FAA sectional charts: chart navy, Class B blue, airspace magenta, terrain tan.
NAVY, BLUE, MAGENTA, TAN, MUTED, GRID = "#16304A", "#2F64A8", "#A8326E", "#C9A66B", "#9AA8B6", "#D5DEE6"
CAUSES = {
    "late_aircraft_delay": ("Late-arriving aircraft", MAGENTA),
    "carrier_delay": ("Airline (crew, maintenance)", NAVY),
    "nas_delay": ("Air traffic system", BLUE),
    "weather_delay": ("Extreme weather", TAN),
    "security_delay": ("Security", MUTED),
}
MONTH_NAMES = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
DAY_NAMES = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
FEATURE_LABELS = {"month": "Month", "day_of_week": "Day of week", "dep_hour": "Departure time",
                  "airline": "Airline", "origin": "Origin airport", "dest": "Destination",
                  "distance": "Distance"}

st.set_page_config(page_title="U.S. Flight Delay Intelligence", page_icon="✈️", layout="wide")


# ---------- data ----------
@st.cache_data
def load_table(name: str) -> pd.DataFrame:
    return pd.read_parquet(DATA_DIR / f"{name}.parquet")


@st.cache_data
def load_json(path: Path) -> dict:
    return json.loads(path.read_text()) if path.exists() else {}


@st.cache_resource
def load_model():
    model_path = MODEL_DIR / "delay_model.json"
    if not model_path.exists():
        return None, None
    import xgboost as xgb
    booster = xgb.Booster()
    booster.load_model(model_path)
    return booster, json.loads((MODEL_DIR / "categories.json").read_text())


def with_rates(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["delay_rate"] = df["delayed"] / df["completed"].replace(0, np.nan)
    df["on_time_pct"] = 100 * (1 - df["delay_rate"])
    df["cancel_pct"] = 100 * df["cancelled"] / df["flights"].replace(0, np.nan)
    df["avg_min_late"] = df["late_minutes"] / df["delayed"].replace(0, np.nan)
    if "departed" in df:
        df["dep_on_time_pct"] = 100 * (1 - df["dep_delayed"] / df["departed"].replace(0, np.nan))
        df["avg_dep_late_min"] = df["dep_late_minutes"] / df["dep_delayed"].replace(0, np.nan)
    return df


def totals(df: pd.DataFrame) -> pd.Series:
    return with_rates(df[["flights", "completed", "delayed", "cancelled", "late_minutes",
                          "departed", "dep_delayed", "dep_late_minutes"]]
                      .sum().to_frame().T).iloc[0]


def style(fig: go.Figure, height: int = 380) -> go.Figure:
    fig.update_layout(
        height=height, margin=dict(l=10, r=10, t=30, b=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=NAVY, size=13), hoverlabel=dict(bgcolor="white"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, title=None),
    )
    fig.update_xaxes(gridcolor=GRID, zeroline=False)
    fig.update_yaxes(gridcolor=GRID, zeroline=False)
    return fig


monthly = load_table("monthly")
airlines = load_table("airlines")
airports = load_table("airports")
name_of = dict(zip(airlines["airline"], airlines["airline_name"]))
city_of = dict(zip(airports["airport"], airports["city"]))


# ---------- sidebar filters ----------
st.sidebar.header("Filters")
years = sorted(monthly["year"].unique())
sel_years = st.sidebar.multiselect("Years", years, default=years)

airport_codes = ["ALL"] + airports["airport"].tolist()
sel_airport = st.sidebar.selectbox(
    "Departure airport", airport_codes,
    index=airport_codes.index(HOME_AIRPORT) if HOME_AIRPORT in airport_codes else 0,
    format_func=lambda c: "All U.S. airports" if c == "ALL" else f"{c} – {city_of.get(c, '')}",
)
sel_airlines = st.sidebar.multiselect(
    "Airlines (leave empty for all)", airlines["airline"].tolist(), format_func=name_of.get)
st.sidebar.caption("Source: U.S. Bureau of Transportation Statistics, on-time performance data.")


def apply_filters(df: pd.DataFrame, airport_col: str = "origin") -> pd.DataFrame:
    out = df[df["year"].isin(sel_years)] if "year" in df else df
    if sel_airport != "ALL":
        out = out[out[airport_col] == sel_airport]
    if sel_airlines and "airline" in out:
        out = out[out["airline"].isin(sel_airlines)]
    return out


f_monthly = apply_filters(monthly)
place = "all U.S. airports" if sel_airport == "ALL" else f"{sel_airport} ({city_of.get(sel_airport, '')})"

st.title("Where U.S. flights run late, and why")
st.markdown(f"Departures from **{place}**, {', '.join(map(str, sel_years)) or 'no years selected'}.")

if f_monthly.empty:
    st.info("No flights match these filters. Add a year or choose another airport in the sidebar.")
    st.stop()

# ---------- KPI row ----------
t = totals(f_monthly)
national = totals(monthly[monthly["year"].isin(sel_years)])
compare = sel_airport != "ALL" or bool(sel_airlines)


def vs_us(value, national_value, unit="pts", digits=1):
    return f"{value - national_value:+.{digits}f} {unit} vs U.S." if compare else None


t_dep_late = 100 - t.dep_on_time_pct
us_dep_late = 100 - national.dep_on_time_pct
k1, k2, k3, k4, k5, k6 = st.columns(6)
k1.metric("Scheduled flights", f"{t.flights:,.0f}")
k2.metric("Arrived on time", f"{t.on_time_pct:.1f}%",
          vs_us(t.on_time_pct, national.on_time_pct),
          help="Arrived less than 15 minutes after the scheduled time (the BTS standard).")
k3.metric("Departed late", f"{t_dep_late:.1f}%",
          vs_us(t_dep_late, us_dep_late), delta_color="inverse",
          help="Left the gate 15 or more minutes after the scheduled time.")
k4.metric("Cancelled", f"{t.cancel_pct:.2f}%",
          vs_us(t.cancel_pct, national.cancel_pct, digits=2), delta_color="inverse")
k5.metric("Avg arrival delay", f"{t.avg_min_late:.0f} min",
          vs_us(t.avg_min_late, national.avg_min_late, "min", 0), delta_color="inverse",
          help="Average minutes late, counting only flights that arrived 15+ minutes late.")
k6.metric("Avg departure delay", f"{t.avg_dep_late_min:.0f} min",
          vs_us(t.avg_dep_late_min, national.avg_dep_late_min, "min", 0), delta_color="inverse",
          help="Average minutes late, counting only flights that left 15+ minutes late.")

tabs = st.tabs(["Trends", "Airlines", "Delay causes", "Time of day", "Routes",
                "Forecast", "Will my flight be late?", "About"])

# ---------- Trends ----------
with tabs[0]:
    m = with_rates(f_monthly.groupby(["year", "month"], as_index=False).sum(numeric_only=True))
    m["month_name"] = m["month"].map(lambda i: MONTH_NAMES[i - 1])
    m["year"] = m["year"].astype(str)
    colors = [MUTED, BLUE, NAVY, MAGENTA][-len(sel_years):]
    has_dep = "dep_on_time_pct" in m
    # Same y-axis scale on both charts so arrivals and departures are easy to compare
    both = pd.concat([m["on_time_pct"], m["dep_on_time_pct"]]) if has_dep else m["on_time_pct"]
    y_range = [max(0, both.min() - 3), min(100, both.max() + 3)]

    def trend_chart(col, label):
        fig = px.line(m, x="month_name", y=col, color="year", markers=True,
                      color_discrete_sequence=colors,
                      category_orders={"month_name": MONTH_NAMES},
                      labels={col: label, "month_name": ""})
        fig.update_traces(hovertemplate="%{fullData.name} %{x}: %{y:.1f}%<extra></extra>")
        fig.update_yaxes(range=y_range)
        return style(fig, 340)

    st.subheader("On-time arrivals")
    st.plotly_chart(trend_chart("on_time_pct", "Arrived on time (%)"), width="stretch")
    worst = m.loc[m["on_time_pct"].idxmin()]
    st.caption(f"Worst month for arrivals: {worst.month_name} {worst.year}, "
               f"with {worst.on_time_pct:.1f}% of flights arriving on time.")

    if has_dep:
        st.subheader("On-time departures")
        st.plotly_chart(trend_chart("dep_on_time_pct", "Departed on time (%)"), width="stretch")
        worst_dep = m.loc[m["dep_on_time_pct"].idxmin()]
        st.caption(f"Worst month for departures: {worst_dep.month_name} {worst_dep.year}, "
                   f"with {worst_dep.dep_on_time_pct:.1f}% of flights leaving on time.")

# ---------- Airlines ----------
with tabs[1]:
    a = with_rates(f_monthly.groupby("airline", as_index=False).sum(numeric_only=True))
    a = a[a["completed"] >= 100].assign(name=lambda d: d["airline"].map(name_of))
    a = a.sort_values("on_time_pct")
    fig = go.Figure(go.Bar(
        x=a["on_time_pct"], y=a["name"], orientation="h", marker_color=BLUE,
        text=a["on_time_pct"].map("{:.1f}%".format), textposition="outside",
        hovertemplate="%{y}: %{x:.1f}% on time<extra></extra>"))
    fig.update_xaxes(title="On-time arrivals (%)",
                     range=[max(0, a["on_time_pct"].min() - 10), 100])
    st.plotly_chart(style(fig, 60 + 32 * len(a)), width="stretch")
    st.dataframe(
        a.sort_values("on_time_pct", ascending=False)[
            ["name", "flights", "on_time_pct", "cancel_pct", "avg_min_late"]],
        hide_index=True, width="stretch",
        column_config={
            "name": "Airline",
            "flights": st.column_config.NumberColumn("Flights", format="%d"),
            "on_time_pct": st.column_config.NumberColumn("On time", format="%.1f%%"),
            "cancel_pct": st.column_config.NumberColumn("Cancelled", format="%.2f%%"),
            "avg_min_late": st.column_config.NumberColumn("Avg minutes late (when late)", format="%.0f"),
        })

# ---------- Delay causes ----------
with tabs[2]:
    cause_cols = list(CAUSES)
    c = f_monthly.groupby("airline", as_index=False)[cause_cols].sum()
    c["total"] = c[cause_cols].sum(axis=1)
    c = c[c["total"] > 0].assign(name=lambda d: d["airline"].map(name_of))
    overall = c[cause_cols].sum() / c["total"].sum()
    top = overall.idxmax()
    st.markdown(f"**{CAUSES[top][0]}** accounts for **{overall[top]:.0%}** of all delay minutes "
                f"in this selection. A late plane in the morning pushes delays onto every flight "
                f"it operates later that day.")
    long = c.melt(id_vars=["name", "total"], value_vars=cause_cols, var_name="cause", value_name="minutes")
    long["share"] = 100 * long["minutes"] / long["total"]
    long["cause_label"] = long["cause"].map(lambda k: CAUSES[k][0])
    fig = px.bar(long, x="share", y="name", color="cause_label", orientation="h",
                 color_discrete_map={v[0]: v[1] for v in CAUSES.values()},
                 category_orders={"cause_label": [v[0] for v in CAUSES.values()]},
                 labels={"share": "Share of delay minutes (%)", "name": "", "cause_label": ""})
    fig.update_traces(hovertemplate="%{y}: %{x:.0f}%<extra></extra>")
    st.plotly_chart(style(fig, 90 + 32 * len(c)), width="stretch")

# ---------- Time of day ----------
with tabs[3]:
    h = apply_filters(load_table("hourly"))
    h = with_rates(h.groupby("dep_hour", as_index=False).sum(numeric_only=True))
    h = h[h["completed"] >= 30]
    if h.empty:
        st.info("Not enough flights in this selection to show a time-of-day pattern.")
    else:
        best, worst = h.loc[h["delay_rate"].idxmin()], h.loc[h["delay_rate"].idxmax()]
        st.markdown(f"Flights leaving around **{int(best.dep_hour):02d}:00** are late "
                    f"{best.delay_rate:.0%} of the time; around **{int(worst.dep_hour):02d}:00** "
                    f"it's {worst.delay_rate:.0%}.")
        fig = go.Figure()
        # Bars on the base axis, line on the overlaying axis: Plotly draws the overlay on top.
        fig.add_bar(x=h["dep_hour"], y=h["flights"], name="Flights (right axis)",
                    marker_color=GRID, opacity=0.8,
                    hovertemplate="%{x}:00, %{y:,} flights<extra></extra>")
        fig.add_scatter(x=h["dep_hour"], y=100 * h["delay_rate"], name="Delay rate (%)",
                        yaxis="y2", mode="lines+markers",
                        line=dict(color=MAGENTA, width=3),
                        marker=dict(size=9, color=MAGENTA, line=dict(color="white", width=1.5)),
                        hovertemplate="%{x}:00, %{y:.1f}% delayed<extra></extra>")
        fig.update_layout(
            yaxis=dict(title="Flights", side="right", showgrid=False),
            yaxis2=dict(title="Delayed 15+ min (%)", overlaying="y", side="left",
                        rangemode="tozero", showgrid=True),
            xaxis=dict(title="Scheduled departure hour", dtick=2))
        st.plotly_chart(style(fig), width="stretch")

# ---------- Routes ----------
with tabs[4]:
    r = apply_filters(load_table("routes"))
    r = with_rates(r.groupby(["origin", "dest"], as_index=False).sum(numeric_only=True))
    default_min = 50 if sel_airport != "ALL" else 300
    min_flights = st.slider("Only routes with at least this many flights", 10, 2000, default_min, step=10)
    r = r[r["completed"] >= min_flights]
    if r.empty:
        st.info("No routes meet that minimum. Lower the slider.")
    else:
        r["route"] = r["origin"] + " to " + r["dest"]
        worst = r.nlargest(15, "delay_rate").sort_values("delay_rate")
        fig = go.Figure(go.Bar(x=100 * worst["delay_rate"], y=worst["route"], orientation="h",
                               marker_color=NAVY,
                               customdata=worst[["completed", "avg_min_late"]],
                               hovertemplate="%{y}: %{x:.1f}% delayed, %{customdata[0]:,} flights, "
                                             "%{customdata[1]:.0f} min late on average<extra></extra>"))
        fig.update_xaxes(title="Delayed 15+ min (%)")
        st.subheader("Most-delayed routes")
        st.plotly_chart(style(fig, 90 + 30 * len(worst)), width="stretch")

# ---------- Forecast ----------
with tabs[5]:
    fc_path = DATA_DIR / "forecast.parquet"
    if not fc_path.exists():
        st.info("Run `python src/forecast.py` to add the forecast.")
    else:
        fc = pd.read_parquet(fc_path)
        metrics = load_json(DATA_DIR / "forecast_metrics.json")
        c1, c2 = st.columns([1, 2])
        series = c1.radio("Series", fc["series"].unique().tolist(), horizontal=True)
        window = c2.radio("Show", ["Last 6 months + forecast", "Last year + forecast", "Everything"],
                          horizontal=True)
        s = fc[fc["series"] == series].sort_values("ds").copy()
        s["actual_7d"] = s["actual"].rolling(7, min_periods=3).mean()
        last_actual = s.loc[s["actual"].notna(), "ds"].max()
        past, future = s[s["ds"] <= last_actual], s[s["ds"] > last_actual]
        lookback = {"Last 6 months + forecast": 182, "Last year + forecast": 365}.get(window)
        x_start = last_actual - pd.Timedelta(days=lookback) if lookback else s["ds"].min()

        fig = go.Figure()
        fig.add_scatter(x=past["ds"], y=100 * past["actual_7d"], name="Actual (7-day average)",
                        line=dict(color=NAVY, width=2),
                        hovertemplate="%{x|%b %d, %Y}: %{y:.1f}% delayed<extra>Actual</extra>")
        fig.add_scatter(x=future["ds"], y=100 * future["yhat_upper"], line=dict(width=0),
                        showlegend=False, hoverinfo="skip")
        fig.add_scatter(x=future["ds"], y=100 * future["yhat_lower"], line=dict(width=0),
                        fill="tonexty", fillcolor="rgba(168,50,110,0.15)",
                        name="80% prediction interval", hoverinfo="skip")
        fig.add_scatter(x=future["ds"], y=100 * future["yhat"], name="Forecast",
                        line=dict(color=MAGENTA, width=3),
                        hovertemplate="%{x|%b %d, %Y}: %{y:.1f}% expected<extra>Forecast</extra>")
        fig.add_shape(type="line", x0=last_actual, x1=last_actual, y0=0, y1=1, yref="paper",
                      line=dict(color=MUTED, width=1.5, dash="dot"))
        fig.add_annotation(x=last_actual, y=1, yref="paper", text=" Forecast starts",
                           showarrow=False, xanchor="left", yanchor="top",
                           font=dict(color=MUTED, size=12))
        fig.update_yaxes(title="Delayed 15+ min (%)")
        fig.update_xaxes(range=[x_start, s["ds"].max()],
                         rangeslider=dict(visible=True, thickness=0.1))
        st.plotly_chart(style(fig, 500), width="stretch")
        st.caption("Drag the handles on the slider under the chart to zoom into any period.")
        if series in metrics:
            mm = metrics[series]
            verdict = (f"Prophet was {mm['improvement_pct']}% more accurate than the baseline."
                       if mm["improvement_pct"] >= 0 else
                       f"The simple baseline was {abs(mm['improvement_pct'])}% more accurate over this test period.")
            st.caption(f"Tested on the last {mm['holdout_days']} days of data before refitting: "
                       f"Prophet was off by {mm['prophet_mae_pts']} percentage points on average, "
                       f"versus {mm['baseline_mae_pts']} for a simple recent-average baseline. {verdict}")

# ---------- Predictor ----------
with tabs[6]:
    booster, cats = load_model()
    model_metrics = load_json(DATA_DIR / "model_metrics.json")
    if booster is None:
        st.info("Run `python src/train_model.py` to enable predictions.")
    else:
        routes = load_table("route_distance")
        st.markdown("Estimate the chance a flight arrives 15+ minutes late, using only what you "
                    "know when booking. The model can't see weather, so treat this as a baseline risk.")
        c1, c2 = st.columns(2)
        c3 = st.container()  # airline gets its own full-width row so long names fit
        origins = [a for a in cats["origin"] if a in set(routes["origin"])]
        p_origin = c1.selectbox("From", origins,
                                index=origins.index(HOME_AIRPORT) if HOME_AIRPORT in origins else 0,
                                format_func=lambda x: f"{x} – {city_of.get(x, '')}")
        dests = sorted(set(routes.loc[routes["origin"] == p_origin, "dest"]) & set(cats["dest"]))
        p_dest = c2.selectbox("To", dests, format_func=lambda x: f"{x} – {city_of.get(x, '')}")
        # Every airline flying from the departure airport; nonstop carriers on this route first
        rt = load_table("routes")
        at_origin = rt[rt["origin"] == p_origin].groupby("airline")["flights"].sum()
        nonstop = set(rt.loc[(rt["origin"] == p_origin) & (rt["dest"] == p_dest), "airline"])
        airline_opts = sorted((a for a in at_origin.index if a in set(cats["airline"])),
                              key=lambda a: (a not in nonstop, -at_origin[a]))
        # Regional airlines operate flights sold under the big airlines' brands
        REGIONAL_BRANDS = {
            "YX": "American Eagle / Delta Connection / United Express",
            "OO": "United Express / Delta Connection / American Eagle / Alaska",
            "9E": "Delta Connection",
            "MQ": "American Eagle",
            "OH": "American Eagle",
            "QX": "Alaska",
        }

        def airline_label(x):
            label = name_of.get(x, x)
            if x in REGIONAL_BRANDS:
                label += f" ({REGIONAL_BRANDS[x]})"
            return label + (" (nonstop)" if x in nonstop else "")

        p_airline = c3.selectbox(
            "Airline", airline_opts,
            format_func=airline_label,
            help=f"Airlines that fly from {p_origin}. Those marked (nonstop) fly this exact route.")
        if p_airline in REGIONAL_BRANDS:
            brand_list = REGIONAL_BRANDS[p_airline].split(" / ")
            brand_text = (brand_list[0] if len(brand_list) == 1 else
                          ", ".join(brand_list[:-1]) + (", or " if len(brand_list) > 2 else " or ") + brand_list[-1])
            st.caption(f"{name_of.get(p_airline, p_airline)} is a regional airline. It doesn't sell "
                       f"tickets under its own name, so on your booking this flight may show as "
                       f"{brand_text}.")
        if p_airline not in nonstop:
            st.caption(f"{name_of.get(p_airline, p_airline)} doesn't fly {p_origin} to {p_dest} nonstop, "
                       "so this is an estimate for a hypothetical nonstop flight. For a connecting "
                       "trip, check each leg separately: from " + p_origin + " to the airline's hub, "
                       "then from the hub to " + p_dest + ".")
        c4, c5 = st.columns(2)
        p_date = c4.date_input("Travel date", date.today() + timedelta(days=21))
        p_hour = c5.slider("Scheduled departure hour", 5, 23, 17, format="%d:00")

        if st.button("Estimate delay risk", type="primary"):
            import xgboost as xgb
            dist = routes.loc[(routes["origin"] == p_origin) & (routes["dest"] == p_dest), "distance"]
            X = pd.DataFrame({
                "month": [p_date.month], "day_of_week": [p_date.isoweekday()], "dep_hour": [p_hour],
                "airline": [p_airline], "origin": [p_origin], "dest": [p_dest],
                "distance": [float(dist.iloc[0]) if len(dist) else np.nan],
            })
            for col in ("month", "day_of_week", "dep_hour"):
                X[col] = X[col].astype("int32")
            X["distance"] = X["distance"].astype("float32")
            for col in ("airline", "origin", "dest"):
                X[col] = pd.Categorical(X[col], categories=cats[col])
            dm = xgb.DMatrix(X, enable_categorical=True)
            prob = float(booster.predict(dm)[0])
            contrib = booster.predict(dm, pred_contribs=True)[0][:-1]
            base = model_metrics.get("base_delay_rate", np.nan)

            r1, r2 = st.columns([1, 2])
            r1.metric("Chance of arriving 15+ min late", f"{prob:.0%}",
                      f"{100 * (prob - base):+.0f} pts vs average flight" if not np.isnan(base) else None,
                      delta_color="inverse")
            r1.caption(f"{name_of.get(p_airline, p_airline)}, {p_origin} to {p_dest}, "
                       f"{DAY_NAMES[p_date.weekday()]} at {p_hour}:00.")
            expl = pd.DataFrame({"factor": [FEATURE_LABELS[f] for f in X.columns], "impact": contrib})
            expl = expl.reindex(expl["impact"].abs().sort_values().index)
            fig = go.Figure(go.Bar(
                x=expl["impact"], y=expl["factor"], orientation="h",
                marker_color=[MAGENTA if v > 0 else BLUE for v in expl["impact"]],
                hovertemplate="%{y}: %{x:+.2f}<extra></extra>"))
            fig.update_xaxes(title="Pushes risk down  |  pushes risk up  (SHAP, log-odds)")
            r2.plotly_chart(style(fig, 300), width="stretch")

        if model_metrics:
            st.caption(f"Model: XGBoost, tested on flights after {model_metrics['test_period_start']} "
                       f"it never saw. ROC-AUC {model_metrics['roc_auc']}, "
                       f"PR-AUC {model_metrics['pr_auc']} (random guessing: "
                       f"{model_metrics['pr_auc_baseline']}).")

# ---------- About ----------
GITHUB_URL = "https://github.com/sridharreddy22/flight-delay-intelligence"
LINKEDIN_URL = "https://www.linkedin.com/in/sridharreddy22aekn1990"

with tabs[7]:
    month_full = ["January", "February", "March", "April", "May", "June", "July",
                  "August", "September", "October", "November", "December"]
    all_years = sorted(monthly["year"].unique())
    total_flights = int(monthly["flights"].sum())
    by_month = with_rates(monthly.groupby("month", as_index=False).sum(numeric_only=True))
    worst_m = by_month.loc[by_month["delay_rate"].idxmax()]
    best_m = by_month.loc[by_month["delay_rate"].idxmin()]
    about_metrics = load_json(DATA_DIR / "model_metrics.json")

    st.subheader("Why I built this")
    st.markdown(
        "I've always been interested in aviation, and like most travelers I've sat at a gate "
        "watching the board flip from \"On time\" to \"Delayed.\" I wanted to know whether "
        "delays are just bad luck, or whether there are real patterns hiding in the data. "
        f"So I took every domestic flight reported by U.S. airlines from {all_years[0]} to "
        f"{all_years[-1]}, **{total_flights:,} flights** in total, and started digging.\n\n"
        "The dashboard opens on St. Louis Lambert (STL) because that's my home airport, "
        "but you can switch to any airport in the country using the sidebar.")

    st.subheader("Where the data comes from")
    st.markdown(
        "Everything here comes from the U.S. Bureau of Transportation Statistics (BTS). "
        "The larger U.S. airlines are required to report every scheduled domestic flight: "
        "when it was supposed to leave and arrive, when it actually did, whether it was "
        "cancelled, and what caused any delay. The data is public and free for anyone to use.")

    st.subheader("What counts as \"late\"")
    st.markdown(
        "I use the same rule the government uses. A flight is **delayed** if it arrives "
        "15 minutes or more after its scheduled time. For departures, it's the same idea: "
        "leaving the gate 15 or more minutes late. Cancelled and diverted flights are "
        "counted separately, so they don't blur the delay numbers.")

    st.subheader("A few things I found")
    st.markdown(
        f"- **{month_full[int(worst_m.month) - 1]} is the worst month to fly.** Across all "
        f"years, {worst_m.delay_rate:.0%} of {month_full[int(worst_m.month) - 1]} flights arrived late, "
        f"compared with {best_m.delay_rate:.0%} in {month_full[int(best_m.month) - 1]}, the calmest month."
        + (" Summer schedules are packed, afternoon thunderstorms are common, and one late plane "
           "can make the later flights it operates late too." if 6 <= int(worst_m.month) <= 8 else "")
        + "\n"
        "- **The Northeast has the roughest disruptions.** Its airports have the highest "
        "cancellation rate and the longest delays. The West is the most reliable region.\n"
        "- **You can see a single bad day in the data.** On July 19, 2024, the CrowdStrike "
        "software failure took down airline computer systems. That day, 16.5% of flights were "
        "cancelled and more than half were delayed, and it took airlines several days to recover.\n"
        "- **Fall is usually the calmest season, but not in 2025.** During the government "
        "shutdown, air traffic control was short-staffed and the FAA ordered flight cuts. "
        "November 2025 cancellations hit 2.49%, more than five times the November before.")

    st.subheader("How I built it")
    st.markdown(
        "I wrote Python scripts that download the monthly files from BTS, clean them, and run "
        "quality checks on every month. For example, I checked that each flight's delay causes "
        "add up to its total delay. Those checks caught a bug in my own code that was quietly "
        "dropping 587 real flights, which I fixed.\n\n"
        "From there, I summarized the data for this dashboard, built a forecast of future delay "
        "rates with Prophet, and trained a machine-learning model with XGBoost that estimates "
        "how likely a given flight is to be late.")

    st.subheader("Being honest about the limits")
    model_line = ""
    if about_metrics:
        model_line = (
            f" In testing on flights it had never seen, it ranked a late flight above an on-time "
            f"one {about_metrics['roc_auc']:.0%} of the time, where a coin flip would get 50%.")
    st.markdown(
        "- **The delay predictor only knows what you'd know when booking a ticket:** the "
        "airline, the route, the date, and the departure time. It can't see tomorrow's "
        "thunderstorm or a broken plane, and those cause most delays." + model_line +
        " So treat it as a rough risk estimate, not a promise.\n"
        "- **The forecast learns from past seasons.** That works when the future looks like "
        "the past. The 2025 shutdown broke the usual pattern, and over that period a simple "
        "recent average actually predicted better than my model. I left that result in "
        "instead of hiding it, because knowing when a model doesn't work matters as much "
        "as knowing when it does.\n"
        "- **The data covers the airlines required to report to BTS**, which carry most U.S. "
        "domestic passengers, but not every small carrier. All times are scheduled local times.")

    st.subheader("About me")
    links = " · ".join(f"[{name}]({url})" for name, url in
                       [("GitHub", GITHUB_URL), ("LinkedIn", LINKEDIN_URL)] if url)
    st.markdown(
        "I'm **Sridhar Reddy Somireddy**, a data analyst based in St. Louis with a Master of "
        "Science in Data Analytics from Saint Louis University. I enjoy turning messy, "
        "real-world data into answers people can actually use, and aviation is one of my "
        "favorite subjects to do that with. I'm currently looking for data analyst roles "
        "and internships, so if you'd like to talk, I'd be glad to hear from you."
        + (f"\n\n{links}" if links else ""))
