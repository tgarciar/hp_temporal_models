# streamlit_app.py
import os
from datetime import date
from typing import Tuple

import numpy as np
import pandas as pd
import streamlit as st
import seaborn as sns

# Plotly
import plotly.express as px
import plotly.io as pio

# ---------------------------------------------
# PAGE + THEME (Streamlit shell + CSS)
# ---------------------------------------------
st.set_page_config(
    page_title="Battery Arbitrage Dashboard",
    page_icon="⚡",
    layout="wide",
)

st.markdown("""
<style>
.metric-card {
  background: #1c1f26;
  border: 1px solid #2a2f3a;
  border-radius: 12px;
  padding: 16px;
  text-align: center;           /* 👈 center everything inside */
}
.metric-title {
  color: #9aa4b2;
  font-size: 0.9rem;
  margin-bottom: 6px;
  text-align: center;           /* 👈 explicitly center text */
}
.metric-value {
  color: #e8edf2;
  font-size: 1.6rem;
  font-weight: 700;
  text-align: center;           /* 👈 explicitly center value */
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------
# Matplotlib/Seaborn + Plotly THEME (your spec)
# ---------------------------------------------
company_colors = [
    "#2f0ac2","#187cc4","#a8f500","#0bad76","#6322c3","#9a67e4",
    "#c2fbe8","#f58849","#f2e22c","#0072ce","#6ac227","#37136C"
]

# Matplotlib (used only for font loading; plotting is Plotly)
FONT_DIR = "/home/tgarciar/.local/share/fonts/Plus Jakarta Sans"
if FONT_DIR and os.path.isdir(FONT_DIR):
    for path in fm.findSystemFonts(fontpaths=[FONT_DIR], fontext="ttf"):
        fm.fontManager.addfont(path)
    try:
        fm._load_fontmanager(try_read_cache=False)
    except Exception:
        pass

mpl.rcParams["font.family"] = ["Plus Jakarta Sans", "DejaVu Sans"]
mpl.rcParams["axes.prop_cycle"] = mpl.cycler(color=company_colors)

# Seaborn (for global styling of any mpl fallback)
sns.set_theme(
    context="paper",
    style="darkgrid",
    palette=company_colors,
    font="Plus Jakarta Sans",
    font_scale=1.1,
    color_codes=True,
    rc={
        "axes.facecolor":  "#1e1e1e",
        "figure.facecolor":"#1e1e1e",
        "axes.edgecolor":  "white",
        "axes.labelcolor": "white",
        "xtick.color":     "white",
        "ytick.color":     "white",
        "text.color":      "white",
        "legend.frameon":  True,
        "legend.facecolor":"#2b2b2b",
        "legend.edgecolor":"white",
        "grid.color":      "#444444",
        "axes.titleweight":"bold",
        "grid.alpha": 0.22,
    }
)

# Plotly defaults (dark to match your style)
pio.templates.default = "plotly_dark"

# ---------------------------------------------
# DATA LOADING
# ---------------------------------------------
CUTOFF_END = pd.Timestamp("2025-08-11 23:00:00")

@st.cache_data(show_spinner=False)
def load_prices() -> pd.DataFrame:
    """
    Return a DataFrame with ['LocalDatetime','Euro per MWh','Country'].
    Clips to >= 2022-01-01 and <= 2025-08-11 23:00.
    """
    df = pd.read_parquet("bidding_zone_prices.parquet")
    df = df[['LocalDatetime','Euro per MWh','Country']]
    df["LocalDatetime"] = pd.to_datetime(df["LocalDatetime"])
    df = df[(df["LocalDatetime"] >= "2022-01-01 00:00:00") &
            (df["LocalDatetime"] <= CUTOFF_END)]
    # Optional keep-list of countries
    keep = ["ES", "FI", "NL", "DE", "FR", "DE-LU"]
    if "Country" in df.columns:
        df = df[df["Country"].isin(keep)]
    return df

# ---------------------------------------------
# OPTIMIZER CORE (same logic as your notebook)
# ---------------------------------------------
def _best_windows_one_day_fixed(day_prices: np.ndarray,
                                charge_len: int,
                                discharge_len: int,
                                capacity_mwh: float):
    assert len(day_prices) == 24, "Expect exactly 24 hourly prices per day."
    if charge_len + discharge_len > 24:
        return {"shutdown": True, "gross_arbitrage_EUR": 0.0,
                "charge_start_hour": None, "discharge_start_hour": None,
                "charge_len": charge_len, "discharge_len": discharge_len,
                "avg_price_charge": None, "avg_price_discharge": None}

    p = day_prices.astype(float)
    p_ext = np.concatenate([p, p])
    cs = np.zeros(len(p_ext) + 1)
    cs[1:] = np.cumsum(p_ext)

    def win_avg(start, L):
        return (cs[start + L] - cs[start]) / L

    best, best_profit = None, -np.inf
    for c_start in range(24):
        c_end = c_start + charge_len
        d_min = c_end
        d_max = c_start + 24 - discharge_len
        if d_min > d_max:
            continue
        c_avg = win_avg(c_start, charge_len)
        for d_start_un in range(d_min, d_max + 1):
            d_avg = win_avg(d_start_un, discharge_len)
            profit = (d_avg - c_avg) * capacity_mwh
            if profit > best_profit:
                best_profit = profit
                best = {
                    "charge_start_hour": c_start % 24,
                    "discharge_start_hour": d_start_un % 24,
                    "charge_len": charge_len,
                    "discharge_len": discharge_len,
                    "avg_price_charge": float(c_avg),
                    "avg_price_discharge": float(d_avg),
                    "gross_arbitrage_EUR": float(profit),
                    "shutdown": False
                }

    if best is None or best_profit <= 0:
        return {
            "charge_start_hour": None,
            "discharge_start_hour": None,
            "charge_len": charge_len,
            "discharge_len": discharge_len,
            "avg_price_charge": None,
            "avg_price_discharge": None,
            "gross_arbitrage_EUR": 0.0,
            "shutdown": True
        }
    return best


def _year_factor_map(index_dates, perf_drop_rate_per_year=None, perf_multiplier_by_year=None):
    years = pd.Index(pd.to_datetime(index_dates).year, name="year")
    yrs_sorted = np.unique(years.values)
    if perf_multiplier_by_year is not None:
        mp = pd.Series({y: float(perf_multiplier_by_year.get(y, 1.0)) for y in yrs_sorted}, name="perf_mult")
    else:
        rate = float(perf_drop_rate_per_year) if perf_drop_rate_per_year else 0.0
        if rate <= 0:
            mp = pd.Series({y: 1.0 for y in yrs_sorted}, name="perf_mult")
        else:
            y0 = yrs_sorted.min()
            mp = pd.Series({y: (1.0 - rate) ** (y - y0) for y in yrs_sorted}, name="perf_mult")
    return mp


def _choose_forced_shutdown_days(dates, extra_shutdowns_per_year=0, seed=None):
    rng = np.random.default_rng(seed)
    dates = pd.to_datetime(pd.Index(dates).date).unique()
    years = pd.Series(dates).map(lambda d: d.year)
    forced = set()
    if isinstance(extra_shutdowns_per_year, dict):
        for y, n in extra_shutdowns_per_year.items():
            year_days = np.array([pd.Timestamp(d) for d, yy in zip(dates, years) if yy == int(y)])
            if len(year_days) == 0 or int(n) <= 0:
                continue
            n_pick = min(int(n), len(year_days))
            picks = rng.choice(year_days, size=n_pick, replace=False)
            forced.update(pd.Timestamp(p) for p in picks)
    else:
        n_per_year = int(extra_shutdowns_per_year or 0)
        if n_per_year > 0:
            for y in np.unique(years):
                year_days = np.array([pd.Timestamp(d) for d, yy in zip(dates, years) if yy == y])
                n_pick = min(n_per_year, len(year_days))
                picks = rng.choice(year_days, size=n_pick, replace=False)
                forced.update(pd.Timestamp(p) for p in picks)
    return forced


def optimize_arbitrage_sweep(
    prices_df: pd.DataFrame,
    price_col: str = "Euro per MWh",
    ts_col: str = "LocalDatetime",
    capacity_mwh: float = 1.5,
    date_start="2022-01-01", date_end=None,
    charge_len_options=range(6, 13),
    discharge_len_options=range(6, 13),
    perf_drop_rate_per_year: float | None = None,
    perf_multiplier_by_year: dict | None = None,
    extra_shutdowns_per_year: int | dict = 0,
    rand_seed: int | None = 42,
    opex_by_year: dict | None = None
):
    df = prices_df.copy()
    if ts_col in df.columns:
        df[ts_col] = pd.to_datetime(df[ts_col])
        df = df.set_index(ts_col)
    df = df.sort_index()

    # filter dates
    idx_tz = df.index.tz
    def _to_index_tz(x):
        if x is None: return None
        x = pd.to_datetime(x)
        if idx_tz is None:
            return x.tz_localize(None) if x.tzinfo is not None else x
        return (x.tz_localize(idx_tz) if x.tzinfo is None else x.tz_convert(idx_tz))
    d0, d1 = _to_index_tz(date_start), _to_index_tz(date_end)
    if d0 is not None or d1 is not None:
        df = df.loc[d0:d1]
    if df.empty:
        return (pd.DataFrame(), pd.DataFrame(), pd.DataFrame())

    s = df[price_col].astype(float).clip(lower=0)
    perf_map = _year_factor_map(s.index, perf_drop_rate_per_year, perf_multiplier_by_year)

    pairs = [(Lc, Ld) for Lc in charge_len_options for Ld in discharge_len_options if (Lc + Ld) <= 24]
    if not pairs:
        raise ValueError("No valid charge/discharge length pairs (sum must be <= 24).")

    daily_all, hourly_all = [], []
    forced_shutdown_dates = _choose_forced_shutdown_days(s.index, extra_shutdowns_per_year, rand_seed)

    for (Lc, Ld) in pairs:
        daily_rows, hourly_rows = [], []

        for day, grp in s.groupby(s.index.date):
            day_ts = pd.Timestamp(day)
            year = day_ts.year
            perf_mult = perf_map.loc[year] if year in perf_map.index else 1.0
            capacity_eff = capacity_mwh * float(perf_mult)

            hourly_prices = grp.groupby(grp.index.hour).mean()
            if len(hourly_prices) != 24:
                continue
            hourly_prices = hourly_prices.reindex(range(24))
            prices_24 = hourly_prices.to_numpy()

            res = _best_windows_one_day_fixed(prices_24, Lc, Ld, capacity_eff)
            res["date"] = day_ts
            res["charge_len"] = Lc
            res["discharge_len"] = Ld

            forced = (day_ts.normalize() in forced_shutdown_dates)
            if forced:
                res.update({
                    "shutdown": True,
                    "gross_arbitrage_EUR": 0.0,
                    "avg_price_charge": None,
                    "avg_price_discharge": None,
                    "charge_start_hour": None,
                    "discharge_start_hour": None
                })

            daily_rows.append(res)

            if res["shutdown"]:
                for hr in range(24):
                    hourly_rows.append({
                        "date": day_ts, "hour": hr, "price": float(prices_24[hr]),
                        "role": "idle", "energy_mwh": 0.0, "cashflow_eur": 0.0,
                        "shutdown": True, "charge_len": Lc, "discharge_len": Ld
                    })
            else:
                c_start, d_start = int(res["charge_start_hour"]), int(res["discharge_start_hour"])
                charge_hours    = {(c_start + i) % 24 for i in range(Lc)}
                discharge_hours = {(d_start + i) % 24 for i in range(Ld)}
                charge_rate_mwh    = capacity_eff / Lc
                discharge_rate_mwh = capacity_eff / Ld
                for hr in range(24):
                    price = float(prices_24[hr])
                    if hr in charge_hours:
                        role, e, cash = "charge", charge_rate_mwh, - price * charge_rate_mwh
                    elif hr in discharge_hours:
                        role, e, cash = "discharge", discharge_rate_mwh, + price * discharge_rate_mwh
                    else:
                        role, e, cash = "idle", 0.0, 0.0
                    hourly_rows.append({
                        "date": day_ts, "hour": hr, "price": price,
                        "role": role, "energy_mwh": e, "cashflow_eur": cash,
                        "shutdown": False, "charge_len": Lc, "discharge_len": Ld
                    })

        daily_df = pd.DataFrame(daily_rows).set_index("date").sort_index()

        def _fmt(start, L):
            if start is None or L is None or pd.isna(start) or pd.isna(L):
                return None
            end = (int(start) + int(L)) % 24
            return f"{int(start):02d}:00–{end:02d}:00"

        daily_df["best_charge_window"] = None
        daily_df["best_discharge_window"] = None
        if not daily_df.empty:
            mask = ~daily_df["shutdown"]
            daily_df.loc[mask, "best_charge_window"] = daily_df.loc[mask].apply(
                lambda r: _fmt(r["charge_start_hour"], r["charge_len"]), axis=1
            )
            daily_df.loc[mask, "best_discharge_window"] = daily_df.loc[mask].apply(
                lambda r: _fmt(r["discharge_start_hour"], r["discharge_len"]), axis=1
            )

        hourly_df = pd.DataFrame(hourly_rows).set_index(["date","hour"]).sort_index()
        daily_all.append(daily_df)
        hourly_all.append(hourly_df)

    daily_all  = pd.concat(daily_all, axis=0).sort_index()
    hourly_all = pd.concat(hourly_all, axis=0).sort_index()

    daily_all["year"] = daily_all.index.year
    daily_all["forced_shutdown"] = False
    if forced_shutdown_dates:
        forced_dates_index = pd.Index([pd.Timestamp(d) for d in forced_shutdown_dates])
        daily_all.loc[daily_all.index.normalize().isin(forced_dates_index), "forced_shutdown"] = True

    annual = (
        daily_all
        .groupby(["year","charge_len","discharge_len"])
        .agg(
            days=("shutdown","size"),
            shutdown_days=("shutdown","sum"),
            forced_shutdown_days=("forced_shutdown","sum"),
            gross_arbitrage_EUR=("gross_arbitrage_EUR","sum")
        )
        .reset_index()
    )

    # We won't set opex here; we’ll compute proration downstream for scorecards & plots
    annual["opex_eur"] = 0.0
    annual["net_profit_eur"] = annual["gross_arbitrage_EUR"] - annual["opex_eur"]

    daily_cols = [
        "shutdown","gross_arbitrage_EUR","avg_price_charge","avg_price_discharge",
        "charge_len","discharge_len","charge_start_hour","discharge_start_hour",
        "best_charge_window","best_discharge_window","year","forced_shutdown"
    ]
    hourly_cols = ["price","role","energy_mwh","cashflow_eur","shutdown","charge_len","discharge_len"]

    daily_all  = daily_all.reindex(columns=daily_cols)
    hourly_all = hourly_all.reindex(columns=hourly_cols)

    annual = annual[["year","charge_len","discharge_len","days","shutdown_days","forced_shutdown_days",
                     "gross_arbitrage_EUR","opex_eur","net_profit_eur"]].sort_values(
                        ["year","charge_len","discharge_len"]
                     )
    return daily_all, hourly_all, annual

# ---------------------------------------------
# HELPERS: OPEX PRORATION + PLOT DATA
# ---------------------------------------------
def build_cumsum_gross_df(daily_df: pd.DataFrame) -> pd.DataFrame:
    """Cumulative gross only (no OPEX)."""
    if daily_df.empty:
        return pd.DataFrame(columns=["date","scenario","cumsum"])
    rows = []
    for (Lc, Ld), df_s in daily_df.groupby(["charge_len","discharge_len"]):
        scen = f"{int(Lc)}c-{int(Ld)}d"
        s = (df_s["gross_arbitrage_EUR"]
             .groupby(df_s.index.normalize()).sum()
             .sort_index())
        cs = s.cumsum()
        rows.append(pd.DataFrame({"date": cs.index, "scenario": scen, "cumsum": cs.values}))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["date","scenario","cumsum"])

def build_annual_gross(daily_df: pd.DataFrame) -> pd.DataFrame:
    """Gross per year, per scenario (no OPEX)."""
    if daily_df.empty:
        return pd.DataFrame(columns=["year","scenario","gross"])
    out = (daily_df
           .assign(year=lambda d: d.index.year,
                   scenario=lambda d: d["charge_len"].astype(int).astype(str)+"c-"+d["discharge_len"].astype(int).astype(str)+"d")
           .groupby(["year","scenario"], as_index=False)["gross_arbitrage_EUR"].sum()
           .rename(columns={"gross_arbitrage_EUR":"gross"}))
    return out


def days_in_year(y: int) -> int:
    return 366 if pd.Timestamp(f"{y}-12-31").is_leap_year else 365

def compute_prorated_opex_per_scenario(daily_df: pd.DataFrame, opex_year: float) -> pd.Series:
    """
    For each scenario (charge_len, discharge_len), charge:
      - full OPEX for every complete year before the last year present,
      - prorated OPEX for the last (possibly incomplete) year by elapsed days/total days.
    Returns a Series indexed by (charge_len, discharge_len) with total OPEX per scenario.
    """
    if daily_df.empty:
        return pd.Series(dtype=float)

    out = {}
    for (Lc, Ld), df_s in daily_df.groupby(["charge_len","discharge_len"]):
        dates = df_s.index.normalize()
        y_min, y_max = int(dates.min().year), int(dates.max().year)
        # Full years strictly before the last year
        n_full_years = max(0, y_max - y_min)
        full_opex = n_full_years * opex_year
        # Prorated OPEX on the final year
        y = y_max
        year_start = pd.Timestamp(f"{y}-01-01")
        last_date  = dates.max()
        elapsed_days = (last_date - year_start).days + 1
        pror = (elapsed_days / days_in_year(y)) * opex_year
        out[(Lc, Ld)] = full_opex + pror
    return pd.Series(out)

def build_cumsum_with_opex_df(daily_df: pd.DataFrame, opex_year: float) -> pd.DataFrame:
    """
    Build a tidy DataFrame with columns [date, scenario, cumsum]
    where OPEX is injected as:
      - full OPEX on Dec 31 for completed years,
      - prorated OPEX on the last available date of the final year.
    """
    rows = []
    if daily_df.empty:
        return pd.DataFrame(columns=["date","scenario","cumsum"])

    for (Lc, Ld), df_s in daily_df.groupby(["charge_len","discharge_len"]):
        scen = f"{int(Lc)}c-{int(Ld)}d"
        s = (df_s["gross_arbitrage_EUR"]
             .groupby(df_s.index.normalize()).sum()
             .sort_index())

        if s.empty:
            continue

        # Determine years & last date
        years = s.index.year.unique()
        y_max = years.max()

        # Inject full OPEX at 31/12 for complete years
        for y in years:
            if y < y_max:
                opex = opex_year
                target = pd.Timestamp(f"{y}-12-31")
                if target not in s.index:
                    # if no 31/12 datapoint, use last available date in that year
                    yr_mask = s.index.year == y
                    if not yr_mask.any():
                        continue
                    target = s.index[yr_mask][-1]
                s.loc[target] = s.get(target, 0.0) - opex

        # Inject prorated OPEX at last date for the final year
        yr_mask_last = s.index.year == y_max
        last_date = s.index[yr_mask_last].max()
        pror = ( (last_date - pd.Timestamp(f"{y_max}-01-01")).days + 1 ) / days_in_year(y_max)
        s.loc[last_date] = s.get(last_date, 0.0) - opex_year * pror

        s = s.sort_index()
        cs = s.cumsum()
        rows.append(
            pd.DataFrame({"date": cs.index, "scenario": scen, "cumsum": cs.values})
        )
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["date","scenario","cumsum"])

def build_negative_days_month_df(daily_df: pd.DataFrame) -> pd.DataFrame:
    """
    Count negative-revenue days by Month-YYYY for the **winning scenario** (max revenue).
    Returns columns [month, count].
    """
    if daily_df.empty:
        return pd.DataFrame(columns=["month","count"])

    scen_revenue = daily_df.groupby(["charge_len","discharge_len"])["gross_arbitrage_EUR"].sum()
    winner = scen_revenue.idxmax()  # (Lc, Ld)
    df_w = daily_df[(daily_df["charge_len"]==winner[0]) & (daily_df["discharge_len"]==winner[1])].copy()
    df_w["date"] = df_w.index.normalize()
    df_w["month"] = df_w["date"].dt.strftime("%Y-%m")
    neg = df_w[df_w["gross_arbitrage_EUR"] < 0]
    counts = neg.groupby("month").size().rename("count").reset_index()
    return counts.sort_values("month")

def build_annual_prorated_net(daily_df: pd.DataFrame, opex_year: float) -> pd.DataFrame:
    """
    Compute yearly gross/opex/net per (charge_len, discharge_len), with OPEX rules:
      - Full OPEX for each complete year before the final year present for that scenario
      - Prorated OPEX for the final (possibly partial) year by elapsed days / days in year
    Returns columns: year, charge_len, discharge_len, scenario, gross, opex, net
    """
    if daily_df.empty:
        return pd.DataFrame(columns=["year","charge_len","discharge_len","scenario","gross","opex","net"])

    rows = []
    for (Lc, Ld), df_s in daily_df.groupby(["charge_len","discharge_len"]):
        scen = f"{int(Lc)}c-{int(Ld)}d"
        df_s = df_s.sort_index()
        dates = df_s.index.normalize()
        years = sorted(dates.year.unique().tolist())
        last_year = max(years)

        for y in years:
            gross = df_s.loc[df_s.index.year == y, "gross_arbitrage_EUR"].sum()
            if y < last_year:
                opex = float(opex_year)  # completed year
            else:
                # prorate by elapsed days in the final year
                yr_dates = dates[dates.year == y]
                if len(yr_dates) == 0:
                    continue
                last_date = yr_dates.max()
                elapsed_days = (last_date - pd.Timestamp(f"{y}-01-01")).days + 1
                opex = float(opex_year) * (elapsed_days / days_in_year(y))
            rows.append({
                "year": int(y),
                "charge_len": int(Lc),
                "discharge_len": int(Ld),
                "scenario": scen,
                "gross": float(gross),
                "opex": float(opex),
                "net": float(gross - opex),
            })
    return pd.DataFrame(rows)


def plot_yearly_net_stacked_mpl(df_yearly: pd.DataFrame, ax=None, title="Yearly Net Profit (stacked)"):
    """
    Alternate 'different look' version using Matplotlib/Seaborn (not Plotly).
    Stacks scenarios horizontally per year.
    """
    import matplotlib.pyplot as plt
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 5))

    if df_yearly.empty:
        ax.set_title(title)
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        return ax

    years = sorted(df_yearly["year"].unique())
    scenarios = sorted(df_yearly["scenario"].unique())
    # build matrix [len(years) x len(scenarios)]
    data = np.array([
        [df_yearly[(df_yearly["year"] == y) & (df_yearly["scenario"] == s)]["net"].sum()
         for s in scenarios]
        for y in years
    ])

    left = np.zeros(len(years))
    for i, s in enumerate(scenarios):
        ax.barh(years, data[:, i], left=left, label=s)
        left += data[:, i]

    ax.set_title(title)
    ax.set_xlabel("Net Profit [EUR]")
    ax.set_ylabel("Year")
    ax.legend(title="Scenario", bbox_to_anchor=(1.02, 1), loc="upper left")
    return ax



# ---------------------------------------------
# UI – Controls
# ---------------------------------------------
st.title("⚡ Earnings Potentials — Results Explorer")

with st.container():
    # 3 columns: Filters & dates | Asset & economics | Windows & action
    c1, c2, c3, c4 = st.columns([1, 1, 1,1 ])

    # --- Column 1: Country + Dates ---
    with c1:
        prices_df = load_prices()
        countries = (
            sorted(prices_df["Country"].dropna().unique().tolist())
            if not prices_df.empty else ["ES","DE","FI","FR","NL"]
        )
        country = st.selectbox(
            "Country",
            countries,
            index=0,
            help="DE - Germany, ES - Spain, FI - Finland, FR - France, NL - Netherlands",
        )

        # Cap picker to 11/08/2025
        daterange = st.date_input(
            "Date range",
            value=(date(2022, 1, 1), date(2025, 8, 11)),
            min_value=date(2015, 1, 1),
            max_value=date(2025, 8, 11),
            help="Available Data: 01/01/2022 - 11/08/2025",
        )
        date_start, date_end = (
            pd.to_datetime(daterange[0]),
            pd.to_datetime(daterange[1]),
        ) if isinstance(daterange, tuple) else (pd.to_datetime(daterange), None)


    # --- Column 2: Battery & economics ---
    with c2:
        capacity_mwh = st.number_input(
            "Battery capacity (MWh)",
            min_value=0.1,
            value=1.5,
            step=0.1,
        )
        perf_drop = st.number_input(
            "Performance drop rate per year",
            min_value=0.0,
            max_value=0.2,
            value=0.0,
            step=0.005,
            help="Fractional; e.g., 0.01 = 1%/yr",
        )

    with c3:
        charge_len_single = st.selectbox(
                    "Charging hours",
                    options=list(range(6, 13)),
                    index=2,  # default 8h
                )
        discharge_min, discharge_max = st.select_slider(
                    "Discharge window (hours)",
                    options=list(range(6, 13)),
                    value=(6, 12),
                )


    # --- Column 3: Windows & action ---
    with c4:
        opex_year = st.number_input(
                    "OPEX per year (EUR)",
                    min_value=0.0,
                    value=5000.0,
                    step=500.0,
                )

        forced_shutdowns = st.number_input(
                    "Extra shutdowns per year",
                    min_value=0,
                    value=0,
                    step=1,
                )
        # Run button at the bottom of col 3
        run_btn = st.button("🚀 Run the model", use_container_width=True, type="primary")

st.divider()


st.divider()

# ---------------------------------------------
# RUN
# ---------------------------------------------
daily_all = hourly_all = annual = pd.DataFrame()

if run_btn:
    if prices_df.empty:
        st.warning("No data loaded. Replace load_prices() with your real data.")
    else:
        df_use = prices_df[prices_df["Country"] == country].copy()

        def run_optimizer(df_use) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
            daily_all, hourly_all, annual = optimize_arbitrage_sweep(
                df_use,
                price_col="Euro per MWh",
                ts_col="LocalDatetime",
                capacity_mwh=capacity_mwh,
                date_start=date_start, date_end=date_end,
                charge_len_options=[int(charge_len_single)],                 # single charging hours
                discharge_len_options=range(int(discharge_min), int(discharge_max)+1),
                perf_drop_rate_per_year=perf_drop if perf_drop > 0 else None,
                extra_shutdowns_per_year=int(forced_shutdowns),
                opex_by_year=None,     # we'll handle OPEX downstream
            )
            return daily_all, hourly_all, annual

        with st.spinner("Running optimization…"):
            daily_all, hourly_all, annual = run_optimizer(df_use)

            # Optional: flip natural shutdowns off (keep days even if profit==0)

            allow_neg_days = False
            if allow_neg_days and not daily_all.empty:
                natural_mask = (daily_all["shutdown"] == True) & (~daily_all["forced_shutdown"])
                daily_all.loc[natural_mask, "shutdown"] = False

# ---------------------------------------------
# SCORECARDS (Avg days, Max revenue, Avg OPEX with proration, Net)
# ---------------------------------------------
if not daily_all.empty:
    # Avg number of days across scenarios
    days_per_scen = daily_all.groupby(["charge_len","discharge_len"]).size()
    avg_days = int(days_per_scen.mean()) if len(days_per_scen) else 0

    # Max revenue & winning scenario
    scen_revenue = daily_all.groupby(["charge_len","discharge_len"])["gross_arbitrage_EUR"].sum()
    if not scen_revenue.empty:
        max_revenue = float(scen_revenue.max())
        winner = scen_revenue.idxmax()  # (Lc, Ld)
        winner_label = f"{winner[0]}c-{winner[1]}d"
    else:
        max_revenue, winner_label = 0.0, "—"

    # Avg OPEX across scenarios with end-of-year proration
    opex_per_scen = compute_prorated_opex_per_scenario(daily_all, float(opex_year))
    avg_opex = float(opex_per_scen.mean()) if not opex_per_scen.empty else 0.0

    net_profit = max_revenue - avg_opex

    colA, colB, colC, colD = st.columns(4)

    def card(col, title, value):
        col.markdown(
            f"""
            <div class="metric-card">
              <div class="metric-title">{title}</div>
              <div class="metric-value">{value}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    card(colA, "Total Days", f"{avg_days:,}")
    card(colB, f"Max Revenue ({winner_label})", f"€ {max_revenue:,.0f}")
    card(colC, "Total OPEX", f"€ {avg_opex:,.0f}")
    card(colD, "Net Profit (Max – Avg OPEX)", f"€ {net_profit:,.0f}")

    st.divider()

# ---------------------------------------------
# PLOTS (Plotly Express)
# ---------------------------------------------

if not daily_all.empty:

    st.subheader("Charts")

    st.markdown("""
    <div style="background-color:#2b2b2b; padding:12px; border-radius:8px; border: 1px solid #444;">
    <b>Important notes:</b>
    <ul>
    <li>2025 data until 11 of August.</li>
    <li>Naive model: 12 hours of Charging and Discharging (C: 6 AM – 5 PM, D: 6 PM – 5 AM).</li>
    <li>OPEX is added every last day of the period to see the real effect.</li>
    <li>2022: Price crisis in European Countries (Ukraine/Russia War and Post-Covid).</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

    left, right = st.columns([2, 1])
    with left:
        st.subheader("Earnings Potentials by Scenarios (OPEX applied at period-end)")
        cumsum_df = build_cumsum_with_opex_df(daily_all, float(opex_year))
        if not cumsum_df.empty:
            fig = px.line(
                cumsum_df, x="date", y="cumsum", color="scenario",
                color_discrete_sequence=company_colors,
                labels={"date":"Date","cumsum":"Cumulative profit [EUR]","scenario":"Scenario"}
            )
            fig.update_layout(margin=dict(l=10,r=10,t=10,b=10))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No data to plot.")

    with right:
        st.subheader("Yearly Net Profit (€) — Stacked by Scenario")

        yearly_df = build_annual_prorated_net(daily_all, float(opex_year))
        if not yearly_df.empty:
            import plotly.express as px

            fig2 = px.bar(
                yearly_df,
                x="net",                # horizontal value axis
                y="year",               # categories on Y
                color="scenario",       # stack by scenario
                orientation="h",
                barmode="group",
                text_auto= True,
                color_discrete_sequence=company_colors,
                labels={"net":"Net Profit (€)", "year":"Year", "scenario":"Scenario"},
                title=""
            )
            fig2.update_layout(
                yaxis=dict(type="category"),  # ensure 2019,2020,... not treated as continuous
                margin=dict(l=10, r=10, t=10, b=10),
            )
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("No yearly data available for the selected filters.")


elif run_btn:
    st.info("No results to display. Check data filters and try again.")
