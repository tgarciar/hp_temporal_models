# streamlit_app.py
import os
from datetime import date
from typing import Tuple

import numpy as np
import pandas as pd
import streamlit as st

import matplotlib as mpl
import matplotlib.font_manager as fm
import seaborn as sns

# Plotly
import plotly.express as px
import plotly.io as pio
import plotly.graph_objects as go

# ---------------------------------------------
# PAGE + THEME (Streamlit shell + CSS)
# ---------------------------------------------
st.set_page_config(
    page_title="Energy Arbitrage Dashboard",
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
  text-align: center;
}
.metric-title {
  color: #9aa4b2;
  font-size: 0.9rem;
  margin-bottom: 6px;
  text-align: center;
}
.metric-value {
  color: #e8edf2;
  font-size: 1.6rem;
  font-weight: 700;
  text-align: center;
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
    # Make it robust to Series or Index/array
    dt = pd.to_datetime(index_dates, errors="coerce")
    if isinstance(dt, pd.Series):
        years_arr = dt.dt.year.to_numpy()
    else:
        years_arr = pd.Index(dt).year

    yrs_sorted = np.unique(years_arr)
    if perf_multiplier_by_year is not None:
        mp = pd.Series({int(y): float(perf_multiplier_by_year.get(int(y), 1.0)) for y in yrs_sorted},
                       name="perf_mult")
    else:
        rate = float(perf_drop_rate_per_year) if perf_drop_rate_per_year else 0.0
        if rate <= 0:
            mp = pd.Series({int(y): 1.0 for y in yrs_sorted}, name="perf_mult")
        else:
            y0 = int(yrs_sorted.min())
            mp = pd.Series({int(y): (1.0 - rate) ** (int(y) - y0) for y in yrs_sorted}, name="perf_mult")
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
# NAIVE BASELINE (fixed 06–18 charge, 18–06 discharge)
# ---------------------------------------------
def build_naive_fixed_windows(
    prices_df: pd.DataFrame,
    price_col: str = "Euro per MWh",
    ts_col: str = "LocalDatetime",
    capacity_mwh: float = 1.5,
    date_start=None, date_end=None,
    charge_start_hour: int = 6,
    charge_len: int = 12,
    perf_drop_rate_per_year: float | None = None,
    perf_multiplier_by_year: dict | None = None,
    extra_shutdowns_per_year: int | dict = 0,
    rand_seed: int | None = 42,
):
    """
    Baseline strategy:
      - Charge on fixed window [charge_start_hour, charge_start_hour+charge_len)
      - Discharge on remaining hours
      - Capacity is split evenly across charge hours and discharge hours,
        with optional performance degradation and forced shutdowns.
    Returns (daily_df, hourly_df) in the SAME schema used by optimize_arbitrage_sweep.
    """
    if prices_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    df = prices_df.copy()
    df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce")
    df = df.dropna(subset=[ts_col, price_col]).sort_values(ts_col)

    # Date filter
    if date_start is not None:
        df = df[df[ts_col] >= pd.to_datetime(date_start)]
    if date_end is not None:
        df = df[df[ts_col] <= pd.to_datetime(date_end)]
    if df.empty:
        return pd.DataFrame(), pd.DataFrame()

    df["date"] = df[ts_col].dt.normalize()
    df["hour"] = df[ts_col].dt.hour
    df["price"] = df[price_col].astype(float).clip(lower=0)

    # Per-year performance factor
    perf_map = _year_factor_map(df["date"], perf_drop_rate_per_year, perf_multiplier_by_year)

    # Forced shutdown days
    forced_shutdown_dates = _choose_forced_shutdown_days(df["date"], extra_shutdowns_per_year, rand_seed)
    forced_shutdown_dates = set(pd.Timestamp(d).normalize() for d in forced_shutdown_dates)

    # Fixed windows
    charge_hours = {(int(charge_start_hour) + i) % 24 for i in range(int(charge_len))}
    discharge_len = 24 - int(charge_len)
    discharge_start_hour = (int(charge_start_hour) + int(charge_len)) % 24

    rows_hourly = []
    rows_daily = []

    # Build per-day
    for day, grp in df.groupby(df["date"]):
        day_ts = pd.Timestamp(day)
        year = int(day_ts.year)
        perf_mult = float(perf_map.loc[year]) if year in perf_map.index else 1.0
        cap_eff = float(capacity_mwh) * perf_mult

        # Determine if forced shutdown
        is_forced = (day_ts in forced_shutdown_dates)

        # Ensure all 24 hours are represented (fill if any hours missing)
        prices_by_hour = grp.groupby("hour")["price"].mean().reindex(range(24), fill_value=np.nan)
        # If a price is missing for an hour, skip energy/cashflow on that hour (treated as idle)
        for hr in range(24):
            price = float(prices_by_hour.iloc[hr]) if not pd.isna(prices_by_hour.iloc[hr]) else np.nan
            role = "idle"
            energy = 0.0
            cash = 0.0

            if not is_forced and not np.isnan(price):
                if hr in charge_hours:
                    role = "charge"
                else:
                    role = "discharge"

            rows_hourly.append({
                "date": day_ts, "hour": hr,
                "price": (0.0 if np.isnan(price) else price),
                "role": (role if not is_forced else "idle"),
                "energy_mwh": 0.0,  # fill later after we know counts
                "cashflow_eur": 0.0,
                "shutdown": bool(is_forced),
                "charge_len": int(charge_len),
                "discharge_len": int(discharge_len),
            })

        # After roles assigned, compute split based on actual counts present
        day_mask = [(r["date"] == day_ts) for r in rows_hourly]
        day_rows_idx = [i for i, m in enumerate(day_mask) if m][-24:]  # last 24 we just appended
        roles = [rows_hourly[i]["role"] for i in day_rows_idx]
        prices = [rows_hourly[i]["price"] for i in day_rows_idx]

        n_charge = sum(1 for r in roles if r == "charge")
        n_discharge = sum(1 for r in roles if r == "discharge")

        charge_rate = (cap_eff / n_charge) if n_charge > 0 else 0.0
        discharge_rate = (cap_eff / n_discharge) if n_discharge > 0 else 0.0

        daily_gross = 0.0
        for i, idx in enumerate(day_rows_idx):
            role = roles[i]
            price = prices[i]
            if np.isnan(price) or is_forced or role == "idle":
                rows_hourly[idx]["energy_mwh"] = 0.0
                rows_hourly[idx]["cashflow_eur"] = 0.0
                continue
            if role == "charge":
                e = charge_rate
                c = - price * e
            else:  # discharge
                e = discharge_rate
                c = + price * e
            rows_hourly[idx]["energy_mwh"] = e
            rows_hourly[idx]["cashflow_eur"] = c
            daily_gross += c

        # Build daily row
        # Average prices over roles (avoid divide-by-zero)
        if n_charge > 0:
            avg_charge = float(np.nanmean([p for p, r in zip(prices, roles) if r == "charge"]))
        else:
            avg_charge = None
        if n_discharge > 0:
            avg_discharge = float(np.nanmean([p for p, r in zip(prices, roles) if r == "discharge"]))
        else:
            avg_discharge = None

        rows_daily.append({
            "date": day_ts,
            "shutdown": bool(is_forced),
            "gross_arbitrage_EUR": float(0.0 if is_forced else daily_gross),
            "avg_price_charge": avg_charge,
            "avg_price_discharge": avg_discharge,
            "charge_len": int(charge_len),
            "discharge_len": int(discharge_len),
            "charge_start_hour": int(charge_start_hour),
            "discharge_start_hour": int(discharge_start_hour),
            "best_charge_window": None,
            "best_discharge_window": None,
            "forced_shutdown": bool(is_forced),
            "year": year
        })

    hourly_df = pd.DataFrame(rows_hourly).set_index(["date","hour"]).sort_index()
    hourly_df = hourly_df[["price","role","energy_mwh","cashflow_eur","shutdown","charge_len","discharge_len"]]

    daily_df = pd.DataFrame(rows_daily).set_index("date").sort_index()
    daily_df = daily_df[[
        "shutdown","gross_arbitrage_EUR","avg_price_charge","avg_price_discharge",
        "charge_len","discharge_len","charge_start_hour","discharge_start_hour",
        "best_charge_window","best_discharge_window","year","forced_shutdown"
    ]]

    return daily_df, hourly_df

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
                opex = float(opex_year)
            else:
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
    import matplotlib.pyplot as plt
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 5))

    if df_yearly.empty:
        ax.set_title(title)
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        return ax

    years = sorted(df_yearly["year"].unique())
    scenarios = sorted(df_yearly["scenario"].unique())
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

def _season_from_month(m: int) -> str:
    # Northern hemisphere seasons (ES, DE, FI, FR, NL)
    if m in (12, 1, 2):  return "Winter"
    if m in (3, 4, 5):   return "Spring"
    if m in (6, 7, 8):   return "Summer"
    return "Autumn"

# ---------------------------------------------
# UI – Controls
# ---------------------------------------------
st.title("⚡ Earnings Potentials — Results Explorer")


@st.cache_data(show_spinner=False)
def load_prices_full() -> pd.DataFrame:
    """
    Return a DataFrame with ['LocalDatetime','Euro per MWh','Country'] plus
    derived columns: Hour, Day, Month, Year, Season.
    Clips only the UPPER bound to CUTOFF_END so we can include pre-2022 history.
    """
    df = pd.read_parquet("bidding_zone_prices.parquet")
    df = df[['LocalDatetime','Euro per MWh','Country']]
    df["LocalDatetime"] = pd.to_datetime(df["LocalDatetime"], errors="coerce")
    df = df.dropna(subset=["LocalDatetime", "Euro per MWh"])

    # Keep only known countries (same keep-list)
    keep = ["ES", "FI", "NL", "DE", "FR"]
    if "Country" in df.columns:
        df = df[df["Country"].isin(keep)]

    # Clip only the TOP end so we retain pre-2022 data
    df = df[df["LocalDatetime"] <= CUTOFF_END]

    # --- Derived columns ---
    dt = df["LocalDatetime"].dt
    df["Hour"]  = dt.hour
    df["Day"]   = dt.day
    df["Month"] = dt.month
    df["Year"]  = dt.year
    df["Season"] = dt.month.map(_season_from_month)  # uses your helper from above

    return df



with st.container():
    c1, c2, c3, c4 = st.columns([1, 1, 1, 1])

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
            index=1,
            key="country_filter",
            help="DE - Germany, ES - Spain, FI - Finland, FR - France, NL - Netherlands",
        )

        # Cap picker to 11/08/2025
        daterange = st.date_input(
            "Date range",
            value=(date(2022, 1, 1), date(2025, 8, 11)),
            min_value=date(2012, 1, 1),
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

    # --- Column 3: Windows ---
    with c3:
        charge_len_single = st.selectbox(
            "Charging hours (optimized)",
            options=list(range(6, 13)),
            index=2,  # default 8h
        )
        discharge_min, discharge_max = st.select_slider(
            "Discharge window (hours, optimized)",
            options=list(range(6, 13)),
            value=(6, 12),
        )

    # --- Column 4: OPEX + forced + run ---
    with c4:
        opex_year = st.number_input(
            "OPEX per year (EUR)",
            min_value=0.0,
            value=1500.0,
            step=500.0,
        )
        forced_shutdowns = st.number_input(
            "Extra shutdowns per year",
            min_value=0,
            value=0,
            step=1,
        )
        run_btn = st.button("🚀 Run the model", use_container_width=True, type="primary")

st.divider()

# ---------------------------------------------
# RUN
# ---------------------------------------------
daily_all = hourly_all = annual = pd.DataFrame()


tab1, tab2 = st.tabs(["📊 Tab 1: Scenarios Explorer", "📈 Tab 2: Price Analysis"])

with tab1:

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
                    charge_len_options=[int(charge_len_single)],  # single charging hours
                    discharge_len_options=range(int(discharge_min), int(discharge_max)+1),
                    perf_drop_rate_per_year=perf_drop if perf_drop > 0 else None,
                    extra_shutdowns_per_year=int(forced_shutdowns),
                    opex_by_year=None,     # OPEX handled downstream
                )
                return daily_all, hourly_all, annual

            with st.spinner("Running optimization…"):
                daily_all, hourly_all, annual = run_optimizer(df_use)

                # --- Naive baseline ---
                daily_naive, hourly_naive = build_naive_fixed_windows(
                    df_use,
                    price_col="Euro per MWh",
                    ts_col="LocalDatetime",
                    capacity_mwh=capacity_mwh,
                    date_start=date_start, date_end=date_end,
                    charge_start_hour=6,
                    charge_len=12,  # 06–18 charge
                    perf_drop_rate_per_year=perf_drop if perf_drop > 0 else None,
                    extra_shutdowns_per_year=int(forced_shutdowns),
                    rand_seed=42,
                )

                # Merge results (always run; used for comparison in charts)
                if not daily_naive.empty:
                    daily_all = pd.concat([daily_all, daily_naive], axis=0).sort_index() if not daily_all.empty else daily_naive
                    hourly_all = pd.concat([hourly_all, hourly_naive], axis=0).sort_index() if not hourly_all.empty else hourly_naive

                # Optional: flip natural shutdowns off (keep days even if profit==0)
                allow_neg_days = False
                if allow_neg_days and not daily_all.empty:
                    natural_mask = (daily_all["shutdown"] == True) & (~daily_all["forced_shutdown"])
                    daily_all.loc[natural_mask, "shutdown"] = False

    # ---------------------------------------------
    # RESULTS — show only after clicking "Run the model"
    # ---------------------------------------------
    if run_btn and not daily_all.empty:
        # --- SCORECARDS (Avg days, Max revenue, Avg OPEX with proration, Net) ---
        days_per_scen = daily_all.groupby(["charge_len","discharge_len"]).size()
        avg_days = int(days_per_scen.mean()) if len(days_per_scen) else 0

        scen_revenue = daily_all.groupby(["charge_len","discharge_len"])["gross_arbitrage_EUR"].sum()
        if not scen_revenue.empty:
            max_revenue = float(scen_revenue.max())
            winner = scen_revenue.idxmax()  # (Lc, Ld)
            winner_label = f"{winner[0]}c-{winner[1]}d"
            if winner_label == "12c-12d":
                winner_label = "12c-12d (Naive)"
        else:
            max_revenue, winner_label = 0.0, "—"

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

        card(colA, "Avg. Days per Scenario", f"{avg_days:,}")
        card(colB, f"Max Revenue ({winner_label})", f"€ {max_revenue:,.0f}")
        card(colC, "Total OPEX", f"€ {avg_opex:,.0f}")
        card(colD, "Net Profit (Max – Avg OPEX)", f"€ {net_profit:,.0f}")

        st.divider()

        # --- CHARTS: Left = Cumsum | Right = Yearly ---
        st.subheader("Charts")

        st.markdown("""
        <div style="background-color:#2b2b2b; padding:12px; border-radius:8px; border: 1px solid #444;">
        <b>Important notes:</b>
        <ul>
            <li>2025 data until 11 of August.</li>
            <li>Optimization searches best daily charge/discharge windows; the Naive baseline is fixed 06–18 charge / 18–06 discharge.</li>
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
                cumsum_df = cumsum_df.copy()
                # Always show Naive and give it a friendly label
                cumsum_df["scenario"] = cumsum_df["scenario"].replace({"12c-12d": "12c-12d (Naive)"})
                fig = px.line(
                    cumsum_df, x="date", y="cumsum", color="scenario",
                    color_discrete_sequence=company_colors,
                    labels={"date":"Date","cumsum":"Cumulative profit [EUR]","scenario":"Scenario"}
                )
                # Force Naive line style (white)
                for trace in fig.data:
                    if trace.name == "12c-12d (Naive)":
                        trace.line.color = "white"
                fig.update_layout(margin=dict(l=10, r=10, t=10, b=10))
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No data to plot.")

        with right:
            st.subheader("Yearly Net Profit (€) by Scenario")
            yearly_df = build_annual_prorated_net(daily_all, float(opex_year))
            if not yearly_df.empty:
                yearly_df = yearly_df.copy()
                # Friendly label for the baseline
                yearly_df["scenario"] = yearly_df["scenario"].replace({"12c-12d": "12c-12d (Naive)"})

                # Round NET to 2 decimals for plotting & labels
                yearly_df["net"] = yearly_df["net"].astype(float).round(0)

                fig2 = px.bar(
                    yearly_df,
                    x="net",
                    y="year",
                    color="scenario",
                    orientation="h",
                    barmode="group",
                    text=yearly_df["net"],                 # we’ll format this below
                    color_discrete_sequence=company_colors,
                    labels={"net": "Net Profit (€)", "year": "Year", "scenario": "Scenario"},
                    title=""
                )

                # Force Naive color to white
                for trace in fig2.data:
                    if trace.name == "12c-12d (Naive)":
                        trace.marker.color = "white"

                # Format bar labels to € and 2 decimals; nicer hover; axis ticks to 2 decimals + €
                fig2.update_traces(
                    texttemplate="€ %{x:,.0f}",
                    textposition="outside",
                    hovertemplate="<b>Year:</b> %{y}<br><b>Scenario:</b> %{fullData.name}"
                                "<br><b>Net:</b> € %{x:,.0f}<extra></extra>"
                )

                # Let text hang outside the bars without being clipped
                fig2.update_traces(cliponaxis=False)

                # Add space at the end so all labels are visible
                x_max = float(yearly_df["net"].max() + 5000)
                x_min = float(yearly_df["net"].min())
                span = (x_max - x_min) if x_max != x_min else max(abs(x_max), 1.0)
                pad  = 0.10 * span  # 10% padding on both sides

                fig2.update_xaxes(range=[x_min - pad, x_max + pad])

                fig2.update_layout(
                    yaxis=dict(type="category"),
                    xaxis=dict(tickformat=",.2f", tickprefix="€ ", separatethousands=True),
                    margin=dict(l=10, r=10, t=10, b=10),
                )

                st.plotly_chart(fig2, use_container_width=True)
            else:
                st.info("No yearly data available for the selected filters.")

        # ---------------------------------------------
        # DISTRIBUTIONS — Optimized only (no bandwidth control)
        # ---------------------------------------------
        def kde_curve(values: np.ndarray, xmin=0.0, xmax=23.0, step=0.1):
            """Gaussian KDE with Silverman's bandwidth; returns (xs, density)."""
            values = np.asarray(values, dtype=float)
            values = values[~np.isnan(values)]
            n = len(values)
            xs = np.arange(xmin, xmax + step, step)
            if n == 0:
                return xs, np.zeros_like(xs)
            std = np.std(values) if np.std(values) > 0 else 1.0
            h = 1.06 * std * (n ** (-1/5))
            h = max(h, 0.3)  # floor to avoid overly spiky curves
            diffs = (xs[:, None] - values[None, :]) / h
            dens = np.exp(-0.5 * diffs**2).sum(axis=1) / (n * h * np.sqrt(2*np.pi))
            return xs, dens

        # Base DF, exclude Naive (12c-12d) and shutdown days
        dist_df = (
            daily_all.reset_index().rename(columns={"index":"date"})
            .query("shutdown == False")
            .copy()
        )
        dist_df["Scenario"] = dist_df["charge_len"].astype(int).astype(str) + "c-" + dist_df["discharge_len"].astype(int).astype(str) + "d"
        dist_df = dist_df[dist_df["Scenario"] != "12c-12d"]

        # ---------- Section: by Season ----------
        st.divider()
        st.subheader("Start Hour Distributions — by Season (Optimized only)")

        df_season = dist_df.copy()
        df_season["Season"] = df_season["date"].dt.month.apply(_season_from_month)
        season_order = ["Winter","Spring","Summer","Autumn"]
        df_season["Season"] = pd.Categorical(df_season["Season"], categories=season_order, ordered=True)

        s_left, s_right = st.columns(2)

        with s_left:
            st.markdown("**Charge start hour**")
            fig_s_c = px.histogram(
                df_season,
                x="charge_start_hour",
                color="Season",
                barmode="overlay",
                nbins=24,
                histnorm="probability density",
                category_orders={"Season": season_order},
                color_discrete_sequence=company_colors,
                labels={"charge_start_hour":"Hour of day (0–23)"}
            )
            fig_s_c.update_traces(opacity=0.45)
            fig_s_c.update_layout(bargap=0.1, xaxis=dict(dtick=2, range=[0,23]), yaxis_title="Density")
            # KDE overlays
            for i, sname in enumerate(season_order):
                vals = df_season.loc[df_season["Season"] == sname, "charge_start_hour"].to_numpy()
                xs, ys = kde_curve(vals)
                fig_s_c.add_trace(go.Scatter(x=xs, y=ys, mode="lines", name=f"{sname} KDE",
                                            line=dict(width=2, color=company_colors[i % len(company_colors)])))
            st.plotly_chart(fig_s_c, use_container_width=True)

        with s_right:
            st.markdown("**Discharge start hour**")
            fig_s_d = px.histogram(
                df_season,
                x="discharge_start_hour",
                color="Season",
                barmode="overlay",
                nbins=24,
                histnorm="probability density",
                category_orders={"Season": season_order},
                color_discrete_sequence=company_colors,
                labels={"discharge_start_hour":"Hour of day (0–23)"}
            )
            fig_s_d.update_traces(opacity=0.45)
            fig_s_d.update_layout(bargap=0.1, xaxis=dict(dtick=2, range=[0,23]), yaxis_title="Density")
            for i, sname in enumerate(season_order):
                vals = df_season.loc[df_season["Season"] == sname, "discharge_start_hour"].to_numpy()
                xs, ys = kde_curve(vals)
                fig_s_d.add_trace(go.Scatter(x=xs, y=ys, mode="lines", name=f"{sname} KDE",
                                            line=dict(width=2, color=company_colors[i % len(company_colors)])))
            st.plotly_chart(fig_s_d, use_container_width=True)

        # ---------- Section: by Model/Scenario ----------
        st.divider()
        st.subheader("Start Hour Distributions — by Model/Scenario (Optimized only)")

        scen_order = sorted(
            dist_df["Scenario"].unique(),
            key=lambda s: (int(s.split('c-')[0]), int(s.split('c-')[1].rstrip('d')))
        )
        dist_df["Scenario"] = pd.Categorical(dist_df["Scenario"], categories=scen_order, ordered=True)

        m_left, m_right = st.columns(2)

        with m_left:
            st.markdown("**Charge start hour**")
            fig_m_c = px.histogram(
                dist_df,
                x="charge_start_hour",
                color="Scenario",
                barmode="overlay",
                nbins=24,
                histnorm="probability density",
                category_orders={"Scenario": scen_order},
                color_discrete_sequence=company_colors,
                labels={"charge_start_hour":"Hour of day (0–23)"}
            )
            fig_m_c.update_traces(opacity=0.45)
            fig_m_c.update_layout(bargap=0.1, xaxis=dict(dtick=2, range=[0,23]), yaxis_title="Density")
            for i, scen in enumerate(scen_order):
                vals = dist_df.loc[dist_df["Scenario"] == scen, "charge_start_hour"].to_numpy()
                xs, ys = kde_curve(vals)
                fig_m_c.add_trace(go.Scatter(x=xs, y=ys, mode="lines", name=f"{scen} KDE",
                                            line=dict(width=2, color=company_colors[i % len(company_colors)])))
            st.plotly_chart(fig_m_c, use_container_width=True)

        with m_right:
            st.markdown("**Discharge start hour**")
            fig_m_d = px.histogram(
                dist_df,
                x="discharge_start_hour",
                color="Scenario",
                barmode="overlay",
                nbins=24,
                histnorm="probability density",
                category_orders={"Scenario": scen_order},
                color_discrete_sequence=company_colors,
                labels={"discharge_start_hour":"Hour of day (0–23)"}
            )
            fig_m_d.update_traces(opacity=0.45)
            fig_m_d.update_layout(bargap=0.1, xaxis=dict(dtick=2, range=[0,23]), yaxis_title="Density")
            for i, scen in enumerate(scen_order):
                vals = dist_df.loc[dist_df["Scenario"] == scen, "discharge_start_hour"].to_numpy()
                xs, ys = kde_curve(vals)
                fig_m_d.add_trace(go.Scatter(x=xs, y=ys, mode="lines", name=f"{scen} KDE",
                                            line=dict(width=2, color=company_colors[i % len(company_colors)])))
            st.plotly_chart(fig_m_d, use_container_width=True)
        # ---------------------------------------------
        # Tab 1 — Intraday deviation heatmap (PX, PeriodIndex-safe)
        # ---------------------------------------------
        st.divider()
        st.subheader("Intraday Price Deviation from Monthly Average (2022–present)")

        df_heat = prices_df.copy()
        df_heat = df_heat[(df_heat["Country"] == country) &
                        (df_heat["LocalDatetime"] >= pd.Timestamp("2022-01-01"))]
        if date_end is not None:
            df_heat = df_heat[df_heat["LocalDatetime"] <= pd.to_datetime(date_end)]

        if df_heat.empty:
            st.info("No data available for the selected filters.")
        else:
            # Features
            df_heat["MonthYear"] = df_heat["LocalDatetime"].dt.to_period("M")
            df_heat["Hour"]      = df_heat["LocalDatetime"].dt.hour

            # Hourly mean per Month-Year
            hourly_mm = (
                df_heat.groupby(["MonthYear","Hour"])["Euro per MWh"]
                    .mean()
                    .rename("Euro_per_MWh_hourly")
                    .reset_index()
            )

            # Monthly mean, then deviation (hourly - monthly)
            hourly_mm["Euro_per_MWh_monthly"] = (
                hourly_mm.groupby("MonthYear")["Euro_per_MWh_hourly"].transform("mean")
            )
            hourly_mm["Hour_minus_Month_Avg"] = np.round(
                hourly_mm["Euro_per_MWh_hourly"] - hourly_mm["Euro_per_MWh_monthly"], 0
            )

            # Pivot; ensure 0..23 present; sort chronologically
            pivot = hourly_mm.pivot(index="MonthYear", columns="Hour", values="Hour_minus_Month_Avg")
            pivot = pivot.reindex(columns=range(24))
            pivot.index = pivot.index.astype("period[M]").sort_values()
        # ---- Build inputs safe for Plotly JSON ----
        Z = pivot.to_numpy()
        y_labels = [str(p) for p in pivot.index]      # 'YYYY-MM'
        x_labels = [str(h) for h in pivot.columns]    # '0'..'23'

        # Symmetric range so 0 sits in the center
        vmax = float(np.nanmax(np.abs(Z))) if np.isfinite(Z).any() else 1.0

        # Diverging scale with white midpoint
        colorscale = [
            [0.00, company_colors[2]],
            [0.37, company_colors[3]],
            [0.50, "#f0f2f1"],   # white-ish midpoint
            [0.63, company_colors[5]],
            [1.00, company_colors[4]],
        ]

        # Height scales with number of months (≈ row count)
        height_px = max(700, 26 * len(y_labels))

        fig_hm = px.imshow(
            Z,
            x=x_labels,
            y=y_labels,
            color_continuous_scale=colorscale,
            zmin=-vmax, zmax=vmax,         # center around 0
            aspect="auto",
            labels=dict(color="€/MWh vs monthly avg", x="Hour of day", y="Month–Year"),
            title=f"{country} — Intraday Price Deviation from Monthly Average (2022–{int(df_heat['LocalDatetime'].dt.year.max())})"
        )

        # --- Annotations (numbers in each cell) ---
        # Round to integers and hide NaNs
        Z_round = np.where(np.isfinite(Z), np.round(Z).astype(int), np.nan)
        text = [[("" if not np.isfinite(val) else f"{int(val)}") for val in row] for row in Z_round]

        fig_hm.update_traces(
            text=text,
            texttemplate="%{text}",
            textfont=dict(size=10)
        )

        # Cosmetics
        fig_hm.update_layout(
            margin=dict(l=10, r=10, t=60, b=10),
            title_x=0,
            coloraxis_colorbar=dict(ticksuffix=" €"),
            height=height_px
        )
        fig_hm.update_xaxes(dtick=1)

        import calendar

        # Pretty month labels like "January - 2023"
        y_display = [f"{calendar.month_name[p.month]} - {p.year}" for p in pivot.index]

        fig_hm.update_yaxes(
            type="category",
            categoryorder="array",
            categoryarray=y_labels,   # keep chronological order
            tickmode="array",
            tickvals=y_labels,        # every row (the underlying categories)
            ticktext=y_display,       # shown text: "MMMM - YYYY"
            autorange="reversed",
            automargin=True,
            tickfont=dict(size=10)
        )

        st.plotly_chart(fig_hm, use_container_width=True)

    elif run_btn:
        st.info("No results to display. Check data filters and try again.")

with tab2:
    st.title("📈 Price Analysis:")

    df_full = load_prices_full()
    if df_full.empty:
        st.info("No data available to display.")
    else:
        # Use the country selected in Tab 1; fallback to first available just in case
        selected_country = st.session_state.get("country_filter")
        if selected_country is None or selected_country not in df_full["Country"].unique():
            selected_country = sorted(df_full["Country"].dropna().unique().tolist())[0]

        # Small badge to show which country is active (no extra selectbox)
        st.markdown(f"**Country:** `{selected_country}`")

        st.markdown("""
        <div style="background-color:#2b2b2b; padding:12px; border-radius:8px; border: 1px solid #444;">
        <b>Electricity prices: a shifting intraday pattern:</b>
        <ul>
            <li>Europe’s 24-hour price profile has changed markedly with the rapid build-out of renewables and the energy shock following Russia’s invasion of Ukraine. Since 2022, average prices are generally higher, yet midday hours increasingly cluster near €0/MWh when solar output is abundant. Pre-2022, the lowest prices typically appeared overnight; today, the trough more often occurs between ~10:00 and 15:00. The magnitude of this shift varies by country and season (weather conditions), reflecting each system’s generation mix and interconnections.</li>

        """, unsafe_allow_html=True)

        df_c = df_full[df_full["Country"] == selected_country].copy()

        left_col, right_col = st.columns(2)

        # Left: BEFORE 2022-01-01
        with left_col:
            st.subheader("Average Intraday Profile by Season — **Before 2022**")
            df_pre = df_c[df_c["LocalDatetime"] < pd.Timestamp("2022-01-01")]
            if df_pre.empty:
                st.info("No pre-2022 data for the selected country.")
            else:
                agg_pre = (
                    df_pre.groupby(["Season", "Hour"], as_index=False)["Euro per MWh"]
                    .mean()
                    .sort_values(["Season", "Hour"])
                )
                fig_pre = px.line(
                    agg_pre,
                    x="Hour",
                    y="Euro per MWh",
                    color="Season",
                    markers=True,
                    color_discrete_sequence=company_colors,
                    labels={"Hour":"Hour of day", "Euro per MWh":"€/MWh"},
                    title=f"{selected_country} — Average Intraday Profile by Season (≤ 2021)"
                )
                fig_pre.update_layout(margin=dict(l=10, r=10, t=40, b=10), xaxis=dict(dtick=2, range=[0, 23]))
                st.plotly_chart(fig_pre, use_container_width=True)

        # Right: FROM 2022-01-01
        with right_col:
            st.subheader("Average Intraday Profile by Season — **From 2022**")
            df_post = df_c[df_c["LocalDatetime"] >= pd.Timestamp("2022-01-01")]
            if df_post.empty:
                st.info("No 2022+ data for the selected country.")
            else:
                agg_post = (
                    df_post.groupby(["Season", "Hour"], as_index=False)["Euro per MWh"]
                    .mean()
                    .sort_values(["Season", "Hour"])
                )
                y_min = int(df_post["Year"].min())
                y_max = int(df_post["Year"].max())
                fig_post = px.line(
                    agg_post,
                    x="Hour",
                    y="Euro per MWh",
                    color="Season",
                    markers=True,
                    color_discrete_sequence=company_colors,
                    labels={"Hour":"Hour of day", "Euro per MWh":"€/MWh"},
                    title=f"{selected_country} — Average Intraday Profile by Season ({y_min}–{y_max})"
                )
                fig_post.update_layout(margin=dict(l=10, r=10, t=40, b=10), xaxis=dict(dtick=2, range=[0, 23]))
                st.plotly_chart(fig_post, use_container_width=True)

# -------------------------
        # Distributions: Pre vs Post (Plotly Express) — 2 columns
        # -------------------------
        st. divider()

        st.subheader("Price Distribution — Pre vs Post 2022")

        df_v = df_c.dropna(subset=["Euro per MWh", "LocalDatetime"]).copy()
        df_v["Period"] = np.where(
            df_v["LocalDatetime"] < pd.Timestamp("2022-01-01"),
            "2015–2021",
            "2022–2025"
        )
        season_order = ["Winter", "Spring", "Summer", "Autumn"]
        df_v["Season"] = pd.Categorical(df_v["Season"], categories=season_order, ordered=True)

        palette_pre_post = [company_colors[4], company_colors[2]]

        import plotly.graph_objects as go

        # Safety: clamp extremes for readability
        ymax = float(np.nanpercentile(df_v["Euro per MWh"], 99.8))

        pre_color  = company_colors[4]
        post_color = company_colors[2]

        pre_vals  = df_v.loc[df_v["Period"] == "2015–2021", "Euro per MWh"]
        post_vals = df_v.loc[df_v["Period"] == "2022–2025", "Euro per MWh"]

        col1, col2 = st.columns([1, 1.8])

        # ----------------------------
        # --- Column 1: Overlaid density histograms (PX) ---
        with col1:
            fig_hist = px.histogram(
                df_v,
                x="Euro per MWh",
                color="Period",
                nbins=120,
                histnorm="probability density",   # density like seaborn's stat='density'
                barmode="overlay",
                opacity=0.5,
                category_orders={"Period": ["2015–2021", "2022–2025"]},
                color_discrete_sequence=palette_pre_post,
                labels={"Euro per MWh": "€/MWh"},
                title="Price Distribution — Pre vs Post 2022"
            )
            fig_hist.update_layout(
                margin=dict(l=10, r=10, t=60, b=10),
                height=480,
                title_x=0,                 # center title
                legend_title_text=""
            )
            fig_hist.update_yaxes(title_text="Density")

            # Optional: clamp extreme tails for readability (p99.5)
            xmax = float(np.nanpercentile(df_v["Euro per MWh"], 99.5))
            xmin = float(np.nanpercentile(df_v["Euro per MWh"], 0.5))
            fig_hist.update_xaxes(range=[xmin, xmax])

            # Optional: show median lines per period
            med_pre  = float(np.nanmedian(df_v.loc[df_v["Period"]=="2015–2021", "Euro per MWh"]))
            med_post = float(np.nanmedian(df_v.loc[df_v["Period"]=="2022–2025", "Euro per MWh"]))
            fig_hist.add_vline(x=med_pre,  line_dash="dash", line_width=1, line_color=palette_pre_post[0],
                            annotation_text="Median 2015–2021", annotation_position="top left")
            fig_hist.add_vline(x=med_post, line_dash="dash", line_width=1, line_color=palette_pre_post[1],
                            annotation_text="Median 2022–2025", annotation_position="top right")

            st.plotly_chart(fig_hist, use_container_width=True)

        # -------------------------------
        # Column 2 — By Season (split violins per category)
        # -------------------------------
            with col2:
                import plotly.graph_objects as go

                # if you don't already have these defined above:
                # pre_color  = company_colors[4]
                # post_color = company_colors[2]
                # ymax = float(np.nanpercentile(df_v["Euro per MWh"], 99.5))

                fig_season = go.Figure()

                legend_pre_shown = False
                legend_post_shown = False

                for season in season_order:
                    # Pre
                    vals_pre = df_v.loc[
                        (df_v["Season"] == season) & (df_v["Period"] == "2015–2021"),
                        "Euro per MWh"
                    ]
                    if len(vals_pre):
                        fig_season.add_trace(go.Violin(
                            x=[season] * len(vals_pre),
                            y=vals_pre,
                            name="2015–2021",
                            legendgroup="pre",
                            scalegroup="all",
                            side="negative",
                            line_color=pre_color,
                            meanline_visible=True,
                            points=False,
                            jitter=0.05,
                            scalemode="count",
                            width=1,
                            spanmode="soft",
                            showlegend=(not legend_pre_shown)   # show once
                        ))
                        legend_pre_shown = True

                    # Post
                    vals_post = df_v.loc[
                        (df_v["Season"] == season) & (df_v["Period"] == "2022–2025"),
                        "Euro per MWh"
                    ]
                    if len(vals_post):
                        fig_season.add_trace(go.Violin(
                            x=[season] * len(vals_post),
                            y=vals_post,
                            name="2022–2025",
                            legendgroup="post",
                            scalegroup="all",
                            side="positive",
                            line_color=post_color,
                            meanline_visible=True,
                            points=False,
                            jitter=0.05,
                            scalemode="count",
                            width=0.9,
                            spanmode="soft",
                            showlegend=(not legend_post_shown)  # show once
                        ))
                        legend_post_shown = True

                fig_season.update_layout(
                    title="Violin Plots — Pre vs Post 2022 by Season",
                    title_x=0,
                    violingap=0.01,
                    violingroupgap=0.10,
                    violinmode="overlay",
                    margin=dict(l=10, r=10, t=60, b=10),
                    height=480,
                    legend_title_text="",        # legend visible here
                    showlegend=True
                )
                fig_season.update_yaxes(range=[0, ymax], title_text="€/MWh")
                fig_season.update_xaxes(title_text="Season")

                st.plotly_chart(fig_season, use_container_width=True)




        st. divider()
        st.subheader("Intraday Price Profiles by Season — Yearly Lines + Pre/Post Averages")

        import matplotlib.pyplot as plt
        from matplotlib.colors import to_rgb

        def lighten(hex_color: str, amount: float = 0.6):
            r, g, b = to_rgb(hex_color)
            return (r + (1 - r) * amount, g + (1 - g) * amount, b + (1 - b) * amount)

        def pick_years(yrs, k=3):
            """Pick first k and last k years from an array-like."""
            yrs = sorted(set(int(y) for y in yrs))
            if len(yrs) <= 2*k:
                return yrs
            return yrs[:k] + yrs[-k:]

        df_plot = df_c.set_index("LocalDatetime").copy()
        seasons = ["Winter", "Spring", "Summer", "Autumn"]

        PRE_COLOR  = company_colors[4]  # 2015–2021 avg
        POST_COLOR = company_colors[2]  # 2022–2025 avg
        PRE_LIGHT  = lighten(PRE_COLOR, 0.4)
        POST_LIGHT = lighten(POST_COLOR, 0.4)

        fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=False, sharey=False)
        fig.patch.set_alpha(0.0)
        for ax in axes.flat:
            ax.set_facecolor("none")

        for ax, season in zip(axes.flat, seasons):
            grp_pre  = df_plot.loc[(df_plot.index < "2022-01-01") & (df_plot["Season"] == season)]
            grp_post = df_plot.loc[(df_plot.index >= "2022-01-01") & (df_plot["Season"] == season)]

            # Years to annotate
            years_pre  = pick_years(grp_pre["Year"].unique(), k=4)
            years_post = pick_years(grp_post["Year"].unique(), k=2)

            # --- Yearly PRE ---
            for year, grp in grp_pre.groupby("Year"):
                curve = grp.groupby("Hour")["Euro per MWh"].mean()
                if curve.empty: continue
                ax.plot(curve.index, curve.values, color=PRE_LIGHT, lw=0.5, ls="--", label="_nolegend_")
                if int(year) in years_pre:
                    x_last = int(curve.index.max())
                    y_last = float(curve.loc[x_last])
                    ax.annotate(
                        str(int(year)),
                        xy=(x_last, y_last),
                        xytext=(6, 0),
                        textcoords="offset points",
                        ha="left", va="center",
                        fontsize=6, color=PRE_LIGHT
                    )

            # --- Yearly POST ---
            for year, grp in grp_post.groupby("Year"):
                curve = grp.groupby("Hour")["Euro per MWh"].mean()
                if curve.empty: continue
                ax.plot(curve.index, curve.values, color=POST_LIGHT, lw=0.5,  ls="--", label="_nolegend_")
                if int(year) in years_post:
                    x_last = int(curve.index.max())
                    y_last = float(curve.loc[x_last])
                    ax.annotate(
                        str(int(year)),
                        xy=(x_last, y_last),
                        xytext=(6, 0),
                        textcoords="offset points",
                        ha="left", va="center",
                        fontsize=6, color=POST_LIGHT,
                    )

            # --- Bold averages ---
            pre_avg  = grp_pre.groupby("Hour")["Euro per MWh"].mean()
            post_avg = grp_post.groupby("Hour")["Euro per MWh"].mean()
            if not pre_avg.empty:
                ax.plot(pre_avg.index, pre_avg.values, marker="s", lw=2,markersize = 3.5, color=PRE_COLOR, label="2015–2021 avg")
            if not post_avg.empty:
                ax.plot(post_avg.index, post_avg.values, marker="s",markersize = 3.5, lw=2, color=POST_COLOR, label="2022–2025 avg")


            for spine in ax.spines.values():
                spine.set_visible(False)
            ax.set_xlim(-0.5, 26)   # extra space for labels
            ax.grid(True, alpha=0.10)
            ax.set_title(season, fontsize=7)
            ax.set_xlabel("Hour of Day", fontsize=6)
            ax.set_ylabel("€/MWh", fontsize=6)
            ax.tick_params(labelsize=6)
        # global legend
        handles, labels = axes[0, 0].get_legend_handles_labels()
        by_label = {lbl: h for h, lbl in zip(handles, labels) if lbl != "_nolegend_"}
        fig.legend(
            by_label.values(), by_label.keys(),
            loc="upper center", ncol=2, bbox_to_anchor=(0.5, 0.99), fontsize=6
)
        plt.tight_layout(rect=[0, 0.05, 1, 0.95])


        st.pyplot(fig)
