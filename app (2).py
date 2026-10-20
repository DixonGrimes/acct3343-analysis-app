import math

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Accrual Red Flags vs Quality-Value", layout="wide")

# ------------------------------------------------------------------
# EDIT THIS SECTION: your pitch company and your written conclusions
# ------------------------------------------------------------------
EXCLUDE = ["FIG"]  # 60 Second Pitch company: never a recommendation

CONCLUSIONS = """
**Top SELL name(s):** _Replace this with your own case: which company, what drives its
flags (look at its cash flow statement and 10-K), and what could prove you wrong._

**Top BUY name(s):** _Replace this with your own case: why the profitability and value
are real, beyond the score, and the main risk._
"""

# ------------------------------------------------------------------
# 1. LOAD DATA  (built once in Colab, saved as universe.csv)
# ------------------------------------------------------------------
@st.cache_data
def load():
    d = pd.read_csv("universe.csv")
    return d[~d["ticker"].isin(EXCLUDE)].copy()


raw = load()


# ------------------------------------------------------------------
# 2. MEASURES  (missing stays missing: no fillna(0))
# ------------------------------------------------------------------
def div(a, b):
    """a / b, but NaN when the denominator is zero or missing."""
    return a / b.where(b != 0)


def norm_cdf(x):
    return x.apply(lambda v: np.nan if pd.isna(v) else 0.5 * (1 + math.erf(v / math.sqrt(2))))


PMAN_COLS = [
    "total_revenue", "total_revenue_prev", "receivables", "receivables_prev",
    "gross_profit_prev", "current_assets", "current_assets_prev", "net_ppe",
    "net_ppe_prev", "depreciation", "depreciation_prev", "sga", "sga_prev",
    "current_liabilities", "current_liabilities_prev", "long_term_debt",
    "long_term_debt_prev", "total_assets_prev"]
HAS_PMAN = all(c in raw.columns for c in PMAN_COLS)


def add_measures(data):
    d = data.copy()
    ta = d["total_assets"].where(d["total_assets"] > 0)
    # SELL side: earnings-quality red flags (higher = worse)
    d["STA"] = (d["net_income"] - d["operating_cash_flow"]) / ta
    op_assets = d["total_assets"] - d["cash_and_st_investments"]
    op_liabs = d["total_assets"] - d["total_debt"] - d["stockholders_equity"]
    d["SNOA"] = (op_assets - op_liabs) / ta
    # BUY side: profitability and value (higher = better)
    d["GPA"] = d["gross_profit"] / ta
    mc = d["market_cap"].where(d["market_cap"] > 0)
    d["BM"] = d["stockholders_equity"] / mc

    if HAS_PMAN:  # Beneish 8-variable model, needs two years of statements
        tap = d["total_assets_prev"]
        dsri = div(div(d["receivables"], d["total_revenue"]),
                   div(d["receivables_prev"], d["total_revenue_prev"]))
        gmi = div(div(d["gross_profit_prev"], d["total_revenue_prev"]),
                  div(d["gross_profit"], d["total_revenue"]))
        aqi = div(1 - div(d["current_assets"] + d["net_ppe"], ta),
                  1 - div(d["current_assets_prev"] + d["net_ppe_prev"], tap))
        sgi = div(d["total_revenue"], d["total_revenue_prev"])
        depi = div(div(d["depreciation_prev"], d["depreciation_prev"] + d["net_ppe_prev"]),
                   div(d["depreciation"], d["depreciation"] + d["net_ppe"]))
        sgai = div(div(d["sga"], d["total_revenue"]),
                   div(d["sga_prev"], d["total_revenue_prev"]))
        lvgi = div(div(d["current_liabilities"] + d["long_term_debt"], ta),
                   div(d["current_liabilities_prev"] + d["long_term_debt_prev"], tap))
        tata = d["STA"]
        d["M_score"] = (-4.84 + 0.920 * dsri + 0.528 * gmi + 0.404 * aqi + 0.892 * sgi
                        + 0.115 * depi - 0.172 * sgai + 4.679 * tata - 0.327 * lvgi)
        d["PMAN"] = norm_cdf(d["M_score"])
    return d


df = add_measures(raw)


def rank_pct(s, groups=None):
    """Percentile rank 0-100. Optionally rank within groups (e.g. industry)."""
    r = s.rank(pct=True) if groups is None else s.groupby(groups).rank(pct=True)
    return r * 100


# ------------------------------------------------------------------
# 3. SIDEBAR FILTERS  (every one feeds the screens below)
# ------------------------------------------------------------------
st.sidebar.header("Universe filters")
if "sector" in df.columns and df["sector"].nunique() > 1:
    sectors = sorted(df["sector"].dropna().unique())
    pick_sec = st.sidebar.multiselect("Sectors to include", sectors, default=sectors)
else:
    pick_sec = None
industries = sorted(df["industry"].dropna().unique())
pick_ind = st.sidebar.multiselect("Industries to include", industries, default=industries)
min_cap = st.sidebar.slider("Minimum market cap ($M)", 0, 20000, 300, 50)
within = st.sidebar.checkbox("Rank within each industry (instead of whole universe)", value=False)

st.sidebar.header("SELL: earnings-quality red flags")
sta_pct = st.sidebar.slider("STA: flag above this percentile", 50, 99, 80)
snoa_pct = st.sidebar.slider("SNOA: flag above this percentile", 50, 99, 80)
if HAS_PMAN:
    pman_pct = st.sidebar.slider("Beneish PMAN: flag above this percentile", 50, 99, 80)
n_measures = 3 if HAS_PMAN else 2
min_flags = st.sidebar.slider("Red flags required (N)", 1, n_measures, min(2, n_measures))

st.sidebar.header("BUY: profitability + value")
w_gpa = st.sidebar.slider("Weight on profitability (GPA) vs value (B/M), %", 0, 100, 50, 5)
buy_pct = st.sidebar.slider("BUY: combined rank at or above this percentile", 50, 99, 85)
min_gpa = st.sidebar.slider("Minimum GPA", -0.2, 0.6, 0.0, 0.01)
drop_sell = st.sidebar.checkbox("Keep SELL names out of the BUY list", value=True)

# ------------------------------------------------------------------
# 4. APPLY FILTERS, THEN SCREEN
# ------------------------------------------------------------------
mask = df["industry"].isin(pick_ind) & (df["market_cap"] >= min_cap * 1e6)
if pick_sec is not None:
    mask &= df["sector"].isin(pick_sec)
u = df[mask].copy()

groups = u["industry"] if within else None
measures = ["STA", "SNOA", "GPA", "BM"] + (["PMAN"] if HAS_PMAN else [])
for c in measures:
    u[c + "_pct"] = rank_pct(u[c], groups)

u["flag_STA"] = u["STA_pct"] > sta_pct
u["flag_SNOA"] = u["SNOA_pct"] > snoa_pct
u["flags"] = u["flag_STA"].astype(int) + u["flag_SNOA"].astype(int)
if HAS_PMAN:
    u["flag_PMAN"] = u["PMAN_pct"] > pman_pct
    u["flags"] += u["flag_PMAN"].astype(int)
u["SELL"] = u["flags"] >= min_flags

w = w_gpa / 100
u["BUY_score"] = w * u["GPA_pct"] + (1 - w) * u["BM_pct"]  # NaN if either is missing
u["BUY"] = (u["BUY_score"] >= buy_pct) & (u["GPA"] >= min_gpa)
if drop_sell:
    u["BUY"] = u["BUY"] & ~u["SELL"]

sell = u[u["SELL"]].sort_values(["flags", "STA"], ascending=False)
buy = u[u["BUY"]].sort_values("BUY_score", ascending=False)

# ------------------------------------------------------------------
# 5. DISPLAY
# ------------------------------------------------------------------
st.title("Accrual Red Flags (SELL) vs Quality-Value (BUY)")
st.caption("Universe: US-listed technology companies reporting in USD. Source: Yahoo Finance "
           "via yfinance, latest annual statements. Move any sidebar control and every number "
           "below recalculates.")

with st.expander("How to use this app", expanded=False):
    st.markdown(
        "SELL and BUY use **different analyses**. SELL looks for companies whose reported "
        "earnings are not backed by cash (accrual red flags). BUY looks for companies that are "
        "both profitable and cheap. Use the sidebar to change the universe, the cutoffs, and "
        "how many red flags a company needs before it is flagged.")

m1, m2, m3, m4 = st.columns(4)
m1.metric("Companies screened", len(u))
m2.metric("SELL candidates", len(sell))
m3.metric("BUY candidates", len(buy))
m4.metric("Rank basis", "Within industry" if within else "Whole universe")

sell_cols = ["ticker", "name", "industry", "flags", "STA", "SNOA"] + (["PMAN"] if HAS_PMAN else [])
buy_cols = ["ticker", "name", "industry", "BUY_score", "GPA", "BM"]

left, right = st.columns(2)
with left:
    st.subheader("SELL candidates")
    st.caption(f"Flagged on at least {min_flags} of {n_measures} red flags.")
    st.dataframe(sell[sell_cols].round(3), hide_index=True, use_container_width=True)
    st.download_button("Download SELL list (CSV)", sell[sell_cols].to_csv(index=False),
                       "sell_candidates.csv", "text/csv")
with right:
    st.subheader("BUY candidates")
    st.caption("Ranked on profitability (GPA) and value (B/M); no overlap with SELL measures.")
    st.dataframe(buy[buy_cols].round(3), hide_index=True, use_container_width=True)
    st.download_button("Download BUY list (CSV)", buy[buy_cols].to_csv(index=False),
                       "buy_candidates.csv", "text/csv")

# ---- chart: points recolor as sliders move ----
u["Group"] = np.where(u["SELL"], "SELL", np.where(u["BUY"], "BUY", "Neither"))
st.subheader("Where the flagged companies sit")
c1, c2, c3 = st.columns(3)
x_axis = c1.selectbox("X axis", measures, index=0)
y_axis = c2.selectbox("Y axis", measures, index=measures.index("GPA"))
zoom = c3.checkbox("Zoom to the central 98% of data", value=True)
plot = u.dropna(subset=[x_axis, y_axis])
fig = px.scatter(
    plot, x=x_axis, y=y_axis, color="Group", hover_name="name", hover_data=["ticker", "industry"],
    category_orders={"Group": ["SELL", "BUY", "Neither"]},
    color_discrete_map={"SELL": "#c0392b", "BUY": "#1e8449", "Neither": "#b0b7c3"})
if zoom and len(plot) > 10:
    fig.update_xaxes(range=[plot[x_axis].quantile(0.01), plot[x_axis].quantile(0.99)])
    fig.update_yaxes(range=[plot[y_axis].quantile(0.01), plot[y_axis].quantile(0.99)])
st.plotly_chart(fig, use_container_width=True)

# ---- drill-down ----
st.subheader("Company drill-down")
if len(u):
    t = st.selectbox("Company", u.sort_values("ticker")["ticker"])
    row = u[u["ticker"] == t].iloc[0]
    st.write(f"**{row['name']}** ({t}), {row['industry']}. Status: **{row['Group']}**, "
             f"red flags: {int(row['flags'])} of {n_measures}.")
    cols = st.columns(len(measures))
    for col, c in zip(cols, measures):
        v, p = row[c], row[c + "_pct"]
        med = u[c].median()
        col.metric(c, "n/a" if pd.isna(v) else f"{v:.3f}",
                   None if pd.isna(p) else f"{p:.0f}th percentile", delta_color="off")
        col.caption(f"Universe median: {med:.3f}")
else:
    st.info("No companies match the current filters.")

# ---- facts to help you write conclusions (not conclusions themselves) ----
with st.expander("Top names right now (starting points for your research)"):
    st.write("**Top SELL names:**", ", ".join(sell["ticker"].head(5)) or "none")
    st.write("**Top BUY names:**", ", ".join(buy["ticker"].head(5)) or "none")

st.header("Investment conclusions")
st.markdown(CONCLUSIONS)

# ---- data quality ----
with st.expander("Data quality and coverage"):
    cov = pd.DataFrame({
        "Companies in file": [len(df)] * len(measures),
        "Computable": [int(df[c].notna().sum()) for c in measures],
        "Coverage %": [round(100 * df[c].notna().mean(), 1) for c in measures]},
        index=measures)
    st.dataframe(cov)
    st.markdown(
        f"- {len(df)} companies in the file; {len(u)} pass the current filters.\n"
        "- Missing values stay missing. A company missing a measure cannot be flagged on it, "
        "and is left out of that measure's ranking. Nothing is filled with zero.\n"
        "- Only USD reporters are kept, because book-to-market compares book equity to a "
        "US-dollar market cap.\n"
        + ("" if HAS_PMAN else
           "- Beneish PMAN is switched off because `universe.csv` lacks the prior-year columns."))

with st.expander("Methodology and formulas"):
    st.markdown("""
**SELL analysis: earnings quality (higher = worse)**
- **STA** = (Net income - Operating cash flow) / Total assets. Earnings not backed by cash.
- **SNOA** = (Operating assets - Operating liabilities) / Total assets. Operating assets = Total
  assets - Cash and ST investments; operating liabilities = Total assets - Total debt - Equity.
- **Beneish PMAN** = Normal CDF of the 8-variable M-score:
  M = -4.84 + 0.920 DSRI + 0.528 GMI + 0.404 AQI + 0.892 SGI + 0.115 DEPI - 0.172 SGAI
  + 4.679 TATA - 0.327 LVGI. Needs two years of statements. TATA is the same as STA above.
- **Rule:** flag a company when at least N of the measures exceed their percentile cutoffs.

**BUY analysis: profitability and value (higher = better)**
- **GPA** = Gross profit / Total assets.
- **B/M** = Book equity / Market cap.
- **Rule:** weighted average of the GPA and B/M percentile ranks at or above the cutoff,
  with a minimum GPA. SELL names can be kept out of BUY.

**Method notes**
- Percentiles are computed among companies that pass the sidebar filters, either across the
  whole universe or within each industry (checkbox).
- Value is a weaker signal for technology companies because much of their value is intangible
  and does not appear in book equity. The weight slider lets you tilt toward profitability.
- Excluded: financial companies (accrual measures do not fit banks), non-USD reporters,
  and the pitch company.
""")
