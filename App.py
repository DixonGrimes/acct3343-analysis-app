import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Accrual Red Flags vs Quality-Value", layout="wide")

# ------------------------------------------------------------------
# 1. LOAD DATA  (built once in Colab, saved as universe.csv)
# ------------------------------------------------------------------
EXCLUDE = ["FIG"]  # 60 Second Pitch company: never a recommendation


@st.cache_data
def load():
    d = pd.read_csv("universe.csv")
    return d[~d["ticker"].isin(EXCLUDE)].copy()


raw = load()

# ------------------------------------------------------------------
# 2. MEASURES  (missing stays missing: no fillna(0))
# ------------------------------------------------------------------
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
    return d


df = add_measures(raw)

# ------------------------------------------------------------------
# 3. SIDEBAR FILTERS  (every one feeds the screens below)
# ------------------------------------------------------------------
st.sidebar.header("Universe filters")
industries = sorted(df["industry"].dropna().unique())
pick_ind = st.sidebar.multiselect("Industries to include", industries, default=industries)
min_cap = st.sidebar.slider("Minimum market cap ($M)", 0, 5000, 300, 50)

st.sidebar.header("SELL: accrual red flags")
sta_pct = st.sidebar.slider("STA: flag above this percentile", 50, 99, 80)
snoa_pct = st.sidebar.slider("SNOA: flag above this percentile", 50, 99, 80)
min_flags = st.sidebar.slider("Red flags required", 1, 2, 2)

st.sidebar.header("BUY: profitability + value")
buy_pct = st.sidebar.slider("BUY: combined rank at or above this percentile", 50, 99, 85)
min_gpa = st.sidebar.slider("Minimum GPA", -0.2, 0.6, 0.0, 0.01)
drop_sell = st.sidebar.checkbox("Keep SELL names out of BUY list", value=True)

# ------------------------------------------------------------------
# 4. APPLY FILTERS, THEN SCREEN
# ------------------------------------------------------------------
u = df[df["industry"].isin(pick_ind) & (df["market_cap"] >= min_cap * 1e6)].copy()

# percentile ranks computed WITHIN the filtered universe
for c in ["STA", "SNOA", "GPA", "BM"]:
    u[c + "_pct"] = u[c].rank(pct=True) * 100

u["flag_STA"] = u["STA_pct"] > sta_pct
u["flag_SNOA"] = u["SNOA_pct"] > snoa_pct
u["flags"] = u["flag_STA"].astype(int) + u["flag_SNOA"].astype(int)
u["SELL"] = u["flags"] >= min_flags

u["BUY_score"] = (u["GPA_pct"] + u["BM_pct"]) / 2  # NaN if either is missing
u["BUY"] = (u["BUY_score"] >= buy_pct) & (u["GPA"] >= min_gpa)
if drop_sell:
    u["BUY"] = u["BUY"] & ~u["SELL"]

sell = u[u["SELL"]].sort_values(["flags", "STA"], ascending=False)
buy = u[u["BUY"]].sort_values("BUY_score", ascending=False)

# ------------------------------------------------------------------
# 5. DISPLAY
# ------------------------------------------------------------------
st.title("Accrual Red Flags (SELL) vs Quality-Value (BUY)")
st.caption("Industry: Industrials. Data from Yahoo Finance via yfinance, latest annual statements.")

m1, m2, m3 = st.columns(3)
m1.metric("Companies screened", len(u))
m2.metric("SELL candidates", len(sell))
m3.metric("BUY candidates", len(buy))

left, right = st.columns(2)
with left:
    st.subheader("SELL candidates")
    st.dataframe(
        sell[["ticker", "name", "industry", "flags", "STA", "SNOA"]].round(3),
        hide_index=True, use_container_width=True)
with right:
    st.subheader("BUY candidates")
    st.dataframe(
        buy[["ticker", "name", "industry", "BUY_score", "GPA", "BM"]].round(3),
        hide_index=True, use_container_width=True)

# chart: points recolor as sliders move
u["Group"] = np.where(u["SELL"], "SELL", np.where(u["BUY"], "BUY", "Neither"))
st.subheader("Where the flagged companies sit")
x_axis = st.selectbox("X axis", ["STA", "SNOA", "GPA", "BM"], index=0)
y_axis = st.selectbox("Y axis", ["STA", "SNOA", "GPA", "BM"], index=2)
plot = u.dropna(subset=[x_axis, y_axis])
fig = px.scatter(
    plot, x=x_axis, y=y_axis, color="Group", hover_name="name",
    hover_data=["ticker"],
    color_discrete_map={"SELL": "#c0392b", "BUY": "#1e8449", "Neither": "#b0b7c3"})
st.plotly_chart(fig, use_container_width=True)

# drill-down
st.subheader("Company drill-down")
if len(u):
    t = st.selectbox("Company", u.sort_values("ticker")["ticker"])
    row = u[u["ticker"] == t].iloc[0]
    st.write(f"**{row['name']}** ({t}), {row['industry']}")
    cols = st.columns(4)
    for col, c in zip(cols, ["STA", "SNOA", "GPA", "BM"]):
        v, p = row[c], row[c + "_pct"]
        col.metric(c, "n/a" if pd.isna(v) else f"{v:.3f}",
                   None if pd.isna(p) else f"{p:.0f}th percentile", delta_color="off")

# conclusions: WRITE THESE YOURSELF after you inspect the lists
st.header("Investment conclusions")
st.markdown("""
**Top SELL name(s):** _write your case: what drives the flags, what the cash flow
statement shows, what could prove you wrong._

**Top BUY name(s):** _write your case: why profitability and value are real here,
beyond the score._
""")

with st.expander("Methodology and formulas"):
    st.markdown("""
- **STA** = (Net income - Operating cash flow) / Total assets. High = earnings not backed by cash.
- **SNOA** = (Operating assets - Operating liabilities) / Total assets, where operating assets = Total assets - Cash & ST investments, and operating liabilities = Total assets - Total debt - Equity. High = balance sheet bloated with non-cash assets.
- **GPA** = Gross profit / Total assets. High = better quality.
- **B/M** = Book equity / Market cap. High = cheaper.
- **SELL**: at least *N* of the two accrual measures above their chosen percentile.
- **BUY**: average of GPA and B/M percentile ranks at or above the cutoff. A different analysis from SELL.
- Percentiles are computed within the filtered universe. Missing data stays missing and is excluded from that measure.
- Excluded: Financial Services (accrual measures do not fit banks) and the pitch company.
""")
