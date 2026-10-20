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

# Optional: add your own company-specific research notes here, e.g.
# CASE_NOTES = {"ABCD": "Receivables grew 40% while revenue grew 10%; ..."}
CASE_NOTES = {}

SELL_CASE = """
**The SELL case: profits that are not backed by cash.** This screen looks for companies whose
reported earnings look better than their cash generation. When net income runs well ahead of
operating cash flow (high STA), when the balance sheet is loaded with non-cash operating assets
(high SNOA), or when the Beneish model estimates elevated odds of earnings manipulation (high
PMAN), reported profits are more likely to reverse in later periods as accruals unwind and the
market re-prices the stock. A company that trips several flags at once is the strongest candidate,
because independent measures agree.

In technology, the usual explanations to check are capitalized software development costs,
receivables growing faster than revenue (sales recognized before cash is collected),
acquisition-related assets, and stock-based compensation. A flag is a reason to investigate, not
proof of a problem, since a fast-growing company can show high accruals for legitimate reasons.
For each top SELL name, the case rests on three things: what is driving the accrual, what the
cash flow statement shows, and what would prove the thesis wrong, such as receivables being
collected on schedule or growth that justifies the working-capital build.
"""

BUY_CASE = """
**The BUY case: profitable businesses at reasonable prices.** This screen looks for companies that
earn a lot of gross profit from their assets (GPA) and trade at a low price relative to book value
(B/M). GPA captures durable, productive profitability, and B/M captures how much the market is
charging for those net assets. Combining the two ranks finds good businesses that are not
expensive, an approach grounded in quality-and-value research. The BUY screen is deliberately
built on different measures from the SELL screen, so a company appears here because of what it
earns and what it costs, not because it merely avoids accrual red flags.

In technology, book-to-market is less informative because much of the value sits in intangibles
that never appear on the balance sheet, so the weight slider lets you tilt the ranking toward
profitability. For each top BUY name, the case rests on whether the margins are sustainable,
whether the low price relative to book value reflects a real discount rather than an asset-light
business model, and the main risks, such as technology obsolescence, customer concentration and
competition.
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
# BACKGROUND: faint stock-chart lines behind the page (text stays readable)
# ------------------------------------------------------------------
def background_css():
    rng = np.random.default_rng(7)
    parts = ['<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1600 900" '
             'preserveAspectRatio="xMidYMid slice">']
    for y in range(100, 900, 160):  # chart gridlines
        parts.append(f'<line x1="0" x2="1600" y1="{y}" y2="{y}" stroke="#8a97a8" '
                     'stroke-width="1" stroke-dasharray="4 10" opacity="0.5"/>')
    specs = [("#2e8b6a", 640, 0.9), ("#3b79b8", 470, 1.1), ("#c0504d", 300, 1.0),
             ("#2e8b6a", 760, 0.8), ("#3b79b8", 200, 1.2)]
    for color, base, vol in specs:
        steps = rng.normal(0.9, 14 * vol, 80)
        steps = np.cumsum(steps)
        ys = base - steps + 0.4 * (steps - steps.mean())
        xs = np.linspace(0, 1600, len(ys))
        pts = " ".join(f"{x:.0f},{max(40, min(870, y)):.0f}" for x, y in zip(xs, ys))
        parts.append(f'<polyline points="{pts}" fill="none" stroke="{color}" '
                     'stroke-width="3" stroke-linejoin="round" stroke-linecap="round"/>')
    parts.append("</svg>")
    import base64
    b64 = base64.b64encode("".join(parts).encode()).decode()
    return f"""
<style>
.stApp::before {{
    content: ""; position: fixed; inset: 0; z-index: 0; pointer-events: none;
    background: url("data:image/svg+xml;base64,{b64}") center / cover no-repeat;
    opacity: 0.16;
}}
@media (prefers-color-scheme: dark) {{ .stApp::before {{ opacity: 0.3; }} }}
[data-testid="stMain"], [data-testid="stMainBlockContainer"], .block-container {{
    position: relative; z-index: 1;
}}
</style>"""


st.markdown(background_css(), unsafe_allow_html=True)

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
st.markdown("##### By Dixon Grimes")
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
st.subheader("Top SELL names")
st.markdown(SELL_CASE)
for _, r in sell.head(3).iterrows():
    line = (f"- **{r['ticker']}** ({r['name']}): {int(r['flags'])} of {n_measures} red flags. "
            f"STA {r['STA']:.3f} ({r['STA_pct']:.0f}th percentile), "
            f"SNOA {r['SNOA']:.3f} ({r['SNOA_pct']:.0f}th percentile)")
    if HAS_PMAN and pd.notna(r["PMAN"]):
        line += f", Beneish PMAN {r['PMAN']:.3f} ({r['PMAN_pct']:.0f}th percentile)"
    st.markdown(line + ".")
    if r["ticker"] in CASE_NOTES:
        st.markdown(f"  Research note: {CASE_NOTES[r['ticker']]}")

st.subheader("Top BUY names")
st.markdown(BUY_CASE)
for _, r in buy.head(3).iterrows():
    st.markdown(f"- **{r['ticker']}** ({r['name']}): combined rank {r['BUY_score']:.0f}th percentile. "
                f"GPA {r['GPA']:.3f} ({r['GPA_pct']:.0f}th percentile), "
                f"B/M {r['BM']:.3f} ({r['BM_pct']:.0f}th percentile).")
    if r["ticker"] in CASE_NOTES:
        st.markdown(f"  Research note: {CASE_NOTES[r['ticker']]}")
st.caption("The names above update with the sidebar filters. They are the current top of each list.")

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
    st.markdown("**Universe**")
    st.markdown(
        f"- **What it is:** {len(df)} US-listed technology companies, drawn from Yahoo Finance's "
        f"stock screener (US region, NYSE and Nasdaq listings, Technology sector) with a market "
        f"capitalization of at least $300 million when the data was pulled. "
        f"They span {df['industry'].nunique()} industries.\n"
        "- **Financial data:** the most recent annual income statement, balance sheet and cash flow "
        "statement from yfinance, plus the prior year for the Beneish measure.\n"
        "- **Excluded:** companies that do not report in US dollars, because book-to-market compares "
        "book equity with a US-dollar market cap; the 60 Second Pitch company, which cannot be a "
        "recommendation; and any company missing a measure, for that measure only.\n"
        "- **Why technology:** accrual measures are meaningful here because of capitalized software, "
        "stock-based compensation and fast-growing receivables, and the sector is large enough for "
        "percentile cutoffs to mean something. The limitation is that book-to-market understates the "
        "value of intangible assets, so value is a weaker signal than profitability.\n"
        f"- **Size:** median market cap ${df['market_cap'].median() / 1e9:.1f}B, "
        f"ranging from ${df['market_cap'].min() / 1e6:,.0f}M to ${df['market_cap'].max() / 1e9:,.0f}B.")
    st.dataframe(df["industry"].value_counts().rename("Companies").to_frame())
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
""")
