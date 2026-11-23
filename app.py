import os
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
import streamlit as st

st.set_page_config(
    page_title="IFM Parity & Foreign Investment Dashboard — USD/INR",
    page_icon="💹",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .main .block-container {
        padding-top: 1.4rem;
        padding-bottom: 2.2rem;
        max-width: 1380px;
    }
    div[data-testid="stMetric"] {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-left: 4px solid #0f4c81;
        border-radius: 10px;
        padding: 0.85rem 1rem;
        box-shadow: 0 1px 3px rgba(15, 23, 42, 0.05);
    }
    .status-badge {
        display: inline-block;
        padding: 0.45rem 0.85rem;
        border-radius: 999px;
        font-weight: 600;
        font-size: 0.9rem;
        margin-top: 0.25rem;
    }
    .badge-undervalued {
        background-color: #fef3c7;
        color: #92400e;
        border: 1px solid #f59e0b;
    }
    .badge-overvalued {
        background-color: #dcfce7;
        color: #166534;
        border: 1px solid #22c55e;
    }
    .insight-card {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 1rem 1.2rem;
        margin-top: 0.5rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def load_dataset(uploaded_xlsx=None, uploaded_cpi=None, uploaded_rate=None):
    if uploaded_xlsx is not None:
        xls = pd.ExcelFile(uploaded_xlsx)
        sheet = "IFM_Model_Data" if "IFM_Model_Data" in xls.sheet_names else "Dashboard_Data"
        df = pd.read_excel(xls, sheet_name=sheet)
    elif os.path.exists("USDINR_IFM_Complete_Merged_Workbook.xlsx"):
        xls = pd.ExcelFile("USDINR_IFM_Complete_Merged_Workbook.xlsx")
        sheet = "IFM_Model_Data" if "IFM_Model_Data" in xls.sheet_names else "Dashboard_Data"
        df = pd.read_excel(xls, sheet_name=sheet)
    elif os.path.exists("All_Merged_IFM_Dataset.xlsx"):
        xls = pd.ExcelFile("All_Merged_IFM_Dataset.xlsx")
        sheet = "IFM_Model_Data" if "IFM_Model_Data" in xls.sheet_names else "Dashboard_Data"
        df = pd.read_excel(xls, sheet_name=sheet)
    elif os.path.exists("Macro_Dashboard_Data.xlsx"):
        df = pd.read_excel("Macro_Dashboard_Data.xlsx", sheet_name="Dashboard_Data")
    else:
        return None

    df["Date"] = pd.to_datetime(df["Date"])
    df["YM"] = df["Date"].dt.to_period("M")

    cpi_df = None
    if uploaded_cpi is not None:
        cpi_df = pd.read_csv(uploaded_cpi)
    elif os.path.exists("in-cpi.csv"):
        cpi_df = pd.read_csv("in-cpi.csv")

    if cpi_df is not None and "IN_CPI_Index" not in df.columns:
        cpi_df["YM"] = pd.to_datetime(cpi_df["month"]).dt.to_period("M")
        cpi_df = cpi_df.rename(columns={"index": "IN_CPI_Index"})
        df = df.merge(cpi_df[["YM", "IN_CPI_Index"]], on="YM", how="left")

    rate_df = None
    if uploaded_rate is not None:
        rate_df = pd.read_csv(uploaded_rate)
    elif os.path.exists("bis-policy-rate-in.csv"):
        rate_df = pd.read_csv("bis-policy-rate-in.csv")

    if rate_df is not None and "IN_Policy_Rate_Pct" not in df.columns:
        rate_df["YM"] = pd.to_datetime(rate_df["month"]).dt.to_period("M")
        rate_df = rate_df.rename(columns={"rate": "IN_Policy_Rate_Pct"})
        df = df.merge(rate_df[["YM", "IN_Policy_Rate_Pct"]], on="YM", how="left")

    df = df.sort_values("Date").reset_index(drop=True)

    if "IN_CPI_Index" in df.columns:
        df["IN_CPI_YoY_Pct"] = df["IN_CPI_Index"].pct_change(12) * 100.0
    if "US_CPI_Index" in df.columns:
        df["US_CPI_YoY_Pct_Clean"] = df["US_CPI_Index"].pct_change(12) * 100.0

    valid_base = df.dropna(subset=["USDINR_FRED_Avg", "IN_CPI_Index", "US_CPI_Index"])
    if len(valid_base) > 0:
        b_idx = valid_base.index[0]
        s0 = df.loc[b_idx, "USDINR_FRED_Avg"]
        in0 = df.loc[b_idx, "IN_CPI_Index"]
        us0 = df.loc[b_idx, "US_CPI_Index"]
        df["PPP_Implied_Spot"] = s0 * ((df["IN_CPI_Index"] / in0) / (df["US_CPI_Index"] / us0))
        df["INR_Misalignment_Pct"] = (
            (df["USDINR_FRED_Avg"] - df["PPP_Implied_Spot"]) / df["PPP_Implied_Spot"]
        ) * 100.0
        df["INR_PPP_Status"] = np.where(
            df["USDINR_FRED_Avg"].isna() | df["PPP_Implied_Spot"].isna(),
            None,
            np.where(
                df["USDINR_FRED_Avg"] > df["PPP_Implied_Spot"],
                "Undervalued (INR weaker than PPP)",
                "Overvalued (INR stronger than PPP)",
            ),
        )

    df["Int_Diff_IN_minus_US_PctPts"] = df["IN_Policy_Rate_Pct"] - df["Fed_Funds_Rate_Pct"]
    df["Annual_Spot_Change_Pct"] = df["USDINR_FRED_Avg"].pct_change(12) * 100.0
    df["Carry_Return_Annualized_Pct"] = (
        df["Int_Diff_IN_minus_US_PctPts"] - df["Annual_Spot_Change_Pct"]
    )

    idx = pd.Series(np.nan, index=df.index, dtype=float)
    valid_carry = df["Carry_Return_Annualized_Pct"].dropna()
    if len(valid_carry) > 0:
        first_i = valid_carry.index[0]
        idx.loc[first_i] = 100.0
        for i in range(first_i + 1, len(df)):
            r = df.loc[i, "Carry_Return_Annualized_Pct"]
            prev = idx.loc[i - 1]
            if pd.notna(prev) and pd.notna(r):
                idx.loc[i] = prev * (1.0 + (r / 100.0) / 12.0)
            elif pd.notna(prev):
                idx.loc[i] = prev
    df["Carry_Trade_Index"] = idx
    df["Infl_Diff_IN_minus_US_PctPts"] = df["IN_CPI_YoY_Pct"] - df["US_CPI_YoY_Pct_Clean"]
    return df.drop(columns=["YM"], errors="ignore")


# --- SIDEBAR ---
with st.sidebar:
    st.markdown("## ⚙️ Model & Data Controls")
    up_xlsx = st.file_uploader("Merged Workbook (.xlsx)", type=["xlsx"])
    up_cpi = st.file_uploader("India CPI CSV (in-cpi.csv)", type=["csv"])
    up_rate = st.file_uploader("BIS Policy Rate CSV (bis-policy-rate-in.csv)", type=["csv"])

df = load_dataset(up_xlsx, up_cpi, up_rate)
if df is None:
    st.error("Dataset not found. Place `USDINR_IFM_Complete_Merged_Workbook.xlsx` in the repository root or upload via the sidebar.")
    st.stop()

fx_choices = [c for c in ["USDINR_FRED_Avg", "USDINR_FRED_MonthEnd", "USDINR_RBI_Avg"] if c in df.columns]
selected_fx = st.sidebar.selectbox("Spot Exchange Rate Series (INR / USD)", fx_choices, index=0)

min_d = df["Date"].min().date()
max_d = df["Date"].max().date()
date_range = st.sidebar.slider(
    "Sample Period",
    min_value=min_d,
    max_value=max_d,
    value=(min_d, max_d),
    format="YYYY-MM",
)

fdf = df[(df["Date"].dt.date >= date_range[0]) & (df["Date"].dt.date <= date_range[1])].copy()
fdf["Spot_Selected"] = fdf[selected_fx]

# Recompute PPP & Carry relative to selected spot series if needed
valid_b = fdf.dropna(subset=["Spot_Selected", "IN_CPI_Index", "US_CPI_Index"])
if len(valid_b) > 0:
    b0 = valid_b.index[0]
    s0 = fdf.loc[b0, "Spot_Selected"]
    in0 = fdf.loc[b0, "IN_CPI_Index"]
    us0 = fdf.loc[b0, "US_CPI_Index"]
    fdf["PPP_Implied_Active"] = s0 * ((fdf["IN_CPI_Index"] / in0) / (fdf["US_CPI_Index"] / us0))
    fdf["Misalignment_Active_Pct"] = ((fdf["Spot_Selected"] - fdf["PPP_Implied_Active"]) / fdf["PPP_Implied_Active"]) * 100.0
else:
    fdf["PPP_Implied_Active"] = fdf.get("PPP_Implied_Spot", np.nan)
    fdf["Misalignment_Active_Pct"] = fdf.get("INR_Misalignment_Pct", np.nan)

fdf["Annual_Spot_Change_Active_Pct"] = fdf["Spot_Selected"].pct_change(12) * 100.0
fdf["Carry_Return_Active_Pct"] = fdf["Int_Diff_IN_minus_US_PctPts"] - fdf["Annual_Spot_Change_Active_Pct"]

idx_series = pd.Series(np.nan, index=fdf.index, dtype=float)
valid_c = fdf["Carry_Return_Active_Pct"].dropna()
if len(valid_c) > 0:
    first_idx = valid_c.index[0]
    idx_series.loc[first_idx] = 100.0
    for pos in range(fdf.index.get_loc(first_idx) + 1, len(fdf)):
        i_cur = fdf.index[pos]
        i_prev = fdf.index[pos - 1]
        r_cur = fdf.loc[i_cur, "Carry_Return_Active_Pct"]
        prev_val = idx_series.loc[i_prev]
        if pd.notna(prev_val) and pd.notna(r_cur):
            idx_series.loc[i_cur] = prev_val * (1.0 + (r_cur / 100.0) / 12.0)
        elif pd.notna(prev_val):
            idx_series.loc[i_cur] = prev_val
fdf["Carry_Index_Active"] = idx_series

# --- HEADER KPI BAR ---
ppp_valid = fdf.dropna(subset=["Spot_Selected", "PPP_Implied_Active"])
latest_row = ppp_valid.iloc[-1] if len(ppp_valid) > 0 else fdf.iloc[-1]
is_undervalued = bool(latest_row.get("Spot_Selected", 0) > latest_row.get("PPP_Implied_Active", 0))

h1, h2, h3, h4 = st.columns(4)
h1.metric("Spot Rate (INR / USD)", f"{latest_row['Spot_Selected']:.2f}" if pd.notna(latest_row.get("Spot_Selected")) else "N/A")
h2.metric("PPP-Implied Spot", f"{latest_row['PPP_Implied_Active']:.2f}" if pd.notna(latest_row.get("PPP_Implied_Active")) else "N/A")
h3.metric("PPP Misalignment", f"{latest_row['Misalignment_Active_Pct']:+.2f}%" if pd.notna(latest_row.get("Misalignment_Active_Pct")) else "N/A")
with h4:
    st.markdown("**INR Valuation Badge (vs. PPP)**")
    if is_undervalued:
        st.markdown('<span class="status-badge badge-undervalued">INR UNDERVALUED (Spot &gt; PPP)</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="status-badge badge-overvalued">INR OVERVALUED (Spot &lt; PPP)</span>', unsafe_allow_html=True)

st.markdown("---")

# --- TABS ---
tab1, tab2, tab3, tab4 = st.tabs([
    "1. PPP & Currency Misalignment",
    "2. IRP & Carry Trade Tracker",
    "3. International Fisher Effect",
    "4. Political Risk, Country Risk & Diversification",
])

with tab1:
    st.subheader("Relative Purchasing Power Parity (PPP) & Currency Misalignment")
    if len(ppp_valid) > 0:
        fig_ppp = go.Figure()
        fig_ppp.add_trace(
            go.Scatter(
                x=ppp_valid["Date"],
                y=ppp_valid["Spot_Selected"],
                name=f"Actual Spot ({selected_fx})",
                mode="lines",
                line=dict(color="#0f4c81", width=2.5),
            )
        )
        fig_ppp.add_trace(
            go.Scatter(
                x=ppp_valid["Date"],
                y=ppp_valid["PPP_Implied_Active"],
                name="PPP-Implied Spot Rate",
                mode="lines",
                line=dict(color="#ea580c", width=2.2, dash="dash"),
            )
        )
        fig_ppp.update_layout(
            title="Actual Spot Rate vs. PPP-Implied Spot Rate (INR per 1 USD)",
            xaxis_title="Date",
            yaxis_title="INR per 1 USD",
            hovermode="x unified",
            template="plotly_white",
            height=420,
        )
        st.plotly_chart(fig_ppp, use_container_width=True)

        fig_mis = px.bar(
            ppp_valid,
            x="Date",
            y="Misalignment_Active_Pct",
            title="Currency Misalignment (%) — Positive = INR Weaker than PPP Implied",
            labels={"Misalignment_Active_Pct": "Misalignment (%)"},
            template="plotly_white",
            height=330,
        )
        fig_mis.update_traces(marker_color="#0f4c81")
        st.plotly_chart(fig_mis, use_container_width=True)

        st.markdown(
            '<div class="insight-card"><strong>Parity Takeaway:</strong> Because USD/INR has risen faster than the India–US CPI inflation differential over much of the sample, actual spot trades above PPP-implied levels (positive misalignment), signaling INR undervaluation relative to goods-price parity.</div>',
            unsafe_allow_html=True,
        )

with tab2:
    st.subheader("Interest Rate Parity (IRP) & Uncovered Carry Trade Tracker")
    carry_valid = fdf.dropna(
        subset=["Int_Diff_IN_minus_US_PctPts", "Annual_Spot_Change_Active_Pct", "Carry_Return_Active_Pct"]
    ).copy()
    idx_valid = fdf.dropna(subset=["Carry_Index_Active"]).copy()

    if len(carry_valid) > 0:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Mean Interest Diff (IN - US)", f"{carry_valid['Int_Diff_IN_minus_US_PctPts'].mean():+.2f} pp")
        m2.metric("Mean Annual Spot Change", f"{carry_valid['Annual_Spot_Change_Active_Pct'].mean():+.2f}%")
        m3.metric("Mean Annualized Carry Return", f"{carry_valid['Carry_Return_Active_Pct'].mean():+.2f}%")
        m4.metric(
            "Carry Trade Index (Start = 100)",
            f"{idx_valid['Carry_Index_Active'].iloc[-1]:.2f}" if len(idx_valid) > 0 else "N/A",
        )

        fig_irp = go.Figure()
        fig_irp.add_trace(
            go.Scatter(
                x=carry_valid["Date"],
                y=carry_valid["Int_Diff_IN_minus_US_PctPts"],
                name="Nominal Interest Differential (IN - US, pp)",
                mode="lines",
                line=dict(color="#0f4c81", width=2.2),
            )
        )
        fig_irp.add_trace(
            go.Scatter(
                x=carry_valid["Date"],
                y=carry_valid["Annual_Spot_Change_Active_Pct"],
                name="Annual Spot Rate Change (% YoY)",
                mode="lines",
                line=dict(color="#dc2626", width=2.0),
            )
        )
        fig_irp.update_layout(
            title="Nominal Interest Rate Differential vs. Annual Spot Rate Appreciation/Depreciation",
            xaxis_title="Date",
            yaxis_title="Percentage / Percentage Points",
            hovermode="x unified",
            template="plotly_white",
            height=400,
        )
        st.plotly_chart(fig_irp, use_container_width=True)

        if len(idx_valid) > 0:
            fig_idx = px.line(
                idx_valid,
                x="Date",
                y="Carry_Index_Active",
                title="Uncovered Carry Trade Profitability Index (Long INR / Short USD, Base = 100)",
                labels={"Carry_Index_Active": "Profitability Index"},
                template="plotly_white",
                height=340,
            )
            fig_idx.update_traces(line=dict(color="#15803d", width=2.4))
            st.plotly_chart(fig_idx, use_container_width=True)

with tab3:
    st.subheader("International Fisher Effect (IFE) Diagnostic")
    ife_valid = fdf.dropna(
        subset=["Infl_Diff_IN_minus_US_PctPts", "Int_Diff_IN_minus_US_PctPts"]
    ).copy()

    if len(ife_valid) >= 3:
        slope, intercept, r_val, p_val, std_err = stats.linregress(
            ife_valid["Infl_Diff_IN_minus_US_PctPts"],
            ife_valid["Int_Diff_IN_minus_US_PctPts"],
        )
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Pearson Correlation (r)", f"{r_val:.4f}")
        k2.metric("R-Squared (R²)", f"{r_val**2:.4f}")
        k3.metric("OLS Slope (β)", f"{slope:.4f}")
        k4.metric("P-Value", f"{p_val:.4e}")

        fig_ife = px.scatter(
            ife_valid,
            x="Infl_Diff_IN_minus_US_PctPts",
            y="Int_Diff_IN_minus_US_PctPts",
            trendline="ols",
            labels={
                "Infl_Diff_IN_minus_US_PctPts": "Inflation Differential (India YoY CPI % − US YoY CPI %, pp)",
                "Int_Diff_IN_minus_US_PctPts": "Nominal Interest Differential (India Policy Rate − Fed Funds %, pp)",
            },
            title="Inflation Differential vs. Nominal Interest Rate Differential",
            template="plotly_white",
            height=430,
        )
        st.plotly_chart(fig_ife, use_container_width=True)

        st.markdown(
            f'<div class="insight-card"><strong>IFE Regression Equation:</strong> $\Delta i_t = {intercept:.4f} + ({slope:.4f})\,\Delta \pi_t$ (Std. Error = {std_err:.4f}, $N = {len(ife_valid)}$). Rather than a 1:1 positive pass-through ($\beta \approx 1$), interest differentials compressed during aggressive US rate hikes despite shifting inflation spreads.</div>',
            unsafe_allow_html=True,
        )

with tab4:
    st.subheader("Factors & Risks Influencing Foreign Investments & Portfolio Diversification")
    c_pol, c_cnt, c_div = st.columns(3)

    with c_pol:
        st.markdown("### Political Risk")
        st.write(
            "Capital account restrictions, regulatory/tax changes, trade policies, and geopolitical shifts can alter cross-border capital flows and trigger sharp exchange rate moves."
        )
        st.write(
            "Policy interventions can keep spot exchange rates away from PPP or UIP benchmarks over extended horizons."
        )

    with c_cnt:
        st.markdown("### Country Risk")
        st.write(
            "Sovereign credit quality, fiscal deficits, current account balances, inflation credibility, and monetary policy divergence (RBI vs. US Fed) drive yield spreads and FX risk premia."
        )
        st.write(
            "Unhedged foreign investors face spot volatility risk that can offset interest rate carry."
        )

    with c_div:
        st.markdown("### International Portfolio Diversification")
        st.write(
            "Allocating capital across global markets reduces idiosyncratic country exposure when correlations are imperfect, but currency risk introduces an additional return and volatility component."
        )
        st.write(
            "Portfolio managers balance diversification benefits against currency hedging costs and long-term PPP valuation signals."
        )
