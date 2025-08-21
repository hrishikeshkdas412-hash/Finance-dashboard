# app.py (polished frontend; logic unchanged)
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from io import StringIO

st.set_page_config(page_title="📊 Valuation & Portfolio Toolkit", layout="wide")

# ---------- Small CSS polish ----------
st.markdown(
    """
    <style>
    .app-header {font-size:22px; font-weight:700; color:#0f4c75; padding-bottom:6px;}
    .section-title {color:#0f4c75; margin-top:6px;}
    .stButton>button {background-color:#0f4c75; color:white; border-radius:6px;}
    .stButton>button:hover {background-color:#3282b8; color:white;}
    .metric {background: #ffffff; padding: 8px; border-radius:8px; box-shadow: 0 1px 3px rgba(0,0,0,0.06);}
    .small-note {font-size:12px; color:#555;}
    </style>
    """,
    unsafe_allow_html=True,
)

# ==============================
# Helpers (unchanged logic)
# ==============================
@st.cache_data
def get_stock_data(stocks, start):
    """Download OHLCV for one or more tickers (stable layout)."""
    return yf.download(stocks, start=start, progress=False, threads=True, auto_adjust=False)


def extract_price_matrix(df: pd.DataFrame, tickers: list, prefer_adjusted=True):
    """
    Return a clean DataFrame of prices with columns = tickers.
    Works for both single ticker and MultiIndex frames.
    """
    if df is None or df.empty:
        raise ValueError("No price data downloaded. Check tickers or date range.")

    # Single ticker (no MultiIndex)
    if not isinstance(df.columns, pd.MultiIndex):
        col = "Adj Close" if (prefer_adjusted and "Adj Close" in df.columns) else "Close"
        if col not in df.columns:
            raise KeyError("Neither 'Adj Close' nor 'Close' found.")
        out = df[[col]].rename(columns={col: tickers[0]}).dropna()
        return out, col

    # MultiIndex
    lvl0 = df.columns.get_level_values(0)
    # fields-first layout: ["Adj Close"]["TICKER"]
    if "Adj Close" in set(lvl0) and prefer_adjusted:
        sub = df["Adj Close"]
        cols = [t for t in tickers if t in sub.columns]
        if not cols:
            raise KeyError("Requested tickers not found under 'Adj Close'.")
        out = sub[cols].dropna()
        return out, "Adj Close"
    if "Close" in set(lvl0):
        sub = df["Close"]
        cols = [t for t in tickers if t in sub.columns]
        if not cols:
            raise KeyError("Requested tickers not found under 'Close'.")
        out = sub[cols].dropna()
        return out, "Close"

    # ticker-first layout: columns like (TICKER, 'Adj Close')
    candidates = []
    used = None
    for t in tickers:
        if (t, "Adj Close") in df.columns and prefer_adjusted:
            candidates.append((t, "Adj Close"))
            used = "Adj Close"
        elif (t, "Close") in df.columns:
            candidates.append((t, "Close"))
            used = "Close" if used is None else used
    if not candidates:
        raise KeyError("Couldn't locate price columns in DataFrame.")
    out = df.loc[:, candidates].copy()
    out.columns = [t for (t, _) in out.columns]
    return out.dropna(), used


def annualize_ret_vol(mean_returns, cov_matrix, weights):
    exp_ret = float(np.sum(weights * mean_returns) * 252.0)
    exp_vol = float(np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252.0))
    return exp_ret, exp_vol


# ==============================
# Efficient Frontier + Extras (same logic, compact figures + metrics)
# ==============================
def plot_efficient_frontier(stock_list, start="2020-01-01", rf=0.02, num_portfolios=5000):
    raw = get_stock_data(stock_list, start)
    try:
        prices, used_col = extract_price_matrix(raw, stock_list, prefer_adjusted=True)
    except Exception:
        prices, used_col = extract_price_matrix(raw, stock_list, prefer_adjusted=False)

    tickers = list(prices.columns)
    rets = prices.pct_change().dropna()
    if rets.empty:
        st.error("Not enough data to compute returns.")
        return
    mean_ret = rets.mean()
    cov = rets.cov()

    # Monte Carlo simulation
    n = len(tickers)
    results = np.zeros((3 + n, num_portfolios))
    for i in range(num_portfolios):
        w = np.random.random(n)
        w /= w.sum()
        p_ret, p_vol = annualize_ret_vol(mean_ret, cov, w)
        sharpe = (p_ret - rf) / p_vol if p_vol > 0 else np.nan
        results[0, i] = p_vol
        results[1, i] = p_ret
        results[2, i] = sharpe
        results[3:, i] = w

    cols = ["Volatility", "Return", "Sharpe"] + tickers
    sim = pd.DataFrame(results.T, columns=cols).dropna(subset=["Sharpe"])
    if sim.empty:
        st.error("Simulation produced no valid portfolios.")
        return

    max_sharpe = sim.loc[sim["Sharpe"].idxmax()]
    min_vol = sim.loc[sim["Volatility"].idxmin()]

    # show top-line metrics
    st.markdown("<div class='app-header'>Portfolio Overview</div>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    try:
        col1.metric("Max Sharpe (ann.)", f"{max_sharpe['Sharpe']:.2f}")
        col2.metric("Max Sharpe Return (ann.)", f"{max_sharpe['Return']:.2%}")
        col3.metric("Min Volatility (ann.)", f"{min_vol['Volatility']:.2%}")
    except Exception:
        # fallback for display if numeric not available
        col1.metric("Max Sharpe (ann.)", str(max_sharpe.get("Sharpe", "")))
        col2.metric("Max Sharpe Return (ann.)", str(max_sharpe.get("Return", "")))
        col3.metric("Min Volatility (ann.)", str(min_vol.get("Volatility", "")))

    # -------- Chart: Efficient Frontier (compact)
    plt.style.use("seaborn-v0_8-darkgrid")
    fig, ax = plt.subplots(figsize=(6, 4))
    scat = ax.scatter(sim["Volatility"], sim["Return"],
                      c=sim["Sharpe"], cmap="viridis", s=10, alpha=0.6, edgecolors="k", linewidths=0.2)
    cb = plt.colorbar(scat, ax=ax)
    cb.set_label("Sharpe Ratio")
    ax.scatter(max_sharpe["Volatility"], max_sharpe["Return"], c="red", marker="*", s=120,
               edgecolors="black", label="Max Sharpe")
    ax.scatter(min_vol["Volatility"], min_vol["Return"], c="blue", marker="*", s=120,
               edgecolors="black", label="Min Volatility")
    ax.set_xlabel("Volatility (ann.)")
    ax.set_ylabel("Return (ann.)")
    ax.set_title("📈 Efficient Frontier")
    ax.legend()
    st.pyplot(fig, use_container_width=False)
    st.caption(f"Prices used: **{used_col}**")

    # -------- Correlation Heatmap (compact)
    st.subheader("🔗 Correlation Between Stocks")
    figc, axc = plt.subplots(figsize=(5, 3))
    sns.heatmap(rets.corr(), annot=True, cmap="coolwarm", ax=axc)
    st.pyplot(figc, use_container_width=False)

    # -------- Portfolio Weights (compact)
    st.subheader("📊 Optimal Portfolios — Weights")
    w1 = max_sharpe[tickers]
    w2 = min_vol[tickers]
    figw, axes = plt.subplots(1, 2, figsize=(8, 3))
    w1.plot(kind="bar", ax=axes[0], color="green", alpha=0.8)
    axes[0].set_title("Max Sharpe")
    w2.plot(kind="bar", ax=axes[1], color="blue", alpha=0.8)
    axes[1].set_title("Min Volatility")
    for ax in axes:
        ax.set_ylim(0, 1)
    st.pyplot(figw, use_container_width=False)

    # -------- Risk Contribution (compact)
    st.subheader("🧮 Risk Contribution (Variance %)")
    def risk_contrib(weights):
        w = weights.values if isinstance(weights, pd.Series) else np.array(weights)
        port_var = float(w @ cov.values @ w)
        contrib = w * (cov.values @ w)
        return pd.Series(contrib / port_var, index=tickers)

    rc1 = risk_contrib(w1)
    rc2 = risk_contrib(w2)
    fig_rc, ax_rc = plt.subplots(1, 2, figsize=(8, 3))
    ax_rc[0].pie(rc1, labels=tickers, autopct="%1.0f%%", startangle=90)
    ax_rc[0].set_title("Max Sharpe RC")
    ax_rc[1].pie(rc2, labels=tickers, autopct="%1.0f%%", startangle=90)
    ax_rc[1].set_title("Min Vol RC")
    st.pyplot(fig_rc, use_container_width=False)

    # -------- Backtest & Drawdown (compact)
    st.subheader("📈 Backtest & Drawdown")
    w_ms = w1.values
    port_ret = rets @ w_ms
    cum = (1 + port_ret).cumprod()
    eq_w = np.repeat(1/len(tickers), len(tickers))
    cum_eq = (1 + (rets @ eq_w)).cumprod()

    # index (NIFTY 50) for comparison & beta calcs with fallback
    idx_ret = None
    idx_cum = None
    try:
        idx_raw = get_stock_data("^NSEI", rets.index.min())
        idx_close, _ = extract_price_matrix(idx_raw, ["^NSEI"], prefer_adjusted=False)
        idx_ret_all = idx_close.pct_change().dropna().iloc[:, 0]
        idx_ret = idx_ret_all.loc[idx_ret_all.index.intersection(cum.index)]
        idx_cum = (1 + idx_ret).cumprod().reindex(cum.index).ffill()
    except Exception:
        try:
            idx_raw = get_stock_data("^BSESN", rets.index.min())
            idx_close, _ = extract_price_matrix(idx_raw, ["^BSESN"], prefer_adjusted=False)
            idx_ret_all = idx_close.pct_change().dropna().iloc[:, 0]
            idx_ret = idx_ret_all.loc[idx_ret_all.index.intersection(cum.index)]
            idx_cum = (1 + idx_ret).cumprod().reindex(cum.index).ffill()
        except Exception:
            idx_ret = None
            idx_cum = None

    fig_b, ax_b = plt.subplots(figsize=(6, 3))
    ax_b.plot(cum.index, cum.values, label="Max Sharpe Portfolio")
    ax_b.plot(cum_eq.index, cum_eq.values, label="Equal Weight", linestyle="--")
    if idx_cum is not None:
        ax_b.plot(idx_cum.index, idx_cum.values, label="Index (NIFTY/Sensex)", linestyle=":")
    ax_b.set_ylabel("Growth of ₹1")
    ax_b.set_title("Cumulative Returns")
    ax_b.legend(fontsize=8)
    st.pyplot(fig_b, use_container_width=False)

    # drawdown
    roll_max = cum.cummax()
    drawdown = cum / roll_max - 1.0
    max_dd = drawdown.min()
    fig_dd, ax_dd = plt.subplots(figsize=(6, 2))
    ax_dd.plot(drawdown.index, drawdown.values, color="crimson")
    ax_dd.set_title(f"Drawdown (Max: {max_dd:.2%})")
    st.pyplot(fig_dd, use_container_width=False)

    # -------- Rolling Volatility & Rolling Beta (compact)
    st.subheader("📉 Rolling Metrics")
    selected = st.multiselect("Select tickers for rolling charts:", tickers, default=tickers[:min(3, len(tickers))])
    window = st.slider("Rolling window (days)", min_value=20, max_value=120, value=60, step=5, key="rolling_win")

    if len(selected) > 0:
        fig_rv, ax_rv = plt.subplots(figsize=(6, 3))
        for t in selected:
            series = rets[t].rolling(window).std() * np.sqrt(252)
            ax_rv.plot(series.index, series.values, label=t)
        ax_rv.set_title("Rolling Volatility (annualized)")
        ax_rv.legend(fontsize=8)
        st.pyplot(fig_rv, use_container_width=False)

        if idx_ret is not None and not idx_ret.empty:
            fig_rb, ax_rb = plt.subplots(figsize=(6, 3))
            for t in selected:
                cov_roll = rets[t].rolling(window).cov(idx_ret)
                beta_roll = cov_roll / idx_ret.rolling(window).var()
                ax_rb.plot(beta_roll.index, beta_roll.values, label=t)
            ax_rb.axhline(1.0, color="k", linestyle="--", linewidth=0.8)
            ax_rb.set_title("Rolling Beta vs Index")
            ax_rb.legend(fontsize=8)
            st.pyplot(fig_rb, use_container_width=False)
        else:
            st.info("Index data unavailable for rolling beta (NIFTY/Sensex not found).")

    # -------- Summary stats tables & weights (unchanged)
    st.subheader("📊 Optimal Portfolios — Summary")
    def port_summary(row):
        w = row[tickers].values
        r, v = annualize_ret_vol(mean_ret, cov, w)
        s = (r - rf) / v if v > 0 else np.nan
        return pd.Series({"Return": r, "Volatility": v, "Sharpe": s})

    summary = pd.DataFrame({"Max Sharpe": port_summary(max_sharpe),
                            "Min Volatility": port_summary(min_vol)}).T
    st.dataframe(summary.style.format({"Return": "{:.2%}", "Volatility": "{:.2%}", "Sharpe": "{:.2f}"}))

    st.write("**Weights (Max Sharpe)**")
    st.dataframe(w1.to_frame("Weight").style.format("{:.2%}"))
    st.write("**Weights (Min Volatility)**")
    st.dataframe(w2.to_frame("Weight").style.format("{:.2%}"))

    # -------- Downloads (unchanged)
    st.subheader("⬇️ Downloads")
    st.download_button("Download all simulated portfolios (CSV)",
                       data=sim.to_csv(index=False).encode("utf-8"),
                       file_name="efficient_frontier_simulations.csv",
                       mime="text/csv")
    st.download_button("Download Max Sharpe weights (CSV)",
                       data=w1.to_csv(header=["Weight"]).encode("utf-8"),
                       file_name="max_sharpe_weights.csv",
                       mime="text/csv")
    st.download_button("Download Min Volatility weights (CSV)",
                       data=w2.to_csv(header=["Weight"]).encode("utf-8"),
                       file_name="min_vol_weights.csv",
                       mime="text/csv")
    backtest_df = pd.DataFrame({"MaxSharpe": cum, "EqualWeight": cum_eq})
    if idx_cum is not None:
        backtest_df["Index"] = idx_cum
    st.download_button("Download backtest (CSV)",
                       data=backtest_df.to_csv().encode("utf-8"),
                       file_name="backtest_cumulative.csv",
                       mime="text/csv")


# ==============================
# DCF Valuation + Extras (compact figures + metrics)
# ==============================
def dcf_section():
    st.markdown("<div class='app-header'>Company Valuation (DCF)</div>", unsafe_allow_html=True)
    company_ticker = st.text_input("Company Ticker (e.g., ASIANPAINT.NS, HDFCBANK.NS, TCS.NS):", "ASIANPAINT.NS")
    shares_out_crore = st.number_input("Shares Outstanding (in crores)", value=100.0, min_value=0.01)
    run = st.button("Run Valuation", key="run_dcf")
    if not run:
        return

    try:
        tkr = yf.Ticker(company_ticker)

        # Financial statements (unchanged)
        with st.expander("📄 Financial Statements", expanded=False):
            if hasattr(tkr, "balance_sheet") and not tkr.balance_sheet.empty:
                st.write("**Balance Sheet**")
                st.dataframe(tkr.balance_sheet.rename(index=str))
            else:
                st.warning("No balance sheet data.")
            if hasattr(tkr, "financials") and not tkr.financials.empty:
                st.write("**Income Statement**")
                st.dataframe(tkr.financials.rename(index=str))
            else:
                st.warning("No income statement data.")
            if hasattr(tkr, "cashflow") and not tkr.cashflow.empty:
                st.write("**Cashflow Statement**")
                st.dataframe(tkr.cashflow.rename(index=str))
            else:
                st.warning("No cashflow data.")

        # DCF inputs
        st.subheader("DCF Assumptions")
        risk_free_rate = st.number_input("Risk-free Rate", value=0.07, key="rf_dcf")
        beta = st.number_input("Beta", value=1.10, key="beta_dcf")
        market_risk_premium = st.number_input("Market Risk Premium", value=0.05, key="mrp_dcf")
        cost_of_debt = st.number_input("Cost of Debt", value=0.08, key="kd_dcf")
        tax_rate = st.number_input("Tax Rate", value=0.25, key="tax_dcf")
        equity_value = st.number_input("Equity Value (₹ crores)", value=50000.0, key="eqval_dcf")
        debt_value = st.number_input("Debt Value (₹ crores)", value=10000.0, key="debtval_dcf")
        current_fcff = st.number_input("Current FCFF (₹ crores)", value=5000.0, key="fcff_dcf")
        terminal_growth = st.number_input("Terminal Growth Rate", value=0.03, min_value=0.0, max_value=0.10, step=0.005, key="tg_dcf")
        forecast_years = st.number_input("Forecast Years", value=5, min_value=1, max_value=10, step=1, key="n_dcf")

        cost_of_equity = risk_free_rate + beta * market_risk_premium
        wacc = (equity_value / (equity_value + debt_value)) * cost_of_equity + \
               (debt_value / (equity_value + debt_value)) * cost_of_debt * (1 - tax_rate)

        # PV of forecast FCFF
        dcf_value = 0.0
        for t in range(1, int(forecast_years) + 1):
            dcf_value += current_fcff * ((1 + terminal_growth) ** t) / ((1 + wacc) ** t)
        # Terminal value (Gordon)
        eps = 1e-9
        denom = max(wacc - terminal_growth, eps)
        terminal_value = (current_fcff * (1 + terminal_growth)) / denom
        terminal_value = terminal_value / ((1 + wacc) ** int(forecast_years))
        intrinsic_value = dcf_value + terminal_value

        per_share = (intrinsic_value * 1e7) / (shares_out_crore * 1e7)  # ₹ crores → ₹
        per_share_value = float(per_share)

        # Top metrics
        col1, col2, col3 = st.columns(3)
        col1.metric("Intrinsic (₹ cr)", f"₹{intrinsic_value:,.2f}")
        last_price_value = None
        hist = yf.download(company_ticker, period="5y", progress=False, auto_adjust=False)
        try:
            last_price_value = float(hist["Close"].iloc[-1]) if not hist.empty else float(tkr.history(period="1d")["Close"].iloc[-1])
            col2.metric("Market Price (last)", f"₹{last_price_value:,.2f}")
            upside = (per_share_value / last_price_value - 1.0) if last_price_value and last_price_value > 0 else np.nan
            col3.metric("Upside vs Market", f"{upside:.1%}" if not np.isnan(upside) else "N/A")
        except Exception:
            col2.metric("Market Price (last)", "N/A")
            col3.metric("Upside vs Market", "N/A")

        st.success(f"📈 DCF Intrinsic Value: ₹{intrinsic_value:,.2f} crores")
        st.info(f"Intrinsic Value per Share (uses entered shares outstanding): **₹{per_share_value:,.2f}**")

        # Historical price chart (compact)
        st.subheader("📉 Historical Price")
        if hist is not None and not hist.empty:
            figp, axp = plt.subplots(figsize=(6, 3))
            axp.plot(hist.index, hist["Close"], label="Close")
            axp.axhline(per_share_value, color="teal", linestyle="--", label="Intrinsic/Share")
            axp.set_title(f"{company_ticker} — 5Y Price (Close)")
            axp.legend(fontsize=8)
            st.pyplot(figp, use_container_width=False)

        # FCFF forecast chart (compact)
        st.subheader("📊 FCFF Forecast")
        years = list(range(1, int(forecast_years) + 1))
        fcff_forecast = [current_fcff * ((1 + terminal_growth) ** t) for t in years]
        fig_fcff, ax_fcff = plt.subplots(figsize=(5, 3))
        ax_fcff.plot(years, fcff_forecast, marker="o")
        ax_fcff.set_xlabel("Year")
        ax_fcff.set_ylabel("FCFF (₹ crores)")
        st.pyplot(fig_fcff, use_container_width=False)

        # Value breakdown (compact)
        st.subheader("💡 Value Breakdown")
        fig_pie, ax_pie = plt.subplots(figsize=(4, 3))
        ax_pie.pie([dcf_value, terminal_value], labels=["PV of Forecast FCFF", "Terminal Value"],
                   autopct="%1.1f%%", startangle=90, colors=["#66c2a5", "#fc8d62"])
        ax_pie.axis("equal")
        st.pyplot(fig_pie, use_container_width=False)

        # Intrinsic vs Market Price (compact)
        st.subheader("📈 Intrinsic vs Market Price")
        try:
            last_price = hist["Close"].iloc[-1] if not hist.empty else tkr.history(period="1d")["Close"].iloc[-1]
            last_price_value = float(last_price)
        except Exception:
            last_price_value = float(0.0)
        fig_cmp, ax_cmp = plt.subplots(figsize=(4, 3))
        ax_cmp.bar(["Market Price", "Intrinsic/Share"], [last_price_value, per_share_value],
                   color=["#e41a1c", "#377eb8"])
        ax_cmp.set_ylabel("₹ per share")
        st.pyplot(fig_cmp, use_container_width=False)

        # Sensitivity Analysis (compact heatmap)
        st.subheader("📊 Sensitivity Analysis (WACC vs Terminal Growth)")
        wacc_range = np.linspace(max(0.01, wacc - 0.02), wacc + 0.02, 5)
        tg_range = np.linspace(max(0.0, terminal_growth - 0.01), terminal_growth + 0.01, 5)
        heat = np.zeros((len(wacc_range), len(tg_range)))
        for i, w in enumerate(wacc_range):
            for j, g in enumerate(tg_range):
                denom = max(w - g, 1e-6)
                dcf_val = sum(current_fcff * ((1 + g) ** t) / ((1 + w) ** t) for t in range(1, int(forecast_years) + 1))
                tv = (current_fcff * (1 + g)) / denom / ((1 + w) ** int(forecast_years))
                heat[i, j] = dcf_val + tv
        fig_sens, ax_sens = plt.subplots(figsize=(6, 3))
        sns.heatmap(heat, annot=True, fmt=".0f",
                    xticklabels=[f"{x:.2%}" for x in tg_range],
                    yticklabels=[f"{x:.2%}" for x in wacc_range],
                    cmap="YlGnBu", ax=ax_sens)
        ax_sens.set_xlabel("Terminal Growth")
        ax_sens.set_ylabel("WACC")
        st.pyplot(fig_sens, use_container_width=False)

        # Peer comparison (unchanged)
        st.subheader("🏷 Peer Comparison (multiples)")
        peers = st.text_input("Enter peer tickers (comma separated, optional)", "HDFCBANK.NS,INFY.NS,TCS.NS", key="peers")
        peer_list = [p.strip() for p in peers.split(",") if p.strip()]
        if peer_list:
            rows = []
            tickers_for_info = [company_ticker] + peer_list
            for tk in tickers_for_info:
                info = {}
                try:
                    info = tkr.get_info() if tk == company_ticker else yf.Ticker(tk).get_info()
                except Exception:
                    info = {}
                rows.append({
                    "Ticker": tk,
                    "Trailing PE": info.get("trailingPE"),
                    "Price to Book": info.get("priceToBook"),
                    "ROE": info.get("returnOnEquity"),
                    "Profit Margin": info.get("profitMargins")
                })
            df_peers = pd.DataFrame(rows)
            st.dataframe(df_peers)
            if df_peers["Price to Book"].notna().any():
                fig_pb, ax_pb = plt.subplots(figsize=(5, 3))
                df_peers.dropna(subset=["Price to Book"]).set_index("Ticker")["Price to Book"].plot(kind="bar", ax=ax_pb)
                ax_pb.set_title("Price to Book Comparison")
                st.pyplot(fig_pb, use_container_width=False)

    except Exception as e:
        st.error(f"Error: {e}")


# ==============================
# UI / Navigation (sidebar)
# ==============================
st.sidebar.markdown("<h3 class='section-title'>Navigation</h3>", unsafe_allow_html=True)
page = st.sidebar.radio("Go to", ["Home", "Company Valuation (DCF)", "Portfolio Analysis"], index=1)

if page == "Home":
    st.title("📊 Valuation & Portfolio Toolkit")
    st.markdown("""
        Welcome — use the sidebar to pick a section.
        - Company Valuation (DCF): run DCF and sensitivity.
        - Portfolio Analysis: run efficient frontier, backtest & rolling metrics.
    """)
    st.markdown("<div class='small-note'>Pro tip: Use few tickers (2–6) for faster portfolio sim.</div>", unsafe_allow_html=True)

elif page == "Company Valuation (DCF)":
    st.title("🏢 Company Valuation (DCF)")
    dcf_section()

elif page == "Portfolio Analysis":
    st.title("📈 Portfolio Analysis (Efficient Frontier)")
    with st.expander("Portfolio Inputs", expanded=True):
        tickers_in = st.text_input("Enter stock tickers (comma separated):", "ASIANPAINT.NS,HDFCBANK.NS,INFY.NS", key="tickers_in")
        stock_list = [t.strip() for t in tickers_in.split(",") if t.strip()]
        start_date = st.date_input("Start Date", pd.to_datetime("2020-01-01"), key="start_date")
        risk_free_rate = st.number_input("Risk-Free Rate", value=0.02, key="rf_rate")
        num_portfolios = st.number_input("Number of simulated portfolios", value=3000, step=100, min_value=100, key="num_portfolios")
        run_port = st.button("Run Portfolio Analysis", key="run_port")
        if run_port:
            try:
                plot_efficient_frontier(stock_list, start=start_date, rf=risk_free_rate, num_portfolios=int(num_portfolios))
            except Exception as e:
                st.error(f"Error: {e}")
