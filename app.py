# app.py
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="📊 Valuation & Portfolio Toolkit", layout="wide")

# ==============================
# Helpers
# ==============================
@st.cache_data
def get_stock_data(stocks, start):
    return yf.download(stocks, start=start, progress=False, threads=True, auto_adjust=False)

def extract_price_matrix(df: pd.DataFrame, tickers: list, prefer_adjusted=True):
    if df is None or df.empty:
        raise ValueError("No price data downloaded. Check tickers or date range.")

    if not isinstance(df.columns, pd.MultiIndex):
        col = "Adj Close" if (prefer_adjusted and "Adj Close" in df.columns) else "Close"
        out = df[[col]].rename(columns={col: tickers[0]}).dropna()
        return out, col

    lvl0 = df.columns.get_level_values(0)
    if "Adj Close" in set(lvl0) and prefer_adjusted:
        sub = df["Adj Close"]
        out = sub[tickers].dropna()
        return out, "Adj Close"
    if "Close" in set(lvl0):
        sub = df["Close"]
        out = sub[tickers].dropna()
        return out, "Close"

    candidates = []
    used = None
    for t in tickers:
        if (t, "Adj Close") in df.columns and prefer_adjusted:
            candidates.append((t, "Adj Close"))
            used = "Adj Close"
        elif (t, "Close") in df.columns:
            candidates.append((t, "Close"))
            used = "Close"
    out = df.loc[:, candidates].copy()
    out.columns = [t for (t, _) in out.columns]
    return out.dropna(), used

def annualize_ret_vol(mean_returns, cov_matrix, weights):
    exp_ret = float(np.sum(weights * mean_returns) * 252.0)
    exp_vol = float(np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252.0))
    return exp_ret, exp_vol

# ==============================
# Efficient Frontier
# ==============================
def plot_efficient_frontier(stock_list, start="2020-01-01", rf=0.02, num_portfolios=5000):
    raw = get_stock_data(stock_list, start)
    try:
        prices, used_col = extract_price_matrix(raw, stock_list, prefer_adjusted=True)
    except Exception:
        prices, used_col = extract_price_matrix(raw, stock_list, prefer_adjusted=False)

    tickers = list(prices.columns)
    rets = prices.pct_change().dropna()
    mean_ret = rets.mean()
    cov = rets.cov()

    # Monte Carlo
    n = len(tickers)
    results = np.zeros((3 + n, num_portfolios))
    for i in range(num_portfolios):
        w = np.random.random(n)
        w /= w.sum()
        p_ret, p_vol = annualize_ret_vol(mean_ret, cov, w)
        sharpe = (p_ret - rf) / p_vol if p_vol > 0 else np.nan
        results[0, i], results[1, i], results[2, i] = p_vol, p_ret, sharpe
        results[3:, i] = w

    cols = ["Volatility", "Return", "Sharpe"] + tickers
    sim = pd.DataFrame(results.T, columns=cols).dropna(subset=["Sharpe"])
    max_sharpe = sim.loc[sim["Sharpe"].idxmax()]
    min_vol = sim.loc[sim["Volatility"].idxmin()]

    # Plot Frontier
    fig, ax = plt.subplots(figsize=(8, 5))
    scat = ax.scatter(sim["Volatility"], sim["Return"], c=sim["Sharpe"],
                      cmap="viridis", s=12, alpha=0.6)
    ax.scatter(max_sharpe["Volatility"], max_sharpe["Return"], c="red", marker="*", s=200, label="Max Sharpe")
    ax.scatter(min_vol["Volatility"], min_vol["Return"], c="blue", marker="*", s=200, label="Min Volatility")
    ax.set_xlabel("Volatility (Risk)")
    ax.set_ylabel("Expected Return")
    ax.set_title("📈 Efficient Frontier")
    ax.legend()
    st.pyplot(fig)

# ==============================
# DCF Section
# ==============================
def dcf_section():
    st.subheader("DCF Inputs")
    ticker = st.text_input("Company Ticker (e.g. TCS.NS)", "TCS.NS")
    shares_out = st.number_input("Shares Outstanding (crores)", value=100.0)
    run = st.button("Run Valuation")
    if not run:
        return
    tkr = yf.Ticker(ticker)
    st.write("**Balance Sheet**")
    st.dataframe(tkr.balance_sheet)
    # Quick intrinsic calc
    fcff = 5000
    wacc = 0.1
    g = 0.03
    years = 5
    pv = sum(fcff * ((1 + g) ** t) / ((1 + wacc) ** t) for t in range(1, years + 1))
    tv = (fcff * (1 + g)) / (wacc - g) / ((1 + wacc) ** years)
    intrinsic = pv + tv
    st.success(f"Intrinsic Value: ₹{intrinsic:,.2f} crores")

# ==============================
# Peer Comparison Section
# ==============================
def peer_comparison():
    st.subheader("Peer Comparison")
    peers = st.text_input("Enter tickers (comma separated)", "TCS.NS,INFY.NS,HDFCBANK.NS")
    peer_list = [p.strip() for p in peers.split(",") if p.strip()]
    rows = []
    for tk in peer_list:
        info = yf.Ticker(tk).info
        rows.append({
            "Ticker": tk,
            "Trailing PE": info.get("trailingPE"),
            "Price to Book": info.get("priceToBook"),
            "Profit Margin": info.get("profitMargins")
        })
    df = pd.DataFrame(rows)
    st.dataframe(df)

# ==============================
# UI with Tabs
# ==============================
st.title("📊 Valuation & Portfolio Toolkit")

tab1, tab2, tab3 = st.tabs(["💰 Company Valuation (DCF)", "📈 Portfolio Analysis", "🏷 Peer Comparison"])

with tab1:
    dcf_section()

with tab2:
    tickers_in = st.text_input("Enter stock tickers (comma separated)", "TCS.NS,INFY.NS,ASIANPAINT.NS")
    stock_list = [t.strip() for t in tickers_in.split(",") if t.strip()]
    if st.button("Run Portfolio Analysis"):
        plot_efficient_frontier(stock_list)

with tab3:
    peer_comparison()
