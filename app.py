import streamlit as st
import yfinance as yf
import pandas as pd
import ta
import google.generativeai as genai
import feedparser
import matplotlib.pyplot as plt

# --- 1. CONFIGURATION ---
st.set_page_config(page_title="AI Trader", layout="wide")
st.title("⚡️ AI Trading Agent (Fastest Stable Model)")

# --- 2. SIDEBAR ---
with st.sidebar:
    st.header("⚙️ Settings")
    api_key = st.text_input("Google API Key", type="password", help="Get free key at aistudio.google.com")
    ticker = st.text_input("Stock Ticker", value="NVDA").upper()
    run_btn = st.button("🚀 Run Analysis", type="primary")
    st.markdown("---")
    st.caption("Powered by Gemini 1.5 Flash")

# --- 3. DATA ENGINE (NOW WITH CACHING) ---
@st.cache_data(ttl=3600)  # <--- CRITICAL FIX: Caches data for 1 hour to prevent bans
def get_data(ticker):
    """Fetches data and handles the 'MultiIndex' bug automatically."""
    try:
        # Get 6mo history
        df = yf.download(ticker, period="6mo", progress=False)
        if df.empty: return None
        
        # FIX: Flatten columns if they look like ('Close', 'AAPL')
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        # Math Indicators
        df['rsi'] = ta.momentum.rsi(df['Close'], window=14)
        df['ema_50'] = ta.trend.ema_indicator(df['Close'], window=50)
        
        latest = df.iloc[-1]
        
        # News Hack (RSS)
        rss_url = f"https://news.google.com/rss/search?q={ticker}+stock&hl=en-US&gl=US&ceid=US:en"
        feed = feedparser.parse(rss_url)
        headlines = [f"- {entry.title}" for entry in feed.entries[:5]]
        
        return {
            "df": df,
            "price": round(float(latest['Close']), 2),
            "rsi": round(float(latest['rsi']), 2),
            "news": "\n".join(headlines)
        }
    except Exception as e:
        return None

# --- 4. APP LOGIC ---
if run_btn and api_key:
    # SWITCHED TO 1.5 FLASH FOR STABILITY & SPEED
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-1.5-flash-latest')
    
    with st.spinner(f"⚡️ Blasting data to Gemini 1.5 Flash..."):
        data = get_data(ticker)
        
    if data:
        # Top Metrics
        col1, col2, col3 = st.columns(3)
        col1.metric("Price", f"${data['price']}")
        col2.metric("RSI", data['rsi'], delta="Overbought" if data['rsi']>70 else "Neutral")
        col3.metric("Trend", "Bullish" if data['price'] > data['df']['ema_50'].iloc[-1] else "Bearish")
        
        # Chart
        st.subheader("Price Action")
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(data['df'].index, data['df']['Close'], label='Price', color='black')
        ax.plot(data['df'].index, data['df']['ema_50'], label='50 EMA', color='blue', alpha=0.5)
        ax.legend()
        st.pyplot(fig)
        
        # AI Debate
        st.divider()
        st.subheader("🤖 The Debate Room")
        
        prompt = f"""
        You are a Wall Street Veteran. Analyze {ticker} at ${data['price']} with RSI {data['rsi']}.
        
        LATEST NEWS:
        {data['news']}
        
        TASK:
        1. **The Bull Case** 🐮: Why is this going to the moon? (Cite news)
        2. **The Bear Case** 🐻: Why is this a trap? (Cite risks)
        3. **The Verdict** ⚖️: Buy, Sell, or Hold? Give a target price.
        
        Format beautifully with Markdown. Be fast and punchy.
        """
        
        response = model.generate_content(prompt)
        st.markdown(response.text)

elif run_btn and not api_key:
    st.warning("⚠️ Please enter your API Key in the sidebar.")
