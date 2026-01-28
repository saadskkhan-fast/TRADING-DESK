import streamlit as st
import yfinance as yf
import pandas as pd
import ta
import google.generativeai as genai
import feedparser
import matplotlib.pyplot as plt

# --- 1. CONFIGURATION ---
st.set_page_config(page_title="AI Trader", layout="wide")
st.title("📈 AI Trading Agent (Gemini 2.0 Flash)")

# --- 2. SIDEBAR ---
with st.sidebar:
    st.header("⚙️ Settings")
    api_key = st.text_input("Google API Key", type="password", help="Get free key at aistudio.google.com")
    ticker = st.text_input("Stock Ticker", value="NVDA").upper()
    run_btn = st.button("🚀 Run Analysis", type="primary")
    st.info("Built with Streamlit & Gemini")

# --- 3. DATA ENGINE ---
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
        st.error(f"Error: {e}")
        return None

# --- 4. APP LOGIC ---
if run_btn and api_key:
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-2.0-flash')
    
    with st.spinner(f"Analyzing {ticker}..."):
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
        st.subheader("🤖 AI Debate Room")
        prompt = f"""
        Analyze {ticker} (${data['price']}) with RSI {data['rsi']}.
        NEWS: {data['news']}
        
        Task:
        1. Write a BULL argument (Why buy?).
        2. Write a BEAR argument (Why sell?).
        3. Give a Final Verdict (Buy/Sell/Hold) with a reason.
        """
        response = model.generate_content(prompt)
        st.markdown(response.text)

elif run_btn and not api_key:
    st.warning("⚠️ Please enter your API Key in the sidebar.")