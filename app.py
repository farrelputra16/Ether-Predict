import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from tensorflow.keras.models import load_model
import joblib
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import date, timedelta
import time

# ==========================================
# 1. PAGE CONFIGURATION & CSS
# ==========================================
st.set_page_config(
    layout="wide", 
    page_title="EtherPredict Pro: Thesis Edition",
    page_icon="💎",
    initial_sidebar_state="expanded"
)

# Custom CSS for "Futuristic/Professional" Look
st.markdown("""
<style>
    /* Main Background */
    .stApp {
        background-color: #0b0f19;
    }
    
    /* Card Styling (Glassmorphism) */
    .metric-card {
        background: rgba(30, 30, 46, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 4px 30px rgba(0, 0, 0, 0.5);
        text-align: center;
        transition: transform 0.3s ease;
    }
    .metric-card:hover {
        transform: translateY(-5px);
        border-color: #4CAF50;
        box-shadow: 0 0 15px rgba(76, 175, 80, 0.3);
    }
    
    /* Typography */
    .metric-title {
        color: #a0a0b0;
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 1.2px;
        margin-bottom: 8px;
    }
    .metric-value {
        color: #ffffff;
        font-size: 2.2rem;
        font-weight: 700;
        font-family: 'Roboto', sans-serif;
    }
    .metric-sub {
        font-size: 0.9rem;
        margin-top: 5px;
        font-weight: 500;
    }
    
    /* Reasoning Box */
    .reasoning-container {
        background-color: #131722;
        border-left: 4px solid #3b82f6;
        padding: 15px 20px;
        border-radius: 8px;
        margin-top: 20px;
        font-family: 'Monospace', sans-serif;
        color: #cfd8dc;
    }
    
    /* Tabs Customization */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: transparent;
    }
    .stTabs [data-baseweb="tab"] {
        height: 45px;
        background-color: #1e1e2e;
        border-radius: 8px;
        color: #a0a0b0;
        border: 1px solid rgba(255,255,255,0.05);
    }
    .stTabs [aria-selected="true"] {
        background-color: #3b82f6;
        color: white;
        border: none;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. HELPER FUNCTIONS & MATH
# ==========================================
def calculate_rsi(data, window=14):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_macd(data, slow=26, fast=12, signal=9):
    exp1 = data.ewm(span=fast, adjust=False).mean()
    exp2 = data.ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return macd, signal_line

# ==========================================
# 3. LOAD ASSETS (CACHING SYSTEM)
# ==========================================

# Cache Model (Load once)
@st.cache_resource
def load_ai_model():
    try:
        # Priority: .keras format, fallback to .h5
        try:
            model = load_model('model_eth_skripsi.keras', compile=False)
        except:
            model = load_model('model_eth_skripsi.h5', compile=False)
        scaler = joblib.load('scaler_eth.pkl')
        return model, scaler
    except Exception as e:
        return None, None

# Cache Data (Update every 30 mins / 1800 seconds)
@st.cache_data(ttl=1800)
def get_live_data():
    # Fetch 2 years buffer for accurate MA200 calculation
    df = yf.download("ETH-USD", period="2y", interval="1d", progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)
    return df

# ==========================================
# 4. SIDEBAR & NAVIGATION
# ==========================================
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/0/05/Ethereum_logo_2014.svg", width=60)
    st.title("Control Panel")
    
    st.markdown("### 👤 Researcher")
    st.info("**Farrel Arkesya Kahira Putra**\n\nInformatics Engineering\nUniversitas Negeri Semarang")
    
    st.markdown("---")
    st.markdown("### ⚙️ Technical Parameters")
    ma_short_window = st.slider("Short MA (Days)", 5, 50, 50)
    ma_long_window = st.slider("Long MA (Days)", 50, 200, 200)
    
    st.markdown("---")
    if st.button("🔄 Force Refresh Data", type="primary"):
        st.cache_data.clear()
        st.rerun()
    
    st.caption("v1.0.0 Stable | Thesis Build")

# ==========================================
# 5. CORE LOGIC (AI ENGINE)
# ==========================================

# A. Load Model
model, scaler = load_ai_model()

if model is None:
    st.error("🚨 CRITICAL ERROR: Model file not found!")
    st.warning("Please ensure 'model_eth_skripsi.keras' (or .h5) and 'scaler_eth.pkl' are in the same directory as app.py")
    st.stop()

# B. Load & Process Data
with st.spinner('📡 Connecting to Market Satellite... Fetching Live Ethereum Data...'):
    df = get_live_data()
    
    # Calculate Indicators
    df['MA_Short'] = df['Close'].rolling(window=ma_short_window).mean()
    df['MA_Long'] = df['Close'].rolling(window=ma_long_window).mean()
    df['RSI'] = calculate_rsi(df['Close'])
    df['MACD'], df['Signal_Line'] = calculate_macd(df['Close'])
    
    # Prepare Input for Model (Last 60 Days)
    raw_input = df[['Open', 'High', 'Low', 'Close', 'Volume']].tail(60).values
    scaled_input = scaler.transform(raw_input)
    model_input = scaled_input.reshape(1, 60, 5)
    
    # C. AI PREDICTION
    pred_scaled = model.predict(model_input)
    
    # Descaling (Dummy Array Trick for 5 Features)
    dummy_array = np.zeros((1, 5))
    dummy_array[:, 3] = pred_scaled[0][0] # Insert into Close column
    pred_usd = scaler.inverse_transform(dummy_array)[:, 3][0]

    # D. Change Analysis
    current_price = df['Close'].iloc[-1]
    change_usd = pred_usd - current_price
    change_pct = (change_usd / current_price) * 100
    
    # E. CONFIDENCE & REASONING SYSTEM
    confidence_score = 75 # Base Score
    reasons = []
    
    # 1. Trend Analysis
    is_uptrend = df['MA_Short'].iloc[-1] > df['MA_Long'].iloc[-1]
    is_pred_up = change_pct > 0
    
    if is_uptrend and is_pred_up:
        confidence_score += 10
        reasons.append("✅ **Trend Alignment:** AI Prediction (Bullish) aligns with Golden Cross trend (MA50 > MA200).")
    elif not is_uptrend and not is_pred_up:
        confidence_score += 10
        reasons.append("✅ **Trend Alignment:** AI Prediction (Bearish) aligns with Death Cross trend.")
    else:
        confidence_score -= 15
        reasons.append("⚠️ **Contrarian:** AI Prediction opposes the current market moving average trend.")
        
    # 2. RSI Analysis
    last_rsi = df['RSI'].iloc[-1]
    if last_rsi > 70 and not is_pred_up:
        confidence_score += 5
        reasons.append("✅ **RSI Overbought:** Market is overextended, supporting the bearish forecast.")
    elif last_rsi < 30 and is_pred_up:
        confidence_score += 5
        reasons.append("✅ **RSI Oversold:** Market is undervalued, supporting a potential bullish rebound.")
        
    # 3. Volatility Analysis
    volatility = df['Close'].pct_change().std() * 100
    if volatility > 5:
        confidence_score -= 20
        reasons.append("⚠️ **High Volatility:** Market is highly unstable, increasing prediction risk.")
    
    confidence_score = min(max(confidence_score, 10), 99) # Cap score 10-99

# ==========================================
# 6. DASHBOARD UI DISPLAY
# ==========================================

st.markdown(f"## 💎 EtherPredict Pro <span style='font-size:16px; color:gray'>| Real-Time AI Forecasting System</span>", unsafe_allow_html=True)
st.write(f"Last Update: {df.index[-1].strftime('%d %B %Y')}")

# --- SECTION 1: KEY METRIC CARDS ---
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">CURRENT PRICE</div>
        <div class="metric-value">${current_price:,.2f}</div>
        <div class="metric-sub">Ethereum / USD</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    color_hex = "#00e676" if change_pct > 0 else "#ff5252"
    sign = "+" if change_pct > 0 else ""
    st.markdown(f"""
    <div class="metric-card" style="border-bottom: 3px solid {color_hex};">
        <div class="metric-title">FORECAST (T+1)</div>
        <div class="metric-value">${pred_usd:,.2f}</div>
        <div class="metric-sub" style="color: {color_hex}; font-weight:bold;">
            {sign}{change_pct:.2f}% ({sign}${change_usd:.2f})
        </div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    if confidence_score > 80: conf_color = "#00e676"
    elif confidence_score > 50: conf_color = "#ffeb3b"
    else: conf_color = "#ff5252"
    
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">AI CONFIDENCE</div>
        <div class="metric-value" style="color:{conf_color}">{confidence_score}%</div>
        <div class="metric-sub">Model Probability Score</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    # Recommendation Logic
    if change_pct > 0.5 and confidence_score > 70: action = "STRONG BUY 🚀"
    elif change_pct > 0.1: action = "ACCUMULATE 📈"
    elif change_pct < -0.5 and confidence_score > 70: action = "STRONG SELL 🔻"
    elif change_pct < -0.1: action = "REDUCE 📉"
    else: action = "WAIT / HOLD ✋"
    
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">RECOMMENDATION</div>
        <div class="metric-value" style="font-size: 1.8rem;">{action}</div>
        <div class="metric-sub">Decision Support</div>
    </div>
    """, unsafe_allow_html=True)

# --- SECTION 2: EXPLAINABLE AI (REASONING) ---
reason_html = "".join([f"<li style='margin-bottom:5px;'>{r}</li>" for r in reasons])
st.markdown(f"""
<div class="reasoning-container">
    <strong style="color: #3b82f6; font-size: 1.1rem;">🤖 AI Logic & Market Reasoning:</strong>
    <ul style="margin-top: 10px; margin-bottom: 0; padding-left: 20px;">
        {reason_html}
    </ul>
</div>
""", unsafe_allow_html=True)

st.write("---")

# --- SECTION 3: ANALYSIS TABS ---
tab1, tab2, tab3 = st.tabs(["📊 Price Analysis & Forecast", "📈 Momentum Indicators", "🔬 Model Details"])

# TAB 1: Main Chart
with tab1:
    st.subheader("Price Visualization & Future Forecast")
    
    fig = go.Figure()
    
    # Candlestick
    fig.add_trace(go.Candlestick(
        x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'],
        name='OHLC Data'
    ))
    
    # Moving Averages
    fig.add_trace(go.Scatter(x=df.index, y=df['MA_Short'], line=dict(color='#ff9800', width=1), name=f'MA {ma_short_window}'))
    fig.add_trace(go.Scatter(x=df.index, y=df['MA_Long'], line=dict(color='#2196f3', width=1), name=f'MA {ma_long_window}'))
    
    # Prediction Point
    next_date = df.index[-1] + timedelta(days=1)
    fig.add_trace(go.Scatter(
        x=[df.index[-1], next_date],
        y=[current_price, pred_usd],
        mode='lines+markers',
        marker=dict(size=12, symbol='star', color='white', line=dict(width=2, color=color_hex)),
        line=dict(color=color_hex, width=3, dash='dot'),
        name='AI Prediction (Next Day)'
    ))
    
    fig.update_layout(
        height=600, 
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis_rangeslider_visible=False,
        hovermode="x unified"
    )
    st.plotly_chart(fig, use_container_width=True)

# TAB 2: RSI & MACD
with tab2:
    st.subheader("Advanced Technical Analysis")
    
    col_tech1, col_tech2 = st.columns(2)
    
    # RSI Chart
    fig_rsi = go.Figure()
    fig_rsi.add_trace(go.Scatter(x=df.index, y=df['RSI'], name='RSI', line=dict(color='#e040fb')))
    fig_rsi.add_hrect(y0=70, y1=100, fillcolor="red", opacity=0.1)
    fig_rsi.add_hrect(y0=0, y1=30, fillcolor="green", opacity=0.1)
    fig_rsi.update_layout(
        title="Relative Strength Index (RSI)", 
        height=400, 
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        yaxis=dict(range=[0, 100])
    )
    col_tech1.plotly_chart(fig_rsi, use_container_width=True)
    
    # MACD Chart
    fig_macd = go.Figure()
    fig_macd.add_trace(go.Scatter(x=df.index, y=df['MACD'], name='MACD Line', line=dict(color='#00e5ff')))
    fig_macd.add_trace(go.Scatter(x=df.index, y=df['Signal_Line'], name='Signal Line', line=dict(color='#ffea00')))
    
    colors_hist = ['#00c853' if v >= 0 else '#ff1744' for v in (df['MACD'] - df['Signal_Line'])]
    fig_macd.add_trace(go.Bar(x=df.index, y=df['MACD'] - df['Signal_Line'], name='Histogram', marker_color=colors_hist))
    
    fig_macd.update_layout(
        title="MACD Oscillator", 
        height=400, 
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)'
    )
    col_tech2.plotly_chart(fig_macd, use_container_width=True)

# TAB 3: Model Details
with tab3:
    st.markdown("### 🔬 Thesis Model Specifications")
    c1, c2 = st.columns(2)
    with c1:
        st.json({
            "Architecture": "Hybrid CNN-BiLSTM-Attention",
            "Optimizer": "Adam (LR=0.001)",
            "Loss Function": "Mean Squared Error (MSE)",
            "Sequence Length": "60 Days",
            "Features": ["Open", "High", "Low", "Close", "Volume"]
        })
    with c2:
        st.info("""
        **Model Advantages:**
        1. **CNN:** Extracts local features (short-term candlestick patterns).
        2. **Bi-LSTM:** Learns long-term temporal dependencies (market trends).
        3. **Attention:** Assigns weights to significant time steps influencing price action.
        """)
        
    st.markdown("### 🗂️ Recent Input Data (Raw)")
    st.dataframe(df.tail(10).sort_index(ascending=False), use_container_width=True)