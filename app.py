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
    initial_sidebar_state="collapsed"
)

# Custom CSS for Mobile-First & Glassmorphism UI
st.markdown("""
<style>
    /* Main Background */
    .stApp { background-color: #0b0f19; }
    
    /* Card Styling */
    .metric-card {
        background: rgba(30, 30, 46, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 4px 30px rgba(0, 0, 0, 0.5);
        text-align: center;
        transition: transform 0.3s ease;
        margin-bottom: 0px; 
    }
    .metric-card:hover {
        transform: translateY(-5px);
        border-color: #4CAF50;
        box-shadow: 0 0 15px rgba(76, 175, 80, 0.3);
    }
    
    /* Typography */
    .metric-title { color: #a0a0b0; font-size: 0.85rem; text-transform: uppercase; letter-spacing: 1.2px; margin-bottom: 8px; }
    .metric-value { color: #ffffff; font-size: 2.2rem; font-weight: 700; font-family: 'Roboto', sans-serif; }
    .metric-sub { font-size: 0.9rem; margin-top: 5px; font-weight: 500; }
    
    /* Reasoning Box */
    .reasoning-container {
        background-color: #131722;
        border-left: 4px solid #3b82f6;
        padding: 15px 20px;
        border-radius: 8px;
        margin-top: 20px;
        font-family: 'Monospace', sans-serif;
        color: #cfd8dc;
        font-size: 0.95rem;
    }
    
    /* Tabs Customization */
    .stTabs [data-baseweb="tab-list"] { gap: 8px; background-color: transparent; flex-wrap: wrap; }
    .stTabs [data-baseweb="tab"] {
        height: 45px; background-color: #1e1e2e; border-radius: 8px; color: #a0a0b0;
        border: 1px solid rgba(255,255,255,0.05); flex-grow: 1; 
    }
    .stTabs [aria-selected="true"] { background-color: #3b82f6; color: white; border: none; }

    /* Responsive Adjustments */
    @media only screen and (max-width: 768px) {
        .metric-card { margin-bottom: 15px !important; padding: 15px; }
        .metric-value { font-size: 1.8rem !important; }
        .block-container { padding-top: 2rem !important; }
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

# Logic to determine AI Confidence Score
def get_confidence_score(row, pred_change_pct):
    score = 75 # Base Score
    
    # 1. Trend Alignment
    is_uptrend = row['MA_Short'] > row['MA_Long']
    is_pred_up = pred_change_pct > 0
    
    if is_uptrend == is_pred_up: score += 10
    else: score -= 15
    
    # 2. RSI Checks
    if row['RSI'] > 70 and not is_pred_up: score += 5
    elif row['RSI'] < 30 and is_pred_up: score += 5
    
    # 3. Volatility Check
    if row['Volatility'] > 5: score -= 20
    
    return min(max(score, 10), 99)

# ==========================================
# 3. LOAD ASSETS (CACHING SYSTEM)
# ==========================================
@st.cache_resource
def load_ai_model():
    try:
        try: model = load_model('model_eth_skripsi.keras', compile=False)
        except: model = load_model('model_eth_skripsi.h5', compile=False)
        scaler = joblib.load('scaler_eth.pkl')
        return model, scaler
    except Exception as e: return None, None

@st.cache_data(ttl=1800)
def get_live_data():
    # Fetch 2 years (730 days) buffer for accurate backtesting & indicators
    df = yf.download("ETH-USD", period="730d", interval="1d", progress=False)
    if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.droplevel(1)
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
    
    st.caption("v2.0.0 Backtest Engine | Thesis Build")

# ==========================================
# 5. CORE LOGIC (AI ENGINE)
# ==========================================

model, scaler = load_ai_model()
if model is None:
    st.error("🚨 CRITICAL ERROR: Model file not found! Please ensure .h5/.keras and .pkl files are present.")
    st.stop()

with st.spinner('📡 Connecting to Market Satellite...'):
    df_full = get_live_data()
    
    # Pre-calculate Indicators for the whole dataframe
    df_full['MA_Short'] = df_full['Close'].rolling(window=ma_short_window).mean()
    df_full['MA_Long'] = df_full['Close'].rolling(window=ma_long_window).mean()
    df_full['RSI'] = calculate_rsi(df_full['Close'])
    df_full['MACD'], df_full['Signal_Line'] = calculate_macd(df_full['Close'])
    df_full['Volatility'] = df_full['Close'].pct_change().std() * 100
    
    # Prepare data for Today's Prediction
    df = df_full.copy()
    raw_input = df[['Open', 'High', 'Low', 'Close', 'Volume']].tail(60).values
    scaled_input = scaler.transform(raw_input)
    model_input = scaled_input.reshape(1, 60, 5)
    
    # Predict Today
    pred_scaled = model.predict(model_input)
    dummy_array = np.zeros((1, 5))
    dummy_array[:, 3] = pred_scaled[0][0] 
    pred_usd = scaler.inverse_transform(dummy_array)[:, 3][0]

    current_price = df['Close'].iloc[-1]
    change_usd = pred_usd - current_price
    change_pct = (change_usd / current_price) * 100
    
    # Live Confidence Score
    last_row = df.iloc[-1]
    confidence_score = get_confidence_score(last_row, change_pct)
    
    # Reasoning Logic
    reasons = []
    is_uptrend = last_row['MA_Short'] > last_row['MA_Long']
    is_pred_up = change_pct > 0
    if is_uptrend and is_pred_up: reasons.append("✅ **Trend Alignment:** AI Bullish + Golden Cross.")
    elif not is_uptrend and not is_pred_up: reasons.append("✅ **Trend Alignment:** AI Bearish + Death Cross.")
    else: reasons.append("⚠️ **Contrarian:** AI signal opposes trend.")
    
    if last_row['RSI'] > 70 and not is_pred_up: reasons.append("✅ **RSI Overbought:** Supports bearish signal.")
    if last_row['RSI'] < 30 and is_pred_up: reasons.append("✅ **RSI Oversold:** Supports bullish signal.")

# ==========================================
# 6. DASHBOARD UI DISPLAY
# ==========================================

st.markdown(f"## 💎 EtherPredict Pro <span style='font-size:16px; color:gray'>| Real-Time AI Forecasting System</span>", unsafe_allow_html=True)
st.write(f"Last Update: {df.index[-1].strftime('%d %B %Y')}")

# --- METRIC CARDS ---
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown(f"""<div class="metric-card"><div class="metric-title">CURRENT PRICE</div><div class="metric-value">${current_price:,.2f}</div></div>""", unsafe_allow_html=True)
with col2:
    color_hex = "#00e676" if change_pct > 0 else "#ff5252"
    sign = "+" if change_pct > 0 else ""
    st.markdown(f"""<div class="metric-card" style="border-bottom: 3px solid {color_hex};"><div class="metric-title">FORECAST (T+1)</div><div class="metric-value">${pred_usd:,.2f}</div><div class="metric-sub" style="color:{color_hex}">{sign}{change_pct:.2f}%</div></div>""", unsafe_allow_html=True)
with col3:
    conf_color = "#00e676" if confidence_score > 75 else "#ffeb3b" if confidence_score > 50 else "#ff5252"
    st.markdown(f"""<div class="metric-card"><div class="metric-title">AI CONFIDENCE</div><div class="metric-value" style="color:{conf_color}">{confidence_score}%</div></div>""", unsafe_allow_html=True)
with col4:
    if change_pct > 0.5 and confidence_score > 70: action = "STRONG BUY 🚀"
    elif change_pct > 0.1: action = "ACCUMULATE 📈"
    elif change_pct < -0.5 and confidence_score > 70: action = "STRONG SELL 🔻"
    elif change_pct < -0.1: action = "REDUCE 📉"
    else: action = "WAIT ✋"
    st.markdown(f"""<div class="metric-card"><div class="metric-title">RECOMMENDATION</div><div class="metric-value" style="font-size: 1.8rem;">{action}</div></div>""", unsafe_allow_html=True)

# --- REASONING BOX ---
reason_html = "".join([f"<li style='margin-bottom:5px;'>{r}</li>" for r in reasons])
st.markdown(f"""<div class="reasoning-container"><strong style="color:#3b82f6;">🤖 AI Logic & Market Reasoning:</strong><ul style="margin:5px 0 0 20px;">{reason_html}</ul></div>""", unsafe_allow_html=True)
st.write("---")

# --- TABS LAYOUT ---
tab1, tab2, tab3, tab4 = st.tabs(["📊 Price Forecast", "📈 Momentum Indicators", "🔬 Model Specs", "🛠️ Backtest Simulator"])

with tab1: # Chart Tab
    st.subheader("Price Visualization & Future Forecast")
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='OHLC Data'))
    fig.add_trace(go.Scatter(x=df.index, y=df['MA_Short'], line=dict(color='#ff9800', width=1), name=f'MA {ma_short_window}'))
    fig.add_trace(go.Scatter(x=df.index, y=df['MA_Long'], line=dict(color='#2196f3', width=1), name=f'MA {ma_long_window}'))
    
    next_date = df.index[-1] + timedelta(days=1)
    fig.add_trace(go.Scatter(x=[df.index[-1], next_date], y=[current_price, pred_usd], mode='lines+markers', marker=dict(size=12, symbol='star', color='white', line=dict(width=2, color=color_hex)), line=dict(color=color_hex, width=3, dash='dot'), name='AI Prediction'))
    
    # Legend at bottom for mobile responsiveness
    fig.update_layout(height=550, template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', xaxis_rangeslider_visible=False, legend=dict(orientation="h", y=-0.15, x=0.5, xanchor="center"), margin=dict(b=50))
    st.plotly_chart(fig, use_container_width=True)

with tab2: # Indicators Tab
    st.subheader("Advanced Technical Analysis")
    c1, c2 = st.columns(2)
    
    # RSI
    fig_rsi = go.Figure()
    fig_rsi.add_trace(go.Scatter(x=df.index, y=df['RSI'], line=dict(color='#e040fb'), name='RSI'))
    fig_rsi.add_hrect(y0=70, y1=100, fillcolor="red", opacity=0.1); fig_rsi.add_hrect(y0=0, y1=30, fillcolor="green", opacity=0.1)
    fig_rsi.update_layout(height=350, title="Relative Strength Index (RSI)", template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', margin=dict(l=10, r=10, t=30, b=10))
    c1.plotly_chart(fig_rsi, use_container_width=True)
    
    # MACD
    fig_macd = go.Figure()
    fig_macd.add_trace(go.Scatter(x=df.index, y=df['MACD'], line=dict(color='#00e5ff'), name='MACD'))
    fig_macd.add_trace(go.Scatter(x=df.index, y=df['Signal_Line'], line=dict(color='#ffea00'), name='Signal'))
    colors_hist = ['#00c853' if v >= 0 else '#ff1744' for v in (df['MACD'] - df['Signal_Line'])]
    fig_macd.add_trace(go.Bar(x=df.index, y=df['MACD'] - df['Signal_Line'], marker_color=colors_hist, name='Hist'))
    fig_macd.update_layout(height=350, title="MACD Oscillator", template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', margin=dict(l=10, r=10, t=30, b=10))
    c2.plotly_chart(fig_macd, use_container_width=True)

with tab3: # Detailed Model Specs Tab
    st.header("🔬 Model Architecture Specifications")
    st.caption("Technical details of the Hybrid Deep Learning model used in this research.")
    
    col_spec1, col_spec2 = st.columns(2)
    
    with col_spec1:
        st.subheader("⚙️ Technical Parameters")
        st.json({
            "Model Architecture": "Hybrid CNN-BiLSTM-Attention",
            "Optimizer": "Adam (Adaptive Moment Estimation)",
            "Learning Rate": 0.001,
            "Loss Function": "Mean Squared Error (MSE)",
            "Lookback Window": "60 Days (Historical Data)",
            "Input Features": ["Open", "High", "Low", "Close", "Volume"]
        })
    
    with col_spec2:
        st.subheader("🧠 Architecture Advantages")
        st.info("""
        **Why is this Model Superior?**
        
        1. **CNN (Convolutional Neural Network):** Specialized in extracting *local features* and short-term patterns from price movements (such as candlestick patterns).
           
        2. **Bi-LSTM (Bidirectional LSTM):** Captures long-term temporal dependencies (trends) from two directions: past to future (Forward) and vice versa (Backward), ensuring no trend information is overlooked.
           
        3. **Attention Mechanism:** Assigns dynamic "weights" to specific days that have the most significant impact on the price, enhancing prediction accuracy during high volatility periods.
        """)
        
    st.markdown("---")
    st.subheader("🗂️ Input Data Sample (Live Fetching)")
    st.dataframe(df.tail(10).sort_index(ascending=False), use_container_width=True)

with tab4: # Backtest Simulator Tab
    st.header("🛠️ Backtest Simulator (Sandbox)")
    st.caption("Test your trading strategy using AI predictions on historical data.")
    
    # 1. Backtest Inputs
    b_col1, b_col2, b_col3 = st.columns(3)
    with b_col1:
        initial_capital = st.number_input("Initial Capital (USD)", value=10000, step=100)
        backtest_days = st.slider("Backtest Duration (Days)", 30, 365, 90)
    with b_col2:
        min_confidence = st.slider("Min. AI Confidence (%)", 0, 100, 70)
        risk_per_trade = st.slider("Risk per Trade (%)", 1, 100, 20)
    with b_col3:
        stop_loss_pct = st.number_input("Stop Loss (%)", value=2.0)
        take_profit_pct = st.number_input("Take Profit (%)", value=4.0)

    # 2. Run Button
    if st.button("🚀 Run Backtest Simulation", type="primary"):
        with st.spinner("⏳ Running time machine... Processing historical data..."):
            
            # Data Preparation for Batch Prediction
            sim_data = df_full.copy()
            total_rows = len(sim_data)
            start_idx = total_rows - backtest_days - 60
            
            if start_idx < 0:
                st.error(f"Insufficient Data! Max backtest: {total_rows-60} days.")
            else:
                # Simulation Loop Initialization
                balance = initial_capital
                trade_log = []
                equity_curve = [initial_capital]
                dates = []
                
                # Setup Batch Inputs to optimize speed
                simulation_range = range(total_rows - backtest_days, total_rows - 1)
                batch_inputs = []
                valid_indices = []
                
                for i in simulation_range:
                    history_window = sim_data.iloc[i-60:i][['Open', 'High', 'Low', 'Close', 'Volume']].values
                    scaled_window = scaler.transform(history_window)
                    batch_inputs.append(scaled_window)
                    valid_indices.append(i)
                
                # BATCH PREDICTION (Faster than looping)
                batch_inputs = np.array(batch_inputs)
                batch_preds_scaled = model.predict(batch_inputs, verbose=0)
                
                # Process Results
                for idx, i in enumerate(valid_indices):
                    current_date = sim_data.index[i]
                    row = sim_data.iloc[i]
                    next_day_row = sim_data.iloc[i+1]
                    
                    # Descale Prediction
                    dummy = np.zeros((1, 5))
                    dummy[:, 3] = batch_preds_scaled[idx][0]
                    pred_price = scaler.inverse_transform(dummy)[:, 3][0]
                    
                    # Logic Calculation
                    pct_change_pred = (pred_price - row['Close']) / row['Close'] * 100
                    conf_score = get_confidence_score(row, pct_change_pred)
                    
                    # Trading Execution
                    pnl = 0
                    if pct_change_pred > 0 and conf_score >= min_confidence:
                        entry_price = row['Close']
                        position_size = balance * (risk_per_trade / 100)
                        
                        target_price = entry_price * (1 + take_profit_pct/100)
                        stop_price = entry_price * (1 - stop_loss_pct/100)
                        
                        next_high = next_day_row['High']
                        next_low = next_day_row['Low']
                        next_close = next_day_row['Close']
                        
                        exit_price = next_close
                        status = "HOLD/CLOSE"
                        
                        if next_low <= stop_price:
                            exit_price = stop_price
                            status = "STOP LOSS ❌"
                        elif next_high >= target_price:
                            exit_price = target_price
                            status = "TAKE PROFIT ✅"
                        
                        coin_amount = position_size / entry_price
                        pnl = (exit_price - entry_price) * coin_amount
                        balance += pnl
                        
                        trade_log.append({
                            "Date": current_date.strftime('%Y-%m-%d'),
                            "Type": "LONG",
                            "Entry": entry_price,
                            "Exit": exit_price,
                            "Confidence": f"{conf_score}%",
                            "Status": status,
                            "PnL (USD)": pnl
                        })
                    
                    equity_curve.append(balance)
                    dates.append(current_date)
                
                # --- RESULTS ---
                total_return = ((balance - initial_capital) / initial_capital) * 100
                win_trades = len([t for t in trade_log if t['PnL (USD)'] > 0])
                total_trades = len(trade_log)
                win_rate = (win_trades / total_trades * 100) if total_trades > 0 else 0
                
                # Display Metrics
                st.write("### 🏁 Simulation Results")
                res_col1, res_col2, res_col3, res_col4 = st.columns(4)
                res_col1.metric("Final Balance", f"${balance:,.2f}")
                res_col2.metric("Total Return", f"{total_return:.2f}%", delta_color="normal")
                res_col3.metric("Total Trades", total_trades)
                res_col4.metric("Win Rate", f"{win_rate:.1f}%")
                
                # Equity Curve Chart
                st.subheader("📈 Equity Curve (Capital Growth)")
                fig_eq = go.Figure()
                fig_eq.add_trace(go.Scatter(x=dates, y=equity_curve[1:], mode='lines', name='Balance', line=dict(color='#00e676', width=2), fill='tozeroy'))
                fig_eq.update_layout(height=400, template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig_eq, use_container_width=True)
                
                # Trade Log Table
                st.subheader("📝 Transaction Log")
                if len(trade_log) > 0:
                    df_log = pd.DataFrame(trade_log)
                    def color_pnl(val):
                        color = '#00e676' if val > 0 else '#ff5252'
                        return f'color: {color}'
                    
                    # Using .map instead of .applymap to avoid FutureWarnings
                    st.dataframe(df_log.style.map(color_pnl, subset=['PnL (USD)']), use_container_width=True)
                else:
                    st.warning("No trades executed with current settings. Try lowering the Min. Confidence.")