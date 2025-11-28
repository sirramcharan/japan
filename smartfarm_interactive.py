# smartfarm_interactive.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error
import joblib
from datetime import timedelta
import os
import time

# ---------- Page Config & Styling ----------
st.set_page_config(
    page_title="SmartFarm AI", 
    page_icon="🌾", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for a cleaner look
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stMetric {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .big-card {
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
        color: white;
        font-weight: bold;
        text-align: center;
        font-size: 24px;
    }
    </style>
    """, unsafe_allow_html=True)

# ---------- Config ----------
DATA_PATH_DEFAULT = "realistic_dummy_dataset.csv" 
OUTPUT_DIR = "analysis_outputs"
MODEL_SAVE_PATH = os.path.join(OUTPUT_DIR, "model_{}.joblib")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------- Logic: Imports & Helper Functions ----------
USE_XGBOOST = False
try:
    import xgboost as xgb
    USE_XGBOOST = True
except Exception:
    USE_XGBOOST = False

@st.cache_data
def load_data(uploaded_file, use_default=True):
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    else:
        if use_default:
            if not os.path.exists(DATA_PATH_DEFAULT):
                return None
            df = pd.read_csv(DATA_PATH_DEFAULT)
        else:
            return None
    df.columns = [c.strip() for c in df.columns]
    if 'Date' not in df.columns:
        st.error("CSV must have a 'Date' column.")
        return None
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    return df

def create_features(df_prod):
    d = df_prod.copy().sort_values('Date').reset_index(drop=True)
    d['year'] = d['Date'].dt.year
    d['month'] = d['Date'].dt.month
    d['dayofweek'] = d['Date'].dt.dayofweek
    d['price_lag_1'] = d['Market_Price_JPY_per_kg'].shift(1)
    d['price_lag_7'] = d['Market_Price_JPY_per_kg'].shift(7)
    d['price_lag_30'] = d['Market_Price_JPY_per_kg'].shift(30)
    d['roll_mean_7'] = d['Market_Price_JPY_per_kg'].shift(1).rolling(7).mean()
    if 'Temperature_C' in d.columns:
        d['temp_roll_3'] = d['Temperature_C'].shift(1).rolling(3).mean()
        d['Temperature_C'] = d['Temperature_C'].astype(float)
    if 'Precipitation_mm' in d.columns:
        d['precip_roll_3'] = d['Precipitation_mm'].shift(1).rolling(3).sum()
        d['Precipitation_mm'] = d['Precipitation_mm'].astype(float)
    d = d.dropna(subset=['price_lag_1']).reset_index(drop=True)
    return d

# ---------- Sidebar ----------
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/606/606063.png", width=80)
    st.title("Settings")
    
    st.markdown("### 📂 Data Source")
    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    use_default = st.checkbox("Use server dataset", value=True)
    
    st.markdown("---")
    st.markdown("### ⚙️ Model Params")
    test_days = st.number_input("Test Window (Days)", min_value=30, max_value=365, value=180)
    
    st.info(f"ML Engine: {'XGBoost 🚀' if USE_XGBOOST else 'RandomForest 🌲'}")

# ---------- Main UI ----------

# Header
col_h1, col_h2 = st.columns([3, 1])
with col_h1:
    st.title("🌾 SmartFarm Forecast")
    st.markdown("AI-driven price prediction for smarter farming decisions.")

# Load Data
df = load_data(uploaded, use_default)
if df is None:
    st.warning("⚠️ No dataset available. Please upload a CSV or enable the default dataset.")
    st.stop()

# 1. Produce Selection & Context
produce_list = sorted(df['Produce_Name'].unique().tolist())

st.markdown("### 1️⃣ Select Produce")
col_sel, col_stats = st.columns([1, 3])

with col_sel:
    produce = st.selectbox("Choose a crop to analyze:", produce_list)

# Prepare Data specific to produce
df_prod_all = df[df['Produce_Name'] == produce].copy().reset_index(drop=True)
last_row = df_prod_all.sort_values('Date').iloc[-1]
last_price = float(last_row['Market_Price_JPY_per_kg'])
last_temp = float(last_row['Temperature_C']) if 'Temperature_C' in df_prod_all.columns else None
last_precip = float(last_row['Precipitation_mm']) if 'Precipitation_mm' in df_prod_all.columns else None
prev_price = float(df_prod_all.sort_values('Date').iloc[-2]['Market_Price_JPY_per_kg'])
price_delta = last_price - prev_price

with col_stats:
    # KPI Metrics
    m1, m2, m3 = st.columns(3)
    m1.metric("Latest Market Price", f"¥{last_price:.0f}", f"{price_delta:.0f} JPY")
    m2.metric("Last Temperature", f"{last_temp:.1f}°C" if last_temp else "N/A")
    m3.metric("Last Precipitation", f"{last_precip:.1f}mm" if last_precip else "N/A")

st.markdown("---")

# 2. Simulation Inputs
st.markdown("### 2️⃣ Simulation: Today's Weather")
st.caption("Input today's weather to generate a recursive forecast for the coming days.")

with st.container():
    c1, c2, c3 = st.columns(3)
    with c1:
        temp_input = st.number_input("🌡️ Today's Temp (°C)", value=float(last_temp) if last_temp else 0.0, format="%.1f")
    with c2:
        precip_input = st.number_input("🌧️ Today's Rain (mm)", value=float(last_precip) if last_precip else 0.0, format="%.1f")
    with c3:
        horizon = st.slider("📅 Forecast Horizon", 1, 30, 7)

# Pre-processing
feat = create_features(df_prod_all)
if feat.shape[0] < 50:
    st.error("Not enough data to train.")
    st.stop()

split_date = feat['Date'].max() - pd.Timedelta(days=int(test_days))
train = feat[feat['Date'] < split_date].copy()
test = feat[feat['Date'] >= split_date].copy()
candidate_features = [c for c in feat.columns if c not in ['Date', 'Produce_Name', 'Market_Price_JPY_per_kg', 'Category', 'Season', 'Quality_Grade'] and np.issubdtype(feat[c].dtype, np.number)]

# Train Button
st.markdown("")
if st.button("🚀 Run Forecast Model", type="primary", use_container_width=True):
    
    with st.spinner("Training model & generating recursive forecast..."):
        # Training Logic
        X_train = train[candidate_features].fillna(0)
        y_train = train['Market_Price_JPY_per_kg']
        X_test = test[candidate_features].fillna(0)
        y_test = test['Market_Price_JPY_per_kg']

        if USE_XGBOOST:
            dtrain = xgb.DMatrix(X_train, label=y_train)
            dtest = xgb.DMatrix(X_test, label=y_test)
            params = {'objective': 'reg:squarederror', 'eval_metric': 'rmse', 'seed': 42}
            bst = xgb.train(params, dtrain, num_boost_round=300, verbose_eval=False)
            predict_fn = lambda X: bst.predict(xgb.DMatrix(X))
        else:
            from sklearn.ensemble import RandomForestRegressor
            rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
            rf.fit(X_train, y_train)
            predict_fn = lambda X: rf.predict(X)

        preds_test = predict_fn(X_test)
        mae_test = mean_absolute_error(y_test, preds_test)

        # Recursive Forecast Logic
        last = feat.sort_values('Date').iloc[-1].copy()
        rows = []
        current = last.copy()
        
        # Simulating weather lists
        recent_temps = list(feat.sort_values('Date')['Temperature_C'].dropna().values[-3:]) if 'Temperature_C' in feat else []
        recent_precips = list(feat.sort_values('Date')['Precipitation_mm'].dropna().values[-3:]) if 'Precipitation_mm' in feat else []
        
        user_temp = float(temp_input)
        user_precip = float(precip_input)

        for i in range(horizon):
            fv = {}
            for f in candidate_features:
                if f == 'Temperature_C': fv[f] = user_temp
                elif f == 'Precipitation_mm': fv[f] = user_precip
                elif f.startswith('temp_roll_3'):
                    temps = (recent_temps + [user_temp])[-3:]
                    fv[f] = float(np.mean(temps)) if temps else 0.0
                elif f.startswith('precip_roll_3'):
                    precs = (recent_precips + [user_precip])[-3:]
                    fv[f] = float(np.sum(precs)) if precs else 0.0
                elif f.startswith('price_lag_'):
                    lag = int(f.split('_')[-1])
                    lookup_date = current['Date'] - pd.Timedelta(days=lag)
                    val = feat[feat['Date'] == lookup_date]['Market_Price_JPY_per_kg']
                    fv[f] = float(val.values[0]) if not val.empty else float(current.get('Market_Price_JPY_per_kg', 0))
                else:
                    fv[f] = float(current.get(f, 0) if f in current else 0)
            
            fv_df = pd.DataFrame([fv]).fillna(0)
            pred_next = float(predict_fn(fv_df)[0])
            next_date = current['Date'] + pd.Timedelta(days=1)
            rows.append({'Date': next_date.date(), 'Predicted_Price': pred_next})
            
            # Update state for recursion
            recent_temps.append(user_temp); recent_temps = recent_temps[-3:]
            recent_precips.append(user_precip); recent_precips = recent_precips[-3:]
            current['Date'] = next_date
            current['Market_Price_JPY_per_kg'] = pred_next
            current['price_lag_1'] = pred_next # Simplification for recursion

        fc_df = pd.DataFrame(rows)
        
        # Calculate Tip logic
        today_price = float(last['Market_Price_JPY_per_kg'])
        next_day_price = fc_df['Predicted_Price'].iloc[0]
        change_pct = (next_day_price - today_price) / max(1e-6, today_price) * 100
        
        if change_pct > 2:
            card_color = "#28a745" # Green
            icon = "📈"
            msg = "Price RISING. Hold stock!"
        elif change_pct < -2:
            card_color = "#dc3545" # Red
            icon = "📉"
            msg = "Price DROPPING. Sell soon."
        else:
            card_color = "#17a2b8" # Blue
            icon = "⚖️"
            msg = "Price STABLE."

    # --- RESULTS DISPLAY ---
    st.markdown("### 3️⃣ Forecast Results")
    
    # The Big Tip Card
    st.markdown(f"""
    <div class="big-card" style="background-color: {card_color};">
        {icon} AI Recommendation: {msg}<br>
        <span style="font-size:16px; opacity:0.9">Expected change tomorrow: {change_pct:.2f}%</span>
    </div>
    """, unsafe_allow_html=True)

    # Tabs for detail
    tab1, tab2, tab3 = st.tabs(["📊 Forecast Chart", "📋 Raw Data", "🛠 Model Health"])

    with tab1:
        # Merging history and forecast for a smooth line
        hist = feat[['Date', 'Market_Price_JPY_per_kg']].copy().tail(21)
        hist = hist.rename(columns={'Market_Price_JPY_per_kg': 'Price'})
        hist['Type'] = 'Historical'
        
        fc_df_plot = fc_df.rename(columns={'Predicted_Price': 'Price'})
        fc_df_plot['Date'] = pd.to_datetime(fc_df_plot['Date'])
        fc_df_plot['Type'] = 'Forecast'
        
        # Add connection point
        connection = pd.DataFrame([{
            'Date': hist.iloc[-1]['Date'],
            'Price': hist.iloc[-1]['Price'],
            'Type': 'Forecast'
        }])
        fc_df_plot = pd.concat([connection, fc_df_plot])
        
        combined_df = pd.concat([hist, fc_df_plot])
        
        # Streamlit Area Chart
        st.subheader(f"Price Trend: {produce}")
        st.line_chart(
            combined_df,
            x="Date",
            y="Price",
            color="Type", # Automatically colors history/forecast differently
        )

    with tab2:
        st.dataframe(fc_df.style.format({"Predicted_Price": "¥{:.2f}"}), use_container_width=True)

    with tab3:
        st.metric("Model Error (MAE)", f"¥{mae_test:.2f}")
        st.write(f"Validation on last {test_days} days.")
        st.caption("Lower MAE means better accuracy.")
