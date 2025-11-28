# smartfarm_interactive.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error
import joblib
import os

# ---------- Page Config & Styling ----------
st.set_page_config(
    page_title="SmartFarm AI", 
    page_icon="🌾", 
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stMetric { background-color: white; padding: 15px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    .big-card { padding: 20px; border-radius: 10px; margin-bottom: 20px; color: white; font-weight: bold; text-align: center; font-size: 24px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
    </style>
    """, unsafe_allow_html=True)

# ---------- Logic & Caching ----------
DATA_PATH_DEFAULT = "realistic_dummy_dataset.csv" 

# Check for XGBoost availability
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
        if use_default and os.path.exists(DATA_PATH_DEFAULT):
            df = pd.read_csv(DATA_PATH_DEFAULT)
        else:
            return None
    df.columns = [c.strip() for c in df.columns]
    if 'Date' not in df.columns: return None
    df['Date'] = pd.to_datetime(df['Date'])
    return df.sort_values('Date').reset_index(drop=True)

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
    return d.dropna(subset=['price_lag_1']).reset_index(drop=True)

# --- KEY FIX: @st.cache_resource ---
# This function runs ONCE per produce/test_days combo. 
# Subsequent runs are instant.
@st.cache_resource(show_spinner="Training AI Model (this happens only once)...")
def train_model(df_prod, test_days):
    # 1. Feature Engineering
    feat = create_features(df_prod)
    if feat.shape[0] < 50: return None, None, None, None, None

    # 2. Split
    split_date = feat['Date'].max() - pd.Timedelta(days=int(test_days))
    train = feat[feat['Date'] < split_date].copy()
    test = feat[feat['Date'] >= split_date].copy()
    
    features = [c for c in feat.columns if c not in ['Date', 'Produce_Name', 'Market_Price_JPY_per_kg', 'Category', 'Season', 'Quality_Grade'] and np.issubdtype(feat[c].dtype, np.number)]
    
    X_train = train[features].fillna(0)
    y_train = train['Market_Price_JPY_per_kg']
    X_test = test[features].fillna(0)
    y_test = test['Market_Price_JPY_per_kg']

    # 3. Train
    if USE_XGBOOST:
        dtrain = xgb.DMatrix(X_train, label=y_train)
        params = {'objective': 'reg:squarederror', 'eval_metric': 'rmse', 'seed': 42}
        model = xgb.train(params, dtrain, num_boost_round=200) # Reduced rounds for speed
    else:
        from sklearn.ensemble import RandomForestRegressor
        model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)

    # 4. Calc Error
    if USE_XGBOOST:
        preds = model.predict(xgb.DMatrix(X_test))
    else:
        preds = model.predict(X_test)
    
    mae = mean_absolute_error(y_test, preds)
    
    # Return everything needed for forecasting
    return model, mae, features, feat.iloc[-1].to_dict(), feat

# ---------- Sidebar ----------
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/606/606063.png", width=80)
    st.title("Settings")
    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    use_default = st.checkbox("Use server dataset", value=True)
    st.markdown("---")
    test_days = st.number_input("Validation Window (Days)", 30, 365, 180)
    st.caption("Change inputs to see instant updates.")

# ---------- Main UI ----------
st.title("🌾 SmartFarm Forecast (Instant)")
col_sel, col_kpi = st.columns([1, 3])

df = load_data(uploaded, use_default)
if df is None:
    st.warning("Please upload data.")
    st.stop()

with col_sel:
    produce_list = sorted(df['Produce_Name'].unique().tolist())
    produce = st.selectbox("Select Crop:", produce_list)

# Get Data for specific produce
df_prod = df[df['Produce_Name'] == produce].copy()

# TRAIN MODEL (Cached)
model, mae, features, last_row_dict, feat_df = train_model(df_prod, test_days)

if model is None:
    st.error("Not enough data to train this crop.")
    st.stop()

# Display KPIs
last_price = last_row_dict['Market_Price_JPY_per_kg']
last_temp = last_row_dict.get('Temperature_C', None)
last_precip = last_row_dict.get('Precipitation_mm', None)

with col_kpi:
    k1, k2, k3 = st.columns(3)
    k1.metric("Current Price", f"¥{last_price:.0f}")
    k2.metric("Last Temp", f"{last_temp:.1f}°C" if last_temp else "-")
    k3.metric("Last Rain", f"{last_precip:.1f}mm" if last_precip else "-")

st.markdown("---")

# Simulation Inputs
st.markdown("### 🔮 Interactive Forecast")
c1, c2, c3 = st.columns(3)
with c1: temp_input = st.number_input("Today's Temp (°C)", value=float(last_temp or 0), step=0.5)
with c2: precip_input = st.number_input("Today's Rain (mm)", value=float(last_precip or 0), step=0.5)
with c3: horizon = st.slider("Days to Forecast", 1, 30, 7)

# --- FAST FORECAST LOOP ---
# This runs instantly because 'model' is already loaded from cache
rows = []
current = last_row_dict.copy()
# Reconstruct recent history for rolling averages
recent_temps = feat_df['Temperature_C'].tail(3).tolist() if 'Temperature_C' in feat_df else []
recent_precips = feat_df['Precipitation_mm'].tail(3).tolist() if 'Precipitation_mm' in feat_df else []

for i in range(horizon):
    fv = {}
    for f in features:
        if f == 'Temperature_C': fv[f] = temp_input
        elif f == 'Precipitation_mm': fv[f] = precip_input
        elif f.startswith('temp_roll_3'):
            temps = (recent_temps + [temp_input])[-3:]
            fv[f] = float(np.mean(temps)) if temps else 0.0
        elif f.startswith('precip_roll_3'):
            precs = (recent_precips + [precip_input])[-3:]
            fv[f] = float(np.sum(precs)) if precs else 0.0
        elif f.startswith('price_lag_'):
            lag = int(f.split('_')[-1])
            # Simple approximation for lag in recursion to speed up
            if lag == 1: fv[f] = current.get('Market_Price_JPY_per_kg', 0)
            else: fv[f] = current.get(f, current.get('Market_Price_JPY_per_kg', 0)) 
        else:
            fv[f] = current.get(f, 0)
    
    # Predict single row
    fv_df = pd.DataFrame([fv]).fillna(0)
    if USE_XGBOOST:
        pred = model.predict(xgb.DMatrix(fv_df))[0]
    else:
        pred = model.predict(fv_df)[0]
    
    # --- FIXED LINE BELOW ---
    next_date = current['Date'] + pd.Timedelta(days=1)
    rows.append({'Date': next_date, 'Price': float(pred)})
    
    # Update state
    recent_temps.append(temp_input)
    recent_precips.append(precip_input)
    current['Date'] = next_date
    current['Market_Price_JPY_per_kg'] = pred

fc_df = pd.DataFrame(rows)

# Recommendation Logic
change_pct = (fc_df['Price'].iloc[0] - last_price) / max(1e-6, last_price) * 100
if change_pct > 1.5:
    color, icon, msg = "#28a745", "📈", "Prices Rising: Hold Stock"
elif change_pct < -1.5:
    color, icon, msg = "#dc3545", "📉", "Prices Falling: Sell Now"
else:
    color, icon, msg = "#17a2b8", "⚖️", "Prices Stable"

# Render UI
st.markdown(f"""
<div class="big-card" style="background-color: {color};">
    {icon} {msg} <br>
    <span style="font-size:16px; opacity:0.9">Tomorrow: {change_pct:+.2f}%</span>
</div>
""", unsafe_allow_html=True)

tab1, tab2 = st.tabs(["📊 Visuals", "📋 Data"])
with tab1:
    # Combine history and forecast for plotting
    hist_data = feat_df[['Date', 'Market_Price_JPY_per_kg']].tail(30).rename(columns={'Market_Price_JPY_per_kg': 'Price'})
    hist_data['Type'] = 'Historical'
    fc_data = fc_df.copy()
    fc_data['Type'] = 'Forecast'
    
    # Link the lines
    link_row = pd.DataFrame([{'Date': hist_data.iloc[-1]['Date'], 'Price': hist_data.iloc[-1]['Price'], 'Type': 'Forecast'}])
    chart_data = pd.concat([hist_data, link_row, fc_data])
    
    st.line_chart(chart_data, x='Date', y='Price', color='Type')

with tab2:
    st.dataframe(fc_df.style.format({"Price": "¥{:.2f}"}), use_container_width=True)
    st.caption(f"Model Accuracy (MAE): ¥{mae:.2f}")
