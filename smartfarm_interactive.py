# smartfarm_interactive.py
# DESIGN UPDATE: Dark Mode Optimized + Zoomed-in Graph

import streamlit as st
import pandas as pd
import numpy as np
import os, hashlib, joblib
import altair as alt

# prefer HistGradientBoosting for speed; fallback if not available
try:
    from sklearn.ensemble import HistGradientBoostingRegressor
    HAVE_HGB = True
except Exception:
    HAVE_HGB = False
from sklearn.linear_model import Ridge

# ---------- CONFIG ----------
st.set_page_config(
    page_title="SmartFarm — AI Forecast", 
    page_icon="🌱", 
    layout="centered"
)

DATA_LOCAL = "realistic_dummy_dataset.csv"   
DATA_RAW_URL = "" 
OUTPUT_DIR = "analysis_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------- UI & STYLING (Dark Mode Optimized) ----------
st.markdown("""
<style>
    /* Import Font */
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700&display=swap');

    /* Global Font Application */
    html, body, [class*="css"] {
        font-family: 'Poppins', sans-serif;
    }
    
    /* Header Styling - White Text for Dark Background */
    .main-header {
        text-align: center;
        margin-bottom: 2rem;
    }
    .main-header h1 {
        font-weight: 700;
        margin-bottom: 0.5rem;
        color: #f1f5f9 !important; /* White/Slate-100 */
    }
    .main-header p {
        color: #94a3b8 !important; /* Slate-400 */
        font-size: 15px;
    }

    /* THE HERO CARD (Kept Light for Contrast) */
    .result-card {
        background: linear-gradient(135deg, #ecfdf5 0%, #d1fae5 100%);
        border: 1px solid #10b981;
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3);
        margin-bottom: 1.5rem;
    }
    /* Text INSIDE the card must be dark to read on light green */
    .price-label {
        color: #047857 !important;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
        font-size: 0.85rem;
        margin-bottom: 0.5rem;
    }
    .big-price {
        color: #064e3b !important;
        font-weight: 800;
        font-size: 3.5rem;
        line-height: 1;
        margin: 0;
    }
    .unit {
        font-size: 1.5rem;
        color: #059669 !important;
        font-weight: 600;
    }
    
    /* Button Styling */
    .stButton > button {
        width: 100%;
        background-color: #10b981;
        color: white !important;
        border: none;
        padding: 0.6rem 1rem;
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.2s ease;
    }
    .stButton > button:hover {
        background-color: #059669;
        box-shadow: 0 4px 12px rgba(16, 185, 129, 0.4);
    }

    /* Table Styling - Light Text for Dark Background */
    .forecast-row {
        display: flex;
        justify-content: space-between;
        padding: 12px 0;
        border-bottom: 1px solid #334155; /* Darker border */
        font-size: 1rem;
    }
    .date-col { color: #94a3b8 !important; } /* Slate-400 */
    .price-col { font-weight: 600; color: #f1f5f9 !important; } /* White */

</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="main-header">
    <h1>🌱 SmartFarm Forecast</h1>
    <p>AI-driven price predictions for your produce.</p>
</div>
""", unsafe_allow_html=True)

# ---------- Data loader ----------
@st.cache_data(show_spinner=False)
def load_df():
    if os.path.exists(DATA_LOCAL):
        df = pd.read_csv(DATA_LOCAL)
    elif DATA_RAW_URL:
        df = pd.read_csv(DATA_RAW_URL)
    else:
        return None
    df.columns = [c.strip() for c in df.columns]
    if 'Date' not in df.columns:
        return None
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    return df

df = load_df()
if df is None:
    st.error("⚠️ Dataset not found. Please ensure 'realistic_dummy_dataset.csv' is in the root directory.")
    st.stop()

if not {'Produce_Name','Market_Price_JPY_per_kg'}.issubset(df.columns):
    st.error("Dataset must include columns: Produce_Name, Market_Price_JPY_per_kg")
    st.stop()

# ---------- Controls ----------
with st.container():
    produces = sorted(df['Produce_Name'].unique().tolist())
    c1, c2, c3 = st.columns([2, 1, 1])
    
    with c1:
        produce = st.selectbox("Select Produce", produces)
    with c2:
        last_temp_val = None
        tmpdf = df[df['Produce_Name']==produce]
        if 'Temperature_C' in tmpdf.columns and not tmpdf['Temperature_C'].isna().all():
            last_temp_val = float(tmpdf.sort_values('Date').iloc[-1]['Temperature_C'])
        temp_today = st.number_input("Today's Temp (°C)", value=float(last_temp_val) if last_temp_val is not None else 20.0, step=0.1, format="%.1f")
    with c3:
        horizon = st.slider("Forecast Days", 1, 14, 7)

# ---------- Feature Engineering & Modeling ----------
def create_features_single(df_prod):
    d = df_prod.copy().sort_values('Date').reset_index(drop=True)
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

df_prod_all = df[df['Produce_Name']==produce].copy().reset_index(drop=True)
feat = create_features_single(df_prod_all)
exclude = ['Date','Produce_Name','Market_Price_JPY_per_kg','Category','Season','Quality_Grade']
candidate_features = [c for c in feat.columns if c not in exclude and np.issubdtype(feat[c].dtype, np.number)]

def _cache_key_for(produce_name, features):
    key = f"{produce_name}|" + ",".join(features)
    return hashlib.sha1(key.encode()).hexdigest()

@st.cache_resource
def train_model_cached(key, X, y, max_samples=3000):
    Xc = X.copy()
    for c in Xc.select_dtypes(include=['float64','int64']).columns:
        Xc[c] = pd.to_numeric(Xc[c], downcast='float')
    if Xc.shape[0] > max_samples:
        Xc = Xc.tail(max_samples).copy()
        yc = y.tail(max_samples).copy()
    else:
        yc = y.copy()
    
    if HAVE_HGB:
        model = HistGradientBoostingRegressor(max_iter=100, max_depth=10)
    else:
        model = Ridge(alpha=1.0)
    model.fit(Xc.fillna(0), yc)
    try:
        joblib.dump(model, os.path.join(OUTPUT_DIR, f"model_{key}.joblib"))
    except Exception:
        pass
    return model

def load_model_if_exists(key):
    path = os.path.join(OUTPUT_DIR, f"model_{key}.joblib")
    if os.path.exists(path):
        try: return joblib.load(path)
        except Exception: return None
    return None

def recursive_forecast(model, feat_full, candidate_features, temp_input, precip_input, h):
    last = feat_full.sort_values('Date').iloc[-1].copy()
    recent_temps = list(feat_full.sort_values('Date').get('Temperature_C', pd.Series()).dropna().values[-3:]) if 'Temperature_C' in feat_full.columns else []
    recent_precips = list(feat_full.sort_values('Date').get('Precipitation_mm', pd.Series()).dropna().values[-3:]) if 'Precipitation_mm' in feat_full.columns else []
    current = last.copy()
    rows = []
    
    for i in range(h):
        fv = {}
        for f in candidate_features:
            if f == 'Temperature_C': fv[f] = float(temp_input)
            elif f == 'Precipitation_mm': fv[f] = float(precip_input)
            elif f.startswith('temp_roll_3'):
                tlist = (recent_temps + [temp_input])[-3:]
                fv[f] = float(np.mean(tlist)) if len(tlist)>0 else float(temp_input)
            elif f.startswith('precip_roll_3'):
                plist = (recent_precips + [precip_input])[-3:]
                fv[f] = float(np.sum(plist)) if len(plist)>0 else float(precip_input)
            elif f.startswith('price_lag_'):
                lag = int(f.split('_')[-1])
                lookup_date = current['Date'] - pd.Timedelta(days=lag)
                val = feat_full[feat_full['Date']==lookup_date]['Market_Price_JPY_per_kg']
                fv[f] = float(val.values[0]) if not val.empty else float(current.get('Market_Price_JPY_per_kg', 0))
            else:
                fv[f] = float(current.get(f, 0) if f in current else 0)
        
        Xrow = pd.DataFrame([fv]).fillna(0)
        try: pred = float(model.predict(Xrow)[0])
        except Exception: pred = float(current.get('Market_Price_JPY_per_kg', 0))
            
        next_date = pd.Timestamp.today().normalize() + pd.Timedelta(days=i+1)
        rows.append({"Date": next_date.date(), "Predicted": round(pred,2)})
        
        recent_temps.append(temp_input); recent_temps = recent_temps[-3:]
        recent_precips.append(precip_input); recent_precips = recent_precips[-3:]
        current['Date'] = next_date
        current['Market_Price_JPY_per_kg'] = pred
        current['price_lag_1'] = pred
    return pd.DataFrame(rows)

# ---------- MAIN ----------
st.markdown("<br>", unsafe_allow_html=True)
if st.button("🚀 Generate Forecast"):
    with st.spinner("Processing..."):
        feat_full = feat.copy() if 'feat' in locals() else create_features_single(df_prod_all)
        
        if feat_full.shape[0] < 5:
            last_price = float(df_prod_all.sort_values('Date').iloc[-1]['Market_Price_JPY_per_kg'])
            fc = pd.DataFrame([{"Date": (df_prod_all['Date'].max() + pd.Timedelta(days=i+1)).date(),
                                "Predicted": round(last_price,2)} for i in range(horizon)])
        else:
            X = feat_full[candidate_features].fillna(0)
            y = feat_full['Market_Price_JPY_per_kg']
            key = _cache_key_for(produce, candidate_features)
            model = load_model_if_exists(key)
            if model is None: model = train_model_cached(key, X, y)
            precip_input = float(feat_full.sort_values('Date').iloc[-1].get('Precipitation_mm',0)) if 'Precipitation_mm' in feat_full.columns else 0.0
            fc = recursive_forecast(model, feat_full, candidate_features, temp_today, precip_input, horizon)

        # 1. The Card (Kept as is - works great on dark)
        tomorrow_val = fc.iloc[0]['Predicted']
        st.markdown(f"""
        <div class="result-card">
            <div class="price-label">Tomorrow's Predicted Price</div>
            <div class="big-price">{tomorrow_val}<span class="unit"> JPY</span></div>
            <div style="color:#059669 !important; font-size: 14px; margin-top:5px;">Per kg • {produce}</div>
        </div>
        """, unsafe_allow_html=True)

        # 2. Graph (Zoomed In + Dark Mode Axes)
        fc['Date'] = pd.to_datetime(fc['Date'])
        
        # We calculate min/max to ensure the scale is not flat 0-100
        y_min = fc['Predicted'].min() * 0.99
        y_max = fc['Predicted'].max() * 1.01

        c = alt.Chart(fc).mark_line(
            color='#10b981', 
            strokeWidth=3, 
            point=True
        ).encode(
            x=alt.X('Date', 
                    axis=alt.Axis(format='%b %d', title='Forecast Date', labelColor='#94a3b8', titleColor='#94a3b8')),
            y=alt.Y('Predicted', 
                    title='Price (JPY)', 
                    # ZERO=FALSE is key here to zoom in
                    scale=alt.Scale(zero=False, domain=[y_min, y_max]), 
                    axis=alt.Axis(labelColor='#94a3b8', titleColor='#94a3b8')),
            tooltip=['Date', 'Predicted']
        ).properties(
            title="Future Trend",
            height=250
        ).configure_title(
            color='#f1f5f9', font='Poppins' # White title
        ).configure_view(
            strokeWidth=0
        )
        
        st.altair_chart(c, use_container_width=True)

        # 3. Table (Dark Mode Text)
        if horizon > 1:
            with st.expander("📄 View Details"):
                html = "<div>"
                for _, row in fc.iloc[1:].iterrows():
                    d_str = row['Date'].strftime('%Y-%m-%d')
                    html += f"<div class='forecast-row'><span class='date-col'>{d_str}</span><span class='price-col'>{row['Predicted']} JPY</span></div>"
                html += "</div>"
                st.markdown(html, unsafe_allow_html=True)
                csv = fc.to_csv(index=False).encode('utf-8')
                st.download_button("Download CSV", data=csv, file_name=f"{produce}_forecast.csv", mime="text/csv")
