# smartfarm_interactive.py
# Minimalistic, fast Streamlit app for short-term produce price forecasts.
# Single-file. Designed to train quickly, cache models, and show a bold "tomorrow" result.
#
# Requirements (minimal): streamlit, pandas, numpy, scikit-learn, joblib
# Put your dataset in the repo root as realistic_dummy_dataset.csv OR set DATA_RAW_URL to the raw github url.

import streamlit as st
import pandas as pd
import numpy as np
import os, io, hashlib, joblib, time
from datetime import timedelta

# prefer HistGradientBoosting for speed; fallback if not available
try:
    from sklearn.ensemble import HistGradientBoostingRegressor
    HAVE_HGB = True
except Exception:
    HAVE_HGB = False
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error

# ---------- CONFIG ----------
st.set_page_config(page_title="SmartFarm — Minimal Forecast", page_icon="🌾", layout="centered")
DATA_LOCAL = "realistic_dummy_dataset.csv"   # file in your repo
DATA_RAW_URL = ""  # optional: "https://raw.githubusercontent.com/<user>/<repo>/main/realistic_dummy_dataset.csv"
OUTPUT_DIR = "analysis_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# UI styling (minimal)
st.markdown("""
<style>
body { background: #fbfbfc; color: #0f172a; font-family: Inter, system-ui, -apple-system, 'Helvetica Neue', Arial; }
.main { max-width:760px; margin:auto; }
.header { text-align:center; padding-top:18px; padding-bottom:6px; }
.h1 { font-weight:700; font-size:22px; margin:0; }
.sub { color:#6b7280; font-size:13px; margin-bottom:18px; }
.controls { display:flex; gap:12px; justify-content:center; margin-bottom:12px; }
.card { background: white; border-radius:12px; padding:18px; box-shadow: 0 6px 18px rgba(2,6,23,0.06); }
.big-price { font-weight:800; font-size:42px; color:#05264b; }
.small-list { color:#0f172a; font-size:14px; }
.btn { margin-top:10px; }
</style>
""", unsafe_allow_html=True)

st.markdown("<div class='main'><div class='header'>"
            "<div class='h1'>SmartFarm — Quick Forecast</div>"
            "<div class='sub'>Enter today’s temperature and pick days to forecast. Predictions are fast and cached.</div>"
            "</div>", unsafe_allow_html=True)

# ---------- Data loader ----------
@st.cache_data(show_spinner=False)
def load_df():
    # try local first
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
    st.error("Dataset not found. Place realistic_dummy_dataset.csv in repo root or set DATA_RAW_URL in the script.")
    st.stop()

# minimal required columns check
if not {'Produce_Name','Market_Price_JPY_per_kg'}.issubset(df.columns):
    st.error("Dataset must include columns: Produce_Name, Market_Price_JPY_per_kg")
    st.stop()

# ---------- Controls ----------
produces = sorted(df['Produce_Name'].unique().tolist())
col1, col2, col3 = st.columns([4,2,2])
with col1:
    produce = st.selectbox("Produce", produces)
with col2:
    # default today temp = last recorded temp for that produce (if available)
    last_temp_val = None
    tmpdf = df[df['Produce_Name']==produce]
    if 'Temperature_C' in tmpdf.columns and not tmpdf['Temperature_C'].isna().all():
        last_temp_val = float(tmpdf.sort_values('Date').iloc[-1]['Temperature_C'])
    temp_today = st.number_input("Today's temp (°C)", value=float(last_temp_val) if last_temp_val is not None else 20.0, step=0.1, format="%.1f")
with col3:
    horizon = st.slider("Days to forecast", 1, 14, 7)

st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

# ---------- Feature engineering (fast) ----------
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
if feat.shape[0] < 10:
    st.warning("Not enough history for reliable forecast. Add more data.")
    # continue but predictions will be baseline

# Candidate features (numeric only)
exclude = ['Date','Produce_Name','Market_Price_JPY_per_kg','Category','Season','Quality_Grade']
candidate_features = [c for c in feat.columns if c not in exclude and np.issubdtype(feat[c].dtype, np.number)]

# ---------- Fast training + caching ----------
def _cache_key_for(produce_name, features):
    key = f"{produce_name}|" + ",".join(features)
    return hashlib.sha1(key.encode()).hexdigest()

@st.cache_resource  # persists across reruns until code changes
def train_model_cached(key, X, y, max_samples=3000):
    """
    Train a reasonably fast model and return it. Cached by key.
    - Uses HistGradientBoosting when available, else Ridge.
    - Downcasts floats for speed and limits rows to most recent max_samples.
    """
    # downcast floats
    Xc = X.copy()
    for c in Xc.select_dtypes(include=['float64','int64']).columns:
        Xc[c] = pd.to_numeric(Xc[c], downcast='float')
    # cap rows: keep most recent
    if Xc.shape[0] > max_samples:
        Xc = Xc.tail(max_samples).copy()
        yc = y.tail(max_samples).copy()
    else:
        yc = y.copy()
    # choose model
    if HAVE_HGB:
        model = HistGradientBoostingRegressor(max_iter=100, max_depth=10)
    else:
        model = Ridge(alpha=1.0)
    model.fit(Xc.fillna(0), yc)
    # try persist to disk
    try:
        joblib.dump(model, os.path.join(OUTPUT_DIR, f"model_{key}.joblib"))
    except Exception:
        pass
    return model

def load_model_if_exists(key):
    path = os.path.join(OUTPUT_DIR, f"model_{key}.joblib")
    if os.path.exists(path):
        try:
            return joblib.load(path)
        except Exception:
            return None
    return None

# ---------- Forecast logic (recursive, light) ----------
def recursive_forecast(model, feat_full, candidate_features, temp_input, precip_input, h):
    # we use last row as current state and update lags iteratively
    last = feat_full.sort_values('Date').iloc[-1].copy()
    recent_temps = list(feat_full.sort_values('Date').get('Temperature_C', pd.Series()).dropna().values[-3:]) if 'Temperature_C' in feat_full.columns else []
    recent_precips = list(feat_full.sort_values('Date').get('Precipitation_mm', pd.Series()).dropna().values[-3:]) if 'Precipitation_mm' in feat_full.columns else []
    current = last.copy()
    rows = []
    for i in range(h):
        fv = {}
        for f in candidate_features:
            if f == 'Temperature_C':
                fv[f] = float(temp_input)
            elif f == 'Precipitation_mm':
                fv[f] = float(precip_input)
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
        try:
            pred = float(model.predict(Xrow)[0])
        except Exception:
            # fallback to last price
            pred = float(current.get('Market_Price_JPY_per_kg', 0))
        # Real-world date anchor
        real_today = pd.Timestamp.today().normalize()
        next_date = real_today + pd.Timedelta(days=i+1)

        rows.append({"Date": next_date.date(), "Predicted": round(pred,2)})
        # update state
        recent_temps.append(temp_input); recent_temps = recent_temps[-3:]
        recent_precips.append(precip_input); recent_precips = recent_precips[-3:]
        current['Date'] = next_date
        current['Market_Price_JPY_per_kg'] = pred
        current['price_lag_1'] = pred
    return pd.DataFrame(rows)

# ---------- MAIN: Forecast button ----------
forecast_button = st.button("Show forecast", key="forecast_btn")

if forecast_button:
    # prepare training set (train on history excluding last few days to avoid leakage)
    feat_full = feat.copy() if 'feat' in locals() else create_features_single(df_prod_all)
    if feat_full.shape[0] < 5:
        st.warning("Insufficient history; showing baseline (last price repeated).")
        last_price = float(df_prod_all.sort_values('Date').iloc[-1]['Market_Price_JPY_per_kg'])
        baseline_preds = pd.DataFrame([{"Date": (df_prod_all['Date'].max() + pd.Timedelta(days=i+1)).date(),
                                        "Predicted": round(last_price,2)} for i in range(horizon)])
        # big tomorrow
        tomorrow = baseline_preds.iloc[0]['Predicted']
        st.markdown(f"<div class='card' style='text-align:center;margin-bottom:12px'><div class='big-price'>{tomorrow} JPY/kg</div>"
                    f"<div style='color:#475569;margin-top:6px'>Predicted for tomorrow</div></div>", unsafe_allow_html=True)
        # small list
        small = baseline_preds.iloc[1:].to_dict('records')
        if small:
            st.markdown("<div class='card small-list'>")
            for r in small:
                st.write(f"{r['Date']} — {r['Predicted']} JPY/kg")
            st.markdown("</div>", unsafe_allow_html=True)
    else:
        # feature + target
        X = feat_full[candidate_features].fillna(0)
        y = feat_full['Market_Price_JPY_per_kg']
        key = _cache_key_for(produce, candidate_features)
        # try load persisted
        model = load_model_if_exists(key)
        if model is None:
            t0 = time.time()
            model = train_model_cached(key, X, y, max_samples=3000)
            t_train = time.time() - t0
            st.info(f"Trained (or loaded) model in {t_train:.1f}s")
        else:
            st.info("Loaded model from disk/cache (fast).")
        # recursive forecast using farmer temp (constant) and no precip input UI (use 0 if absent)
        precip_input = 0.0
        if 'Precipitation_mm' in feat_full.columns:
            precip_input = float(feat_full.sort_values('Date').iloc[-1].get('Precipitation_mm',0))
        fc = recursive_forecast(model, feat_full, candidate_features, temp_today, precip_input, horizon)
        # display: tomorrow big, rest small
        tomorrow_value = fc.iloc[0]['Predicted']
        st.markdown(f"<div class='card' style='text-align:center;margin-bottom:12px'><div class='big-price'>{tomorrow_value} JPY/kg</div>"
                    f"<div style='color:#475569;margin-top:6px'>Predicted price — tomorrow</div></div>", unsafe_allow_html=True)
        # small table for remaining days (if horizon > 1)
        if horizon > 1:
            small_df = fc.iloc[1:].reset_index(drop=True)
            st.markdown("<div class='card small-list'>", unsafe_allow_html=True)
            for _, row in small_df.iterrows():
                st.markdown(f"<div style='padding:6px 0'>{row['Date']} — <b>{row['Predicted']}</b> JPY/kg</div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
        # download button
        csv_bytes = fc.to_csv(index=False).encode('utf-8')
        st.download_button("Download forecast CSV", data=csv_bytes, file_name=f"{produce}_forecast.csv", mime="text/csv")

# small footer
st.markdown("<div style='height:18px'></div><div style='text-align:center;color:#94a3b8;font-size:12px'>Lightweight forecast — cached models for speed. For production, host pre-trained models & serve via API.</div></div>", unsafe_allow_html=True)
