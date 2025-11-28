# smartfarm_interactive.py
# Streamlit app: farmer selects produce, inputs today's temperature (and optional rain), picks horizon,
# app trains quick model and shows a recursive forecast using the farmer-provided weather for future days.
# Cleaned version (no notebook metadata). Falls back to RandomForest if xgboost not available.

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
import joblib
from datetime import timedelta
import os

st.set_page_config(page_title="SmartFarm — Farmer Forecast", layout="wide")
st.title("SmartFarm — Simple Price Forecast for Farmers")

# ---------- Config ----------
DATA_PATH_DEFAULT = r"C:\Users\lucky\Documents\Japan\realistic_dummy_dataset.csv"  # update if your CSV lives elsewhere
OUTPUT_DIR = r"C:\Users\lucky\Documents\Japan\analysis_outputs"
MODEL_SAVE_PATH = os.path.join(OUTPUT_DIR, "model_{}.joblib")
os.makedirs(OUTPUT_DIR, exist_ok=True)
st.sidebar.header("Settings")
# -----------------------------

# ML backend selection (try xgboost then fallback)
USE_XGBOOST = False
try:
    import xgboost as xgb
    USE_XGBOOST = True
except Exception:
    from sklearn.ensemble import RandomForestRegressor

# Helper: load data (uploaded or default)
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
    # normalize column names (strip)
    df.columns = [c.strip() for c in df.columns]
    if 'Date' not in df.columns:
        st.error("CSV must have a 'Date' column.")
        return None
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    return df

uploaded = st.sidebar.file_uploader("Upload CSV (optional)", type=["csv"])
use_default = st.sidebar.checkbox("Use default dataset on server", value=True)

df = load_data(uploaded, use_default)
if df is None:
    st.warning("No dataset available. Upload a CSV or enable the default dataset and ensure the path is correct.")
    st.stop()

st.sidebar.markdown(f"Dataset rows: {df.shape[0]} | Produces: {df['Produce_Name'].nunique() if 'Produce_Name' in df.columns else 0}")

# sanity check
if 'Produce_Name' not in df.columns or 'Market_Price_JPY_per_kg' not in df.columns:
    st.error("CSV must contain at least 'Produce_Name' and 'Market_Price_JPY_per_kg' columns.")
    st.stop()

# Produce selection
produce_list = sorted(df['Produce_Name'].unique().tolist())
produce = st.selectbox("Select produce", produce_list)

# Quick date range and last known weather shown to user
df_prod_all = df[df['Produce_Name'] == produce].copy().reset_index(drop=True)
st.markdown(f"**Data range for {produce}:** {df_prod_all['Date'].min().date()} → {df_prod_all['Date'].max().date()}")

last_row = df_prod_all.sort_values('Date').iloc[-1]
last_temp = float(last_row['Temperature_C']) if 'Temperature_C' in df_prod_all.columns else None
last_precip = float(last_row['Precipitation_mm']) if 'Precipitation_mm' in df_prod_all.columns else None

# Farmer inputs: today's temp and precip (optional)
st.subheader("Farmer inputs (override future weather)")
col1, col2 = st.columns(2)
with col1:
    temp_input = st.number_input("Today's temperature (°C) — edit or keep default",
                                 value=float(last_temp) if last_temp is not None else 0.0, format="%.2f")
with col2:
    precip_input = st.number_input("Today's precipitation (mm) — edit or keep default",
                                   value=float(last_precip) if last_precip is not None else 0.0, format="%.2f")

# Forecast horizon
horizon = st.slider("Forecast horizon (days)", 1, 30, 7)

# Test window for evaluation
test_days = st.number_input("Test window (days) for validation", min_value=30, max_value=730, value=180)

st.markdown(
    "This app trains a quick model on historical data for the chosen produce. "
    "It then performs a recursive forecast for the horizon you pick using the weather values you provide."
)

# ---------- Feature creation ----------
def create_features(df_prod):
    d = df_prod.copy().sort_values('Date').reset_index(drop=True)
    d['year'] = d['Date'].dt.year
    d['month'] = d['Date'].dt.month
    d['dayofweek'] = d['Date'].dt.dayofweek
    # lags and rolling
    d['price_lag_1'] = d['Market_Price_JPY_per_kg'].shift(1)
    d['price_lag_7'] = d['Market_Price_JPY_per_kg'].shift(7)
    d['price_lag_30'] = d['Market_Price_JPY_per_kg'].shift(30)
    d['roll_mean_7'] = d['Market_Price_JPY_per_kg'].shift(1).rolling(7).mean()
    d['roll_std_7'] = d['Market_Price_JPY_per_kg'].shift(1).rolling(7).std()
    if 'Temperature_C' in d.columns:
        d['temp_roll_3'] = d['Temperature_C'].shift(1).rolling(3).mean()
        d['Temperature_C'] = d['Temperature_C'].astype(float)
    if 'Precipitation_mm' in d.columns:
        d['precip_roll_3'] = d['Precipitation_mm'].shift(1).rolling(3).sum()
        d['Precipitation_mm'] = d['Precipitation_mm'].astype(float)
    # Drop rows missing the minimal lag (price_lag_1)
    d = d.dropna(subset=['price_lag_1']).reset_index(drop=True)
    return d

feat = create_features(df_prod_all)

if feat.shape[0] < 50:
    st.warning("Not enough rows after lag creation to train reliably. Consider more data or smaller lags.")

# Train/test split
split_date = feat['Date'].max() - pd.Timedelta(days=int(test_days))
train = feat[feat['Date'] < split_date].copy()
test = feat[feat['Date'] >= split_date].copy()
st.write(f"Train rows: {train.shape[0]} | Test rows: {test.shape[0]} (split date: {split_date.date()})")

# Candidate features (numeric engineered)
exclude = ['Date', 'Produce_Name', 'Market_Price_JPY_per_kg', 'Category', 'Season', 'Quality_Grade']
candidate_features = [c for c in feat.columns if c not in exclude and (np.issubdtype(feat[c].dtype, np.number))]
if len(candidate_features) == 0:
    st.error("No numeric features available for modeling after preprocessing.")
    st.stop()
st.write("Using features:", candidate_features)

# Train & predict button
if st.button("Train model & forecast"):
    X_train = train[candidate_features].fillna(0)
    y_train = train['Market_Price_JPY_per_kg']
    X_test = test[candidate_features].fillna(0)
    y_test = test['Market_Price_JPY_per_kg']

    # Train model: XGBoost if available, else RandomForest
    if USE_XGBOOST:
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dtest = xgb.DMatrix(X_test, label=y_test)
        params = {'objective': 'reg:squarederror', 'eval_metric': 'rmse', 'seed': 42}
        bst = xgb.train(params, dtrain, num_boost_round=300, evals=[(dtest, 'eval')], early_stopping_rounds=25, verbose_eval=False)
        predict_fn = lambda X: bst.predict(xgb.DMatrix(X))
        model_obj = bst
    else:
        # fallback
        from sklearn.ensemble import RandomForestRegressor
        rf = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
        rf.fit(X_train, y_train)
        predict_fn = lambda X: rf.predict(X)
        model_obj = rf

    # test predictions
    preds_test = predict_fn(X_test)
    mae_test = mean_absolute_error(y_test, preds_test)
    st.success(f"Model trained. Test MAE = {mae_test:.3f} JPY/kg")

    # plot actual vs pred (test)
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(test['Date'], y_test.values, label="Actual")
    ax.plot(test['Date'], preds_test, label="Predicted")
    ax.set_title(f"{produce} — Actual vs Pred (test)")
    ax.legend()
    st.pyplot(fig)

    # Save model
    try:
        joblib.dump(model_obj, MODEL_SAVE_PATH.format(produce))
        st.write("Saved model:", MODEL_SAVE_PATH.format(produce))
    except Exception as e:
        st.write("Model not saved (write error):", e)

    # Recursive forecast using farmer-provided weather values
    last = feat.sort_values('Date').iloc[-1].copy()
    rows = []
    current = last.copy()

    # prepare a small queue of recent temperatures/precip for rolling calculation
    recent_temps = list(feat.sort_values('Date')['Temperature_C'].dropna().values[-3:]) if 'Temperature_C' in feat else []
    recent_precips = list(feat.sort_values('Date')['Precipitation_mm'].dropna().values[-3:]) if 'Precipitation_mm' in feat else []

    user_temp = float(temp_input) if temp_input is not None else (recent_temps[-1] if recent_temps else 0.0)
    user_precip = float(precip_input) if precip_input is not None else (recent_precips[-1] if recent_precips else 0.0)

    for i in range(horizon):
        # build feature vector for current day
        fv = {}
        for f in candidate_features:
            if f == 'Temperature_C':
                fv[f] = user_temp
            elif f == 'Precipitation_mm':
                fv[f] = user_precip
            elif f.startswith('temp_roll_3'):
                temps = (recent_temps + [user_temp])[-3:]
                fv[f] = float(np.mean(temps)) if len(temps) > 0 else 0.0
            elif f.startswith('precip_roll_3'):
                precs = (recent_precips + [user_precip])[-3:]
                fv[f] = float(np.sum(precs)) if len(precs) > 0 else 0.0
            elif f.startswith('price_lag_'):
                lag = int(f.split('_')[-1])
                lookup_date = current['Date'] - pd.Timedelta(days=lag)
                val = feat[feat['Date'] == lookup_date]['Market_Price_JPY_per_kg']
                if not val.empty:
                    fv[f] = float(val.values[0])
                else:
                    fv[f] = float(current.get('Market_Price_JPY_per_kg', 0))
            else:
                fv[f] = float(current.get(f, 0) if f in current else 0)
        fv_df = pd.DataFrame([fv]).fillna(0)
        pred_next = float(predict_fn(fv_df)[0])
        next_date = current['Date'] + pd.Timedelta(days=1)
        rows.append({'Date': next_date.date(), 'Predicted_Price': pred_next})

        recent_temps.append(user_temp); recent_temps = recent_temps[-3:]
        recent_precips.append(user_precip); recent_precips = recent_precips[-3:]
        current['Date'] = next_date
        current['Market_Price_JPY_per_kg'] = pred_next
        current['price_lag_1'] = pred_next

    fc_df = pd.DataFrame(rows)
    st.subheader("Forecast")
    st.dataframe(fc_df)

    # action tip using first-day change
    today_price = float(last['Market_Price_JPY_per_kg'])
    change_pct = (fc_df['Predicted_Price'].iloc[0] - today_price) / max(1e-6, today_price) * 100
    if change_pct > 2:
        tip = "Price likely up: consider holding stock 1-3 days to get a better price."
    elif change_pct < -2:
        tip = "Price likely down: consider selling soon or negotiate bulk sale to avoid loss."
    else:
        tip = "Price likely stable: follow your normal plan."

    st.info(f"Tip: {tip} (expected change next day: {change_pct:.2f}%)")

    # show recent predicted trend plot
    fig2, ax2 = plt.subplots(figsize=(9, 3))
    hist = feat[['Date', 'Market_Price_JPY_per_kg']].copy().tail(14)
    ax2.plot(hist['Date'], hist['Market_Price_JPY_per_kg'], label='Recent actuals')
    ax2.plot(pd.to_datetime(fc_df['Date']), fc_df['Predicted_Price'], label='Forecast', marker='o')
    ax2.set_title("Recent actuals + forecast")
    ax2.legend()
    st.pyplot(fig2)

st.markdown("---")
st.write(
    "Notes: This is a lightweight prototype. For production, add: (1) robust rolling-origin CV, "
    "(2) prediction intervals, (3) hyperparameter tuning, (4) an API so farmers get forecasts without re-training locally."
)
