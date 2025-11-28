# smartfarm_interactive.py
# Polished Streamlit app UI for SmartFarm price forecasting
# Single-file app: clean layout, metrics, tabs, downloads, and nice styling.
#
# Requirements (minimal):
# streamlit, pandas, numpy, scikit-learn, joblib
# Optional: xgboost (will be used if available)

import streamlit as st
import pandas as pd
import numpy as np
from datetime import timedelta
from sklearn.metrics import mean_absolute_error
import joblib
import os
import io

# ----- App config -----
st.set_page_config(page_title="SmartFarm — Farmer Forecast", page_icon="🌾", layout="wide")
APP_TITLE = "SmartFarm — Price Forecast"
DATA_PATH_DEFAULT = "realistic_dummy_dataset.csv"  # change to your repo/raw URL if needed
OUTPUT_DIR = "analysis_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)
MODEL_SAVE_PATH = os.path.join(OUTPUT_DIR, "model_{}.joblib")

# ----- Attempt optional ML backends -----
USE_XGBOOST = False
try:
    import xgboost as xgb
    USE_XGBOOST = True
except Exception:
    USE_XGBOOST = False

# ----- CSS styling -----
st.markdown(
    """
    <style>
    /* General */
    .stApp { font-family: 'Segoe UI', Roboto, Arial, sans-serif; }
    .header { display: flex; align-items: center; gap: 16px; }
    .brand-title { font-size:28px; font-weight:700; margin:0; }
    .brand-sub { color: #6b7280; margin:0; font-size:13px; }

    /* Card like boxes */
    .card { background: linear-gradient(90deg, #ffffff, #fbfbff); padding:14px; border-radius:12px; box-shadow: 0 6px 18px rgba(16,24,40,0.06); }
    .metric { display:flex; align-items:baseline; gap:8px; }

    /* Narrow table */
    .small { font-size:13px; color:#374151; }

    /* Tweak sidebar */
    [data-testid="stSidebar"] { background-color: #f8fafc; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ----- Header -----
with st.container():
    col1, col2 = st.columns([0.12, 0.88])
    with col1:
        st.markdown("<div style='font-size:40px'>🌾</div>", unsafe_allow_html=True)
    with col2:
        st.markdown("<div class='header'><div>"
                    f"<h1 class='brand-title'>{APP_TITLE}</h1>"
                    f"<div class='brand-sub'>Simple, actionable price forecasts for farmers — pick produce, enter weather, and forecast.</div>"
                    "</div></div>", unsafe_allow_html=True)

st.markdown("---")

# ----- Sidebar controls -----
st.sidebar.header("Inputs & settings")
uploaded = st.sidebar.file_uploader("Upload CSV (optional)", type=["csv"])
use_default = st.sidebar.checkbox("Use default dataset in repo", value=True)
st.sidebar.markdown("**Model / compute**")
use_xgb_checkbox = st.sidebar.checkbox("Prefer XGBoost (if available)", value=USE_XGBOOST)
st.sidebar.markdown("---")
st.sidebar.markdown("Tips:")
st.sidebar.write("• CSV should include: Date, Produce_Name, Market_Price_JPY_per_kg.\n"
                 "• Optional: Temperature_C, Precipitation_mm, Volume_Sold_kg, Quality_Grade")

# ----- Data loader -----
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
        return None
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    return df

df = load_data(uploaded, use_default)
if df is None:
    st.warning("No dataset loaded. Upload a CSV or enable the default dataset (ensure path is correct).")
    st.stop()

# quick sanity
required_cols = {'Date', 'Produce_Name', 'Market_Price_JPY_per_kg'}
if not required_cols.issubset(set(df.columns)):
    st.error(f"Dataset missing required columns: {required_cols - set(df.columns)}")
    st.stop()

# ----- App body in tabs -----
tab_data, tab_train, tab_forecast, tab_about = st.tabs(["Data", "Train & Validate", "Forecast (interactive)", "About"])

# ---------- Data tab ----------
with tab_data:
    st.subheader("Dataset snapshot")
    c1, c2, c3 = st.columns([1,1,1])
    with c1:
        st.metric("Rows", df.shape[0])
    with c2:
        st.metric("Unique produces", int(df['Produce_Name'].nunique()))
    with c3:
        st.metric("Date range", f"{df['Date'].min().date()} → {df['Date'].max().date()}")

    # sample table
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.write(df.head(12))
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("### Quick correlations (numerical)")
    try:
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        corr = df[num_cols].corr().round(3)
        st.dataframe(corr.style.background_gradient(axis=None))
    except Exception:
        st.write("Correlation unavailable (small data or non-numeric).")

# ---------- Train & Validate tab ----------
with tab_train:
    st.subheader("Model training & quick validation")
    produce_list = sorted(df['Produce_Name'].unique().tolist())
    selected_produce = st.selectbox("Choose produce to train on", produce_list, index=0)

    st.markdown("Small model training for the chosen produce. This can run in seconds for reasonable data sizes.")
    st.info("Tip: Use 'Forecast' tab after training to run interactive forecasts using farmer-provided weather.")

    # small options
    col_a, col_b = st.columns([1,1])
    with col_a:
        test_days = st.number_input("Validation window (days)", min_value=30, max_value=730, value=180)
    with col_b:
        n_estimators = st.number_input("RF trees (if using RF)", min_value=50, max_value=1000, value=200, step=50)

    # filter produce
    df_prod = df[df['Produce_Name'] == selected_produce].copy().reset_index(drop=True)

    def create_features(df_prod):
        d = df_prod.copy().sort_values('Date').reset_index(drop=True)
        d['year'] = d['Date'].dt.year
        d['month'] = d['Date'].dt.month
        d['dayofweek'] = d['Date'].dt.dayofweek
        d['price_lag_1'] = d['Market_Price_JPY_per_kg'].shift(1)
        d['price_lag_7'] = d['Market_Price_JPY_per_kg'].shift(7)
        d['price_lag_30'] = d['Market_Price_JPY_per_kg'].shift(30)
        d['roll_mean_7'] = d['Market_Price_JPY_per_kg'].shift(1).rolling(7).mean()
        d['roll_std_7'] = d['Market_Price_JPY_per_kg'].shift(1).rolling(7).std()
        if 'Temperature_C' in d.columns:
            d['temp_roll_3'] = d['Temperature_C'].shift(1).rolling(3).mean()
        if 'Precipitation_mm' in d.columns:
            d['precip_roll_3'] = d['Precipitation_mm'].shift(1).rolling(3).sum()
        d = d.dropna(subset=['price_lag_1']).reset_index(drop=True)
        return d

    feat = create_features(df_prod)
    st.write(f"Rows after feature engineering: {feat.shape[0]}")

    split_date = feat['Date'].max() - pd.Timedelta(days=int(test_days))
    train = feat[feat['Date'] < split_date].copy()
    test = feat[feat['Date'] >= split_date].copy()

    st.write(f"Train rows: {train.shape[0]} | Test rows: {test.shape[0]} (split: {split_date.date()})")

    # choose features automatically
    exclude = ['Date', 'Produce_Name', 'Market_Price_JPY_per_kg', 'Category', 'Season', 'Quality_Grade']
    candidate_features = [c for c in feat.columns if c not in exclude and np.issubdtype(feat[c].dtype, np.number)]
    st.write("Features used:", candidate_features)

    if st.button("Train quick model now"):
        # prepare data
        X_train = train[candidate_features].fillna(0)
        y_train = train['Market_Price_JPY_per_kg']
        X_test = test[candidate_features].fillna(0)
        y_test = test['Market_Price_JPY_per_kg']

        # train
        if use_xgb_checkbox and USE_XGBOOST:
            dtrain = xgb.DMatrix(X_train, label=y_train)
            dtest = xgb.DMatrix(X_test, label=y_test)
            params = {'objective': 'reg:squarederror', 'eval_metric': 'rmse', 'seed': 42}
            bst = xgb.train(params, dtrain, num_boost_round=200, evals=[(dtest, 'eval')], early_stopping_rounds=25, verbose_eval=False)
            predict_fn = lambda X: bst.predict(xgb.DMatrix(X))
            model = bst
            model_name = "XGBoost"
        else:
            from sklearn.ensemble import RandomForestRegressor
            rf = RandomForestRegressor(n_estimators=int(n_estimators), random_state=42, n_jobs=-1)
            rf.fit(X_train, y_train)
            predict_fn = lambda X: rf.predict(X)
            model = rf
            model_name = "RandomForest"

        preds = predict_fn(X_test)
        mae_v = mean_absolute_error(y_test, preds)
        st.success(f"Trained {model_name}. Test MAE: {mae_v:.3f} JPY/kg")

        # show metrics nicely
        m1, m2, m3 = st.columns(3)
        m1.metric("Test MAE", f"{mae_v:.3f} JPY/kg")
        m2.metric("Train rows", f"{X_train.shape[0]}")
        m3.metric("Test rows", f"{X_test.shape[0]}")

        # save model
        try:
            joblib.dump(model, MODEL_SAVE_PATH.format(selected_produce))
            st.info("Model saved to analysis_outputs/")
        except Exception as e:
            st.write("Could not save model:", e)

        # show importance if RF
        if model_name == "RandomForest":
            try:
                importances = pd.Series(model.feature_importances_, index=candidate_features).sort_values(ascending=False)
                st.subheader("Feature importance")
                st.bar_chart(importances.head(12))
                # export CSV
                buf = io.StringIO()
                importances.to_csv(buf, header=True)
                st.download_button("Download feature importance CSV", data=buf.getvalue(), file_name=f"fi_{selected_produce}.csv")
            except Exception:
                pass

# ---------- Forecast tab ----------
with tab_forecast:
    st.subheader("Interactive forecast for farmers")
    produce_list = sorted(df['Produce_Name'].unique().tolist())
    prod = st.selectbox("Pick produce to forecast", produce_list, index=0)

    df_prod_all = df[df['Produce_Name'] == prod].copy().reset_index(drop=True)
    st.markdown(f"**Available data:** {df_prod_all['Date'].min().date()} → {df_prod_all['Date'].max().date()} | Rows: {df_prod_all.shape[0]}")

    last_row = df_prod_all.sort_values('Date').iloc[-1]
    last_temp = float(last_row['Temperature_C']) if 'Temperature_C' in df_prod_all.columns else 0.0
    last_precip = float(last_row['Precipitation_mm']) if 'Precipitation_mm' in df_prod_all.columns else 0.0

    st.subheader("Farmer inputs")
    c1, c2, c3 = st.columns([1,1,1])
    with c1:
        temp_input = st.number_input("Today's temperature (°C)", value=float(last_temp), format="%.2f")
    with c2:
        precip_input = st.number_input("Today's precipitation (mm)", value=float(last_precip), format="%.2f")
    with c3:
        horizon = st.slider("Forecast horizon (days)", 1, 14, 7)

    # prepare features
    feat_full = create_features(df_prod_all)
    if feat_full.shape[0] < 7:
        st.warning("Not enough history to forecast reliably. Add more data.")
    else:
        # try load model if exists
        model_path = MODEL_SAVE_PATH.format(prod)
        model_loaded = None
        if os.path.exists(model_path):
            try:
                model_loaded = joblib.load(model_path)
                st.info(f"Loaded saved model for {prod}")
            except Exception:
                model_loaded = None

        # if no model, train lightweight RF quickly
        if model_loaded is None:
            st.info("No saved model found — training quick RandomForest (fast).")
            # quick train on full history excluding very recent horizon
            split_date = feat_full['Date'].max() - pd.Timedelta(days=30)
            tr = feat_full[feat_full['Date'] < split_date].copy()
            te = feat_full[feat_full['Date'] >= split_date].copy()
            candidate_features = [c for c in feat_full.columns if c not in ['Date','Produce_Name','Market_Price_JPY_per_kg','Category','Season','Quality_Grade'] and np.issubdtype(feat_full[c].dtype, np.number)]
            from sklearn.ensemble import RandomForestRegressor
            rf = RandomForestRegressor(n_estimators=150, random_state=42, n_jobs=-1)
            if tr.shape[0] >= 20:
                rf.fit(tr[candidate_features].fillna(0), tr['Market_Price_JPY_per_kg'])
                model_loaded = rf
                predict_fn = lambda X: model_loaded.predict(X)
            else:
                st.warning("Not enough data to train a fallback model. Showing simple baseline (yesterday's price).")
                model_loaded = None

        if model_loaded is None:
            # baseline forecast: repeat last price
            last_price = float(feat_full.sort_values('Date').iloc[-1]['Market_Price_JPY_per_kg'])
            fc = [{'Date': (feat_full['Date'].max() + pd.Timedelta(days=i+1)).date(), 'Predicted_Price': last_price} for i in range(horizon)]
            fc_df = pd.DataFrame(fc)
            st.warning("Using baseline (last-price) forecast due to insufficient model/data.")
            st.dataframe(fc_df)
            st.download_button("Download forecast CSV", data=fc_df.to_csv(index=False), file_name=f"{prod}_forecast.csv")
        else:
            # do recursive forecast using farmer-provided constant weather
            last_row = feat_full.sort_values('Date').iloc[-1].copy()
            recent_temps = list(feat_full.sort_values('Date')['Temperature_C'].dropna().values[-3:]) if 'Temperature_C' in feat_full else []
            recent_precips = list(feat_full.sort_values('Date')['Precipitation_mm'].dropna().values[-3:]) if 'Precipitation_mm' in feat_full else []
            user_temp = float(temp_input)
            user_precip = float(precip_input)
            candidate_features = [c for c in feat_full.columns if c not in ['Date','Produce_Name','Market_Price_JPY_per_kg','Category','Season','Quality_Grade'] and np.issubdtype(feat_full[c].dtype, np.number)]

            current = last_row.copy()
            rows = []
            for i in range(horizon):
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
                        val = feat_full[feat_full['Date'] == lookup_date]['Market_Price_JPY_per_kg']
                        if not val.empty:
                            fv[f] = float(val.values[0])
                        else:
                            fv[f] = float(current.get('Market_Price_JPY_per_kg', 0))
                    else:
                        fv[f] = float(current.get(f, 0) if f in current else 0)
                fv_df = pd.DataFrame([fv]).fillna(0)
                pred_next = float(model_loaded.predict(fv_df)[0])
                next_date = current['Date'] + pd.Timedelta(days=1)
                rows.append({'Date': next_date.date(), 'Predicted_Price': pred_next})
                # update queues/state
                recent_temps.append(user_temp); recent_temps = recent_temps[-3:]
                recent_precips.append(user_precip); recent_precips = recent_precips[-3:]
                current['Date'] = next_date
                current['Market_Price_JPY_per_kg'] = pred_next
                current['price_lag_1'] = pred_next

            fc_df = pd.DataFrame(rows)
            st.subheader("Forecast (next days)")
            st.dataframe(fc_df)

            # quick KPIs
            today_price = float(feat_full.sort_values('Date').iloc[-1]['Market_Price_JPY_per_kg'])
            change_pct = (fc_df['Predicted_Price'].iloc[0] - today_price) / max(1e-6, today_price) * 100
            k1, k2, k3 = st.columns(3)
            k1.metric("Price today (JPY/kg)", f"{today_price:.2f}")
            k2.metric("Tomorrow (pred.)", f"{fc_df['Predicted_Price'].iloc[0]:.2f}", delta=f"{change_pct:.2f}%")
            k3.metric("Horizon", f"{horizon} days")

            # download
            st.download_button("Download forecast CSV", data=fc_df.to_csv(index=False), file_name=f"{prod}_forecast.csv", mime="text/csv")

            # show plots (streamlit charts)
            st.subheader("Visual: recent actuals + forecast")
            hist = feat_full[['Date', 'Market_Price_JPY_per_kg']].copy().tail(14)
            hist_plot = hist.set_index('Date').rename(columns={'Market_Price_JPY_per_kg': 'actual'}).resample('D').mean().ffill()
            fc_plot = pd.Series(fc_df['Predicted_Price'].values, index=pd.to_datetime(fc_df['Date']))
            # build combined df
            full_index = pd.date_range(hist_plot.index.min(), fc_plot.index.max(), freq='D')
            actual_col = hist_plot['actual'].reindex(full_index).ffill()
            forecast_col = pd.Series([np.nan]*len(full_index), index=full_index)
            forecast_col.loc[fc_plot.index] = fc_plot.values
            plot_df = pd.DataFrame({"actual": actual_col, "forecast": forecast_col})
            st.line_chart(plot_df)

# ---------- About tab ----------
with tab_about:
    st.header("About SmartFarm")
    st.write(
        "SmartFarm is a lightweight forecasting tool built for the Talent for Japan 2025 challenge.\n\n"
        "This polished UI helps farmers quickly check short-term price expectations and make simple decisions: hold, sell, or wait.\n\n"
        "For production deployment: (1) host pre-trained models on S3 or a small API, (2) add authentication, (3) add prediction intervals and robust CV."
    )

    st.markdown("**Quick links & notes**")
    st.markdown("- Keep datasets small in the repo (< 50MB) or host externally (S3/Google Drive).")
    st.markdown("- For cloud deployment, ensure `requirements.txt` has compatible wheel versions (numpy/pandas/scikit-learn).")

st.markdown("---")
st.caption("Made by Sirram Charan.")
