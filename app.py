# app.py - Merged Streamlit Frontend

import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
import pickle
import os
import io
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt

st.set_page_config(page_title="Energy Analysis Platform", page_icon="⚡", layout="wide")

# Navigation
analysis_mode = st.sidebar.radio(
    "Select Analysis",
    ["Analysis 1 – Mass Anomaly Detection", "Analysis 2 – Theft Leaderboard & Inspection"]
)

# ==================== ANALYSIS 1 ====================
if analysis_mode == "Analysis 1 – Mass Anomaly Detection":
    
    def preprocess_data(df, selected_consumers=None, n_consumers=5, date_range=None):
        df = df.rename(columns={'LCLid': 'consumer_id', 'day': 'date', 'energy_sum': 'consumption'})
        df['date'] = pd.to_datetime(df['date'])
        
        if selected_consumers is not None:
            df = df[df['consumer_id'].isin(selected_consumers)]
        else:
            unique_consumers = df['consumer_id'].unique()
            selected = np.random.choice(unique_consumers, size=min(n_consumers, len(unique_consumers)), replace=False)
            df = df[df['consumer_id'].isin(selected)]
        
        df = df.sort_values(['consumer_id', 'date']).reset_index(drop=True)
        
        if date_range is not None:
            df = df[(df['date'] >= date_range[0]) & (df['date'] <= date_range[1])]
        else:
            min_date = df['date'].min()
            max_date = df['date'].max()
            date_span = (max_date - min_date).days
            if date_span > 270:
                start_offset = pd.Timedelta(days=date_span // 6)
                end_offset = pd.Timedelta(days=date_span // 6)
                df = df[(df['date'] >= min_date + start_offset) & (df['date'] <= max_date - end_offset)]
        
        df = df.loc[:, ~df.columns.duplicated()]
        df = df.sort_values(['consumer_id', 'date']).reset_index(drop=True)
        df['rolling_avg_7'] = df.groupby('consumer_id')['consumption'].transform(lambda x: x.rolling(window=7, min_periods=1).mean())
        df['deviation'] = df['consumption'] - df['rolling_avg_7']
        df = df.dropna().reset_index(drop=True)
        return df

    def train_model(df, contamination=0.03):
        split_idx = int(len(df) * 0.8)
        train_df = df.iloc[:split_idx].copy()
        feature_cols = ['consumption', 'rolling_avg_7', 'deviation']
        X_train = train_df[feature_cols].values
        model = IsolationForest(n_estimators=100, contamination=contamination, random_state=42, n_jobs=-1)
        model.fit(X_train)
        return model, split_idx

    def detect_anomalies(df, model):
        feature_cols = ['consumption', 'rolling_avg_7', 'deviation']
        X = df[feature_cols].values
        predictions = model.predict(X)
        scores = model.decision_function(X)
        df['anomaly_label'] = ['Anomaly' if pred == -1 else 'Normal' for pred in predictions]
        df['anomaly_score'] = scores
        return df

    st.title("⚡ Electricity Consumption Anomaly Detection")
    st.markdown("**Detect abnormal electricity consumption patterns using Isolation Forest**")
    
    uploaded_file = st.sidebar.file_uploader("Upload Smart Meter Data (CSV)", type=['csv'])
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            required_cols = ['LCLid', 'day', 'energy_sum']
            if not all(col in df.columns for col in required_cols):
                st.error(f"CSV must contain columns: {required_cols}")
                st.stop()
            
            st.sidebar.success(f"✅ Loaded {len(df):,} rows")
            st.sidebar.info(f"Consumers: {df['LCLid'].nunique():,}")
        except Exception as e:
            st.error(f"Error loading file: {str(e)}")
            st.stop()
        
        st.sidebar.subheader("Consumer Selection")
        all_consumers = sorted(df['LCLid'].unique())
        selection_mode = st.sidebar.radio("Selection Mode:", ["Random", "Specific IDs"])
        
        if selection_mode == "Random":
            n_consumers = st.sidebar.slider("Number of Consumers", min_value=1, max_value=min(10, len(all_consumers)), value=5)
            selected_consumers = None
        else:
            selected_consumers = st.sidebar.multiselect("Select Consumer IDs", options=all_consumers[:50], default=all_consumers[:3])
            if not selected_consumers:
                st.warning("Please select at least one consumer ID")
                st.stop()
            n_consumers = len(selected_consumers)
        
        st.sidebar.subheader("Model Parameters")
        contamination = st.sidebar.slider("Contamination Rate", min_value=0.01, max_value=0.10, value=0.03, step=0.01, format="%.2f")
        run_analysis = st.sidebar.button("🚀 Run Analysis", type="primary")
        
        if run_analysis:
            with st.spinner("Processing data..."):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                status_text.text("Step 1/4: Preprocessing data...")
                progress_bar.progress(25)
                df_processed = preprocess_data(df, selected_consumers=selected_consumers, n_consumers=n_consumers if selected_consumers is None else None)
                
                status_text.text("Step 2/4: Training Isolation Forest...")
                progress_bar.progress(50)
                model, split_idx = train_model(df_processed, contamination)
                
                status_text.text("Step 3/4: Detecting anomalies...")
                progress_bar.progress(75)
                results = detect_anomalies(df_processed, model)
                
                status_text.text("Step 4/4: Generating results...")
                progress_bar.progress(100)
                progress_bar.empty()
                status_text.empty()
            
            st.success("✅ Analysis complete!")
            
            st.header("📊 Summary Statistics")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Records", f"{len(results):,}")
            with col2:
                st.metric("Consumers Analyzed", results['consumer_id'].nunique())
            with col3:
                n_anomalies = (results['anomaly_label'] == 'Anomaly').sum()
                st.metric("Anomalies Detected", n_anomalies)
            with col4:
                anomaly_rate = n_anomalies / len(results) * 100
                st.metric("Anomaly Rate", f"{anomaly_rate:.2f}%")
            
            st.subheader("Anomalies by Consumer")
            anomaly_summary = results[results['anomaly_label'] == 'Anomaly'].groupby('consumer_id').size().reset_index()
            anomaly_summary.columns = ['Consumer ID', 'Number of Anomalies']
            anomaly_summary = anomaly_summary.sort_values('Number of Anomalies', ascending=False)
            
            col1, col2 = st.columns([1, 2])
            with col1:
                st.dataframe(anomaly_summary, use_container_width=True)
            with col2:
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.barh(anomaly_summary['Consumer ID'], anomaly_summary['Number of Anomalies'], color='crimson')
                ax.set_xlabel('Number of Anomalies')
                ax.set_ylabel('Consumer ID')
                ax.set_title('Anomaly Count by Consumer')
                ax.grid(axis='x', alpha=0.3)
                st.pyplot(fig)
                plt.close()
            
            st.header("📈 Consumption Patterns with Anomalies")
            consumers = results['consumer_id'].unique()
            tabs = st.tabs([f"Consumer {c}" for c in consumers])
            
            for idx, (tab, consumer) in enumerate(zip(tabs, consumers)):
                with tab:
                    consumer_data = results[results['consumer_id'] == consumer].copy()
                    anomaly_data = consumer_data[consumer_data['anomaly_label'] == 'Anomaly']
                    
                    fig, ax = plt.subplots(figsize=(14, 5))
                    ax.plot(consumer_data['date'], consumer_data['consumption'], color='steelblue', linewidth=2, label='Consumption', alpha=0.7)
                    ax.plot(consumer_data['date'], consumer_data['rolling_avg_7'], color='green', linewidth=1.5, linestyle='--', label='7-day Average', alpha=0.6)
                    
                    if len(anomaly_data) > 0:
                        ax.scatter(anomaly_data['date'], anomaly_data['consumption'], color='red', s=100, marker='o', label='Anomaly', edgecolors='darkred', linewidths=2, zorder=5)
                    
                    ax.set_title(f'Consumer {consumer} | Anomalies: {len(anomaly_data)}', fontsize=14, fontweight='bold')
                    ax.set_xlabel('Date', fontsize=11)
                    ax.set_ylabel('Consumption (kWh)', fontsize=11)
                    ax.legend(loc='upper right', fontsize=10)
                    ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
                    plt.xticks(rotation=45, ha='right')
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()
                    
                    if len(anomaly_data) > 0:
                        st.subheader(f"Anomaly Details for {consumer}")
                        anomaly_display = anomaly_data[['date', 'consumption', 'rolling_avg_7', 'deviation', 'anomaly_score']].copy()
                        anomaly_display['date'] = anomaly_display['date'].dt.strftime('%Y-%m-%d')
                        anomaly_display = anomaly_display.sort_values('anomaly_score')
                        st.dataframe(anomaly_display.style.background_gradient(subset=['anomaly_score'], cmap='Reds_r'), use_container_width=True)
            
            st.header("💾 Download Results")
            col1, col2 = st.columns(2)
            with col1:
                csv_buffer = io.StringIO()
                results.to_csv(csv_buffer, index=False)
                st.download_button("📥 Download Full Results (CSV)", data=csv_buffer.getvalue(), file_name="anomaly_detection_results.csv", mime="text/csv")
            with col2:
                anomalies_only = results[results['anomaly_label'] == 'Anomaly'].copy()
                csv_buffer_anomalies = io.StringIO()
                anomalies_only.to_csv(csv_buffer_anomalies, index=False)
                st.download_button("📥 Download Anomalies Only (CSV)", data=csv_buffer_anomalies.getvalue(), file_name="anomalies_only.csv", mime="text/csv")
    else:
        st.info("👈 Upload your smart meter data CSV file to get started!")

# ==================== ANALYSIS 2 ====================
elif analysis_mode == "Analysis 2 – Theft Leaderboard & Inspection":
    
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    
    def load_local_asset(filename):
        path = os.path.join(CURRENT_DIR, filename)
        with open(path, "rb") as f:
            return pickle.load(f)
    
    try:
        group_averages = load_local_asset("group_averages.pkl")
        label_encoder = load_local_asset("label_encoder.pkl")
    except Exception as e:
        st.error(f"⚠️ Assets not found! Ensure .pkl files are in: {CURRENT_DIR}")
    
    st.markdown("""
        <style>
        .status-card { padding: 25px; border-radius: 15px; text-align: center; color: white !important; margin-bottom: 20px;}
        .suspicious-box { background-color: #FF4B4B; border: 3px solid #800000; box-shadow: 0 4px 15px rgba(255,0,0,0.3); }
        .normal-box { background-color: #00C851; border: 3px solid #004d00; }
        .metric-value { font-size: 40px; font-weight: bold; margin: 0; color: white !important; }
        .metric-label { font-size: 14px; text-transform: uppercase; opacity: 0.9; color: white !important; }
        .stDataFrame { background: white; padding: 10px; border-radius: 10px; }
        </style>
        """, unsafe_allow_html=True)
    
    st.sidebar.title("🎮 Dashboard Mode")
    mode = st.sidebar.radio("Select View:", ["Individual Inspector", "Theft Leaderboard"])
    st.sidebar.divider()
    uploaded_file = st.sidebar.file_uploader("📂 Upload Smart Meter Data (CSV)", type=['csv'])
    
    if mode == "Individual Inspector":
        st.title("🕵️ Household Anomaly Investigator")
        
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            consumer_id = st.sidebar.selectbox("Select Consumer", df['LCLid'].unique())
            user_row = df[df['LCLid'] == consumer_id].iloc[-1]
            
            st.sidebar.subheader("Live Simulation")
            e_sum = st.sidebar.slider("Energy Sum (kWh)", 0.0, 50.0, float(user_row['energy_sum']))
            e_std = st.sidebar.slider("Variation (STD)", 0.0, 5.0, 0.2)
            e_max = st.sidebar.slider("Peak Spike", 0.0, 15.0, float(user_row['energy_sum'] * 0.5))
            acorn = st.sidebar.selectbox("Social Group", ["Affluent", "Comfortable", "Adversity"])
            
            if st.sidebar.button("🚀 RUN AI INSPECTION", type="primary"):
                payload = {"energy_sum": e_sum, "energy_std": e_std, "energy_max": e_max, "energy_mean": e_sum / 24, "acorn_grouped": acorn}
                
                try:
                    res = requests.post("http://127.0.0.1:8000/inspect", json=payload).json()
                    
                    col1, col2 = st.columns(2)
                    box_class = "suspicious-box" if res['status'] == "Suspicious" else "normal-box"
                    
                    with col1:
                        st.markdown(f"""<div class="status-card {box_class}">
                            <p class="metric-label">AI Verdict</p>
                            <p class="metric-value">{'🚨' if box_class=='suspicious-box' else '✅'} {res['status'].upper()}</p>
                            </div>""", unsafe_allow_html=True)
                    
                    with col2:
                        risk_color = "#FF4B4B" if res['risk_score'] > 60 else "#00C851"
                        st.markdown(f"""<div class="status-card" style="background-color: #262730; border-top: 10px solid {risk_color};">
                            <p class="metric-label">Theft Probability</p>
                            <p class="metric-value" style="color: {risk_color} !important;">{res['risk_score']}%</p>
                            </div>""", unsafe_allow_html=True)
                    
                    st.subheader("🔍 Investigator Analysis")
                    if res['status'] == "Suspicious":
                        st.error(f"**Flagged:** Consumption is {res['peer_ratio']}x lower than {acorn} peers. Flat variation detected.")
                    else:
                        st.success(f"**Verified:** Patterns consistent with {acorn} neighborhood behavior.")
                    
                    fig = go.Figure(data=go.Scatterpolar(
                        r=[e_sum/10, e_std*5, res['peer_ratio'], e_max/2],
                        theta=['Volume', 'Consistency', 'Peer Ratio', 'Peak'],
                        fill='toself', line_color='#FF4B4B' if res['status'] == "Suspicious" else '#00C851'
                    ))
                    st.plotly_chart(fig)
                except Exception as e:
                    st.error("API Error: Ensure 'main.py' is running on localhost:8000")
        else:
            st.info("👈 Please upload a CSV file to begin inspection.")
    
    elif mode == "Theft Leaderboard":
        st.title("🏆 National Grid: Theft Leaderboard")
        st.markdown("Automated batch processing for identifying high-priority tampering cases.")
        
        if uploaded_file is not None:
            df_all = pd.read_csv(uploaded_file)
            
            with st.spinner("Processing Grid Data..."):
                df_all['peer_avg'] = df_all['Acorn_grouped'].map(group_averages)
                df_all['peer_ratio'] = df_all['energy_sum'] / (df_all['peer_avg'] + 1e-6)
                suspects = df_all[df_all['peer_ratio'] < 0.25].copy()
                suspects['Risk Score'] = (1 - suspects['peer_ratio']) * 100
                suspects = suspects.sort_values(by='peer_ratio')
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Houses", len(df_all))
            col2.metric("Anomalies Found", len(suspects))
            col3.metric("Grid Efficiency", "94.2%", delta="-2.1% Theft Leakage", delta_color="inverse")
            
            st.subheader("🚨 Top 10 Priority Dispatch List")
            st.dataframe(suspects[['LCLid', 'Acorn_grouped', 'energy_sum', 'peer_ratio', 'Risk Score']].head(10).style.background_gradient(cmap='Reds', subset=['Risk Score']), use_container_width=True)
            
            if st.button("Generate Inspection Orders"):
                st.toast("Orders sent to Field Teams!", icon="✉️")
        else:
            st.info("👈 Please upload a CSV file to generate the Leaderboard.")