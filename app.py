import streamlit as st
import pandas as pd
import numpy as np
import json
import joblib
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt

# ------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------

st.set_page_config(
    page_title="Grid Battery AI",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ------------------------------------------------
# CUSTOM CSS (HIGH-END LIGHT THEME)
# ------------------------------------------------

st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap" rel="stylesheet">

<style>
    /* --- ENFORCE LIGHT THEME --- */
    .main, div[data-testid="stSidebar"] {
        background-color: #f1f5f9 !important; /* Slate 100 */
        color: #334155 !important;
    }

    /* --- GLOBAL FONTS --- */
    html, body, [class*="css"]  {
        font-family: 'Poppins', sans-serif;
    }

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* --- SIDEBAR STYLING --- */
    div[data-testid="stSidebar"] {
        background-color: #ffffff !important;
        border-right: 1px solid #e2e8f0 !important;
        padding-top: 0px;
    }

    .sidebar-header-container {
        background: linear-gradient(135deg, #fbbf24 0%, #f59e0b 100%);
        padding: 2rem 1.5rem;
        border-bottom: 1px solid #e2e8f0;
        margin-bottom: 1rem;
        color: white !important;
    }
    .sidebar-title {
        font-size: 1.4rem;
        font-weight: 800;
        margin: 0;
        letter-spacing: -0.5px;
    }
    .sidebar-subtitle {
        font-size: 0.75rem;
        margin-top: 5px;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 1px;
        opacity: 0.9;
    }

    /* Navigation Items */
    .stRadio [role="radiogroup"] {
        flex-direction: column;
        gap: 0.5rem;
        padding: 0 15px;
    }

    .stRadio [role="radiogroup"] > label {
        background-color: transparent;
        border: 1px solid transparent;
        padding: 12px 15px;
        border-radius: 10px;
        color: #64748b;
        transition: all 0.2s ease;
        margin-bottom: 0px;
        cursor: pointer;
        display: flex;
        align-items: center;
        font-size: 0.95rem;
        font-weight: 500;
    }

    .stRadio [role="radiogroup"] > label:hover {
        background-color: #fffbeb; /* Light Amber */
        color: #b45309;
        padding-left: 20px; /* Slide effect */
    }

    .stRadio [role="radiogroup"] > label > div:first-child { display: none; }

    .stRadio [role="radiogroup"] > label[data-baseweb="radio-checked"] {
        background-color: #f59e0b; /* Amber 500 */
        color: white;
        font-weight: 700;
        box-shadow: 0 4px 6px -1px rgba(245, 158, 11, 0.4);
        border-color: #f59e0b;
    }

    /* --- DASHBOARD COMPONENTS --- */
    .dashboard-card {
        background-color: #ffffff;
        border-radius: 16px;
        padding: 1.5rem;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1), 0 1px 2px 0 rgba(0, 0, 0, 0.06);
        border: 1px solid #f1f5f9;
        margin-bottom: 1.5rem;
        height: 100%;
        display: flex;
        flex-direction: column;
    }

    .card-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 1.5rem;
        border-bottom: 1px solid #f1f5f9;
        padding-bottom: 1rem;
    }
    .card-title {
        font-size: 1.1rem;
        font-weight: 700;
        color: #1e293b;
        margin: 0;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    .card-icon {
        background: #fef3c7;
        color: #d97706;
        padding: 8px;
        border-radius: 8px;
        font-size: 1.2rem;
    }

    /* Header Area */
    .main-header {
        background: white;
        padding: 2rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        display: flex;
        justify-content: space-between;
        align-items: center;
        border-left: 5px solid #f59e0b;
    }
    .main-header h1 { margin: 0; color: #0f172a; font-size: 1.8rem; font-weight: 800; line-height: 1.2; }
    .main-header p { margin: 5px 0 0 0; color: #64748b; font-size: 0.9rem; }
    .status-pill {
        background: #dcfce7;
        color: #166534;
        padding: 5px 15px;
        border-radius: 50px;
        font-size: 0.8rem;
        font-weight: 700;
        border: 1px solid #bbf7d0;
    }

    /* KPI Cards */
    .kpi-card {
        background: white;
        padding: 1.5rem;
        border-radius: 16px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        border: 1px solid #e2e8f0;
        display: flex;
        align-items: center;
        gap: 15px;
        transition: transform 0.2s;
    }
    .kpi-card:hover { transform: translateY(-3px); }
    .kpi-icon { font-size: 2rem; color: #f59e0b; background: #fffbeb; padding: 10px; border-radius: 12px; }
    .kpi-text h4 { margin: 0; color: #64748b; font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.5px; font-weight: 600; }
    .kpi-text h2 { margin: 5px 0 0 0; color: #0f172a; font-size: 1.8rem; font-weight: 800; }

    /* Result Card */
    .result-box {
        background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%);
        color: white;
        padding: 3rem;
        border-radius: 24px;
        text-align: center;
        box-shadow: 0 20px 25px -5px rgba(245, 158, 11, 0.4);
        margin-top: 2rem;
        border: 1px solid rgba(255,255,255,0.2);
    }

    /* Buttons */
    .stButton > button {
        background-color: #f59e0b !important;
        color: white !important;
        border: none;
        border-radius: 10px;
        padding: 0.8rem 2rem;
        font-weight: 600;
        font-size: 1rem;
        box-shadow: 0 4px 6px -1px rgba(245, 158, 11, 0.3);
        transition: all 0.2s;
    }
    .stButton > button:hover {
        background-color: #d97706 !important;
        transform: translateY(-2px);
        box-shadow: 0 10px 15px -3px rgba(245, 158, 11, 0.4);
    }
    
    button[k*="load"] {
        background-color: #e2e8f0 !important;
        color: #475569 !important;
        width: auto !important;
        padding: 0.5rem 1.5rem !important;
        font-size: 0.9rem !important;
        margin-bottom: 1rem;
    }
    button[k*="load"]:hover { background-color: #cbd5e1 !important; }

</style>
""", unsafe_allow_html=True)

# ------------------------------------------------
# LOAD MODEL & DATA
# ------------------------------------------------

@st.cache_resource
def load_resources():
    model = joblib.load("battery_rul_model.pkl")
    scaler = joblib.load("scaler.pkl")
    df = pd.read_csv("Battery_RUL.csv")

    # Clean column names
    df.columns = (
        df.columns
        .str.strip()
        .str.replace(" ", "_")
        .str.replace("(", "")
        .str.replace(")", "")
        .str.replace("-", "_")
        .str.replace(".", "")
    )
    return model, scaler, df

model, scaler, df = load_resources()
feature_cols = df.drop(columns=["RUL"]).columns.tolist()

if 'json_input' not in st.session_state: st.session_state.json_input = ""
if 'batch_json' not in st.session_state: st.session_state.batch_json = ""

# ------------------------------------------------
# SIDEBAR
# ------------------------------------------------

st.sidebar.markdown("""
    <div class="sidebar-header-container">
        <div class="sidebar-title">⚡ GridBattery AI</div>
        <div class="sidebar-subtitle">Energy Storage Analytics</div>
    </div>
""", unsafe_allow_html=True)

page = st.sidebar.radio(
    "Navigation",
    [
        "📊 Dashboard Overview",
        "🎯 Single Prediction",
        "📄 JSON API",
        "📦 Batch Process",
        "ℹ System Info"
    ],
    label_visibility="collapsed"
)

# =================================================
# DASHBOARD PAGE (ENHANCED VISUALIZATIONS)
# =================================================

if page == "📊 Dashboard Overview":

    # --- MAIN HEADER ---
    st.markdown("""
        <div class="main-header">
            <div>
                <h1>Predictive Maintenance for<br>Grid-Scale Energy Storage Systems</h1>
                <p>Real-time monitoring and health analysis dashboard</p>
            </div>
            <div class="status-pill">● System Online</div>
        </div>
    """, unsafe_allow_html=True)

    # --- KPI ROW ---
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-icon">📈</div>
                <div class="kpi-text">
                    <h4>Total Cycles</h4>
                    <h2>{len(df):,}</h2>
                </div>
            </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-icon">⏳</div>
                <div class="kpi-text">
                    <h4>Avg RUL</h4>
                    <h2>{round(df['RUL'].mean(), 0)}</h2>
                </div>
            </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-icon">🔋</div>
                <div class="kpi-text">
                    <h4>Max Cycle Index</h4>
                    <h2>{int(df['Cycle_Index'].max())}</h2>
                </div>
            </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-icon">⚠️</div>
                <div class="kpi-text">
                    <h4>Min RUL (Critical)</h4>
                    <h2>{int(df['RUL'].min())}</h2>
                </div>
            </div>
        """, unsafe_allow_html=True)

    # --- CHART ROW 1: DEGRADATION & DISCHARGE ---
    row1_col1, row1_col2 = st.columns([2, 1])

    with row1_col1:
        st.markdown("""
            <div class="dashboard-card">
                <div class="card-header">
                    <div class="card-title"><span class="card-icon">📉</span> RUL Degradation Curve</div>
                </div>
            </div>
        """, unsafe_allow_html=True)
        # Main Trend Line: Cycle Index vs RUL
        fig_trend = px.scatter(
            df.sample(2000), # Sampling for performance if dataset is huge, otherwise remove .sample()
            x="Cycle_Index", 
            y="RUL", 
            color="RUL",
            color_continuous_scale=px.colors.sequential.YlOrRd,
            title="Battery Health Trend (Cycle Index vs RUL)",
            opacity=0.6
        )
        fig_trend.update_layout(plot_bgcolor='rgba(0,0,0,0)', xaxis=dict(showgrid=False), yaxis=dict(gridcolor='#f1f5f9'))
        st.plotly_chart(fig_trend, use_container_width=True)

    with row1_col2:
        st.markdown("""
            <div class="dashboard-card">
                <div class="card-header">
                    <div class="card-title"><span class="card-icon">📊</span> Discharge Analysis</div>
                </div>
            </div>
        """, unsafe_allow_html=True)
        # Scatter: Discharge Time vs RUL
        fig_discharge = px.scatter(
            df.sample(1000),
            x="Discharge_Time_s",
            y="RUL",
            color="Max_Voltage_Dischar_V",
            title="Discharge Time vs RUL",
            size_max=10,
            opacity=0.7
        )
        fig_discharge.update_layout(plot_bgcolor='rgba(0,0,0,0)', margin=dict(l=0, r=0, t=40, b=0))
        st.plotly_chart(fig_discharge, use_container_width=True)

    # --- CHART ROW 2: DISTRIBUTIONS & HEATMAP ---
    row2_col1, row2_col2 = st.columns(2)

    with row2_col1:
        st.markdown("""
            <div class="dashboard-card">
                <div class="card-header">
                    <div class="card-title"><span class="card-icon">📦</span> Feature Distributions</div>
                </div>
            </div>
        """, unsafe_allow_html=True)
        # Box plots for key features
        key_features = ["Discharge_Time_s", "Charging_time_s", "Time_constant_current_s"]
        df_melted = df[key_features].melt(var_name='Feature', value_name='Value')
        fig_box = px.box(df_melted, y="Feature", x="Value", color="Feature", orientation='h')
        fig_box.update_layout(showlegend=False, plot_bgcolor='rgba(0,0,0,0)', xaxis=dict(gridcolor='#f1f5f9'))
        st.plotly_chart(fig_box, use_container_width=True)

    with row2_col2:
        st.markdown("""
            <div class="dashboard-card">
                <div class="card-header">
                    <div class="card-title"><span class="card-icon">🔥</span> Feature Correlation</div>
                </div>
            </div>
        """, unsafe_allow_html=True)
        # Correlation Heatmap
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="YlOrBr", linewidths=.5, ax=ax)
        st.pyplot(fig)

    # --- DATA TABLE ---
    st.markdown("""
        <div class="dashboard-card">
            <div class="card-header">
                <div class="card-title"><span class="card-icon">📑</span> Recent Data Logs</div>
            </div>
        </div>
    """, unsafe_allow_html=True)
    st.dataframe(df.tail(10), use_container_width=True)


# =================================================
# PAGE: SINGLE PREDICTION
# =================================================

elif page == "🎯 Single Prediction":

    st.markdown("""
        <div class="main-header">
            <div>
                <h1>Battery Life Prediction</h1>
                <p>Analyze specific battery parameters for health status</p>
            </div>
            <div style="font-size: 3rem;">🎯</div>
        </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("<div class='dashboard-card'><div class='card-header'><div class='card-title'>Input Parameters</div></div></div>", unsafe_allow_html=True)
        
        Cycle_Index = st.number_input("Cycle Index", min_value=1, max_value=1500, value=500)
        Discharge_Time_s = st.number_input("Discharge Time (s)", min_value=1000.0, max_value=2000.0, value=1500.0)
        Decrement_36_34V_s = st.number_input("Decrement 3.6-3.4V (s)", value=400.0)
        Max_Voltage_Dischar_V = st.number_input("Max Voltage Discharge (V)", value=3.9)
        
    with col2:
        st.markdown("<div class='dashboard-card'><div class='card-header'><div class='card-title'>Charging Metrics</div></div></div>", unsafe_allow_html=True)
        
        Min_Voltage_Charg_V = st.number_input("Min Voltage Charge (V)", value=3.5)
        Time_at_415V_s = st.number_input("Time at 4.15V (s)", value=3000.0)
        Time_constant_current_s = st.number_input("Time Constant Current (s)", value=5000.0)
        Charging_time_s = st.number_input("Charging Time (s)", value=8000.0)

    if st.button("🔮 Predict Battery Life", type="primary"):
        input_data = pd.DataFrame([[
            Cycle_Index, Discharge_Time_s, Decrement_36_34V_s, Max_Voltage_Dischar_V,
            Min_Voltage_Charg_V, Time_at_415V_s, Time_constant_current_s, Charging_time_s
        ]], columns=feature_cols)

        try:
            scaled = scaler.transform(input_data)
            prediction = model.predict(scaled)[0]

            status = "🟢 HEALTHY" if prediction > 800 else ("🟡 WARNING" if prediction > 400 else "🔴 CRITICAL")

            st.markdown(f"""
                <div class="result-box">
                    <p style="opacity:0.9; letter-spacing: 1px; font-weight: 500;">PREDICTED REMAINING USEFUL LIFE</p>
                    <h1 style="font-size: 4rem; margin: 10px 0;">{prediction:.2f}</h1>
                    <div style="background: rgba(255,255,255,0.2); display:inline-block; padding: 8px 25px; border-radius:50px; font-size: 1.2rem; font-weight: 700;">
                        {status}
                    </div>
                </div>
            """, unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Error: {e}")


# =================================================
# PAGE: JSON PREDICTION
# =================================================

elif page == "📄 JSON API":

    st.markdown("""
        <div class="main-header">
            <div>
                <h1>JSON API Input</h1>
                <p>Direct JSON payload processing for integration</p>
            </div>
            <div style="font-size: 3rem;">📄</div>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("<div class='dashboard-card'>", unsafe_allow_html=True)
    
    def load_sample_json():
        sample = {
            "Cycle_Index": 500, "Discharge_Time_s": 1500.0, "Decrement_36_34V_s": 400.0,
            "Max_Voltage_Dischar_V": 3.9, "Min_Voltage_Charg_V": 3.5,
            "Time_at_415V_s": 3000.0, "Time_constant_current_s": 5000.0, "Charging_time_s": 8000.0
        }
        st.session_state.json_input = json.dumps(sample, indent=4)

    st.button("📋 Load Sample Data", on_click=load_sample_json, key='load_json_btn')
    json_input = st.text_area("Paste JSON Payload", key="json_input", height=250)

    if st.button("🚀 Run Prediction"):
        try:
            data = json.loads(json_input)
            df_input = pd.DataFrame([data])
            df_input.columns = (
                df_input.columns.str.strip().str.replace(" ", "_").str.replace("(", "")
                .str.replace(")", "").str.replace("-", "_").str.replace(".", "")
            )
            df_input = df_input[feature_cols]
            prediction = model.predict(scaler.transform(df_input))[0]
            st.success(f"### ✅ Predicted RUL: **{prediction:.2f}** cycles")
        except Exception as e:
            st.error(f"Error: {e}")
    st.markdown("</div>", unsafe_allow_html=True)


# =================================================
# PAGE: BATCH PREDICTION
# =================================================

elif page == "📦 Batch Process":

    st.markdown("""
        <div class="main-header">
            <div>
                <h1>Batch Processing</h1>
                <p>Process multiple battery records simultaneously</p>
            </div>
            <div style="font-size: 3rem;">📦</div>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("<div class='dashboard-card'>", unsafe_allow_html=True)
    
    def load_sample_batch():
        sample = [
            {"Cycle_Index": 500, "Discharge_Time_s": 1200.0, "Decrement_36_34V_s": 300.0,
             "Max_Voltage_Dischar_V": 3.8, "Min_Voltage_Charg_V": 3.4,
             "Time_at_415V_s": 2500.0, "Time_constant_current_s": 4500.0, "Charging_time_s": 7000.0},
            {"Cycle_Index": 1000, "Discharge_Time_s": 1500.0, "Decrement_36_34V_s": 400.0,
             "Max_Voltage_Dischar_V": 3.9, "Min_Voltage_Charg_V": 3.5,
             "Time_at_415V_s": 3000.0, "Time_constant_current_s": 5000.0, "Charging_time_s": 8000.0}
        ]
        st.session_state.batch_json = json.dumps(sample, indent=4)

    st.button("📋 Load Sample Batch", on_click=load_sample_batch, key='load_batch_btn')
    batch_json = st.text_area("Paste Batch JSON List", key="batch_json", height=300)

    if st.button("🚀 Run Batch Prediction"):
        try:
            data = json.loads(batch_json)
            df_input = pd.DataFrame(data)
            df_input.columns = (
                df_input.columns.str.strip().str.replace(" ", "_").str.replace("(", "")
                .str.replace(")", "").str.replace("-", "_").str.replace(".", "")
            )
            df_input = df_input[feature_cols]
            df_input["Predicted_RUL"] = model.predict(scaler.transform(df_input))
            
            st.write("### Results")
            st.dataframe(df_input, use_container_width=True)
            
            csv = df_input.to_csv(index=False).encode()
            st.download_button("📥 Download CSV", csv, "battery_predictions.csv", mime="text/csv")
        except Exception as e:
            st.error(f"Error: {e}")
    st.markdown("</div>", unsafe_allow_html=True)


# =================================================
# PAGE: SYSTEM INFO
# =================================================

elif page == "ℹ System Info":

    st.markdown("""
        <div class="main-header">
            <div>
                <h1>System Information</h1>
                <p>Model architecture and performance details</p>
            </div>
            <div style="font-size: 3rem;">⚙️</div>
        </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("<div class='dashboard-card'><div class='card-header'><div class='card-title'>Model Details</div></div></div>", unsafe_allow_html=True)
        st.write("**Type:** Random Forest Regressor")
        st.write("**Version:** v1.0 (Sklearn)")
        st.metric("R² Score", "0.999")
        st.metric("RMSE", "3.77 Cycles")
    
    with col2:
        st.markdown("<div class='dashboard-card'><div class='card-header'><div class='card-title'>Feature Importance</div></div></div>", unsafe_allow_html=True)
        try:
            imp_df = pd.DataFrame({"Feature": feature_cols, "Importance": model.feature_importances_}).sort_values("Importance", ascending=False)
            fig = px.bar(imp_df, x="Importance", y="Feature", orientation="h", color="Importance", color_continuous_scale=["#f59e0b", "#d97706"])
            fig.update_layout(yaxis={'categoryorder':'total ascending'}, plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig, use_container_width=True)
        except:
            st.warning("Data unavailable")