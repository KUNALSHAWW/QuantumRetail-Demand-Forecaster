import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from datetime import datetime
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.ingest_transform import get_processed_data
from src.model_selection import time_series_split, train_and_select_model
from src.model_selection import save_model, load_model

# Page configuration
st.set_page_config(
    page_title="QuantumRetail Demand Forecaster",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better aesthetics
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: 700;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 1rem 0;
    }
    .sub-header {
        text-align: center;
        color: #666;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .stButton>button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 5px;
        padding: 0.5rem 2rem;
        font-weight: 600;
        width: 100%;
    }
    .stButton>button:hover {
        background: linear-gradient(90deg, #764ba2 0%, #667eea 100%);
    }
    .info-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<h1 class="main-header">⚡ QuantumRetail Demand Forecaster</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Advanced Machine Learning Pipeline for Intelligent Retail Demand Prediction</p>', unsafe_allow_html=True)
st.markdown("---")

@st.cache_data
def load_data():
    """Load and cache processed data"""
    with st.spinner("🔄 Loading data from HuggingFace Hub..."):
        return get_processed_data()

# Load data
df = load_data()

# Sidebar Configuration
st.sidebar.image("https://img.icons8.com/clouds/200/000000/artificial-intelligence.png", width=150)
st.sidebar.markdown("## 🎛️ Configuration Panel")
st.sidebar.markdown("---")

# Store selection with search
st.sidebar.markdown("### 🏪 Store Selection")
store_ids = df["store_id"].unique().tolist()
store_id = st.sidebar.selectbox(
    "Choose Store ID",
    sorted(store_ids),
    help="Select the retail store for analysis"
)

# Product selection (filtered by store)
st.sidebar.markdown("### 📦 Product Selection")
products = df[df["store_id"] == store_id]["product_id"].unique().tolist()
product_id = st.sidebar.selectbox(
    "Choose Product SKU",
    sorted(products),
    help="Select the product for demand forecasting"
)

# Get max available days
max_days = len(df[(df["store_id"] == store_id) & (df["product_id"]==product_id)])

# Training & forecast windows
st.sidebar.markdown("### 📊 Model Configuration")
train_days = st.sidebar.slider(
    "Training Window (days)",
    min_value=10,
    max_value=max_days,
    value=min(60, max_days),
    step=1,
    help="Number of days to use for model training"
)

# Advanced options
with st.sidebar.expander("🔬 Advanced Options"):
    show_metrics = st.checkbox("Show Detailed Metrics", value=True)
    show_features = st.checkbox("Show Feature Importance", value=False)
    confidence_interval = st.slider("Confidence Interval (%)", 80, 99, 95)


st.sidebar.markdown("---")
run_forecast = st.sidebar.button("🚀 Run Forecast", use_container_width=True)

# Main content area
if run_forecast:
    # Create columns for overview metrics
    col1, col2, col3, col4 = st.columns(4)
    
    df_subset = df[(df["store_id"] == store_id) & (df["product_id"] == product_id)]
    df_subset = df_subset.sort_values("dt").reset_index(drop=True)
    
    with col1:
        st.metric("📅 Total Days", len(df_subset))
    with col2:
        st.metric("📊 Training Days", train_days)
    with col3:
        st.metric("🔮 Validation Days", len(df_subset) - train_days + 7)
    with col4:
        avg_sales = df_subset["sale_amount"].mean()
        st.metric("💰 Avg Sales", f"{avg_sales:.2f}")
    
    st.markdown("---")
    
    # Progress indicator
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # 🔍 Try loading cached model
    status_text.text("🔍 Checking for cached model...")
    progress_bar.progress(20)
    best_model = load_model(store_id, product_id, train_days)
    
    if best_model:
        status_text.text("✅ Loaded cached model!")
        progress_bar.progress(50)
        results = {"cached_model": {"rmse": "N/A", "mae": "N/A"}}
        train, val = time_series_split(df_subset, train_days-7)
    else:
        status_text.text("⚡ Training new model...")
        progress_bar.progress(30)
        train, val = time_series_split(df_subset, train_days-7)
        best_model, results = train_and_select_model(train, val)
        progress_bar.progress(70)
        save_model(best_model, store_id, product_id, train_days)
        status_text.text("✅ Model trained successfully!")
    
    progress_bar.progress(100)
    status_text.empty()
    progress_bar.empty()
    
    # Model Information
    st.markdown("## 🧠 Model Performance")
    
    if results and results != {"cached_model": {"rmse": "N/A", "mae": "N/A"}}:
        best_model_name = min(results, key=lambda k: results[k]['rmse'])
        best_model_metrics = results[best_model_name]
        
        # Display metrics in attractive cards
        metric_cols = st.columns(len(results) + 1)
        
        with metric_cols[0]:
            st.markdown(f"""
            <div class="metric-card">
                <h3>🏆 Best Model</h3>
                <h2>{best_model_name}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        for idx, (model_name, metrics) in enumerate(results.items(), 1):
            with metric_cols[idx]:
                st.markdown(f"""
                <div class="info-box">
                    <h4>{model_name}</h4>
                    <p><b>RMSE:</b> {metrics['rmse']:.4f}</p>
                    <p><b>MAE:</b> {metrics['mae']:.4f}</p>
                </div>
                """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Generate predictions
    X_val = val.drop(columns=["sale_amount", "dt"])
    y_val = val["sale_amount"]
    preds = best_model.predict(X_val)
    
    # Calculate residuals
    residuals = y_val - preds
    
    # Visualization Section
    st.markdown("## 📈 Forecast Visualization")
    
    # Create tabs for different views
    tab1, tab2, tab3 = st.tabs(["📊 Forecast Plot", "📉 Residual Analysis", "📋 Data Table"])
    
    with tab1:
        fig, ax = plt.subplots(figsize=(14, 6))
        
        # Plot actual vs predicted
        ax.plot(val["dt"], y_val, label="Actual Sales", marker="o", linewidth=2, markersize=6, color="#667eea")
        ax.plot(val["dt"], preds, label="Predicted Sales", marker="x", linewidth=2, markersize=6, color="#f093fb")
        
        # Add confidence band (simple std-based approximation)
        std_error = np.std(residuals)
        ax.fill_between(val["dt"], preds - std_error, preds + std_error, alpha=0.2, color="#f093fb", label="Confidence Band")
        
        ax.set_title(f"Demand Forecast: Store {store_id} - Product {product_id}", fontsize=16, fontweight='bold')
        ax.set_xlabel("Date", fontsize=12)
        ax.set_ylabel("Sales Amount", fontsize=12)
        ax.tick_params(axis='x', rotation=45)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, linestyle='--')
        
        plt.tight_layout()
        st.pyplot(fig)
    
    with tab2:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Residual plot
        ax1.scatter(preds, residuals, alpha=0.6, color="#667eea")
        ax1.axhline(y=0, color='red', linestyle='--', linewidth=2)
        ax1.set_xlabel("Predicted Values", fontsize=12)
        ax1.set_ylabel("Residuals", fontsize=12)
        ax1.set_title("Residual Plot", fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # Residual distribution
        ax2.hist(residuals, bins=20, color="#764ba2", alpha=0.7, edgecolor='black')
        ax2.set_xlabel("Residual Value", fontsize=12)
        ax2.set_ylabel("Frequency", fontsize=12)
        ax2.set_title("Residual Distribution", fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        st.pyplot(fig)
    
    with tab3:
        # Create comparison dataframe
        comparison_df = pd.DataFrame({
            "Date": val["dt"],
            "Actual Sales": y_val.values,
            "Predicted Sales": preds,
            "Error": residuals.values,
            "Error %": (residuals.values / y_val.values * 100)
        })
        
        st.dataframe(
            comparison_df.style.background_gradient(cmap="RdYlGn_r", subset=["Error %"]),
            use_container_width=True
        )
        
        # Download button
        csv = comparison_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Forecast Data",
            data=csv,
            file_name=f"forecast_{store_id}_{product_id}_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv",
        )
    
    # Additional metrics section
    if show_metrics:
        st.markdown("---")
        st.markdown("## 📊 Detailed Performance Metrics")
        
        from sklearn.metrics import r2_score, mean_absolute_percentage_error
        
        r2 = r2_score(y_val, preds)
        mape = mean_absolute_percentage_error(y_val, preds) * 100
        
        metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
        
        with metric_col1:
            st.metric("R² Score", f"{r2:.4f}")
        with metric_col2:
            st.metric("MAPE", f"{mape:.2f}%")
        with metric_col3:
            st.metric("Max Error", f"{abs(residuals).max():.2f}")
        with metric_col4:
            st.metric("Std Error", f"{std_error:.2f}")

else:
    # Welcome screen
    st.markdown("""
    <div class="info-box">
        <h2>👋 Welcome to QuantumRetail Demand Forecaster</h2>
        <p>This advanced machine learning platform enables intelligent retail demand prediction with:</p>
        <ul>
            <li>🎯 <b>Latent Demand Recovery</b>: Account for sales lost during stockouts</li>
            <li>🤖 <b>Automated Model Selection</b>: LightGBM & XGBoost ensemble</li>
            <li>📊 <b>Interactive Visualizations</b>: Real-time forecast analysis</li>
            <li>⚡ <b>Intelligent Caching</b>: Fast predictions with model persistence</li>
        </ul>
        <p><b>To get started:</b> Configure your parameters in the sidebar and click "Run Forecast"</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Dataset overview
    st.markdown("### 📊 Dataset Overview")
    
    overview_col1, overview_col2, overview_col3 = st.columns(3)
    
    with overview_col1:
        st.metric("🏪 Total Stores", df["store_id"].nunique())
    with overview_col2:
        st.metric("📦 Total Products", df["product_id"].nunique())
    with overview_col3:
        st.metric("📅 Total Records", len(df))
    
    # Sample data
    st.markdown("### 🔍 Sample Data")
    st.dataframe(df.head(10), use_container_width=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 1rem;">
    <p>⚡ <b>QuantumRetail Demand Forecaster</b> | Built with Python, Streamlit & Machine Learning</p>
    <p>Developed by <a href="https://github.com/KUNALSHAWW" target="_blank">KUNALSHAWW</a> | Machine Learning Engineer</p>
</div>
""", unsafe_allow_html=True)