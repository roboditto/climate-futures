"""
Streamlit Dashboard for Climate Futures
Interactive web interface for climate risk analysis and visualization.
"""

import streamlit as st
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from utils import load_config, setup_logging
from data_preprocessing import ClimateDataLoader
from features import ClimateFeatureEngineer
from models.heatwave import HeatwavePredictor
from models.rainfall import RainfallPredictor
from models.flood import FloodPredictor
from risk_model import ClimateRiskIndex
from visualization import ClimateVisualizer
from alerts import ClimateAlertSystem
from flood_simulation import FloodSimulator
from predictions import generate_predictions

# Page configuration
st.set_page_config(
    page_title="Caribbean Climate Risk Dashboard",
    page_icon="🌎",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .risk-extreme {
        color: #8e44ad;
        font-weight: bold;
    }
    .risk-high {
        color: #e74c3c;
        font-weight: bold;
    }
    .risk-moderate {
        color: #e67e22;
        font-weight: bold;
    }
    .risk-low {
        color: #f1c40f;
        font-weight: bold;
    }
    .risk-minimal {
        color: #2ecc71;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_system():
    """Load and initialize the climate system."""
    config = load_config()
    logger = setup_logging(config)
    
    data_loader = ClimateDataLoader(config, logger)
    feature_engineer = ClimateFeatureEngineer(config, logger)
    risk_calculator = ClimateRiskIndex(config, logger)
    visualizer = ClimateVisualizer(config, logger)
    alert_system = ClimateAlertSystem(config, logger)
    
    # Load models
    heatwave_model = HeatwavePredictor(config, logger)
    rainfall_model = RainfallPredictor(config, logger)
    flood_model = FloodPredictor(config, logger)
    
    try:
        heatwave_model.load_model('data/models/heatwave_model.pkl')
        rainfall_model.load_model('data/models/rainfall_model.pkl')
        flood_model.load_model('data/models/flood_model.pkl')
    except:
        st.warning("⚠️ Models not found. Please run quickstart.py first to train models.")
    
    return {
        'config': config,
        'logger': logger,
        'data_loader': data_loader,
        'feature_engineer': feature_engineer,
        'risk_calculator': risk_calculator,
        'visualizer': visualizer,
        'alert_system': alert_system,
        'models': {
            'heatwave': heatwave_model,
            'rainfall': rainfall_model,
            'flood': flood_model
        }
    }

@st.cache_data
def load_data():
    """Load climate data."""
    try:
        df = pd.read_csv('data/processed/climate_features.csv', 
                        index_col='date', parse_dates=True)
        return df
    except:
        st.error("❌ Data not found. Please run quickstart.py first to generate data.")
        return None

def get_risk_color(score):
    """Get color based on risk score."""
    if pd.isna(score):
        return "#95a5a6"
    elif score >= 85:
        return "#8e44ad"  # Extreme
    elif score >= 70:
        return "#e74c3c"  # High
    elif score >= 50:
        return "#e67e22"  # Moderate
    elif score >= 30:
        return "#f1c40f"  # Low
    else:
        return "#2ecc71"  # Minimal

def get_risk_category(score):
    """Get risk category name."""
    if pd.isna(score):
        return "Unknown"
    elif score >= 85:
        return "Extreme"
    elif score >= 70:
        return "High"
    elif score >= 50:
        return "Moderate"
    elif score >= 30:
        return "Low"
    else:
        return "Minimal"

def main():
    # Header
    st.markdown('<h1 class="main-header">🌎 Caribbean Climate Impact Dashboard</h1>', 
                unsafe_allow_html=True)
    st.markdown("**Real-time Climate Risk Analysis & Early Warning System**")
    
    # Load system
    with st.spinner("Loading climate system..."):
        system = load_system()
    
    # Load data
    with st.spinner("Loading climate data..."):
        df = load_data()
    
    if df is None:
        st.stop()
        return
    
    # Generate predictions if not present
    with st.spinner("Generating predictions..."):
        try:
            df = generate_predictions(df, system)
        except Exception as e:
            system['logger'].exception('generate_predictions failed')
            st.warning(f"⚠️ Prediction generation failed: {e}")
    
    # Sidebar
    st.sidebar.title("⚙️ Settings")
    
    # Date selector
    st.sidebar.subheader("📅 Date Range")
    min_date = df.index.min().date()
    max_date = df.index.max().date()
    
    date_range = st.sidebar.date_input(
        "Select date range",
        value=(max_date - timedelta(days=30), max_date),
        min_value=min_date,
        max_value=max_date
    )
    
    # Filter data by date range
    if len(date_range) == 2:
        start_date, end_date = date_range
        # Convert Index to DatetimeIndex for .date access
        dt_index = pd.to_datetime(df.index)
        mask = (dt_index.date >= start_date) & (dt_index.date <= end_date)
        df_filtered = df[mask].copy()
    else:
        df_filtered = df.copy()
    
    # Validate filtered data
    if len(df_filtered) == 0:
        st.error("❌ No data available for the selected date range.")
        st.stop()
        return  # For type checker
    
    # Location selector (simulated)
    st.sidebar.subheader("📍 Location")
    location = st.sidebar.selectbox(
        "Select location",
        ["Kingston, Jamaica", "Port-of-Spain, Trinidad", "Bridgetown, Barbados", 
         "Nassau, Bahamas", "Havana, Cuba"]
    )
    
    # Risk thresholds
    st.sidebar.subheader("⚠️ Alert Thresholds")
    flood_threshold = st.sidebar.slider("Flood Alert (%)", 0, 100, 60)
    heatwave_threshold = st.sidebar.slider("Heatwave Alert (%)", 0, 100, 70)
    
    # Main content
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Overview", 
        "🌡️ Climate Metrics", 
        "⚠️ Risk Analysis",
        "🌊 Flood Simulation",
        "📈 Forecasts"
    ])
    
    with tab1:
        st.header("Current Climate Status")
        
        # Latest data
        if 'overall_risk_score' in df_filtered.columns:
            latest = df_filtered.iloc[-1]
            latest_date = df_filtered.index[-1]
            
            # Metrics row
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                risk_score = latest.get('overall_risk_score', np.nan)
                risk_cat = get_risk_category(risk_score)
                st.metric(
                    "Overall Risk Score",
                    f"{risk_score:.1f}/100" if not pd.isna(risk_score) else "N/A",
                    delta=None
                )
                st.markdown(f"**Category:** :{get_risk_color(risk_score)}[{risk_cat}]")
            
            with col2:
                temp = latest.get('temperature', np.nan)
                st.metric(
                    "Temperature",
                    f"{temp:.1f}°C" if not pd.isna(temp) else "N/A"
                )
            
            with col3:
                precip = latest.get('precipitation', np.nan)
                st.metric(
                    "Precipitation",
                    f"{precip:.1f} mm" if not pd.isna(precip) else "N/A"
                )
            
            with col4:
                humidity = latest.get('humidity', np.nan)
                st.metric(
                    "Humidity",
                    f"{humidity:.1f}%" if not pd.isna(humidity) else "N/A"
                )
            
            st.markdown("---")
            
            # Risk timeline
            st.subheader("📈 Risk Score Timeline")
            
            if 'overall_risk_score' in df_filtered.columns:
                fig = go.Figure()
                
                fig.add_trace(go.Scatter(
                    x=df_filtered.index,
                    y=df_filtered['overall_risk_score'],
                    mode='lines',
                    name='Risk Score',
                    line=dict(color='#1f77b4', width=2),
                    fill='tozeroy',
                    fillcolor='rgba(31, 119, 180, 0.2)'
                ))
                
                # Add threshold lines
                fig.add_hline(y=85, line_dash="dash", line_color="purple", 
                            annotation_text="Extreme")
                fig.add_hline(y=70, line_dash="dash", line_color="red", 
                            annotation_text="High")
                fig.add_hline(y=50, line_dash="dash", line_color="orange", 
                            annotation_text="Moderate")
                
                fig.update_layout(
                    title="Climate Risk Score Over Time",
                    xaxis_title="Date",
                    yaxis_title="Risk Score (0-100)",
                    hovermode='x unified',
                    height=400
                )
                
                st.plotly_chart(fig, width='stretch')
            
            # Active alerts
            st.subheader("🚨 Active Alerts")
            
            alerts = []
            flood_prob = latest.get('flood_prob', 0) * 100
            heatwave_prob = latest.get('heatwave_prob', 0) * 100
            
            if flood_prob > flood_threshold:
                alerts.append({
                    'type': 'FLOOD',
                    'severity': 'HIGH' if flood_prob > 80 else 'MODERATE',
                    'probability': flood_prob,
                    'message': f'High flood risk detected ({flood_prob:.1f}% probability)'
                })
            
            if heatwave_prob > heatwave_threshold:
                alerts.append({
                    'type': 'HEATWAVE',
                    'severity': 'HIGH' if heatwave_prob > 85 else 'MODERATE',
                    'probability': heatwave_prob,
                    'message': f'Heatwave conditions likely ({heatwave_prob:.1f}% probability)'
                })
            
            if alerts:
                for alert in alerts:
                    st.warning(f"⚠️ **{alert['type']} ALERT ({alert['severity']})**: {alert['message']}")
            else:
                st.success("✅ No active alerts at this time")
    
    with tab2:
        st.header("Climate Metrics")
        
        # Temperature trends
        st.subheader("🌡️ Temperature Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if 'temperature' in df_filtered.columns:
                fig = go.Figure()
                
                if 'temperature_max' in df_filtered.columns:
                    fig.add_trace(go.Scatter(
                        x=df_filtered.index,
                        y=df_filtered['temperature_max'],
                        mode='lines',
                        name='Max Temp',
                        line=dict(color='red', width=1)
                    ))
                
                fig.add_trace(go.Scatter(
                    x=df_filtered.index,
                    y=df_filtered['temperature'],
                    mode='lines',
                    name='Avg Temp',
                    line=dict(color='orange', width=2)
                ))
                
                if 'temperature_min' in df_filtered.columns:
                    fig.add_trace(go.Scatter(
                        x=df_filtered.index,
                        y=df_filtered['temperature_min'],
                        mode='lines',
                        name='Min Temp',
                        line=dict(color='blue', width=1)
                    ))
                
                fig.update_layout(
                    title="Temperature Trends",
                    xaxis_title="Date",
                    yaxis_title="Temperature (°C)",
                    hovermode='x unified',
                    height=350
                )
                
                st.plotly_chart(fig, width='stretch')
        
        with col2:
            if 'precipitation' in df_filtered.columns:
                fig = go.Figure()
                
                fig.add_trace(go.Bar(
                    x=df_filtered.index,
                    y=df_filtered['precipitation'],
                    name='Precipitation',
                    marker_color='lightblue'
                ))
                
                fig.update_layout(
                    title="Precipitation Patterns",
                    xaxis_title="Date",
                    yaxis_title="Precipitation (mm)",
                    hovermode='x unified',
                    height=350
                )
                
                st.plotly_chart(fig, width='stretch')
        
        # Additional metrics
        st.subheader("💨 Wind & Pressure")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if 'wind_speed' in df_filtered.columns:
                fig = px.line(
                    df_filtered,
                    y='wind_speed',
                    title='Wind Speed Over Time',
                    labels={'wind_speed': 'Wind Speed (m/s)', 'index': 'Date'}
                )
                fig.update_traces(line_color='green')
                st.plotly_chart(fig, width='stretch')
        
        with col2:
            if 'pressure' in df_filtered.columns:
                fig = px.line(
                    df_filtered,
                    y='pressure',
                    title='Atmospheric Pressure',
                    labels={'pressure': 'Pressure (hPa)', 'index': 'Date'}
                )
                fig.update_traces(line_color='purple')
                st.plotly_chart(fig, width='stretch')
    
    with tab3:
        st.header("Risk Analysis")
        
        # Risk component breakdown
        st.subheader("📊 Risk Components")
        
        if all(col in df_filtered.columns for col in ['flood_prob', 'heatwave_prob']):
            latest = df_filtered.iloc[-1]
            
            # Component scores
            col1, col2, col3 = st.columns(3)
            
            with col1:
                flood_risk = latest.get('flood_prob', 0) * 100
                st.metric("Flood Risk", f"{flood_risk:.1f}%")
                st.progress(flood_risk / 100)
            
            with col2:
                heatwave_risk = latest.get('heatwave_prob', 0) * 100
                st.metric("Heatwave Risk", f"{heatwave_risk:.1f}%")
                st.progress(heatwave_risk / 100)
            
            with col3:
                rainfall_pred = latest.get('rainfall_pred', 0)
                if not pd.isna(rainfall_pred):
                    st.metric("Predicted Rainfall", f"{rainfall_pred:.1f} mm")
                    st.progress(min(rainfall_pred / 100, 1.0))
                else:
                    st.metric("Predicted Rainfall", "N/A")
            
            # Risk distribution
            st.subheader("📈 Risk Distribution")
            
            if 'overall_risk_score' in df_filtered.columns:
                risk_scores = df_filtered['overall_risk_score'].dropna()
                
                # Create categories
                categories = pd.cut(
                    risk_scores,
                    bins=[0, 30, 50, 70, 85, 100],
                    labels=['Minimal', 'Low', 'Moderate', 'High', 'Extreme']
                )
                
                cat_counts = categories.value_counts()
                
                fig = go.Figure(data=[go.Pie(
                    labels=cat_counts.index,
                    values=cat_counts.values,
                    marker=dict(colors=['#2ecc71', '#f1c40f', '#e67e22', '#e74c3c', '#8e44ad']),
                    hole=0.4
                )])
                
                fig.update_layout(
                    title="Risk Category Distribution",
                    height=400
                )
                
                st.plotly_chart(fig, width='stretch')
    
    with tab4:
        st.header("Flood Simulation")
        
        st.markdown("""
        Hydrological flood simulation using Digital Elevation Model (DEM) and 
        D8 flow algorithm to predict water accumulation and flood-prone areas.
        """)
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("⚙️ Simulation Parameters")
            
            rainfall_input = st.slider(
                "Rainfall Amount (mm)",
                min_value=0,
                max_value=200,
                value=100,
                step=10
            )
            
            grid_size = st.selectbox(
                "Grid Resolution",
                [50, 100, 150, 200],
                index=1
            )
            
            if st.button("🌊 Run Simulation"):
                with st.spinner("Running flood simulation..."):
                    simulator = FloodSimulator(system['config'], system['logger'])
                    
                    # Extract country from location
                    country = location.split(',')[-1].strip()
                    
                    # Generate synthetic DEM
                    dem = simulator.generate_synthetic_dem(size=(grid_size, grid_size))
                    
                    # Simulate flood
                    water_depth = simulator.simulate_rainfall_runoff(
                        dem,
                        rainfall_mm=rainfall_input,
                        duration_hours=6
                    )
                    
                    # Store in session state
                    st.session_state['dem'] = dem
                    st.session_state['water_depth'] = water_depth
                    st.session_state['selected_country'] = country
        
        with col2:
            if 'water_depth' in st.session_state:
                st.subheader("🗺️ Flood Depth Map")
                
                # Get country from session state
                country_title = st.session_state.get('selected_country', '')
                title = f"Simulated Flood Water Depth - {country_title}" if country_title else "Simulated Flood Water Depth"
                
                fig = go.Figure(data=go.Heatmap(
                    z=st.session_state['water_depth'],
                    colorscale='Blues',
                    colorbar=dict(title="Water Depth (m)")
                ))
                
                fig.update_layout(
                    title=title,
                    xaxis_title="X (grid cells)",
                    yaxis_title="Y (grid cells)",
                    height=500
                )
                
                st.plotly_chart(fig, width='stretch')
                
                # Statistics
                max_depth = st.session_state['water_depth'].max()
                affected_area = (st.session_state['water_depth'] > 0.05).sum() / st.session_state['water_depth'].size * 100
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Max Water Depth", f"{max_depth:.2f} m")
                with col2:
                    st.metric("Affected Area", f"{affected_area:.1f}%")
    
    with tab5:
        st.header("Climate Forecasts")
        
        st.subheader("📅 14-Day Outlook")
        
        if 'heatwave_prob' in df_filtered.columns:
            # Last 14 days + forecast
            last_14 = df_filtered.tail(14)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Heatwave Probability**")
                
                fig = go.Figure()
                
                fig.add_trace(go.Scatter(
                    x=last_14.index,
                    y=last_14['heatwave_prob'] * 100,
                    mode='lines+markers',
                    name='Heatwave Risk',
                    line=dict(color='red', width=2),
                    marker=dict(size=8)
                ))
                
                fig.add_hline(y=heatwave_threshold, line_dash="dash", 
                            line_color="orange", annotation_text="Alert Threshold")
                
                fig.update_layout(
                    xaxis_title="Date",
                    yaxis_title="Probability (%)",
                    yaxis_range=[0, 100],
                    hovermode='x unified',
                    height=300
                )
                
                st.plotly_chart(fig, width='stretch')
            
            with col2:
                st.markdown("**Flood Probability**")
                
                fig = go.Figure()
                
                fig.add_trace(go.Scatter(
                    x=last_14.index,
                    y=last_14['flood_prob'] * 100,
                    mode='lines+markers',
                    name='Flood Risk',
                    line=dict(color='blue', width=2),
                    marker=dict(size=8)
                ))
                
                fig.add_hline(y=flood_threshold, line_dash="dash", 
                            line_color="orange", annotation_text="Alert Threshold")
                
                fig.update_layout(
                    xaxis_title="Date",
                    yaxis_title="Probability (%)",
                    yaxis_range=[0, 100],
                    hovermode='x unified',
                    height=300
                )
                
                st.plotly_chart(fig, width='stretch')
        
        # Rainfall forecast
        if 'rainfall_pred' in df_filtered.columns:
            st.subheader("🌧️ Rainfall Forecast")
            
            last_14 = df_filtered.tail(14)
            
            fig = go.Figure()
            
            # Actual precipitation
            fig.add_trace(go.Bar(
                x=last_14.index,
                y=last_14['precipitation'],
                name='Actual',
                marker_color='lightblue'
            ))
            
            # Predicted precipitation
            if 'rainfall_pred' in last_14.columns:
                fig.add_trace(go.Scatter(
                    x=last_14.index,
                    y=last_14['rainfall_pred'],
                    name='Predicted',
                    mode='lines+markers',
                    line=dict(color='red', width=2, dash='dash'),
                    marker=dict(size=8)
                ))
            
            fig.update_layout(
                title="Actual vs Predicted Rainfall",
                xaxis_title="Date",
                yaxis_title="Rainfall (mm)",
                hovermode='x unified',
                height=400
            )
            
            st.plotly_chart(fig, width='stretch')
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #7f8c8d;'>
        <p>Caribbean Climate Impact Simulation & Early Warning System</p>
        <p>Data updates daily | Last updated: {}</p>
    </div>
    """.format(datetime.now().strftime("%Y-%m-%d %H:%M")), unsafe_allow_html=True)

if __name__ == "__main__":
    main()
