# Streamlit Dashboard Guide

## Interactive Web Dashboard for Caribbean Climate System

The Streamlit dashboard provides a modern, interactive web interface for exploring climate data, viewing risk assessments, and running flood simulations in real-time.

---

## Quick Start

### 1. Install Streamlit (if not already installed)

```bash
pip install streamlit>=1.28.0
```

Or install all requirements:

```bash
pip install -r requirements.txt
```

### 2. Ensure Data is Available

Before running the dashboard, make sure you have generated the climate data and trained models:

```bash
python quickstart.py
```

This will create:

- `data/processed/climate_features.csv` - Climate data with predictions
- `data/models/*.pkl` - Trained ML models
- `results/` - Visualizations
- `reports/` - Daily reports

### 3. Launch the Dashboard

```bash
streamlit run app.py
```

The dashboard will automatically open in your default web browser at `http://localhost:8501`

---

## Dashboard Features

### **Tab 1: Overview**

- **Current Risk Status**: Real-time overall climate risk score
- **Key Metrics**: Temperature, precipitation, humidity
- **Risk Timeline**: Interactive graph showing risk trends over time
- **Active Alerts**: Current flood and heatwave warnings

### **Tab 2: Climate Metrics**

- **Temperature Analysis**: Min/avg/max temperature trends
- **Precipitation Patterns**: Daily rainfall visualization
- **Wind Speed**: Wind patterns over time
- **Atmospheric Pressure**: Pressure trends

### **Tab 3: Risk Analysis**

- **Risk Components**: Breakdown of flood, heatwave, and rainfall risks
- **Risk Distribution**: Pie chart showing risk category distribution
- **Component Scores**: Individual risk factor analysis

### **Tab 4: Flood Simulation**

- **Interactive Simulation**: Run flood simulations with custom parameters
- **Rainfall Input**: Adjust rainfall amount (0-200mm)
- **Grid Resolution**: Choose simulation detail level
- **Flood Depth Map**: Visualize water accumulation patterns
- **Impact Statistics**: Max depth and affected area metrics

### **Tab 5: Forecasts**

- **14-Day Outlook**: Heatwave and flood probability forecasts
- **Rainfall Forecast**: Predicted vs actual precipitation
- **Alert Thresholds**: Configurable warning levels

---

## Configuration Options

### Sidebar Settings

#### Date Range Selector

- Choose custom date ranges for analysis
- Default: Last 30 days

#### Location Selector

- Kingston, Jamaica
- Port-of-Spain, Trinidad
- Bridgetown, Barbados
- Nassau, Bahamas
- Havana, Cuba

#### Alert Thresholds

- **Flood Alert**: 0-100% (default: 60%)
- **Heatwave Alert**: 0-100% (default: 70%)

---

## Features & Capabilities

### Interactive Visualizations

- **Plotly Charts**: Zoom, pan, and hover for detailed information
- **Real-time Updates**: Data refreshes when filters change
- **Responsive Design**: Works on desktop and mobile devices

### Data Exploration

- **Time Series Analysis**: Track metrics over custom periods
- **Risk Distribution**: Understand risk patterns
- **Component Breakdown**: Analyze individual risk factors

### Simulation Tools

- **Flood Modeling**: Run hydrological simulations
- **Parameter Tuning**: Adjust rainfall and resolution
- **Visual Results**: Heatmaps and statistics

### Alert Management

- **Threshold Configuration**: Set custom warning levels
- **Active Alerts**: See current warnings
- **Risk Categories**: Color-coded severity levels

---

## Usage Tips

### For Best Performance

1. **First Run**: Execute `quickstart.py` to generate all necessary data
2. **Data Updates**: Re-run quickstart to refresh with new data
3. **Browser**: Use Chrome or Firefox for optimal performance
4. **Cache**: Streamlit caches data for faster loading

### Common Workflows

#### Daily Risk Assessment

1. Launch dashboard
2. Check "Overview" tab for current status
3. Review active alerts
4. Explore detailed metrics in other tabs

#### Flood Risk Evaluation

1. Go to "Flood Simulation" tab
2. Set rainfall amount based on forecast
3. Run simulation
4. Analyze affected areas

#### Historical Analysis

1. Use sidebar date range selector
2. Choose past period (e.g., last 90 days)
3. Review trends in "Climate Metrics"
4. Analyze risk distribution

#### Alert Configuration

1. Adjust threshold sliders in sidebar
2. Review which days would trigger alerts
3. Export or screenshot for reports

---

## Customization

### Modify Alert Colors

Edit the CSS in `app.py`:

```python
.risk-extreme { color: #8e44ad; }  # Purple
.risk-high { color: #e74c3c; }     # Red
.risk-moderate { color: #e67e22; } # Orange
.risk-low { color: #f1c40f; }      # Yellow
.risk-minimal { color: #2ecc71; }  # Green
```

### Add New Visualizations

Add custom plots in the respective tabs:

```python
with tab6:  # New tab
    st.header("Custom Analysis")
    fig = px.scatter(df, x='temperature', y='precipitation')
    st.plotly_chart(fig)
```

### Configure Default Settings

Modify default values in the sidebar section:

```python
flood_threshold = st.sidebar.slider("Flood Alert (%)", 0, 100, 60)  # Change 60
heatwave_threshold = st.sidebar.slider("Heatwave Alert (%)", 0, 100, 70)  # Change 70
```

---

## Troubleshooting

### Dashboard Won't Start

```bash
# Check Streamlit installation
pip show streamlit

# Reinstall if needed
pip install --upgrade streamlit
```

### "Data not found" Error

```bash
# Generate data first
python quickstart.py

# Verify files exist
ls data/processed/climate_features.csv
ls data/models/
```

### Slow Performance

- Reduce date range in sidebar
- Lower grid resolution in flood simulation
- Clear browser cache
- Restart Streamlit server

### Port Already in Use

```bash
# Use different port
streamlit run app.py --server.port 8502
```

---

## Deployment Options

### Local Network Access

```bash
streamlit run app.py --server.address 0.0.0.0
```

Access from other devices on network: `http://YOUR_IP:8501`

### Cloud Deployment

#### Streamlit Cloud (Free)

1. Push to GitHub
2. Visit [Streamlit Share](https://share.streamlit.io)
3. Connect repository
4. Deploy!

#### Heroku

1. Create `Procfile`: `web: streamlit run app.py --server.port $PORT`
2. Deploy via Heroku CLI

#### Docker

```dockerfile
FROM python:3.10
COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt
EXPOSE 8501
CMD ["streamlit", "run", "app.py"]
```

---

## Additional Resources

- [Streamlit Documentation](https://docs.streamlit.io)
- [Plotly Graphing Library](https://plotly.com/python/)
- Project README: `README.md`
- Complete Guide: `docs/GUIDE.md`

---

## Support

For issues or questions:

1. Check this guide
2. Review console output for errors
3. Verify data files exist
4. Ensure all dependencies installed

---

**Enjoy exploring Caribbean climate data with the interactive dashboard!**
