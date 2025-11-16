# 🌎 Climate Futures

A **Python-based climate risk model** that predicts **heatwaves, flooding, and storm surge risk** for Caribbean islands using environmental data, machine learning, and simulation engines.

## Project Overview

This system combines:

- Climate science
- Predictive modeling
- Real Caribbean relevance
- Advanced algorithms
- Interactive visualizations
- Research-grade reporting

## Features

### 1. Climate Data Collection

- Rainfall, temperature, sea level, wind data
- Integration with NOAA, NASA POWER, ERA5 datasets
- Caribbean meteorological office data support

### 2. Machine Learning Predictions

- **Heatwave probability** (binary classification)
- **Daily rainfall extremes** (regression)
- **Flood risk modeling** (hydrological simulation)

### 3. Climate Risk Index

- Combined 0–100 danger score
- Daily updates
- Multi-hazard assessment

### 4. Visual Analytics

- Interactive maps with folium
- Time-series graphs with plotly
- Elevation-based flood spread visualization

### 5. Automated Alert System

- Heatwave warnings
- Flash flood alerts
- Storm surge risk notifications

## Tech Stack

**Core Libraries:**

- Python 3.10+
- pandas, numpy
- scikit-learn, xgboost
- matplotlib, plotly, seaborn
- folium, geopandas
- rasterio (DEM processing)

**Optional Advanced:**

- PyTorch (neural networks)
- Flask (web interface)
- Mesa (agent-based simulation)

## Project Structure

```text
Climate Futures/
├── data/                      # Data storage
│   ├── raw/                   # Raw datasets
│   ├── processed/             # Cleaned data
│   └── models/                # Trained ML models
├── src/                       # Source code
│   ├── data_preprocessing.py  # Data cleaning & loading
│   ├── features.py            # Feature engineering
│   ├── models/                # ML model modules
│   │   ├── heatwave.py
│   │   ├── rainfall.py
│   │   └── flood.py
│   ├── flood_simulation.py    # Hydrological simulation
│   ├── risk_model.py          # Climate risk index
│   ├── visualization.py       # Plotting functions
│   └── alerts.py              # Alert generation
├── notebooks/                 # Jupyter notebooks
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_model_training.ipynb
│   └── 04_dashboard.ipynb
├── config/                    # Configuration files
│   └── config.yaml
├── tests/                     # Unit tests
├── app.py                     # Flask application (optional)
├── cli.py                     # Command-line interface
├── requirements.txt           # Dependencies
└── README.md
```

## Installation

```bash
# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Quick Start

```bash
# Run complete analysis
python cli.py --analyze --location "Kingston, Jamaica"

# Generate daily report
python cli.py --daily-report

# Check current risk
python cli.py --risk-score
```

### Web Interface

```bash
# Start Flask dashboard
python app.py
# Visit http://localhost:5000
```

### Jupyter Notebooks

```bash
jupyter notebook notebooks/
```

## Development Timeline

- Data collection & preprocessing
- Feature engineering
- ML model training
- Flood simulation engine
- Climate risk index
- Visualization dashboard
- Alert system
- Final assembly
- Presentation & documentation

## Scientific Approach

### Climate Metrics

- Heat Index & Humidity calculations
- ENSO/SST anomalies
- Rolling averages & trend detection
- Storm surge potential modeling

### Model Performance

- Cross-validation
- Feature importance analysis
- Confusion matrices & ROC curves
- RMSE, MAE for regression

### Validation

- Historical event backtesting
- Comparison with official warnings
- Sensitivity analysis

## Future Enhancements

- [ ] GPU-accelerated neural networks
- [ ] Reinforcement learning for evacuation planning
- [ ] Real-time IoT sensor integration
- [ ] Monte Carlo climate projections (2050)
- [ ] Multi-island comparison dashboard
- [ ] Mobile app development

## Contributing

This project is designed for educational and research purposes to help Caribbean communities prepare for climate impacts.

## License

MIT License

## Acknowledgments

- NOAA Climate Data
- NASA POWER API
- ERA5 Reanalysis
- Caribbean meteorological services

---

Built with love for Caribbean climate resilience
