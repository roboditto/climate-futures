# 🌎 Caribbean Climate Impact System - Complete Guide

## Quick Start (5 Minutes)

### Option 1: Automated Demo

```powershell
# Install dependencies
pip install -r requirements.txt

# Run complete demonstration
python quickstart.py
```

This will:

- Generate 2 years of climate data
- Train all ML models
- Generate predictions and risk scores
- Create visualizations
- Generate alerts and reports

### Option 2: Command-Line Interface

```powershell
# Complete analysis with visualizations
python cli.py --analyze --visualize

# Train models only
python cli.py --train

# Generate daily report
python cli.py --daily-report

# Check current risk score
python cli.py --risk-score
```

### Option 3: Interactive Notebook

```powershell
jupyter notebook notebooks/01_complete_climate_system.ipynb
```

---

## Detailed Component Guide

### 1. Data Preprocessing

**Module:** `src/data_preprocessing.py`

**Features:**

- Load climate data from NASA POWER API or generate synthetic data
- Clean and validate data (handle missing values, outliers)
- Calculate derived metrics (heat index, storm surge potential)
- Save processed data for reuse

**Example:**

```python
from data_preprocessing import ClimateDataLoader

loader = ClimateDataLoader(config, logger)

# Generate synthetic data
df = loader.generate_sample_data(days=730)

# Or fetch real data
df = loader.fetch_nasa_power_data('20220101', '20231231')

# Clean and process
df_clean = loader.clean_data(df)
df_processed = loader.calculate_derived_metrics(df_clean)
```

---

### 2. Feature Engineering

**Module:** `src/features.py`

**Creates 100+ features including:**

- **Temporal:** Month, day of year, cyclical encoding, seasons
- **Rolling windows:** 3, 7, 14, 30-day averages/max/min
- **Lag features:** 1, 2, 3, 7, 14-day lags
- **Climate indices:** Heat index, humidity index, drought index
- **Interactions:** Temperature × humidity, wind × pressure
- **Extreme indicators:** Binary flags for extreme events

**Example:**

```python
from features import ClimateFeatureEngineer

engineer = ClimateFeatureEngineer(config, logger)
df_features = engineer.create_all_features(df_processed)

print(f"Created {len(df_features.columns)} features")
```

---

### 3. Machine Learning Models

#### A. Heatwave Prediction Model

**Module:** `src/models/heatwave.py`

**Type:** Binary Classification (RandomForest)

**Predicts:** Probability of heatwave in next 3 days

**Performance Metrics:**

- Accuracy: ~0.85-0.92
- ROC-AUC: ~0.88-0.94
- Cross-validation: 5-fold

**Example:**

```python
from models.heatwave import HeatwavePredictor

predictor = HeatwavePredictor(config, logger)

# Create target variable
df['is_heatwave'] = predictor.create_target_variable(df)

# Train
X, y = predictor.prepare_features(df)
metrics = predictor.train(X, y)

# Predict
probabilities = predictor.predict(X)
```

#### B. Rainfall Prediction Model

**Module:** `src/models/rainfall.py`

**Type:** Regression (XGBoost)

**Predicts:** Daily rainfall amount (mm)

**Performance Metrics:**

- RMSE: ~8-15 mm
- MAE: ~5-10 mm
- R²: ~0.65-0.80

**Example:**

```python
from models.rainfall import RainfallPredictor

predictor = RainfallPredictor(config, logger)

X, y = predictor.prepare_features(df, forecast_days=1)
metrics = predictor.train(X, y)

rainfall_mm = predictor.predict(X)
```

#### C. Flood Risk Model

**Module:** `src/models/flood.py`

**Type:** Binary Classification (RandomForest)

**Predicts:** Flood probability based on rainfall + terrain

**Inputs:**

- Predicted rainfall
- Cumulative rainfall
- Elevation
- Slope
- Coastal proximity

**Example:**

```python
from models.flood import FloodPredictor

predictor = FloodPredictor(config, logger)

df['is_flood_risk'] = predictor.create_target_variable(df)
X, y = predictor.prepare_features(df, include_elevation=True)
metrics = predictor.train(X, y)
```

---

### 4. Flood Simulation Engine

**Module:** `src/flood_simulation.py`

**Technique:** Cellular Automata + D8 Flow Direction Algorithm

**Process:**

1. Generate/load Digital Elevation Model (DEM)
2. Calculate flow directions (D8 algorithm)
3. Simulate water accumulation from rainfall
4. Apply Manning's equation for flow velocity
5. Generate water depth grid

**Example:**

```python
from flood_simulation import FloodSimulator

simulator = FloodSimulator(config, logger)

# Generate terrain
dem = simulator.generate_synthetic_dem(size=(150, 150))

# Simulate 100mm rainfall
water_depth = simulator.simulate_rainfall_runoff(
    dem, 
    rainfall_mm=100, 
    duration_hours=6
)

# Visualize
simulator.plot_simulation(dem, water_depth)
```

---

### 5. Climate Risk Index (Day 9)

**Module:** `src/risk_model.py`

**Formula:**

```txt
Risk Score = 0.40 × Flood Probability +
             0.30 × Heatwave Probability +
             0.30 × Rainfall Severity
```

**Output:** 0-100 scale

**Categories:**

- 0-30: Minimal Risk (Green)
- 30-50: Low Risk (Yellow)
- 50-70: Moderate Risk (Orange)
- 70-85: High Risk (Red)
- 85-100: Extreme Risk (Purple)

**Example:**

```python
from risk_model import ClimateRiskIndex

risk_calc = ClimateRiskIndex(config, logger)

risk_score = risk_calc.calculate_risk_score(
    flood_prob=0.75,
    heatwave_prob=0.60,
    rainfall_severity=85
)

category = risk_calc.get_risk_category(risk_score)
print(f"Risk: {risk_score}/100 - {category['name']}")
```

---

### 6. Visualization Dashboard

**Module:** `src/visualization.py`

**Visualizations:**

1. **Risk Dashboard:** Interactive HTML with 4 panels
2. **Heatwave Forecast:** 7-14 day temperature + probability
3. **Rainfall Forecast:** Daily rainfall bars with thresholds
4. **Risk Gauge:** Speedometer-style current risk
5. **Calendar Heatmap:** Monthly risk patterns

**Technologies:**

- **Matplotlib/Seaborn:** Static plots (PNG/PDF)
- **Plotly:** Interactive HTML visualizations
- **Folium:** Map layers (planned)

**Example:**

```python
from visualization import ClimateVisualizer

viz = ClimateVisualizer(config, logger)

# Create dashboard
viz.plot_risk_dashboard(df, save_path='dashboard.html')

# Forecasts
viz.plot_heatwave_forecast(df, days_ahead=14)
viz.plot_rainfall_forecast(df, days_ahead=14)

# Current risk
viz.plot_risk_gauge(current_risk_score)
```

---

### 7. Alert Generation System (Day 12)

**Module:** `src/alerts.py`

**Alert Triggers:**

- **Heatwave:** Probability > 70%
- **Flood:** Probability > 60%
- **Heavy Rainfall:** > 100mm predicted
- **Combined Risk:** Score > 70

**Output Formats:**

- Console (terminal)
- JSON (machine-readable)
- PDF/PNG Reports (planned)

**Example:**

```python
from alerts import ClimateAlertSystem

alert_system = ClimateAlertSystem(config, logger)

# Check for alerts
alerts = alert_system.generate_alerts(
    flood_prob=0.85,
    heatwave_prob=0.70,
    rainfall_mm=120,
    risk_score=78
)

# Generate daily report
alert_system.generate_daily_report(risk_summary, alerts)
```

---

## Configuration

All settings are in `config/config.yaml`:

### Key Settings

```yaml
location:
  default_island: "Jamaica"
  latitude: 18.1096
  longitude: -77.2975

models:
  heatwave:
    threshold_temperature: 32.0  # °C
    consecutive_days: 3
  
  rainfall:
    extreme_threshold: 50.0  # mm/day
  
  flood:
    rainfall_threshold: 75.0  # mm/day

risk_index:
  weights:
    flood_risk: 0.40
    heatwave_prob: 0.30
    rainfall_severity: 0.30
```

---

## Data Sources

### Supported Sources

1. **NASA POWER API** (Free, no account required)
   - Global coverage
   - Daily meteorological data
   - Variables: Temperature, rainfall, humidity, wind, pressure

2. **NOAA Climate Data** (Requires API key)
   - Station-based observations
   - High accuracy

3. **ERA5 Reanalysis** (Requires Copernicus account)
   - Historical climate data
   - Advanced variables

4. **Synthetic Data Generator**
   - For testing and demonstrations
   - Realistic Caribbean climate patterns

---

## Testing & Validation

### Model Performance

```python
# Run all tests
pytest tests/

# Test specific module
pytest tests/test_models.py

# With coverage
pytest --cov=src tests/
```

### Validation Approaches

1. **Cross-validation:** 5-fold CV on all models
2. **Temporal split:** Train on past, test on recent
3. **Feature importance:** Analyze top predictive features
4. **Backtesting:** Compare predictions to historical events

---

## Results & Outputs

### Generated Files

**Models:** `data/models/`

- `heatwave_model.pkl`
- `rainfall_model.pkl`
- `flood_model.pkl`

**Reports:** `reports/`

- Daily risk reports (TXT)
- Alert logs (JSON)

**Visualizations:** `results/`

- `risk_dashboard.html` - Interactive dashboard
- `heatwave_forecast.png` - Temperature forecast
- `rainfall_forecast.png` - Rainfall forecast
- `flood_simulation.png` - Flood depth map
- `risk_gauge.html` - Current risk indicator

**Data:** `data/processed/`

- `climate_data_processed.csv`
- `climate_features.csv`

---

## Use Cases

### 1. Daily Risk Assessment

```powershell
python cli.py --daily-report
```

Generate morning briefing for emergency management.

### 2. Event Forecasting

Monitor specific hazards (heatwaves, floods) 7-14 days ahead.

### 3. Historical Analysis

Analyze past climate patterns and validate predictions.

### 4. Community Planning

Identify high-risk periods for resource allocation.

### 5. Research & Education

Demonstrate climate modeling techniques.

---

## Advanced Features

### Planned Enhancements

- [ ] **Real-time data integration** from weather stations
- [ ] **Ensemble models** combining multiple algorithms
- [ ] **Neural networks** (LSTM) for time series
- [ ] **Flask web application** for public access
- [ ] **Mobile app** for alerts
- [ ] **IoT sensor integration**
- [ ] **Monte Carlo simulations** for 2050 projections
- [ ] **Multi-island comparison**

---

## Contributing

### Adding New Features

1. Create module in `src/`
2. Add configuration to `config/config.yaml`
3. Write tests in `tests/`
4. Update documentation
5. Create example in notebook

---

## License & Attribution

### MIT License

Data sources:

- NASA POWER Project
- NOAA National Centers for Environmental Information
- Caribbean meteorological services

---

## Acknowledgments

Built for Caribbean climate resilience and community preparedness.

Special thanks to:

- NASA POWER team for open climate data
- Caribbean meteorological offices
- Climate science community

---

## Support

For issues or questions:

1. Check documentation
2. Review example notebooks
3. Examine configuration files
4. Run `python cli.py --help`

---

**Stay informed. Stay prepared. Stay safe.**
