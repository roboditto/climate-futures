# Getting Started with the Climate Futures System

Welcome! This guide will help you get the system running in **under 10 minutes**.

---

[## Prerequisites

- **Python 3.10+** installed
- **pip** package manager
- **Windows PowerShell** (or terminal of choice)
- **10 GB** free disk space

---

## Installation Steps

### Step 1: Navigate to Project Directory

```powershell
cd "c:\Users\allen\Downloads\Career\SCHOOL\Olympiad\Coding Competition\Climate Futures"
```

### Step 2: Create Virtual Environment (Recommended)

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### Step 3: Install Dependencies

```powershell
pip install -r requirements.txt
```

This will install:

- Data science: pandas, numpy, scipy
- ML: scikit-learn, xgboost
- Visualization: matplotlib, seaborn, plotly
- Geospatial: geopandas, rasterio
- Web: flask (optional)

**Installation time:** ~5-10 minutes

---

## Choose Your Experience

### Option A: Quick Demo (Recommended for First Time)

Run the automated demonstration in the src directory:

```powershell
python quickstart.py
```

**What it does:**

1. Generates 2 years of Caribbean climate data
2. Creates 100+ scientifically meaningful features
3. Trains 3 ML models (heatwave, rainfall, flood)
4. Calculates climate risk scores
5. Generates alerts and reports
6. Creates visualizations
7. Runs a flood simulation

**Time:** ~3-5 minutes

**Output:**

- Models: `data/models/`
- Reports: `reports/demo_daily_report.txt`
- Visualizations: `results/`

---

### Option B: Command-Line Interface

For more control, use the CLI:

```powershell
# Complete analysis with all visualizations
python cli.py --analyze --visualize

# Just train the models
python cli.py --train --days 730

# Get today's risk assessment
python cli.py --risk-score

# Generate daily report
python cli.py --daily-report
```

**Available options:**

```text
--analyze         Run complete analysis
--train           Train ML models
--daily-report    Generate daily report
--risk-score      Show current risk
--visualize       Create visualizations
--days N          Number of days of data (default: 730)
--real-data       Fetch real NASA POWER data
```

---

### Option C: Interactive Jupyter Notebook

For exploratory analysis:

```powershell
jupyter notebook
```

Then open: `notebooks/01_complete_climate_system.ipynb`

**Contains:**

- Step-by-step explanations
- Interactive visualizations
- Customizable parameters
- Full code examples

---

## Project Structure

```text
Climate Futures/
├── src/
│  ├── config/
│  │   └── config.yaml          # All settings (thresholds, weights, etc.)
│  │
│  ├── data/
│  │    ├── cache/               # Fast memory retrieval for frequently used data    
│  │    ├── raw/                 # Original data
│  │    ├── processed/           # Cleaned data
│  │    └── models/              # Trained ML models (*.pkl)
│  ├── data_preprocessing.py   # Data loading & cleaning
│  ├── features.py             # Feature engineering
│  ├── models/
│  │    ├── heatwave.py         # Heatwave classifier
│  │    ├── rainfall.py         # Rainfall regressor
│  │    └── flood.py            # Flood classifier
│  ├── flood_simulation.py     # Hydrological simulation
│  ├── risk_model.py           # Climate risk index
│  ├── cli.py                  # Command-line interface
│  ├── visualization.py        # All visualizations
│  ├── alerts.py               # Alert generation
│  ├── utils.py                # Helper functions
│  └── quickstart.py           # Automated demo
├── logs/
│   └── climate_system.log
├── notebooks/
│   └── 01_complete_climate_system.ipynb
├── reports/                 # Generated reports
├── results/                 # Visualizations
├── GETTING_STARTED.md       # Prerequisites and approaches to using the app
├── GUIDE.md                 # Detailed documentation
├── PROJECT_SUMMARY.md       # Project abstract with additional coverage
├── README.md                # Project overview
├── requirements.txt         # Python dependencies
└── STREAMLIT_GUIDE.md       # Guide when using streamlit to view app
```

---

## Understanding the Outputs

### 1. Risk Dashboard (`results/risk_dashboard.html`)

**Open in browser** to see:

- Risk score timeline
- Category distribution (pie chart)
- Component scores (flood, heatwave, rainfall)
- Hazard correlation plot

**Interactive:** Hover for details, zoom, pan

### 2. Forecasts (`results/*.png`)

**Heatwave Forecast:**

- Temperature trend (14 days)
- Heatwave probability overlay
- Threshold line

**Rainfall Forecast:**

- Daily rainfall bars
- Color-coded by intensity
- Extreme threshold marked

### 3. Flood Simulation (`results/flood_simulation.png`)

**3-panel view:**

- Digital Elevation Model (terrain)
- Water depth after 100mm rain
- Flood risk zones overlay

### 4. Daily Report (`reports/demo_daily_report.txt`)

**Text-based summary:**

```text
CARIBBEAN CLIMATE IMPACT SYSTEM
DAILY RISK ASSESSMENT REPORT

Risk Score: 42.5/100
Risk Category: Low
Primary Hazards: Low Risk

Component Analysis:
  - Flood Probability: 25.3%
  - Heatwave Probability: 18.7%
  - Rainfall Severity: 12.5

Recommendations:
Low climate risk - normal precautions apply
```

---

## Customization

### Adjust Risk Thresholds

Edit `config/config.yaml`:

```yaml
models:
  heatwave:
    threshold_temperature: 32.0  # Change to 30.0 for more sensitivity
    consecutive_days: 3          # Change to 2 for earlier alerts

risk_index:
  weights:
    flood_risk: 0.40      # Adjust based on your priorities
    heatwave_prob: 0.30
    rainfall_severity: 0.30
```

### Change Location

```yaml
location:
  default_island: "Barbados"  # Change island
  latitude: 13.1939
  longitude: -59.5432
```

---

## Verify Installation

Run this quick test:

```powershell
python -c "import pandas, sklearn, xgboost, plotly; print('All packages installed correctly!')"
```

Expected output: `All packages installed correctly!`

---

## Sample Workflow

### Typical Daily Use

**Morning:**

```powershell
python cli.py --daily-report
```

Check risk assessment and alerts.

**Planning:**

```powershell
python cli.py --visualize
```

Review 14-day forecasts.

**Analysis:**
Open `results/risk_dashboard.html` in browser.

**Exploration:**
Launch Jupyter notebook for deeper analysis.

---

## Troubleshooting

### Issue: Import errors

**Solution:**

```powershell
pip install --upgrade -r requirements.txt
```

### Issue: "No module named 'src'"

**Solution:** Run from project root directory.

### Issue: Slow performance

**Solution:** Reduce data size:

```powershell
python cli.py --analyze --days 365
```

### Issue: Matplotlib plots don't show

**Solution:** Install GUI backend:

```powershell
pip install pyqt5
```

---

## Learning Path

### Beginner

1. Run `quickstart.py`
2. Explore visualizations in `results/`
3. Read the daily report

### Intermediate

1. Use CLI with different options
2. Modify `config.yaml` settings
3. Open Jupyter notebook

### Advanced

1. Read source code in `src/`
2. Add custom features
3. Integrate real data sources
4. Deploy as web app

---

## Next Steps

1. **Read GUIDE.md** for detailed component documentation
2. **Explore notebooks/** for interactive examples
3. **Review config/config.yaml** to understand all settings
4. **Run tests:** `pytest tests/` (if tests are available)
5. **Customize models** for your specific use case

---

## Key Features to Try

### 1. Flood Simulation

```powershell
python -c "from src.flood_simulation import FloodSimulator; from src.utils import load_config; s = FloodSimulator(load_config()); dem = s.generate_synthetic_dem(); w = s.simulate_rainfall_runoff(dem, 150); s.plot_simulation(dem, w)"
```

### 2. Scenario Analysis

Modify risk inputs in `src/risk_model.py` `main()` function and run:

```powershell
python src/risk_model.py
```

### 3. Feature Importance

```powershell
python src/models/heatwave.py
# Opens plot showing top predictive features
```

---

## Tips

1. **Start with synthetic data** (fast) before trying real NASA POWER data
2. **Use smaller date ranges** (--days 365) for faster testing
3. **Check results/ folder** after each run for new visualizations
4. **Save interesting configurations** in separate YAML files
5. **Export predictions** to CSV for use in other tools

---

## You're Ready

The Climate Futures System is now set up and ready to use.

**Start with:**

```powershell
python quickstart.py
```

**Then explore:**

- Open `results/risk_dashboard.html` in your browser
- Read `reports/demo_daily_report.txt`
- Check out the visualizations in `results/`

---

**Questions?** Check `GUIDE.md` for detailed documentation.
