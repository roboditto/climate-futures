"""
Quick Start Script for Climate Futures System
Runs a complete demonstration of all features.
"""

import sys
import os
from pathlib import Path
import numpy as np

# Set UTF-8 encoding for Windows console
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

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


def main():
    """Run complete system demonstration."""
    print("\n" + "="*80)
    print("🌎 CARIBBEAN CLIMATE IMPACT SIMULATION & EARLY WARNING SYSTEM")
    print("="*80)
    print("\nThis demonstration will:")
    print("  1. Generate synthetic climate data (2 years)")
    print("  2. Engineer scientifically meaningful features")
    print("  3. Train 3 machine learning models")
    print("  4. Calculate climate risk scores")
    print("  5. Generate visualizations and alerts")
    print("  6. Create a flood simulation")
    print("\n" + "="*80 + "\n")
    
    # Initialize
    config = load_config()
    logger = setup_logging(config)
    
    data_loader = ClimateDataLoader(config, logger)
    feature_engineer = ClimateFeatureEngineer(config, logger)
    risk_calculator = ClimateRiskIndex(config, logger)
    visualizer = ClimateVisualizer(config, logger)
    alert_system = ClimateAlertSystem(config, logger)
    flood_simulator = FloodSimulator(config, logger)
    
    # Create output directories
    os.makedirs('results', exist_ok=True)
    os.makedirs('reports', exist_ok=True)
    os.makedirs('data/models', exist_ok=True)
    
    # Step 1: Generate/Load Data
    print("\n📊 STEP 1: Generating Climate Data...")
    print("-" * 80)
    
    try:
        df = data_loader.load_processed_data('climate_features.csv')
        print(f"✅ Loaded existing data: {len(df)} records")
    except FileNotFoundError:
        print("Generating 2 years of synthetic Caribbean climate data...")
        df_raw = data_loader.generate_sample_data(days=730)
        df_clean = data_loader.clean_data(df_raw)
        df_processed = data_loader.calculate_derived_metrics(df_clean)
        
        print("\n🔧 STEP 2: Engineering Features...")
        print("-" * 80)
        df = feature_engineer.create_all_features(df_processed)
        data_loader.save_processed_data(df, 'climate_features.csv')
        
        print(f"✅ Created {len(df.columns)} features from {len(df)} days of data")
    
    # Step 3: Train Models
    print("\n🤖 STEP 3: Training Machine Learning Models...")
    print("-" * 80)
    
    # Heatwave Model
    heatwave_model = HeatwavePredictor(config, logger)
    df['is_heatwave'] = heatwave_model.create_target_variable(df)
    X_hw, y_hw = heatwave_model.prepare_features(df)
    hw_metrics = heatwave_model.train(X_hw, y_hw)
    heatwave_model.save_model('data/models/heatwave_model.pkl')
    print(f"✅ Heatwave Model: Accuracy={hw_metrics['accuracy']:.3f}, AUC={hw_metrics['roc_auc']:.3f}")
    
    # Rainfall Model
    rainfall_model = RainfallPredictor(config, logger)
    X_rf, y_rf = rainfall_model.prepare_features(df, forecast_days=1)
    rf_metrics = rainfall_model.train(X_rf, y_rf)
    rainfall_model.save_model('data/models/rainfall_model.pkl')
    print(f"✅ Rainfall Model: RMSE={rf_metrics['rmse']:.2f} mm, R²={rf_metrics['r2']:.3f}")
    
    # Flood Model
    flood_model = FloodPredictor(config, logger)
    df['is_flood_risk'] = flood_model.create_target_variable(df)
    X_fl, y_fl = flood_model.prepare_features(df)
    fl_metrics = flood_model.train(X_fl, y_fl)
    flood_model.save_model('data/models/flood_model.pkl')
    print(f"✅ Flood Model: Accuracy={fl_metrics['accuracy']:.3f}, AUC={fl_metrics['roc_auc']:.3f}")
    
    # Step 4: Generate Predictions
    print("\n🔮 STEP 4: Generating Predictions...")
    print("-" * 80)
    
    # Predictions
    df['heatwave_prob'] = heatwave_model.predict(X_hw)
    
    # Rainfall predictions are 1-day ahead, so we need to align them
    rainfall_preds = rainfall_model.predict(X_rf)
    # Initialize with NaN
    df['rainfall_pred'] = np.nan
    # Assign predictions to the rows that have valid features (exclude last day)
    df.loc[df.index[:len(rainfall_preds)], 'rainfall_pred'] = rainfall_preds
    
    df['flood_prob'] = flood_model.predict(X_fl)
    
    # Calculate risk scores (only for rows with all predictions)
    df = risk_calculator.calculate_batch_risk(df)
    print(f"✅ Predictions generated for {len(df)} days ({df['rainfall_pred'].notna().sum()} with rainfall forecast)")

    # Short ERA5 regrid test: attempt to download & regrid a small hourly window for immediate demo use
    try:
        from era5_helper import download_era5_hourly, regrid_era5_to_grid, CDSAPI_AVAILABLE
        import numpy as _np
        if not _np.exists('data'):
            os.makedirs('data', exist_ok=True)
        regrid_path = 'data/era5_test_regrid.npy'
        if not os.path.exists(regrid_path):
            if CDSAPI_AVAILABLE:
                print('\n🌐 Attempting short ERA5 download+regrid for demo (this may take a minute)...')
                # use latest date in df as reference
                latest_date = df.index[-1].date()
                start = (latest_date - np.timedelta64(1, 'D')).astype('datetime64[D]').astype(str)
                end = latest_date.astype('datetime64[D]').astype(str)
                bounds = (17.7, -78.4, 18.5, -76.1)  # Jamaica example bbox
                nc_out = 'data/era5_test.nc'
                nc = download_era5_hourly(bounds, start, end, out_path=nc_out)
                if nc:
                    arr = regrid_era5_to_grid(nc, target_shape=(50, 50), target_bounds=bounds)
                    if arr is not None:
                        _np.save(regrid_path, arr)
                        print(f'✅ ERA5 regrid saved to {regrid_path}')
                    else:
                        print('⚠️ ERA5 regrid returned None')
                else:
                    print('⚠️ ERA5 download failed or was skipped')
            else:
                print('ℹ️ CDS API not available; skipping ERA5 demo step')
        else:
            print(f'ℹ️ Found existing ERA5 regrid: {regrid_path}')
    except Exception as e:
        print('⚠️ ERA5 demo step failed:', e)
    
    # Step 5: Current Risk Assessment
    print("\n📊 STEP 5: Current Climate Risk Assessment...")
    print("-" * 80)
    
    latest = df.iloc[-1]
    risk_summary = risk_calculator.generate_risk_summary(
        flood_prob=latest['flood_prob'],
        heatwave_prob=latest['heatwave_prob'],
        rainfall_mm=latest['rainfall_pred'],
        temperature=latest.get('temperature', 28)
    )
    
    print(f"\n🎯 Overall Risk Score: {risk_summary['overall_risk_score']}/100")
    print(f"📈 Risk Category: {risk_summary['risk_category']}")
    print(f"⚠️  Primary Hazards: {', '.join(risk_summary['primary_hazards'])}")
    print(f"\n💡 Top Recommendation:")
    if risk_summary['recommendations']:
        print(f"   {risk_summary['recommendations'][0]}")
    
    # Step 6: Generate Alerts
    print("\n🚨 STEP 6: Checking for Active Alerts...")
    print("-" * 80)
    
    alerts = alert_system.generate_alerts(
        flood_prob=latest['flood_prob'],
        heatwave_prob=latest['heatwave_prob'],
        rainfall_mm=latest['rainfall_pred'],
        risk_score=risk_summary['overall_risk_score']
    )
    
    if alerts:
        print(f"⚠️  {len(alerts)} active alert(s):")
        for i, alert in enumerate(alerts, 1):
            print(f"   {i}. [{alert['type'].upper()}] {alert['severity'].upper()}")
    else:
        print("✅ No active alerts - conditions are normal")
    
    # Generate daily report
    report_path = 'reports/demo_daily_report.txt'
    alert_system.generate_daily_report(risk_summary, alerts, save_path=report_path)
    print(f"✅ Daily report saved to: {report_path}")
    
    # Step 7: Flood Simulation
    print("\n🌊 STEP 7: Running Flood Simulation...")
    print("-" * 80)
    
    # Generate DEM
    dem = flood_simulator.generate_synthetic_dem(size=(100, 100), elevation_range=(0, 80))
    
    # Simulate 100mm rainfall event
    water_depth = flood_simulator.simulate_rainfall_runoff(dem, rainfall_mm=100, duration_hours=6)
    flood_simulator.plot_simulation(dem, water_depth, save_path='results/flood_simulation.png')
    
    print(f"✅ Max flood depth: {water_depth.max():.3f} m")
    print(f"✅ Simulation visualization saved to: results/flood_simulation.png")
    
    # Step 8: Create Visualizations
    print("\n🎨 STEP 8: Creating Visualizations...")
    print("-" * 80)
    
    # Risk dashboard
    visualizer.plot_risk_dashboard(df.tail(90), save_path='results/risk_dashboard.html')
    print("✅ Risk dashboard: results/risk_dashboard.html")
    
    # Forecasts
    visualizer.plot_heatwave_forecast(df, days_ahead=14, save_path='results/heatwave_forecast.png')
    print("✅ Heatwave forecast: results/heatwave_forecast.png")
    
    visualizer.plot_rainfall_forecast(df, days_ahead=14, save_path='results/rainfall_forecast.png')
    print("✅ Rainfall forecast: results/rainfall_forecast.png")
    
    # Risk gauge
    visualizer.plot_risk_gauge(risk_summary['overall_risk_score'], save_path='results/risk_gauge.html')
    print("✅ Risk gauge: results/risk_gauge.html")
    
    # Final Summary
    print("\n" + "="*80)
    print("✅ DEMONSTRATION COMPLETE!")
    print("="*80)
    print("\n📁 Generated Files:")
    print("   Models: data/models/")
    print("   Reports: reports/")
    print("   Visualizations: results/")
    print("\n🔍 Next Steps:")
    print("   1. Explore the generated visualizations in results/")
    print("   2. Read the daily report in reports/")
    print("   3. Run 'python cli.py --help' for more options")
    print("   4. Open notebooks/01_complete_climate_system.ipynb for detailed analysis")
    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    main()
