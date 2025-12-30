"""
Command-Line Interface
Main entry point for the Climate Futures System.
"""

import argparse
import sys
import os
from pathlib import Path

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
import pandas as pd
import numpy as np


class ClimateSystem:
    """
    Main Climate Futures System.
    """
    
    def __init__(self):
        """Initialize the system."""
        self.config = load_config()
        self.logger = setup_logging(self.config)
        
        # Initialize components
        self.data_loader = ClimateDataLoader(self.config, self.logger)
        self.feature_engineer = ClimateFeatureEngineer(self.config, self.logger)
        self.heatwave_model = HeatwavePredictor(self.config, self.logger)
        self.rainfall_model = RainfallPredictor(self.config, self.logger)
        self.flood_model = FloodPredictor(self.config, self.logger)
        self.risk_calculator = ClimateRiskIndex(self.config, self.logger)
        self.visualizer = ClimateVisualizer(self.config, self.logger)
        self.alert_system = ClimateAlertSystem(self.config, self.logger)
        
        self.logger.info("Climate Futures System initialized")
    
    def setup_data(self, days: int = 730, use_real_data: bool = False):
        """
        Set up climate data.
        
        Parameters:
        -----------
        days : int
            Number of days of data to generate/load
        use_real_data : bool
            Whether to attempt to fetch real data
        """
        self.logger.info("Setting up climate data...")
        
        try:
            # Try to load existing processed data
            df = self.data_loader.load_processed_data('climate_features.csv')
            self.logger.info(f"Loaded existing data: {len(df)} records")
        except FileNotFoundError:
            self.logger.info("No existing data found, generating new data...")
            
            if use_real_data:
                # Attempt to fetch real data from NASA POWER
                try:
                    from datetime import datetime, timedelta
                    end_date = datetime.now()
                    start_date = end_date - timedelta(days=days)
                    
                    df_raw = self.data_loader.fetch_nasa_power_data(
                        start_date.strftime('%Y%m%d'),
                        end_date.strftime('%Y%m%d')
                    )
                except Exception as e:
                    self.logger.warning(f"Failed to fetch real data: {e}")
                    self.logger.info("Falling back to generated data...")
                    df_raw = self.data_loader.generate_sample_data(days=days)
            else:
                df_raw = self.data_loader.generate_sample_data(days=days)
            
            # Process data
            df_clean = self.data_loader.clean_data(df_raw)
            df_processed = self.data_loader.calculate_derived_metrics(df_clean)
            
            # Engineer features
            df = self.feature_engineer.create_all_features(df_processed)
            
            # Save
            self.data_loader.save_processed_data(df, 'climate_features.csv')
        
        return df
    
    def train_models(self, df: pd.DataFrame):
        """
        Train all ML models.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Feature data
        """
        self.logger.info("Training machine learning models...")
        
        # Create targets
        df['is_heatwave'] = self.heatwave_model.create_target_variable(df)
        df['is_flood_risk'] = self.flood_model.create_target_variable(df)
        
        # Train heatwave model
        self.logger.info("Training heatwave model...")
        X_hw, y_hw = self.heatwave_model.prepare_features(df)
        hw_metrics = self.heatwave_model.train(X_hw, y_hw)
        self.heatwave_model.save_model('data/models/heatwave_model.pkl')
        
        # Train rainfall model
        self.logger.info("Training rainfall model...")
        X_rf, y_rf = self.rainfall_model.prepare_features(df, forecast_days=1)
        rf_metrics = self.rainfall_model.train(X_rf, y_rf)
        self.rainfall_model.save_model('data/models/rainfall_model.pkl')
        
        # Train flood model
        self.logger.info("Training flood model...")
        X_fl, y_fl = self.flood_model.prepare_features(df)
        fl_metrics = self.flood_model.train(X_fl, y_fl)
        self.flood_model.save_model('data/models/flood_model.pkl')
        
        # Print metrics
        print("\n" + "="*80)
        print("MODEL TRAINING RESULTS")
        print("="*80)
        print(f"\n🌡️  Heatwave Model:")
        print(f"   Accuracy: {hw_metrics['accuracy']:.3f}")
        print(f"   ROC-AUC: {hw_metrics['roc_auc']:.3f}")
        
        print(f"\n🌧️  Rainfall Model:")
        print(f"   RMSE: {rf_metrics['rmse']:.2f} mm")
        print(f"   R²: {rf_metrics['r2']:.3f}")
        
        print(f"\n🌊 Flood Model:")
        print(f"   Accuracy: {fl_metrics['accuracy']:.3f}")
        print(f"   ROC-AUC: {fl_metrics['roc_auc']:.3f}")
        print("="*80 + "\n")
    
    def generate_predictions(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate predictions for all models.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Feature data
        
        Returns:
        --------
        pd.DataFrame
            Data with predictions
        """
        self.logger.info("Generating predictions...")
        
        df_pred = df.copy()
        
        # Load models if not already trained
        try:
            self.heatwave_model.load_model('data/models/heatwave_model.pkl')
            self.rainfall_model.load_model('data/models/rainfall_model.pkl')
            self.flood_model.load_model('data/models/flood_model.pkl')
        except FileNotFoundError:
            self.logger.warning("Models not found, training new models...")
            self.train_models(df)
        
        # Prepare features
        df_pred['is_heatwave'] = self.heatwave_model.create_target_variable(df)
        df_pred['is_flood_risk'] = self.flood_model.create_target_variable(df)
        
        X_hw, _ = self.heatwave_model.prepare_features(df_pred)
        X_rf, _ = self.rainfall_model.prepare_features(df_pred, forecast_days=1)
        X_fl, _ = self.flood_model.prepare_features(df_pred)
        
        # Generate predictions
        df_pred['heatwave_prob'] = self.heatwave_model.predict(X_hw)
        df_pred['rainfall_pred'] = self.rainfall_model.predict(X_rf)
        df_pred['flood_prob'] = self.flood_model.predict(X_fl)
        
        # Calculate risk scores
        df_pred = self.risk_calculator.calculate_batch_risk(
            df_pred,
            flood_col='flood_prob',
            heatwave_col='heatwave_prob',
            rainfall_col='rainfall_pred'
        )
        
        return df_pred
    
    def get_current_risk(self, df: pd.DataFrame) -> dict:
        """
        Get current risk assessment.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Data with predictions
        
        Returns:
        --------
        dict
            Current risk summary
        """
        latest = df.iloc[-1]
        
        # Format date safely - check if index name is a timestamp
        date_str = None
        try:
            import pandas as pd
            if isinstance(latest.name, pd.Timestamp):
                date_str = latest.name.strftime('%Y-%m-%d')
        except:
            pass
        
        summary = self.risk_calculator.generate_risk_summary(
            flood_prob=latest['flood_prob'],
            heatwave_prob=latest['heatwave_prob'],
            rainfall_mm=latest['rainfall_pred'],
            temperature=latest.get('temperature', 28),
            date=date_str
        )
        
        return summary
    
    def generate_daily_report(self, df: pd.DataFrame):
        """
        Generate comprehensive daily report.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Data with predictions
        """
        # Get current risk
        risk_summary = self.get_current_risk(df)
        
        # Generate alerts
        latest = df.iloc[-1]
        alerts = self.alert_system.generate_alerts(
            flood_prob=latest['flood_prob'],
            heatwave_prob=latest['heatwave_prob'],
            rainfall_mm=latest['rainfall_pred'],
            risk_score=risk_summary['overall_risk_score']
        )
        
        # Generate report (respect output_dir if provided)
        from datetime import datetime
        outdir = self.config.get('output_dir', '.')
        report_dir = os.path.join(outdir, 'reports') if outdir else 'reports'
        os.makedirs(report_dir, exist_ok=True)
        report_path = os.path.join(report_dir, f"daily_report_{datetime.now().strftime('%Y%m%d')}.txt")
        report = self.alert_system.generate_daily_report(risk_summary, alerts, save_path=report_path)

        print(report)
        print(f"\n✅ Report saved to: {report_path}")
    
    def create_visualizations(self, df: pd.DataFrame):
        """
        Create all visualizations.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Data with predictions
        """
        self.logger.info("Creating visualizations...")
        
        # Determine results directory (allow override via config)
        outdir = self.config.get('output_dir', '.')
        results_dir = os.path.join(outdir, 'results') if outdir else 'results'
        os.makedirs(results_dir, exist_ok=True)

        # Risk dashboard
        self.visualizer.plot_risk_dashboard(df, save_path=os.path.join(results_dir, 'risk_dashboard.html'))

        # Forecasts
        self.visualizer.plot_heatwave_forecast(df, days_ahead=14, save_path=os.path.join(results_dir, 'heatwave_forecast.png'))
        self.visualizer.plot_rainfall_forecast(df, days_ahead=14, save_path=os.path.join(results_dir, 'rainfall_forecast.png'))

        # Current risk gauge
        latest_risk = df['climate_risk_score'].iloc[-1]
        self.visualizer.plot_risk_gauge(latest_risk, save_path=os.path.join(results_dir, 'risk_gauge.html'))

        print(f"\n✅ Visualizations created in '{results_dir}' folder")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description='Climate Futures Simulation & Early Warning System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Complete analysis with visualizations
  python cli.py --analyze --visualize
  
  # Train models only
  python cli.py --train
  
  # Generate daily report
  python cli.py --daily-report
  
  # Get current risk score
  python cli.py --risk-score
  
  # Create visualizations
  python cli.py --visualize
        """
    )
    
    parser.add_argument('--analyze', action='store_true',
                       help='Run complete analysis')
    parser.add_argument('--train', action='store_true',
                       help='Train ML models')
    parser.add_argument('--daily-report', action='store_true',
                       help='Generate daily risk report')
    parser.add_argument('--risk-score', action='store_true',
                       help='Show current risk score')
    parser.add_argument('--visualize', action='store_true',
                       help='Create visualizations')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Directory to write outputs (reports, results).')
    parser.add_argument('--coastline-shp', type=str, default=None,
                       help='Path to coastline shapefile for masking/mapping.')
    parser.add_argument('--days', type=int, default=730,
                       help='Number of days of data (default: 730)')
    parser.add_argument('--real-data', action='store_true',
                       help='Attempt to fetch real climate data')
    
    args = parser.parse_args()
    
    # If no arguments, show help
    if len(sys.argv) == 1:
        parser.print_help()
        return
    
    # Initialize system
    print("\n🌎 Climate Futures System")
    print("="*80)
    
    system = ClimateSystem()
    # Apply CLI-specified output directory and coastline shapefile to system config
    if args.output_dir:
        system.config['output_dir'] = args.output_dir
        os.makedirs(args.output_dir, exist_ok=True)
        os.makedirs(os.path.join(args.output_dir, 'results'), exist_ok=True)
        os.makedirs(os.path.join(args.output_dir, 'reports'), exist_ok=True)
    if args.coastline_shp:
        system.config['coastline_shapefile'] = args.coastline_shp
    
    # Setup data
    df = system.setup_data(days=args.days, use_real_data=args.real_data)
    
    # Execute requested operations
    if args.train or args.analyze:
        system.train_models(df)
    
    # Generate predictions
    df_pred = system.generate_predictions(df)
    
    if args.risk_score or args.analyze:
        print("\n📊 CURRENT CLIMATE RISK")
        print("="*80)
        risk_summary = system.get_current_risk(df_pred)
        print(f"Risk Score: {risk_summary['overall_risk_score']}/100")
        print(f"Category: {risk_summary['risk_category']}")
        print(f"Primary Hazards: {', '.join(risk_summary['primary_hazards'])}")
        print("="*80)
    
    if args.daily_report or args.analyze:
        system.generate_daily_report(df_pred)
    
    if args.visualize or args.analyze:
        system.create_visualizations(df_pred)
    
    print("\n✅ Climate Futures System - Operations Complete\n")


if __name__ == "__main__":
    main()
