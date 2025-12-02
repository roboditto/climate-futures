"""
Rainfall Prediction Model
Regression model to predict daily rainfall extremes.
"""

import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Tuple, Optional, List
import logging
import joblib


class RainfallPredictor:
    """
    Predicts daily rainfall using machine learning.
    """
    
    def __init__(self, config: Dict, logger: Optional[logging.Logger] = None):
        """
        Initialize rainfall predictor.
        
        Parameters:
        -----------
        config : dict
            Configuration dictionary
        logger : logging.Logger, optional
            Logger instance
        """
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        self.model_config = config['models']['rainfall']
        self.model = None
        self.feature_importance = None
        self.extreme_threshold = self.model_config['extreme_threshold']
    
    def prepare_features(self, df: pd.DataFrame,
                        target_col: str = 'precipitation',
                        forecast_days: int = 1,
                        exclude_cols: Optional[List[str]] = None) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare features and target for training.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Feature dataframe
        target_col : str
            Target column name (precipitation)
        forecast_days : int
            Number of days ahead to forecast
        exclude_cols : list, optional
            Columns to exclude from features
        
        Returns:
        --------
        tuple
            (X, y) features and target
        """
        # Default exclusions
        if exclude_cols is None:
            exclude_cols = []
        
        # Exclude precipitation-related columns (to avoid data leakage)
        always_exclude = [
            target_col,
            'precipitation_rolling_sum_3d',
            'precipitation_rolling_sum_7d',
            'precipitation_rolling_sum_14d',
            'precipitation_rolling_sum_30d',
            'precipitation_7day',
            'precipitation_30day'
        ]
        exclude_cols.extend(always_exclude)
        
        # Select feature columns
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        X = df[feature_cols].copy()
        
        # Shift target by forecast_days (predict future rainfall)
        y = df[target_col].shift(-forecast_days)
        
        # Remove rows with missing target (at the end)
        valid_idx = ~y.isna()
        X = X[valid_idx]
        y = y[valid_idx]
        
        # Handle missing values in features
        X = X.fillna(X.mean())
        X = X.replace([np.inf, -np.inf], np.nan).fillna(X.mean())
        
        self.logger.info(f"Prepared {len(feature_cols)} features for rainfall prediction")
        self.logger.info(f"Forecasting {forecast_days} days ahead")
        
        return X, y
    
    def train(self, X: pd.DataFrame, y: pd.Series,
              test_size: float = 0.2) -> Dict:
        """
        Train the rainfall prediction model.
        
        Parameters:
        -----------
        X : pd.DataFrame
            Features
        y : pd.Series
            Target (rainfall)
        test_size : float
            Proportion of data for testing
        
        Returns:
        --------
        dict
            Training results and metrics
        """
        self.logger.info("Training rainfall prediction model...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size,
            random_state=self.config['training']['random_state']
        )
        
        # Initialize model
        self.model = XGBRegressor(
            **self.model_config['hyperparameters'],
            n_jobs=-1
        )
        
        # Train model
        self.model.fit(X_train, y_train)
        
        # Predictions
        y_pred = self.model.predict(X_test)
        
        # Metrics
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        
        metrics = {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'y_test': y_test,
            'y_pred': y_pred
        }
        
        # Cross-validation
        cv_scores = cross_val_score(
            self.model, X_train, y_train, cv=5,
            scoring='neg_mean_squared_error'
        )
        metrics['cv_rmse_mean'] = np.sqrt(-cv_scores.mean())
        metrics['cv_rmse_std'] = np.sqrt(cv_scores.std())
        
        # Feature importance
        self.feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        self.logger.info(f"Training complete. RMSE: {rmse:.2f} mm, MAE: {mae:.2f} mm, R²: {r2:.3f}")
        
        return metrics
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict rainfall.
        
        Parameters:
        -----------
        X : pd.DataFrame
            Features
        
        Returns:
        --------
        np.ndarray
            Predicted rainfall (mm)
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        # Handle missing values
        X_clean = X.fillna(X.mean()).replace([np.inf, -np.inf], np.nan).fillna(X.mean())
        
        predictions = self.model.predict(X_clean)
        
        # Ensure non-negative rainfall
        predictions = np.maximum(predictions, 0)
        
        return predictions
    
    def calculate_severity(self, rainfall_mm: float) -> str:
        """
        Calculate rainfall severity category.
        
        Parameters:
        -----------
        rainfall_mm : float
            Rainfall amount in mm
        
        Returns:
        --------
        str
            Severity category
        """
        if rainfall_mm < 2.5:
            return "Light"
        elif rainfall_mm < 10:
            return "Moderate"
        elif rainfall_mm < 50:
            return "Heavy"
        elif rainfall_mm < 100:
            return "Very Heavy"
        else:
            return "Extreme"
    
    def save_model(self, filepath: str) -> None:
        """Save trained model."""
        joblib.dump(self.model, filepath)
        self.logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str) -> None:
        """Load trained model."""
        self.model = joblib.load(filepath)
        self.logger.info(f"Model loaded from {filepath}")
    
    def plot_predictions(self, metrics: Dict, 
                        save_path: Optional[str] = None) -> None:
        """
        Plot actual vs predicted rainfall.
        
        Parameters:
        -----------
        metrics : dict
            Training metrics containing y_test and y_pred
        save_path : str, optional
            Path to save figure
        """
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        y_test = metrics['y_test']
        y_pred = metrics['y_pred']
        
        # Scatter plot
        axes[0].scatter(y_test, y_pred, alpha=0.5)
        axes[0].plot([y_test.min(), y_test.max()], 
                     [y_test.min(), y_test.max()], 
                     'r--', lw=2)
        axes[0].set_xlabel('Actual Rainfall (mm)')
        axes[0].set_ylabel('Predicted Rainfall (mm)')
        axes[0].set_title(f'Actual vs Predicted\nR² = {metrics["r2"]:.3f}')
        axes[0].grid(True, alpha=0.3)
        
        # Residuals
        residuals = y_test - y_pred
        axes[1].scatter(y_pred, residuals, alpha=0.5)
        axes[1].axhline(y=0, color='r', linestyle='--', lw=2)
        axes[1].set_xlabel('Predicted Rainfall (mm)')
        axes[1].set_ylabel('Residuals (mm)')
        axes[1].set_title(f'Residual Plot\nRMSE = {metrics["rmse"]:.2f} mm')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Prediction plot saved to {save_path}")
        
        plt.show()
    
    def plot_feature_importance(self, top_n: int = 20,
                               save_path: Optional[str] = None) -> None:
        """
        Plot feature importance.
        
        Parameters:
        -----------
        top_n : int
            Number of top features to plot
        save_path : str, optional
            Path to save figure
        """
        if self.feature_importance is None:
            raise ValueError("No feature importance available. Train model first.")
        
        plt.figure(figsize=(10, 8))
        top_features = self.feature_importance.head(top_n)
        sns.barplot(data=top_features, y='feature', x='importance')
        plt.title(f'Top {top_n} Features for Rainfall Prediction')
        plt.xlabel('Importance')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Feature importance plot saved to {save_path}")
        
        plt.show()


def main():
    """Example usage."""
    from utils import load_config, setup_logging
    from data_preprocessing import ClimateDataLoader
    
    config = load_config()
    logger = setup_logging(config)
    
    # Load features
    loader = ClimateDataLoader(config, logger)
    df = loader.load_processed_data('climate_features.csv')
    
    # Initialize predictor
    predictor = RainfallPredictor(config, logger)
    
    # Prepare features (predict 1 day ahead)
    X, y = predictor.prepare_features(df, forecast_days=1)
    
    # Train
    metrics = predictor.train(X, y)
    
    print("\nRainfall Prediction Results:")
    print(f"RMSE: {metrics['rmse']:.2f} mm")
    print(f"MAE: {metrics['mae']:.2f} mm")
    print(f"R²: {metrics['r2']:.3f}")
    print(f"CV RMSE: {metrics['cv_rmse_mean']:.2f} (+/- {metrics['cv_rmse_std']:.2f})")
    
    # Save model
    predictor.save_model('data/models/rainfall_model.pkl')
    
    # Plot results
    predictor.plot_predictions(metrics, save_path='results/rainfall_predictions.png')
    predictor.plot_feature_importance(save_path='results/rainfall_feature_importance.png')


if __name__ == "__main__":
    main()
