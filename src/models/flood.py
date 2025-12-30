"""
Flood Risk Model
Predicts flood susceptibility using terrain and climate data.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Tuple, Optional, List
import logging
import joblib


class FloodPredictor:
    """
    Predicts flood risk using climate and terrain data.
    """
    
    def __init__(self, config: Dict, logger: Optional[logging.Logger] = None):
        """
        Initialize flood predictor.
        
        Parameters:
        -----------
        config : dict
            Configuration dictionary
        logger : logging.Logger, optional
            Logger instance
        """
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        self.model_config = config['models']['flood']
        self.model = None
        self.feature_importance = None
        self.rainfall_threshold = self.model_config['rainfall_threshold']
    
    def create_target_variable(self, df: pd.DataFrame) -> pd.Series:
        """
        Create binary target variable for flood events.
        
        Flood risk is based on:
        - High rainfall (> threshold)
        - Cumulative rainfall over past days
        - Historical flooding patterns
        
        Parameters:
        -----------
        df : pd.DataFrame
            Climate data
        
        Returns:
        --------
        pd.Series
            Binary target (1 = flood risk, 0 = no flood risk)
        """
        self.logger.info(f"Creating flood risk target (threshold={self.rainfall_threshold} mm)")
        
        # Multiple criteria for flood risk
        criteria = []
        
        # 1. Heavy single-day rainfall
        if 'precipitation' in df.columns:
            heavy_rain = (df['precipitation'] > self.rainfall_threshold).astype(int)
            criteria.append(heavy_rain)
        
        # 2. High cumulative rainfall
        if 'precipitation_7day' in df.columns:
            high_cumulative = (df['precipitation_7day'] > self.rainfall_threshold * 2).astype(int)
            criteria.append(high_cumulative)
        elif 'precipitation' in df.columns:
            cumulative_7d = df['precipitation'].rolling(window=7, min_periods=1).sum()
            high_cumulative = (cumulative_7d > self.rainfall_threshold * 2).astype(int)
            criteria.append(high_cumulative)
        
        # 3. Saturated conditions (consecutive wet days)
        if 'consecutive_wet_days' in df.columns:
            saturated = (df['consecutive_wet_days'] > 5).astype(int)
            criteria.append(saturated)
        
        # Combine criteria (any one triggers flood risk)
        if criteria:
            is_flood_risk = pd.Series(np.maximum.reduce(criteria), index=df.index)
        else:
            is_flood_risk = pd.Series(0, index=df.index)
        
        self.logger.info(f"Flood risk days: {is_flood_risk.sum()} ({is_flood_risk.mean()*100:.1f}%)")
        
        return is_flood_risk
    
    def add_elevation_features(self, df: pd.DataFrame,
                              elevation_percentile: float = 25.0) -> pd.DataFrame:
        """
        Add synthetic elevation features (in absence of real DEM data).
        
        Parameters:
        -----------
        df : pd.DataFrame
            Climate data
        elevation_percentile : float
            Percentile for low-lying areas
        
        Returns:
        --------
        pd.DataFrame
            Data with elevation features
        """
        self.logger.info("Adding synthetic elevation features...")
        
        df_elev = df.copy()
        
        # Simulate elevation distribution (0-100m for coastal areas)
        np.random.seed(42)
        n = len(df)
        
        # Synthetic elevation (bimodal: coastal lowlands + inland hills)
        elevation = np.concatenate([
            np.random.gamma(2, 10, int(n*0.7)),  # Lowlands
            np.random.gamma(5, 15, n - int(n*0.7))  # Hills
        ])
        np.random.shuffle(elevation)
        elevation = elevation[:n]
        
        df_elev['elevation_m'] = elevation
        
        # Low-lying area indicator
        low_threshold = np.percentile(elevation, elevation_percentile)
        df_elev['is_lowland'] = (elevation < low_threshold).astype(int)
        
        # Slope (derived from elevation changes - simplified)
        df_elev['slope_degrees'] = np.abs(np.gradient(elevation)) * 10  # Synthetic slope
        df_elev['is_flat'] = (df_elev['slope_degrees'] < 2).astype(int)
        
        # Distance to coast (synthetic - assume closer is riskier)
        df_elev['coastal_proximity'] = np.random.beta(2, 5, n) * 100  # 0-100km
        df_elev['is_coastal'] = (df_elev['coastal_proximity'] < 10).astype(int)
        
        self.logger.info("Added synthetic elevation features")
        
        return df_elev
    
    def prepare_features(self, df: pd.DataFrame,
                        target_col: str = 'is_flood_risk',
                        include_elevation: bool = True,
                        exclude_cols: Optional[List[str]] = None) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare features and target for training.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Feature DataFrame
        target_col : str
            Target column name
        include_elevation : bool
            Whether to add elevation features
        exclude_cols : list, optional
            Columns to exclude from features
        
        Returns:
        --------
        tuple
            (X, y) features and target
        """
        df_prep = df.copy()
        
        # Add elevation features if requested
        if include_elevation and 'elevation_m' not in df_prep.columns:
            df_prep = self.add_elevation_features(df_prep)
        
        # Default exclusions
        if exclude_cols is None:
            exclude_cols = []
        
        # Always exclude target and highly correlated features
        always_exclude = [
            target_col,
            'precipitation',  # Too direct
            'precipitation_7day',
            'precipitation_30day'
        ]
        exclude_cols.extend(always_exclude)
        
        # Select feature columns
        feature_cols = [col for col in df_prep.columns if col not in exclude_cols]
        
        X = df_prep[feature_cols].copy()
        y = df_prep[target_col].copy()
        
        # Handle missing values
        X = X.fillna(X.mean())
        X = X.replace([np.inf, -np.inf], np.nan).fillna(X.mean())
        
        self.logger.info(f"Prepared {len(feature_cols)} features for flood prediction")
        
        return X, y
    
    def train(self, X: pd.DataFrame, y: pd.Series,
              test_size: float = 0.2) -> Dict:
        """
        Train the flood prediction model.
        
        Parameters:
        -----------
        X : pd.DataFrame
            Features
        y : pd.Series
            Target
        test_size : float
            Proportion of data for testing
        
        Returns:
        --------
        dict
            Training results and metrics
        """
        self.logger.info("Training flood prediction model...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size,
            random_state=self.config['training']['random_state'],
            stratify=y
        )
        
        # Initialize model
        self.model = RandomForestClassifier(
            **self.model_config['hyperparameters'],
            n_jobs=-1
        )
        
        # Train model
        self.model.fit(X_train, y_train)
        
        # Check if model learned both classes
        n_classes = len(self.model.classes_)
        
        # Predictions
        y_pred = self.model.predict(X_test)
        
        # Handle single class case
        if n_classes == 1:
            self.logger.warning(f"Only one class present in training data: {self.model.classes_[0]}")
            y_pred_proba = np.zeros(len(X_test))
            roc_auc = 0.5
        else:
            y_pred_proba = self.model.predict_proba(X_test)[:, 1]
            roc_auc = roc_auc_score(y_test, y_pred_proba)
        
        # Metrics
        metrics = {
            'accuracy': self.model.score(X_test, y_test),
            'classification_report': classification_report(y_test, y_pred),
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'roc_auc': roc_auc
        }
        
        # Cross-validation (only if we have both classes)
        if n_classes > 1:
            cv_scores = cross_val_score(
                self.model, X_train, y_train, cv=5, scoring='roc_auc'
            )
            metrics['cv_auc_mean'] = cv_scores.mean()
            metrics['cv_auc_std'] = cv_scores.std()
        else:
            metrics['cv_auc_mean'] = 0.5
            metrics['cv_auc_std'] = 0.0
        
        # Feature importance
        self.feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        self.logger.info(f"Training complete. Accuracy: {metrics['accuracy']:.3f}, AUC: {metrics['roc_auc']:.3f}")
        
        return metrics
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict flood probability.
        
        Parameters:
        -----------
        X : pd.DataFrame
            Features
        
        Returns:
        --------
        np.ndarray
            Probability of flood
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        # Handle missing values
        X_clean = X.fillna(X.mean()).replace([np.inf, -np.inf], np.nan).fillna(X.mean())
        
        # Handle single class case
        if len(self.model.classes_) == 1:
            return np.zeros(len(X_clean))
        
        return self.model.predict_proba(X_clean)[:, 1]
    
    def save_model(self, filepath: str) -> None:
        """Save trained model."""
        joblib.dump(self.model, filepath)
        self.logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str) -> None:
        """Load trained model."""
        self.model = joblib.load(filepath)
        self.logger.info(f"Model loaded from {filepath}")
    
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
        plt.title(f'Top {top_n} Features for Flood Prediction')
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
    predictor = FloodPredictor(config, logger)
    
    # Create target
    df['is_flood_risk'] = predictor.create_target_variable(df)
    
    # Prepare features
    X, y = predictor.prepare_features(df, include_elevation=True)
    
    # Train
    metrics = predictor.train(X, y)
    
    print("\nFlood Prediction Results:")
    print(f"Accuracy: {metrics['accuracy']:.3f}")
    print(f"ROC-AUC: {metrics['roc_auc']:.3f}")
    print(f"CV AUC: {metrics['cv_auc_mean']:.3f} (+/- {metrics['cv_auc_std']:.3f})")
    print("\nClassification Report:")
    print(metrics['classification_report'])
    
    # Save model
    predictor.save_model('data/models/flood_model.pkl')
    
    # Plot feature importance
    predictor.plot_feature_importance(save_path='results/flood_feature_importance.png')


if __name__ == "__main__":
    main()
