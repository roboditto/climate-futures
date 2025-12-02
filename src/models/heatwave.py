"""
Heatwave Prediction Model
Binary classification model to predict heatwave events.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Tuple, Optional, List
import logging
import joblib


class HeatwavePredictor:
    """
    Predicts heatwave probability using machine learning.
    """
    
    def __init__(self, config: Dict, logger: Optional[logging.Logger] = None):
        """
        Initialize heatwave predictor.
        
        Parameters:
        -----------
        config : dict
            Configuration dictionary
        logger : logging.Logger, optional
            Logger instance
        """
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        self.model_config = config['models']['heatwave']
        self.model = None
        self.feature_importance = None
        self.threshold = self.model_config['threshold_temperature']
        self.consecutive_days = self.model_config['consecutive_days']
    
    def create_target_variable(self, df: pd.DataFrame) -> pd.Series:
        """
        Create binary target variable for heatwave events.
        
        A heatwave is defined as consecutive days above threshold temperature.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Climate data with temperature
        
        Returns:
        --------
        pd.Series
            Binary target (1 = heatwave, 0 = no heatwave)
        """
        self.logger.info(f"Creating heatwave target (threshold={self.threshold}°C, consecutive={self.consecutive_days} days)")
        
        # Identify days above threshold
        above_threshold = (df['temperature'] > self.threshold).astype(int)
        
        # Count consecutive days
        consecutive = (
            above_threshold.groupby((above_threshold != above_threshold.shift()).cumsum()).cumsum()
        )
        
        # Heatwave if consecutive days >= threshold
        is_heatwave = (consecutive >= self.consecutive_days).astype(int)
        
        self.logger.info(f"Heatwave days: {is_heatwave.sum()} ({is_heatwave.mean()*100:.1f}%)")
        
        return is_heatwave
    
    def prepare_features(self, df: pd.DataFrame,
                        target_col: str = 'is_heatwave',
                        exclude_cols: Optional[List[str]] = None) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare features and target for training.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Feature dataframe
        target_col : str
            Target column name
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
        
        # Always exclude these
        always_exclude = [
            target_col, 'temperature_max', 'temperature_min', 'temperature',
            'heat_index', 'wet_bulb_temp'  # Too correlated with target
        ]
        exclude_cols.extend(always_exclude)
        
        # Select feature columns
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        X = df[feature_cols].copy()
        y = df[target_col].copy()
        
        # Handle missing values
        X = X.fillna(X.mean())
        
        # Remove infinite values
        X = X.replace([np.inf, -np.inf], np.nan).fillna(X.mean())
        
        self.logger.info(f"Prepared {len(feature_cols)} features for training")
        
        return X, y
    
    def train(self, X: pd.DataFrame, y: pd.Series,
              test_size: float = 0.2) -> Dict:
        """
        Train the heatwave prediction model.
        
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
        self.logger.info("Training heatwave prediction model...")
        
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
            y_pred_proba = np.zeros(len(X_test))  # All probabilities are 0 for the positive class
            roc_auc = 0.5  # Random performance
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
        Predict heatwave probability.
        
        Parameters:
        -----------
        X : pd.DataFrame
            Features
        
        Returns:
        --------
        np.ndarray
            Probability of heatwave
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        # Handle missing values
        X_clean = X.fillna(X.mean()).replace([np.inf, -np.inf], np.nan).fillna(X.mean())
        
        # Handle single class case
        if len(self.model.classes_) == 1:
            return np.zeros(len(X_clean))  # All probabilities are 0 for positive class
        
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
        plt.title(f'Top {top_n} Features for Heatwave Prediction')
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
    predictor = HeatwavePredictor(config, logger)
    
    # Create target
    df['is_heatwave'] = predictor.create_target_variable(df)
    
    # Prepare features
    X, y = predictor.prepare_features(df)
    
    # Train
    metrics = predictor.train(X, y)
    
    print("\nHeatwave Prediction Results:")
    print(f"Accuracy: {metrics['accuracy']:.3f}")
    print(f"ROC-AUC: {metrics['roc_auc']:.3f}")
    print(f"CV AUC: {metrics['cv_auc_mean']:.3f} (+/- {metrics['cv_auc_std']:.3f})")
    print("\nClassification Report:")
    print(metrics['classification_report'])
    
    # Save model
    predictor.save_model('data/models/heatwave_model.pkl')
    
    # Plot feature importance
    predictor.plot_feature_importance(save_path='results/heatwave_feature_importance.png')


if __name__ == "__main__":
    main()
