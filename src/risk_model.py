"""
Climate Risk Index Model - Day 9
Combines all predictions into a unified 0-100 risk score.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple
import logging


class ClimateRiskIndex:
    """
    Calculates comprehensive climate risk score.
    """
    
    def __init__(self, config: Dict, logger: Optional[logging.Logger] = None):
        """
        Initialize risk index calculator.
        
        Parameters:
        -----------
        config : dict
            Configuration dictionary
        logger : logging.Logger, optional
            Logger instance
        """
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        self.risk_config = config['risk_index']
        self.weights = self.risk_config['weights']
        self.thresholds = self.risk_config['thresholds']
        self.categories = self.risk_config['categories']
    
    def calculate_risk_score(self, 
                            flood_prob: float,
                            heatwave_prob: float,
                            rainfall_severity: float,
                            rainfall_mm: Optional[float] = None) -> float:
        """
        Calculate combined climate risk score (0-100).
        
        Parameters:
        -----------
        flood_prob : float
            Flood probability (0-1)
        heatwave_prob : float
            Heatwave probability (0-1)
        rainfall_severity : float
            Rainfall severity measure
        rainfall_mm : float, optional
            Actual rainfall in mm (for normalization)
        
        Returns:
        --------
        float
            Risk score (0-100)
        """
        # Normalize rainfall severity to 0-1 scale
        if rainfall_mm is not None:
            # Normalize based on extreme threshold (100mm = 1.0)
            rainfall_norm = min(rainfall_mm / 100.0, 1.0)
        else:
            # Assume rainfall_severity is already normalized
            rainfall_norm = min(rainfall_severity / 100.0, 1.0)
        
        # Calculate weighted combination
        risk_score = (
            self.weights['flood_risk'] * flood_prob * 100 +
            self.weights['heatwave_prob'] * heatwave_prob * 100 +
            self.weights['rainfall_severity'] * rainfall_norm * 100
        )
        
        # Ensure 0-100 range
        risk_score = np.clip(risk_score, 0, 100)
        
        return risk_score
    
    def get_risk_category(self, risk_score: float) -> Dict[str, any]:
        """
        Get risk category based on score.
        
        Parameters:
        -----------
        risk_score : float
            Risk score (0-100)
        
        Returns:
        --------
        dict
            Category information
        """
        for category in self.categories:
            min_val, max_val = category['range']
            if min_val <= risk_score < max_val or (risk_score == 100 and max_val == 100):
                return {
                    'name': category['name'],
                    'color': category['color'],
                    'range': category['range'],
                    'score': risk_score
                }
        
        # Default to extreme if above all thresholds
        return {
            'name': self.categories[-1]['name'],
            'color': self.categories[-1]['color'],
            'range': self.categories[-1]['range'],
            'score': risk_score
        }
    
    def calculate_risk_trend(self, risk_scores: np.ndarray,
                            window: int = 7) -> str:
        """
        Calculate risk trend (increasing, decreasing, stable).
        
        Parameters:
        -----------
        risk_scores : np.ndarray
            Array of recent risk scores
        window : int
            Number of recent scores to consider
        
        Returns:
        --------
        str
            Trend description
        """
        if len(risk_scores) < 2:
            return "Insufficient data"
        
        recent = risk_scores[-window:]
        
        # Calculate linear trend
        x = np.arange(len(recent))
        coeffs = np.polyfit(x, recent, 1)
        slope = coeffs[0]
        
        if slope > 2:
            return "Rapidly increasing ⬆️"
        elif slope > 0.5:
            return "Increasing ↗️"
        elif slope < -2:
            return "Rapidly decreasing ⬇️"
        elif slope < -0.5:
            return "Decreasing ↘️"
        else:
            return "Stable ➡️"
    
    def generate_risk_summary(self,
                            flood_prob: float,
                            heatwave_prob: float,
                            rainfall_mm: float,
                            temperature: float,
                            date: Optional[str] = None) -> Dict:
        """
        Generate comprehensive risk summary.
        
        Parameters:
        -----------
        flood_prob : float
            Flood probability (0-1)
        heatwave_prob : float
            Heatwave probability (0-1)
        rainfall_mm : float
            Predicted rainfall in mm
        temperature : float
            Predicted temperature in °C
        date : str, optional
            Date of prediction
        
        Returns:
        --------
        dict
            Comprehensive risk summary
        """
        # Calculate overall risk score
        risk_score = self.calculate_risk_score(
            flood_prob, heatwave_prob, rainfall_mm
        )
        
        # Get category
        category = self.get_risk_category(risk_score)
        
        # Identify primary hazards
        hazards = []
        if heatwave_prob > 0.6:
            hazards.append("Heatwave")
        if flood_prob > 0.5:
            hazards.append("Flooding")
        if rainfall_mm > 50:
            hazards.append("Heavy Rainfall")
        
        if not hazards:
            hazards.append("Low Risk")
        
        # Generate recommendations
        recommendations = self.generate_recommendations(
            flood_prob, heatwave_prob, rainfall_mm, temperature
        )
        
        summary = {
            'date': date or pd.Timestamp.now().strftime('%Y-%m-%d'),
            'overall_risk_score': round(risk_score, 1),
            'risk_category': category['name'],
            'risk_color': category['color'],
            'primary_hazards': hazards,
            'component_scores': {
                'flood_probability': round(flood_prob * 100, 1),
                'heatwave_probability': round(heatwave_prob * 100, 1),
                'rainfall_severity': round(min(rainfall_mm, 100), 1)
            },
            'weather_forecast': {
                'temperature': round(temperature, 1),
                'rainfall': round(rainfall_mm, 1)
            },
            'recommendations': recommendations
        }
        
        return summary
    
    def generate_recommendations(self,
                                flood_prob: float,
                                heatwave_prob: float,
                                rainfall_mm: float,
                                temperature: float) -> list:
        """
        Generate safety recommendations based on conditions.
        
        Parameters:
        -----------
        flood_prob : float
            Flood probability
        heatwave_prob : float
            Heatwave probability
        rainfall_mm : float
            Rainfall amount
        temperature : float
            Temperature
        
        Returns:
        --------
        list
            List of recommendations
        """
        recommendations = []
        
        # Flood recommendations
        if flood_prob > 0.7:
            recommendations.append("⚠️ HIGH FLOOD RISK: Avoid low-lying areas and stay informed about local warnings")
            recommendations.append("🚗 Do not attempt to drive through flooded areas")
        elif flood_prob > 0.5:
            recommendations.append("🌊 Moderate flood risk: Monitor weather updates closely")
        
        # Heatwave recommendations
        if heatwave_prob > 0.7:
            recommendations.append("🌡️ EXTREME HEAT WARNING: Stay hydrated and limit outdoor activities")
            recommendations.append("❄️ Seek air-conditioned environments during peak heat hours")
        elif heatwave_prob > 0.5:
            recommendations.append("☀️ High temperatures expected: Take precautions against heat stress")
        
        # Rainfall recommendations
        if rainfall_mm > 100:
            recommendations.append("🌧️ HEAVY RAINFALL: Expect travel disruptions and potential flash flooding")
        elif rainfall_mm > 50:
            recommendations.append("☔ Significant rainfall expected: Carry rain gear and allow extra travel time")
        
        # Temperature-specific
        if temperature > 35:
            recommendations.append("🥵 Extreme heat: Check on vulnerable family members and neighbors")
        
        # Combined risks
        if flood_prob > 0.6 and rainfall_mm > 75:
            recommendations.append("🚨 CRITICAL: Combined flood and heavy rain threat - take immediate precautions")
        
        if not recommendations:
            recommendations.append("✅ Low climate risk - normal precautions apply")
        
        return recommendations
    
    def calculate_batch_risk(self, df: pd.DataFrame,
                           flood_col: str = 'flood_prob',
                           heatwave_col: str = 'heatwave_prob',
                           rainfall_col: str = 'rainfall_pred') -> pd.DataFrame:
        """
        Calculate risk scores for a batch of predictions.
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame with prediction columns
        flood_col : str
            Flood probability column name
        heatwave_col : str
            Heatwave probability column name
        rainfall_col : str
            Rainfall prediction column name
        
        Returns:
        --------
        pd.DataFrame
            DataFrame with added risk scores and categories
        """
        self.logger.info("Calculating batch risk scores...")
        
        df_risk = df.copy()
        
        # Calculate risk scores
        risk_scores = []
        risk_categories = []
        
        for idx, row in df.iterrows():
            flood_prob = row.get(flood_col, 0)
            heatwave_prob = row.get(heatwave_col, 0)
            rainfall = row.get(rainfall_col, 0)
            
            score = self.calculate_risk_score(flood_prob, heatwave_prob, rainfall)
            category = self.get_risk_category(score)
            
            risk_scores.append(score)
            risk_categories.append(category['name'])
        
        df_risk['climate_risk_score'] = risk_scores
        df_risk['risk_category'] = risk_categories
        
        self.logger.info(f"Calculated risk scores for {len(df_risk)} records")
        
        return df_risk


def main():
    """Example usage."""
    from utils import load_config, setup_logging
    
    config = load_config()
    logger = setup_logging(config)
    
    # Initialize risk index
    risk_calculator = ClimateRiskIndex(config, logger)
    
    # Example scenarios
    scenarios = [
        {
            'name': 'Normal Day',
            'flood_prob': 0.1,
            'heatwave_prob': 0.2,
            'rainfall_mm': 5,
            'temperature': 28
        },
        {
            'name': 'Moderate Risk',
            'flood_prob': 0.5,
            'heatwave_prob': 0.6,
            'rainfall_mm': 30,
            'temperature': 32
        },
        {
            'name': 'High Risk - Flood',
            'flood_prob': 0.8,
            'heatwave_prob': 0.3,
            'rainfall_mm': 85,
            'temperature': 29
        },
        {
            'name': 'High Risk - Heatwave',
            'flood_prob': 0.2,
            'heatwave_prob': 0.9,
            'rainfall_mm': 0,
            'temperature': 37
        },
        {
            'name': 'Extreme Risk - Combined',
            'flood_prob': 0.9,
            'heatwave_prob': 0.8,
            'rainfall_mm': 120,
            'temperature': 36
        }
    ]
    
    print("=" * 80)
    print("CARIBBEAN CLIMATE RISK INDEX - SCENARIO ANALYSIS")
    print("=" * 80)
    
    for scenario in scenarios:
        print(f"\n{'='*80}")
        print(f"Scenario: {scenario['name']}")
        print('-' * 80)
        
        summary = risk_calculator.generate_risk_summary(
            flood_prob=scenario['flood_prob'],
            heatwave_prob=scenario['heatwave_prob'],
            rainfall_mm=scenario['rainfall_mm'],
            temperature=scenario['temperature']
        )
        
        print(f"Overall Risk Score: {summary['overall_risk_score']}/100")
        print(f"Risk Category: {summary['risk_category']}")
        print(f"Primary Hazards: {', '.join(summary['primary_hazards'])}")
        print(f"\nComponent Scores:")
        print(f"  - Flood Probability: {summary['component_scores']['flood_probability']}%")
        print(f"  - Heatwave Probability: {summary['component_scores']['heatwave_probability']}%")
        print(f"  - Rainfall Severity: {summary['component_scores']['rainfall_severity']}")
        print(f"\nRecommendations:")
        for rec in summary['recommendations']:
            print(f"  {rec}")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
