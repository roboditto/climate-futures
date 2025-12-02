"""
Alert Generation System
Generates automated climate warnings and reports.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import logging
import json
import os


class ClimateAlertSystem:
    """
    Generates and manages climate alerts.
    """
    
    def __init__(self, config: Dict, logger: Optional[logging.Logger] = None):
        """
        Initialize alert system.
        
        Parameters:
        -----------
        config : dict
            Configuration dictionary
        logger : logging.Logger, optional
            Logger instance
        """
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        self.alert_config = config.get('alerts', {})
        self.triggers = self.alert_config.get('triggers', {})
        self.output_formats = self.alert_config.get('output_formats', ['console'])
        
        # Create reports directory
        reports_path = config.get('storage', {}).get('reports_path', 'reports/')
        os.makedirs(reports_path, exist_ok=True)
    
    def check_heatwave_alert(self, heatwave_prob: float) -> Optional[Dict]:
        """
        Check if heatwave alert should be triggered.
        
        Parameters:
        -----------
        heatwave_prob : float
            Heatwave probability (0-1)
        
        Returns:
        --------
        dict or None
            Alert details if triggered, None otherwise
        """
        threshold = self.triggers.get('heatwave', {}).get('threshold', 0.7)
        
        if heatwave_prob >= threshold:
            return {
                'type': 'heatwave',
                'severity': 'high' if heatwave_prob > 0.85 else 'moderate',
                'probability': heatwave_prob,
                'message': self.triggers['heatwave']['message'],
                'timestamp': datetime.now().isoformat()
            }
        
        return None
    
    def check_flood_alert(self, flood_prob: float) -> Optional[Dict]:
        """
        Check if flood alert should be triggered.
        
        Parameters:
        -----------
        flood_prob : float
            Flood probability (0-1)
        
        Returns:
        --------
        dict or None
            Alert details if triggered, None otherwise
        """
        threshold = self.triggers.get('flood', {}).get('threshold', 0.6)
        
        if flood_prob >= threshold:
            return {
                'type': 'flood',
                'severity': 'high' if flood_prob > 0.8 else 'moderate',
                'probability': flood_prob,
                'message': self.triggers['flood']['message'],
                'timestamp': datetime.now().isoformat()
            }
        
        return None
    
    def check_rainfall_alert(self, rainfall_mm: float) -> Optional[Dict]:
        """
        Check if heavy rainfall alert should be triggered.
        
        Parameters:
        -----------
        rainfall_mm : float
            Predicted rainfall in mm
        
        Returns:
        --------
        dict or None
            Alert details if triggered, None otherwise
        """
        threshold = self.triggers.get('rainfall', {}).get('threshold', 100)
        
        if rainfall_mm >= threshold:
            return {
                'type': 'rainfall',
                'severity': 'extreme' if rainfall_mm > 150 else 'high',
                'amount': rainfall_mm,
                'message': self.triggers['rainfall']['message'],
                'timestamp': datetime.now().isoformat()
            }
        
        return None
    
    def check_combined_risk_alert(self, risk_score: float) -> Optional[Dict]:
        """
        Check if combined climate risk alert should be triggered.
        
        Parameters:
        -----------
        risk_score : float
            Climate risk score (0-100)
        
        Returns:
        --------
        dict or None
            Alert details if triggered, None otherwise
        """
        threshold = self.triggers.get('combined_risk', {}).get('threshold', 70)
        
        if risk_score >= threshold:
            return {
                'type': 'combined_risk',
                'severity': 'extreme' if risk_score > 85 else 'high',
                'risk_score': risk_score,
                'message': self.triggers['combined_risk']['message'],
                'timestamp': datetime.now().isoformat()
            }
        
        return None
    
    def generate_alerts(self,
                       flood_prob: float,
                       heatwave_prob: float,
                       rainfall_mm: float,
                       risk_score: float) -> List[Dict]:
        """
        Generate all applicable alerts.
        
        Parameters:
        -----------
        flood_prob : float
            Flood probability
        heatwave_prob : float
            Heatwave probability
        rainfall_mm : float
            Rainfall amount
        risk_score : float
            Overall risk score
        
        Returns:
        --------
        list
            List of active alerts
        """
        alerts = []
        
        # Check each alert type
        heatwave_alert = self.check_heatwave_alert(heatwave_prob)
        if heatwave_alert:
            alerts.append(heatwave_alert)
        
        flood_alert = self.check_flood_alert(flood_prob)
        if flood_alert:
            alerts.append(flood_alert)
        
        rainfall_alert = self.check_rainfall_alert(rainfall_mm)
        if rainfall_alert:
            alerts.append(rainfall_alert)
        
        combined_alert = self.check_combined_risk_alert(risk_score)
        if combined_alert:
            alerts.append(combined_alert)
        
        self.logger.info(f"Generated {len(alerts)} alerts")
        
        return alerts
    
    def format_alert_console(self, alerts: List[Dict]) -> str:
        """
        Format alerts for console output.
        
        Parameters:
        -----------
        alerts : list
            List of alerts
        
        Returns:
        --------
        str
            Formatted alert text
        """
        if not alerts:
            return "✅ No active climate alerts\n"
        
        output = []
        output.append("\n" + "=" * 80)
        output.append("🚨 CLIMATE ALERT SYSTEM - ACTIVE WARNINGS")
        output.append("=" * 80)
        output.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        output.append("=" * 80 + "\n")
        
        for i, alert in enumerate(alerts, 1):
            output.append(f"\n[ALERT {i}] {alert['type'].upper()} - {alert['severity'].upper()}")
            output.append("-" * 80)
            output.append(f"📢 {alert['message']}")
            
            if 'probability' in alert:
                output.append(f"   Probability: {alert['probability']*100:.1f}%")
            if 'amount' in alert:
                output.append(f"   Amount: {alert['amount']:.1f} mm")
            if 'risk_score' in alert:
                output.append(f"   Risk Score: {alert['risk_score']:.1f}/100")
            
            output.append(f"   Issued: {alert['timestamp']}")
            output.append("-" * 80)
        
        output.append("\n" + "=" * 80)
        output.append(f"Total Active Alerts: {len(alerts)}")
        output.append("=" * 80 + "\n")
        
        return "\n".join(output)
    
    def save_alert_json(self, alerts: List[Dict], filepath: str) -> None:
        """
        Save alerts to JSON file.
        
        Parameters:
        -----------
        alerts : list
            List of alerts
        filepath : str
            Output file path
        """
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump({
                'generated_at': datetime.now().isoformat(),
                'alert_count': len(alerts),
                'alerts': alerts
            }, f, indent=2)
        
        self.logger.info(f"Alerts saved to {filepath}")
    
    def generate_daily_report(self,
                            risk_summary: Dict,
                            alerts: List[Dict],
                            save_path: Optional[str] = None) -> str:
        """
        Generate comprehensive daily report.
        
        Parameters:
        -----------
        risk_summary : dict
            Risk summary from ClimateRiskIndex
        alerts : list
            Active alerts
        save_path : str, optional
            Path to save report
        
        Returns:
        --------
        str
            Formatted report
        """
        report = []
        
        # Header
        report.append("=" * 80)
        report.append("CARIBBEAN CLIMATE IMPACT SYSTEM")
        report.append("DAILY RISK ASSESSMENT REPORT")
        report.append("=" * 80)
        report.append(f"Date: {risk_summary.get('date', datetime.now().strftime('%Y-%m-%d'))}")
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("=" * 80)
        
        # Overall Risk
        report.append("\n📊 OVERALL CLIMATE RISK")
        report.append("-" * 80)
        report.append(f"Risk Score: {risk_summary['overall_risk_score']}/100")
        report.append(f"Risk Category: {risk_summary['risk_category']}")
        report.append(f"Primary Hazards: {', '.join(risk_summary['primary_hazards'])}")
        
        # Component Scores
        report.append("\n📈 COMPONENT ANALYSIS")
        report.append("-" * 80)
        components = risk_summary['component_scores']
        report.append(f"Flood Probability: {components['flood_probability']}%")
        report.append(f"Heatwave Probability: {components['heatwave_probability']}%")
        report.append(f"Rainfall Severity: {components['rainfall_severity']}")
        
        # Weather Forecast
        report.append("\n🌤️  WEATHER FORECAST")
        report.append("-" * 80)
        forecast = risk_summary['weather_forecast']
        report.append(f"Temperature: {forecast['temperature']}°C")
        report.append(f"Rainfall: {forecast['rainfall']} mm")
        
        # Active Alerts
        report.append("\n🚨 ACTIVE ALERTS")
        report.append("-" * 80)
        if alerts:
            for i, alert in enumerate(alerts, 1):
                report.append(f"{i}. [{alert['type'].upper()}] {alert['message']}")
        else:
            report.append("No active alerts")
        
        # Recommendations
        report.append("\n💡 RECOMMENDATIONS")
        report.append("-" * 80)
        for i, rec in enumerate(risk_summary['recommendations'], 1):
            report.append(f"{i}. {rec}")
        
        # Footer
        report.append("\n" + "=" * 80)
        report.append("For emergency information, contact local authorities.")
        report.append("Stay informed and stay safe!")
        report.append("=" * 80)
        
        report_text = "\n".join(report)
        
        # Save if path provided
        if save_path:
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write(report_text)
            self.logger.info(f"Daily report saved to {save_path}")
        
        return report_text
    
    def process_alerts(self,
                      flood_prob: float,
                      heatwave_prob: float,
                      rainfall_mm: float,
                      risk_score: float) -> None:
        """
        Generate and output alerts in configured formats.
        
        Parameters:
        -----------
        flood_prob : float
            Flood probability
        heatwave_prob : float
            Heatwave probability
        rainfall_mm : float
            Rainfall amount
        risk_score : float
            Risk score
        """
        # Generate alerts
        alerts = self.generate_alerts(flood_prob, heatwave_prob, rainfall_mm, risk_score)
        
        # Output in configured formats
        if 'console' in self.output_formats:
            print(self.format_alert_console(alerts))
        
        if 'json' in self.output_formats:
            reports_path = self.config.get('storage', {}).get('reports_path', 'reports/')
            json_path = os.path.join(reports_path, f'alerts_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
            self.save_alert_json(alerts, json_path)


def main():
    """Example usage."""
    from utils import load_config, setup_logging
    from risk_model import ClimateRiskIndex
    
    config = load_config()
    logger = setup_logging(config)
    
    # Initialize systems
    alert_system = ClimateAlertSystem(config, logger)
    risk_calculator = ClimateRiskIndex(config, logger)
    
    # Test scenarios
    scenarios = [
        {
            'name': 'Normal Conditions',
            'flood_prob': 0.2,
            'heatwave_prob': 0.3,
            'rainfall_mm': 10,
            'temperature': 28
        },
        {
            'name': 'High Flood Risk',
            'flood_prob': 0.85,
            'heatwave_prob': 0.2,
            'rainfall_mm': 120,
            'temperature': 27
        },
        {
            'name': 'Extreme Heatwave',
            'flood_prob': 0.1,
            'heatwave_prob': 0.92,
            'rainfall_mm': 0,
            'temperature': 38
        },
        {
            'name': 'Multiple Hazards',
            'flood_prob': 0.75,
            'heatwave_prob': 0.80,
            'rainfall_mm': 95,
            'temperature': 35
        }
    ]
    
    for scenario in scenarios:
        print(f"\n{'='*80}")
        print(f"SCENARIO: {scenario['name']}")
        print('='*80)
        
        # Generate risk summary
        risk_summary = risk_calculator.generate_risk_summary(
            flood_prob=scenario['flood_prob'],
            heatwave_prob=scenario['heatwave_prob'],
            rainfall_mm=scenario['rainfall_mm'],
            temperature=scenario['temperature']
        )
        
        # Generate alerts
        alerts = alert_system.generate_alerts(
            flood_prob=scenario['flood_prob'],
            heatwave_prob=scenario['heatwave_prob'],
            rainfall_mm=scenario['rainfall_mm'],
            risk_score=risk_summary['overall_risk_score']
        )
        
        # Display alerts
        print(alert_system.format_alert_console(alerts))
        
        # Generate daily report
        report_path = f"reports/daily_report_{scenario['name'].replace(' ', '_')}.txt"
        report = alert_system.generate_daily_report(risk_summary, alerts, save_path=report_path)
        
        print(f"\n✅ Report saved to: {report_path}")


if __name__ == "__main__":
    main()
