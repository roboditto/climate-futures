import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Dict, Optional, List
import logging


class ClimateVisualizer:
    """
    Handles all climate visualization needs.
    """
    
    def __init__(self, config: Dict, logger: Optional[logging.Logger] = None):
        """
        Initialize visualizer.
        
        Parameters:
        -----------
        config : dict
            Configuration dictionary
        logger : logging.Logger, optional
            Logger instance
        """
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        self.viz_config = config.get('visualization', {})
        
        # Set style
        plt.style.use(self.viz_config.get('style', 'seaborn-v0_8-darkgrid'))
        sns.set_palette(self.viz_config.get('color_palette', 'viridis'))
    
    def plot_time_series(self, df: pd.DataFrame,
                        columns: List[str],
                        title: str = "Climate Time Series",
                        save_path: Optional[str] = None) -> None:
        """
        Plot multiple time series.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Data with datetime index
        columns : list
            Column names to plot
        title : str
            Plot title
        save_path : str, optional
            Path to save figure
        """
        fig = make_subplots(
            rows=len(columns), cols=1,
            subplot_titles=columns,
            shared_xaxes=True,
            vertical_spacing=0.05
        )
        
        for i, col in enumerate(columns, 1):
            if col in df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=df.index,
                        y=df[col],
                        name=col,
                        mode='lines',
                        line=dict(width=2)
                    ),
                    row=i, col=1
                )
        
        fig.update_layout(
            title=title,
            height=300 * len(columns),
            showlegend=False,
            template='plotly_white'
        )
        
        if save_path:
            fig.write_html(save_path)
            self.logger.info(f"Time series plot saved to {save_path}")
        
        fig.show()
    
    def plot_risk_dashboard(self, df: pd.DataFrame,
                           save_path: Optional[str] = None) -> None:
        """
        Create comprehensive risk dashboard.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Data with risk scores and predictions
        save_path : str, optional
            Path to save figure
        """
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Climate Risk Score Over Time',
                'Risk Category Distribution',
                'Component Scores',
                'Heatwave vs Flood Probability'
            ),
            specs=[
                [{"type": "scatter"}, {"type": "pie"}],
                [{"type": "bar"}, {"type": "scatter"}]
            ]
        )
        
        # 1. Risk score timeline
        if 'climate_risk_score' in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df['climate_risk_score'],
                    name='Risk Score',
                    fill='tozeroy',
                    line=dict(color='red', width=2)
                ),
                row=1, col=1
            )
        
        # 2. Risk category pie chart
        if 'risk_category' in df.columns:
            category_counts = df['risk_category'].value_counts()
            fig.add_trace(
                go.Pie(
                    labels=category_counts.index,
                    values=category_counts.values,
                    name='Categories'
                ),
                row=1, col=2
            )
        
        # 3. Component scores (latest)
        if all(col in df.columns for col in ['flood_prob', 'heatwave_prob']):
            latest = df.iloc[-1]
            components = {
                'Flood': latest.get('flood_prob', 0) * 100,
                'Heatwave': latest.get('heatwave_prob', 0) * 100,
                'Rainfall': min(latest.get('rainfall_pred', 0), 100)
            }
            
            fig.add_trace(
                go.Bar(
                    x=list(components.keys()),
                    y=list(components.values()),
                    name='Component Scores',
                    marker_color=['#3498db', '#e74c3c', '#2ecc71']
                ),
                row=2, col=1
            )
        
        # 4. Scatter: Heatwave vs Flood
        if 'heatwave_prob' in df.columns and 'flood_prob' in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df['heatwave_prob'] * 100,
                    y=df['flood_prob'] * 100,
                    mode='markers',
                    name='Days',
                    marker=dict(
                        size=8,
                        color=df.get('climate_risk_score', range(len(df))),
                        colorscale='Reds',
                        showscale=True,
                        colorbar=dict(title="Risk Score")
                    )
                ),
                row=2, col=2
            )
        
        fig.update_layout(
            title="Caribbean Climate Risk Dashboard",
            height=800,
            showlegend=False,
            template='plotly_white'
        )
        
        # Update axes
        fig.update_yaxes(title_text="Risk Score (0-100)", row=1, col=1)
        fig.update_yaxes(title_text="Score (%)", row=2, col=1)
        fig.update_xaxes(title_text="Heatwave Probability (%)", row=2, col=2)
        fig.update_yaxes(title_text="Flood Probability (%)", row=2, col=2)
        
        if save_path:
            fig.write_html(save_path)
            self.logger.info(f"Risk dashboard saved to {save_path}")
        
        fig.show()
    
    def plot_heatwave_forecast(self, df: pd.DataFrame,
                              days_ahead: int = 7,
                              save_path: Optional[str] = None) -> None:
        """
        Plot heatwave forecast.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Data with heatwave predictions
        days_ahead : int
            Number of days to show
        save_path : str, optional
            Path to save figure
        """
        recent = df.tail(days_ahead)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Temperature
        ax.plot(recent.index, recent.get('temperature', []), 
                'o-', color='red', linewidth=2, markersize=8, label='Temperature')
        
        # Heatwave probability (secondary axis)
        ax2 = ax.twinx()
        if 'heatwave_prob' in recent.columns:
            ax2.fill_between(recent.index, 0, recent['heatwave_prob'] * 100,
                            alpha=0.3, color='orange', label='Heatwave Probability')
        
        # Threshold line
        threshold = self.config['models']['heatwave']['threshold_temperature']
        ax.axhline(y=threshold, color='darkred', linestyle='--', 
                   linewidth=2, label=f'Heatwave Threshold ({threshold}°C)')
        
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Temperature (°C)', fontsize=12, color='red')
        ax2.set_ylabel('Heatwave Probability (%)', fontsize=12, color='orange')
        ax.tick_params(axis='y', labelcolor='red')
        ax2.tick_params(axis='y', labelcolor='orange')
        
        ax.set_title(f'{days_ahead}-Day Heatwave Forecast', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper left')
        ax2.legend(loc='upper right')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.viz_config.get('figure_dpi', 300), 
                       bbox_inches='tight')
            self.logger.info(f"Heatwave forecast saved to {save_path}")
            plt.close()
        else:
            plt.show()
    
    def plot_rainfall_forecast(self, df: pd.DataFrame,
                              days_ahead: int = 7,
                              save_path: Optional[str] = None) -> None:
        """
        Plot rainfall forecast.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Data with rainfall predictions
        days_ahead : int
            Number of days to show
        save_path : str, optional
            Path to save figure
        """
        recent = df.tail(days_ahead)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Get extreme threshold from config
        extreme_threshold = self.config['models']['rainfall']['extreme_threshold']
        
        # Rainfall bars
        rainfall_col = 'rainfall_pred' if 'rainfall_pred' in recent.columns else 'precipitation'
        if rainfall_col in recent.columns:
            bars = ax.bar(recent.index, recent[rainfall_col], 
                         color='skyblue', edgecolor='navy', linewidth=1.5)
            
            # Color bars by intensity
            for i, (idx, row) in enumerate(recent.iterrows()):
                rainfall = row[rainfall_col]
                if rainfall > extreme_threshold:
                    bars[i].set_color('darkred')
                    bars[i].set_label('Extreme' if i == 0 else '')
                elif rainfall > extreme_threshold * 0.5:
                    bars[i].set_color('orange')
        
        # Threshold line
        ax.axhline(y=extreme_threshold, color='red', linestyle='--',
                  linewidth=2, label=f'Extreme Threshold ({extreme_threshold} mm)')
        
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Rainfall (mm)', fontsize=12)
        ax.set_title(f'{days_ahead}-Day Rainfall Forecast', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        ax.legend()
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.viz_config.get('figure_dpi', 300),
                       bbox_inches='tight')
            self.logger.info(f"Rainfall forecast saved to {save_path}")
            plt.close()
        else:
            plt.show()
    
    def plot_risk_gauge(self, risk_score: float,
                       save_path: Optional[str] = None) -> None:
        """
        Create gauge chart for current risk level.
        
        Parameters:
        -----------
        risk_score : float
            Current risk score (0-100)
        save_path : str, optional
            Path to save figure
        """
        from risk_model import ClimateRiskIndex
        
        risk_calc = ClimateRiskIndex(self.config)
        category = risk_calc.get_risk_category(risk_score)
        
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=risk_score,
            title={'text': f"Climate Risk Level<br><span style='font-size:0.8em'>{category['name']}</span>"},
            delta={'reference': 50, 'increasing': {'color': "red"}},
            gauge={
                'axis': {'range': [None, 100], 'tickwidth': 1},
                'bar': {'color': category['color']},
                'steps': [
                    {'range': [0, 30], 'color': "#d5f4e6"},
                    {'range': [30, 50], 'color': "#fff9c4"},
                    {'range': [50, 70], 'color': "#ffccbc"},
                    {'range': [70, 85], 'color': "#ffab91"},
                    {'range': [85, 100], 'color': "#ef5350"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 85
                }
            }
        ))
        
        fig.update_layout(
            height=400,
            template='plotly_white'
        )
        
        if save_path:
            fig.write_html(save_path)
            self.logger.info(f"Risk gauge saved to {save_path}")
        
        fig.show()
    
    def create_climate_heatmap(self, df: pd.DataFrame,
                              value_col: str = 'climate_risk_score',
                              save_path: Optional[str] = None) -> None:
        """
        Create calendar heatmap of climate risk.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Data with datetime index
        value_col : str
            Column to visualize
        save_path : str, optional
            Path to save figure
        """
        # Prepare data
        df_cal = df.copy()
        dt_index = pd.to_datetime(df_cal.index)
        df_cal['year'] = dt_index.year
        df_cal['month'] = dt_index.month
        df_cal['day'] = dt_index.day
        
        # Pivot for heatmap
        pivot = df_cal.pivot_table(
            values=value_col,
            index='day',
            columns='month',
            aggfunc='mean'
        )
        
        # Plot
        fig, ax = plt.subplots(figsize=(14, 8))
        sns.heatmap(pivot, cmap='RdYlGn_r', center=50, vmin=0, vmax=100,
                   cbar_kws={'label': value_col}, ax=ax)
        
        ax.set_title(f'{value_col.replace("_", " ").title()} Calendar Heatmap',
                    fontsize=14, fontweight='bold')
        ax.set_xlabel('Month', fontsize=12)
        ax.set_ylabel('Day of Month', fontsize=12)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.viz_config.get('figure_dpi', 300),
                       bbox_inches='tight')
            self.logger.info(f"Calendar heatmap saved to {save_path}")
            plt.close()
        else:
            plt.show()


def main():
    """Example usage."""
    from utils import load_config, setup_logging
    from data_preprocessing import ClimateDataLoader
    import os
    
    config = load_config()
    logger = setup_logging(config)
    
    # Create results directory
    os.makedirs('results', exist_ok=True)
    
    # Load data
    loader = ClimateDataLoader(config, logger)
    
    try:
        df = loader.load_processed_data('climate_features.csv')
    except:
        print("Generating sample data...")
        df = loader.generate_sample_data(days=90)
        df = loader.clean_data(df)
        df = loader.calculate_derived_metrics(df)
    
    # Add some predictions (simulated)
    np.random.seed(42)
    df['heatwave_prob'] = np.random.beta(2, 5, len(df))
    df['flood_prob'] = np.random.beta(2, 8, len(df))
    df['rainfall_pred'] = df.get('precipitation', np.random.gamma(2, 10, len(df)))
    df['climate_risk_score'] = (
        df['heatwave_prob'] * 30 +
        df['flood_prob'] * 40 +
        np.clip(df['rainfall_pred'] / 100, 0, 1) * 30
    )
    df['risk_category'] = pd.cut(
        df['climate_risk_score'],
        bins=[0, 30, 50, 70, 85, 100],
        labels=['Minimal', 'Low', 'Moderate', 'High', 'Extreme']
    )
    
    # Initialize visualizer
    viz = ClimateVisualizer(config, logger)
    
    print("Creating visualizations...")
    
    # 1. Risk dashboard
    viz.plot_risk_dashboard(df, save_path='results/risk_dashboard.html')
    
    # 2. Heatwave forecast
    viz.plot_heatwave_forecast(df, days_ahead=14, save_path='results/heatwave_forecast.png')
    
    # 3. Rainfall forecast
    viz.plot_rainfall_forecast(df, days_ahead=14, save_path='results/rainfall_forecast.png')
    
    # 4. Risk gauge (current)
    current_risk = df['climate_risk_score'].iloc[-1]
    viz.plot_risk_gauge(current_risk, save_path='results/risk_gauge.html')
    
    print("\nVisualizations created successfully!")
    print("Check the 'results' folder for output files.")


if __name__ == "__main__":
    main()
