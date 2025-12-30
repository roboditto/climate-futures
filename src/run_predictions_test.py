import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
import numpy as np
from predictions import generate_predictions
import os

DATA_PATH = 'data/processed/climate_features.csv'

# Load or synthesize data
if os.path.exists(DATA_PATH):
    df = pd.read_csv(DATA_PATH, index_col=0, parse_dates=True)
    print(f'Loaded {len(df)} rows from {DATA_PATH}')
else:
    print(f'{DATA_PATH} not found; synthesizing demo data')
    # create past-only data ending 2025-12-15
    end = pd.to_datetime('2025-12-15')
    start = end - pd.Timedelta(days=60)
    idx = pd.date_range(start=start, end=end, freq='D')
    df = pd.DataFrame(index=idx)
    # include minimal columns used by predictions
    df['precipitation'] = np.random.rand(len(idx)) * 10
    df['temperature'] = 28 + np.random.randn(len(idx))

# Build minimal system with dummy models and risk calculator
class DummyModel:
    def predict(self, X):
        # return zeros of len(X)
        return np.zeros(len(X))

class DummyRisk:
    def calculate_batch_risk(self, df_in):
        df = df_in.copy()
        df['flood_prob'] = 0.05
        df['heatwave_prob'] = 0.05
        df['overall_risk_score'] = 5.0
        return df

system = {
    'models': {
        'heatwave': DummyModel(),
        'rainfall': DummyModel(),
        'flood': DummyModel()
    },
    'risk_calculator': DummyRisk(),
    'logger': None
}

# Run predictions with a 14-day forecast horizon
df_pred = generate_predictions(df, system, forecast_days=14)

print('\nLast 30 rows of predictions:')
print(df_pred[['rainfall_pred','flood_prob','heatwave_prob','overall_risk_score']].tail(30))

# Print the last index value
print('\nLast date in index:', df_pred.index.max())
print('Today:', pd.to_datetime(pd.Timestamp.today()).normalize())
