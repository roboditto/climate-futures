"""
Small test runner to reproduce prediction generation errors.
This script imports helpers from `app.py`, loads the data and system,
then calls `generate_predictions` and prints any exceptions/outputs.
"""

import traceback
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent))

from app import load_system, load_data, generate_predictions

if __name__ == '__main__':
    try:
        print('Loading system...')
        system = load_system()
        print('System loaded.')

        print('Loading data...')
        df = load_data()
        print('Data loaded. Shape:', None if df is None else df.shape)

        if df is None:
            raise SystemExit('No data available to test predictions.')

        print('Generating predictions...')
        df_pred = generate_predictions(df, system)
        print('Predictions generated. Columns added:')
        cols = [c for c in ['heatwave_prob','rainfall_pred','flood_prob','overall_risk_score'] if c in df_pred.columns]
        print(cols)
        print('\nLast row:')
        print(df_pred.tail(1).T)

    except Exception as e:
        print('\nERROR during prediction test:')
        traceback.print_exc()