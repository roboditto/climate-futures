"""
Prediction helpers for dashboard.
Provides a safe `generate_predictions` that aligns features to trained models
and logs full tracebacks when prediction fails.
"""
from typing import Dict
import pandas as pd
import numpy as np
import traceback
import streamlit as st


def generate_predictions(df: pd.DataFrame, system: Dict) -> pd.DataFrame:
    df_pred = df.copy()

    def _build_X_for_model(model_wrapper, df_src: pd.DataFrame) -> pd.DataFrame:
        model_obj = getattr(model_wrapper, 'model', None)
        if model_obj is None:
            return df_src.copy()

        feature_names = None
        if hasattr(model_obj, 'feature_names_in_'):
            try:
                feature_names = list(model_obj.feature_names_in_)
            except Exception:
                feature_names = None

        if feature_names is None:
            try:
                booster = getattr(model_obj, 'get_booster', None)
                if booster is not None:
                    b = booster()
                    feature_names = list(b.feature_names)
            except Exception:
                feature_names = None

        if feature_names:
            X = df_src.reindex(columns=feature_names).copy()
        else:
            X = df_src.select_dtypes(include=[np.number]).copy()

        X = X.fillna(X.mean())
        X = X.replace([np.inf, -np.inf], np.nan).fillna(X.mean())
        return X

    # Heatwave
    try:
        heatwave_model = system['models']['heatwave']
        X_hw = _build_X_for_model(heatwave_model, df_pred)
        df_pred['heatwave_prob'] = heatwave_model.predict(X_hw)
    except Exception as e:
        tb = traceback.format_exc()
        logger = system.get('logger') if isinstance(system, dict) else None
        if logger is not None:
            logger.exception('Heatwave prediction failed')
        st.warning(f"⚠️ Could not generate heatwave predictions: {str(e)}. See logs for details.")
        if logger is None:
            st.text_area('Heatwave traceback', tb, height=200)
        df_pred['heatwave_prob'] = 0.3

    # Rainfall
    try:
        rainfall_model = system['models']['rainfall']
        X_rf = _build_X_for_model(rainfall_model, df_pred)
        rainfall_preds = rainfall_model.predict(X_rf)
        df_pred['rainfall_pred'] = np.nan
        df_pred.loc[df_pred.index[:len(rainfall_preds)], 'rainfall_pred'] = rainfall_preds
        df_pred['rainfall_pred'] = df_pred['rainfall_pred'].bfill()
    except Exception as e:
        tb = traceback.format_exc()
        logger = system.get('logger') if isinstance(system, dict) else None
        if logger is not None:
            logger.exception('Rainfall prediction failed')
        st.warning(f"⚠️ Could not generate rainfall predictions: {str(e)}. See logs for details.")
        if logger is None:
            st.text_area('Rainfall traceback', tb, height=200)
        df_pred['rainfall_pred'] = df_pred['precipitation'].rolling(window=3, min_periods=1).mean()

    # Flood
    try:
        flood_model = system['models']['flood']
        X_fl = _build_X_for_model(flood_model, df_pred)
        df_pred['flood_prob'] = flood_model.predict(X_fl)
    except Exception as e:
        tb = traceback.format_exc()
        logger = system.get('logger') if isinstance(system, dict) else None
        if logger is not None:
            logger.exception('Flood prediction failed')
        st.warning(f"⚠️ Could not generate flood predictions: {str(e)}. See logs for details.")
        if logger is None:
            st.text_area('Flood traceback', tb, height=200)
        df_pred['flood_prob'] = 0.2

    # Risk calculation
    try:
        risk_calculator = system['risk_calculator']
        df_pred = risk_calculator.calculate_batch_risk(df_pred)
        if 'climate_risk_score' in df_pred.columns:
            df_pred['overall_risk_score'] = df_pred['climate_risk_score']
    except Exception as e:
        tb = traceback.format_exc()
        logger = system.get('logger') if isinstance(system, dict) else None
        if logger is not None:
            logger.exception('Risk calculation failed')
        st.warning(f"⚠️ Could not calculate risk scores: {str(e)}. See logs for details.")
        if logger is None:
            st.text_area('Risk calculation traceback', tb, height=200)
        rainfall_col = df_pred.get('rainfall_pred', pd.Series(0, index=df_pred.index))
        if isinstance(rainfall_col, (int, float)):
            rainfall_normalized = 0
        else:
            rainfall_normalized = (rainfall_col / 100.0).clip(0, 1) * 100
        df_pred['overall_risk_score'] = (
            df_pred.get('flood_prob', 0) * 100 * 0.4 +
            df_pred.get('heatwave_prob', 0) * 100 * 0.4 +
            rainfall_normalized * 0.2
        )

    # Ensure no NaN in key columns
    df_pred['heatwave_prob'] = df_pred['heatwave_prob'].fillna(0.3)
    df_pred['flood_prob'] = df_pred['flood_prob'].fillna(0.2)
    df_pred['rainfall_pred'] = df_pred['rainfall_pred'].fillna(df_pred['precipitation'].rolling(3, min_periods=1).mean())
    df_pred['overall_risk_score'] = df_pred['overall_risk_score'].fillna(df_pred['overall_risk_score'].mean())

    return df_pred
