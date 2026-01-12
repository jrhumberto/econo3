"""
Modulo com os modelos econometricos
"""
import pandas as pd
import numpy as np
from statsmodels.regression.linear_model import OLS
from statsmodels.api import add_constant
from statsmodels.discrete.discrete_model import Logit, Probit
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.api import VAR
from statsmodels.stats.stattools import durbin_watson
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import streamlit as st


def suggest_model(df: pd.DataFrame) -> dict:
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    n_rows = len(df)
    date_cols = [col for col in df.columns if 'data' in col.lower() or 'date' in col.lower()]
    
    suggestion = {
        'numeric_cols': len(numeric_cols),
        'categorical_cols': len(categorical_cols),
        'n_rows': n_rows,
    }
    
    if len(date_cols) > 0 and len(numeric_cols) >= 1:
        suggestion['model'] = 'ARIMA'
        suggestion['reason'] = "Dataset contem coluna de data, ideal para series temporais."
        suggestion['description'] = "ARIMA: Modelo para previsao de series temporais."
    elif len(numeric_cols) >= 1:
        binary_cols = [col for col in numeric_cols if df[col].nunique() == 2]
        if len(binary_cols) > 0:
            suggestion['model'] = 'Regressao Logistica (Logit)'
            suggestion['reason'] = "Dataset contem variavel binaria."
            suggestion['description'] = "Logit: Modelo para variaveis dependentes binarias."
        else:
            suggestion['model'] = 'Regressao Linear (OLS)'
            suggestion['reason'] = "Dataset com variaveis numericas, ideal para regressao."
            suggestion['description'] = "OLS: Modelo classico de regressao linear."
    else:
        suggestion['model'] = 'Random Forest'
        suggestion['reason'] = "Modelo versatil para analise exploratoria."
        suggestion['description'] = "Random Forest: Modelo de machine learning."
    
    return suggestion


def run_ols_regression(df: pd.DataFrame, y_var: str, x_vars: list) -> dict:
    try:
        data = df[[y_var] + x_vars].dropna()
        y = data[y_var]
        X = data[x_vars]
        X = add_constant(X)
        
        model = OLS(y, X)
        results = model.fit()
        
        residuals = results.resid
        fitted = results.fittedvalues
        dw = durbin_watson(residuals)
        
        return {
            'model': results,
            'summary': results.summary(),
            'r_squared': results.rsquared,
            'r_squared_adj': results.rsquared_adj,
            'f_statistic': results.fvalue,
            'durbin_watson': dw,
            'residuals': residuals,
            'fitted': fitted,
            'y_true': y,
            'data': data,
        }
    except Exception as e:
        st.error(f"Erro ao executar modelo OLS: {str(e)}")
        return None


def run_logit_model(df: pd.DataFrame, y_var: str, x_vars: list) -> dict:
    try:
        data = df[[y_var] + x_vars].dropna()
        if data[y_var].nunique() != 2:
            st.error("A variavel dependente deve ser binaria (0/1)!")
            return None
        
        y = data[y_var]
        X = data[x_vars]
        X = add_constant(X)
        
        model = Logit(y, X)
        results = model.fit()
        
        return {
            'model': results,
            'summary': results.summary(),
            'predictions': results.predict(X),
            'data': data
        }
    except Exception as e:
        st.error(f"Erro ao executar modelo Logit: {str(e)}")
        return None


def run_probit_model(df: pd.DataFrame, y_var: str, x_vars: list) -> dict:
    try:
        data = df[[y_var] + x_vars].dropna()
        if data[y_var].nunique() != 2:
            st.error("A variavel dependente deve ser binaria (0/1)!")
            return None
        
        y = data[y_var]
        X = data[x_vars]
        X = add_constant(X)
        
        model = Probit(y, X)
        results = model.fit()
        
        return {
            'model': results,
            'summary': results.summary(),
            'predictions': results.predict(X),
            'data': data
        }
    except Exception as e:
        st.error(f"Erro ao executar modelo Probit: {str(e)}")
        return None


def run_arima_model(df: pd.DataFrame, ts_var: str, order: tuple) -> dict:
    try:
        data = df[ts_var].dropna()
        model = ARIMA(data, order=order)
        results = model.fit()
        forecast = results.forecast(steps=10)
        
        return {
            'model': results,
            'summary': results.summary(),
            'data': data,
            'forecast': forecast,
            'fitted': results.fittedvalues
        }
    except Exception as e:
        st.error(f"Erro ao executar modelo ARIMA: {str(e)}")
        return None


def run_var_model(df: pd.DataFrame, vars_list: list, lags: int) -> dict:
    try:
        data = df[vars_list].dropna()
        model = VAR(data)
        results = model.fit(lags)
        
        return {
            'model': results,
            'summary': results.summary(),
            'data': data
        }
    except Exception as e:
        st.error(f"Erro ao executar modelo VAR: {str(e)}")
        return None


def run_random_forest(df: pd.DataFrame, y_var: str, x_vars: list, n_estimators: int, test_size: float) -> dict:
    try:
        data = df[[y_var] + x_vars].dropna()
        y = data[y_var]
        X = data[x_vars]
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        
        model = RandomForestRegressor(n_estimators=n_estimators, random_state=42)
        model.fit(X_train, y_train)
        
        y_pred_test = model.predict(X_test)
        
        r2 = r2_score(y_test, y_pred_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
        mae = mean_absolute_error(y_test, y_pred_test)
        
        return {
            'model': model,
            'r2': r2,
            'rmse': rmse,
            'mae': mae,
            'feature_importance': dict(zip(x_vars, model.feature_importances_)),
            'y_test': y_test,
            'y_pred_test': y_pred_test,
        }
    except Exception as e:
        st.error(f"Erro ao executar Random Forest: {str(e)}")
        return None
