"""
Modulo de visualizacoes
"""
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import scipy.stats as stats

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


def create_correlation_heatmap(data):
    fig, ax = plt.subplots(figsize=(10, 8))
    corr = data.corr()
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', center=0, ax=ax)
    ax.set_title('Matriz de Correlacao', fontsize=16, fontweight='bold')
    plt.tight_layout()
    return fig


def create_residuals_plot(results):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    residuals = results['residuals']
    fitted = results['fitted']
    
    axes[0, 0].scatter(fitted, residuals, alpha=0.5)
    axes[0, 0].axhline(y=0, color='r', linestyle='--')
    axes[0, 0].set_xlabel('Valores Ajustados')
    axes[0, 0].set_ylabel('Residuos')
    axes[0, 0].set_title('Residuos vs Valores Ajustados')
    
    axes[0, 1].hist(residuals, bins=30, edgecolor='black', alpha=0.7)
    axes[0, 1].set_xlabel('Residuos')
    axes[0, 1].set_ylabel('Frequencia')
    axes[0, 1].set_title('Distribuicao dos Residuos')
    
    standardized_residuals = residuals / np.std(residuals)
    axes[1, 0].scatter(fitted, np.sqrt(np.abs(standardized_residuals)), alpha=0.5)
    axes[1, 0].set_xlabel('Valores Ajustados')
    axes[1, 0].set_ylabel('Raiz |Residuos Padronizados|')
    axes[1, 0].set_title('Scale-Location')
    
    axes[1, 1].plot(residuals)
    axes[1, 1].axhline(y=0, color='r', linestyle='--')
    axes[1, 1].set_xlabel('Observacao')
    axes[1, 1].set_ylabel('Residuos')
    axes[1, 1].set_title('Residuos vs Ordem')
    
    plt.tight_layout()
    return fig


def create_qq_plot(results):
    fig, ax = plt.subplots(figsize=(8, 6))
    residuals = results['residuals']
    stats.probplot(residuals, dist="norm", plot=ax)
    ax.set_title('Q-Q Plot', fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def create_prediction_plot(results):
    fig, ax = plt.subplots(figsize=(10, 6))
    
    y_true = results['y_true']
    y_pred = results['fitted']
    
    ax.scatter(y_true, y_pred, alpha=0.5)
    
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Predicao Perfeita')
    
    ax.set_xlabel('Valores Reais')
    ax.set_ylabel('Valores Preditos')
    ax.set_title('Valores Reais vs Preditos', fontsize=16, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def create_feature_importance_plot(results):
    fig, ax = plt.subplots(figsize=(10, 6))
    
    features = list(results['feature_importance'].keys())
    importance = list(results['feature_importance'].values())
    
    sorted_idx = np.argsort(importance)
    features = [features[i] for i in sorted_idx]
    importance = [importance[i] for i in sorted_idx]
    
    ax.barh(features, importance)
    ax.set_xlabel('Importancia')
    ax.set_title('Importancia das Features', fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    return fig


def create_time_series_plot(results):
    fig, ax = plt.subplots(figsize=(12, 6))
    
    data = results['data']
    fitted = results['fitted']
    forecast = results['forecast']
    
    ax.plot(data.index, data.values, label='Dados Reais', linewidth=2)
    ax.plot(fitted.index, fitted.values, label='Valores Ajustados', linewidth=2, linestyle='--')
    
    forecast_index = range(len(data), len(data) + len(forecast))
    ax.plot(forecast_index, forecast.values, label='Previsao', linewidth=2, linestyle=':')
    
    ax.set_xlabel('Tempo')
    ax.set_ylabel('Valor')
    ax.set_title('Serie Temporal - ARIMA', fontsize=16, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig
