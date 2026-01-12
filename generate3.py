"""
Parte 3: Gera visualizations.py, export.py, requirements.txt e README.md
"""


def write_file(filepath, content):
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"  [OK] {filepath}")


def generate_visualizations_py():
    content = '''"""
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
'''
    write_file('econometric_system/visualizations.py', content)


def generate_export_py():
    content = '''"""
Modulo de exportacao de resultados
"""
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from io import BytesIO
import streamlit as st


def export_to_pdf(results):
    try:
        from visualizations import (
            create_residuals_plot, create_qq_plot,
            create_prediction_plot, create_correlation_heatmap
        )
        
        buffer = BytesIO()
        
        with PdfPages(buffer) as pdf:
            fig = plt.figure(figsize=(11, 8.5))
            fig.text(0.5, 0.95, 'Analise Econometrica', ha='center', fontsize=20, fontweight='bold')
            summary_text = str(results.get('summary', 'Resumo nao disponivel'))
            fig.text(0.1, 0.1, summary_text, fontsize=8, verticalalignment='top', family='monospace')
            pdf.savefig(fig, bbox_inches='tight')
            plt.close()
            
            if 'residuals' in results:
                fig = create_residuals_plot(results)
                pdf.savefig(fig, bbox_inches='tight')
                plt.close()
                
                fig = create_qq_plot(results)
                pdf.savefig(fig, bbox_inches='tight')
                plt.close()
                
                fig = create_prediction_plot(results)
                pdf.savefig(fig, bbox_inches='tight')
                plt.close()
            
            if 'data' in results:
                fig = create_correlation_heatmap(results['data'])
                pdf.savefig(fig, bbox_inches='tight')
                plt.close()
        
        buffer.seek(0)
        return buffer
    
    except Exception as e:
        st.error(f"Erro ao gerar PDF: {str(e)}")
        return None


def export_to_png(results):
    try:
        from visualizations import (
            create_residuals_plot, create_qq_plot,
            create_prediction_plot, create_correlation_heatmap
        )
        
        png_buffers = []
        
        if 'residuals' in results:
            fig = create_residuals_plot(results)
            buffer = BytesIO()
            fig.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
            buffer.seek(0)
            png_buffers.append(buffer)
            plt.close()
            
            fig = create_qq_plot(results)
            buffer = BytesIO()
            fig.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
            buffer.seek(0)
            png_buffers.append(buffer)
            plt.close()
            
            fig = create_prediction_plot(results)
            buffer = BytesIO()
            fig.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
            buffer.seek(0)
            png_buffers.append(buffer)
            plt.close()
        
        if 'data' in results:
            fig = create_correlation_heatmap(results['data'])
            buffer = BytesIO()
            fig.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
            buffer.seek(0)
            png_buffers.append(buffer)
            plt.close()
        
        return png_buffers
    
    except Exception as e:
        st.error(f"Erro ao gerar PNG: {str(e)}")
        return None
'''
    write_file('econometric_system/export.py', content)


def generate_requirements_txt():
    content = "streamlit>=1.28.0\n"
    content += "pandas>=2.0.0\n"
    content += "numpy>=1.24.0\n"
    content += "matplotlib>=3.7.0\n"
    content += "seaborn>=0.12.0\n"
    content += "plotly>=5.14.0\n"
    content += "statsmodels>=0.14.0\n"
    content += "scikit-learn>=1.3.0\n"
    content += "scipy>=1.11.0\n"
    content += "openpyxl>=3.1.0\n"
    write_file('econometric_system/requirements.txt', content)


def generate_readme_md():
    content = "# Sistema de Analise Econometrica\n\n"
    content += "Sistema completo de analise econometrica desenvolvido em Streamlit Python.\n\n"
    content += "## Funcionalidades\n\n"
    content += "- Autenticacao de usuarios\n"
    content += "- Upload de arquivos CSV e XLSX\n"
    content += "- Sugestao automatica de modelos\n"
    content += "- Multiplos modelos econometricos:\n"
    content += "  - Regressao Linear (OLS)\n"
    content += "  - Regressao Logistica (Logit)\n"
    content += "  - Modelo Probit\n"
    content += "  - ARIMA (Series Temporais)\n"
    content += "  - VAR (Vetores Autoregressivos)\n"
    content += "  - Random Forest\n"
    content += "- Visualizacoes completas\n"
    content += "- Exportacao em PDF e PNG\n\n"
    content += "## Instalacao\n\n"
    content += "```bash\n"
    content += "cd econometric_system\n"
    content += "pip install -r requirements.txt\n"
    content += "```\n\n"
    content += "## Executar\n\n"
    content += "```bash\n"
    content += "streamlit run app.py\n"
    content += "```\n\n"
    content += "## Login\n\n"
    content += "- Usuario: admin\n"
    content += "- Senha: admin123\n\n"
    content += "Outros usuarios: demo/demo123, user/user123\n"
    write_file('econometric_system/README.md', content)


def run_generate3():
    print("=== PARTE 3: Gerando visualizations.py, export.py, requirements.txt, README.md ===")
    generate_visualizations_py()
    generate_export_py()
    generate_requirements_txt()
    generate_readme_md()
    print("Parte 3 concluida!")


if __name__ == "__main__":
    run_generate3()
