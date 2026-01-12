#!/usr/bin/env python3
"""
Script para gerar automaticamente toda a estrutura do projeto
Sistema de Análise Econométrica em Streamlit
"""
import os
from pathlib import Path

def create_directory_structure():
    """Cria a estrutura de diretórios do projeto"""
    directories = [
        'econometric_system',
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✓ Diretório criado: {directory}")

def write_file(filepath, content):
    """Escreve conteúdo em um arquivo"""
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"✓ Arquivo criado: {filepath}")

def create_files():
    """Cria todos os arquivos do projeto com seus conteúdos"""
    
    files_content = {
        'econometric_system/app.py': '''import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from io import BytesIO
import base64
from datetime import datetime

# Importações para modelos econométricos
from statsmodels.regression.linear_model import OLS
from statsmodels.api import add_constant
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.discrete.discrete_model import Logit, Probit
from statsmodels.stats.diagnostic import het_breuschpagan, het_white
from statsmodels.stats.stattools import durbin_watson
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import scipy.stats as stats

# Importar módulos personalizados
from auth import authenticate_user, check_authentication
from models import (
    suggest_model,
    run_ols_regression,
    run_logit_model,
    run_probit_model,
    run_arima_model,
    run_var_model,
    run_random_forest
)
from visualizations import (
    create_correlation_heatmap,
    create_residuals_plot,
    create_qq_plot,
    create_prediction_plot,
    create_feature_importance_plot,
    create_time_series_plot,
    create_acf_pacf_plots
)
from export import export_to_pdf, export_to_png

# Configuração da página
st.set_page_config(
    page_title="Sistema de Análise Econométrica",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS customizado
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stButton>button {
        width: 100%;
    }
    .upload-text {
        font-size: 1.2rem;
        font-weight: bold;
        color: #1f77b4;
    }
    </style>
""", unsafe_allow_html=True)

# Inicializar session state
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False
if 'username' not in st.session_state:
    st.session_state.username = None
if 'data' not in st.session_state:
    st.session_state.data = None
if 'model_results' not in st.session_state:
    st.session_state.model_results = None

def main():
    # Verificar autenticação
    if not st.session_state.authenticated:
        show_login_page()
    else:
        show_main_app()

def show_login_page():
    st.title("🔐 Sistema de Análise Econométrica")
    st.subheader("Faça login para continuar")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("---")
        username = st.text_input("👤 Usuário", placeholder="Digite seu usuário")
        password = st.text_input("🔑 Senha", type="password", placeholder="Digite sua senha")
        
        if st.button("Entrar", type="primary"):
            if authenticate_user(username, password):
                st.session_state.authenticated = True
                st.session_state.username = username
                st.success("Login realizado com sucesso!")
                st.rerun()
            else:
                st.error("Usuário ou senha incorretos!")
        
        st.markdown("---")
        st.info("**Demo:** usuário: `admin` | senha: `admin123`")

def show_main_app():
    # Sidebar
    with st.sidebar:
        st.title("📊 Menu Principal")
        st.write(f"**Usuário:** {st.session_state.username}")
        
        if st.button("🚪 Sair"):
            st.session_state.authenticated = False
            st.session_state.username = None
            st.session_state.data = None
            st.session_state.model_results = None
            st.rerun()
        
        st.markdown("---")
        
        # Upload de arquivo
        st.subheader("📁 Upload de Dados")
        uploaded_file = st.file_uploader(
            "Escolha um arquivo CSV ou XLSX",
            type=['csv', 'xlsx'],
            help="Faça upload dos seus dados para análise"
        )
        
        if uploaded_file is not None:
            try:
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                else:
                    df = pd.read_excel(uploaded_file)
                
                st.session_state.data = df
                st.success(f"✅ Arquivo carregado: {len(df)} linhas, {len(df.columns)} colunas")
            except Exception as e:
                st.error(f"Erro ao carregar arquivo: {str(e)}")
    
    # Conteúdo principal
    st.title("📈 Sistema de Análise Econométrica")
    
    if st.session_state.data is not None:
        show_data_analysis()
    else:
        show_welcome_screen()

def show_welcome_screen():
    st.markdown("""
    ### 👋 Bem-vindo ao Sistema de Análise Econométrica!
    
    Este sistema permite que você:
    - 📤 Faça upload de dados em formato CSV ou XLSX
    - 🤖 Receba sugestões automáticas de modelos econométricos
    - 📊 Execute diversos modelos: OLS, Logit, Probit, ARIMA, VAR, Random Forest
    - 📈 Visualize resultados através de gráficos interativos
    - 💾 Exporte gráficos em formato PDF ou PNG
    
    **Para começar, faça upload de um arquivo no menu lateral.**
    """)
    
    # Exemplos de estrutura de dados
    with st.expander("📋 Exemplo de estrutura de dados"):
        st.markdown("""
        **Para regressão linear:**

        - Uma coluna deve ser a variável dependente (target)
        - As demais colunas devem ser variáveis independentes (features)
        - Não devem haver valores faltantes ou duplicados

        **Exemplo:**

        | target | feature1 | feature2 |
        |--------|----------|----------|
        | 10     | 2        | 3        |
        | 20     | 4        | 5        |
        | 30     | 6        | 7        |
        """)

def show_data_analysis():
    # Análise de dados aqui
    st.write("Análise de dados será implementada aqui")

def create_all_files():
    """Cria todos os arquivos do projeto"""
    
    app_content = open('app.py', 'r', encoding='utf-8').read()
    write_file('econometric_system/app.py', app_content)
    
    auth_content = open('auth.py', 'r', encoding='utf-8').read()
    write_file('econometric_system/auth.py', auth_content)
    
    models_content = open('models.py', 'r', encoding='utf-8').read()
    write_file('econometric_system/models.py', models_content)
    
    viz_content = open('visualizations.py', 'r', encoding='utf-8').read()
    write_file('econometric_system/visualizations.py', viz_content)
    
    export_content = open('export.py', 'r', encoding='utf-8').read()
    write_file('econometric_system/export.py', export_content)
    
    req_content = open('requirements.txt', 'r', encoding='utf-8').read()
    write_file('econometric_system/requirements.txt', req_content)
    
    readme_content = open('README.md', 'r', encoding='utf-8').read()
    write_file('econometric_system/README.md', readme_content)

if __name__ == "__main__":
    print("=" * 60)
    print("GERADOR DE PROJETO - SISTEMA DE ANÁLISE ECONOMÉTRICA")
    print("=" * 60)
    print()
    
    print("Criando estrutura de diretórios...")
    create_directory_structure()
    print()
    
    print("Copiando arquivos do projeto...")
    try:
        create_all_files()
    except FileNotFoundError as e:
        print(f"\nERRO: {e}")
        print("\nEste script deve ser executado no mesmo diretório dos arquivos fonte:")
        print("  - app.py")
        print("  - auth.py")
        print("  - models.py")
        print("  - visualizations.py")
        print("  - export.py")
        print("  - requirements.txt")
        print("  - README.md")
        return
    
    print()
    print("=" * 60)
    print("PROJETO CRIADO COM SUCESSO!")
    print("=" * 60)
    print()
    print("Para executar o projeto:")
    print("  1. cd econometric_system")
    print("  2. pip install -r requirements.txt")
    print("  3. streamlit run app.py")
    print()
    print("Credenciais de login:")
    print("  Usuário: admin")
    print("  Senha: admin123")
    print()
