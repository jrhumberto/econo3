"""
Sistema de Analise Econometrica
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime

from auth import authenticate_user
from models import (
    suggest_model, run_ols_regression, run_logit_model,
    run_probit_model, run_arima_model, run_var_model, run_random_forest
)
from visualizations import (
    create_correlation_heatmap, create_residuals_plot,
    create_qq_plot, create_prediction_plot,
    create_feature_importance_plot, create_time_series_plot
)
from export import export_to_pdf, export_to_png

st.set_page_config(page_title="Sistema Econometrico", page_icon="chart", layout="wide")


def init_session_state():
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
    if 'username' not in st.session_state:
        st.session_state.username = None
    if 'data' not in st.session_state:
        st.session_state.data = None
    if 'model_results' not in st.session_state:
        st.session_state.model_results = None


def main():
    init_session_state()
    if not st.session_state.authenticated:
        show_login()
    else:
        show_main_app()


def show_login():
    st.title("Sistema de Analise Econometrica")
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.subheader("Login")
        with st.form("login_form"):
            username = st.text_input("Usuario")
            password = st.text_input("Senha", type="password")
            submit = st.form_submit_button("Entrar", use_container_width=True)
            if submit:
                if authenticate_user(username, password):
                    st.session_state.authenticated = True
                    st.session_state.username = username
                    st.rerun()
                else:
                    st.error("Usuario ou senha invalidos!")
        st.info("Usuarios: admin/admin123, demo/demo123, user/user123")


def show_main_app():
    st.sidebar.title(f"Ola, {st.session_state.username}!")
    if st.sidebar.button("Sair", use_container_width=True):
        st.session_state.authenticated = False
        st.session_state.username = None
        st.session_state.data = None
        st.session_state.model_results = None
        st.rerun()
    st.sidebar.markdown("---")
    st.title("Sistema de Analise Econometrica")
    if st.session_state.data is None:
        show_upload_section()
    else:
        show_data_analysis()


def show_upload_section():
    st.subheader("Upload de Dados")
    uploaded_file = st.file_uploader("Selecione um arquivo CSV ou Excel", type=['csv', 'xlsx', 'xls'])
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            st.session_state.data = df
            st.success(f"Arquivo carregado: {len(df)} linhas, {len(df.columns)} colunas")
            st.rerun()
        except Exception as e:
            st.error(f"Erro ao carregar arquivo: {str(e)}")


def show_data_analysis():
    df = st.session_state.data
    if st.sidebar.button("Carregar novo arquivo"):
        st.session_state.data = None
        st.session_state.model_results = None
        st.rerun()
    tab1, tab2, tab3, tab4 = st.tabs(["Exploracao", "Sugestao", "Modelagem", "Exportar"])
    with tab1:
        show_data_exploration(df)
    with tab2:
        show_model_suggestion(df)
    with tab3:
        show_modeling(df)
    with tab4:
        show_export_options()


def show_data_exploration(df):
    st.subheader("Exploracao dos Dados")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Linhas", len(df))
    with col2:
        st.metric("Colunas", len(df.columns))
    with col3:
        st.metric("Valores Nulos", df.isnull().sum().sum())
    with col4:
        st.metric("Memoria", f"{df.memory_usage(deep=True).sum() / 1024:.2f} KB")
    st.markdown("### Visualizacao dos Dados")
    st.dataframe(df.head(20), use_container_width=True)
    st.markdown("### Estatisticas Descritivas")
    st.dataframe(df.describe(), use_container_width=True)


def show_model_suggestion(df):
    st.subheader("Sugestao Automatica de Modelo")
    suggestion = suggest_model(df)
    st.markdown(f"### Modelo Recomendado: **{suggestion['model']}**")
    st.markdown(f"**Justificativa:** {suggestion['reason']}")
    st.markdown(f"- Variaveis numericas: {suggestion['numeric_cols']}")
    st.markdown(f"- Variaveis categoricas: {suggestion['categorical_cols']}")
    st.markdown(f"- Total de observacoes: {suggestion['n_rows']}")
    with st.expander("Mais informacoes"):
        st.markdown(suggestion['description'])


def show_modeling(df):
    st.subheader("Modelagem Econometrica")
    model_type = st.selectbox("Escolha o modelo:", [
        "Regressao Linear (OLS)", "Regressao Logistica (Logit)", "Modelo Probit",
        "ARIMA (Series Temporais)", "VAR (Vetores Autoregressivos)", "Random Forest"
    ])
    st.markdown("---")
    if model_type == "Regressao Linear (OLS)":
        run_ols_interface(df)
    elif model_type == "Regressao Logistica (Logit)":
        run_logit_interface(df)
    elif model_type == "Modelo Probit":
        run_probit_interface(df)
    elif model_type == "ARIMA (Series Temporais)":
        run_arima_interface(df)
    elif model_type == "VAR (Vetores Autoregressivos)":
        run_var_interface(df)
    elif model_type == "Random Forest":
        run_random_forest_interface(df)


def run_ols_interface(df):
    st.markdown("### Regressao Linear (OLS)")
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    col1, col2 = st.columns(2)
    with col1:
        y_var = st.selectbox("Variavel Dependente (Y):", numeric_cols)
    with col2:
        x_vars = st.multiselect("Variaveis Independentes (X):", [c for c in numeric_cols if c != y_var])
    if st.button("Executar Modelo OLS", type="primary"):
        if len(x_vars) == 0:
            st.error("Selecione pelo menos uma variavel independente!")
        else:
            with st.spinner("Executando modelo..."):
                results = run_ols_regression(df, y_var, x_vars)
                if results:
                    st.session_state.model_results = results
                    display_ols_results(results)


def display_ols_results(results):
    st.success("Modelo executado com sucesso!")
    st.markdown("### Resumo do Modelo")
    st.text(str(results['summary']))
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("R2", f"{results['r_squared']:.4f}")
    with col2:
        st.metric("R2 Ajustado", f"{results['r_squared_adj']:.4f}")
    with col3:
        st.metric("F-statistic", f"{results['f_statistic']:.2f}")
    with col4:
        st.metric("Durbin-Watson", f"{results['durbin_watson']:.4f}")
    st.markdown("### Visualizacoes")
    t1, t2, t3, t4 = st.tabs(["Residuos", "Q-Q Plot", "Predicoes", "Correlacao"])
    with t1:
        st.pyplot(create_residuals_plot(results))
    with t2:
        st.pyplot(create_qq_plot(results))
    with t3:
        st.pyplot(create_prediction_plot(results))
    with t4:
        st.pyplot(create_correlation_heatmap(results['data']))


def run_logit_interface(df):
    st.markdown("### Regressao Logistica (Logit)")
    st.info("Modelo para variavel dependente binaria (0 ou 1)")
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    col1, col2 = st.columns(2)
    with col1:
        y_var = st.selectbox("Variavel Dependente Binaria (Y):", numeric_cols)
    with col2:
        x_vars = st.multiselect("Variaveis Independentes (X):", [c for c in numeric_cols if c != y_var])
    if st.button("Executar Modelo Logit", type="primary"):
        if len(x_vars) == 0:
            st.error("Selecione pelo menos uma variavel independente!")
        else:
            with st.spinner("Executando modelo..."):
                results = run_logit_model(df, y_var, x_vars)
                if results:
                    st.session_state.model_results = results
                    st.success("Modelo executado com sucesso!")
                    st.text(str(results['summary']))


def run_probit_interface(df):
    st.markdown("### Modelo Probit")
    st.info("Similar ao Logit, usado para variaveis binarias")
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    col1, col2 = st.columns(2)
    with col1:
        y_var = st.selectbox("Variavel Dependente Binaria (Y):", numeric_cols)
    with col2:
        x_vars = st.multiselect("Variaveis Independentes (X):", [c for c in numeric_cols if c != y_var])
    if st.button("Executar Modelo Probit", type="primary"):
        if len(x_vars) == 0:
            st.error("Selecione pelo menos uma variavel independente!")
        else:
            with st.spinner("Executando modelo..."):
                results = run_probit_model(df, y_var, x_vars)
                if results:
                    st.session_state.model_results = results
                    st.success("Modelo executado com sucesso!")
                    st.text(str(results['summary']))


def run_arima_interface(df):
    st.markdown("### Modelo ARIMA")
    st.info("Modelo para analise de series temporais")
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    col1, col2 = st.columns(2)
    with col1:
        ts_var = st.selectbox("Selecione a serie temporal:", numeric_cols)
    with col2:
        auto_params = st.checkbox("Detectar parametros automaticamente", value=True)
    if not auto_params:
        c1, c2, c3 = st.columns(3)
        with c1:
            p = st.number_input("p (AR):", min_value=0, max_value=5, value=1)
        with c2:
            d = st.number_input("d (I):", min_value=0, max_value=2, value=1)
        with c3:
            q = st.number_input("q (MA):", min_value=0, max_value=5, value=1)
    else:
        p, d, q = 1, 1, 1
    if st.button("Executar Modelo ARIMA", type="primary"):
        with st.spinner("Executando modelo..."):
            results = run_arima_model(df, ts_var, (p, d, q))
            if results:
                st.session_state.model_results = results
                st.success("Modelo executado com sucesso!")
                st.text(str(results['summary']))
                st.pyplot(create_time_series_plot(results))


def run_var_interface(df):
    st.markdown("### Modelo VAR")
    st.info("Vetores Autoregressivos para multiplas series temporais")
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    vars_selected = st.multiselect("Selecione as variaveis (2 ou mais):", numeric_cols)
    lags = st.slider("Numero de lags:", min_value=1, max_value=10, value=2)
    if st.button("Executar Modelo VAR", type="primary"):
        if len(vars_selected) < 2:
            st.error("Selecione pelo menos 2 variaveis!")
        else:
            with st.spinner("Executando modelo..."):
                results = run_var_model(df, vars_selected, lags)
                if results:
                    st.session_state.model_results = results
                    st.success("Modelo executado com sucesso!")
                    st.text(str(results['summary']))


def run_random_forest_interface(df):
    st.markdown("### Random Forest")
    st.info("Modelo de machine learning para previsao")
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    col1, col2 = st.columns(2)
    with col1:
        y_var = st.selectbox("Variavel Target (Y):", numeric_cols)
    with col2:
        x_vars = st.multiselect("Features (X):", [c for c in numeric_cols if c != y_var])
    c1, c2 = st.columns(2)
    with c1:
        n_estimators = st.slider("Numero de arvores:", 10, 500, 100)
    with c2:
        test_size = st.slider("Tamanho do teste (%):", 10, 40, 20)
    if st.button("Executar Random Forest", type="primary"):
        if len(x_vars) == 0:
            st.error("Selecione pelo menos uma feature!")
        else:
            with st.spinner("Treinando modelo..."):
                results = run_random_forest(df, y_var, x_vars, n_estimators, test_size/100)
                if results:
                    st.session_state.model_results = results
                    st.success("Modelo treinado com sucesso!")
                    c1, c2, c3 = st.columns(3)
                    with c1:
                        st.metric("R2 Score", f"{results['r2']:.4f}")
                    with c2:
                        st.metric("RMSE", f"{results['rmse']:.4f}")
                    with c3:
                        st.metric("MAE", f"{results['mae']:.4f}")
                    st.pyplot(create_feature_importance_plot(results))


def show_export_options():
    st.subheader("Exportar Resultados")
    if st.session_state.model_results is None:
        st.warning("Execute um modelo primeiro!")
        return
    st.markdown("### Opcoes de Download")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Exportar como PDF")
        if st.button("Gerar PDF", use_container_width=True):
            with st.spinner("Gerando PDF..."):
                pdf_buffer = export_to_pdf(st.session_state.model_results)
                if pdf_buffer:
                    st.download_button(
                        label="Download PDF", data=pdf_buffer,
                        file_name=f"analise_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                        mime="application/pdf", use_container_width=True
                    )
    with col2:
        st.markdown("#### Exportar como PNG")
        if st.button("Gerar PNG", use_container_width=True):
            with st.spinner("Gerando imagens..."):
                png_files = export_to_png(st.session_state.model_results)
                if png_files:
                    for i, png_buffer in enumerate(png_files):
                        st.download_button(
                            label=f"Download Grafico {i+1}", data=png_buffer,
                            file_name=f"grafico_{i+1}.png", mime="image/png",
                            use_container_width=True
                        )


if __name__ == "__main__":
    main()
