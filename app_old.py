import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pyspark.sql import SparkSession
import os
import logging

# Configuração de logging
debug_mode = True  # Ativar para logs detalhados
logging.basicConfig(level=logging.DEBUG if debug_mode else logging.INFO)
logger = logging.getLogger(__name__)

# Certifique-se de definir treino_path e itens_path antes de usá-los
treino_path = "caminho/para/treino"
itens_path = "caminho/para/itens"

@st.cache_resource
def init_spark():
    try:
        spark = SparkSession.builder \
            .appName("G1 Recommendations") \
            .config("spark.sql.parquet.datetimeRebaseModeInRead", "CORRECTED") \
            .config("spark.sql.parquet.datetimeRebaseModeInWrite", "CORRECTED") \
            .getOrCreate()
        return spark
    except Exception as e:
        st.error(f"Erro ao inicializar SparkSession: {e}")
        return None

spark = init_spark()

if spark:
    st.sidebar.success("✅ Conexão com Spark estabelecida")
    # Carregar dados ou executar operações que dependem do Spark
    treino, itens = load_data()
else:
    st.sidebar.error("❌ SparkSession não inicializada. Dados não serão carregados.")
    treino, itens = pd.DataFrame(), pd.DataFrame()

# Load data
@st.cache_data
def load_data():
    treino_path = "/app/datalake/silver/treino"
    itens_path = "/app/datalake/silver/itens"
    
    logger.debug(f"Verificando caminhos: {treino_path}, {itens_path}")
    
    if spark is None:
        logger.warning("SparkSession não inicializada. Dados não serão carregados.")
        st.warning("SparkSession não inicializada. Dados não serão carregados.")
        return pd.DataFrame(), pd.DataFrame()
    
    if not os.path.exists(treino_path) or not os.path.exists(itens_path):
        logger.warning("Arquivos Parquet não encontrados. Verifique o caminho dentro do container.")
        st.warning("Arquivos Parquet não encontrados. Verifique o caminho dentro do container.")
        return pd.DataFrame(), pd.DataFrame()
    
    try:
        logger.info("Carregando arquivos Parquet...")
        treino = spark.read.parquet(treino_path).toPandas()
        itens = spark.read.parquet(itens_path).toPandas()
        logger.info("Dados carregados com sucesso.")
    except Exception as e:
        logger.error(f"Erro ao carregar dados: {e}")
        st.error(f"Erro ao carregar dados: {e}")
        return pd.DataFrame(), pd.DataFrame()
    
    return treino, itens

# Sidebar navigation
st.sidebar.title("Dashboard G1 - Sistema de Recomendação")
page = st.sidebar.radio("Escolha uma página:", ["Página 1", "Página 2"])

# Introdução
if page == "Página 1":
    st.title("Exploração de Dados para o Sistema de Recomendação")
    st.write(
        "Este dashboard apresenta a análise exploratória dos dados utilizados no sistema de recomendação. "
        "Cada aba segue uma jornada de descoberta, baseada na exploração dos padrões e insights dos usuários."
    )

# Distribuição de Usuários
elif page == "Página 2":
    if treino.empty:
        st.warning("Dados não carregados.")
    else:
        st.title("Distribuição por Tipo de Usuário")
        user_type_results = treino.groupby("userType").size().reset_index(name="count")
        fig = px.bar(user_type_results, x="userType", y="count", title="Distribuição por Tipo de Usuário")
        st.plotly_chart(fig)

# Engajamento
elif section == "Engajamento":
    if treino.empty:
        st.warning("Dados não carregados.")
    else:
        st.title("Análise de Engajamento")
        engagement_results = treino.groupby("userType").agg({
            "numberOfClicksHistory": "mean",
            "timeOnPageHistory": "mean",
            "scrollPercentageHistory": "mean",
            "pageVisitsCountHistory": "mean"
        }).reset_index()

        fig = go.Figure()
        fig.add_trace(go.Bar(x=engagement_results["userType"], y=engagement_results["numberOfClicksHistory"], name="Média de Clicks"))
        fig.add_trace(go.Bar(x=engagement_results["userType"], y=engagement_results["timeOnPageHistory"], name="Tempo na Página"))
        fig.add_trace(go.Bar(x=engagement_results["userType"], y=engagement_results["scrollPercentageHistory"], name="Scroll (%)"))
        fig.add_trace(go.Bar(x=engagement_results["userType"], y=engagement_results["pageVisitsCountHistory"], name="Visitas"))
        fig.update_layout(barmode='group', title="Métricas de Engajamento")
        st.plotly_chart(fig)

# Análise Temporal
elif section == "Análise Temporal":
    if treino.empty:
        st.warning("Dados não carregados.")
    else:
        st.title("Evolução Temporal de Interações e Usuários Únicos")
        treino["date"] = pd.to_datetime(treino["year"].astype(str) + "-" + treino["month"].astype(str) + "-01")
        temporal_results = treino.groupby("date").agg({"userId": "nunique", "sessionId": "count"}).reset_index()
        temporal_results.columns = ["date", "unique_users", "interactions"]

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=temporal_results["date"], y=temporal_results["interactions"], mode='lines+markers', name='Interações'))
        fig.add_trace(go.Scatter(x=temporal_results["date"], y=temporal_results["unique_users"], mode='lines+markers', name='Usuários Únicos'))
        fig.update_layout(title="Análise Temporal de Interações e Usuários", xaxis_title="Data", yaxis_title="Contagem")
        st.plotly_chart(fig)

# Cold-Start
elif section == "Cold-Start":
    if treino.empty:
        st.warning("Dados não carregados.")
    else:
        st.title("Análise de Cold-Start")
        treino["interaction_level"] = pd.cut(treino.groupby("userId")["sessionId"].transform("count"), bins=[0, 4, 9, 19, float("inf")], labels=["Muito Baixo", "Baixo", "Médio", "Alto"])
        cold_start_analysis = treino["interaction_level"].value_counts(normalize=True) * 100
        fig = px.bar(cold_start_analysis, x=cold_start_analysis.index, y=cold_start_analysis.values, title="Distribuição de Usuários por Nível de Interação")
        st.plotly_chart(fig)

# Análise de Correlação
elif section == "Análise de Correlação":
    st.title("🔗 Análise de Correlação")
    
    # Supondo que 'correlation_matrix' seja um DataFrame com as correlações
    correlation_matrix = get_correlation_matrix()  # Função hipotética
    
    fig = px.imshow(correlation_matrix, 
                    labels=dict(x="Métricas", y="Métricas", color="Correlação"),
                    x=correlation_matrix.columns,
                    y=correlation_matrix.index,
                    color_continuous_scale='Blues')
    
    fig.update_layout(
        plot_bgcolor=THEME_COLORS['background'],
        paper_bgcolor=THEME_COLORS['background'],
        font=dict(color=THEME_COLORS['text'])
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("""
    ### 🔍 Insights de Correlação
    - Identifique relações entre diferentes métricas de engajamento.
    - Utilize essas correlações para ajustar estratégias de recomendação.
    """)

st.write("**Caminho dos dados:**", treino_path, itens_path)
