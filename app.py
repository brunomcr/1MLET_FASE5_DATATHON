import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pyspark.sql import SparkSession
import os
import logging

# Configuração de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuração de tema e cores
THEME_COLORS = {
    'primary': '#2196F3',
    'secondary': '#FF9800',
    'background': '#000000',
    'surface': '#1E1E1E',   
    'text': '#FFFFFF'       
}

# Configuração da página
st.set_page_config(
    page_title="Dashboard G1 - Sistema de Recomendação",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personalizado
st.markdown("""
    <style>
    .main {
        background-color: #000000;  /* Fundo preto */
        color: #FFFFFF;              /* Texto branco */
    }
    .stMetric {
        background-color: #1E1E1E;   /* Cor de superfície */
        color: #FFFFFF;              /* Texto branco */
    }
    </style>
""", unsafe_allow_html=True)

# Inicializar Spark e carregar dados
@st.cache_resource
def init_spark():
    spark = SparkSession.builder \
        .appName("G1 Recommendations") \
        .config("spark.sql.parquet.datetimeRebaseModeInRead", "CORRECTED") \
        .config("spark.sql.parquet.datetimeRebaseModeInWrite", "CORRECTED") \
        .getOrCreate()
    
    # Ler os arquivos parquet com particionamento
    treino = spark.read \
        .option("mergeSchema", "false") \
        .parquet("/app/datalake/silver/treino") \
    
    itens = spark.read \
    .option("mergeSchema", "false") \
    .parquet("datalake/silver/itens") \
    
    # Registrar as tabelas temporárias
    treino.createOrReplaceTempView("tab_treino")
    itens.createOrReplaceTempView("tab_itens")
    
    return spark

# Cache para queries frequentes
@st.cache_data
def get_basic_metrics():
    return {
        "total_users": spark.sql('SELECT COUNT(DISTINCT userId) FROM tab_treino').collect()[0][0],
        "total_interactions": spark.sql('SELECT COUNT(*) FROM tab_treino').collect()[0][0],
        "avg_interactions": spark.sql('SELECT CAST(COUNT(*)/COUNT(DISTINCT userId) AS INT) FROM tab_treino').collect()[0][0],
        "total_news": spark.sql('SELECT COUNT(DISTINCT history) FROM tab_treino').collect()[0][0]
    }

@st.cache_data
def get_user_distribution():
    return spark.sql("""
        SELECT 
            userType,
            COUNT(DISTINCT userId) as unique_users,
            CAST(AVG(numberOfClicksHistory) as INT) as avg_clicks,
            CAST(AVG(timeOnPageHistory) as INT) as avg_time,
            CAST(AVG(scrollPercentageHistory) as INT) as avg_scroll
        FROM tab_treino
        GROUP BY userType
        ORDER BY unique_users DESC
    """).toPandas()

@st.cache_data
def get_cold_start_analysis(_spark):
    return _spark.sql("""
        WITH user_interactions AS (
            SELECT userId, COUNT(*) as interaction_count
            FROM tab_treino
            GROUP BY userId
        )
        SELECT 
            CASE 
                WHEN interaction_count < 5 THEN 'Muito Baixo (< 5)'
                WHEN interaction_count < 10 THEN 'Baixo (5-9)'
                WHEN interaction_count < 20 THEN 'Médio (10-19)'
                ELSE 'Alto (20+)'
            END as nivel_interacao,
            COUNT(*) as num_users,
            CAST((COUNT(*) * 100.0 / SUM(COUNT(*)) OVER ()) as INT) as percentual
        FROM user_interactions
        GROUP BY 
            CASE 
                WHEN interaction_count < 5 THEN 'Muito Baixo (< 5)'
                WHEN interaction_count < 10 THEN 'Baixo (5-9)'
                WHEN interaction_count < 20 THEN 'Médio (10-19)'
                ELSE 'Alto (20+)'
            END
        ORDER BY num_users DESC
    """).toPandas()

@st.cache_data
def get_recency_analysis():
    return spark.sql("""
        WITH user_recency AS (
            SELECT 
                userId,
                DATEDIFF(MAX(timestampHistory), MIN(timestampHistory)) as days_active
            FROM tab_treino
            GROUP BY userId
        )
        SELECT 
            CASE 
                WHEN days_active <= 1 THEN '1 dia'
                WHEN days_active <= 7 THEN '1 semana'
                WHEN days_active <= 30 THEN '1 mês'
                ELSE 'Mais de 1 mês'
            END as periodo,
            COUNT(*) as num_users
        FROM user_recency
        GROUP BY 
            CASE 
                WHEN days_active <= 1 THEN '1 dia'
                WHEN days_active <= 7 THEN '1 semana'
                WHEN days_active <= 30 THEN '1 mês'
                ELSE 'Mais de 1 mês'
            END
        ORDER BY num_users DESC
    """).toPandas()

@st.cache_data
def get_content_analysis():
    return spark.sql("""
        SELECT 
            i.page as category,
            COUNT(*) as total_reads,
            COUNT(DISTINCT t.userId) as unique_readers,
            CAST(AVG(t.timeOnPageHistory) as INT) as avg_time
        FROM tab_treino t
        JOIN tab_itens i ON t.history = i.page
        GROUP BY i.page
        HAVING COUNT(*) > 100
        ORDER BY total_reads DESC
        LIMIT 10
    """).toPandas()

@st.cache_data
def get_temporal_distribution(_spark):
    try:
        temporal_dist = _spark.sql("""
            SELECT 
                DATE(timestampHistory) as data,
                COUNT(*) as total_interacoes,
                COUNT(DISTINCT userId) as usuarios_unicos
            FROM tab_treino
            GROUP BY DATE(timestampHistory)
            ORDER BY data
        """).toPandas()
        
        logger.info(f"Dados de distribuição temporal: {temporal_dist.head()}")
        return temporal_dist
    except Exception as e:
        logger.error(f"Erro ao obter distribuição temporal: {str(e)}")
        return pd.DataFrame()  # Retorna um DataFrame vazio

@st.cache_data
def get_engagement_metrics():
    return spark.sql("""
        SELECT 
            userType,
            CAST(AVG(numberOfClicksHistory) as INT) as media_clicks,
            CAST(AVG(timeOnPageHistory)/60 as INT) as media_tempo_minutos,
            CAST(AVG(scrollPercentageHistory) as INT) as media_scroll
        FROM tab_treino
        GROUP BY userType
        ORDER BY media_tempo_minutos DESC
    """).toPandas()

@st.cache_data
def get_hourly_pattern():
    return spark.sql("""
        SELECT 
            HOUR(timestampHistory) as hora,
            COUNT(*) as total_acessos,
            COUNT(DISTINCT userId) as usuarios_unicos
        FROM tab_treino
        GROUP BY HOUR(timestampHistory)
        ORDER BY hora
    """).toPandas()

@st.cache_data
def get_top_categories():
    return spark.sql("""
        SELECT 
            i.page as category,
            COUNT(*) as total_reads,
            COUNT(DISTINCT t.userId) as unique_readers,
            CAST(AVG(t.timeOnPageHistory)/60 as INT) as avg_time_minutes,
            CAST(AVG(t.scrollPercentageHistory) as INT) as avg_scroll
        FROM tab_treino t
        JOIN tab_itens i ON t.history = i.page
        GROUP BY i.page
        HAVING COUNT(*) > 100
        ORDER BY total_reads DESC
        LIMIT 10
    """).toPandas()

@st.cache_data
def get_correlation_metrics():
    return spark.sql("""
        SELECT 
            CORR(numberOfClicksHistory, timeOnPageHistory) as corr_clicks_time,
            CORR(numberOfClicksHistory, scrollPercentageHistory) as corr_clicks_scroll,
            CORR(timeOnPageHistory, scrollPercentageHistory) as corr_time_scroll
        FROM tab_treino
    """).toPandas()

# Inicializar Spark e carregar dados
try:
    spark = init_spark()
    #st.sidebar.success("✅ Conexão com Spark estabelecida")
    print("Conexão com Spark estabelecida com sucesso")
except Exception as e:
    st.sidebar.error(f"❌ Erro ao conectar com Spark: {str(e)}")
    st.stop()

# Sidebar com navegação mais limpa
st.sidebar.title("📊 Dashboard G1")

# Menu de navegação simplificado
page = st.sidebar.radio(
    "Navegação",
    options=["Início", "Visão Geral", "Perfil dos Usuários", "Cold Start", 
             "Recência e Engajamento", "Análise de Conteúdo", 
             "Distribuição Temporal", "Conclusão"]
)

# Mover as informações de debug para uma seção expansível
with st.sidebar.expander("ℹ️ Informações de Debug", expanded=False):
    st.write("Verificando caminhos:")
    st.write(f"Conteúdo de /app/datalake/silver/:")
    st.write(os.listdir("/app/datalake/silver/"))
    st.write("✅ Arquivo treino carregado")
    st.write(f"Registros treino: {spark.sql('SELECT COUNT(*) FROM tab_treino').collect()[0][0]:,}")
    st.write("✅ Arquivo itens carregado")
    st.write(f"Registros itens: {spark.sql('SELECT COUNT(*) FROM tab_itens').collect()[0][0]:,}")
    st.write("✅ Conexão com Spark estabelecida")

# Funções de visualização usando dados cacheados
def show_objective(objective_text):
    """Exibe o objetivo da análise para a guia atual."""
    st.markdown(f"### Objetivo\n{objective_text}")

def show_visao_geral():
    show_objective("Esta seção fornece uma visão geral das interações dos usuários e do desempenho do conteúdo.")
    st.title("🎯 Visão Geral")
    
    metrics = get_basic_metrics()
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total de Usuários", f"{metrics['total_users']:,}")
    with col2:
        st.metric("Total de Interações", f"{metrics['total_interactions']:,}")
    with col3:
        st.metric("Média Interações/Usuário", f"{metrics['avg_interactions']:,}")
    with col4:
        st.metric("Total de Notícias", f"{metrics['total_news']:,}")

def show_perfil_usuarios(spark):
    show_objective("Analise os perfis dos usuários para identificar padrões de comportamento e segmentar usuários com base em interações.")
    st.title("👥 Análise do Perfil dos Usuários")
    
    # Obter dados de distribuição de usuários
    user_dist = get_user_distribution()
    
    if user_dist.empty:
        st.warning("Nenhum dado disponível para a distribuição de usuários.")
        return
    
    # Criar colunas para os gráficos
    col1, col2 = st.columns(2)

    with col1:
        # Gráfico de Distribuição de Usuários
        fig1 = px.bar(
            user_dist,
            x='userType',
            y='unique_users',
            title='Distribuição de Usuários por Tipo',
            color_discrete_sequence=[THEME_COLORS['primary']]
        )
        fig1.update_layout(
            plot_bgcolor=THEME_COLORS['background'],  # Fundo do gráfico
            paper_bgcolor=THEME_COLORS['background'],  # Fundo do gráfico
            font=dict(color=THEME_COLORS['text'])
        )
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        # Gráfico de Métricas de Engajamento
        engagement_metrics = get_engagement_metrics()
        fig2 = px.bar(
            engagement_metrics,
            x='userType',
            y=['media_clicks', 'media_tempo_minutos', 'media_scroll'],
            title='Métricas de Engajamento por Tipo de Usuário',
            barmode='group',
            color_discrete_sequence=[THEME_COLORS['primary'], THEME_COLORS['secondary'], '#4CAF50']
        )
        fig2.update_layout(
            plot_bgcolor=THEME_COLORS['background'],  # Fundo do gráfico
            paper_bgcolor=THEME_COLORS['background'],  # Fundo do gráfico
            font=dict(color=THEME_COLORS['text'])
        )
        st.plotly_chart(fig2, use_container_width=True)

    st.subheader("📅 Padrões de Acesso")
    hourly_pattern = get_hourly_pattern()
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(x=hourly_pattern['hora'], y=hourly_pattern['total_acessos'],
                            name='Total de Acessos', mode='lines'))
    fig3.add_trace(go.Scatter(x=hourly_pattern['hora'], y=hourly_pattern['usuarios_unicos'],
                            name='Usuários Únicos', mode='lines'))
    fig3.update_layout(
        plot_bgcolor=THEME_COLORS['background'],  # Fundo do gráfico
        paper_bgcolor=THEME_COLORS['background'],  # Fundo do gráfico
        font=dict(color=THEME_COLORS['text']),     # Texto do gráfico
        title='Distribuição de Acessos por Hora do Dia'
    )
    st.plotly_chart(fig3, use_container_width=True)

    st.markdown("""
    ### 👥 Insights sobre os Usuários
    - Diferentes perfis apresentam padrões distintos de consumo
    - Horários de pico bem definidos ao longo do dia
    - Variação significativa no tempo de leitura entre tipos de usuário
    - Oportunidade para personalização temporal das recomendações
    """)

def show_cold_start(spark):
    show_objective("Aborde o desafio do cold start desenvolvendo estratégias para melhorar a experiência de novos usuários com dados limitados.")
    st.title("🆕 Análise de Cold Start")
    
    cold_start = get_cold_start_analysis(spark)
    
    if cold_start.empty:
        st.warning("Nenhum dado disponível para a análise de cold start.")
        return
    
    fig = go.Figure()
    fig.add_trace(go.Pie(
        labels=cold_start['nivel_interacao'],
        values=cold_start['num_users'],
        marker=dict(colors=[THEME_COLORS['primary'], THEME_COLORS['secondary'], '#4CAF50', '#FF9800']),
        textinfo='percent+label'
    ))
    
    fig.update_layout(
        title='Distribuição de Usuários por Nível de Interação',
        plot_bgcolor=THEME_COLORS['background'],  # Fundo do gráfico
        paper_bgcolor=THEME_COLORS['background'],  # Fundo do gráfico
        font=dict(color=THEME_COLORS['text'])      # Texto do gráfico
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("""
    ### 🔍 Desafios Identificados
    - Alto percentual de usuários com poucas interações
    - Necessidade de estratégias para novos usuários
    - Importância do primeiro contato
    
    ### ⚡ Estratégias Sugeridas
    - Usar popularidade global para novos usuários
    - Implementar recomendações baseadas em contexto
    - Coletar informações mínimas no cadastro
    """)

def show_recencia_engajamento():
    show_objective("Explore métricas de recência e engajamento para entender a atividade dos usuários ao longo do tempo.")
    st.title("Análise de Recência e Engajamento")
    recency = get_recency_analysis()
    
    fig = go.Figure()
    fig.add_trace(go.Bar(x=recency['periodo'], y=recency['num_users'],
                        marker=dict(color=THEME_COLORS['primary'])))
    fig.update_layout(
        plot_bgcolor=THEME_COLORS['background'],  # Fundo do gráfico
        paper_bgcolor=THEME_COLORS['background'],  # Fundo do gráfico
        font=dict(color=THEME_COLORS['text']),     # Texto do gráfico
        title='Distribuição de Usuários por Período de Atividade'
    )
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("""
    ### ⏰ Padrões Temporais
    - Diferentes níveis de atividade ao longo do tempo
    - Importância da recência nas interações
    - Ciclos de engajamento identificados
    
    ### 🎯 Recomendações
    - Priorizar conteúdo recente
    - Reativar usuários inativos
    - Balancear novidade e relevância
    """)

def show_analise_conteudo():
    show_objective("Analise o desempenho do conteúdo para avaliar quais tipos de conteúdo geram mais engajamento.")
    st.title("Análise de Conteúdo")
    
    top_cats = get_top_categories()
    
    # Gráfico de Top Categorias mais Lidas
    fig1 = px.bar(top_cats, x='category', y=['total_reads', 'unique_readers'],
                  title='Top 10 Categorias mais Lidas',
                  barmode='group')
    fig1.update_layout(
        plot_bgcolor=THEME_COLORS['background'],  # Fundo do gráfico
        paper_bgcolor=THEME_COLORS['background'],  # Fundo do gráfico
        font=dict(color=THEME_COLORS['text'])
    )
    st.plotly_chart(fig1, use_container_width=True)
    
    # Gráfico de Métricas de Engajamento por Categoria
    fig2 = px.bar(top_cats, x='category', y=['avg_time_minutes', 'avg_scroll'],
                  title='Métricas de Engajamento por Categoria',
                  barmode='group')
    fig2.update_layout(
        plot_bgcolor=THEME_COLORS['background'],  # Fundo do gráfico
        paper_bgcolor=THEME_COLORS['background'],  # Fundo do gráfico
        font=dict(color=THEME_COLORS['text'])
    )
    st.plotly_chart(fig2, use_container_width=True)

    st.subheader("🔄 Correlação entre Métricas de Engajamento")
    correlation = get_correlation_metrics()
    
    st.write("Correlações:")
    st.write("- Clicks vs Tempo: ", round(correlation['corr_clicks_time'][0], 2))
    st.write("- Clicks vs Scroll: ", round(correlation['corr_clicks_scroll'][0], 2))
    st.write("- Tempo vs Scroll: ", round(correlation['corr_time_scroll'][0], 2))

    st.markdown("""
    ### 📰 Insights sobre Conteúdo
    - Categorias populares têm padrões distintos de engajamento
    - Tempo de leitura varia significativamente entre categorias
    - Correlação interessante entre métricas de engajamento
    - Oportunidade para recomendações baseadas em padrões de consumo
    """)

# Função para mostrar a distribuição temporal das interações
def show_temporal_distribution(spark):
    show_objective("Descubra padrões temporais e sazonalidades nas interações dos usuários ao longo do tempo.")
    st.title("📈 Distribuição Temporal das Interações")
    
    temporal_dist = get_temporal_distribution(spark)
    
    if temporal_dist.empty:
        st.warning("Nenhum dado disponível para a distribuição temporal.")
        return
    
    # Gráfico com dados válidos
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=temporal_dist['data'],
        y=temporal_dist['total_interacoes'],
        name='Total de Interações',
        line=dict(color=THEME_COLORS['primary'], width=2)
    ))
    fig.add_trace(go.Scatter(
        x=temporal_dist['data'],
        y=temporal_dist['usuarios_unicos'],
        name='Usuários Únicos',
        line=dict(color=THEME_COLORS['secondary'], width=2)
    ))
    
    fig.update_layout(
        plot_bgcolor=THEME_COLORS['background'],  # Fundo do gráfico
        paper_bgcolor=THEME_COLORS['background'],  # Fundo do gráfico
        font=dict(color=THEME_COLORS['text']),     # Texto do gráfico
        title='Distribuição Temporal das Interações'
    )
    
    st.subheader("📈 Distribuição Temporal das Interações")
    temporal_dist = get_temporal_distribution(spark)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=temporal_dist['data'], y=temporal_dist['total_interacoes'],
                            name='Total de Interações', mode='lines'))
    fig.add_trace(go.Scatter(x=temporal_dist['data'], y=temporal_dist['usuarios_unicos'],
                            name='Usuários Únicos', mode='lines'))
    fig.update_layout(
        plot_bgcolor=THEME_COLORS['background'],  # Fundo do gráfico
        paper_bgcolor=THEME_COLORS['background'],  # Fundo do gráfico
        font=dict(color=THEME_COLORS['text']),     # Texto do gráfico
        title='Distribuição Temporal das Interações'
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    ### 📊 Insights Principais
    - Volume significativo de dados com mais de 8 milhões de interações
    - Base diversificada de usuários com diferentes padrões de consumo
    - Padrões temporais claros nas interações
    - Oportunidade para personalização em escala
    """)

def show_home():
    
    st.title("📊 Dashboard G1 - Sistema de Recomendação")
    
    st.markdown("""
    Bem-vindo ao Dashboard G1, uma plataforma interativa para análise exploratória de dados do nosso sistema de recomendação. 
    Este dashboard foi desenvolvido para fornecer insights valiosos sobre o comportamento dos usuários e o desempenho do conteúdo.

    ### Objetivos da Análise:
    - **Entender o Perfil dos Usuários**: Identificar padrões de comportamento e segmentar usuários com base em suas interações.
    - **Analisar o Desempenho do Conteúdo**: Avaliar quais tipos de conteúdo geram mais engajamento e como os usuários interagem com eles.
    - **Explorar Padrões Temporais**: Descobrir tendências e sazonalidades nas interações dos usuários ao longo do tempo.
    - **Abordar o Desafio do Cold Start**: Desenvolver estratégias para melhorar a experiência de novos usuários com base em dados limitados.

    Navegue pelas diferentes seções para explorar os dados e descobrir insights que podem ajudar a otimizar nosso sistema de recomendação.
    """)

# Adicione a função de conclusão
def show_conclusion():
    st.title("🔍 Conclusão da Análise")
    
    st.markdown("""
    Após uma análise detalhada dos dados, chegamos às seguintes conclusões:

    - **Diversidade de Usuários**: Identificamos uma base diversificada de usuários com diferentes padrões de consumo, o que sugere a necessidade de personalização.
    - **Conteúdo Popular**: Certas categorias de conteúdo geram mais engajamento, indicando áreas de foco para futuras recomendações.
    - **Padrões Temporais**: Observamos padrões claros de interação ao longo do tempo, o que pode ser utilizado para otimizar o timing das recomendações.
    - **Desafios do Cold Start**: Estratégias específicas são necessárias para engajar novos usuários e melhorar sua experiência inicial.

    Continuaremos a monitorar e ajustar nosso sistema de recomendação com base nesses insights para oferecer uma experiência cada vez mais personalizada e eficaz.
    """)

# Execução da página selecionada
if page == "Início":
    show_home()
elif page == "Visão Geral":
    show_visao_geral()
elif page == "Perfil dos Usuários":
    show_perfil_usuarios(spark)
elif page == "Cold Start":
    show_cold_start(spark)
elif page == "Recência e Engajamento":
    show_recencia_engajamento()
elif page == "Análise de Conteúdo":
    show_analise_conteudo()
elif page == "Distribuição Temporal":
    show_temporal_distribution(spark)
elif page == "Conclusão":
    show_conclusion() 