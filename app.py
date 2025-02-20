import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pyspark.sql import SparkSession
import os
import logging
import concurrent.futures
import time
from typing import Dict, Any

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
        .config("spark.driver.memory", "4g") \
        .config("spark.executor.memory", "4g") \
        .config("spark.sql.shuffle.partitions", "10") \
        .config("spark.sql.autoBroadcastJoinThreshold", "10m") \
        .config("spark.memory.fraction", "0.8") \
        .config("spark.memory.storageFraction", "0.3") \
        .config("spark.sql.adaptive.enabled", "true") \
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
        .config("spark.sql.adaptive.skewJoin.enabled", "true") \
        .config("spark.sql.inMemoryColumnarStorage.compressed", "true") \
        .config("spark.sql.inMemoryColumnarStorage.batchSize", "10000") \
        .config("spark.sql.shuffle.partitions", "8") \
        .getOrCreate()
    
    # Configurações adicionais após a criação da sessão
    spark.conf.set("spark.sql.execution.arrow.pyspark.enabled", "true")
    spark.conf.set("spark.sql.execution.arrow.maxRecordsPerBatch", "10000")
    spark.conf.set("spark.sql.shuffle.partitions", "8")
    spark.conf.set("spark.default.parallelism", "8")
    
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
def get_cold_start_analysis():
    return spark.sql("""
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
            CAST((COUNT(*) * 100.0 / SUM(COUNT(*)) OVER ()) AS DECIMAL(10,2)) as percentual
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
def get_temporal_distribution():
    return spark.sql("""
        SELECT 
            DAYOFWEEK(timestampHistory) as dia_semana,
            HOUR(timestampHistory) as hora,
            COUNT(DISTINCT userId) as num_usuarios,
            i.page as categoria
        FROM tab_treino t
        JOIN tab_itens i ON t.history = i.page
        GROUP BY 
            DAYOFWEEK(timestampHistory),
            HOUR(timestampHistory),
            i.page
        ORDER BY dia_semana, hora
    """).toPandas()

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

# @st.cache_data
# def get_content_similarity():
#     return spark.sql("""
#         WITH filtered_articles AS (
#             -- Pré-filtrar artigos com número mínimo de leitores
#             SELECT history, COUNT(DISTINCT userId) as reader_count
#             FROM tab_treino
#             GROUP BY history
#             HAVING COUNT(DISTINCT userId) >= 10
#         ),
#         sampled_interactions AS (
#             -- Amostrar apenas uma parte dos dados para análise
#             SELECT t.userId, t.history
#             FROM tab_treino t
#             JOIN filtered_articles fa ON t.history = fa.history
#             WHERE userId IN (
#                 SELECT DISTINCT userId 
#                 FROM tab_treino 
#                 GROUP BY userId 
#                 HAVING COUNT(*) >= 5
#                 LIMIT 10000
#             )
#         ),
#         article_pairs AS (
#             -- Calcular co-ocorrências com dados amostrados
#             SELECT 
#                 a.history as article1,
#                 b.history as article2,
#                 COUNT(DISTINCT a.userId) as co_occurrences
#             FROM sampled_interactions a
#             JOIN sampled_interactions b 
#                 ON a.userId = b.userId 
#                 AND a.history < b.history
#             GROUP BY a.history, b.history
#             HAVING COUNT(DISTINCT a.userId) >= 3
#         )
#         SELECT 
#             ap.article1,
#             ap.article2,
#             ap.co_occurrences,
#             fa1.reader_count as readers_article1,
#             fa2.reader_count as readers_article2,
#             CAST(ap.co_occurrences / SQRT(fa1.reader_count * fa2.reader_count) AS DOUBLE) as similarity_score
#         FROM article_pairs ap
#         JOIN filtered_articles fa1 ON ap.article1 = fa1.history
#         JOIN filtered_articles fa2 ON ap.article2 = fa2.history
#         ORDER BY similarity_score DESC
#         LIMIT 20
#     """).toPandas()

@st.cache_data
def get_reading_sequence():
    return spark.sql("""
        WITH ordered_reads AS (
            SELECT 
                userId,
                history,
                timestampHistory,
                LAG(history) OVER (PARTITION BY userId ORDER BY timestampHistory) as prev_article
            FROM tab_treino
        )
        SELECT 
            prev_article,
            history as next_article,
            COUNT(*) as frequency
        FROM ordered_reads
        WHERE prev_article IS NOT NULL
        GROUP BY prev_article, history
        HAVING COUNT(*) > 10
        ORDER BY frequency DESC
        LIMIT 15
    """).toPandas()

@st.cache_data
def get_seasonality_analysis():
    return spark.sql("""
        SELECT 
            DAYOFWEEK(timestampHistory) as dia_semana,
            HOUR(timestampHistory) as hora,
            COUNT(*) as total_leituras,
            COUNT(DISTINCT userId) as usuarios_unicos
        FROM tab_treino
        GROUP BY DAYOFWEEK(timestampHistory), HOUR(timestampHistory)
        ORDER BY dia_semana, hora
    """).toPandas()

def load_data_sequential() -> Dict[str, Any]:
    data_cache = {}
    
    # Placeholder para mostrar progresso
    progress_placeholder = st.empty()
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Lista de todas as funções de carregamento de dados em ordem de prioridade
    data_loaders = [
        ('basic_metrics', lambda: get_basic_metrics(), 'Métricas Básicas'),
        ('user_distribution', lambda: get_user_distribution(), 'Distribuição de Usuários'),
        ('cold_start', lambda: get_cold_start_analysis(), 'Análise de Cold Start'),
        ('recency', lambda: get_recency_analysis(), 'Análise de Recência'),
        ('content_analysis', lambda: get_content_analysis(), 'Análise de Conteúdo'),
        ('temporal_dist', lambda: get_temporal_distribution(), 'Distribuição Temporal'),
        ('engagement', lambda: get_engagement_metrics(), 'Métricas de Engajamento'),
        ('hourly_pattern', lambda: get_hourly_pattern(), 'Padrões por Hora'),
        ('top_categories', lambda: get_top_categories(), 'Top Categorias'),
        ('correlation', lambda: get_correlation_metrics(), 'Correlações'),
        ('seasonality', lambda: get_seasonality_analysis(), 'Sazonalidade'),
        ('reading_sequence', lambda: get_reading_sequence(), 'Sequências de Leitura')
    ]
    
    total_queries = len(data_loaders)
    
    for idx, (name, func, description) in enumerate(data_loaders, 1):
        try:
            # Atualizar status
            progress = idx / total_queries
            progress_bar.progress(progress)
            status_text.text(f"Carregando {description}... ({idx}/{total_queries})")
            
            # Executar query
            result = func()
            data_cache[name] = result
            
        except Exception as e:
            st.error(f"Erro ao carregar {description}: {str(e)}")
    
    # Limpar elementos de progresso
    progress_placeholder.empty()
    progress_bar.empty()
    status_text.empty()
    
    return data_cache

def main():
    # Configuração inicial do Spark
    try:
        global spark, data_cache  # Tornar data_cache global também
        spark = init_spark()
        print("Conexão com Spark estabelecida com sucesso")
    except Exception as e:
        st.error(f"❌ Erro ao conectar com Spark: {str(e)}")
        st.stop()
    
    # Configurar o layout principal
    st.sidebar.title("Analise Exploratoria de Dados")
    
    # Menu de navegação
    page = st.sidebar.radio(
        "Navegação",
        options=["Início", "Visão Geral", "Perfil dos Usuários", "Cold Start", 
                "Recência e Engajamento", "Análise de Conteúdo", 
                "Distribuição Temporal", "Padrões Avançados", "Conclusão"]
    )
    
    # Carregar dados sequencialmente com feedback visual
    with st.spinner('Inicializando análise de dados...'):
        data_cache = load_data_sequential()
        
        # Verificar dados essenciais
        required_data = ['basic_metrics', 'user_distribution', 'cold_start']
        missing_data = [key for key in required_data if key not in data_cache]
        
        if missing_data:
            st.error(f"Falha ao carregar dados essenciais: {', '.join(missing_data)}")
            st.stop()

    # Título principal apenas na página inicial
    if page == "Início":
        st.title("📊 Dashboard G1 - Sistema de Recomendação")
    
    # Execução da página selecionada
    if page == "Início":
        show_home()
    elif page == "Visão Geral":
        show_visao_geral(data_cache)
    elif page == "Perfil dos Usuários":
        show_perfil_usuarios(data_cache)
    elif page == "Cold Start":
        show_cold_start(data_cache)
    elif page == "Recência e Engajamento":
        show_recencia_engajamento(data_cache)
    elif page == "Análise de Conteúdo":
        show_analise_conteudo(data_cache)
    elif page == "Distribuição Temporal":
        show_temporal_distribution(data_cache)
    elif page == "Conclusão":
        show_conclusion()
    elif page == "Padrões Avançados":
        show_advanced_patterns(data_cache)

def show_objective(text):
    """Exibe o objetivo da seção atual."""
    st.markdown(f"#### Objetivo\n{text}")

def show_visao_geral(data_cache):
    """Mostra a visão geral do sistema de recomendação."""
    metrics = data_cache.get('basic_metrics')
    if metrics is None:
        st.error("Dados básicos não disponíveis")
        return
    
    st.subheader("Visão Geral")
    
    show_objective("""
    Fornecer uma visão abrangente do sistema de recomendação, apresentando métricas-chave 
    de engajamento, distribuição de usuários e principais indicadores de desempenho.
    """)
    
    # Métricas principais
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total de Usuários", f"{metrics['total_users']:,}")
    with col2:
        st.metric("Total de Interações", f"{metrics['total_interactions']:,}")
    with col3:
        st.metric("Média Interações/Usuário", f"{metrics['avg_interactions']:,}")
    with col4:
        st.metric("Total de Notícias", f"{metrics['total_news']:,}")
    
    # Visualizações
    st.subheader("📊 Distribuição de Interações")
    
    user_dist = data_cache.get('user_distribution')
    if user_dist is not None and not user_dist.empty:
        fig = px.bar(
            user_dist,
            x='userType',
            y='unique_users',
            title='Distribuição de Usuários por Tipo',
            color_discrete_sequence=[THEME_COLORS['primary']]
        )
        fig.update_layout(
            plot_bgcolor=THEME_COLORS['background'],
            paper_bgcolor=THEME_COLORS['background'],
            font=dict(color=THEME_COLORS['text'])
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Novos insights mais detalhados
    st.markdown(f"""
    ### 📈 Insights Principais
    - **Volume de Dados**: Base com {metrics['total_users']:,} usuários ativos e {metrics['total_interactions']:,} interações
    - **Engajamento**: Média de {metrics['avg_interactions']:,} interações por usuário
    - **Diversidade de Conteúdo**: {metrics['total_news']:,} notícias diferentes consumidas
    - **Oportunidades**:
        - Personalização baseada no histórico de interações
        - Segmentação por padrões de consumo
        - Otimização da distribuição de conteúdo
    """)

def show_perfil_usuarios(data_cache):
    st.subheader("👥 Análise do Perfil dos Usuários")
    
    show_objective("""
    Compreender os diferentes perfis de usuários, seus padrões de comportamento e preferências, 
    visando melhorar a segmentação e personalização das recomendações.
    """)
    
    user_dist = data_cache.get('user_distribution')
    if user_dist is None or user_dist.empty:
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
        engagement_metrics = data_cache['engagement']
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
    hourly_pattern = data_cache['hourly_pattern']
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

    # Novos insights mais estruturados
    st.markdown("""
    ### 👥 Insights sobre os Usuários
    - **Segmentação de Perfis**:
        - Identificados padrões distintos de consumo por tipo de usuário
        - Variação significativa no tempo médio de leitura
        - Diferentes níveis de engajamento por segmento

    - **Comportamento Temporal**:
        - Picos de acesso em horários comerciais
        - Padrões distintos entre dias úteis e finais de semana
        - Oportunidades de engajamento em horários específicos

    - **Métricas de Engajamento**:
        - Correlação entre tempo de leitura e scroll
        - Diferentes padrões de navegação por perfil
        - Identificação de usuários mais engajados

    - **Recomendações**:
        - Personalização por segmento de usuário
        - Adaptação do conteúdo ao horário de acesso
        - Estratégias específicas por perfil de engajamento
    """)

def show_cold_start(data_cache):
    st.subheader("🆕 Análise de Cold Start")
    
    show_objective("""
    Analisar o desafio de novos usuários e usuários com poucas interações, buscando 
    estratégias efetivas para melhorar a experiência inicial e aumentar o engajamento.
    """)
    
    cold_start = data_cache['cold_start']
    
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
    
    # Calcular percentuais para insights
    total_users = cold_start['num_users'].sum()
    low_interactions = cold_start[cold_start['nivel_interacao'] == 'Muito Baixo (< 5)']['num_users'].iloc[0]
    high_interactions = cold_start[cold_start['nivel_interacao'] == 'Alto (20+)']['num_users'].iloc[0]
    
    low_percent = (low_interactions / total_users) * 100
    high_percent = (high_interactions / total_users) * 100
    
    st.markdown(f"""
    ### 🔍 Análise do Cold Start
    - **Distribuição de Interações**:
        - {low_percent:.1f}% dos usuários têm menos de 5 interações
        - {high_percent:.1f}% são usuários altamente ativos (20+ interações)
        - Desafio crítico com novos usuários

    - **Desafios Identificados**:
        - Baixa retenção inicial de novos usuários
        - Limitação de dados para personalização
        - Necessidade de engajamento rápido

    - **Estratégias Propostas**:
        1. **Recomendações Iniciais**:
            - Conteúdo mais popular da plataforma
            - Tendências atuais e trending topics
            - Mix de categorias para descoberta de interesses

        2. **Coleta de Informações**:
            - Interesses básicos no cadastro
            - Preferências de categorias
            - Horários preferenciais de leitura

        3. **Engajamento Progressivo**:
            - Feedback rápido sobre recomendações
            - Gamificação das primeiras interações
            - Personalização gradual do conteúdo
    """)

def show_recencia_engajamento(data_cache):
    st.subheader("📊 Análise de Recência e Engajamento")
    
    show_objective("""
    Avaliar os padrões de recência nas interações dos usuários e seus níveis de engajamento, 
    identificando oportunidades para retenção e reativação de usuários.
    """)
    
    recency = data_cache['recency']
    
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
    
    # Calcular métricas para insights
    total_users = recency['num_users'].sum()
    active_week = recency[recency['periodo'] == '1 semana']['num_users'].iloc[0]
    week_percent = (active_week / total_users) * 100
    
    st.markdown(f"""
    ### ⏰ Análise de Recência e Engajamento

    - **Padrões de Atividade**:
        - {week_percent:.1f}% dos usuários ativos na última semana
        - Forte correlação entre recência e engajamento
        - Ciclos claros de engajamento identificados

    - **Comportamento Temporal**:
        - Picos de atividade em horários específicos
        - Padrões semanais de engajamento
        - Sazonalidade no consumo de conteúdo

    - **Métricas de Retenção**:
        - Taxa de retorno por segmento
        - Tempo médio entre interações
        - Durabilidade do engajamento

    - **Estratégias Recomendadas**:
        1. **Conteúdo**:
            - Priorização de notícias recentes
            - Mix entre trending e personalizado
            - Adaptação ao contexto temporal

        2. **Retenção**:
            - Notificações personalizadas
            - Reengajamento de inativos
            - Campanhas baseadas em recência

        3. **Otimização**:
            - Timing das recomendações
            - Balanceamento de conteúdo
            - Personalização por padrão de uso
    """)

def show_analise_conteudo(data_cache):
    st.subheader("📰 Análise de Conteúdo")
    
    show_objective("""
    Examinar o desempenho e impacto de diferentes tipos de conteúdo, identificando 
    padrões de consumo e preferências para otimizar as recomendações.
    """)
    
    top_cats = data_cache['top_categories']
    
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
    correlation = data_cache['correlation']
    
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

def show_temporal_distribution(data_cache):
    st.subheader("📈 Distribuição Temporal das Interações")
    
    show_objective("""
    Analisar os padrões de consumo de conteúdo ao longo da semana e horários do dia, 
    identificando correlações entre categorias específicas e momentos de maior engajamento.
    """)
    
    temporal_dist = data_cache['temporal_dist']
    
    if temporal_dist.empty:
        st.warning("Nenhum dado disponível para a distribuição temporal.")
        return
    
    # Calcular total de usuários por categoria
    categoria_counts = temporal_dist.groupby('categoria')['num_usuarios'].sum().sort_values(ascending=False)
    
    # Criar seletor de categoria com opção "Todas" e ordenado por número de usuários
    categorias = ["Todas"] + list(categoria_counts.index)
    categoria_selecionada = st.selectbox(
        "Selecione a categoria de conteúdo:",
        options=categorias,
        index=0  # Começa com "Todas" selecionado
    )
    
    # Filtrar dados pela categoria selecionada (ou não)
    if categoria_selecionada == "Todas":
        df_filtered = temporal_dist
        titulo = 'Distribuição de Usuários por Dia e Hora: Todas as Categorias'
    else:
        df_filtered = temporal_dist[temporal_dist['categoria'] == categoria_selecionada]
        titulo = f'Distribuição de Usuários por Dia e Hora: {categoria_selecionada}'
    
    # Criar scatter plot
    fig = px.scatter(
        df_filtered,
        x='dia_semana',
        y='hora',
        size='num_usuarios',  # Tamanho dos pontos baseado no número de usuários
        color='categoria' if categoria_selecionada == "Todas" else 'num_usuarios',  # Cor por categoria quando mostrar todas
        title=titulo,
        labels={
            'dia_semana': 'Dia da Semana',
            'hora': 'Hora do Dia',
            'num_usuarios': 'Número de Usuários Únicos',
            'categoria': 'Categoria'
        },
        hover_data={
            'dia_semana': False,  # Não mostrar o número do dia
            'hora': True,
            'num_usuarios': True,
            'categoria': True if categoria_selecionada == "Todas" else False
        }
    )
    
    # Personalizar o layout
    fig.update_layout(
        plot_bgcolor=THEME_COLORS['background'],
        paper_bgcolor=THEME_COLORS['background'],
        font=dict(color=THEME_COLORS['text']),
        xaxis=dict(
            ticktext=['Domingo', 'Segunda', 'Terça', 'Quarta', 'Quinta', 'Sexta', 'Sábado'],
            tickvals=[1, 2, 3, 4, 5, 6, 7],
            gridcolor='rgba(128, 128, 128, 0.2)',
            title_font=dict(size=14)
        ),
        yaxis=dict(
            ticktext=[f'{i:02d}:00' for i in range(24)],
            tickvals=list(range(24)),
            gridcolor='rgba(128, 128, 128, 0.2)',
            title_font=dict(size=14)
        ),
        coloraxis_colorbar_title='Número de Usuários' if categoria_selecionada != "Todas" else 'Categoria',
        showlegend=True,
        height=700  # Aumentar altura do gráfico para melhor visualização
    )
    
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    ### 📊 Insights sobre Padrões Temporais
    
    - **Padrões por Categoria**:
        - Diferentes categorias mostram padrões únicos de consumo
        - Horários de pico variam por tipo de conteúdo
        - Comportamentos distintos entre dias úteis e fins de semana
    
    - **Comportamento dos Usuários**:
        - Preferências claras por horários específicos
        - Variação significativa no engajamento ao longo do dia
        - Padrões consistentes por categoria
    
    - **Oportunidades Identificadas**:
        1. **Timing de Publicação**:
            - Alinhar publicações com picos de audiência
            - Programar conteúdo baseado em padrões históricos
            - Otimizar notificações por categoria
    
        2. **Personalização Temporal**:
            - Recomendar conteúdo baseado no horário
            - Adaptar mix de categorias ao momento do dia
            - Considerar contexto temporal nas recomendações
    
        3. **Estratégias de Engajamento**:
            - Identificar janelas de oportunidade por categoria
            - Desenvolver estratégias específicas por período
            - Maximizar alcance em horários de pico
    """)

def show_advanced_patterns(data_cache):
    st.subheader("🔄 Padrões Avançados de Leitura")
    
    show_objective("""
    Análise aprofundada dos padrões de leitura para identificar comportamentos sazonais 
    e sequências de consumo de conteúdo, visando melhorar as recomendações.
    """)
    
    # Usar dados do cache
    seasonality = data_cache['seasonality']
    sequences = data_cache['reading_sequence']
    
    # Mapa de calor de sazonalidade
    st.subheader("📊 Padrão de Leitura por Dia e Hora")
    fig_heat = px.density_heatmap(
        seasonality,
        x='hora',
        y='dia_semana',
        z='total_leituras',
        title='Distribuição de Leituras ao Longo da Semana',
        labels={
            'hora': 'Hora do Dia',
            'dia_semana': 'Dia da Semana',
            'total_leituras': 'Volume de Leituras'
        }
    )
    fig_heat.update_layout(
        plot_bgcolor=THEME_COLORS['background'],
        paper_bgcolor=THEME_COLORS['background'],
        font=dict(color=THEME_COLORS['text'])
    )
    st.plotly_chart(fig_heat, use_container_width=True)
    
    # Insights sobre sazonalidade
    st.markdown("""
    ### 📅 Insights sobre Padrões Temporais
    - **Picos de Atividade**: Maior volume de leituras durante horários comerciais (10h-15h)
    - **Dias Úteis vs. Fim de Semana**: Padrão distinto de consumo entre dias da semana
    - **Períodos de Baixa**: Menor atividade durante madrugada (0h-5h)
    - **Oportunidades**:
        - Programar envio de recomendações antes dos horários de pico
        - Adaptar conteúdo recomendado conforme período do dia
        - Estratégias específicas para aumentar engajamento em períodos de baixa
    """)
    
    # Visualização de sequência de leitura
    st.subheader("🔄 Sequências de Leitura Mais Comuns")
    fig_seq = px.bar(
        sequences.head(10),
        x='frequency',
        y='prev_article',
        orientation='h',
        title='Top 10 Sequências de Leitura',
        labels={
            'frequency': 'Frequência',
            'prev_article': 'Artigo Anterior'
        }
    )
    fig_seq.update_layout(
        plot_bgcolor=THEME_COLORS['background'],
        paper_bgcolor=THEME_COLORS['background'],
        font=dict(color=THEME_COLORS['text'])
    )
    st.plotly_chart(fig_seq, use_container_width=True)
    
    # Insights sobre sequências
    st.markdown("""
    ### 🔄 Insights sobre Sequências de Leitura
    - **Padrões de Navegação**: Identificadas sequências frequentes de leitura entre artigos relacionados
    - **Conteúdo Âncora**: Alguns artigos funcionam como "hub", levando a múltiplas leituras subsequentes
    - **Comportamento do Usuário**:
        - Tendência a seguir temas relacionados em sequência
        - Forte correlação entre artigos de economia/concursos
    
    ### 💡 Recomendações Estratégicas
    1. **Personalização Temporal**:
        - Adaptar recomendações ao horário e dia da semana
        - Priorizar conteúdo relevante nos horários de pico
    
    2. **Sequenciamento Inteligente**:
        - Utilizar padrões de sequência para prever próximas leituras
        - Recomendar conteúdo baseado em caminhos de leitura comuns
    
    3. **Otimização de Conteúdo**:
        - Identificar e promover conteúdos "âncora"
        - Criar clusters de conteúdo baseados em padrões de sequência
    """)

def show_conclusion():
    st.subheader("🔍 Conclusão da Análise")
      
    st.markdown("""
    Após uma análise detalhada dos dados, chegamos às seguintes conclusões:

    - **Diversidade de Usuários**: Identificamos uma base diversificada de usuários com diferentes padrões de consumo, o que sugere a necessidade de personalização.
    - **Conteúdo Popular**: Certas categorias de conteúdo geram mais engajamento, indicando áreas de foco para futuras recomendações.
    - **Padrões Temporais**: Observamos padrões claros de interação ao longo do tempo, o que pode ser utilizado para otimizar o timing das recomendações.
    - **Desafios do Cold Start**: Estratégias específicas são necessárias para engajar novos usuários e melhorar sua experiência inicial.

    Continuaremos a monitorar e ajustar nosso sistema de recomendação com base nesses insights para oferecer uma experiência cada vez mais personalizada e eficaz.
    """)

def show_home():
    st.markdown("""
    Bem-vindo ao Dashboard G1, uma plataforma interativa para análise exploratória de dados do nosso sistema de recomendação. 
    Este dashboard foi desenvolvido para fornecer insights valiosos sobre o comportamento dos usuários e o desempenho do conteúdo.

    ### Objetivo:
    - **Entender o Perfil dos Usuários**: Identificar padrões de comportamento e segmentar usuários com base em suas interações.
    - **Analisar o Desempenho do Conteúdo**: Avaliar quais tipos de conteúdo geram mais engajamento e como os usuários interagem com eles.
    - **Explorar Padrões Temporais**: Descobrir tendências e sazonalidades nas interações dos usuários ao longo do tempo.
    - **Abordar o Desafio do Cold Start**: Desenvolver estratégias para melhorar a experiência de novos usuários com base em dados limitados.

    Navegue pelas diferentes seções para explorar os dados e descobrir insights que podem ajudar a otimizar nosso sistema de recomendação.
    """)

if __name__ == "__main__":
    main() 