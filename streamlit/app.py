from pyspark.sql import SparkSession
from pyspark.sql.functions import countDistinct, when, col, explode, split, corr, expr, rand, percentile_approx, avg
import plotly.express as px
import pandas as pd
import streamlit as st
from streamlit_option_menu import option_menu
import plotly.graph_objects as go
import numpy as np
import time
import json
import os
from datetime import datetime

st.set_page_config(
    page_title="Análise Exploratória - Recomendador de Notícias",
    page_icon="",
    layout="wide"
)

@st.cache_resource
def init_spark():
    spark = SparkSession.builder.appName("NewsRecommenderEDA") \
            .config("spark.sql.parquet.datetimeRebaseModeInRead", "CORRECTED") \
            .config("spark.sql.parquet.datetimeRebaseModeInWrite", "CORRECTED") \
            .config("spark.driver.memory", "8g") \
            .config("spark.executor.memory", "8g") \
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
    return spark


def load_data(spark):
    treino = spark.read \
        .option("mergeSchema", "false") \
        .parquet("datalake/silver/treino") \
        .coalesce(2)
    
    itens = spark.read \
        .option("mergeSchema", "false") \
        .parquet("datalake/silver/itens") \
        .coalesce(2)
    
    return treino, itens

@st.cache_data
def get_news_per_user(_spark):
    """Cache para os dados da Análise 1"""
    return _spark.sql("""
        SELECT userId, COUNT(history) as news_count
        FROM tab_treino
        GROUP BY userId
    """).toPandas()

@st.cache_data
def get_time_distribution(_spark):
    """Cache para os dados da Análise 2"""
    time_dist = _spark.sql("""
        SELECT hour, COUNT(*) as count
        FROM tab_treino
        GROUP BY hour
        ORDER BY hour
    """).toPandas()
    
    time_week_dist = _spark.sql("""
        SELECT dayofweek, hour, COUNT(*) as count
        FROM tab_treino
        GROUP BY dayofweek, hour
        ORDER BY dayofweek, hour
    """).toPandas()
    
    time_series = _spark.sql("""
        SELECT year, month, day, COUNT(*) as count
        FROM tab_treino
        GROUP BY year, month, day
        ORDER BY year, month, day
    """).toPandas()
    
    time_series['year_month_day'] = time_series['year'].astype(str) + '-' + \
                                   time_series['month'].astype(str) + '-' + \
                                   time_series['day'].astype(str)
    
    return time_dist, time_week_dist, time_series

@st.cache_data
def get_engagement_data(_spark):
    """Cache para os dados da Análise 3"""
    engagement_df = _spark.sql("""
        SELECT history,
            timeOnPageHistory, 
            numberOfClicksHistory, 
            scrollPercentageHistory, 
            interaction_score
        FROM tab_treino
    """)

    engagement_df_grouped = engagement_df.groupBy("history").agg(
        avg("timeOnPageHistory").alias("avg_timeOnPageHistory"),
        avg("numberOfClicksHistory").alias("avg_numberOfClicksHistory"),
        avg("scrollPercentageHistory").alias("avg_scrollPercentageHistory"),
        avg("interaction_score").alias("avg_interaction_score")
    )

    filtered_df_grouped = engagement_df_grouped.filter(
        (col("avg_timeOnPageHistory") > 0) & 
        (col("avg_timeOnPageHistory") < 500000) & 
        (col("avg_numberOfClicksHistory") < 500)  
    )

    return filtered_df_grouped.sample(fraction=0.1).toPandas()

@st.cache_data
def get_retention_data(_spark):
    """Cache para os dados da Análise 4"""
    retention_df = _spark.sql("""
        SELECT year, month, day, userId, COUNT(*) as visits
        FROM tab_treino
        GROUP BY year, month, day, userId
    """).toPandas()

    time_retention_df = _spark.sql("""
        SELECT year, month, day, COUNT(DISTINCT userId) as unique_users
        FROM tab_treino
        GROUP BY year, month, day
        ORDER BY year, month, day
    """).toPandas()

    time_retention_df["date"] = pd.to_datetime(
        time_retention_df[["year", "month", "day"]].assign(
            year=lambda x: x["year"].astype(str),
            month=lambda x: x["month"].astype(str).str.zfill(2),
            day=lambda x: x["day"].astype(str).str.zfill(2)
        ).agg("-".join, axis=1)
    )

    return retention_df, time_retention_df

@st.cache_data
def get_user_type_data(_spark):
    """Cache para os dados da Análise 5"""
    return _spark.sql("""
        SELECT 
            CONCAT(year, '-', LPAD(month, 2, '0'), '-', LPAD(day, 2, '0')) AS date, 
            userType, 
            COUNT(*) AS count
        FROM tab_treino
        GROUP BY year, month, day, userType
        ORDER BY date
    """).toPandas()

@st.cache_data
def get_recency_data(_spark):
    """Cache para os dados da Análise 6"""
    return _spark.sql("""
        SELECT i.year as item_year, i.month as item_month, i.day as item_day, 
               i.days_since_published, COUNT(*) as access_count
        FROM tab_treino AS t
        JOIN tab_itens AS i
        ON t.history = i.page
        GROUP BY i.year, i.month, i.day, i.days_since_published
    """).toPandas()

@st.cache_data
def get_news_overlap_data(_spark):
    """Cache para os dados da Análise 7"""
    return _spark.sql("""
        SELECT t.history AS news_id, COUNT(DISTINCT t.userId) AS user_count
        FROM tab_treino AS t
        GROUP BY t.history
    """).toPandas()

@st.cache_data
def get_news_popularity_data(_spark):
    """Cache para os dados de popularidade das notícias (Análise 7 e 8)"""

    news_popularity_df = _spark.sql("""
        SELECT userId, split(history, ',') AS news_array
        FROM tab_treino
    """).withColumn("news_id", explode(col("news_array")))

    news_popularity_df = news_popularity_df.groupBy("news_id") \
        .agg(countDistinct("userId").alias("unique_users"))

    return news_popularity_df.toPandas()

@st.cache_data
def get_clicks_time_correlation(_spark):
    """Cache para os dados da Análise 9"""
    correlation = _spark.sql("""
        SELECT corr(numberOfClicksHistory, timeOnPageHistory) as correlation 
        FROM tab_treino
    """).collect()[0]["correlation"]
    
    sample_df = _spark.sql("""
        SELECT numberOfClicksHistory, timeOnPageHistory 
        FROM tab_treino
    """).sample(fraction=0.1, seed=42).toPandas()
    
    return correlation, sample_df

@st.cache_data
def get_user_interactions_data(_spark):
    """Cache para os dados da Análise 10"""
    return _spark.sql("""
        SELECT userId, COUNT(history) AS interaction_count
        FROM tab_treino
        GROUP BY userId
    """).toPandas()

@st.cache_data
def get_news_distribution_data(_spark):
    """Cache para os dados da Análise 8"""
    news_popularity_df = _spark.sql("""
        SELECT history AS news_id, COUNT(DISTINCT userId) AS unique_users
        FROM tab_treino
        GROUP BY history
    """)

    news_popularity_df = news_popularity_df.withColumn(
        "user_bins",
        when(col("unique_users") <= 10, "1-10")
        .when((col("unique_users") > 10) & (col("unique_users") <= 100), "11-100")
        .when((col("unique_users") > 100) & (col("unique_users") <= 1000), "101-1000")
        .when((col("unique_users") > 1000) & (col("unique_users") <= 10000), "1001-10000")
        .otherwise("10001+")
    )

    return news_popularity_df.toPandas()

@st.cache_data
def load_monitoring_results():
    """Carrega os resultados mais recentes do monitoramento"""
    try:
        monitoring_path = "/app/models/monitoring"
        
        if not os.path.exists(monitoring_path):
            return None
        
        monitoring_files = [f for f in os.listdir(monitoring_path) 
                          if f.startswith('monitoring_results_') and f.endswith('.json')]
        
        if not monitoring_files:
            return None
        
        latest_file = max(monitoring_files)
        file_path = os.path.join(monitoring_path, latest_file)
        
        with open(file_path, 'r') as f:
            return json.load(f)
            
    except Exception as e:
        return None

def plot_feature_importance(feature_data):
    """Plota o gráfico de importância das features"""
    df = pd.DataFrame({
        'Feature': [f"Feature {i}" for i in range(len(feature_data))],
        'Importance': list(feature_data.values())
    }).sort_values('Importance', ascending=False).head(20)

    fig = px.bar(df, 
                 x='Feature', 
                 y='Importance',
                 title='Top 20 Features Mais Importantes',
                 color='Importance',
                 color_continuous_scale='viridis')
    
    fig.update_layout(
        xaxis_title="Features",
        yaxis_title="Importância Relativa",
        showlegend=False
    )
    
    return fig

def plot_interaction_metrics(interaction_data):
    """Plota métricas de interação"""
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=['Total', 'Treino', 'Teste'],
        y=[interaction_data['total_interactions'],
           interaction_data['train_interactions'],
           interaction_data['test_interactions']],
        name='Número de Interações'
    ))
    
    fig.update_layout(
        title='Distribuição de Interações',
        yaxis_title='Número de Interações',
        showlegend=True
    )
    
    return fig

def show_analysis_1(spark):  
    st.markdown("<h1 style='font-size: 32px;'>Análise 1: Distribuição do número de notícias lidas por usuário</h1>", unsafe_allow_html=True)
    
    st.markdown("------------------------------------------------------------")
    
    st.markdown("<h1 style='font-size: 26px;'>Objetivos</h1>", unsafe_allow_html=True)
    
    st.markdown("""
    - Compreender o padrão de consumo de notícias pelos usuários
    - Identificar se a distribuição do consumo é equilibrada ou se há concentração em poucos usuários
    - Detectar possíveis outliers, como usuários altamente engajados ou bots
    - Ajudar a definir estratégias diferenciadas de recomendação para usuários casuais e frequentes
    ------------------------------------------------------------
    """)
    st.markdown("<h1 style='font-size: 26px;'>Histograma da Distribuição de Notícias Lidas por Usuário</h1>", unsafe_allow_html=True)

    news_per_user = get_news_per_user(spark)

    fig_hist = px.histogram(news_per_user, x='news_count', nbins=30)
    st.plotly_chart(fig_hist, use_container_width=True)

    st.markdown("""
    **Observações:**
    - A grande maioria dos usuários lê poucas notícias, o que indica que o consumo é altamente concentrado
    - Há uma cauda longa na distribuição, com alguns usuários lendo milhares de notícias
    - Isso pode indicar a presença de usuários extremamente engajados ou até mesmo bots

    **O que isso significa para o modelo?**
    - A recomendação personalizada pode ser mais relevante para usuários que consomem muitas notícias, pois há mais dados sobre suas preferências
    - Para usuários casuais, recomendações baseadas em popularidade ou tendências podem ser mais eficazes

    **Ação recomendada:**
    - Separar os usuários em grupos (casuais, medianos e altamente engajados) para testar recomendações diferenciadas
    - Filtrar possíveis bots ao identificar usuários com consumo anormalmente alto
    - Criar estratégias para engajar usuários com poucos acessos, oferecendo recomendações mais diversificadas ou guiadas por tendências
    ------------------------------------------------------------
    """)

    st.markdown("<h1 style='font-size: 26px;'>Distribuição de Notícias Lidas por Usuário (Boxplot)</h1>", unsafe_allow_html=True)

    fig_box = px.box(news_per_user, y='news_count')
    st.plotly_chart(fig_box, use_container_width=True)

    st.markdown("""
    **Observações:**
    - O boxplot evidencia a presença de outliers extremos, que se destacam do restante da distribuição
    - A grande maioria dos usuários consome um número pequeno de notícias, enquanto alguns poucos consomem milhares

    **O que isso significa para o modelo?**
    - Como os outliers podem distorcer métricas médias e padrões de recomendação, é importante tratá-los adequadamente
    - O modelo pode precisar de pesos diferentes para usuários casuais e altamente engajados

    **Ação recomendada:**
    - Remover outliers extremos ou tratá-los separadamente para evitar distorções
    - Criar um modelo híbrido, onde a recomendação para usuários frequentes seja altamente personalizada e a recomendação para novos usuários seja baseada em popularidade
    - Considerar limites superiores para o número de notícias lidas ao calcular estatísticas médias
    ------------------------------------------------------------
    """)
    st.markdown("<h1 style='font-size: 26px;'>Conclusões Finais</h1>", unsafe_allow_html=True)

    st.markdown(
        """
    - O comportamento de consumo é muito desigual, exigindo abordagens diferentes para diferentes perfis de usuários.
    - Modelos baseados em popularidade podem ser úteis para novos usuários, enquanto modelos mais personalizados beneficiam os usuários mais engajados.
    - O tratamento de outliers e segmentação de usuários pode melhorar a precisão e a relevância das recomendações.
    ------------------------------------------------------------
    """)

def show_analysis_2(spark):
    st.markdown("<h1 style='font-size: 32px;'>Análise 2: Distribuição temporal das interações dos usuários</h1>", unsafe_allow_html=True)

    st.markdown("------------------------------------------------------------")

    st.markdown("<h1 style='font-size: 26px;'>Objetivos</h1>", unsafe_allow_html=True)

    st.markdown(
        """
        - Identificar horários de pico de leitura e padrões sazonais.
        - Verificar diferenças no comportamento de leitura entre diferentes horários e dias da semana.
        - Fornecer insights para a introdução de features temporais no modelo de recomendação.
        - Entender a evolução das interações ao longo do tempo para identificar tendências.
        ------------------------------------------------------------
        """
    )
    st.markdown("<h1 style='font-size: 26px; ;'>Distribuição de Acessos por Hora do Dia</h1>", unsafe_allow_html=True)

    time_dist, time_week_dist, time_series = get_time_distribution(spark)

    fig_time = px.bar(time_dist, x='hour', y='count')
    st.plotly_chart(fig_time, use_container_width=True)

    st.markdown(
        """
        **Observações:**
        - O volume de acessos é menor na madrugada e atinge picos a partir das 10h, com um crescimento contínuo até o início da noite.
        - O maior volume de acessos ocorre entre 12h e 18h, indicando que este pode ser um período crítico para recomendações personalizadas.
        - O período da madrugada apresenta acessos mais baixos, sugerindo que a atividade dos usuários é mínima entre 2h e 6h.

        **O que isso significa para o modelo?**
        - Modelos baseados em recência podem precisar considerar a hora do dia para evitar recomendar conteúdos fora do horário de maior engajamento.
        - Recomendações feitas pela manhã podem se beneficiar de tendências do dia anterior, enquanto à noite podem ser baseadas no consumo do próprio dia.

        **Ação recomendada:**
        - Criar um fator de ajuste temporal para favorecer recomendações em horários de pico.
        - Testar modelos de recomendação que diferenciam usuários matutinos e noturnos.
        - Analisar se a taxa de conversão das recomendações varia ao longo do dia.
        ------------------------------------------------------------
        """
    )
    st.markdown("<h1 style='font-size: 26px; ;'>Acessos por Hora e Dia da Semana (Heatmap)</h1>", unsafe_allow_html=True)

    fig_heatmap = px.density_heatmap(time_week_dist, x='hour', y='dayofweek', z='count', color_continuous_scale='Blues')
    st.plotly_chart(fig_heatmap, use_container_width=True)

    st.markdown(
        """
        **Observações:**
        - Os acessos aumentam durante o horário comercial e início da noite, com picos mais evidentes nos dias úteis.
        - O fim de semana apresenta um padrão de acessos mais distribuído, sem picos tão intensos quanto os dias úteis.
        - O período da manhã durante os dias úteis tem um volume crescente de acessos, enquanto nos finais de semana esse crescimento é menos acentuado.

        **O que isso significa para o modelo?**
        - Usuários podem ter padrões de leitura distintos entre dias úteis e finais de semana.
        - Recomendações podem ser otimizadas levando em conta a sazonalidade do dia da semana.
        - Notícias mais acessadas durante a semana podem perder relevância no final de semana, sugerindo que a recência pode ter impacto diferenciado.

        **Ação recomendada:**
        - Criar features temporais no modelo de recomendação considerando o dia da semana e horário.
        - Testar se recomendações de tendências da semana funcionam no final de semana ou se precisam ser ajustadas.
        - Analisar se o engajamento do usuário varia conforme o dia e ajustar a estratégia de recomendação.
        ------------------------------------------------------------
        """
    )

    st.markdown("<h1 style='font-size: 26px; ;'>Evolução das Interações ao Longo do Tempo</h1>", unsafe_allow_html=True)

    fig_line = px.line(time_series, x='year_month_day', y='count')
    st.plotly_chart(fig_line, use_container_width=True)

    st.markdown(
        """
        **Observações:**
        - Há flutuações regulares no volume de interações, possivelmente refletindo um ciclo semanal de consumo de notícias.
        - Os acessos tendem a diminuir em alguns períodos específicos, sugerindo sazonalidade.
        - Alguns picos de interação podem estar associados a eventos de grande impacto.

        **O que isso significa para o modelo?**
        - A recência e a sazonalidade são fatores críticos para a recomendação de notícias.
        - Eventos sazonais podem influenciar fortemente o consumo de notícias, e o modelo deve ser capaz de se adaptar rapidamente.
        - Recomendações baseadas em tendências podem precisar de ajustes dependendo do dia da semana e período do mês.

        **Ação recomendada:**
        - Criar um mecanismo de ajuste dinâmico para recomendações baseadas na variação da demanda ao longo do tempo.
        - Explorar a inclusão de eventos sazonais no modelo de recomendação.
        - Monitorar a taxa de aceitação das recomendações ao longo do tempo para identificar padrões e possíveis melhorias.
        ------------------------------------------------------------
        """
    )

    st.markdown("<h1 style='font-size: 26px; ;'>Conclusões Finais</h1>", unsafe_allow_html=True)
    st.markdown(
        """
        - Horários e dias da semana influenciam fortemente o consumo de notícias, o que pode ser explorado no modelo.
        - A recência deve ser considerada em diferentes escalas temporais para manter a relevância das recomendações.
        - Modelos temporais podem melhorar a precisão das sugestões ao considerar padrões de engajamento diários e semanais.
        - A adaptação a eventos sazonais pode otimizar a experiência do usuário, garantindo que as recomendações permaneçam relevantes.
        ------------------------------------------------------------
        """
    )

def show_analysis_3(spark):
    st.markdown("<h1 style='font-size: 32px;'>Análise 3: Relação entre tempo na página e engajamento</h1>", unsafe_allow_html=True)

    st.markdown("------------------------------------------------------------")
    
    st.markdown("<h1 style='font-size: 26px;'>Objetivos</h1>", unsafe_allow_html=True)

    st.markdown("""
    - Avaliar se o tempo que um usuário passa em uma página está relacionado com seu nível de engajamento.
    - Identificar padrões entre tempo na página e diferentes métricas de interação, como número de cliques, porcentagem de scroll e score de interação.
    - Compreender se o tempo na página pode ser um indicador relevante para o modelo de recomendação.
    - Determinar se existem outliers ou comportamentos anômalos que devem ser tratados.
    ------------------------------------------------------------
    """)

    st.markdown("<h1 style='font-size: 26px;'>Relação entre Tempo na Página e Número de Cliques</h1>", unsafe_allow_html=True)

    sample_df_grouped = get_engagement_data(spark)
    sample_df_grouped['avg_timeOnPageHistory'] = sample_df_grouped['avg_timeOnPageHistory'] / 60000

    fig_scatter_clicks = px.scatter(
        sample_df_grouped, x='avg_timeOnPageHistory', y='avg_numberOfClicksHistory',
        labels={'avg_timeOnPageHistory': 'Média de Tempo na Página (minutos)', 
                'avg_numberOfClicksHistory': 'Média de Número de Cliques'},
        trendline='ols',
        trendline_color_override='red' 
    )
    st.plotly_chart(fig_scatter_clicks, use_container_width=True)

    st.markdown(
        """
        **Observações:**
        - Existe uma leve tendência de aumento no número de cliques conforme o tempo na página cresce.
        - No entanto, a dispersão dos dados é alta, o que sugere que o tempo na página, isoladamente, não determina o número de cliques.
        - Há alguns outliers com muitos cliques, possivelmente indicando páginas com conteúdo altamente interativo.

        **O que isso significa para o modelo?**
        - O número de cliques pode ser um fator de engajamento relevante, mas não é um indicador determinístico.
        - Pode ser necessário combinar essa métrica com outras para criar uma feature mais robusta.

        **Ação recomendada:**
        - Criar uma feature combinada entre tempo na página e cliques, normalizando os valores.
        - Investigar se há um limite de tempo além do qual os cliques não aumentam significativamente.
        ------------------------------------------------------------
        """)

    st.markdown("<h1 style='font-size: 26px;'>Relação entre Tempo na Página e Scroll (%)</h1>", unsafe_allow_html=True)

    fig_scatter_scroll = px.scatter(
        sample_df_grouped, x='avg_timeOnPageHistory', y='avg_scrollPercentageHistory',
        labels={'avg_timeOnPageHistory': 'Média de Tempo na Página (minutos)', 
                'avg_scrollPercentageHistory': 'Média de Porcentagem de Scroll'},
        trendline='ols',
        trendline_color_override='red' 
    )
    st.plotly_chart(fig_scatter_scroll, use_container_width=True)

    st.markdown(
        """
        **Observações:**
        - A maioria dos usuários parece realizar pouco scroll, independentemente do tempo que passam na página.
        - Alguns outliers indicam sessões com scroll muito alto, possivelmente erros de registro ou comportamentos específicos.

        **O que isso significa para o modelo?**
        - O scroll pode não ser um indicador confiável de engajamento, especialmente quando o tempo na página é longo, mas o scroll é baixo.
        - Pode haver casos em que os usuários deixam a página aberta sem interagir com ela.

        **Ação recomendada:**
        - Analisar se há um threshold mínimo de scroll para considerar a interação válida.
        - Combinar o scroll com outras métricas, como tempo médio de leitura e cliques.
        ------------------------------------------------------------
        """)

    st.markdown("<h1 style='font-size: 26px;'>Relação entre Tempo na Página e Score de Interação</h1>", unsafe_allow_html=True)

    fig_scatter_interaction = px.scatter(
        sample_df_grouped, x='avg_timeOnPageHistory', y='avg_interaction_score',
        labels={'avg_timeOnPageHistory': 'Média de Tempo na Página (minutos)', 
                'avg_interaction_score': 'Média de Score de Interação'},
        trendline='ols',
        trendline_color_override='red' 
    )
    st.plotly_chart(fig_scatter_interaction, use_container_width=True)

    st.markdown(
        """
        **Observações:**
        - A relação entre tempo na página e o score de interação é quase linear, sugerindo que quanto mais tempo um usuário passa na página, maior é seu score de interação.
        - Isso indica que o score de interação já pode estar incorporando o tempo de leitura como um fator relevante.

        **O que isso significa para o modelo?**
        - O score de interação parece ser um bom indicador de envolvimento do usuário com a notícia.
        - Esse score pode ser mais útil do que usar tempo na página ou cliques isoladamente.

        **Ação recomendada:**
        - Utilizar o score de interação como uma feature central no modelo de recomendação.
        - Testar o impacto de combinar essa métrica com cliques e tempo de leitura para melhorar a precisão das recomendações.
        ------------------------------------------------------------
        """)

    st.markdown("<h1 style='font-size: 26px;'>Conclusões Finais</h1>", unsafe_allow_html=True)

    st.markdown(
        """
        - O tempo na página tem correlação com o score de interação, mas não necessariamente com o número de cliques ou a porcentagem de scroll.
        - A porcentagem de scroll pode não ser um bom indicador isolado e pode precisar de ajustes no modelo.
        - Notícias com altos scores de interação devem ser priorizadas, pois refletem um engajamento mais realista.
        - O modelo pode se beneficiar de uma feature combinada que englobe tempo na página, cliques e score de interação.
        ------------------------------------------------------------
        """)

def show_analysis_4(spark):
    st.markdown("<h1 style='font-size: 32px;'>Análise 4: Taxa de retorno dos usuários ao longo do tempo</h1>", unsafe_allow_html=True)

    st.markdown("------------------------------------------------------------")

    st.markdown("<h1 style='font-size: 26px;'>Objetivos</h1>", unsafe_allow_html=True)

    st.markdown(
        """
        - Identificar a frequência com que os usuários retornam à plataforma.
        - Analisar padrões de retorno ao longo do tempo para entender tendências e sazonalidade.
        - Avaliar se há um padrão de fidelização dos usuários, diferenciando usuários casuais e recorrentes.
        - Fornecer insights para estratégias de retenção e engajamento no modelo de recomendação.
        ------------------------------------------------------------
        """
    )

    st.markdown("<h1 style='font-size: 26px;'>Distribuição da Taxa de Retorno dos Usuários</h1>", unsafe_allow_html=True)

    retention_df, time_retention_df = get_retention_data(spark)

    fig_hist_retention = px.histogram(
        retention_df, x="visits", nbins=30,
        labels={"visits": "Número de Retornos"}
    )
    st.plotly_chart(fig_hist_retention, use_container_width=True)

    st.markdown(
        """
        **Observações:**
        - A maioria dos usuários acessa a plataforma poucas vezes, com uma distribuição altamente concentrada nos primeiros retornos.
        - Há uma cauda longa indicando que um pequeno número de usuários retorna várias vezes, chegando a mais de 40 retornos.
        - Essa discrepância sugere a presença de dois perfis distintos: usuários casuais e usuários altamente engajados.

        **O que isso significa para o modelo?**
        - O modelo de recomendação pode beneficiar-se da diferenciação entre usuários casuais e recorrentes.
        - Usuários casuais podem receber recomendações baseadas em popularidade e tendências gerais.
        - Usuários recorrentes podem receber recomendações mais personalizadas, baseadas em histórico detalhado de navegação.

        **Ação recomendada:**
        - Criar segmentações de usuários com base no número de retornos para oferecer experiências diferenciadas.
        - Implementar estratégias para engajar usuários com poucos retornos e incentivar novas visitas.
        - Avaliar se padrões de retorno se correlacionam com outros fatores como tempo na página e número de cliques.
        ------------------------------------------------------------
        """
    )

    st.markdown("<h1 style='font-size: 26px;'>Evolução da Taxa de Retorno dos Usuários ao Longo do Tempo</h1>", unsafe_allow_html=True)

    fig_line_retention = px.line(
        time_retention_df, x="date", y="unique_users",
        labels={"date": "Data", "unique_users": "Usuários Únicos por Dia"}
    )
    st.plotly_chart(fig_line_retention, use_container_width=True)

    st.markdown(
        """
        **Observações:**
        - A taxa de retorno dos usuários segue um padrão cíclico, com quedas periódicas seguidas de aumentos bruscos.
        - Esses picos podem estar relacionados a eventos específicos, como notícias de grande repercussão que atraem mais usuários.
        - A tendência geral mostra uma variação significativa ao longo do tempo, indicando que o engajamento não é constante.

        **O que isso significa para o modelo?**
        - A sazonalidade pode impactar a eficiência do modelo de recomendação, pois a base de usuários ativos varia ao longo do tempo.
        - O modelo pode se beneficiar de features temporais para ajustar recomendações com base no momento da interação.
        - Estratégias de retenção podem ser reforçadas em períodos de baixa interação para evitar perda de usuários.

        **Ação recomendada:**
        - Analisar eventos e fatores que podem estar influenciando os picos de retorno.
        - Criar um sistema de recomendação dinâmico que se adapte a padrões sazonais de engajamento.
        - Desenvolver estratégias para manter usuários ativos mesmo em períodos de baixa interação.
        ------------------------------------------------------------
        """
    )

    st.markdown("<h1 style='font-size: 26px;'>Conclusões Finais</h1>", unsafe_allow_html=True)

    st.markdown(
        """
        **Impacto no modelo de recomendação:**

        - A retenção de usuários é um fator crítico para o sucesso da recomendação personalizada.
        - A segmentação entre usuários casuais e recorrentes pode melhorar a assertividade das recomendações.
        - Incorporar variáveis temporais e padrões sazonais pode tornar o modelo mais robusto e responsivo às mudanças no comportamento dos usuários.
        - Estratégias de engajamento devem ser direcionadas para aumentar a taxa de retorno, garantindo uma base de usuários ativa e crescente.
        ------------------------------------------------------------
        """
    )

def show_analysis_5(spark):
    st.markdown("<h1 style='font-size: 32px;'>Análise 5: Proporção de usuários logados vs. anônimos</h1>", unsafe_allow_html=True)

    st.markdown("------------------------------------------------------------")

    st.markdown("<h1 style='font-size: 26px;'>Objetivos</h1>", unsafe_allow_html=True)

    st.markdown(
        """
        - Compreender a distribuição entre usuários logados e anônimos ao longo do tempo.
        - Verificar se há mudanças sazonais na proporção de usuários logados e anônimos.
        - Avaliar o impacto desse fator na personalização das recomendações, visto que usuários logados possuem histórico mais completo.
        - Adaptar as estratégias de recomendação para cada perfil de usuário.
        ------------------------------------------------------------
        """
    )

    user_type_time_df = get_user_type_data(spark)

    user_type_time_df["date"] = pd.to_datetime(user_type_time_df["date"])

    color_map = {
        "Logged": "#1f77b4", 
        "Non-Logged": "#aec7e8"
    }

    st.markdown("<h1 style='font-size: 26px;'>Evolução da Proporção de Usuários Logados vs. Anônimos</h1>", unsafe_allow_html=True)

    fig_area_user_type = px.area(
        user_type_time_df, x="date", y="count", color="userType",
        labels={"userType": "Tipo de Usuário", "count": "Quantidade", "date": "Data"},
        color_discrete_map=color_map
    )
    st.plotly_chart(fig_area_user_type, use_container_width=True)

    st.markdown(
        """
        **Observações:**
        - A proporção de usuários logados e anônimos se mantém relativamente estável ao longo do tempo.
        - Pequenas oscilações podem indicar eventos ou fatores externos que afetam o login dos usuários.
        - A maior proporção de usuários anônimos pode impactar a qualidade da recomendação personalizada, pois há menos dados históricos disponíveis para esses usuários.

        **O que isso significa para o modelo de recomendação?**
        - Usuários logados têm um histórico de interações mais rico, permitindo recomendações mais personalizadas e sofisticadas.
        - Para usuários anônimos, pode ser necessário adotar abordagens baseadas em tendências e popularidade.
        - Estratégias como incentivo ao login podem melhorar a experiência do usuário e a eficácia do modelo de recomendação.

        **Ações recomendadas:**
        - Desenvolver estratégias híbridas: modelos personalizados para usuários logados e recomendações baseadas em popularidade para anônimos.
        - Criar incentivos para que mais usuários façam login, como recomendações exclusivas ou conteúdo personalizado.
        - Monitorar tendências que possam influenciar a taxa de login ao longo do tempo.
        ------------------------------------------------------------
        """
    )

    st.markdown("<h1 style='font-size: 26px;'>Evolução da Quantidade de Usuários Logados vs. Anônimos</h1>", unsafe_allow_html=True)

    fig_bar_user_type = px.bar(
        user_type_time_df, x="date", y="count", color="userType",
        labels={"userType": "Tipo de Usuário", "count": "Quantidade", "date": "Data"},
        barmode="stack",
        color_discrete_map=color_map
    )
    st.plotly_chart(fig_bar_user_type, use_container_width=True)

    st.markdown(
        """
        **Observações:**
        - O volume total de usuários apresenta flutuações ao longo do tempo.
        - O número de usuários anônimos é consistentemente maior do que o de usuários logados.
        - Picos e quedas na atividade podem indicar eventos sazonais, mudanças no tráfego do site ou fatores externos que influenciam o login.

        **O que isso significa para o modelo de recomendação?**
        - A predominância de usuários anônimos reforça a necessidade de recomendações baseadas em contexto, popularidade e tendências globais.
        - Eventos sazonais podem afetar padrões de login e interação, o que pode ser explorado para criar recomendações mais relevantes.
        - Notícias mais acessadas durante a semana podem perder relevância no final de semana, sugerindo que a recência pode ter impacto diferenciado.

        **Ações recomendadas:**
        - Investigar períodos de picos e quedas para entender o que impulsiona o login dos usuários.
        - Criar filtros ou categorias diferenciadas para recomendações baseadas em comportamento de usuários logados vs. anônimos.
        - Integrar dados temporais ao modelo para antecipar mudanças no padrão de login e consumo de conteúdo.
        ------------------------------------------------------------
        """
    )

    st.markdown("<h1 style='font-size: 26px;'>Conclusões Finais</h1>", unsafe_allow_html=True)

    st.markdown(
        """
        - A segmentação entre usuários logados e anônimos é essencial para ajustar estratégias de recomendação.
        - Estratégias híbridas podem melhorar a experiência de usuários com e sem login.
        - O incentivo ao login pode trazer benefícios tanto para a personalização quanto para o engajamento da plataforma.
        ------------------------------------------------------------
        """
    )

def show_analysis_6(spark):
    st.markdown("<h1 style='font-size: 32px;'>Análise 6: Padrões de consumo de notícias recentes vs. antigas</h1>", unsafe_allow_html=True)

    st.markdown("------------------------------------------------------------")

    st.markdown("<h1 style='font-size: 26px;'>Objetivos</h1>", unsafe_allow_html=True)

    st.markdown("""
    - Avaliar o impacto da recência no consumo de notícias.
    - Identificar se os usuários tendem a acessar notícias mais recentes ou se ainda há demanda por notícias antigas.
    - Verificar se a recência deve ser uma feature relevante para o modelo de recomendação.
    - Auxiliar na mitigação do problema de cold-start, propondo estratégias para novos conteúdos.
    ------------------------------------------------------------
    """)

    st.markdown("<h1 style='font-size: 26px;'>Distribuição do Tempo Desde a Publicação das Notícias Acessadas</h1>", unsafe_allow_html=True)

    recency_df = get_recency_data(spark)

    fig_hist_recency = px.histogram(
        recency_df, x="days_since_published", nbins=50,
        labels={"days_since_published": "Dias desde a Publicação", "access_count": "Número de Acessos"}
    )
    st.plotly_chart(fig_hist_recency, use_container_width=True)

    st.markdown("""
    **Observações:**
    - A distribuição sugere que a maior parte das notícias acessadas tem um tempo relativamente alto desde a publicação.
    - No entanto, há um comportamento consistente de acesso ao longo do tempo, sem uma queda brusca.
    - Um pequeno número de notícias muito antigas ainda recebe visualizações.

    **O que isso significa para o modelo?**
    - O consumo de notícias não se concentra apenas em conteúdos recentes, indicando que um modelo baseado apenas em recência pode não ser ideal.
    - Alguns conteúdos mais antigos podem continuar sendo relevantes, o que sugere que fatores como popularidade ou relevância histórica podem ser úteis na recomendação.

    **Ações Recomendadas:**
    - Criar um filtro de recência adaptativo no modelo, priorizando notícias novas, mas sem descartar completamente conteúdos mais antigos com alto engajamento.
    - Analisar se categorias específicas (como política ou esportes) têm padrões de consumo diferentes.
    - Testar um peso de decaimento temporal para ajustar a importância da recência na recomendação.
    ------------------------------------------------------------
    """)

    st.markdown("<h1 style='font-size: 26px;'>Tendência de Consumo de Notícias Antigas vs. Recentes</h1>", unsafe_allow_html=True)

    time_recency_df = recency_df.groupby(["days_since_published"]).agg({"access_count": "sum"}).reset_index()

    fig_line_recency = px.line(
        time_recency_df, x="days_since_published", y="access_count",
        labels={"days_since_published": "Dias desde a Publicação", "access_count": "Total de Acessos"}
    )
    st.plotly_chart(fig_line_recency, use_container_width=True)

    st.markdown("""
    **Observações:**
    - Há um pico massivo de consumo nos primeiros dias após a publicação da notícia.
    - Após esse período inicial, o consumo cai drasticamente, indicando que a maioria dos usuários busca conteúdo recente.
    - No entanto, algumas notícias antigas ainda aparecem com um pequeno volume de acessos residuais.

    **O que isso significa para o modelo?**
    - Para um sistema de recomendação de notícias, a recência é um fator crítico, mas não absoluto.
    - Notícias virais ou evergreen podem continuar sendo acessadas, exigindo um tratamento especial para evitar que o modelo descarte conteúdos importantes.
    - A baixa demanda por notícias antigas sugere que o modelo deve dar menos peso a conteúdos mais antigos, mas sem removê-los completamente.

    **Ações Recomendadas:**
    - Implementar um decaimento exponencial para priorizar notícias novas, reduzindo a pontuação de conteúdos antigos.
    - Criar uma feature de "vida útil da notícia", identificando conteúdos que permanecem populares por mais tempo (como reportagens especiais ou investigações).
    - Ajustar o modelo para recomendar notícias antigas apenas se houver relevância contextual, como eventos históricos relacionados a tópicos atuais.
    ------------------------------------------------------------
    """)

    st.markdown("<h1 style='font-size: 26px;'>Conclusões Finais</h1>", unsafe_allow_html=True)

    st.markdown("""
    - O consumo de notícias segue um padrão esperado: a maioria dos acessos ocorre logo após a publicação.
    - Para evitar o problema do cold-start, é importante considerar recência como um fator de recomendação, mas não como único critério.
    - Algumas notícias antigas ainda possuem valor, sugerindo a inclusão de um mecanismo para detectar conteúdos de longa relevância.
    - Testar um modelo híbrido que combine recência, popularidade e interesses do usuário.
    - Avaliar diferentes categorias de notícias para entender se há variações nos padrões de consumo.
    - Criar um sistema dinâmico que adapte a importância da recência conforme o tipo de notícia e o perfil do usuário.
    ------------------------------------------------------------
    """)

def show_analysis_7(spark):
    st.markdown("<h1 style='font-size: 32px;'>Análise 7: Sobreposição de acessos entre diferentes usuários</h1>", unsafe_allow_html=True)

    st.markdown("------------------------------------------------------------")

    st.markdown("<h1 style='font-size: 26px;'>Objetivos</h1>", unsafe_allow_html=True)

    st.markdown("""
    - Identificar a distribuição da popularidade das notícias com base no número de usuários únicos que as acessaram.
    - Avaliar se a maioria das notícias é consumida por um pequeno número de usuários ou se há um equilíbrio na distribuição.
    - Compreender a relação entre o número de acessos e a exclusividade das notícias, para auxiliar na personalização do modelo de recomendação.
    ------------------------------------------------------------
    """)

    st.markdown("<h1 style='font-size: 26px;'>Distribuição de Notícias por Número de Usuários</h1>", unsafe_allow_html=True)

    news_overlap_df = get_news_overlap_data(spark)

    fig_news_overlap = px.histogram(
        news_overlap_df, x="user_count", nbins=50,
        labels={"user_count": "Número de Usuários Únicos", "count": "Quantidade de Notícias"}
    )
    st.plotly_chart(fig_news_overlap, use_container_width=True)

    st.markdown("""
    **Observações:**
    - A grande maioria das notícias é acessada por um número muito pequeno de usuários.
    - A distribuição apresenta uma cauda longa, com poucas notícias sendo altamente acessadas por muitos usuários.
    - Isso sugere que a maior parte das notícias tem um consumo nichado, sendo lida por um público restrito.

    **O que isso significa para o modelo?**
    - O modelo de recomendação pode precisar priorizar diferentes abordagens para notícias populares e notícias de nicho.
    - As notícias consumidas por muitos usuários podem ser recomendadas com base em tendências gerais.
    - Já as notícias de baixa popularidade podem exigir técnicas de recomendação personalizadas, baseadas em preferências individuais.

    **Ação recomendada:**
    - Implementar um sistema híbrido que combine recomendações populares com recomendações personalizadas.
    - Avaliar a possibilidade de recomendar conteúdos menos acessados para expandir o engajamento do usuário.
    ------------------------------------------------------------
    """)

    st.markdown("<h1 style='font-size: 26px;'>Distribuição de Notícias por Faixa de Popularidade (Decil)</h1>", unsafe_allow_html=True)

    news_popularity_df = get_news_popularity_data(spark)
    
    bins = pd.qcut(news_popularity_df["unique_users"], q=10, duplicates='drop')
    news_popularity_df["popularity_decile"] = bins.astype(str)

    decile_counts = news_popularity_df.groupby("popularity_decile").size().reset_index(name="count")
    
    fig = px.bar(
        decile_counts, x="popularity_decile", y="count",
        labels={"popularity_decile": "Número de Usuários Únicos (Decil)", "count": "Quantidade de Notícias"},
        text_auto=True
    )

    fig.update_layout(
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.3,
            xanchor="center",
            x=0.5
        )
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    **Observações**
    - A grande maioria das notícias foi acessada por um número muito pequeno de usuários: mais de 210 mil notícias foram vistas por no máximo 10 usuários.
    - Apenas um pequeno número de notícias ultrapassa 1000 usuários únicos, mostrando que poucas notícias se tornam amplamente populares.
    - A curva segue um padrão de distribuição de cauda longa, ou seja, poucas notícias se tornam muito populares enquanto a grande maioria é consumida por poucos usuários.

    **O que isso significa para o modelo de recomendação?**
    - O consumo de notícias é altamente desigual, com muitas matérias tendo acessos extremamente baixos.
    - Um sistema de recomendação baseado apenas em popularidade pode não ser ideal, pois grande parte das notícias não se torna viral.
    - Estratégias como recomendações personalizadas (baseadas no perfil do usuário) podem ser mais eficazes do que simplesmente promover as notícias mais populares.
    - O cold-start para notícias pouco acessadas pode ser um desafio, exigindo abordagens como recomendação baseada em conteúdo.

    **Ações Recomendadas**
    - Segmentar as notícias: Criar estratégias diferenciadas para conteúdos de nicho e conteúdos virais.
    - Considerar modelos híbridos: Combinar recomendações baseadas em popularidade para usuários casuais e recomendações personalizadas para usuários frequentes.
    - Aprimorar a descoberta de notícias menos acessadas, possivelmente através de categorização por temas e afinidade com o usuário.
    - Explorar novas métricas de relevância, como tempo médio na página, taxa de rolagem e cliques para identificar engajamento além da popularidade.
    ------------------------------------------------------------
    """)

    st.markdown("<h1 style='font-size: 26px;'>Conclusões Finais</h1>", unsafe_allow_html=True)

    st.markdown("""
    - A maioria das notícias tem um público pequeno e nichado, o que exige um sistema de recomendação capaz de identificar interesses específicos de cada usuário.
    - Modelos baseados apenas na popularidade podem não ser a melhor abordagem para um sistema de recomendação eficiente neste cenário.
    - Estratégias híbridas, combinando popularidade, personalização e exploração de novos conteúdos, podem ser a melhor alternativa para maximizar o engajamento dos usuários.
    ------------------------------------------------------------
    """)


def show_analysis_8(spark):
    st.markdown("<h1 style='font-size: 32px;'>Análise 8: Sobreposição de acessos entre diferentes usuários</h1>", unsafe_allow_html=True)

    st.markdown("------------------------------------------------------------")

    st.markdown("<h1 style='font-size: 26px;'>Objetivos</h1>", unsafe_allow_html=True)

    st.markdown("""
    - Identificar a distribuição do número de usuários únicos por notícia.
    - Verificar se a maioria das notícias recebe poucos acessos ou se algumas são amplamente consumidas.
    - Avaliar a viabilidade de recomendar notícias populares vs. nichadas para diferentes perfis de usuários.
    - Compreender o impacto da popularidade da notícia na estratégia de recomendação.
    ------------------------------------------------------------
    """)

    news_bins_df = get_news_distribution_data(spark)
    news_bins_count = news_bins_df.groupby("user_bins").size().reset_index(name="count")

    bins_order = ["1-10", "11-100", "101-1000", "1001-10000", "10001+"]
    news_bins_count["user_bins"] = pd.Categorical(news_bins_count["user_bins"], categories=bins_order, ordered=True)
    news_bins_count = news_bins_count.sort_values("user_bins")

    st.markdown("<h1 style='font-size: 26px;'>Distribuição de Notícias por Faixa de Popularidade</h1>", unsafe_allow_html=True) 

    fig_bins_news = px.bar(
        news_bins_count, x="user_bins", y="count",
        labels={"user_bins": "Número de Usuários Únicos", "count": "Quantidade de Notícias"},
        text_auto=True
    )

    st.plotly_chart(fig_bins_news, use_container_width=True)

    st.markdown("""
    **Observações**
    - A maioria esmagadora das notícias foi acessada por um número muito pequeno de usuários: mais de 210 mil notícias foram vistas por no máximo 10 usuários.
    - Apenas um pequeno número de notícias ultrapassa 1000 usuários únicos, mostrando que poucas notícias se tornam muito populares.
    - A curva segue um padrão de distribuição de cauda longa, ou seja, poucas notícias se tornam muito populares enquanto a grande maioria é consumida por poucos usuários.

    **O que isso significa para o modelo de recomendação?**
    - O consumo de notícias é altamente desigual, com muitas matérias tendo acessos extremamente baixos.
    - Um sistema de recomendação baseado apenas em popularidade pode não ser ideal, pois grande parte das notícias não se torna viral.
    - Estratégias como recomendações personalizadas (baseadas no perfil do usuário) podem ser mais eficazes do que simplesmente promover as notícias mais populares.
    - O cold-start para notícias pouco acessadas pode ser um desafio, exigindo abordagens como recomendação baseada em conteúdo.

    **Ações Recomendadas**
    - Segmentar as notícias: Criar estratégias diferenciadas para conteúdos de nicho e conteúdos virais.
    - Considerar modelos híbridos: Combinar recomendações baseadas em popularidade para usuários casuais e recomendações personalizadas para usuários frequentes.
    - Aprimorar a descoberta de notícias menos acessadas, possivelmente através de categorização por temas e afinidade com o usuário.
    - Explorar novas métricas de relevância, como tempo médio na página, taxa de rolagem e cliques para identificar engajamento além da popularidade.
    ------------------------------------------------------------
    """)

    st.markdown("<h1 style='font-size: 26px;'>Conclusões Finais</h1>", unsafe_allow_html=True)

    st.markdown("""
    - A distribuição desigual do consumo de notícias sugere que a popularidade sozinha não pode ser o único critério para recomendações eficazes.
    - Para maximizar a relevância, o modelo deve equilibrar a recomendação de notícias populares e menos conhecidas, dependendo do perfil do usuário.
    - Notícias menos acessadas ainda podem ser valiosas para públicos específicos, reforçando a necessidade de personalização.
    ------------------------------------------------------------
    """)

def show_analysis_9(spark):
    st.markdown("<h1 style='font-size: 32px;'>Análise 9: Calcular a correlação entre número de cliques e tempo de leitura</h1>", unsafe_allow_html=True)
    
    st.markdown("------------------------------------------------------------")

    st.markdown("<h1 style='font-size: 26px;'>Objetivos</h1>", unsafe_allow_html=True)

    st.markdown("""
    - Identificar se há uma relação direta entre o número de cliques e o tempo de leitura.
    - Avaliar se um maior número de cliques significa que o usuário passou mais tempo na página.
    - Compreender se o tempo de permanência pode ser um indicador de engajamento para o modelo de recomendação.
    - Detectar possíveis padrões ou outliers que possam impactar a modelagem.
    ------------------------------------------------------------
    """)

    st.markdown("<h1 style='font-size: 26px;'>Correlação entre Número de Cliques e Tempo de Leitura</h1>", unsafe_allow_html=True)

    correlation, sample_df = get_clicks_time_correlation(spark)

    fig_clicks_time = px.scatter(
        sample_df, x="numberOfClicksHistory", y="timeOnPageHistory",
        labels={"numberOfClicksHistory": "Número de Cliques", "timeOnPageHistory": "Tempo de Leitura (ms)"},
        trendline="ols",
        trendline_color_override="red"
    )

    st.plotly_chart(fig_clicks_time, use_container_width=True)

    st.write(f"Correlação entre número de cliques e tempo de leitura: {correlation:.4f}")

    st.markdown("""
    **Observações**
    - Há uma tendência de correlação positiva fraca entre cliques e tempo de leitura, como indicado pela linha de tendência em vermelho.
    - Muitos pontos estão concentrados próximos da origem, sugerindo que a maioria das interações ocorre em um curto período de tempo, independentemente da quantidade de cliques.
    - Existem outliers extremos, onde alguns usuários passam um tempo muito alto na página, independentemente do número de cliques.
    - O comportamento sugere que nem sempre um maior número de cliques resulta em um tempo de leitura significativamente maior.
    
    **O que isso significa para o modelo de recomendação?**
    
    - O tempo de leitura isoladamente pode não ser um preditor confiável de engajamento, já que usuários podem abrir a página e não necessariamente consumi-la por completo.
    - Recomendações baseadas somente no número de cliques podem não capturar corretamente o nível de interesse do usuário.
    - Estratégias híbridas devem ser exploradas, combinando cliques, tempo de leitura, e outras métricas como taxa de rolagem e número de visitas repetidas.
    
    **Ações Recomendadas**
    
    - Filtrar outliers: Excluir registros de tempo de leitura anormalmente altos ou baixos para evitar distorções nas métricas.
    - Incorporar métricas complementares: Além de cliques e tempo de leitura, utilizar taxa de rolagem e interações subsequentes como indicativos de interesse.
    - Analisar padrões de engajamento: Segmentar usuários em grupos de comportamento (rápido consumo, leitura profunda, baixa interação) para personalizar recomendações.
    ------------------------------------------------------------
    """)

    st.markdown("<h1 style='font-size: 26px;'>Conclusões Finais</h1>", unsafe_allow_html=True)

    st.markdown("""
    - Embora haja uma leve correlação entre número de cliques e tempo de leitura, ela não é forte o suficiente para ser usada isoladamente como um indicador de interesse.
    - O modelo pode ser aprimorado ao considerar métricas combinadas de engajamento, como tempo médio na página + taxa de rolagem + interações repetidas.
    - Estratégias personalizadas podem ser úteis para diferenciar leituras rápidas de usuários casuais e consumo aprofundado de usuários altamente engajados.
    ------------------------------------------------------------
    """)

def show_analysis_10(spark):
    st.markdown("<h1 style='font-size: 32px;'>Análise 10: Perfis de Usuários com Base na Quantidade de Interações</h1>", unsafe_allow_html=True)

    st.markdown("------------------------------------------------------------")

    st.markdown("<h1 style='font-size: 26px;'>Objetivos</h1>", unsafe_allow_html=True)

    st.markdown("""
    - Identificar a distribuição de usuários com base na quantidade de interações.
    - Determinar se existe uma relação entre alta/baixa interação e padrões de consumo.
    - Avaliar a necessidade de segmentação de usuários para estratégias de recomendação diferenciadas.
    - Detectar possíveis outliers, como bots ou heavy users, que podem impactar a recomendação.
    ------------------------------------------------------------
    """)

    st.markdown("<h1 style='font-size: 26px;'>Distribuição de Usuários por Faixa de Interação</h1>", unsafe_allow_html=True)

    user_interactions_df = get_user_interactions_data(spark)

    deciles = np.percentile(user_interactions_df["interaction_count"], np.arange(0, 110, 10))
    deciles = np.unique(deciles) 

    decile_labels = [f"{int(deciles[i])}-{int(deciles[i+1])}" for i in range(len(deciles)-1)]

    user_interactions_df["interaction_decile"] = pd.cut(
        user_interactions_df["interaction_count"], 
        bins=deciles, 
        labels=decile_labels, 
        include_lowest=True
    )

    decile_counts = user_interactions_df["interaction_decile"].value_counts().sort_index().reset_index()
    decile_counts.columns = ["Interaction Range", "User Count"]

    decile_counts["Cumulative %"] = (decile_counts["User Count"].cumsum() / decile_counts["User Count"].sum()) * 100

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=decile_counts["Interaction Range"], 
        y=decile_counts["User Count"], 
        name="Usuários por Faixa",
        text=decile_counts["User Count"], 
        textposition="outside"
    ))

    fig.add_trace(go.Scatter(
        x=decile_counts["Interaction Range"], 
        y=decile_counts["Cumulative %"], 
        name="Percentual Acumulado",
        mode="lines+markers",
        yaxis="y2",
        line=dict(color='red') 
    ))

    fig.update_layout(
        xaxis_title="Faixa de Interações",
        yaxis_title="Quantidade de Usuários",
        yaxis2=dict(
            title="Percentual Acumulado",
            overlaying="y",
            side="right",
            showgrid=False
        ),
        legend=dict(
            orientation="h",  
            yanchor="bottom", 
            y=-0.3, 
            xanchor="center", 
            x=0.5
        ),
        bargap=0.1,
        showlegend=True
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    **Observações:**
    - A grande maioria dos usuários (aproximadamente 342 mil) possui apenas 1 a 2 interações, indicando um consumo extremamente esporádico.
    - A queda no número de usuários conforme o nível de interação aumenta é bastante acentuada:
        - 2-3 interações: 40 mil usuários.
        - 3-4 interações: 24 mil usuários.
        - 4-10 interações: 57 mil usuários.
        - 10-35 interações: 54 mil usuários.
        - 35+ interações: 57 mil usuários.
    - O gráfico mostra um comportamento de cauda longa, onde poucos usuários possuem alta frequência de interações, enquanto a maioria interage muito pouco com a plataforma.
    - O percentual acumulado reforça essa assimetria: uma pequena fração dos usuários representa a maior parte das interações.
    
    **O que isso significa para o modelo?**
    
    - Usuários casuais (1-2 interações) representam a maioria, sugerindo que estratégias de recomendação devem focar em conteúdos populares para engajá-los.
    - Usuários moderadamente ativos (4-10 interações) podem se beneficiar de um modelo híbrido, que combina recomendações baseadas em popularidade e histórico de consumo.
    - Usuários altamente engajados (35+ interações) devem receber recomendações altamente personalizadas, considerando seu histórico detalhado.
    - A presença de usuários com centenas ou milhares de interações pode indicar heavy users ou até mesmo bots, que devem ser analisados separadamente.
    
    **Ações Recomendadas:**
    
    1. **Segmentação de Perfis de Usuários:**
        - Criar modelos diferenciados para usuários casuais, moderadamente ativos e altamente engajados.
        - Para usuários casuais: recomendações baseadas em notícias populares e tendências.
        - Para usuários frequentes: recomendações baseadas no comportamento histórico e padrões de similaridade.
    
    2. **Detecção de Outliers:**
    
        - Identificar e analisar usuários com um número excepcionalmente alto de interações, evitando distorções na recomendação.
        - Verificar a existência de bots ou acessos automatizados.
    
    3. **Ajuste no Modelo de Recomendação:**
    
        - Introduzir pesos diferenciados para os diferentes segmentos de usuários.
        - Testar abordagens que incentivem usuários casuais a aumentar seu engajamento.
    
    4. **Análise Temporal:**
    
        - Avaliar padrões sazonais para entender se o engajamento dos usuários varia ao longo do tempo.
        - Identificar momentos de pico que podem ser explorados para melhorar a personalização da recomendação.
    ------------------------------------------------------------
    """)

    st.markdown("<h1 style='font-size: 26px;'>Conclusões Finais</h1>", unsafe_allow_html=True)

    st.markdown("""
    - A distribuição altamente desigual do número de interações reforça a necessidade de segmentação de usuários no sistema de recomendação.
    - Usuários casuais devem receber sugestões de conteúdos populares para aumentar sua interação inicial.
    - Usuários altamente engajados precisam de um modelo personalizado que aproveite seu comportamento detalhado.
    - Estratégias de filtragem de bots e análise de heavy users podem ajudar a evitar distorções nas recomendações.
    - Ajustes dinâmicos na recomendação, considerando padrões temporais, podem tornar o sistema mais eficiente e responsivo.
    ------------------------------------------------------------
    """)

def show_general_eda_conclusion(spark):
    st.markdown("<h1 style='font-size: 32px;'>Conclusão Final da Análise Exploratória de Dados</h1>", unsafe_allow_html=True)
    
    st.markdown("------------------------------------------------------------")

    st.markdown("<h1 style='font-size: 26px;'>Objetivo</h1>", unsafe_allow_html=True)

    st.markdown("""
    - A análise exploratória buscou entender os padrões de consumo de notícias no G1, identificando fatores que impactam a personalização das recomendações. O desafio principal envolve equilibrar recência, engajamento e popularidade para diferentes perfis de usuários.
    -----------------
    """)

    st.markdown("<h1 style='font-size: 26px;'>Principais Descobertas e Impacto no Modelo</h1>", unsafe_allow_html=True)

    st.markdown("""
    **Padrão de Consumo e Segmentação de Usuários**
    - A maioria dos usuários lê poucas notícias, enquanto um pequeno grupo consome intensamente.
    - O comportamento varia entre usuários casuais, recorrentes e altamente engajados.
    
    **Ações Recomendadas:**
    - Implementar modelos diferenciados para cada perfil.
    - Criar estratégias para detectar e tratar bots.
    - Testar recomendações híbridas: personalização para usuários frequentes, tendências para casuais.
    
    -----------------
    
    **Recência vs. Popularidade**
    - A maioria das notícias tem alto consumo logo após a publicação, mas algumas continuam relevantes ao longo do tempo.
    
    **Ações Recomendadas:**
    - Implementar um peso adaptativo para a recência, ajustando conforme a categoria da notícia.
    - Detectar conteúdos atemporais e evitar descartá-los prematuramente.
    - Explorar um modelo híbrido (recência + relevância + engajamento).
    
    -----------------
    
    **Engajamento e Tempo na Página**
    - O tempo de leitura isoladamente não indica alto engajamento.
    - Cliques, tempo de leitura e taxa de rolagem combinados são melhores indicadores.
    
    **Ações Recomendadas:**
    - Criar um score de engajamento que combine métricas de interação.
    - Filtrar outliers (sessões anormais).
    - Avaliar padrões distintos de leitura para ajustar recomendações.
    
    -----------------
    
    **Retorno e Fidelização**
    - Usuários apresentam padrões cíclicos de acesso, influenciados por eventos sazonais.
    
    **Ações Recomendadas:**
    - Ajustar recomendações conforme horário e dia da semana.
    - Criar um sistema dinâmico de retenção, incentivando novas visitas.
    - Monitorar o impacto de notícias de grande repercussão no engajamento.
    
    -----------------
    
    **Usuários Logados vs. Anônimos**
    - A maioria dos acessos ocorre sem login, reduzindo a personalização possível.
    
    **Ações Recomendadas:**
    - Recomendação híbrida: personalizada para logados, baseada em popularidade para anônimos.
    - Criar incentivos ao login, aumentando a base de dados históricos.
    
    -----------------
    
    **Popularidade e Exploração de Notícias**
    - Poucas notícias dominam a atenção, enquanto a maioria tem baixa visibilidade.
    - O consumo segue um padrão de cauda longa.
    
    **Ações Recomendadas:**
    - Balancear recomendações entre tendências e conteúdos de nicho.
    - Explorar recomendações por similaridade para expandir o consumo de notícias menos acessadas.
    - Considerar uma abordagem que priorize recomendações de notícias novas, visto que notícias antigas, mesmo que populares, podem perder relevância.
    
    -----------------
    """)

    st.markdown("<h1 style='font-size: 26px;'>Impacto nas Features de Engenharia</h1>", unsafe_allow_html=True)

    st.markdown("""
    | **Feature**             | **Justificativa** |
    |-------------------------|------------------------------------------|
    | **Recência**            | Influência direta na relevância.         |
    | **Popularidade**        | Algumas notícias continuam relevantes por mais tempo. |
    | **Engajamento do usuário** | Tempo na página, cliques e rolagem são mais úteis quando combinados. |
    | **Padrão de consumo**   | Usuários casuais e recorrentes precisam de abordagens diferentes. |
    | **Horário e sazonalidade** | O consumo varia ao longo do dia e semana. |
    
    .
                
    **Próximos Passos:**
    - Refinar modelo híbrido (TF-IDF + LightFM) com pesos ajustáveis para recência e engajamento.
    - Implementar estratégias de retenção e descoberta de conteúdos.
    - Desenvolver um sistema adaptável a eventos sazonais e preferências individuais.
    
    -----------------
    """)

    st.markdown("<h1 style='font-size: 26px;'>Conclusão Final</h1>", unsafe_allow_html=True)

    st.markdown("""
    - A análise revelou que a recência é essencial, mas deve ser equilibrada com popularidade e engajamento. A segmentação de usuários e o uso de modelos híbridos são essenciais para uma recomendação eficiente. O próximo passo é testar diferentes abordagens, ajustando os pesos das features para otimizar a precisão e a experiência do usuário.
    
    - Modelos distintos para usuários casuais e recorrentes.
    - Recência como fator crítico, mas não único.
    - Engajamento medido por múltiplas métricas combinadas.
    - Abordagem híbrida para equilibrar popularidade e personalização.
    """)

def pre_cache_all_data(spark):
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    total_functions = 11 
    cached_functions = 0

    try:
        cache_functions = [
            (get_news_per_user, "Cacheando dados de distribuição de leituras..."),
            (get_time_distribution, "Cacheando dados de distribuição temporal..."),
            (get_engagement_data, "Cacheando dados de engajamento..."),
            (get_retention_data, "Cacheando dados de retenção..."),
            (get_user_type_data, "Cacheando dados de tipos de usuário..."),
            (get_recency_data, "Cacheando dados de recência..."),
            (get_news_overlap_data, "Cacheando dados de sobreposição..."),
            (get_news_popularity_data, "Cacheando dados de popularidade..."),
            (get_clicks_time_correlation, "Cacheando dados de correlação..."),
            (get_user_interactions_data, "Cacheando dados de interações..."),
            (get_news_distribution_data, "Cacheando dados de distribuição de notícias...")
        ]

        for func, description in cache_functions:
            status_text.text(description)
            func(spark)  
            cached_functions += 1
            progress_bar.progress(cached_functions / total_functions)
            time.sleep(0.5)  

        progress_bar.progress(1.0)
        status_text.text("Todos os dados foram cacheados com sucesso!")
        return True

    except Exception as e:
        status_text.text(f"❌ Erro ao cachear dados: {str(e)}")
        return False

def show_home():
    st.title("Análise Exploratória - Sistema de Recomendação G1")
    
    st.markdown("""
    ### Sobre o Projeto
    
    Esta análise exploratória faz parte do desenvolvimento de um sistema de recomendação de notícias para o portal G1, 
    um dos maiores portais de notícias do Brasil. O objetivo é compreender profundamente os padrões de consumo de 
    notícias dos usuários para criar recomendações mais precisas e personalizadas.
    
    ### Principais Aspectos Analisados
    
    - **Comportamento dos Usuários**: Padrões de leitura, horários de acesso e engajamento
    - **Consumo de Conteúdo**: Distribuição de leituras, popularidade das notícias e recência
    - **Perfis de Usuário**: Segmentação entre usuários casuais e frequentes
    - **Métricas de Engajamento**: Tempo de leitura, cliques e interações
    
    ### Base de Dados
    
    - **Período Analisado**: Dados históricos de interações dos usuários com notícias do G1
    - **Volume de Dados**: Milhões de interações processadas
    - **Métricas Coletadas**: Tempo de leitura, cliques, scroll, tipo de usuário e outros indicadores
    
    ### Objetivo do Sistema de Recomendação
    
    Desenvolver um sistema capaz de:
    - Personalizar recomendações com base no perfil do usuário
    - Equilibrar notícias populares e conteúdo personalizado
    - Considerar a temporalidade das notícias
    - Melhorar o engajamento dos usuários com o portal
    """)

    loading_message = st.empty()
    loading_message.markdown("#### Inicializando o Sistema\nAguarde enquanto preparamos os dados para uma experiência mais rápida...")

    try:
        spark = init_spark()
        treino, itens = load_data(spark)
        treino.createOrReplaceTempView("tab_treino")
        itens.createOrReplaceTempView("tab_itens")

        cache_status = pre_cache_all_data(spark)

        if cache_status:
            success_message = st.empty()
            success_message.success("#### Sistema Pronto!")
            time.sleep(2)
            loading_message.empty()
            success_message.empty()
        else:
            st.warning("⚠️ Alguns dados podem demorar para carregar durante a navegação.")

    except Exception as e:
        st.error(f"❌ Erro ao inicializar o sistema: {str(e)}")

def main():
    # Inicializar Spark
    spark = init_spark()
    
    # Menu lateral
    with st.sidebar:
        selected = option_menu(
            menu_title=None,
            options=[
                "Home",
                "Análise 1: Distribuição de Leituras",
                "Análise 2: Distribuição Temporal",
                "Análise 3: Relação Tempo e Engajamento",
                "Análise 4: Taxa de Retorno",
                "Análise 5: Usuários Logados vs Anônimos",
                "Análise 6: Padrões de Consumo",
                "Análise 7: Sobreposição de Acessos",
                "Análise 8: Distribuição de Usuários",
                "Análise 9: Correlações",
                "Análise 10: Perfis de Usuários",
                "Conclusões Gerais",
                "Monitoramento do Modelo"
            ],
            icons=[
                "house", "newspaper", "people", "clock", "arrow-repeat",
                "person", "calendar", "intersect", "star", "graph-up",
                "bar-chart", "check-circle", "speedometer"
            ],
            menu_icon=None,
            default_index=0,
            orientation="vertical",
            styles={
                "container": {"padding": "0!important", "background-color": "#262730"},
                "icon": {"color": "#fafafa", "font-size": "16px"},
                "nav-link": {
                    "font-size": "16px",
                    "text-align": "left",
                    "margin": "0px",
                    "--hover-color": "#414757",
                    "color": "#fafafa",
                    "padding": "12px 15px"
                },
                "nav-link-selected": {
                    "background-color": "#00c04b",
                    "color": "#ffffff",
                    "font-weight": "bold"
                },
                "menu-title": {"display": "none"}
            }
        )

    # Roteamento das páginas
    if selected == "Home":
        show_home()
    elif selected == "Análise 1: Distribuição de Leituras":
        show_analysis_1(spark)
    elif selected == "Análise 2: Distribuição Temporal":
        show_analysis_2(spark)
    elif selected == "Análise 3: Relação Tempo e Engajamento":
        show_analysis_3(spark)
    elif selected == "Análise 4: Taxa de Retorno":
        show_analysis_4(spark)
    elif selected == "Análise 5: Usuários Logados vs Anônimos":
        show_analysis_5(spark)
    elif selected == "Análise 6: Padrões de Consumo":
        show_analysis_6(spark)
    elif selected == "Análise 7: Sobreposição de Acessos":
        show_analysis_7(spark)
    elif selected == "Análise 8: Distribuição de Usuários":
        show_analysis_8(spark)
    elif selected == "Análise 9: Correlações":
        show_analysis_9(spark)
    elif selected == "Análise 10: Perfis de Usuários":
        show_analysis_10(spark)
    elif selected == "Conclusões Gerais":
        show_general_eda_conclusion(spark)
    elif selected == "Monitoramento do Modelo":
        from pages.model_monitoring import show_monitoring_page
        show_monitoring_page()

if __name__ == "__main__":
    main()

