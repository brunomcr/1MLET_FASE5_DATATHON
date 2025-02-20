from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, unix_timestamp, datediff, current_timestamp, 
    hour, dayofweek, month, lag, count, avg, stddev, max,
    collect_list, struct, explode, array, expr, 
    from_unixtime, window, sum, when, lit, udf,
    hours, days
)
from pyspark.sql.window import Window
from pyspark.ml.feature import BucketedRandomProjectionLSH
from utils.logger import logger
from pyspark.sql.types import DoubleType
import numpy as np

class AdvancedFeatureEngineering:
    def __init__(self, spark: SparkSession):
        self.spark = spark
        
        # Registrar UDF para similaridade cosseno
        @udf(returnType=DoubleType())
        def cosine_similarity(vec1, vec2):
            if vec1 is None or vec2 is None:
                return None
            return float(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))
        
        self.spark.udf.register("cosine_similarity", cosine_similarity)
        
    def log_step(self, message: str):
        logger.info(message)
        
    def add_temporal_features(self, df):
        """
        Adiciona features temporais avançadas:
        - Tempo desde a última interação
        - Padrões de horário/dia da semana
        - Velocidade de consumo de conteúdo
        """
        self.log_step("Adding temporal features...")
        
        # Window specs
        user_window = Window.partitionBy("userId").orderBy("timestampHistory")
        
        return df.withColumn(
            "timestamp", from_unixtime(col("timestampHistory"))
        ).withColumn(
            "hour_of_day", hour("timestamp")
        ).withColumn(
            "day_of_week", dayofweek("timestamp")
        ).withColumn(
            "time_since_last_interaction",
            col("timestampHistory") - lag("timestampHistory").over(user_window)
        ).withColumn(
            "avg_time_between_reads",
            avg("time_since_last_interaction").over(user_window)
        ).withColumn(
            "reading_velocity",
            col("timeOnPageHistory") / col("scrollPercentageHistory")
        )
    
    def add_content_engagement_features(self, df):
        """
        Features baseadas no engajamento com o conteúdo:
        - Score de engajamento normalizado
        - Tempo relativo na página
        - Profundidade de scroll relativa
        """
        self.log_step("Adding engagement features...")
        
        # Definir janela para normalização
        engagement_window = Window.partitionBy("userId")
        
        return df.withColumn(
            "normalized_time",
            col("timeOnPageHistory") / max(col("timeOnPageHistory")).over(engagement_window)
        ).withColumn(
            "normalized_scroll",
            col("scrollPercentageHistory") / max(col("scrollPercentageHistory")).over(engagement_window)
        ).withColumn(
            "engagement_score",
            (col("normalized_time") + col("normalized_scroll")) / 2
        ).withColumn(
            "above_avg_engagement",
            col("engagement_score") > avg(col("engagement_score")).over(engagement_window)
        )
    
    def add_sequence_features(self, df):
        """
        Features baseadas em sequências de leitura:
        - Categorias frequentes
        - Transições entre conteúdos
        - Padrões de sessão
        """
        self.log_step("Adding sequence features...")
        
        # Window para análise de sequência
        sequence_window = Window.partitionBy("userId").orderBy("timestampHistory")
        session_window = Window.partitionBy("userId", "session_id")
        
        return df.withColumn(
            "session_id",
            expr("""sum(case when 
                time_since_last_interaction > 1800 
                then 1 else 0 end) over (partition by userId order by timestampHistory)""")
        ).withColumn(
            "items_in_session",
            count("history").over(session_window)
        ).withColumn(
            "prev_item",
            lag("history").over(sequence_window)
        )
    
    def add_recency_features(self, df, items_df):
        """
        Features baseadas em recência:
        - Idade do conteúdo
        - Popularidade recente
        - Tendências temporais
        """
        self.log_step("Adding recency features...")
        
        # Join com informações dos itens usando alias
        df_with_items = df.alias("treino").join(
            items_df.select(
                col("page"),
                "issued"
            ),
            col("treino.history") == col("page"),
            "left"
        )
        
        return df_with_items.withColumn(
            "content_age_hours",
            (unix_timestamp(current_timestamp()) - unix_timestamp(col("issued"))) / 3600
        ).withColumn(
            "recency_score",
            expr("exp(-content_age_hours / (24.0 * 7.0))")  # Decay de 7 dias
        )
    
    def add_content_similarity_features(self, df, items_df):
        """
        Adiciona features de similaridade entre conteúdos para melhorar 
        recomendações do LightFM, especialmente para cold-start:
        - Similaridade com item anterior (captura sequência de leitura)
        - Diversidade na sessão (ajuda na diversificação das recomendações)
        """
        self.log_step("Adding content similarity features...")
        
        # Join com items para pegar os embeddings usando alias explícito
        df_with_embeddings = df.alias("interactions").join(
            items_df.alias("items"),
            col("interactions.history") == col("items.page"),
            "left"
        ).select(
            "interactions.*",  # Mantém todas as colunas do df original
            col("items.tfidf_features")  # Adiciona apenas o embedding do item
        )
        
        # Window para comparar com item anterior
        sequence_window = Window.partitionBy("userId").orderBy("timestampHistory")
        session_window = Window.partitionBy("userId", "session_id")
        
        return df_with_embeddings.withColumn(
            "prev_item_embedding", 
            lag("tfidf_features").over(sequence_window)
        ).withColumn(
            "content_similarity",
            expr("cosine_similarity(tfidf_features, prev_item_embedding)")
        ).withColumn(
            "session_diversity",
            avg("content_similarity").over(session_window)
        )

    def add_trend_features(self, df):
        """
        Features de tendências otimizadas:
        - Calcula max timestamp por história primeiro
        - Depois faz as contagens em uma única passada
        """
        self.log_step("Adding trend features...")
        
        # 1. Primeiro calcula o timestamp mais recente por história
        window_max = Window.partitionBy("history")
        df_with_max = df.withColumn(
            "max_ts", 
            max("timestampHistory").over(window_max)
        )
        
        # 2. Depois faz as contagens usando o max_ts como referência
        return df_with_max.withColumn(
            "views_last_24h",
            sum(
                when(
                    col("timestampHistory") >= col("max_ts") - 86400, 
                    1
                ).otherwise(0)
            ).over(window_max)
        ).withColumn(
            "views_last_7d",
            sum(
                when(
                    col("timestampHistory") >= col("max_ts") - 604800, 
                    1
                ).otherwise(0)
            ).over(window_max)
        ).withColumn(
            "trend_score",
            col("views_last_24h") / (col("views_last_7d") / 7)
        ).withColumn(
            "is_trending",
            col("trend_score") > 1.5
        ).drop("max_ts")

    def process_features(self, treino_df, items_df):
        """
        Pipeline completo de feature engineering
        """
        self.log_step("Starting advanced feature engineering...")
        
        # Features existentes
        df_temporal = self.add_temporal_features(treino_df)
        df_engagement = self.add_content_engagement_features(df_temporal)
        df_sequence = self.add_sequence_features(df_engagement)
        df_recency = self.add_recency_features(df_sequence, items_df)
        
        # Novas features
        df_similarity = self.add_content_similarity_features(df_recency, items_df)
        df_final = self.add_trend_features(df_similarity)
        
        return df_final 