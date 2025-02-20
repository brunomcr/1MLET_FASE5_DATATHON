from pyspark.sql.functions import (
    monotonically_increasing_id, explode, split, 
    col, count, when, array, lit, struct, row_number,
    sequence, arrays_zip, hour, dayofweek, month, lag, current_timestamp, min,
    from_unixtime, avg, max, expr
)
from pyspark.ml.feature import StringIndexer
from pyspark.sql.types import ArrayType, FloatType
from pyspark.sql.window import Window
from pyspark.sql.functions import udf
import numpy as np
from utils.logger import logger
import gc
import shutil
import os
from configs.config import Config

# UDF para converter vetor em array
@udf(returnType=ArrayType(FloatType()))
def vector_to_array(v):
    return v.toArray().tolist()

class FeatureEngineering:
    def __init__(self, spark):
        self.spark = spark
        self.user_indexer = None
        self.page_indexer = None
        
    def log_step(self, message):
        logger.info(message)
        
    def create_id_mappings(self, treino_df, items_df):
        """
        Criar mapeamento numérico para userId e page
        """
        self.log_step("Creating ID mappings...")
        
        # User ID mapping
        self.user_indexer = StringIndexer(
            inputCol="userId", 
            outputCol="user_idx",
            handleInvalid="keep"
        ).fit(treino_df)
        
        # Page ID mapping - usando todos os pages únicos
        unique_pages = (items_df.select("page")
                       .union(treino_df.select(explode(split("history", ",")).alias("page")))
                       .distinct())
        
        self.page_indexer = StringIndexer(
            inputCol="page", 
            outputCol="item_idx",
            handleInvalid="keep"
        ).fit(unique_pages)
        
        return self.user_indexer, self.page_indexer
    
    def create_interaction_matrix(self, treino_indexed):
        """Criar matriz de interações com features temporais e validação"""
        
        # Inicializar validador
        validator = FeatureValidator()
        
        # Window specs
        user_window = Window.partitionBy("userId").orderBy("timestampHistory")
        group_window = Window.partitionBy("history")
        global_window = Window.orderBy(lit(1))
        
        # 1. Normalizar métricas base
        interactions = (treino_indexed
            .withColumn("clicks_normalized", 
                col("numberOfClicksHistory") / max("numberOfClicksHistory").over(global_window))
            .withColumn("time_normalized",
                col("timeOnPageHistory") / max("timeOnPageHistory").over(global_window))
            .withColumn("scroll_normalized",
                col("scrollPercentageHistory") / 100.0))
        
        # 2. Criar features com pesos calibrados
        interactions = (interactions
            .select(
                col("user_idx"),
                col("userId"),  # Necessário para window
                col("timestampHistory"),  # Necessário para cálculos temporais
                explode(split("history", ",")).alias("page"),
                
                # Features temporais
                hour(from_unixtime("timestampHistory")).alias("hour"),
                dayofweek(from_unixtime("timestampHistory")).alias("dayofweek"),
                month(from_unixtime("timestampHistory")).alias("month"),
                
                # Features de recência
                expr("""
                    exp(-(unix_timestamp() - timestampHistory) / 86400.0)
                """).alias("recency_score"),
                
                # Score composto
                (col("clicks_normalized") * 0.4 +
                 col("time_normalized") * 0.35 +
                 col("scroll_normalized") * 0.25
                ).alias("base_score")
            ))
        
        # 3. Adicionar features de sequência
        interactions = (interactions
            .withColumn("time_since_last",
                col("timestampHistory") - lag("timestampHistory", 1)
                .over(user_window))
            .withColumn("session_score",
                when(col("time_since_last") < 1800, 1.2)
                .otherwise(1.0))
        )
        
        # 4. Calcular score final e selecionar colunas finais
        interactions = (interactions
            .withColumn("interaction_score",
                col("base_score") * 
                col("recency_score") * 
                col("session_score")
            )
            .select(  # Seleção final das colunas
                "user_idx",
                "page",
                "hour",
                "dayofweek",
                "month",
                "interaction_score"
            ))
        
        # 5. Validar features
        validator.validate_feature(interactions, "interaction_score", [0.0, 1.2])
        
        # Indexar pages
        interactions = self.page_indexer.transform(interactions)
        
        return interactions
    
    def prepare_item_features(self, items_df):
        """
        Preparar features dos itens para o LightFM usando TF-IDF
        """
        # Selecionar apenas TF-IDF features e índices
        item_features = (items_df
            .select(
                "item_idx",
                # Explodir o vetor TF-IDF em colunas individuais
                explode(arrays_zip(
                    sequence(lit(0), lit(99)).alias("feature_idx"),  # índices de 0 a 99 (100 features)
                    vector_to_array("tfidf_features").alias("embedding_value")  # converter vetor para array
                )).alias("features")
            )
            .select(
                "item_idx",
                col("features.feature_idx").alias("feature_idx"),
                col("features.embedding_value").alias("embedding_value")
            ))
        
        return item_features
    
    def prepare_lightfm_matrices(self, treino_path, items_path, advanced_features_path, output_path):
        """Prepara as matrizes para o LightFM"""
        logger.info("Preparing LightFM matrices...")
        
        # Carregar dados
        treino_df = self.spark.read.parquet(treino_path)
        items_df = self.spark.read.parquet(items_path)
        advanced_df = self.spark.read.parquet(advanced_features_path)
        
        # Join com alias para evitar ambiguidade
        enriched_df = treino_df.alias("treino") \
            .join(advanced_df.alias("advanced"), 
                  on=["userId", "history"], 
                  how="left")
        
        # Features do usuário atualizadas
        user_features = enriched_df.select(
            col("treino.userId"),
            col("treino.userType"),
            # Features temporais
            'avg_time_between_reads',
            'reading_velocity',
            # Features de engajamento
            'engagement_score',
            'normalized_time',
            'normalized_scroll',
            # Features de sequência
            'items_in_session',
            'session_diversity',
            # Features de tendências
            'trend_score',
            # Features contextuais
            'typical_hours',  # Mantendo apenas as features que existem
            # Features calculadas
            'recency_score'
        ).distinct()
        
        # 2. Features do Item
        item_features = items_df.join(
            enriched_df.groupBy('history').agg(
                avg('engagement_score').alias('avg_engagement'),
                avg('recency_score').alias('avg_recency')
            ),
            items_df.page == enriched_df.history,
            'left'
        )
        
        # 3. Matriz de Interações
        interactions = enriched_df.select(
            'userId',
            'history',
            # Peso da interação baseado nas features avançadas
            (col('engagement_score') * 
             col('recency_score') * 
             when(col('above_avg_engagement'), 1.2).otherwise(1.0)
            ).alias('interaction_weight')
        )
        
        # Salvar as matrizes processadas
        user_features.write.mode("overwrite").parquet(f"{output_path}/user_features")
        item_features.write.mode("overwrite").parquet(f"{output_path}/item_features")
        interactions.write.mode("overwrite").parquet(f"{output_path}/interactions")
        
        return {
            "n_users": user_features.select("userId").distinct().count(),
            "n_items": item_features.select("page").distinct().count(),
            "n_interactions": interactions.count()
        }

class FeatureValidator:
    def __init__(self):
        self.validation_metrics = {}
    
    def validate_feature(self, df, feature_name, expected_range=None):
        """Validar uma feature específica"""
        stats = df.select(
            count(when(col(feature_name).isNull(), True)).alias("null_count"),
            min(feature_name).alias("min_value"),
            max(feature_name).alias("max_value"),
            avg(feature_name).alias("mean"),
            expr(f"percentile({feature_name}, 0.5)").alias("median")
        ).collect()[0]
        
        is_valid = True
        if expected_range:
            is_valid = (stats.min_value >= expected_range[0] and 
                       stats.max_value <= expected_range[1])
        
        self.validation_metrics[feature_name] = {
            "null_count": stats.null_count,
            "min_value": stats.min_value,
            "max_value": stats.max_value,
            "mean": stats.mean,
            "median": stats.median,
            "is_valid": is_valid
        }
        
        return is_valid 