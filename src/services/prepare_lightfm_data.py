from pyspark.sql import SparkSession
from pyspark.sql.functions import col, avg
from pyspark.ml.feature import MinMaxScaler, VectorAssembler
from scipy.sparse import csr_matrix
import numpy as np
import os
from utils.logger import logger
import scipy.sparse as sp


class LightFMDataPreparer:
    def __init__(self, spark: SparkSession, silver_path_treino_normalized: str, silver_path_itens_embeddings: str,
                 gold_path_lightfm_interactions: str, gold_path_lightfm_user_features: str,
                 gold_path_lightfm_item_features: str):
        self.spark = spark
        self.silver_path_treino_normalized = silver_path_treino_normalized
        self.silver_path_itens_embeddings = silver_path_itens_embeddings
        self.gold_path_lightfm_interactions = gold_path_lightfm_interactions
        self.gold_path_lightfm_user_features = gold_path_lightfm_user_features
        self.gold_path_lightfm_item_features = gold_path_lightfm_item_features

    def create_id_mappings(self, treino_df, items_embeddings_df):
        """Criar mapeamento numérico para userId e page."""
        logger.info("Creating ID mappings...")

        user_ids = treino_df.select("userId").distinct().rdd.flatMap(lambda x: x).collect()
        page_ids = items_embeddings_df.select("page").distinct().rdd.flatMap(lambda x: x).collect()

        user_id_map = {user_id: idx for idx, user_id in enumerate(user_ids)}
        page_id_map = {page: idx for idx, page in enumerate(page_ids)}

        return user_id_map, page_id_map

    def build_interaction_matrix(self, treino_df, user_id_map, page_id_map):
        """Construir a matriz de interações usando adjusted_score."""
        logger.info("Building interaction matrix...")

        # Usar rdd.zipWithIndex() para mapear IDs
        interactions = []
        treino_rdd = treino_df.rdd.zipWithIndex()  # Adiciona um índice a cada linha

        # Processar cada partição separadamente
        def process_partition(iterator):
            for row, index in iterator:
                user_idx = user_id_map[row.userId]
                for page in row.history:
                    if page in page_id_map:
                        page_idx = page_id_map[page]
                        yield (user_idx, page_idx, row.adjusted_score)

        # Usar mapPartitions para processar as interações
        interactions_rdd = treino_rdd.mapPartitions(process_partition)

        # Coletar os resultados
        interactions = interactions_rdd.collect()

        # Criar matriz esparsa
        interaction_matrix = csr_matrix(
            (np.array([score for _, _, score in interactions]),
             (np.array([user for user, _, _ in interactions]),
              np.array([item for _, item, _ in interactions]))),
            shape=(len(user_id_map), len(page_id_map))
        )

        return interaction_matrix

    def create_user_features(self, treino_df):
        """Criar features do usuário e normalizá-las."""
        logger.info("Creating user features...")

        user_features = treino_df.select(
            "userId",
            "userType",
            "avg_time_on_page",
            "numberOfClicksHistory",
            "timeOnPageHistory",
            "interaction_score",
            "time_since_last_interaction",
            "recency_weight"
        )

        # Normalizar as features
        assembler = VectorAssembler(inputCols=user_features.columns[1:], outputCol="features_vec")
        user_features = assembler.transform(user_features)

        scaler = MinMaxScaler(inputCol="features_vec", outputCol="user_features")
        scaler_model = scaler.fit(user_features)
        user_features = scaler_model.transform(user_features)

        return user_features

    def create_item_features(self, items_embeddings_df):
        """Criar features dos itens a partir do vetor TF-IDF."""
        logger.info("Creating item features...")

        item_features = items_embeddings_df.select("page", "features")
        return item_features

    def prepare_data(self):
        """Método principal para preparar os dados para o LightFM."""
        logger.info("Preparing data for LightFM...")

        # Ler dados
        treino_df = self.spark.read.parquet(self.silver_path_treino_normalized)
        items_embeddings_df = self.spark.read.parquet(self.silver_path_itens_embeddings)

        # Criar mapeamentos
        user_id_map, page_id_map = self.create_id_mappings(treino_df, items_embeddings_df)

        # Construir matriz de interações
        interaction_matrix = self.build_interaction_matrix(treino_df, user_id_map, page_id_map)

        # Criar features do usuário
        user_features = self.create_user_features(treino_df)

        # Criar features dos itens
        item_features = self.create_item_features(items_embeddings_df)

        # Salvar a matriz de interações como .npz
        try:
            logger.info("Salvando a matriz de interações como .npz...")
            sp.save_npz(os.path.join(self.gold_path_lightfm_interactions, "interaction_matrix.npz"), interaction_matrix)
            logger.info("Matriz de interações salva com sucesso como .npz.")
        except Exception as e:
            logger.error(f"Erro ao salvar a matriz de interações: {str(e)}")

        # Salvar features do usuário como .npz e Parquet
        try:
            # logger.info("Salvando as features do usuário como .npz...")
            # user_features_npz = user_features.select("userId", "user_features").toPandas().set_index("userId").values
            # user_features_npz = user_features_npz.astype(np.float32)  # Convertendo para float32
            # user_features_csr = sp.csr_matrix(user_features_npz)  # Converter para csr_matrix
            # sp.save_npz(os.path.join(self.gold_path_lightfm_user_features, "user_features.npz"), user_features_csr)
            # logger.info("Features do usuário salvas com sucesso como .npz.")

            logger.info("Salvando as features do usuário como Parquet...")
            user_features.write.mode("overwrite").option("compression", "snappy").option("maxRecordsPerFile",
                                                                                         "10000").parquet(
                os.path.join(self.gold_path_lightfm_user_features, "user_features.parquet")
            )
            logger.info("Features do usuário salvas com sucesso como Parquet.")
        except Exception as e:
            logger.error(f"Erro ao salvar as features do usuário: {str(e)}")

        # Salvar features dos itens como .npz e Parquet
        try:
            # logger.info("Salvando as features dos itens como .npz...")
            # item_features_npz = item_features.select("page", "features").toPandas().set_index("page").values
            # item_features_npz = item_features_npz.astype(np.float32)  # Convertendo para float32
            # item_features_csr = sp.csr_matrix(item_features_npz)  # Converter para csr_matrix
            # sp.save_npz(os.path.join(self.gold_path_lightfm_item_features, "item_features.npz"), item_features_csr)
            # logger.info("Features dos itens salvas com sucesso como .npz.")

            logger.info("Salvando as features dos itens como Parquet...")
            item_features.write.mode("overwrite").option("compression", "snappy").option("maxRecordsPerFile",
                                                                                         "10000").parquet(
                os.path.join(self.gold_path_lightfm_item_features, "item_features.parquet")
            )
            logger.info("Features dos itens salvas com sucesso como Parquet.")
        except Exception as e:
            logger.error(f"Erro ao salvar as features dos itens: {str(e)}")

        logger.info("Data preparation for LightFM completed.")
