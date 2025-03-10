from pyspark.sql import SparkSession
from pyspark.sql.functions import col, avg, first
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
        """
        Initialize the LightFMDataPreparer.

        Args:
            spark (SparkSession): A Spark session for processing data.
            silver_path_treino_normalized (str): Path to the normalized 'Treino' dataset.
            silver_path_itens_embeddings (str): Path to the item embeddings dataset.
            gold_path_lightfm_interactions (str): Path to save the LightFM interactions matrix.
            gold_path_lightfm_user_features (str): Path to save the user features.
            gold_path_lightfm_item_features (str): Path to save the item features.

        Returns:
            None
        """
        self.spark = spark
        self.silver_path_treino_normalized = silver_path_treino_normalized
        self.silver_path_itens_embeddings = silver_path_itens_embeddings
        self.gold_path_lightfm_interactions = gold_path_lightfm_interactions
        self.gold_path_lightfm_user_features = gold_path_lightfm_user_features
        self.gold_path_lightfm_item_features = gold_path_lightfm_item_features

    def create_id_mappings(self, treino_df, items_embeddings_df):
        """
        Create numerical mappings for userId and page.

        Args:
            treino_df (DataFrame): The DataFrame containing training data.
            items_embeddings_df (DataFrame): The DataFrame containing item embeddings.

        Returns:
            tuple: A tuple containing two dictionaries: user_id_map and page_id_map.
        """
        logger.info("Creating ID mappings...")

     
        logger.info("Sample of training pages:")
        treino_df.select("history").distinct().show(5, truncate=False)

        logger.info("Sample of items pages:")
        items_embeddings_df.select("page").distinct().show(5, truncate=False)

        user_ids = treino_df.select("userId").distinct().rdd.flatMap(lambda x: x).collect()
        page_ids = items_embeddings_df.select("page").distinct().rdd.flatMap(lambda x: x).collect()

        user_id_map = {user_id: idx for idx, user_id in enumerate(user_ids)}
        page_id_map = {page: idx for idx, page in enumerate(page_ids)}

        logger.info(f"Created mappings - Users: {len(user_id_map)}, Pages: {len(page_id_map)}")

    
        total_pages = treino_df.select("history").distinct().count()
        pages_in_mapping = sum(1 for page in treino_df.select("history").distinct().rdd.flatMap(lambda x: x).collect()
                               if page in page_id_map)

        logger.info(f"Page coverage: {pages_in_mapping}/{total_pages} "
                    f"({pages_in_mapping / total_pages:.2%} of training pages in mapping)")

        return user_id_map, page_id_map

    def build_interaction_matrix(self, treino_df, user_id_map, page_id_map):
        """
        Build the interaction matrix using adjusted_score.

        Args:
            treino_df (DataFrame): The DataFrame containing training data.
            user_id_map (dict): A mapping of user IDs to numerical indices.
            page_id_map (dict): A mapping of page IDs to numerical indices.

        Returns:
            csr_matrix: A sparse matrix representing user-item interactions.
        """
        try:
            logger.info("Building interaction matrix...")
            logger.info(f"Input DataFrame size: {treino_df.count()} rows")
            logger.info(f"Number of users: {len(user_id_map)}")
            logger.info(f"Number of pages: {len(page_id_map)}")

        
            logger.info("Sample of page_id_map:")
            sample_pages = list(page_id_map.keys())[:5]
            logger.info(f"First 5 pages in mapping: {sample_pages}")

        
            def process_partition(iterator):
                users = []
                items = []
                scores = []
                partition_count = 0
                error_count = 0

                for row, index in iterator:
                    try:
                        partition_count += 1

                
                        if partition_count == 1:
                            logger.info(f"Processing partition - First row data:")
                            logger.info(f"userId: {row.userId}")
                            logger.info(f"history: {row.history}")
                            logger.info(f"adjusted_score: {row.adjusted_score}")

                        if not row.userId or not row.history or not hasattr(row, 'adjusted_score'):
                            logger.warning(f"Missing data in row {index}")
                            continue

                        user_idx = user_id_map.get(row.userId)
                        if user_idx is None:
                            logger.warning(f"User ID {row.userId} not found in mapping")
                            continue

                 
                        page = row.history.strip()
                        page_idx = page_id_map.get(page)

                        if page_idx is not None:
                            try:
                                score = float(row.adjusted_score)
                                if score > 0: 
                                 
                                    yield (user_idx, page_idx, score)

                                    if partition_count == 1: 
                                        logger.info(f"First successful interaction: user={row.userId}, "
                                                    f"page={page}, score={score}")
                            except (ValueError, TypeError) as e:
                                logger.warning(f"Invalid score value for user {row.userId}, page {page}: {str(e)}")
                                error_count += 1
                        else:
                            if partition_count == 1: 
                                logger.warning(
                                    f"Page {page} not found in mapping. Available pages sample: {sample_pages[:2]}")

                    except Exception as e:
                        error_count += 1
                        logger.error(f"Error processing row {index}: {str(e)}")
                        continue

                  
                    if partition_count % 1000 == 0:
                        logger.info(f"Processed {partition_count} rows in partition")

                logger.info(f"Partition completed: {partition_count} rows processed, {error_count} errors")

          
            interactions_rdd = treino_df.rdd.zipWithIndex().mapPartitions(process_partition)

          
            logger.info("Collecting interactions from all partitions...")
            interactions = interactions_rdd.collect()

          
            users, items, scores = [], [], []
            for user_idx, page_idx, score in interactions:
                users.append(user_idx)
                items.append(page_idx)
                scores.append(score)

            total_interactions = len(users)
            logger.info(f"Total interactions collected: {total_interactions}")

        
            users = np.array(users)
            items = np.array(items)
            scores = np.array(scores)

            logger.info(f"Creating sparse matrix with {len(scores)} non-zero elements")
            logger.info(f"Users range: {users.min()}-{users.max()}")
            logger.info(f"Items range: {items.min()}-{items.max()}")
            logger.info(f"Scores range: {scores.min():.4f}-{scores.max():.4f}")

            interaction_matrix = sp.csr_matrix(
                (scores, (users, items)),
                shape=(len(user_id_map), len(page_id_map))
            )

            logger.info(f"Created interaction matrix with shape {interaction_matrix.shape} "
                        f"and {interaction_matrix.nnz} non-zero elements")

            return interaction_matrix

        except Exception as e:
            logger.error(f"Error building interaction matrix: {str(e)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise

    def create_user_features(self, treino_df):
        """
        Create user features and normalize them.

        Args:
            treino_df (DataFrame): The DataFrame containing training data.

        Returns:
            DataFrame: A DataFrame containing normalized user features.
        """
        logger.info("Creating user features...")

      
        user_features = treino_df.groupBy("userId").agg(
            first("userType").alias("userType"),
            avg("avg_time_on_page").alias("avg_time_on_page"),
            avg("numberOfClicksHistory").alias("numberOfClicksHistory"),
            avg("timeOnPageHistory").alias("timeOnPageHistory"),
            avg("interaction_score").alias("interaction_score"),
            avg("time_since_last_interaction").alias("time_since_last_interaction"),
            avg("recency_weight").alias("recency_weight")
        )

        logger.info(f"Number of unique users: {user_features.count()}")

        # Normalizar as features
        assembler = VectorAssembler(
            inputCols=[
                "userType",
                "avg_time_on_page",
                "numberOfClicksHistory",
                "timeOnPageHistory",
                "interaction_score",
                "time_since_last_interaction",
                "recency_weight"
            ],
            outputCol="features_vec"
        )
        user_features = assembler.transform(user_features)

        scaler = MinMaxScaler(inputCol="features_vec", outputCol="user_features")
        scaler_model = scaler.fit(user_features)
        user_features = scaler_model.transform(user_features)

       
        unique_users = user_features.count()
        logger.info(f"Created user features for {unique_users} users")

        return user_features

    def create_item_features(self, items_embeddings_df):
        """
        Create item features from the TF-IDF vector.

        Args:
            items_embeddings_df (DataFrame): The DataFrame containing item embeddings.

        Returns:
            DataFrame: A DataFrame containing item features.
        """
        logger.info("Creating item features...")

        item_features = items_embeddings_df.select("page", "features")
        return item_features

    def prepare_data(self):
        """
        Main method to prepare data for LightFM.

        Args:
            None

        Returns:
            None
        """
        logger.info("Preparing data for LightFM...")

     
        treino_df = self.spark.read.parquet(self.silver_path_treino_normalized)
        items_embeddings_df = self.spark.read.parquet(self.silver_path_itens_embeddings)

       
        user_id_map, page_id_map = self.create_id_mappings(treino_df, items_embeddings_df)

       
        interaction_matrix = self.build_interaction_matrix(treino_df, user_id_map, page_id_map)

        
        user_features = self.create_user_features(treino_df)

       
        item_features = self.create_item_features(items_embeddings_df)

       
        try:
            logger.info("Salvando a matriz de interações como .npz...")
            sp.save_npz(os.path.join(self.gold_path_lightfm_interactions, "interaction_matrix.npz"), interaction_matrix)
            logger.info("Matriz de interações salva com sucesso como .npz.")
        except Exception as e:
            logger.error(f"Erro ao salvar a matriz de interações: {str(e)}")

       
        try:
            logger.info("Salvando as features do usuário como Parquet...")
            user_features.write.mode("overwrite").option("compression", "snappy").option("maxRecordsPerFile",
                                                                                         "10000").parquet(
                os.path.join(self.gold_path_lightfm_user_features, "user_features.parquet")
            )
            logger.info("Features do usuário salvas com sucesso como Parquet.")
        except Exception as e:
            logger.error(f"Erro ao salvar as features do usuário: {str(e)}")

       
        try:  
            logger.info("Salvando as features dos itens como Parquet...")
            item_features.write.mode("overwrite").option("compression", "snappy").option("maxRecordsPerFile",
                                                                                         "10000").parquet(
                os.path.join(self.gold_path_lightfm_item_features, "item_features.parquet")
            )
            logger.info("Features dos itens salvas com sucesso como Parquet.")
        except Exception as e:
            logger.error(f"Erro ao salvar as features dos itens: {str(e)}")

        logger.info("Data preparation for LightFM completed.")
