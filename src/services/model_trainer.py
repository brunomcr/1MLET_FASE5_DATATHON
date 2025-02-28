from lightfm import LightFM
from lightfm.evaluation import precision_at_k, recall_at_k, auc_score
import numpy as np
from scipy.sparse import csr_matrix
from utils.logger import logger
from pyspark.sql.functions import col, rand
import pickle
import os

class LightFMTrainer:
    def __init__(self, spark):
        """
        Initialize the LightFMTrainer.

        Args:
            spark (SparkSession): A Spark session for processing data.

        This constructor initializes the LightFMTrainer with the provided Spark session and sets up
        the LightFM model with specified hyperparameters, including learning rate, loss function,
        number of components, and regularization parameters for both items and users.

        Returns:
            None
        """
        self.spark = spark
        self.model = LightFM(
            learning_rate=0.05,
            loss='warp',        
            no_components=100,   
            item_alpha=1e-6,    
            user_alpha=1e-6,    
            random_state=42
        )
        
    def prepare_matrices(self, interactions_path, item_features_path):
        """
        Prepare matrices for training the LightFM model in an optimized way.

        Args:
            interactions_path (str): The path to the interactions data in parquet format.
            item_features_path (str): The path to the item features data in parquet format.

        Returns:
            tuple: A tuple containing the interaction matrix and item features matrix.
        """
        logger.info(f"Lendo dados do caminho: {interactions_path}")
        
       
        interactions_df = self.spark.read.parquet(interactions_path)
        interactions_rows = interactions_df.collect()
        
        interactions_np = np.array([
            [float(row['user_idx']), 
             float(row['item_idx']), 
             float(row['interaction_score'])] 
            for row in interactions_rows
        ])
        
       
        unique_item_ids = np.unique(interactions_np[:, 1])
        item_id_map = {old_idx: new_idx for new_idx, old_idx in enumerate(unique_item_ids)}
        
        
        interactions_np[:, 1] = np.array([item_id_map[item_id] for item_id in interactions_np[:, 1]])
        
       
        item_features_df = (self.spark.read
            .parquet(item_features_path)
            .filter(col("item_idx").isin(unique_item_ids.tolist())))
        
        item_features_rows = item_features_df.collect()
        item_features_np = np.array([
            [float(row['item_idx']), 
             float(row['feature_idx']), 
             float(row['embedding_value'])]
            for row in item_features_rows
        ])
        
      
        item_features_np[:, 0] = np.array([
            item_id_map[item_id] for item_id in item_features_np[:, 0]
        ])
        
       
        n_users = int(np.max(interactions_np[:, 0]) + 1)
        n_items = len(unique_item_ids)
        
        interaction_matrix = csr_matrix(
            (
                interactions_np[:, 2],  
                (
                    interactions_np[:, 0].astype(int),  
                    interactions_np[:, 1].astype(int)   
                )
            ),
            shape=(n_users, n_items)
        )
        
        item_features_matrix = csr_matrix(
            (
                item_features_np[:, 2],  
                (
                    item_features_np[:, 0].astype(int),  
                    item_features_np[:, 1].astype(int)   
                )
            ),
            shape=(n_items, 100)  
        )
        
       
        logger.info(f"""
        Estatísticas do dataset:
        - Total de interações: {len(interactions_rows)}
        - Usuários únicos: {n_users}
        - Itens únicos: {n_items}
        """)
        
        return interaction_matrix, item_features_matrix
        
    def train(self, interactions, item_features=None, epochs=30):
        """
        Train the LightFM model.

        Args:
            interactions (csr_matrix): The interaction matrix for training.
            item_features (csr_matrix, optional): The item features matrix. Defaults to None.
            epochs (int, optional): The number of epochs to train the model. Defaults to 30.

        Returns:
            None
        """
        logger.info("Iniciando treinamento...")
        
        self.model.fit(
            interactions=interactions,
            item_features=item_features,
            epochs=epochs,
            verbose=True
        )
        
        logger.info("Treinamento concluído")
        
    def evaluate(self, test_interactions, item_features=None):
        """
        Evaluate the model on test data.

        Args:
            test_interactions (csr_matrix): The interaction matrix for testing.
            item_features (csr_matrix, optional): The item features matrix. Defaults to None.

        Returns:
            dict: A dictionary containing evaluation metrics (precision@10, recall@10, auc).
        """
        try:
           
            precision = precision_at_k(
                self.model, 
                test_interactions, 
                k=10,
               
            ).mean()
            
            recall = recall_at_k(
                self.model, 
                test_interactions, 
                k=10,
                
            ).mean()
            
            auc = auc_score(
                self.model, 
                test_interactions,
                
            ).mean()
            
            metrics = {
                'precision@10': precision,
                'recall@10': recall,
                'auc': auc
            }
            
            logger.info(f"Métricas de avaliação: {metrics}")
            return metrics
            
        except Exception as e:
            logger.error(f"Erro na avaliação: {str(e)}")
        
            return {
                'precision@10': 0.0,
                'recall@10': 0.0,
                'auc': 0.0
            }
    
    def save_model(self, path):
        """
        Save the trained model to a specified path.

        Args:
            path (str): The path where the model will be saved.

        Returns:
            None
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(self.model, f)
        logger.info(f"Modelo salvo em {path}")
    
    def load_model(self, path):
        """
        Load a saved model from a specified path.

        Args:
            path (str): The path from where the model will be loaded.

        Returns:
            None
        """
        with open(path, 'rb') as f:
            self.model = pickle.load(f)
        logger.info(f"Modelo carregado de {path}") 