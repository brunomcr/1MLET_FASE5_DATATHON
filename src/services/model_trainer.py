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
        self.spark = spark
        self.model = LightFM(
            learning_rate=0.05,
            loss='warp',        # Weighted Approximate-Rank Pairwise
            no_components=100,   # Ajustar para match com número de features TF-IDF
            item_alpha=1e-6,     # L2 regularização para item features
            user_alpha=1e-6,     # L2 regularização para user features
            random_state=42
        )
        
    def prepare_matrices(self, interactions_path, item_features_path):
        """
        Preparar matrizes para treinamento do LightFM de forma otimizada
        """
        logger.info(f"Lendo dados do caminho: {interactions_path}")
        
        # 1. Ler e converter interações para numpy
        interactions_df = self.spark.read.parquet(interactions_path)
        interactions_rows = interactions_df.collect()
        
        interactions_np = np.array([
            [float(row['user_idx']), 
             float(row['item_idx']), 
             float(row['interaction_score'])] 
            for row in interactions_rows
        ])
        
        # 2. Obter itens únicos e criar mapeamento
        unique_item_ids = np.unique(interactions_np[:, 1])
        item_id_map = {old_idx: new_idx for new_idx, old_idx in enumerate(unique_item_ids)}
        
        # 3. Remapear índices das interações
        interactions_np[:, 1] = np.array([item_id_map[item_id] for item_id in interactions_np[:, 1]])
        
        # 4. Ler e filtrar features dos itens
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
        
        # 5. Remapear índices das features
        item_features_np[:, 0] = np.array([
            item_id_map[item_id] for item_id in item_features_np[:, 0]
        ])
        
        # 6. Criar matrizes esparsas
        n_users = int(np.max(interactions_np[:, 0]) + 1)
        n_items = len(unique_item_ids)
        
        interaction_matrix = csr_matrix(
            (
                interactions_np[:, 2],  # interaction_score
                (
                    interactions_np[:, 0].astype(int),  # user_idx
                    interactions_np[:, 1].astype(int)   # item_idx remapeado
                )
            ),
            shape=(n_users, n_items)
        )
        
        item_features_matrix = csr_matrix(
            (
                item_features_np[:, 2],  # embedding_value
                (
                    item_features_np[:, 0].astype(int),  # item_idx remapeado
                    item_features_np[:, 1].astype(int)   # feature_idx
                )
            ),
            shape=(n_items, 100)  # 100 features TF-IDF
        )
        
        # Logging
        logger.info(f"""
        Estatísticas do dataset:
        - Total de interações: {len(interactions_rows)}
        - Usuários únicos: {n_users}
        - Itens únicos: {n_items}
        """)
        
        return interaction_matrix, item_features_matrix
        
    def train(self, interactions, item_features=None, epochs=30):
        """
        Treinar o modelo
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
        Avaliar modelo em dados de teste
        """
        try:
            # Avaliar usando apenas interações, sem features
            precision = precision_at_k(
                self.model, 
                test_interactions, 
                k=10,
                # Não passar item_features nem user_features
            ).mean()
            
            recall = recall_at_k(
                self.model, 
                test_interactions, 
                k=10,
                # Não passar item_features nem user_features
            ).mean()
            
            auc = auc_score(
                self.model, 
                test_interactions,
                # Não passar item_features nem user_features
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
            # Retornar métricas vazias em caso de erro
            return {
                'precision@10': 0.0,
                'recall@10': 0.0,
                'auc': 0.0
            }
    
    def save_model(self, path):
        """
        Salvar modelo treinado
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(self.model, f)
        logger.info(f"Modelo salvo em {path}")
    
    def load_model(self, path):
        """
        Carregar modelo salvo
        """
        with open(path, 'rb') as f:
            self.model = pickle.load(f)
        logger.info(f"Modelo carregado de {path}") 