from lightfm import LightFM
from lightfm.evaluation import precision_at_k, recall_at_k, auc_score, reciprocal_rank
import numpy as np
import scipy.sparse as sp
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, rand
import os
import pickle
from utils.logger import logger
from services.model_monitoring import ModelMonitoring
from datetime import datetime
import json
import time


class LightFMTrainer:
    def __init__(self, spark=None, interactions_path=None, user_features_path=None, item_features_path=None,
                 model_output_path=None, test_ratio=0.2):
        self.spark = spark
        self.interactions_path = interactions_path
        self.user_features_path = user_features_path
        self.item_features_path = item_features_path
        self.model_output_path = model_output_path
        self.model = LightFM(
            learning_rate=0.05,
            loss='warp',        # Weighted Approximate-Rank Pairwise
            no_components=100,   # Ajustar para match com número de features TF-IDF
            item_alpha=1e-6,     # L2 regularização para item features
            user_alpha=1e-6,     # L2 regularização para user features
            random_state=42
        )
        self.test_ratio = test_ratio
        self.train = None
        self.test = None
        self.monitoring = None
        
        # Validar test_ratio
        if not 0 < test_ratio < 1:
            raise ValueError("test_ratio must be between 0 and 1")
            
    def load_data(self):
        """Load interaction matrix and features"""
        try:
            logger.info("Starting data loading...")

            # Load interaction matrix
            try:
                logger.info(f"Loading interaction matrix from {self.interactions_path}")
                self.interaction_matrix = sp.load_npz(self.interactions_path)
                logger.info(f"Interaction matrix loaded: shape={self.interaction_matrix.shape}, "
                            f"non-zero elements={self.interaction_matrix.nnz}, "
                            f"density={self.interaction_matrix.nnz / (self.interaction_matrix.shape[0] * self.interaction_matrix.shape[1]):.4%}")

                # Validar se a matriz não está vazia
                if self.interaction_matrix.nnz == 0:
                    raise ValueError("Loaded interaction matrix is empty")

            except Exception as e:
                logger.error(f"Error loading interaction matrix: {str(e)}")
                raise

            # Load user features if available
            if self.user_features_path:
                try:
                    logger.info(f"Loading user features from {self.user_features_path}")
                    user_features_df = self.spark.read.parquet(self.user_features_path)
                    self.user_features = self._convert_features_to_sparse(user_features_df, "user_features")
                    logger.info(f"User features loaded: shape={self.user_features.shape}")
                except Exception as e:
                    logger.warning(f"Could not load user features: {str(e)}")
                    self.user_features = None

            # Load item features if available
            if self.item_features_path:
                try:
                    logger.info(f"Loading item features from {self.item_features_path}")
                    item_features_df = self.spark.read.parquet(self.item_features_path)
                    self.item_features = self._convert_features_to_sparse(item_features_df, "features")
                    logger.info(f"Item features loaded: shape={self.item_features.shape}")
                except Exception as e:
                    logger.warning(f"Could not load item features: {str(e)}")
                    self.item_features = None

            logger.info("Data loading completed.")

        except Exception as e:
            logger.error(f"Error in load_data: {str(e)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise

    def _convert_features_to_sparse(self, df, feature_col):
        """Convert DataFrame features to sparse matrix"""
        try:
            logger.info("Starting conversion to sparse matrix for user features...")
            
            # Primeiro, vamos verificar a estrutura dos dados
            sample_row = df.select(feature_col).first()
            if sample_row is None:
                raise ValueError(f"Nenhum dado encontrado na coluna {feature_col}")
                
            # Se os dados já estiverem em formato de array
            features_array = np.array([row[feature_col] for row in df.select(feature_col).collect()])
            
            # Garantir que temos uma matriz 2D
            if features_array.ndim > 2:
                features_array = features_array.reshape(features_array.shape[0], -1)
            
            # Converter para matriz esparsa
            sparse_matrix = sp.csr_matrix(features_array)
            
            logger.info(f"Matriz esparsa criada com shape: {sparse_matrix.shape}")
            logger.info("Conversion to sparse matrix for user features completed.")
            return sparse_matrix
            
        except Exception as e:
            logger.error(f"Erro ao converter features para matriz esparsa: {str(e)}")
            logger.error(f"Shape dos dados: {features_array.shape if 'features_array' in locals() else 'N/A'}")
            return None

    def split_data(self):
        """Split data into train and test sets"""
        try:
            logger.info("Starting data splitting...")
            
            # Get indices of all non-zero elements
            nonzero = self.interaction_matrix.nonzero()
            num_interactions = len(nonzero[0])
            
            # Generate random mask for test set
            np.random.seed(42)
            test_mask = np.random.rand(num_interactions) < self.test_ratio
            
            # Create train and test matrices
            train_matrix = self.interaction_matrix.copy()
            test_matrix = sp.csr_matrix(self.interaction_matrix.shape)
            
            # Assign interactions to train/test sets
            for i in range(num_interactions):
                if test_mask[i]:
                    train_matrix[nonzero[0][i], nonzero[1][i]] = 0
                    test_matrix[nonzero[0][i], nonzero[1][i]] = self.interaction_matrix[nonzero[0][i], nonzero[1][i]]
            
            # Compress matrices
            self.train = train_matrix.tocsr()
            self.test = test_matrix.tocsr()
            
            logger.info(f"Train shape: {self.train.shape}, Test shape: {self.test.shape}")
            logger.info(f"Train interactions: {self.train.nnz}, Test interactions: {self.test.nnz}")
            
            logger.info("Data splitting completed.")
            
        except Exception as e:
            logger.error(f"Error splitting data: {str(e)}")
            raise

    def train_model(self, epochs=30):
        """Treinar o modelo"""
        try:
            logger.info(f"Training model for {epochs} epochs...")
            start_time = time.time()
            
            # Ajustar no_components para match com as features
            if self.item_features is not None:
                self.model.no_components = self.item_features.shape[1]
                logger.info(f"Adjusted model components to match item features: {self.model.no_components}")
            
            self.model.fit(
                interactions=self.train,
                item_features=self.item_features,
                epochs=epochs,
                verbose=True
            )
            
            training_time = time.time() - start_time
            logger.info(f"Model training completed in {training_time:.2f} seconds.")
            
            # Avaliar e salvar métricas após o treinamento
            self.evaluate_model(training_time)
            
            # Salvar o modelo
            self.save_model()
            
        except Exception as e:
            logger.error(f"Error in model training: {str(e)}")
            raise

    def evaluate_model(self, training_time=None, sample_size=100.0):
        """Avaliar modelo em dados de teste"""
        try:
            logger.info("Starting model evaluation...")
            logger.info(f"Evaluating with {sample_size}% of the data...")
            
            # Apply sample size to test data
            if sample_size < 100.0:
                logger.info(f"Sampling {sample_size}% of the test data for evaluation...")
                num_test_interactions = self.test.nnz
                sample_indices = np.random.choice(
                    num_test_interactions, 
                    size=int(num_test_interactions * (sample_size / 100.0)), 
                    replace=False
                )
                sampled_test = sp.csr_matrix(self.test.shape)
                sampled_test[self.test.nonzero()[0][sample_indices], self.test.nonzero()[1][sample_indices]] = self.test.data[sample_indices]
                test_data = sampled_test
            else:
                test_data = self.test

            # 1. Métricas básicas de desempenho
            logger.info("Calculating precision@10...")
            precision = precision_at_k(
                self.model, 
                test_data, 
                item_features=self.item_features,
                k=10
            ).mean()
            
            logger.info("Calculating recall@10...")
            recall = recall_at_k(
                self.model, 
                test_data, 
                item_features=self.item_features,
                k=10
            ).mean()
            
            logger.info("Calculating AUC...")
            auc = auc_score(
                self.model, 
                test_data,
                item_features=self.item_features
            ).mean()

            # 2. Métricas de qualidade das recomendações
            logger.info("Calculating NDCG@10...")
            ndcg = precision_at_k(
                self.model, 
                test_data, 
                item_features=self.item_features,
                k=10
            ).mean()
            
            logger.info("Calculating MRR...")
            mrr = reciprocal_rank(
                self.model, 
                test_data, 
                item_features=self.item_features
            ).mean()

            # 3. Estatísticas dos embeddings
            logger.info("Calculating embedding statistics...")
            user_embedding_norms = np.linalg.norm(self.model.user_embeddings, axis=1)
            item_embedding_norms = np.linalg.norm(self.model.item_embeddings, axis=1)
            embedding_stats = {
                "user_embedding_norm_mean": float(np.mean(user_embedding_norms)),
                "user_embedding_norm_std": float(np.std(user_embedding_norms)),
                "item_embedding_norm_mean": float(np.mean(item_embedding_norms)),
                "item_embedding_norm_std": float(np.std(item_embedding_norms))
            }

            # 4. Calcular importância das features
            logger.info("Calculating feature importance...")
            if self.item_features is not None:
                item_embeddings = self.model.item_embeddings
                feature_importance = np.abs(item_embeddings).mean(axis=0)
                feature_importance = feature_importance / feature_importance.sum()
                feature_importance_dict = {f"feature_{i}": float(imp) for i, imp in enumerate(feature_importance)}
            else:
                feature_importance_dict = {}

            # 5. Análise de distribuição de interações
            logger.info("Calculating interaction distribution...")
            interaction_stats = {
                "total_interactions": int(self.train.nnz + self.test.nnz),
                "train_interactions": int(self.train.nnz),
                "test_interactions": int(self.test.nnz),
                "interactions_per_user": float(self.train.nnz / self.train.shape[0]),
                "interactions_per_item": float(self.train.nnz / self.train.shape[1]),
                "sparsity": float(1 - (self.train.nnz / (self.train.shape[0] * self.train.shape[1])))
            }

            # 6. Estabilidade do modelo
            logger.info("Calculating model stability...")
            stability_metrics = {
                "user_coverage": float(np.sum(self.train.getnnz(axis=1) > 0) / self.train.shape[0]),
                "item_coverage": float(np.sum(self.train.getnnz(axis=0) > 0) / self.train.shape[1]),
                "cold_start_users": int(np.sum(self.train.getnnz(axis=1) == 0)),
                "cold_start_items": int(np.sum(self.train.getnnz(axis=0) == 0))
            }

            # 7. Preparar métricas completas para serialização
            logger.info("Starting serialization of metrics...")
            metrics = {
                "timestamp": datetime.now().isoformat(),
                "model_summary": {
                    "model_type": "LightFM",
                    "embedding_dim": self.model.no_components,
                    "loss_function": self.model.loss,
                    "learning_schedule": "adagrad",
                    "learning_rate": float(self.model.learning_rate),
                    "last_modified": datetime.now().isoformat(),
                    "model_size_mb": os.path.getsize(self.model_output_path) / (1024 * 1024) if os.path.exists(self.model_output_path) else None
                },
                "hyperparameters": {
                    "no_components": int(self.model.no_components),
                    "learning_rate": float(self.model.learning_rate),
                    "loss": str(self.model.loss),
                    "item_alpha": float(self.model.item_alpha),
                    "user_alpha": float(self.model.user_alpha),
                    "random_state": 42
                },
                "performance_metrics": {
                    "precision@10": float(precision),
                    "recall@10": float(recall),
                    "auc": float(auc),
                    "ndcg@10": float(ndcg),
                    "mrr": float(mrr),
                    "training_time": float(training_time) if training_time is not None else None
                },
                "feature_importance": feature_importance_dict,
                "interaction_distribution": interaction_stats,
                "model_stability": stability_metrics,
                "embedding_stats": embedding_stats,
                "dataset_info": {
                    "n_users": int(self.train.shape[0]),
                    "n_items": int(self.train.shape[1]),
                    "n_features": int(self.item_features.shape[1]) if self.item_features is not None else 0,
                    "n_train_interactions": int(self.train.nnz),
                    "n_test_interactions": int(self.test.nnz)
                },
                "sample_size": sample_size
            }
            
            # Criar diretório de monitoramento se não existir
            monitoring_dir = os.path.join(os.path.dirname(self.model_output_path), 'monitoring')
            os.makedirs(monitoring_dir, exist_ok=True)
            
            # Salvar métricas em arquivo JSON
            monitoring_file = os.path.join(monitoring_dir, f'monitoring_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
            try:
                # Primeiro, verificar se podemos serializar o objeto completo
                json_str = json.dumps(metrics, indent=4)
                
                # Se a serialização foi bem sucedida, então gravamos no arquivo
                with open(monitoring_file, 'w') as f:
                    f.write(json_str)
                    f.flush()
                    os.fsync(f.fileno())  # Força a gravação no disco
                
                # Verificar se o arquivo foi gravado corretamente
                if os.path.exists(monitoring_file):
                    with open(monitoring_file, 'r') as f:
                        _ = json.load(f)  # Tenta ler o arquivo para validar
                
                logger.info(f"Métricas de avaliação salvas com sucesso em: {monitoring_file}")
            except Exception as json_error:
                logger.error(f"Erro ao salvar arquivo JSON: {str(json_error)}")
                # Se houver erro, tenta salvar em um arquivo de backup
                backup_file = os.path.join(monitoring_dir, f'monitoring_results_backup_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
                with open(backup_file, 'w') as f:
                    json.dump(metrics, f, indent=4, default=str)
                logger.info(f"Arquivo de backup salvo em: {backup_file}")
            
            logger.info("Serialization of metrics completed.")
            return metrics
            
        except Exception as e:
            logger.error(f"Erro na avaliação: {str(e)}")
            error_metrics = {
                "timestamp": datetime.now().isoformat(),
                "error": str(e),
                "metrics": {
                    "precision@10": 0.0,
                    "recall@10": 0.0,
                    "auc": 0.0
                }
            }
            
            try:
                monitoring_dir = os.path.join(os.path.dirname(self.model_output_path), 'monitoring')
                os.makedirs(monitoring_dir, exist_ok=True)
                monitoring_file = os.path.join(monitoring_dir, f'model_metrics_error_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
                with open(monitoring_file, 'w') as f:
                    json.dump(error_metrics, f, indent=4)
            except:
                pass
                
            return error_metrics

    def save_model(self):
        """Save trained model"""
        try:
            os.makedirs(os.path.dirname(self.model_output_path), exist_ok=True)
            with open(self.model_output_path, 'wb') as f:
                pickle.dump(self.model, f)
            logger.info(f"Model saved to {self.model_output_path}")
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            raise

    def sample_data(self, sample_ratio):
        """Sample a portion of the interaction matrix"""
        if not 0 < sample_ratio <= 1:
            raise ValueError("sample_ratio must be between 0 and 1")
            
        try:
            logger.info(f"Starting data sampling...")
            
            # Get indices of all non-zero elements
            nonzero = self.interaction_matrix.nonzero()
            num_interactions = len(nonzero[0])
            
            # Generate random mask for sampling
            np.random.seed(42)
            sample_mask = np.random.rand(num_interactions) < sample_ratio
            
            # Create sampled matrix
            sampled_matrix = sp.csr_matrix(self.interaction_matrix.shape)
            
            # Copy selected interactions to sampled matrix
            for i in range(num_interactions):
                if sample_mask[i]:
                    sampled_matrix[nonzero[0][i], nonzero[1][i]] = self.interaction_matrix[nonzero[0][i], nonzero[1][i]]
            
            self.interaction_matrix = sampled_matrix
            logger.info(f"Sampled matrix has {sampled_matrix.nnz} interactions")
            
            logger.info("Data sampling completed.")
            
        except Exception as e:
            logger.error(f"Error sampling data: {str(e)}")
            raise
