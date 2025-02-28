from lightfm.evaluation import precision_at_k, recall_at_k, auc_score
import numpy as np
from scipy import stats
from datetime import datetime
import json
import os
from utils.logger import logger
import pandas as pd
from typing import Dict, List, Tuple, Any
import pickle
import scipy.sparse as sp


class ModelMonitoring:
    def __init__(self, model_path: str, monitoring_output_path: str):
        """
        Inicializa o monitoramento do modelo
        
        Args:
            model_path: Caminho para o modelo salvo
            monitoring_output_path: Caminho para salvar os resultados do monitoramento
        """
        self.model_path = model_path
        self.monitoring_output_path = monitoring_output_path
        self.model = None
        self.monitoring_results = {}
        
        # Criar diretório se não existir
        os.makedirs(monitoring_output_path, exist_ok=True)
        
    def load_model(self):
        """Carrega o modelo salvo"""
        try:
            with open(self.model_path, 'rb') as f:
                self.model = pickle.load(f)
            logger.info("Modelo carregado com sucesso para monitoramento")
        except Exception as e:
            logger.error(f"Erro ao carregar o modelo para monitoramento: {str(e)}")
            raise
            
    def get_model_summary(self) -> Dict[str, Any]:
        """Retorna um resumo do modelo"""
        try:
            if self.model is None:
                self.load_model()
                
            return {
                "model_type": "LightFM",
                "embedding_dim": self.model.no_components,
                "loss_function": self.model.loss,
                "learning_schedule": self.model.learning_schedule,
                "learning_rate": self.model.learning_rate,
                "model_size_mb": os.path.getsize(self.model_path) / (1024 * 1024),
                "last_modified": datetime.fromtimestamp(os.path.getmtime(self.model_path)).strftime('%Y-%m-%d %H:%M:%S')
            }
        except Exception as e:
            logger.error(f"Erro ao gerar resumo do modelo: {str(e)}")
            return {
                "error": "Não foi possível gerar o resumo do modelo",
                "details": str(e)
            }
        
    def get_hyperparameters(self) -> Dict[str, Any]:
        """Retorna os hiperparâmetros do modelo"""
        try:
            if self.model is None:
                self.load_model()
                
            return {
                "no_components": self.model.no_components,
                "learning_rate": self.model.learning_rate,
                "loss": self.model.loss,
                "item_alpha": self.model.item_alpha,
                "user_alpha": self.model.user_alpha,
                "max_sampled": self.model.max_sampled,
                "random_state": self.model.random_state
            }
        except Exception as e:
            logger.error(f"Erro ao recuperar hiperparâmetros: {str(e)}")
            return {"error": "Não foi possível recuperar os hiperparâmetros"}
        
    def calculate_performance_metrics(self, test_interactions, k_values=[5, 10, 20]) -> Dict[str, float]:
        """Calcula métricas de desempenho para diferentes valores de k"""
        try:
            if self.model is None:
                self.load_model()
                
            metrics = {}
            
            for k in k_values:
                metrics[f'precision@{k}'] = float(precision_at_k(self.model, test_interactions, k=k).mean())
                metrics[f'recall@{k}'] = float(recall_at_k(self.model, test_interactions, k=k).mean())
                
            metrics['auc'] = float(auc_score(self.model, test_interactions).mean())
            
            return metrics
        except Exception as e:
            logger.error(f"Erro ao calcular métricas de desempenho: {str(e)}")
            return {"error": "Não foi possível calcular as métricas de desempenho"}
        
    def analyze_interaction_distribution(self, interaction_matrix) -> Dict[str, Any]:
        """Analisa a distribuição das interações"""
        try:
            # Garantir que a matriz está no formato CSR
            if not sp.isspmatrix_csr(interaction_matrix):
                interaction_matrix = interaction_matrix.tocsr()
            
            user_interactions = np.asarray(interaction_matrix.sum(axis=1)).flatten()
            item_interactions = np.asarray(interaction_matrix.sum(axis=0)).flatten()
            
            return {
                "user_interactions": {
                    "mean": float(np.mean(user_interactions)),
                    "median": float(np.median(user_interactions)),
                    "std": float(np.std(user_interactions)),
                    "min": float(np.min(user_interactions)),
                    "max": float(np.max(user_interactions)),
                    "distribution": np.histogram(user_interactions, bins=50)[0].tolist()
                },
                "item_interactions": {
                    "mean": float(np.mean(item_interactions)),
                    "median": float(np.median(item_interactions)),
                    "std": float(np.std(item_interactions)),
                    "min": float(np.min(item_interactions)),
                    "max": float(np.max(item_interactions)),
                    "distribution": np.histogram(item_interactions, bins=50)[0].tolist()
                }
            }
        except Exception as e:
            logger.error(f"Erro ao analisar distribuição de interações: {str(e)}")
            return {"error": "Não foi possível analisar a distribuição de interações"}
        
    def analyze_model_stability(self, train_interactions, test_interactions) -> Dict[str, Any]:
        """Analisa a estabilidade do modelo entre treino e teste"""
        try:
            if self.model is None:
                self.load_model()
                
            # Garantir que as matrizes estão no formato CSR
            if not sp.isspmatrix_csr(train_interactions):
                train_interactions = train_interactions.tocsr()
            if not sp.isspmatrix_csr(test_interactions):
                test_interactions = test_interactions.tocsr()
                
            # Usar apenas uma amostra para predições para evitar memória excessiva
            max_users = 10000
            max_items = 10000
            
            n_users = min(train_interactions.shape[0], max_users)
            n_items = min(train_interactions.shape[1], max_items)
            
            user_indices = np.random.choice(train_interactions.shape[0], n_users, replace=False)
            item_indices = np.random.choice(train_interactions.shape[1], n_items, replace=False)
            
            train_predictions = self.model.predict(
                user_ids=user_indices,
                item_ids=item_indices
            )
            
            test_predictions = self.model.predict(
                user_ids=user_indices,
                item_ids=item_indices
            )
            
            return {
                "prediction_stats": {
                    "train_mean": float(np.mean(train_predictions)),
                    "train_std": float(np.std(train_predictions)),
                    "test_mean": float(np.mean(test_predictions)),
                    "test_std": float(np.std(test_predictions)),
                },
                "ks_test": float(stats.ks_2samp(train_predictions, test_predictions)[0])
            }
        except Exception as e:
            logger.error(f"Erro ao analisar estabilidade do modelo: {str(e)}")
            return {"error": "Não foi possível analisar a estabilidade do modelo"}
        
    def analyze_feature_importance(self, item_features=None, user_features=None) -> Dict[str, List[float]]:
        """Analisa a importância das features baseado nos embeddings do modelo"""
        try:
            if self.model is None:
                self.load_model()
                
            importance = {}
            
            # Analisar embeddings do modelo mesmo sem features
            if hasattr(self.model, 'item_embeddings'):
                item_embeddings = np.abs(self.model.item_embeddings)
                importance["item_embeddings"] = np.mean(item_embeddings, axis=0).tolist()
            
            if hasattr(self.model, 'user_embeddings'):
                user_embeddings = np.abs(self.model.user_embeddings)
                importance["user_embeddings"] = np.mean(user_embeddings, axis=0).tolist()
            
            # Analisar features específicas se fornecidas
            if item_features is not None and sp.issparse(item_features):
                item_feature_importance = np.abs(item_features.mean(axis=0)).A1
                importance["item_features"] = item_feature_importance.tolist()
            
            if user_features is not None and sp.issparse(user_features):
                user_feature_importance = np.abs(user_features.mean(axis=0)).A1
                importance["user_features"] = user_feature_importance.tolist()
            
            return importance
        except Exception as e:
            logger.error(f"Erro ao analisar importância das features: {str(e)}")
            return {"error": "Não foi possível analisar a importância das features"}
        
    def save_monitoring_results(self):
        """Salva os resultados do monitoramento"""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_file = os.path.join(self.monitoring_output_path, f'monitoring_results_{timestamp}.json')
            
            with open(output_file, 'w') as f:
                json.dump(self.monitoring_results, f, indent=4)
            
            logger.info(f"Resultados do monitoramento salvos em: {output_file}")
        except Exception as e:
            logger.error(f"Erro ao salvar resultados do monitoramento: {str(e)}")
        
    def run_full_monitoring(self, train_interactions, test_interactions, 
                          item_features=None, user_features=None) -> Dict[str, Any]:
        """Executa todas as análises de monitoramento"""
        try:
            # Garantir que as matrizes de interação estão no formato correto
            if not sp.isspmatrix_csr(train_interactions):
                train_interactions = train_interactions.tocsr()
            if not sp.isspmatrix_csr(test_interactions):
                test_interactions = test_interactions.tocsr()
            
            self.monitoring_results = {
                "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                "model_summary": self.get_model_summary(),
                "hyperparameters": self.get_hyperparameters(),
                "performance_metrics": self.calculate_performance_metrics(test_interactions),
                "interaction_distribution": self.analyze_interaction_distribution(train_interactions),
                "model_stability": self.analyze_model_stability(train_interactions, test_interactions),
                "feature_importance": self.analyze_feature_importance(item_features, user_features)
            }
            
            self.save_monitoring_results()
            return self.monitoring_results
        except Exception as e:
            logger.error(f"Erro ao executar monitoramento completo: {str(e)}")
            return {
                "error": "Falha ao executar monitoramento completo",
                "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                "details": str(e)
            } 