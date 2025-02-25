import logging
from typing import List, Dict, Tuple

from lightfm import LightFM
import numpy as np
import scipy.sparse as sp
import pickle
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, udf, array
from pyspark.sql.types import ArrayType, FloatType, StringType, StructType, StructField


logger = logging.getLogger(__name__)


class LightFMPredictor:
    def __init__(
        self, 
        spark_session: SparkSession, 
        model_path: str, 
        user_features_path: str, 
        item_features_path: str
    ):
        """
        Initialize the predictor
        
        Args:
            spark_session: Active Spark session
            model_path: Path to saved LightFM model
            user_features_path: Path to user features parquet
            item_features_path: Path to item features parquet
        """
        self.spark = spark_session
        self.model = self._load_model(model_path)
        self.user_features_path = user_features_path
        self.item_features_path = item_features_path
        self.user_features = None
        self.item_features = None
        self.item_ids = None
        self._load_features()

    def _load_model(self, model_path: str) -> LightFM:
        """Load the trained LightFM model"""
        logger.info(f"Loading model from {model_path}")
        try:
            with open(model_path, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise

    def _load_features(self):
        """Load and prepare user and item features"""
        logger.info("Loading features...")
        
        try:
            # Load user features
            user_features_df = self.spark.read.parquet(self.user_features_path)
            self.user_features = self._convert_features_to_sparse(user_features_df, "user_features")
            
            # Load item features
            item_features_df = self.spark.read.parquet(self.item_features_path)
            self.item_features = self._convert_features_to_sparse(item_features_df, "features")
            
            # Store item IDs for mapping predictions back to items
            self.item_ids = [row.page for row in item_features_df.select("page").collect()]
            
            logger.info("Features loaded successfully")
        except Exception as e:
            logger.error(f"Error loading features: {str(e)}")
            raise

    def _convert_features_to_sparse(self, df, feature_col):
        """Convert DataFrame features to sparse matrix"""
        features_list = [row[feature_col].toArray() for row in df.select(feature_col).collect()]
        features_array = np.array(features_list)
        if features_array.ndim > 2:
            features_array = features_array.reshape(features_array.shape[0], -1)
        return sp.csr_matrix(features_array)

    def predict_for_user(self, user_id: str, n_items: int = 10) -> list:
        """
        Generate recommendations for a single user
        
        Args:
            user_id: User ID to generate predictions for
            n_items: Number of items to recommend
            
        Returns:
            List of tuples (item_id, score)
        """
        try:
            # Get user features
            user_df = self.spark.read.parquet(self.user_features_path).filter(col("userId") == user_id)
            if user_df.count() == 0:
                logger.warning(f"User {user_id} not found in features")
                return []

            user_features = self._convert_features_to_sparse(user_df, "user_features")
            
            # Generate predictions for all items
            n_items_total = len(self.item_ids)
            user_ids = np.repeat(0, n_items_total)  # Create array of same length as items
            item_ids = np.arange(n_items_total)
            
            # Generate predictions
            scores = self.model.predict(
                user_ids=user_ids,
                item_ids=item_ids,
                user_features=user_features,
                item_features=self.item_features
            )

            # Normalize scores to [0,1] range
            scores = (scores - scores.min()) / (scores.max() - scores.min())
            
            # Get top N items
            top_items_idx = np.argsort(-scores)[:n_items]
            recommendations = [(self.item_ids[idx], float(scores[idx])) for idx in top_items_idx]
            
            return recommendations

        except Exception as e:
            logger.error(f"Error generating predictions for user {user_id}: {str(e)}")
            return []

    def batch_predict(self, user_ids: list, n_items: int = 10) -> dict:
        """
        Generate predictions for multiple users
        
        Args:
            user_ids: List of user IDs
            n_items: Number of items to recommend per user
            
        Returns:
            Dictionary mapping user IDs to their recommendations
        """
        recommendations = {}
        for user_id in user_ids:
            recommendations[user_id] = self.predict_for_user(user_id, n_items)
        return recommendations