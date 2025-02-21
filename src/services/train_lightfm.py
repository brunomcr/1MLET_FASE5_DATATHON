from lightfm import LightFM
from lightfm.evaluation import precision_at_k, auc_score
import numpy as np
import scipy.sparse as sp
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
import os
import pickle
from utils.logger import logger


class LightFMTrainer:
    def __init__(self, spark=None, interactions_path=None, user_features_path=None, item_features_path=None,
                 model_output_path=None):
        self.spark = spark
        self.interactions_path = interactions_path
        self.user_features_path = user_features_path
        self.item_features_path = item_features_path
        self.model_output_path = model_output_path
        self.model = None

    def load_data(self):
        """Load interaction matrix and features"""
        logger.info("Loading data...")

        # Load interaction matrix
        self.interaction_matrix = sp.load_npz(self.interactions_path)

        # Load user features from Parquet
        user_features_df = self.spark.read.parquet(self.user_features_path)

        # Load item features from Parquet
        item_features_df = self.spark.read.parquet(self.item_features_path)

        # Convert features to sparse matrices
        self.user_features = self._convert_features_to_sparse(user_features_df, "user_features")
        self.item_features = self._convert_features_to_sparse(item_features_df, "features")

        logger.info(f"Data loaded: {self.interaction_matrix.shape[0]} users, {self.interaction_matrix.shape[1]} items")

    def _convert_features_to_sparse(self, df, feature_col):
        """Convert DataFrame features to sparse matrix"""
        features_array = np.array(df.select(feature_col).collect())
        return sp.csr_matrix(features_array)

    def split_data(self, test_ratio=0.2):
        """Split data into train and test sets"""
        logger.info("Splitting data...")

        # Create mask for test set
        test_mask = np.random.random(self.interaction_matrix.shape) < test_ratio

        # Ensure each user has at least one interaction in training set
        test_mask = self._ensure_min_interactions(test_mask)

        # Create train and test matrices
        self.train = self.interaction_matrix.multiply(~test_mask)
        self.test = self.interaction_matrix.multiply(test_mask)

        logger.info("Data split completed")

    def _ensure_min_interactions(self, test_mask):
        """Ensure each user has at least one interaction in training set"""
        n_users = test_mask.shape[0]
        for user_id in range(n_users):
            user_interactions = self.interaction_matrix[user_id].nonzero()[1]
            if len(user_interactions) > 0:
                if test_mask[user_id].sum() >= len(user_interactions):
                    # Keep at least one interaction in training
                    test_mask[user_id, np.random.choice(user_interactions)] = False
        return test_mask

    def train_model(self, epochs=30):
        """Train the LightFM model"""
        logger.info("Training model...")

        # Initialize model with optimal parameters
        self.model = LightFM(
            learning_rate=0.05,
            loss='warp',  # Weighted Approximate-Rank Pairwise
            no_components=100,  # Match with TF-IDF features
            item_alpha=1e-6,  # L2 regularization for item features
            user_alpha=1e-6,  # L2 regularization for user features
            random_state=42
        )

        # Train model
        self.model.fit(
            interactions=self.train,
            user_features=self.user_features,
            item_features=self.item_features,
            epochs=epochs,
            num_threads=4,
            verbose=True
        )

        logger.info("Model training completed")

    def evaluate_model(self):
        """Evaluate model performance"""
        logger.info("Evaluating model...")

        # Calculate metrics
        train_precision = precision_at_k(
            self.model,
            self.train,
            k=10,
            user_features=self.user_features,
            item_features=self.item_features,
            num_threads=4
        ).mean()

        test_precision = precision_at_k(
            self.model,
            self.test,
            k=10,
            user_features=self.user_features,
            item_features=self.item_features,
            num_threads=4
        ).mean()

        train_auc = auc_score(
            self.model,
            self.train,
            user_features=self.user_features,
            item_features=self.item_features,
            num_threads=4
        ).mean()

        test_auc = auc_score(
            self.model,
            self.test,
            user_features=self.user_features,
            item_features=self.item_features,
            num_threads=4
        ).mean()

        metrics = {
            'train_precision@10': train_precision,
            'test_precision@10': test_precision,
            'train_auc': train_auc,
            'test_auc': test_auc
        }

        logger.info(f"Evaluation metrics: {metrics}")
        return metrics

    def save_model(self):
        """Save trained model and features"""
        logger.info("Saving model and features...")

        # Create models directory if it doesn't exist
        os.makedirs(os.path.dirname(self.model_output_path), exist_ok=True)

        # Save model
        with open(self.model_output_path, 'wb') as f:
            pickle.dump(self.model, f)

        logger.info(f"Model saved to {self.model_output_path}")
