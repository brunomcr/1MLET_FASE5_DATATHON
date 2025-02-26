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
                 model_output_path=None, test_ratio=0.2):
        """
        Initialize the LightFMTrainer.

        Args:
            spark (SparkSession, optional): A Spark session for processing data.
            interactions_path (str, optional): Path to the interaction matrix.
            user_features_path (str, optional): Path to the user features.
            item_features_path (str, optional): Path to the item features.
            model_output_path (str, optional): Path to save the trained model.
            test_ratio (float, optional): The ratio of data to be used for testing. Must be between 0 and 1.

        Raises:
            ValueError: If test_ratio is not between 0 and 1.

        Returns:
            None
        """
        self.spark = spark
        self.interactions_path = interactions_path
        self.user_features_path = user_features_path
        self.item_features_path = item_features_path
        self.model_output_path = model_output_path
        self.model = None
        self.test_ratio = test_ratio
        self.train = None 
        self.test = None  

       
        if not 0 < test_ratio < 1:
            raise ValueError("test_ratio must be between 0 and 1")

    def load_data(self):
        """Load interaction matrix and features.

        This method loads the interaction matrix and user/item features from the specified paths,
        converting them into sparse matrices for model training.

        Returns:
            None
        """
        try:
            logger.info("Loading data...")

           
            try:
                logger.info(f"Loading interaction matrix from {self.interactions_path}")
                self.interaction_matrix = sp.load_npz(self.interactions_path)
                logger.info(f"Interaction matrix loaded: shape={self.interaction_matrix.shape}, "
                            f"non-zero elements={self.interaction_matrix.nnz}, "
                            f"density={self.interaction_matrix.nnz / (self.interaction_matrix.shape[0] * self.interaction_matrix.shape[1]):.4%}")

              
                if self.interaction_matrix.nnz == 0:
                    raise ValueError("Loaded interaction matrix is empty")

            except Exception as e:
                logger.error(f"Error loading interaction matrix: {str(e)}")
                raise

          
            try:
                logger.info(f"Loading user features from {self.user_features_path}")
                user_features_df = self.spark.read.parquet(self.user_features_path)
                user_count = user_features_df.count()
                logger.info(f"User features loaded: {user_count} users")
            except Exception as e:
                logger.error(f"Error loading user features: {str(e)}")
                raise

          
            try:
                logger.info(f"Loading item features from {self.item_features_path}")
                item_features_df = self.spark.read.parquet(self.item_features_path)
                item_count = item_features_df.count()
                logger.info(f"Item features loaded: {item_count} items")
            except Exception as e:
                logger.error(f"Error loading item features: {str(e)}")
                raise

            
            try:
                logger.info("Converting user features to sparse matrix...")
                self.user_features = self._convert_features_to_sparse(user_features_df, "user_features")
                logger.info(f"User features matrix shape: {self.user_features.shape}")

                logger.info("Converting item features to sparse matrix...")
                self.item_features = self._convert_features_to_sparse(item_features_df, "features")
                logger.info(f"Item features matrix shape: {self.item_features.shape}")
            except Exception as e:
                logger.error(f"Error converting features to sparse matrices: {str(e)}")
                raise


            try:
                if self.user_features.shape[0] != self.interaction_matrix.shape[0]:
                    raise ValueError(
                        f"Mismatch in user dimensions: interaction_matrix={self.interaction_matrix.shape[0]}, "
                        f"user_features={self.user_features.shape[0]}")

                if self.item_features.shape[0] != self.interaction_matrix.shape[1]:
                    raise ValueError(
                        f"Mismatch in item dimensions: interaction_matrix={self.interaction_matrix.shape[1]}, "
                        f"item_features={self.item_features.shape[0]}")
            except Exception as e:
                logger.error(f"Error in final validations: {str(e)}")
                raise

            logger.info("Data loading completed successfully")

        except Exception as e:
            logger.error(f"Error in load_data: {str(e)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise

    def _convert_features_to_sparse(self, df, feature_col):
        """Convert DataFrame features to sparse matrix.

        Args:
            df (DataFrame): The DataFrame containing features.
            feature_col (str): The name of the column containing the features to convert.

        Returns:
            csr_matrix: A sparse matrix representation of the features.
        """
       
        features_list = [row[feature_col].toArray() for row in df.select(feature_col).collect()]
       
        features_array = np.array(features_list)
       
        if features_array.ndim > 2:
            features_array = features_array.reshape(features_array.shape[0], -1)
        return sp.csr_matrix(features_array)

    def split_data(self):
        """Split interaction matrix into train and test sets.

        This method creates a train-test split based on the specified test ratio, ensuring that
        the train and test sets do not overlap.

        Returns:
            None
        """
        logger.info("Splitting data...")
        logger.info(f"Interaction matrix shape: {self.interaction_matrix.shape}")
        logger.info(f"Number of non-zero interactions: {self.interaction_matrix.nnz}")
        logger.info(f"Using test_ratio: {self.test_ratio}")

        try:
           
            logger.info("Creating test mask...")

           
            nonzero = self.interaction_matrix.nonzero()
            num_nonzero = len(nonzero[0])

        
            num_test = int(num_nonzero * self.test_ratio)

         
            test_indices = np.random.choice(num_nonzero, num_test, replace=False)

       
            train_mask = sp.csr_matrix(self.interaction_matrix.shape, dtype=np.bool_)
            test_mask = sp.csr_matrix(self.interaction_matrix.shape, dtype=np.bool_)

        
            train_indices = np.ones(num_nonzero, dtype=bool)
            train_indices[test_indices] = False

            train_mask[nonzero[0][train_indices], nonzero[1][train_indices]] = True
            test_mask[nonzero[0][test_indices], nonzero[1][test_indices]] = True

            logger.info(f"Train mask shape: {train_mask.shape}, non-zero elements: {train_mask.nnz}")
            logger.info(f"Test mask shape: {test_mask.shape}, non-zero elements: {test_mask.nnz}")

           
            self.train = self.interaction_matrix.multiply(train_mask)  
            self.test = self.interaction_matrix.multiply(test_mask)  

            logger.info(f"Train matrix: {self.train.nnz} interactions")
            logger.info(f"Test matrix: {self.test.nnz} interactions")

        except Exception as e:
            logger.error(f"Error splitting data: {str(e)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise

    def train_model(self, epochs=30):
        """Train the LightFM model.

        Args:
            epochs (int, optional): The number of epochs to train the model. Defaults to 30.

        Returns:
            None
        """
        logger.info("Training model...")

       
        self.model = LightFM(
            learning_rate=0.05,
            loss='warp',    
            no_components=100,  
            item_alpha=1e-6,  
            user_alpha=1e-6,  
            random_state=42
        )

       
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
        """Evaluate model performance.

        This method calculates precision and AUC metrics for both the training and testing datasets.

        Returns:
            dict: A dictionary containing evaluation metrics (train_precision@10, test_precision@10, train_auc, test_auc).
        """
        logger.info("Evaluating model...")

       
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
        """Save trained model and features.

        This method saves the trained LightFM model to the specified output path.

        Returns:
            None
        """
        logger.info("Saving model and features...")


        os.makedirs(os.path.dirname(self.model_output_path), exist_ok=True)

       
        with open(self.model_output_path, 'wb') as f:
            pickle.dump(self.model, f)

        logger.info(f"Model saved to {self.model_output_path}")
