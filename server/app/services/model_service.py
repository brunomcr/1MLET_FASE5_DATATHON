import logging
from typing import Dict, List

from app.ml.lightfm_predictor import LightFMPredictor
from app.schemas.prediction import Prediction


logger = logging.getLogger(__name__)


class ModelService:
    """Service layer for handling model predictions and business logic"""

    def __init__(self, predictor: LightFMPredictor):
        """
        Initialize the model service
        
        Args:
            predictor: Initialized LightFM predictor instance
        """
        self.predictor = predictor

    def predict(self, user_id: str, num_recommendations: int) -> List[Prediction]:
        """
        Generate article recommendations for a user
        
        Args:
            user_id: ID of the user to generate recommendations for
            num_recommendations: Number of recommendations to generate
            
        Returns:
            List of ArticlePrediction objects containing article IDs and probabilities
        """
        try:
            # Get raw predictions from the model
            predictions = self.predictor.predict_for_user(user_id=user_id, n_items=num_recommendations)
            
            # Convert to Prediction objects
            return [
                Prediction(article_id=article_id, score=score)
                for article_id, score in predictions
            ]
            
        except Exception as e:
            logger.error(f"Error generating predictions for user {user_id}: {str(e)}")
            raise

    def predict_batch(self, user_ids: List[str], num_recommendations: int) -> Dict[str, List[Prediction]]:
        """
        Generate recommendations for multiple users
        
        Args:
            user_ids: List of user IDs to generate recommendations for
            num_recommendations: Number of recommendations per user
            
        Returns:
            Dictionary mapping user IDs to lists of ArticlePrediction objects
        """
        try:
            # Get raw predictions from the model
            predictions_dict = self.predictor.batch_predict(user_ids=user_ids, n_items=num_recommendations)
            
            # Convert to Prediction objects
            return {
                user_id: [
                    Prediction(article_id=article_id, score=score)
                    for article_id, score in predictions
                ]
                for user_id, predictions in predictions_dict.items()
            }
            
        except Exception as e:
            logger.error(f"Error generating batch predictions: {str(e)}")
            raise