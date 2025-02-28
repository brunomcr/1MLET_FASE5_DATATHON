import pytest
from unittest.mock import Mock, patch

from app.services.model_service import ModelService
from app.schemas.prediction import Prediction


class TestModelService:
    @pytest.fixture
    def mock_predictor(self):
        """Create a mock LightFMPredictor"""
        mock = Mock()
        # Set up default return values
        mock.predict_for_user.return_value = [("article1", 0.9), ("article2", 0.8)]
        mock.batch_predict.return_value = {
            "user1": [("article1", 0.9), ("article2", 0.8)],
            "user2": [("article3", 0.7), ("article4", 0.6)]
        }
        return mock
    
    @pytest.fixture
    def model_service(self, mock_predictor):
        """Create a ModelService with a mock predictor"""
        return ModelService(predictor=mock_predictor)
    
    def test_predict_success(self, model_service, mock_predictor):
        """Test successful prediction for a single user"""
        # Arrange
        user_id = "test_user"
        num_recommendations = 2
        
        # Act
        result = model_service.predict(user_id, num_recommendations)
        
        # Assert
        assert len(result) == 2
        assert isinstance(result[0], Prediction)
        assert result[0].article_id == "article1"
        assert result[0].score == 0.9
        assert result[1].article_id == "article2"
        assert result[1].score == 0.8
        
        # Verify predictor was called correctly
        mock_predictor.predict_for_user.assert_called_once_with(
            user_id=user_id, n_items=num_recommendations
        )
    
    def test_predict_error_handling(self, model_service, mock_predictor):
        """Test error handling in predict method"""
        # Arrange
        mock_predictor.predict_for_user.side_effect = Exception("Prediction error")
        
        # Act & Assert
        with pytest.raises(Exception) as exc_info:
            model_service.predict("user1", 5)
        
        assert "Prediction error" in str(exc_info.value)
    
    def test_predict_empty_results(self, model_service, mock_predictor):
        """Test handling of empty prediction results"""
        # Arrange
        mock_predictor.predict_for_user.return_value = []
        
        # Act
        result = model_service.predict("user1", 5)
        
        # Assert
        assert isinstance(result, list)
        assert len(result) == 0
    
    def test_predict_batch_success(self, model_service, mock_predictor):
        """Test successful batch prediction"""
        # Arrange
        user_ids = ["user1", "user2"]
        num_recommendations = 2
        
        # Act
        result = model_service.predict_batch(user_ids, num_recommendations)
        
        # Assert
        assert len(result) == 2
        assert "user1" in result
        assert "user2" in result
        
        # Check user1 predictions
        user1_predictions = result["user1"]
        assert len(user1_predictions) == 2
        assert user1_predictions[0].article_id == "article1"
        assert user1_predictions[0].score == 0.9
        
        # Check user2 predictions
        user2_predictions = result["user2"]
        assert len(user2_predictions) == 2
        assert user2_predictions[0].article_id == "article3"
        assert user2_predictions[0].score == 0.7
        
        # Verify predictor was called correctly
        mock_predictor.batch_predict.assert_called_once_with(
            user_ids=user_ids, n_items=num_recommendations
        )
    
    def test_predict_batch_error_handling(self, model_service, mock_predictor):
        """Test error handling in batch predict method"""
        # Arrange
        mock_predictor.batch_predict.side_effect = Exception("Batch prediction error")
        
        # Act & Assert
        with pytest.raises(Exception) as exc_info:
            model_service.predict_batch(["user1", "user2"], 5)
        
        assert "Batch prediction error" in str(exc_info.value)
    
    def test_predict_batch_empty_results(self, model_service, mock_predictor):
        """Test handling of empty batch prediction results"""
        # Arrange
        mock_predictor.batch_predict.return_value = {}
        
        # Act
        result = model_service.predict_batch(["user1", "user2"], 5)
        
        # Assert
        assert isinstance(result, dict)
        assert len(result) == 0
    
    def test_predict_batch_partial_results(self, model_service, mock_predictor):
        """Test handling of partial batch prediction results"""
        # Arrange
        mock_predictor.batch_predict.return_value = {
            "user1": [("article1", 0.9)],
            "user2": []  # Empty results for user2
        }
        
        # Act
        result = model_service.predict_batch(["user1", "user2"], 5)
        
        # Assert
        assert len(result) == 2
        assert len(result["user1"]) == 1
        assert len(result["user2"]) == 0 