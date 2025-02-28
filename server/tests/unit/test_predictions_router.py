import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, MagicMock

from app.schemas.prediction import Prediction
from app.services.model_service import ModelService
from app.routers.v1.predictions import get_model_service

class TestPredictionsRouter:
    def test_predict_success(self, test_client, mock_model_service, sample_predictions):
        # Arrange
        user_id = "test_user"
        num_recommendations = 3
        mock_model_service.predict.return_value = sample_predictions

        # Act
        response = test_client.post(
            "/v1/predict/",
            json={"user_id": user_id, "num_recommendations": num_recommendations},
        )

        # Assert
        assert response.status_code == 200
        assert response.json() == {
            "user_id": user_id,
            "predictions": [
                {"article_id": "article1", "score": 0.9},
                {"article_id": "article2", "score": 0.8},
                {"article_id": "article3", "score": 0.7},
            ],
        }
        mock_model_service.predict.assert_called_once_with(
            user_id=user_id, num_recommendations=num_recommendations
        )

    def test_predict_invalid_request(self, test_client, mock_model_service):
        # Test missing user_id
        response = test_client.post(
            "/v1/predict/",
            json={"num_recommendations": 3},
        )
        assert response.status_code == 422

        # Test missing num_recommendations
        response = test_client.post(
            "/v1/predict/",
            json={"user_id": "test_user"},
        )
        assert response.status_code == 422

        # Test invalid num_recommendations
        response = test_client.post(
            "/v1/predict/",
            json={"user_id": "test_user", "num_recommendations": 0},
        )
        assert response.status_code == 422

        # Verify model service was never called
        mock_model_service.predict.assert_not_called()

    def test_predict_service_error(self, test_client, mock_model_service):
        # Arrange
        mock_model_service.predict.side_effect = Exception("Model error")

        # Act
        response = test_client.post(
            "/v1/predict/",
            json={"user_id": "test_user", "num_recommendations": 3},
        )

        # Assert
        assert response.status_code == 500
        assert "Model error" in response.json()["detail"]

    def test_predict_batch_success(self, test_client, mock_model_service, sample_predictions):
        # Arrange
        requests = [
            {"user_id": "user1", "num_recommendations": 3},
            {"user_id": "user2", "num_recommendations": 3},
        ]
        mock_model_service.predict.side_effect = [
            sample_predictions,
            sample_predictions,
        ]

        # Act
        response = test_client.post("/v1/predict/batch", json=requests)

        # Assert
        assert response.status_code == 200
        assert len(response.json()) == 2
        for user_response in response.json():
            assert len(user_response["predictions"]) == 3
            assert user_response["predictions"][0]["score"] == 0.9

        assert mock_model_service.predict.call_count == 2

    def test_predict_batch_invalid_request(self, test_client, mock_model_service):
        # Test empty batch
        response = test_client.post("/v1/predict/batch", json=[])
        assert response.status_code == 422

        # Test invalid items in batch
        response = test_client.post(
            "/v1/predict/batch",
            json=[{"user_id": "user1"}],  # Missing num_recommendations
        )
        assert response.status_code == 422

        mock_model_service.predict.assert_not_called()

    def test_predict_batch_service_error(self, test_client, mock_model_service):
        # Arrange
        requests = [
            {"user_id": "user1", "num_recommendations": 3},
            {"user_id": "user2", "num_recommendations": 3},
        ]
        mock_model_service.predict.side_effect = Exception("Batch error")

        # Act
        response = test_client.post("/v1/predict/batch", json=requests)

        # Assert
        assert response.status_code == 500
        assert "Batch error" in response.json()["detail"]
        
    def test_predict_with_different_num_recommendations(self, test_client, mock_model_service, sample_predictions):
        # Arrange
        user_id = "test_user"
        num_recommendations = 2
        # Only return 2 predictions instead of 3
        mock_model_service.predict.return_value = sample_predictions[:2]

        # Act
        response = test_client.post(
            "/v1/predict/",
            json={"user_id": user_id, "num_recommendations": num_recommendations},
        )

        # Assert
        assert response.status_code == 200
        assert len(response.json()["predictions"]) == 2
        mock_model_service.predict.assert_called_once_with(
            user_id=user_id, num_recommendations=num_recommendations
        )

    def test_get_model_service(self, mock_model_service):
        # Create a mock request with a mock app that has a container
        mock_request = MagicMock()
        mock_request.app.container.model_service.return_value = mock_model_service
        
        # Call the function
        service = get_model_service(mock_request)
        
        # Verify it returns the mock service
        assert service == mock_model_service
        # Verify it accessed the container correctly
        mock_request.app.container.model_service.assert_called_once() 