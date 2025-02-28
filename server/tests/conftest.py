import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock

from app.main import app
from app.services.model_service import ModelService
from app.routers.v1.predictions import get_model_service

@pytest.fixture
def mock_model_service():
    return Mock(spec=ModelService)

@pytest.fixture
def test_client(mock_model_service):
    """Create a test client with mocked dependencies"""
    # Store original dependencies
    original_dependencies = app.dependency_overrides.copy()
    
    # Override the model service dependency
    app.dependency_overrides[get_model_service] = lambda: mock_model_service
    
    # Create test client
    client = TestClient(app)
    
    yield client
    
    # Restore original dependencies
    app.dependency_overrides = original_dependencies

@pytest.fixture
def sample_predictions():
    """Provide sample prediction data for tests"""
    from app.schemas.prediction import Prediction
    return [
        Prediction(article_id="article1", score=0.9),
        Prediction(article_id="article2", score=0.8),
        Prediction(article_id="article3", score=0.7),
    ]