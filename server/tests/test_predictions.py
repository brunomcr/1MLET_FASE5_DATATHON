from unittest.mock import Mock
from dependency_injector import containers, providers
from fastapi.testclient import TestClient
from app.main import app

def test_predict_single():
    # Create mock model service
    mock_model_service = Mock()
    mock_model_service.predict.return_value = (1.0, 0.95)
    
    # Override container
    app.container.model_service.override(providers.Object(mock_model_service))
    
    client = TestClient(app)
    response = client.post("/api/v1/predict", json={"features": [1.0, 2.0, 3.0]})
    
    assert response.status_code == 200
    assert response.json() == {"prediction": 1.0, "probability": 0.95}

def test_predict_batch():
    # Create mock model service
    mock_model_service = Mock()
    mock_model_service.predict_batch.return_value = [
        (1.0, 0.95),
        (0.0, 0.85)
    ]
    
    # Override container
    app.container.model_service.override(providers.Object(mock_model_service))
    
    client = TestClient(app)
    response = client.post(
        "/api/v1/predict/batch", 
        json={"features_batch": [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]}
    )
    
    assert response.status_code == 200
    assert response.json() == {"predictions": [[1.0, 0.95], [0.0, 0.85]]} 