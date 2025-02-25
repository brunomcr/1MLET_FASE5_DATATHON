from typing import List
from dependency_injector.wiring import inject, Provide
from fastapi import APIRouter, HTTPException, Depends

from app.containers import Container
from app.schemas.prediction.requests.prediction_request import PredictionRequest
from app.schemas.prediction.responses.prediction_response import PredictionResponse
from app.services.model_service import ModelService

router = APIRouter(prefix="/v1/predict", tags=["Prediction, Recommendation System, Machine Learning"])

@router.post("/", response_model=PredictionResponse)
@inject
async def predict(
    request: PredictionRequest,
    model_service: ModelService = Depends(Provide[Container.model_service])
) -> PredictionResponse:
    """
    Get article recommendations for a single user
    """
    try:
        predictions = model_service.predict(
            user_id=request.user_id,
            num_recommendations=request.num_recommendations
        )
        return PredictionResponse(
            user_id=request.user_id,
            predictions=predictions
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/batch", response_model=List[PredictionResponse])
@inject
async def predict_batch(
    requests: List[PredictionRequest],
    model_service: ModelService = Depends(Provide[Container.model_service])
) -> List[PredictionResponse]:
    """
    Get article recommendations for multiple users
    """
    try:
        responses = []
        for request in requests:
            predictions = model_service.predict(
                user_id=request.user_id,
                num_recommendations=request.num_recommendations
            )
            responses.append(
                PredictionResponse(
                    user_id=request.user_id,
                    predictions=predictions
                )
            )
        return responses
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) 