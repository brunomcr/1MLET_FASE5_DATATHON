from pydantic import BaseModel, Field
from typing import List

from app.schemas.prediction import Prediction


class PredictionResponse(BaseModel):
    user_id: str = Field(..., description="User ID the predictions were generated for")
    predictions: List[Prediction] = Field(
        ..., 
        description="List of article predictions with their probabilities"
    )
