from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    user_id: str = Field(..., description="User ID to generate predictions for")
    num_recommendations: int = Field(
        ..., 
        description="Number of recommendations to generate",
        gt=0
    )
