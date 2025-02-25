from pydantic import BaseModel, Field


class Prediction(BaseModel):
    article_id: str = Field(..., description="Predicted article ID")
    score: float = Field(
        ..., 
        description="Prediction probability/score",
        ge=0.0,
        le=1.0
    )
