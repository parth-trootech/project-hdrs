from datetime import datetime
from typing import Optional

from pydantic import BaseModel


# Pydantic model for User Registration (Signup)
class UserCreate(BaseModel):
    user_email: str
    user_password: str

    class Config:
        from_attributes = True


# Pydantic model for User Login
class UserLogin(BaseModel):
    user_email: str
    user_password: str

    class Config:
        from_attributes = True


# Pydantic model for Image Upload
class ImageUploadCreate(BaseModel):
    image_path: str

    class Config:
        from_attributes = True


# Pydantic model for Prediction Result (to store predicted digit and confidence score)
class PredictionResultCreate(BaseModel):
    image_id: int
    predicted_digit: int
    confidence_score: Optional[float] = None  # Make confidence score optional

    class Config:
        from_attributes = True


# Response Model for User
class UserResponse(BaseModel):
    user_id: int
    user_email: str

    class Config:
        from_attributes = True


# Response Model for Image Upload (with prediction result)
class ImageUploadResponse(BaseModel):
    image_id: int
    user_id: int
    image_path: str
    upload_time: datetime
    predictions: Optional[list[PredictionResultCreate]]  # List of predictions for this image

    class Config:
        from_attributes = True


# Response Model for Prediction Result
class PredictionResultResponse(BaseModel):
    prediction_id: int
    image_id: int
    predicted_digit: int
    confidence_score: Optional[float]
    prediction_time: datetime

    class Config:
        from_attributes = True


# Added PredictionRequest model here

class PredictionRequest(BaseModel):
    image_id: int
