from pydantic import BaseModel, Field

class PredictRequest(BaseModel):
    image_b64: str = Field(..., description="Imagen en base64 (PNG/JPG)")

class PredictResponse(BaseModel):
    cleaned_image_b64: str = Field(..., description="Imagen procesada (base64 PNG)")