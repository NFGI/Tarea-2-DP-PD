from pydantic import BaseModel, Field, field_validator
import base64
import binascii

class PredictRequest(BaseModel):
    image_b64: str = Field(
        ...,
        description="Imagen en base64 (PNG/JPG). Debe representar una imagen en escala de grises o RGB."
    )

    @field_validator("image_b64")
    @classmethod
    def validate_b64(cls, v: str) -> str:
        # Validación básica: no vacío y longitud mínima razonable
        if not isinstance(v, str) or len(v.strip()) < 50:
            raise ValueError("image_b64 debe ser una cadena base64 no vacía y de longitud razonable.")
        # Validación estricta de base64 (sin decodificar imágenes grandes)
        try:
            # validate=True hace que falle si hay caracteres no válidos o padding incorrecto
            base64.b64decode(v, validate=True)
        except (binascii.Error, ValueError):
            raise ValueError("image_b64 no es una cadena base64 válida.")
        return v

class PredictResponse(BaseModel):
    cleaned_image_b64: str = Field(
        ...,
        description="Imagen procesada (PNG) codificada en base64."
    )
