import os
import io
import base64
import binascii
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

# Ruta del modelo (por defecto: model/model.h5)
MODEL_PATH = os.getenv("MODEL_PATH", os.path.join("model", "model.h5"))

# Tamaño esperado por el autoencoder (alto, ancho)
IMG_SIZE = (128, 128)

_model = None  # caché en memoria del modelo

def load_model_file():
    """
    Carga el modelo Keras desde MODEL_PATH con validaciones claras.
    Devuelve el modelo cacheado en llamadas posteriores.
    """
    global _model
    if _model is not None:
        return _model

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Modelo no encontrado en {MODEL_PATH}")
    size = os.path.getsize(MODEL_PATH)
    if size == 0:
        raise FileNotFoundError(f"El archivo del modelo está vacío: {MODEL_PATH}")

    try:
        _model = load_model(MODEL_PATH)
    except Exception as e:
        # Mensaje explícito para ayudar a diagnosticar incompatibilidades de versión/formato
        raise RuntimeError(f"No se pudo cargar el modelo Keras desde {MODEL_PATH}: {e}")
    return _model

def _decode_image_b64(image_b64: str) -> bytes:
    """
    Decodifica base64 con validación estricta y devuelve bytes.
    Lanza ValueError si el base64 no es válido.
    """
    try:
        return base64.b64decode(image_b64, validate=True)
    except (binascii.Error, ValueError):
        raise ValueError("El campo image_b64 no contiene base64 válido.")

def preprocess_image_b64(image_b64: str) -> np.ndarray:
    """
    Convierte base64 -> PIL -> array float32 normalizado y con shape (1, H, W, C).
    Fuerza escala de grises (1 canal) para el autoencoder definido.
    """
    img_bytes = _decode_image_b64(image_b64)

    try:
        img = Image.open(io.BytesIO(img_bytes))
    except Exception:
        raise ValueError("No se pudo abrir la imagen decodificada. Verifica el formato (PNG/JPG).")

    # Convertimos a escala de grises (L) para ajustarnos al modelo (1 canal)
    img = img.convert("L")
    img = img.resize(IMG_SIZE)
    arr = np.array(img, dtype=np.float32) / 255.0  # [0,1]
    arr = np.expand_dims(arr, axis=(0, -1))  # (1, H, W, 1)
    return arr

def postprocess_to_b64(pred: np.ndarray) -> str:
    """
    Toma una predicción (H, W, 1) en [0,1], la convierte a PNG en memoria y retorna base64.
    """
    if pred.ndim == 3 and pred.shape[-1] == 1:
        pred = pred[..., 0]
    if pred.max() <= 1.0:
        pred_img = (pred * 255.0).clip(0, 255).astype(np.uint8)
    else:
        # Por si el modelo devuelve en [0,255]
        pred_img = pred.clip(0, 255).astype(np.uint8)

    pil_img = Image.fromarray(pred_img, mode="L")
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")

def predict_image_b64(image_b64: str) -> str:
    """
    Pipeline de inferencia: decodifica → preprocesa → predice → reencapsula en base64.
    """
    mdl = load_model_file()
    arr = preprocess_image_b64(image_b64)
    try:
        denoised = mdl.predict(arr, verbose=0)[0]  # (H, W, 1)
    except Exception as e:
        raise RuntimeError(f"Error al realizar la predicción: {e}")
    return postprocess_to_b64(denoised)
