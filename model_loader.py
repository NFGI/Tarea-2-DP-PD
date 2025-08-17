import os
import io
import base64
import numpy as np
from PIL import Image

from tensorflow.keras.models import load_model

MODEL_PATH = os.getenv("MODEL_PATH", os.path.join("model", "model.h5"))
_model = None

IMG_SIZE = (128, 128)  # alto, ancho (como entrenaste)

def load_model_file():
    global _model
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Modelo no encontrado en {MODEL_PATH}")
        _model = load_model(MODEL_PATH)
    return _model

def preprocess_image_b64(image_b64: str) -> np.ndarray:
    img_bytes = base64.b64decode(image_b64)
    img = Image.open(io.BytesIO(img_bytes)).convert("L")  # grayscale
    img = img.resize(IMG_SIZE)
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = np.expand_dims(arr, axis=(0, -1))  # (1, H, W, 1)
    return arr

def postprocess_to_b64(pred: np.ndarray) -> str:
    # pred esperado: (H, W, 1) con valores [0,1]
    pred_img = (pred * 255).clip(0, 255).astype(np.uint8).reshape(IMG_SIZE)
    pil_img = Image.fromarray(pred_img, mode="L")
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")

def predict_image_b64(image_b64: str) -> str:
    mdl = load_model_file()
    arr = preprocess_image_b64(image_b64)
    denoised = mdl.predict(arr, verbose=0)[0]  # (H, W, 1)
    return postprocess_to_b64(denoised)
