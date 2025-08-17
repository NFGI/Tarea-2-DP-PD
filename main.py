from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from schemas import PredictRequest, PredictResponse
from model_loader import load_model_file, predict_image_b64

app = FastAPI(title="Image Denoising API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    try:
        load_model_file()
    except FileNotFoundError as e:
        # La API levanta igual, /predict dará 503 si no hay modelo
        print(f"[WARN] {e}")

@app.get("/")
def root():
    return {"message": "OK — visita /docs para probar la API"}

@app.get("/health")
def health():
    try:
        load_model_file()
        return {"status": "healthy"}
    except FileNotFoundError:
        return {"status": "degraded", "detail": "Modelo no encontrado"}

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    try:
        out_b64 = predict_image_b64(req.image_b64)
        return PredictResponse(cleaned_image_b64=out_b64)
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en predicción: {e}")
