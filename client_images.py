import base64
import requests

# URL de la API (cambia por la de Render cuando la tengas desplegada)
API_URL = "http://127.0.0.1:8000/predict"

# Ruta de la imagen de prueba (sucia)
INPUT_IMAGE = "test_dirty.png"
OUTPUT_IMAGE = "resultado.png"

def image_to_base64(path):
    """Convierte una imagen en base64"""
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def base64_to_image(b64_string, path):
    """Convierte base64 a imagen y guarda en disco"""
    img_bytes = base64.b64decode(b64_string)
    with open(path, "wb") as f:
        f.write(img_bytes)

def main():
    # Cargar imagen de prueba
    img_b64 = image_to_base64(INPUT_IMAGE)

    # Crear payload
    payload = {"image_base64": img_b64}

    # Hacer POST a la API
    response = requests.post(API_URL, json=payload)

    if response.status_code == 200:
        result = response.json()
        print("✅ Respuesta recibida correctamente")

        # Guardar imagen resultante
        base64_to_image(result["cleaned_image_base64"], OUTPUT_IMAGE)
        print(f"🖼 Imagen procesada guardada en {OUTPUT_IMAGE}")
    else:
        print("❌ Error en la petición:", response.status_code, response.text)

if __name__ == "__main__":
    main()
