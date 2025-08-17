import os
import sys
import time
import base64
import requests
from typing import List

# === Config ===
API_BASE = "https://tarea-2-dp-pd.onrender.com"
PREDICT_URL = f"{API_BASE.rstrip('/')}/predict"

# Rutas por defecto (las que me diste)
DEFAULT_INPUTS = [
    r"C:\Users\nicol\Desktop\denoising-dirty-documents\test\145.png",
    r"C:\Users\nicol\Desktop\denoising-dirty-documents\test\52.png",
    r"C:\Users\nicol\Desktop\denoising-dirty-documents\test\76.png",
]

def image_to_b64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def b64_to_file(b64_str: str, out_path: str) -> None:
    data = base64.b64decode(b64_str)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "wb") as f:
        f.write(data)

def pick_inputs_from_args() -> List[str]:
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    return args if args else DEFAULT_INPUTS

def main():
    inputs = pick_inputs_from_args()
    print(f"→ Endpoint: {PREDICT_URL}")
    print(f"→ Casos: {inputs}")

    ok, fail = 0, 0
    for i, in_path in enumerate(inputs, 1):
        if not os.path.isfile(in_path):
            print(f"[{i}] ⚠️ No existe: {in_path}")
            fail += 1
            continue

        try:
            img_b64 = image_to_b64(in_path)
            payload = {"image_b64": img_b64}

            t0 = time.time()
            r = requests.post(PREDICT_URL, json=payload, timeout=120)
            dt = time.time() - t0

            print(f"[{i}] {os.path.basename(in_path)} → {r.status_code} ({dt:0.2f}s)")
            if not r.ok:
                print("    Respuesta:", r.text[:300], "...")
                fail += 1
                continue

            data = r.json()
            out_path = os.path.join("results", f"cleaned_{i}.png")
            b64_to_file(data["cleaned_image_b64"], out_path)
            print(f"    ✅ Guardado: {out_path}")
            ok += 1

        except requests.exceptions.RequestException as e:
            print(f"    ❌ Error de red/timeout: {e}")
            fail += 1
        except KeyError:
            print(f"    ❌ La respuesta no trae 'cleaned_image_b64'. Respuesta: {r.text[:300]} ...")
            fail += 1
        except Exception as e:
            print(f"    ❌ Error inesperado: {e}")
            fail += 1

    print("\n=== Resumen ===")
    print(f"OK:   {ok}")
    print(f"Fail: {fail}")
    print("Resultados en: results/")

if __name__ == "__main__":
    main()