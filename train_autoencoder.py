import os
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Concatenate
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from sklearn.model_selection import train_test_split

# === RUTAS ===
base_dir = r"C:\Users\nicol\Desktop\denoising-dirty-documents"
train_dir = os.path.join(base_dir, "train")
train_cleaned_dir = os.path.join(base_dir, "train_cleaned")
test_dir = os.path.join(base_dir, "test")

IMG_SIZE = (128, 128)

def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        path = os.path.join(folder, filename)
        if not os.path.isfile(path):
            continue
        img = load_img(path, target_size=IMG_SIZE, color_mode="grayscale")
        arr = img_to_array(img) / 255.0
        images.append(arr)
    return np.array(images, dtype=np.float32)

X_dirty = load_images_from_folder(train_dir)
X_clean = load_images_from_folder(train_cleaned_dir)

X_train_dirty, X_val_dirty, X_train_clean, X_val_clean = train_test_split(
    X_dirty, X_clean, test_size=0.2, random_state=42
)

# Modelo
input_img = Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 1))
x1 = Conv2D(32, (3,3), activation="relu", padding="same")(input_img)
x2 = MaxPooling2D((2,2), padding="same")(x1)
x2 = Conv2D(64, (3,3), activation="relu", padding="same")(x2)
x3 = MaxPooling2D((2,2), padding="same")(x2)
x3 = Conv2D(128, (3,3), activation="relu", padding="same")(x3)

d3 = Conv2D(128, (3,3), activation="relu", padding="same")(x3)
d3 = UpSampling2D((2,2))(d3)
d3 = Concatenate()([d3, x2])
d2 = Conv2D(64, (3,3), activation="relu", padding="same")(d3)
d2 = UpSampling2D((2,2))(d2)
d2 = Concatenate()([d2, x1])
d1 = Conv2D(32, (3,3), activation="relu", padding="same")(d2)
output_img = Conv2D(1, (3,3), activation="sigmoid", padding="same")(d1)

autoencoder = Model(input_img, output_img)
autoencoder.compile(optimizer="adam", loss="binary_crossentropy")

EPOCHS = 20
BATCH_SIZE = 16

history = autoencoder.fit(
    X_train_dirty, X_train_clean,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_data=(X_val_dirty, X_val_clean),
    verbose=1
)

os.makedirs("model", exist_ok=True)
out_path = os.path.join("model", "model.h5")
autoencoder.save(out_path)
print(f"Modelo guardado en {out_path}")
