# utils/data_loader.py
import os
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from .constants import img_height, img_width, color_map

def rgb_to_class(mask):
    """Convierte la máscara de RGB a clases."""
    mask_class = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.int32)
    for class_id, color in color_map.items():
        mask_class[np.all(mask == color, axis=-1)] = class_id
    return mask_class

def load_data(img_dir, mask_dir):
    images = []
    masks = []
    
    for img_filename in os.listdir(img_dir):
        img_path = os.path.join(img_dir, img_filename)
        mask_path = os.path.join(mask_dir, img_filename)  # Suponemos que nombres coinciden

        # Cargar y redimensionar imagen
        img = load_img(img_path, target_size=(img_height, img_width))
        img = img_to_array(img) / 255.0  # Normalizar entre 0 y 1

        # Cargar y convertir máscara de RGB a índices de clase
        mask = load_img(mask_path, target_size=(img_height, img_width), color_mode="rgb")
        mask = img_to_array(mask).astype(np.int32)
        mask = rgb_to_class(mask)

        images.append(img)
        masks.append(mask)

    return np.array(images), np.array(masks)
w