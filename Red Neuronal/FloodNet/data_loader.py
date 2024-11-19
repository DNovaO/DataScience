import os
import cv2
import numpy as np
from tensorflow.keras.utils import to_categorical

def load_data(img_dir, mask_dir, img_height, img_width, num_classes):
    images = []
    masks = []
    for img_file in os.listdir(img_dir):
        img_path = os.path.join(img_dir, img_file)
        mask_file = img_file.replace('.jpg', '_lab.png')
        mask_path = os.path.join(mask_dir, mask_file)

        if not os.path.exists(img_path) or not os.path.exists(mask_path):
            print(f"Advertencia: {img_path} o {mask_path} no existe.")
            continue

        img = cv2.imread(img_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if img is None or mask is None:
            print(f"Advertencia: No se pudo leer {img_path} o {mask_path}.")
            continue

        img = cv2.resize(img, (img_width, img_height))
        mask = cv2.resize(mask, (img_width, img_height), interpolation=cv2.INTER_NEAREST)

        images.append(img)
        masks.append(mask)

    images = np.array(images, dtype=np.float32) / 255.0
    masks = np.array(masks, dtype=np.int32)

    # Verificar valores únicos en las máscaras
    print("Valores únicos en las máscaras antes de clip:", np.unique(masks))
    
    # Limitar valores al rango permitido
    masks = np.clip(masks, 0, num_classes - 1)

    if num_classes > 1:
        masks = to_categorical(masks, num_classes=num_classes)

    return images, masks
