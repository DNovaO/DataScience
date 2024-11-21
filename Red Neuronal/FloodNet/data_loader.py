import os
import tensorflow as tf
import numpy as np
from tensorflow.keras.utils import to_categorical

def remap_mask_values(mask):
    # Create a mapping of unique values to 0-(num_classes-1)
    unique_values = np.unique(mask)
    value_map = {val: i for i, val in enumerate(unique_values)}
    
    # Remap the mask values
    for old_value, new_value in value_map.items():
        mask[mask == old_value] = new_value
    
    return mask

def preprocess_image(image, mask, target_size=(128, 128)):
    # Redimensionar imagen
    image = tf.image.resize(image, target_size, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    # Redimensionar máscara
    mask = tf.image.resize(mask, target_size, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    return image, mask

def load_data(img_dir, mask_dir, img_height, img_width):
    # Listas para almacenar imágenes y máscaras
    images = []
    masks = []

    # Obtener listas de archivos de imágenes y máscaras
    image_files = sorted([f for f in os.listdir(img_dir) if f.endswith('.jpg')])
    mask_files = sorted([f for f in os.listdir(mask_dir) if f.endswith('.png')])

    # Verificar que el número de imágenes y máscaras coincida
    if len(image_files) != len(mask_files):
        raise ValueError("El número de imágenes y máscaras no coincide.")

    for img_file, mask_file in zip(image_files, mask_files):
        # Cargar imagen
        img_path = os.path.join(img_dir, img_file)
        img = tf.io.read_file(img_path)
        img = tf.image.decode_image(img, channels=3)
        img = tf.image.convert_image_dtype(img, tf.float32)  # Normalizar a [0,1]

        # Cargar máscara
        mask_path = os.path.join(mask_dir, mask_file)
        mask = tf.io.read_file(mask_path)
        mask = tf.image.decode_image(mask, channels=1)
        mask = tf.image.convert_image_dtype(mask, tf.uint8)  # Asegurar que la máscara sea entera

        # Preprocesar imagen y máscara
        img, mask = preprocess_image(img, mask, target_size=(img_height, img_width))

        images.append(img)
        masks.append(mask)

    # Convertir listas a arrays de NumPy
    images = np.array(images, dtype=np.float32)
    masks = np.array(masks, dtype=np.uint8)

    # Remap mask values
    masks = np.array([remap_mask_values(mask) for mask in masks])

    # Verify unique values after remapping
    unique_values = np.unique(masks)
    print(f"Valores únicos en las máscaras después de remapear: {unique_values}")

    # Ensure num_classes matches the actual number of classes
    num_classes = len(unique_values)
    print(f"Número de clases ajustado a: {num_classes}")

    # Convert masks to one-hot encoding
    masks = to_categorical(masks, num_classes=num_classes)

    return images, masks, num_classes