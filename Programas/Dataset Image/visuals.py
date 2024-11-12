# visuals.py
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.models import load_model
from utils.constants import img_height, img_width, color_map
from utils.data_loader import load_data

# Función para mapear la máscara de clases a colores
def class_to_rgb(mask):
    """Convierte una máscara de clases a una máscara RGB usando el mapa de colores."""
    rgb_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    for class_index, color in enumerate(color_map):
        rgb_mask[mask == class_index] = color
    return rgb_mask

# Definir rutas de las imágenes y máscaras de prueba
img_dir_test = 'D:\\FloodNet-Supervised_v1.0\\test\\test-org-img'
mask_dir_test = 'D:\\ColorMasks-FloodNetv1.0\\ColorMasks-TestSet'

# Cargar datos de prueba
x_test, y_test = load_data(img_dir_test, mask_dir_test)

# Cargar el modelo entrenado
model = load_model('unet_floodnet_model.h5')
print("Modelo cargado exitosamente desde 'unet_floodnet_model.h5'")

# Hacer predicciones sobre el conjunto de prueba
predictions = model.predict(x_test)
predictions = np.argmax(predictions, axis=-1)  # Convertir predicciones a clases

# Visualizar y calcular precisión para algunas imágenes de prueba
for i in range(5):  # Visualizar las primeras 5 imágenes
    # Imagen original
    original_img = x_test[i]

    # Máscara real
    true_mask = y_test[i]

    # Máscara predicha
    pred_mask = predictions[i]

    # Convertir máscaras a RGB para visualización
    true_mask_rgb = class_to_rgb(true_mask)
    pred_mask_rgb = class_to_rgb(pred_mask)

    # Calcular precisión
    correct_pixels = np.sum(true_mask == pred_mask)
    total_pixels = true_mask.size
    accuracy = (correct_pixels / total_pixels) * 100

    # Mostrar imágenes en una ventana
    plt.figure(figsize=(15, 5))

    # Imagen original
    plt.subplot(1, 3, 1)
    plt.imshow(original_img)
    plt.title("Imagen Original")

    # Máscara real
    plt.subplot(1, 3, 2)
    plt.imshow(true_mask_rgb)
    plt.title("Máscara Real")

    # Máscara predicha
    plt.subplot(1, 3, 3)
    plt.imshow(pred_mask_rgb)
    plt.title(f"Predicción (Precisión: {accuracy:.2f}%)")

    # Mostrar la ventana con las tres imágenes
    plt.show()
