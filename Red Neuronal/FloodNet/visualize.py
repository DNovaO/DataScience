# visualize.py
import matplotlib.pyplot as plt
import numpy as np
from constants import color_map

def decode_mask(mask):
    """
    Decodifica una máscara segmentada en colores para su visualización.
    """
    height, width = mask.shape
    decoded = np.zeros((height, width, 3), dtype=np.uint8)

    for class_id, color in color_map.items():
        decoded[mask == class_id] = color

    return decoded

def visualize_results(images, masks, predictions, num_samples=5):
    """
    Visualiza imágenes originales, máscaras reales y predicciones.
    """
    for i in range(num_samples):
        plt.figure(figsize=(15, 5))

        # Imagen original
        plt.subplot(1, 3, 1)
        plt.imshow(images[i])
        plt.title("Imagen Original")
        plt.axis("off")

        # Máscara real
        plt.subplot(1, 3, 2)
        decoded_mask = decode_mask(masks[i])
        plt.imshow(decoded_mask)
        plt.title("Máscara Real")
        plt.axis("off")

        # Predicción
        plt.subplot(1, 3, 3)
        decoded_prediction = decode_mask(predictions[i])
        plt.imshow(decoded_prediction)
        plt.title("Predicción")
        plt.axis("off")

        plt.show()