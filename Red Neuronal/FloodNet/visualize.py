import matplotlib.pyplot as plt
import numpy as np


color_map = {
    0: [0, 0, 0],      
    1: [255, 0, 0],    
    2: [0, 255, 0],    
    3: [0, 0, 255],    
  
}

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
        
        if len(predictions[i].shape) == 3:  
            predictions[i] = np.argmax(predictions[i], axis=-1)

        # Verificar dimensiones y valores únicos
        print(f"Imagen {i+1}:")
        print(f"Dimensiones - Imagen: {images[i].shape}, Máscara: {masks[i].shape}, Predicción: {predictions[i].shape}")
        print(f"Valores únicos en Máscara: {np.unique(masks[i])}")
        print(f"Valores únicos en Predicción: {np.unique(predictions[i])}")

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
