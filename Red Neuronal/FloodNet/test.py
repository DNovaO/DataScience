import matplotlib.pyplot as plt
import numpy as np

from tensorflow.keras.models import load_model
from data_loader import load_data
from constants import img_height, img_width, color_map

# Cargar datos de prueba
x_test, y_test, num_classes_test = load_data('D:/FloodNet-Supervised_v1.0/test/test-org-img', 'D:/FloodNet-Supervised_v1.0/test/test-label-img', img_height, img_width)

print(f"Número de clases en el conjunto de prueba: {num_classes_test}")

# Cargar modelo entrenado
model = load_model('floodnet_model_best.keras')

# Generar predicciones
predictions = model.predict(x_test)
predictions_classes = np.argmax(predictions, axis=-1)

# Visualizar resultados
for i in range(5):  # Mostrar 5 imágenes
    plt.figure(figsize=(15, 5))
    
    # Imagen original
    plt.subplot(1, 3, 1)
    plt.imshow(x_test[i])
    plt.title('Imagen Original')
    plt.axis('off')
    
    # Máscara verdadera
    plt.subplot(1, 3, 2)
    true_mask = y_test[i].argmax(axis=-1)
    plt.imshow(true_mask, cmap='jet', vmin=0, vmax=num_classes_test-1)
    plt.title('Máscara Verdadera')
    plt.axis('off')

    # Predicción
    plt.subplot(1, 3, 3)
    plt.imshow(predictions_classes[i], cmap='jet', vmin=0, vmax=num_classes_test-1)
    plt.title('Predicción')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
