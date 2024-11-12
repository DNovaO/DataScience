# main_test.py
import os
import numpy as np
from tensorflow.keras.models import load_model
from utils.data_loader import load_data
from utils.constants import img_height, img_width

# Definir rutas de las imágenes y máscaras de prueba
img_dir_test = 'D:\\FloodNet-Supervised_v1.0\\test\\test-org-img'
mask_dir_test = 'D:\\ColorMasks-FloodNetv1.0\\ColorMasks-TestSet'

# Cargar datos de prueba
x_test, y_test = load_data(img_dir_test, mask_dir_test)

# Cargar el modelo entrenado
model = load_model('unet_floodnet_model.h5')
print("Modelo cargado exitosamente desde 'unet_floodnet_model.h5'")

# Evaluar el modelo en el conjunto de prueba
score = model.evaluate(x_test, y_test, verbose=1)
print("Test loss:", score[0])
print("Test accuracy:", score[1])

# Realizar predicciones para visualizar resultados (opcional)
predictions = model.predict(x_test)
np.save('predictions.npy', predictions)  # Guardar las predicciones para visualización posterior
print("Predicciones guardadas en 'predictions.npy'")
