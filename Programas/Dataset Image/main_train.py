# main_train.py
import os
from .models.unet import unet_model
from .utils.data_loader import load_data
from .utils.constants import img_height, img_width, num_classes

# Definir rutas de las imágenes y máscaras de entrenamiento
img_dir_train = 'D:\\FloodNet-Supervised_v1.0\\train\\train-org-img'
mask_dir_train = 'D:\\ColorMasks-FloodNetv1.0\\ColorMasks-TrainSet'

# Cargar datos de entrenamiento
x_train, y_train = load_data(img_dir_train, mask_dir_train)

# Crear y compilar el modelo
model = unet_model((img_height, img_width, 3))
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Entrenar el modelo
history = model.fit(x_train, y_train, batch_size=16, epochs=50, validation_split=0.2, verbose=1)

# Guardar el modelo entrenado
model.save('unet_floodnet_model.h5')
print("Modelo entrenado y guardado en 'unet_floodnet_model.h5'")
