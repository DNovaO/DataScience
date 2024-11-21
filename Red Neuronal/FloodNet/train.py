from unet_model import unet_model
from data_loader import load_data
from constants import img_height, img_width
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import matplotlib.pyplot as plt

# Directorios del dataset
img_dir_train = 'D:/FloodNet-Supervised_v1.0/train/train-org-img'
mask_dir_train = 'D:/ColorMasks-FloodNetv1.0/ColorMasks-TrainSet'
img_dir_val = 'D:/FloodNet-Supervised_v1.0/val/val-org-img'
mask_dir_val = 'D:/FloodNet-Supervised_v1.0/val/val-label-img'

# Cargar datos
x_train, y_train, num_classes_train = load_data(img_dir_train, mask_dir_train, img_height, img_width)
x_val, y_val, num_classes_val = load_data(img_dir_val, mask_dir_val, img_height, img_width)

# Verificar que el número de clases sea el mismo en ambos conjuntos
if num_classes_train != num_classes_val:
    raise ValueError("El número de clases en el conjunto de entrenamiento y validación no coincide.")

# Crear modelo
model = unet_model((img_height, img_width, 3), num_classes_train)
model.summary()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Callbacks
checkpoint = ModelCheckpoint('floodnet_model_best.keras', save_best_only=True, monitor='val_loss', mode='min')
early_stopping = EarlyStopping(patience=50, monitor='val_loss', mode='min', restore_best_weights=True)

# Entrenar
history = model.fit(
    x_train, y_train,
    validation_data=(x_val, y_val),
    epochs=100,
    batch_size=32,
    callbacks=[checkpoint, early_stopping]
)

# Guardar el modelo final
model.save('floodnet_model_final.keras')

# Visualizar el historial de entrenamiento
plt.figure(figsize=(12, 4))
plt.subplot(121)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.legend()

plt.subplot(122)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()

plt.tight_layout()
plt.show()