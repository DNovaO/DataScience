from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical

import numpy as np
modelo = load_model('best_model_87.keras')
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Preprocesar los datos
x_test = x_test.astype('float32') / 255.0  # Normalizar los valores de los píxeles
y_test_categorical = to_categorical(y_test, 10)  # One-hot encoding de las etiquetas

predicciones = modelo.predict(x_test[:10])

# Mapeo de las clases CIFAR-10
clases = ['avión', 'automóvil', 'pájaro', 'gato', 'ciervo', 'perro', 'rana', 'caballo', 'barco', 'camión']

# Variables para contar las correctas e incorrectas
correctas = 0
incorrectas = 0

# Comparar predicciones con las etiquetas reales
for i in range(10):
    clase_predicha = np.argmax(predicciones[i])  # Clase predicha
    clase_real = y_test[i][0]  # Clase real
    
    # Verificar si la predicción es correcta
    if clase_predicha == clase_real:
        correctas += 1
    else:
        incorrectas += 1
    
    # Mostrar el resultado para cada imagen
    print(f'Imagen {i+1}:')
    print(f'    Clase predicha: {clases[clase_predicha]}')
    print(f'    Clase real: {clases[clase_real]}')
    print(f'    {"Correcto" if clase_predicha == clase_real else "Incorrecto"}\n')

# Calcular porcentajes
porcentaje_correctas = (correctas / 10) * 100
porcentaje_incorrectas = (incorrectas / 10) * 100

# Mostrar los resultados finales
print(f'Porcentaje de predicciones correctas: {porcentaje_correctas}%')
print(f'Porcentaje de predicciones incorrectas: {porcentaje_incorrectas}%')