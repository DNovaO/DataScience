import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Cargar los datos del archivo CSV
read_inegi_dataset = 'data/datos_inegi.csv'  # Cambia esto por la ruta a tu archivo
data = pd.read_csv(read_inegi_dataset)

# Mostrar las columas del archivo
print(data.head())

# Elegimos un estado de interés
estado = '04'  # Cambia esto por el estado o región de interés
data_estado = data[data['MUN_REGIS'] == estado]

# Seleccionar tres variables de interés
variables_interes = ['EDAD_MADR', 'ESCOL_MAD', 'HIJOS_VIVO']  # Cambia esto por las variables de interés
data_estado = data_estado[variables_interes]

# Eliminar filas con datos faltantes en las variables de interés
data_estado.dropna(inplace=True)

# Generar un heatmap de correlación entre las variables
correlation_matrix = data_estado.corr()

# Configurar el tamaño del gráfico
plt.figure(figsize=(10, 8))

# Generar el heatmap
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)

# Agregar título
plt.title(f'Heatmap de Correlación de Variables en {estado}')

# Mostrar el gráfico
plt.show()