import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Cargar el archivo CSV
file_path = r'C:\Users\bryan\Documents\ITQ\Semestre 8\Ciencia de Datos\DS env\Programas\data\datos_inegi.csv'
data = pd.read_csv(file_path)

# Mostrar las primeras filas del DataFrame para verificar las columnas
print(data.head())
print(data.columns)

# Seleccionar los datos de un estado en específico
estado = 1
data_estado = data[data['mun_regis'] == estado]

# Verificar si el filtrado devolvió datos
print(f"Número de filas para el estado {estado}: {len(data_estado)}")

# Seleccionar las columnas de interés
variables_interes = ['edad_madn', 'edad_padn']

# Asegurarnos de que las columnas existan en el DataFrame
for col in variables_interes:
    if col not in data_estado.columns:
        print(f"Advertencia: La columna '{col}' no se encontró en los datos")
        variables_interes.remove(col)

data_estado = data_estado[variables_interes]
data_estado.dropna(inplace=True)  # Eliminamos las filas con valores nulos

# Verificar si hay datos después de eliminar valores nulos
print(f"Número de filas después de eliminar valores nulos: {len(data_estado)}")

# Asegurarnos de que 'edad_madn' y 'edad_padn' existen antes de crear el heatmap
if 'edad_madn' in data_estado.columns and 'edad_padn' in data_estado.columns:
    # Generar un histograma 2D entre 'edad_madn' y 'edad_padn'
    z, x_edges, y_edges = np.histogram2d(data_estado['edad_madn'], data_estado['edad_padn'], bins=[50, 50])

    # Configurar el tamaño del gráfico
    plt.figure(figsize=(12, 10))

    # Generar el heatmap
    sns.heatmap(z.T, cmap='YlOrRd', cbar_kws={'label': 'Frecuencia'})
    
    # Configurar los ejes
    plt.xlabel('Edad Materna')
    plt.ylabel('Edad Paterna')
    plt.title(f'Heatmap de Edad Materna vs Edad Paterna en el estado {estado}')

    # Ajustar las etiquetas de los ejes
    plt.xticks(np.linspace(0, 50, 6), np.linspace(x_edges.min(), x_edges.max(), 6).astype(int))
    plt.yticks(np.linspace(0, 50, 6), np.linspace(y_edges.min(), y_edges.max(), 6).astype(int))

    # Mostrar el gráfico
    plt.tight_layout()
    plt.show()
else:
    print("No se pudo generar el heatmap porque faltan columnas necesarias.")