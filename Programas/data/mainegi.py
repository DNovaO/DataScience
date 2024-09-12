import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Cargar los datos del archivo CSV
file_path = r'C:\Users\bryan\Documents\ITQ\Semestre 8\Ciencia de Datos\DS env\Programas\data\datos_inegi.csv'  # Ruta absoluta
data = pd.read_csv(file_path)

# Mostrar las primeras filas del dataframe para entender la estructura de los datos
print(data.head())

estado = '1'
data_estado = data[data['mun_regis'] == estado]

# Seleccionar las columnas a mostrar
variables_interes = ['edad_madn', 'ESCOL_MAD', 'SEXO']
data_estado = data_estado[variables_interes]

# Eliminamos los residuos
data_estado.dropna(inplace=True)

# Convertir las variables categóricas en variables dummy
data_estado = pd.get_dummies(data_estado, columns=['ESCOL_MAD', 'SEXO'], drop_first=True)

# Generar un heatmap de correlación entre las variables
correlation_matrix = data_estado.corr()

# Configurar el tamaño del gráfico
plt.figure(figsize=(10, 8))

# Generar el heatmap
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title(f'Heatmap de Correlación de Variables en {estado}')
plt.show()