# Regresion Lineal
# 1) Hacer un split data con los datos normalizados de diabetesnorm.csv
# - El 70% de columnas (6) tienen que ser aleatorias asi como el otro 30% (5 Columnas)
# - Dataframe llamado test y train, dividir entre 70 y 30, el 30 es con la ultima columna.
# 2) Generar el modelo de regresion lineal: linear_regression, con train_data y train_labels
# - Columnas: 1-10 con train_data, 11 para train_labels
# 3) Hacer la evaluacion como: y_pred = model.predict(test)
# - Calcular el RMSE y MSE: MSE(y, y_pred), RMSE(y, y_pred)
# 4) Hacer una funcion para cada uno de los pasos.
# 5) Hacer linear_regression con train_data (1-10 columnas), train_labels (Columna 11)

import os
import joblib
import random
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Funcion para crear el directorio de salida si no existe
def check_output_dir():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    output_dir = os.path.join(base_dir, 'output')
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    return output_dir

# Funcion para cargar los datos de diabetes
def load_diabetes_data(file_path):
    return pd.read_csv(file_path)

# Funcion para normalizar los datos
def normalize_diabetes_data(data):
    scaler = StandardScaler()
    # Asumimos que todas las columnas menos la última necesitan normalización
    data_scaled = scaler.fit_transform(data.iloc[:, :-1])
    norm_data = pd.DataFrame(data_scaled, columns=data.columns[:-1])
    norm_data['Y'] = data['Y']  # Añadir la columna de salida de nuevo
    return norm_data

# Funcion para dividir los datos en entrenamiento y prueba
def split_data(data, test_size):
    # Seleccionar 6 columnas aleatorias para entrenamiento y prueba
    feature_columns = list(data.columns[:-1])  # Todas las columnas excepto 'Y'
    selected_columns = random.sample(feature_columns, 6)
    
    train_data, test_data = train_test_split(data, test_size=test_size)
    
    train_input = train_data[selected_columns]
    train_output = train_data['Y']

    test_input = test_data[selected_columns]
    test_output = test_data['Y']

    return train_input, train_output, test_input, test_output, selected_columns
# Funcion para entrenar un modelo de regresión lineal
def simple_linear_regression(train_input, train_output):
    model = LinearRegression()
    model.fit(train_input, train_output)
    return model

# Funcion para obtener los coeficientes del modelo
def get_coefficients(model):
    return model.coef_

# Funcion para obtener el error cuadrático medio
def get_mean_squared_error(test_output, test_predictions):
    return mean_squared_error(test_output, test_predictions)

# Funcion para obtener el coeficiente de determinación (R2)
def get_coefficient_of_determination(model, test_input, test_output):
    return model.score(test_input, test_output)

# Funcion para guardar el modelo
def save_model(model, output_dir):
    model_path = os.path.join(output_dir, 'linear_regression_model.pkl')
    joblib.dump(model, model_path)
    return model_path

# Funcion para cargar el modelo
def load_model(model_path):
    model = joblib.load(model_path)
    return model

# Funcion para hacer predicciones
def test_predictions(model, test_input):
    return model.predict(test_input)

# Funcion para graficar la regresión
def plot_regression(test_input, test_output, test_predictions, model, output_dir):
    plt.figure(figsize=(10, 6))
    plt.scatter(test_output, test_predictions, color='blue', alpha=0.5)
    plt.plot([test_output.min(), test_output.max()], [test_output.min(), test_output.max()], 'r--', lw=2)
    plt.xlabel('Valores reales')
    plt.ylabel('Predicciones')
    plt.title('Regresión Lineal: Valores reales vs Predicciones')
    
    # Añadir texto con el R2 score
    r2 = get_coefficient_of_determination(model, test_input, test_output)
    plt.text(0.05, 0.95, f'R2 Score: {r2:.2f}', transform=plt.gca().transAxes, verticalalignment='top')    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'regression_plot.png'))
    plt.close()
    print("Gráfico de regresión guardado como 'regression_plot.png'")