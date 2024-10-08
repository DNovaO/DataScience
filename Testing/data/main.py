# Ciencia de Datos
# Modelo entrenamiento y testing
# Regresión Lineal

import process as proc

# Ruta al archivo CSV
# file_path = r'C:\Users\bryan\Documents\ITQ\Semestre 8\Ciencia de Datos\DS env\Testing\data\diabetesnorm.csv'
file_path = r'C:\Users\Bry\Documents\ITQ\Semestre 8\Ciencia de Datos\Data-Science\Testing\data\diabetesnorm.csv'
# Cargar el dataset
dataset = proc.load_diabetes_data(file_path)

# Normalizar los datos
norm_dataset = proc.normalize_diabetes_data(dataset)

training_input, training_output, test_input, test_output, selected_columns = proc.split_data(norm_dataset, 0.3)

model = proc.simple_linear_regression(training_input, training_output)

# Realizar predicciones
test_predictions = model.predict(test_input)

# Obtener coeficientes
coefficients = proc.get_coefficients(model)
print("Coeficiente: ", coefficients)
for col, coef in zip(training_input.columns, coefficients):
    print(f"{col}: {coef}")

# Calcular y mostrar el MSE
MSE = proc.get_mean_squared_error(test_output, test_predictions)
print("Error cuadrático medio: ", MSE)

# Calcular y mostrar el R2 Score
R2 = proc.get_coefficient_of_determination(model, test_input, test_output)
print("R2 Score: ", R2)

# Guardar el modelo
output_dir = proc.check_output_dir()
proc.save_model(model, output_dir)

# Generar y guardar el gráfico de regresión
proc.plot_regression(test_input, test_output, test_predictions, model, output_dir)