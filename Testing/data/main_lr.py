# Ciencia de Datos
# Modelo entrenamiento y testing
# Regresión Logística

import process_lr as proc

# Ruta al archivo CSV
# file_path = r'C:\Users\bryan\Documents\ITQ\Semestre 8\Ciencia de Datos\DS env\Testing\data\diabetesnorm.csv'
file_path = r'C:\Users\Bry\Documents\ITQ\Semestre 8\Ciencia de Datos\Data-Science\Testing\data\diabetesnorm.csv'

dataset = proc.load_diabetes_data(file_path)
norm_dataset = proc.normalize_diabetes_data(dataset)

training_input, training_output, test_input, test_output, selected_columns = proc.split_data(norm_dataset, 0.3)

# Entrenar el modelo de regresión logística
model = proc.logistic_regression(training_input, training_output)

# Calcular el accuracy
accuracy = proc.get_accuracy(model, test_input, test_output)
print(f"Accuracy: {accuracy}")

# Obtener la matriz de confusión
conf_matrix = proc.get_confusion_matrix(model, test_input, test_output)
print(f"Matriz de confusión:\n{conf_matrix}")
print(f"Forma de los datos de entrenamiento: {training_input.shape}")
print(f"Distribución de la variable objetivo: {training_output.value_counts(normalize=True)}")

# Guardar el modelo
output_dir = proc.check_output_dir()
proc.save_model(model, output_dir)

# Generar la curva ROC
proc.plot_roc_curve(model, test_input, test_output, output_dir)