import sys
import os
import importlib

# Agregar la carpeta raíz del proyecto al sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Importar dinamicamente el modulo
imports = importlib.import_module('utils.imports')
read_diabetes = getattr(imports, 'read_diabetes')

# Ejecutar la funcion
dataset = read_diabetes("data/diabetes.tab.txt")
print(dataset)
print("ok")

# Ejecutar la funcion
visuals = importlib.import_module('utils.visuals')

# Generar y guardar las visualizaciones
visuals.save_histogram(dataset, "AGE")
visuals.save_correlation(dataset, "BMI", "S6")
visuals.save_all_correlations(dataset, dataset.corr())
visuals.save_all_correlations_one_image(dataset)