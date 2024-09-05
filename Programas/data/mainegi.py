import sys
import os
import importlib

# Agregar la carpeta raíz del proyecto al sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Buscar el arhivo de INEGI
imports = importlib.import_module('utils.imports')
read_inegi = getattr(imports, 'read_inegi')

# Ejecutar la funcion
dataset = read_inegi("data/conjunto_de_datos_natalidad_2022.csv") # Catalogos de datos
print("ok")

# Ejecutar la funcion
visuals = importlib.import_module('utils.visuals')

# Generar y guardar las visualizaciones
visuals.save_heatmap(dataset, "edad_madn", "edad_padn", "dia_nac")