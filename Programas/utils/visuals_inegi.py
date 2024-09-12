import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os

# 1. Leer datos del csv inegi 2022
# 2. Construir las primeras 10 columnas normalizadas
# 3. Construir un heatmap usando 3 variables del csv

def check_output_dir():
    # Obtener la ruta absoluta del directorio donde se encuentra el script actual (main.py)
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    output_dir = os.path.join(base_dir, 'output')
    
    # Crear el directorio si no existe
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    return output_dir