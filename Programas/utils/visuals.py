# 1. Funcion para genererar y guardar las visualizaciones (Histogramas(dataframe,"column"))
# 2. Crear una funcion, la cual genere un folder para guardar las imcolumnnes generadas en png

import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import numpy as np
import pandas as pd
import os

def check_output_dir():
    # Obtener la ruta absoluta del directorio donde se encuentra el script actual (main.py)
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    output_dir = os.path.join(base_dir, 'output')
    
    # Crear el directorio si no existe
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    return output_dir

"""
# Crear la funcion con seaborn para generar un histograma por pares de columna
def save_histogram(dataframe, column):
    output_dir = check_output_dir()
    sns.set(style="whitegrid")
    sns.histplot(data=dataframe, x=column, kde=True)
    plt.title(f"Histograma de {column}")
    plt.xlabel(column)
    plt.ylabel("Frecuencia")
    plt.savefig(f"{output_dir}/histogram_{column}.png")
    plt.close()

# Funcion para generar un histograma por columnas
def save_histograms(dataframe):
    output_dir = check_output_dir()
    for column in dataframe.columns:
        save_histogram(dataframe, column)
    
def save_correlation(dataframe, column1, column2):
    output_dir = check_output_dir()
    sns.set(style="whitegrid")
    sns.scatterplot(data=dataframe, x=column1, y=column2)
    
    plt.title(f"Correlación entre {column1} y {column2} : {dataframe[column1].corr(dataframe[column2])}")
    plt.xlabel(column1)
    plt.ylabel(column2)
    plt.savefig(f"{output_dir}/correlation_{column1}_{column2}.png")
    plt.clf()
    plt.close()

def save_all_correlations(dataframe, correlation):
    output_dir = check_output_dir()

    plt.savefig(f"{output_dir}/correlation_all.png")
    plt.clf()
    
    num_columns = correlation.shape[1]
    for i in range(num_columns):
        for j in range(i + 1, num_columns):
            save_correlation(dataframe, correlation.columns[i], correlation.columns[j])
    
    plt.close()

def save_all_correlations_one_image(dataframe):
    output_dir = check_output_dir()
    sns.pairplot(dataframe, hue = "SEX")
    plt.savefig(f"{output_dir}/ALL_histograms.png")
    plt.close()
    pass
"""
# SEGUNDA PARTE --> INEGI
# Crear la funcion con seaborn para generar un heatmap

def save_heatmap(dataframe, x_col, y_col, z_col):
    output_dir = check_output_dir()

    # Pivotar el DataFrame para obtener una matriz de valores
    pivot_table = dataframe.pivot_table(index=y_col, columns=x_col, values=z_col, aggfunc='mean')

    # Crear el heatmap
    plt.figure(figsize=(10, 8))
    sns.set(style="whitegrid")
    
    # Generamos el heatmap
    sns.heatmap(pivot_table, annot=True, fmt=".1f", cmap="coolwarm", linewidths=0.5)

    # Título y etiquetas
    plt.title(f"Heatmap de {z_col} con {x_col} y {y_col}")
    plt.xlabel(x_col)
    plt.ylabel(y_col)

    # Guardar la imagen
    plt.savefig(f"{output_dir}/heatmap_{x_col}_{y_col}_{z_col}.png")
    plt.close()