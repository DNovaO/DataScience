# 1. Funcion para genererar y guardar las visualizaciones (Histogramas(dataframe,"column"))
# 2. Crear una funcion, la cual genere un folder para guardar las imcolumnnes generadas en png

import seaborn as sns
#import utils.processing as proc
import matplotlib.pyplot as plt
#import plotly.graph_objects as go
#import plotly.express as px
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