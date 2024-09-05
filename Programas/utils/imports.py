import pandas as pd # Se encarga de leer datos desde txt
import seaborn as sns # Se encarga de generar visualizaciones

pd.set_option('display.max_rows', None) # Mostrar todas las filas
pd.set_option('display.max_columns', None) # Mostrar todas las columnas
# Para volver al set default, solo quitamos el valor None, o bien, borrar las lineas anteriores.
"""
def read_diabetes(path):
    # Lee el archivo y lo retorna
    return pd.read_csv(path, delimiter='\t') # Agregamos el delimitador para que haga el split correctamente
"""
    
def read_inegi(path):
    try:
        # Lee el archivo y lo retorna
        dataframe = pd.read_csv(path)
        return dataframe
    except FileNotFoundError as e:
        print(f"El archivo {path} no se encuentra en la ruta especificada")
        raise e
    except Exception as e:
        print(f"Error al leer el archivo {path}")
        raise e