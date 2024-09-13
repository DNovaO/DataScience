import pandas as pd
from sklearn.preprocessing import StandardScaler

# Lee el archivo TXT
df = pd.read_csv(r'C:\Users\Oscar Garcia\Documents\GitHub\DataScience\Programas\data\diabetes.tab.txt', sep='\t')

# Convierte a tipo float
df = df.astype(float)

# Elimina espacios en blanco
df = df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)

# Reemplaza valores nulos con el promedio de la columna
df = df.fillna(df.mean())

# Escala los valores
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df)
df_scaled = pd.DataFrame(df_scaled, columns=df.columns)

# Guarda los datos normalizados
df_scaled.to_csv('diabetesnorm.csv', index=False)

print("Datos normalizados guardados en 'diabetesnorm.csv'")
