# utils/constants.py
img_height = 256
img_width = 256
num_classes = 9

# Diccionario para el mapa de colores de las clases (etiquetas en la máscara)
color_map = [
    (0, 0, 0),        # background - Negro
    (255, 0, 0),      # building-flooded - Rojo
    (180, 120, 120),  # building-non-flooded - Gris claro
    (160, 150, 20),   # road-flooded - Marrón
    (140, 140, 140),  # road-non-flooded - Gris
    (61, 230, 250),   # water - Celeste
    (0, 82, 255),     # tree - Azul
    (255, 0, 245),    # vehicle - Fucsia
    (255, 235, 0),    # pool - Amarillo
    (4, 250, 7)       # grass - Verde
]
