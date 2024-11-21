# constants.py

# Dimensiones de las imágenes
img_height = 128
img_width = 128
num_classes = 9 # Número de clases

# Paleta de colores para las clases
color_map = {
    0: (0, 0, 0),       # Fondo/No clase
    1: (128, 0, 0),     # Construcciones urbanas
    2: (0, 128, 0),     # Vegetación
    3: (128, 128, 0),   # Agua
    4: (0, 0, 128),     # Carreteras
    5: (128, 0, 128),   # Escombros
    6: (0, 128, 128),   # Vehículos
    7: (64, 64, 64),    # Infraestructura dañada
    8: (192, 192, 192)  # Infraestructura no dañada
}