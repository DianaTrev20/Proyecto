import tensorflow
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models

# Parámetros de configuración
ancho_imagen, alto_imagen = 128, 128  # Dimensiones de entrada
tamano_lote = 32  # Tamaño de lote para entrenamiento
epocas = 1000  # Número de épocas

# Directorios de las imágenes
directorio_entrenamiento = "dataSet/entrenamiento"
directorio_validacion = "dataSet/validacion"

# Generador de imágenes para entrenamiento (con aumentación de datos)
generador_entrenamiento = ImageDataGenerator(
    rescale=1.0 / 255,  # Normaliza los valores de los píxeles entre 0 y 1 (antes estaban entre 0 y 255).
    
    rotation_range=50,  # Rota las imágenes aleatoriamente hasta 50 grados en cualquier dirección.
    
    width_shift_range=0.3,  # Desplaza la imagen horizontalmente hasta un 30% del ancho total.
    height_shift_range=0.3,  # Desplaza la imagen verticalmente hasta un 30% de la altura total.
    
    shear_range=15,  # Aplica una transformación de corte (shear) de hasta 15 grados, distorsionando la imagen.
    
    zoom_range=[0.5, 1.5],  # Hace zoom aleatorio entre el 50% y el 150% del tamaño original de la imagen.
    
    vertical_flip=True,  # Voltea la imagen verticalmente de manera aleatoria.
    horizontal_flip=True  # Voltea la imagen horizontalmente de manera aleatoria.
)

# Generador de imágenes para validación (sin aumentación, solo normalización)
generador_validacion = ImageDataGenerator(rescale=1.0 / 255)

# Cargar imágenes desde directorios
datos_entrenamiento = generador_entrenamiento.flow_from_directory(
    directorio_entrenamiento,
    target_size=(ancho_imagen, alto_imagen),
    batch_size=tamano_lote,
    class_mode="categorical"
)

datos_validacion = generador_validacion.flow_from_directory(
    directorio_validacion,
    target_size=(ancho_imagen, alto_imagen),
    batch_size=tamano_lote,
    class_mode="categorical"
)

# ========================== Definición de la Red Convolucional ==========================

RED_NEURONAL_CONVOLUCIONAL = models.Sequential([
   
   # Primera capa convolucional (extracción de características básicas)
    layers.Conv2D(32, (3, 3), activation="relu", input_shape=(ancho_imagen, alto_imagen, 3)),
    layers.MaxPooling2D((2, 2)),  # Reducción de dimensionalidad,pasa por 32 filtros, y un kernel de 3x3 usando relu para mantener  valores positivos y 3 canale de color

    # Segunda capa convolucional (más filtros para detectar patrones más abstractos)
    layers.Conv2D(64, (3, 3), activation="relu"),#
    layers.MaxPooling2D((2, 2)),#Reduce la dimensionalidad de la imagen para disminuir la cantidad de datos y hacer el modelo más eficiente/Toma una ventana de 2x2 píxeles y selecciona el valor máximo dentro de esa región

    # Tercera capa convolucional (aumento de filtros para captar más detalles del rostro)
    layers.Conv2D(128, (3,  3), activation="relu"),
    layers.MaxPooling2D((2, 2)),

    # Cuarta capa convolucional (mayor profundidad para refinar características)
    layers.Conv2D(256, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),

    # Regularización para evitar sobreajuste
    layers.Dropout(0.5),

    # Aplanar las características obtenidas y pasarlas a una red densa
    layers.Flatten(),
    layers.Dense(128, activation="relu"),  # Capa oculta con 128 neuronas
    layers.Dense(3, activation="softmax")  # Capa de salida con 3 neuronas (una por persona)
])

# Compilación del modelo
RED_NEURONAL_CONVOLUCIONAL.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# Entrenamiento de la red
entrenar_red = RED_NEURONAL_CONVOLUCIONAL.fit(
    datos_entrenamiento,
    epochs=epocas,
    validation_data=datos_validacion
)

# Evaluación de la red
puntaje = RED_NEURONAL_CONVOLUCIONAL.evaluate(datos_validacion)
print(f"Pérdida: {puntaje[0]:.4f}, Precisión: {puntaje[1]:.4f}")

# Guardar el modelo entrenado
RED_NEURONAL_CONVOLUCIONAL.save("modelo_clasificador_rostros.h5")
RED_NEURONAL_CONVOLUCIONAL.save_weights("pesos.weights.h5")
