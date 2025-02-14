import tensorflow
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models

# Parámetros de configuración
# Dimensiones de entrada
ancho_imagen, alto_imagen = 128, 128 
tamano_lote = 32  
epocas = 1000 

# Directorios de nuestro dataset
directorio_entrenamiento = "dataSet/entrenamiento"
directorio_validacion = "dataSet/validacion"

# Generador de imágenes para entrenamiento (con aumentación de datos)
generador_entrenamiento = ImageDataGenerator(
    rescale=1.0 / 255, 
    
    rotation_range=50,  
    
    width_shift_range=0.3, 
    height_shift_range=0.3,  
    
    shear_range=15,  
    zoom_range=[0.5, 1.5],  
    vertical_flip=True,  
    horizontal_flip=True  )

# Generador de imágenes para validación (sin aumentación, solo normalización)
generador_validacion = ImageDataGenerator(rescale=1.0 / 255)

# Carga de imágenes desde directorios
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
     # Reducción de dimensionalidad,pasa por 32 filtros, y un kernel de 3x3 usando relu para mantener  valores positivos y 3 canale de color
    layers.MaxPooling2D((2, 2)), 

    # Segunda capa convolucional (más filtros para detectar patrones más abstractos)
    layers.Conv2D(64, (3, 3), activation="relu"),#
    #Reduce la dimensionalidad de la imagen para disminuir la cantidad de datos y hacer el modelo más eficiente/Toma una ventana de 2x2 píxeles y selecciona el valor máximo dentro de esa región
    layers.MaxPooling2D((2, 2)),

    # Tercera capa convolucional (aumento de filtros para captar más detalles del rostro)
    layers.Conv2D(128, (3,  3), activation="relu"),
    layers.MaxPooling2D((2, 2)),

    # Cuarta capa convolucional (mayor profundidad para refinar características)
    layers.Conv2D(256, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),

    # Regularización para evitar sobreajuste
    layers.Dropout(0.5),

    # Aplana las características obtenidas y las pasa a una red densa
    layers.Flatten(),
    layers.Dense(128, activation="relu"),  
    layers.Dense(3, activation="softmax") 
])

# Compilación del modelo
RED_NEURONAL_CONVOLUCIONAL.compile(
    #Usando el optimizador adam ajusta los pesos automaticamente y usando clasificacion multiclase
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

entrenar_red = RED_NEURONAL_CONVOLUCIONAL.fit(
    datos_entrenamiento,
    epochs=epocas,
    validation_data=datos_validacion
)
#Se evalúa la precisión y la pérdida en los datos de validación.
puntaje = RED_NEURONAL_CONVOLUCIONAL.evaluate(datos_validacion)
#Se evalúa la precisión y la pérdida en los datos de validación.
print(f"Pérdida: {puntaje[0]:.4f}, Precisión: {puntaje[1]:.4f}")

#H5 muestra la ARQUTECTURA(CAPAS Y CONEXIONES) Y PESOS
RED_NEURONAL_CONVOLUCIONAL.save("modelo_clasificador_rostros.h5")
#weights.h5 muestra SOLO PESOS
RED_NEURONAL_CONVOLUCIONAL.save_weights("pesos.weights.h5")
