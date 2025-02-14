from flask import Flask, render_template, Response, request, redirect, url_for, session, jsonify, send_from_directory
import cv2
import tensorflow as tf
import numpy as np
import os
import time
import glob

app = Flask(__name__)
app.secret_key = 'supersecretkey'

# Cargar el modelo de reconocimiento facial
modelo = tf.keras.models.load_model("modelo_clasificador_rostros.h5")
usuarios = {
    "betsy": {"nombre": "Betsy Daniela Velazquez Osorio", "edad": 22, "modelo_id": 0,"img":"betsy"},
    "diana": {"nombre": "Diana Paola Velazquez Treviño", "edad": 21, "modelo_id": 1,"img":"diana"},
    "eder": {"nombre": "Eder Meneses Galván", "edad": 21, "modelo_id": 0,"img":"eder"}
}

# Cargar el clasificador de rostros Haar Cascade early store
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

captura = cv2.VideoCapture(0)
upload_folder = "static/uploads"
os.makedirs(upload_folder, exist_ok=True)

# Contador de validaciones exitosas
validaciones_exitosas = 0


@app.route('/')
def index():
    return render_template("index.html", usuarios=usuarios)


@app.route('/seleccionar/<usuario>')
def seleccionar_usuario(usuario):
    if usuario in usuarios:
        session['usuario'] = usuario
        print(f"Usuario seleccionado: {usuario}")  # Debugging
        return redirect(url_for('validacion'))
    return redirect(url_for('index'))


@app.route('/validacion')
def validacion():
    if 'usuario' not in session:
        return redirect(url_for('index'))
    return render_template("validacion.html", usuario=session['usuario'])


@app.route('/capturar_imagen', methods=['GET'])
def capturar_imagen():
    global validaciones_exitosas
    if 'usuario' not in session:
        return jsonify({"mensaje": "Usuario no seleccionado", "autorizado": False})

    usuario_id = usuarios[session['usuario']]['modelo_id']
    ret, frame = captura.read()

    if not ret:
        return jsonify({"mensaje": "Error al capturar el frame", "autorizado": False})

    # Convertir a escala de grises
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # **Detectar nivel de luz**
    brillo_promedio = np.mean(gray)
    umbral_luz = 50  # Ajusta este valor según pruebas

    if brillo_promedio < umbral_luz:
        validaciones_exitosas = 0  # Reiniciar si el ambiente es oscuro
        return jsonify({
            "mensaje": "Ambiente demasiado oscuro",
            "autorizado": False,
            "imagen": None,
            "validaciones": validaciones_exitosas
        })

    # Detectar rostros
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) == 0:
        validaciones_exitosas = 0  # Reiniciar si no detecta rostros
        return jsonify({
            "mensaje": "No se detectó ningún rostro",
            "autorizado": False,
            "imagen": None,
            "validaciones": validaciones_exitosas
        })

    for (x, y, w, h) in faces:
        rostro = frame[y:y + h, x:x + w]
        rostro = cv2.resize(rostro, (128, 128)) / 255.0  # Normalización
        rostro = np.expand_dims(rostro, axis=0)

        prediccion = modelo.predict(rostro)
        confianza = np.max(prediccion)  # Confianza de la predicción

        if np.argmax(prediccion) == usuario_id and confianza > 0.7:  # Umbral de confianza
            validaciones_exitosas += 0.1
        else:
            validaciones_exitosas = 0

        # Guardar imagen
        imagen_path = os.path.join(upload_folder, "captura.jpg")
        cv2.imwrite(imagen_path, frame)

        # **Autenticación basada en tiempo continuo**
        if validaciones_exitosas >= 1:  # 3 segundos acumulados
            session['autenticado'] = True
            limpiar_carpeta()
            return jsonify({
                "mensaje": "Autenticación exitosa",
                "autorizado": True,
                "imagen": "captura.jpg",
                "validaciones": validaciones_exitosas
            })

    return jsonify({
        "mensaje": "Rostro no reconocido o tiempo insuficiente",
        "autorizado": False,
        "imagen": "captura.jpg",
        "validaciones": validaciones_exitosas
    })



@app.route('/imagen_actual')
def imagen_actual():
    return send_from_directory(upload_folder, "captura.jpg")


@app.route('/bienvenida')
def bienvenida():
    if 'autenticado' in session and session['autenticado']:
        usuario = usuarios[session['usuario']]
        return redirect(url_for('pagina_bienvenida', usuario=usuario['nombre'], edad=usuario['edad'], img=usuario['img']))
    return redirect(url_for('index'))

@app.route('/pagina_bienvenida')
def pagina_bienvenida():
    return render_template("bienvenida.html")



def limpiar_carpeta():
    """ Elimina todas las imágenes en la carpeta uploads. """
    archivos = glob.glob(os.path.join(upload_folder, "*"))
    for archivo in archivos:
        os.remove(archivo)


if __name__ == '__main__':
    app.run(debug=True)
