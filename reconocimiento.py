"""
reconocimiento.py — Módulo de Reconocimiento en Tiempo Real
============================================================
Carga un modelo entrenado (EigenFace, FisherFace o LBPH) y realiza
predicciones de emociones en tiempo real desde la webcam.

Interfaz dividida:
  - Izquierda: Frame de la webcam con detección y etiquetas
  - Derecha: Imagen de emoji representativo de la emoción

Funcionalidades:
  - Selección dinámica del modelo
  - Evaluación de umbral de confianza
  - Overlay con información de predicción
  - Registro en base de datos
"""

import cv2
import numpy as np
import os
import sys
import time
from database import Database


# ===================== CONFIGURACIÓN =====================

# Ruta al clasificador Haar Cascade
HAARCASCADE_PATH = os.path.join("haarcascades", "haarcascade_frontalface_default.xml")

# Carpeta de modelos entrenados
MODELOS_DIR = "modelos"

# Carpeta de emojis
EMOJIS_DIR = "emojis"

# Tamaño estandarizado (debe coincidir con entrenamiento.py)
FACE_SIZE = (150, 150)

# Mapeo de etiquetas numéricas a emociones
ETIQUETA_A_EMOCION = {
    0: 'felicidad',
    1: 'enojo',
    2: 'asombro',
    3: 'tristeza',
    4: 'neutral',
    5: 'miedo',
    6: 'disgusto'
}

# Umbrales de confianza por modelo (valores menores = mayor confianza)
# EigenFace y FisherFace usan distancias: valores altos = mala confianza
# LBPH usa distancias: valores más bajos = mejor match
UMBRALES = {
    'EigenFace': 6000.0,    # Umbral para EigenFace
    'FisherFace': 4000.0,   # Umbral para FisherFace
    'LBPH': 120.0           # Umbral para LBPH
}

# Archivos de modelos
ARCHIVOS_MODELO = {
    '1': ('EigenFace', os.path.join(MODELOS_DIR, 'modelo_eigenface.xml')),
    '2': ('FisherFace', os.path.join(MODELOS_DIR, 'modelo_fisherface.xml')),
    '3': ('LBPH', os.path.join(MODELOS_DIR, 'modelo_lbph.xml'))
}

# Colores para la interfaz (BGR)
COLORES_EMOCION = {
    'felicidad': (0, 255, 255),    # Amarillo
    'enojo': (0, 0, 255),          # Rojo
    'asombro': (255, 255, 0),      # Cian
    'tristeza': (255, 100, 0),     # Azul
    'neutral': (200, 200, 200),    # Gris
    'miedo': (128, 0, 128),        # Púrpura
    'disgusto': (0, 128, 0),       # Verde oscuro
    'No identificado': (100, 100, 100)
}


def seleccionar_modelo():
    """
    Muestra un menú en consola para seleccionar el modelo a usar.
    
    Returns:
        tuple: (nombre_modelo, ruta_archivo) del modelo seleccionado.
    """
    print("\n" + "=" * 50)
    print("  SELECCIÓN DE MODELO")
    print("=" * 50)

    for tecla, (nombre, ruta) in ARCHIVOS_MODELO.items():
        existe = "✓" if os.path.exists(ruta) else "✗ (no entrenado)"
        print(f"  [{tecla}] {nombre:15s} {existe}")

    print()
    while True:
        opcion = input("  Seleccione modelo (1/2/3): ").strip()
        if opcion in ARCHIVOS_MODELO:
            nombre, ruta = ARCHIVOS_MODELO[opcion]
            if os.path.exists(ruta):
                return nombre, ruta
            else:
                print(f"  [ERROR] El modelo '{nombre}' no ha sido entrenado.")
                print("          Ejecute primero 'entrenamiento.py'.")
        else:
            print("  [ERROR] Opción no válida. Use 1, 2 o 3.")


def cargar_modelo(nombre_modelo, ruta_modelo):
    """
    Carga un modelo de reconocimiento facial según el tipo especificado.
    
    Args:
        nombre_modelo (str): Nombre del modelo (EigenFace, FisherFace, LBPH).
        ruta_modelo (str): Ruta al archivo XML del modelo.
        
    Returns:
        cv2.face.FaceRecognizer: Modelo cargado y listo para predicción.
    """
    print(f"\n[INFO] Cargando modelo: {nombre_modelo}...")

    # Instanciar el reconocedor según el tipo
    if nombre_modelo == 'EigenFace':
        modelo = cv2.face.EigenFaceRecognizer_create()
    elif nombre_modelo == 'FisherFace':
        modelo = cv2.face.FisherFaceRecognizer_create()
    elif nombre_modelo == 'LBPH':
        modelo = cv2.face.LBPHFaceRecognizer_create()
    else:
        print(f"[ERROR] Modelo desconocido: {nombre_modelo}")
        sys.exit(1)

    # Leer los parámetros del modelo entrenado
    modelo.read(ruta_modelo)
    print(f"[OK] Modelo '{nombre_modelo}' cargado desde: {ruta_modelo}")

    return modelo


def cargar_emoji(emocion):
    """
    Carga la imagen del emoji correspondiente a la emoción detectada.
    Busca archivos .png o .jpg en la carpeta /emojis/.
    
    Args:
        emocion (str): Nombre de la emoción.
        
    Returns:
        numpy.ndarray | None: Imagen del emoji o None si no se encuentra.
    """
    # Intentar cargar PNG primero, luego JPG
    for ext in ['.png', '.jpg', '.jpeg']:
        ruta = os.path.join(EMOJIS_DIR, f"{emocion}{ext}")
        if os.path.exists(ruta):
            emoji = cv2.imread(ruta, cv2.IMREAD_COLOR)
            if emoji is not None:
                return emoji

    return None


def crear_emoji_texto(emocion, tamaño=(200, 200)):
    """
    Crea una imagen con el nombre de la emoción como fallback
    cuando no hay imagen de emoji disponible.
    
    Args:
        emocion (str): Nombre de la emoción.
        tamaño (tuple): Dimensiones de la imagen (ancho, alto).
        
    Returns:
        numpy.ndarray: Imagen generada con texto.
    """
    img = np.zeros((tamaño[1], tamaño[0], 3), dtype=np.uint8)
    color = COLORES_EMOCION.get(emocion, (200, 200, 200))

    # Dibujar un círculo de fondo
    centro = (tamaño[0] // 2, tamaño[1] // 2 - 20)
    cv2.circle(img, centro, 60, color, -1)

    # Poner el texto de la emoción
    texto = emocion.upper()
    font_scale = 0.5 if len(texto) > 8 else 0.6
    tam_texto = cv2.getTextSize(texto, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)[0]
    pos_texto = ((tamaño[0] - tam_texto[0]) // 2, tamaño[1] - 30)
    cv2.putText(img, texto, pos_texto, cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, (255, 255, 255), 2)

    return img


def construir_interfaz(frame, emoji_img):
    """
    Construye la interfaz dividida usando cv2.hconcat():
    - Izquierda: Frame de la webcam con detección
    - Derecha: Imagen del emoji de la emoción
    
    Args:
        frame: Frame de video con anotaciones.
        emoji_img: Imagen del emoji a mostrar.
        
    Returns:
        numpy.ndarray: Imagen concatenada (interfaz completa).
    """
    alto_frame = frame.shape[0]
    ancho_frame = frame.shape[1]

    # Panel derecho: fondo oscuro con emoji centrado
    panel_ancho = 250
    panel = np.zeros((alto_frame, panel_ancho, 3), dtype=np.uint8)

    # Redimensionar emoji para que quepa en el panel
    emoji_max = min(panel_ancho - 20, alto_frame // 2)
    emoji_redim = cv2.resize(emoji_img, (emoji_max, emoji_max))

    # Centrar el emoji en el panel
    y_offset = (alto_frame - emoji_max) // 2
    x_offset = (panel_ancho - emoji_max) // 2
    panel[y_offset:y_offset + emoji_max, x_offset:x_offset + emoji_max] = emoji_redim

    # Concatenar horizontalmente: webcam | panel emoji
    interfaz = cv2.hconcat([frame, panel])

    return interfaz


def ejecutar_reconocimiento():
    """
    Función principal de reconocimiento en tiempo real.
    
    Captura video de la webcam, detecta rostros con Haar Cascade,
    predice la emoción usando el modelo seleccionado, y muestra
    una interfaz dividida con el emoji correspondiente.
    """
    # Verificar clasificador Haar Cascade
    if not os.path.exists(HAARCASCADE_PATH):
        print(f"[ERROR] Clasificador no encontrado: {HAARCASCADE_PATH}")
        sys.exit(1)

    # Seleccionar y cargar modelo
    nombre_modelo, ruta_modelo = seleccionar_modelo()
    modelo = cargar_modelo(nombre_modelo, ruta_modelo)
    umbral = UMBRALES[nombre_modelo]

    # Cargar clasificador Haar Cascade
    face_cascade = cv2.CascadeClassifier(HAARCASCADE_PATH)
    if face_cascade.empty():
        print("[ERROR] No se pudo cargar el clasificador Haar Cascade.")
        sys.exit(1)

    # Inicializar base de datos
    db = Database()

    # Pre-cargar emojis para todas las emociones
    emojis = {}
    for emocion in ETIQUETA_A_EMOCION.values():
        emoji = cargar_emoji(emocion)
        if emoji is not None:
            emojis[emocion] = emoji
        else:
            emojis[emocion] = crear_emoji_texto(emocion)
    emojis['No identificado'] = crear_emoji_texto('No identificado')
    print("[OK] Emojis cargados.")

    # Iniciar captura de video
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] No se pudo abrir la cámara web.")
        sys.exit(1)
    print("[OK] Cámara web iniciada.")
    print(f"[INFO] Modelo activo: {nombre_modelo} (umbral: {umbral})")
    print("[INFO] Presione ESC para salir.")

    # Variables para FPS y conteo
    frame_count = 0
    fps_inicio = time.time()
    fps_actual = 0
    ultima_emocion = 'neutral'

    while True:
        # Leer frame de la cámara
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] No se pudo leer el frame.")
            break

        # Voltear horizontalmente (efecto espejo)
        frame = cv2.flip(frame, 1)

        # Calcular FPS
        frame_count += 1
        if frame_count % 30 == 0:
            fps_actual = 30 / (time.time() - fps_inicio)
            fps_inicio = time.time()

        # Convertir a escala de grises para detección
        gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detectar rostros
        rostros = face_cascade.detectMultiScale(
            gris,
            scaleFactor=1.3,
            minNeighbors=5,
            minSize=(80, 80)
        )

        emocion_detectada = ultima_emocion

        for (x, y, w, h) in rostros:
            # Recortar el área del rostro
            rostro_gris = gris[y:y + h, x:x + w]

            # Redimensionar al tamaño usado en el entrenamiento
            rostro_redim = cv2.resize(rostro_gris, FACE_SIZE)

            # Realizar predicción con el modelo
            etiqueta, confianza = modelo.predict(rostro_redim)

            # Evaluar la confianza contra el umbral del modelo
            if confianza > umbral:
                # Confianza por encima del umbral = no identificado
                emocion_detectada = 'No identificado'
                texto_emocion = "No identificado"
                color = COLORES_EMOCION['No identificado']
            else:
                # Predicción válida
                emocion_detectada = ETIQUETA_A_EMOCION.get(etiqueta, 'neutral')
                confianza_pct = max(0, 100 - (confianza / umbral * 100))
                texto_emocion = f"{emocion_detectada.capitalize()} ({confianza_pct:.0f}%)"
                color = COLORES_EMOCION.get(emocion_detectada, (200, 200, 200))

                # Guardar detección en la base de datos (cada 30 frames)
                if frame_count % 30 == 0:
                    db.guardar_deteccion(
                        persona_id=None,
                        nombre_persona="Usuario",
                        emocion=emocion_detectada,
                        confianza=confianza,
                        modelo_usado=nombre_modelo
                    )

            ultima_emocion = emocion_detectada

            # Dibujar rectángulo alrededor del rostro
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

            # Fondo para el texto
            cv2.rectangle(frame, (x, y - 35), (x + w, y), color, -1)

            # Texto de la emoción
            cv2.putText(frame, texto_emocion, (x + 5, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        # Información en la parte superior del frame
        cv2.putText(frame, f"Modelo: {nombre_modelo} | FPS: {fps_actual:.0f}",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(frame, "ESC: Salir",
                    (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

        # Obtener emoji de la última emoción detectada
        emoji_actual = emojis.get(emocion_detectada, emojis['neutral'])

        # Construir interfaz dividida: webcam | emoji
        interfaz = construir_interfaz(frame, emoji_actual)

        # Mostrar la interfaz completa
        cv2.imshow("Reconocimiento de Emociones - Sistema en Tiempo Real", interfaz)

        # Capturar tecla presionada
        tecla = cv2.waitKey(1) & 0xFF

        # ESC (código 27): Salir del programa
        if tecla == 27:
            print("\n[INFO] Saliendo del programa...")
            break

    # Liberar recursos
    cap.release()
    cv2.destroyAllWindows()
    db.cerrar()
    print("[OK] Recursos liberados. Programa finalizado.")


# ===================== PUNTO DE ENTRADA =====================
if __name__ == "__main__":
    print("=" * 60)
    print("  SISTEMA DE RECONOCIMIENTO EN TIEMPO REAL")
    print("  Reconocimiento Facial con Análisis de Emociones")
    print("=" * 60)

    ejecutar_reconocimiento()
