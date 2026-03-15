"""
captura_rostros.py — Módulo de Captura de Rostros
==================================================
Captura rostros desde la webcam usando OpenCV y los almacena en
subcarpetas organizadas por emoción dentro de /data/.
Utiliza Haar Cascade para la detección facial en tiempo real.
"""

import cv2
import numpy as np
import os
import sys


# ===================== CONFIGURACIÓN =====================

# Ruta al clasificador Haar Cascade para detección frontal de rostros
HAARCASCADE_PATH = os.path.join("haarcascades", "haarcascade_frontalface_default.xml")

# Carpeta base donde se almacenan los datos faciales
DATA_DIR = "data"

# Tamaño estandarizado para las imágenes de rostros (ancho x alto)
FACE_SIZE = (150, 150)

# Número máximo de imágenes a capturar por emoción
MAX_IMAGENES = 200

# Diccionario de emociones con sus teclas de activación
EMOCIONES = {
    '1': 'felicidad',
    '2': 'enojo',
    '3': 'asombro',
    '4': 'tristeza',
    '5': 'neutral',
    '6': 'miedo',
    '7': 'disgusto'
}

# Colores para la interfaz (BGR)
COLOR_RECT = (0, 255, 0)       # Verde para el rectángulo del rostro
COLOR_TEXTO = (255, 255, 255)  # Blanco para texto
COLOR_ACTIVO = (0, 255, 255)   # Amarillo para emoción activa
COLOR_FONDO = (40, 40, 40)     # Gris oscuro para el panel


def crear_carpetas():
    """
    Crea la estructura de carpetas para almacenar las imágenes
    de cada emoción si no existen.
    """
    for emocion in EMOCIONES.values():
        ruta = os.path.join(DATA_DIR, emocion)
        os.makedirs(ruta, exist_ok=True)
    print("[INFO] Carpetas de emociones verificadas/creadas.")


def contar_imagenes(emocion):
    """
    Cuenta las imágenes existentes en la carpeta de una emoción.
    
    Args:
        emocion (str): Nombre de la emoción.
        
    Returns:
        int: Número de imágenes en la carpeta.
    """
    ruta = os.path.join(DATA_DIR, emocion)
    if not os.path.exists(ruta):
        return 0
    return len([f for f in os.listdir(ruta) if f.endswith(('.jpg', '.png'))])


def dibujar_panel_info(frame, emocion_actual, capturando, contador):
    """
    Dibuja un panel informativo en la parte inferior del frame con
    las instrucciones y el estado actual de la captura.
    
    Args:
        frame: Frame de video actual.
        emocion_actual (str): Emoción seleccionada actualmente.
        capturando (bool): Si se está capturando activamente.
        contador (int): Número de imágenes capturadas.
    """
    alto, ancho = frame.shape[:2]
    panel_alto = 200
    
    # Fondo semitransparente para el panel
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, alto - panel_alto), (ancho, alto), COLOR_FONDO, -1)
    cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
    
    # Título del panel
    y_base = alto - panel_alto + 25
    cv2.putText(frame, "=== CAPTURA DE ROSTROS ===", (10, y_base),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_TEXTO, 1)
    
    # Instrucciones de teclas
    y_base += 25
    cv2.putText(frame, "Teclas [1-7]: Seleccionar emocion | [SPACE]: Capturar | [ESC]: Salir",
                (10, y_base), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)
    
    # Lista de emociones con conteo
    y_base += 30
    for tecla, emocion in EMOCIONES.items():
        total = contar_imagenes(emocion)
        color = COLOR_ACTIVO if emocion == emocion_actual else (150, 150, 150)
        indicador = " <<" if emocion == emocion_actual else ""
        texto = f"[{tecla}] {emocion.capitalize():12s} ({total:3d}/{MAX_IMAGENES}){indicador}"
        cv2.putText(frame, texto, (10, y_base),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        y_base += 18
    
    # Estado de captura
    if capturando:
        cv2.putText(frame, f"CAPTURANDO: {emocion_actual} [{contador}/{MAX_IMAGENES}]",
                    (ancho // 2 - 150, alto - panel_alto - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)


def capturar_rostros(nombre_persona="persona"):
    """
    Función principal de captura de rostros.
    
    Abre la webcam, detecta rostros con Haar Cascade, y permite
    al usuario capturar imágenes categorizadas por emoción.
    
    Args:
        nombre_persona (str): Identificador de la persona para el nombre de archivo.
    """
    # Verificar que el clasificador existe
    if not os.path.exists(HAARCASCADE_PATH):
        print(f"[ERROR] No se encontró el clasificador: {HAARCASCADE_PATH}")
        print("        Descarga 'haarcascade_frontalface_default.xml' y colócalo en /haarcascades/")
        sys.exit(1)

    # Crear carpetas de emociones
    crear_carpetas()

    # Cargar el clasificador Haar Cascade
    face_cascade = cv2.CascadeClassifier(HAARCASCADE_PATH)
    if face_cascade.empty():
        print("[ERROR] No se pudo cargar el clasificador Haar Cascade.")
        sys.exit(1)
    print("[OK] Clasificador Haar Cascade cargado.")

    # Iniciar captura de video desde la webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] No se pudo abrir la cámara web.")
        sys.exit(1)
    print("[OK] Cámara web iniciada.")
    print("[INFO] Usa las teclas 1-7 para seleccionar emoción, SPACE para capturar, ESC para salir.")

    # Variables de estado
    emocion_actual = 'neutral'
    capturando = False
    contador = 0

    while True:
        # Leer frame de la cámara
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] No se pudo leer el frame de la cámara.")
            break

        # Voltear el frame horizontalmente (efecto espejo)
        frame = cv2.flip(frame, 1)

        # Convertir a escala de grises para la detección
        gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detectar rostros usando Haar Cascade
        # scaleFactor: Compensación por cercanía a la cámara
        # minNeighbors: Cuántos vecinos necesita un candidato para ser retenido
        # minSize: Tamaño mínimo del rostro detectado
        rostros = face_cascade.detectMultiScale(
            gris,
            scaleFactor=1.3,
            minNeighbors=5,
            minSize=(80, 80)
        )

        for (x, y, w, h) in rostros:
            # Dibujar rectángulo alrededor del rostro detectado
            cv2.rectangle(frame, (x, y), (x + w, y + h), COLOR_RECT, 2)

            # Mostrar etiqueta de la emoción sobre el rectángulo
            etiqueta = f"{emocion_actual.capitalize()}"
            cv2.putText(frame, etiqueta, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_RECT, 2)

            # Si se está capturando activamente
            if capturando:
                total_actual = contar_imagenes(emocion_actual)
                if total_actual < MAX_IMAGENES:
                    # Recortar el área exacta del rostro
                    rostro_recortado = gris[y:y + h, x:x + w]

                    # Redimensionar a tamaño estandarizado
                    rostro_redim = cv2.resize(rostro_recortado, FACE_SIZE)

                    # Generar nombre de archivo único
                    nombre_archivo = f"{nombre_persona}_{emocion_actual}_{total_actual:04d}.jpg"
                    ruta_guardado = os.path.join(DATA_DIR, emocion_actual, nombre_archivo)

                    # Guardar la imagen
                    cv2.imwrite(ruta_guardado, rostro_redim)
                    contador += 1
                    print(f"  [CAPTURA] {ruta_guardado} ({total_actual + 1}/{MAX_IMAGENES})")
                else:
                    print(f"[INFO] Límite de {MAX_IMAGENES} imágenes alcanzado para '{emocion_actual}'.")
                    capturando = False

        # Dibujar panel informativo
        dibujar_panel_info(frame, emocion_actual, capturando, contador)

        # Mostrar el frame con las anotaciones
        cv2.imshow("Captura de Rostros - Sistema de Reconocimiento", frame)

        # Capturar tecla presionada
        tecla = cv2.waitKey(1) & 0xFF

        # ESC (código 27): Salir del programa
        if tecla == 27:
            print("\n[INFO] Saliendo del programa...")
            break

        # Teclas 1-7: Seleccionar emoción
        tecla_chr = chr(tecla) if tecla < 128 else ''
        if tecla_chr in EMOCIONES:
            emocion_actual = EMOCIONES[tecla_chr]
            capturando = False
            contador = 0
            print(f"\n[SELECCIÓN] Emoción cambiada a: {emocion_actual.upper()}")

        # SPACE: Iniciar/detener captura
        if tecla == 32:  # Barra espaciadora
            capturando = not capturando
            if capturando:
                contador = 0
                print(f"\n[CAPTURA] Iniciando captura para: {emocion_actual.upper()}")
            else:
                print(f"[CAPTURA] Captura detenida. Total capturadas: {contador}")

    # Liberar recursos de la cámara
    cap.release()
    # Destruir todas las ventanas de OpenCV
    cv2.destroyAllWindows()
    print("[OK] Recursos liberados. Programa finalizado.")


# ===================== PUNTO DE ENTRADA =====================
if __name__ == "__main__":
    print("=" * 60)
    print("  SISTEMA DE CAPTURA DE ROSTROS")
    print("  Reconocimiento Facial con Análisis de Emociones")
    print("=" * 60)

    # Solicitar nombre de la persona
    nombre = input("\nIngrese el nombre de la persona: ").strip()
    if not nombre:
        nombre = "persona"
    
    print(f"\n[INFO] Capturando rostros para: {nombre}")
    print("[INFO] Emociones disponibles:")
    for tecla, emocion in EMOCIONES.items():
        print(f"  [{tecla}] {emocion.capitalize()}")
    print()
    
    capturar_rostros(nombre_persona=nombre)
