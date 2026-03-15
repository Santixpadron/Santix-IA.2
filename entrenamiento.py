"""
entrenamiento.py — Módulo de Entrenamiento de Modelos
=====================================================
Entrena tres modelos de reconocimiento facial de OpenCV:
- EigenFaceRecognizer
- FisherFaceRecognizer
- LBPHFaceRecognizer

Lee las imágenes desde /data/, asigna etiquetas numéricas por
emoción, y guarda los modelos resultantes en /modelos/.
"""

import cv2
import numpy as np
import os
import sys
import time


# ===================== CONFIGURACIÓN =====================

# Carpeta con los datos de entrenamiento
DATA_DIR = "data"

# Carpeta donde se guardan los modelos entrenados
MODELOS_DIR = "modelos"

# Tamaño estandarizado de las imágenes (debe coincidir con captura_rostros.py)
FACE_SIZE = (150, 150)

# Mapeo de emociones a etiquetas numéricas
EMOCIONES = {
    'felicidad': 0,
    'enojo': 1,
    'asombro': 2,
    'tristeza': 3,
    'neutral': 4,
    'miedo': 5,
    'disgusto': 6
}

# Mapeo inverso: etiqueta numérica -> nombre de emoción
ETIQUETA_A_EMOCION = {v: k for k, v in EMOCIONES.items()}


def cargar_datos():
    """
    Lee todas las imágenes de las subcarpetas de /data/ y las convierte
    en arreglos NumPy con sus etiquetas correspondientes.
    
    Returns:
        tuple: (rostros, etiquetas, info) donde:
            - rostros: lista de arrays NumPy con las imágenes
            - etiquetas: lista de enteros con las etiquetas
            - info: diccionario con conteos por emoción
    """
    rostros = []
    etiquetas = []
    info = {}

    print("\n[INFO] Cargando datos de entrenamiento...")
    print("-" * 50)

    for emocion, etiqueta in EMOCIONES.items():
        ruta_emocion = os.path.join(DATA_DIR, emocion)

        # Verificar que la carpeta existe
        if not os.path.exists(ruta_emocion):
            print(f"  [ADVERTENCIA] Carpeta no encontrada: {ruta_emocion}")
            info[emocion] = 0
            continue

        # Leer todas las imágenes de la carpeta
        imagenes = [f for f in os.listdir(ruta_emocion) if f.endswith(('.jpg', '.png'))]
        info[emocion] = len(imagenes)

        if len(imagenes) == 0:
            print(f"  [ADVERTENCIA] Sin imágenes para: {emocion}")
            continue

        for nombre_archivo in imagenes:
            ruta_imagen = os.path.join(ruta_emocion, nombre_archivo)

            # Leer la imagen en escala de grises
            imagen = cv2.imread(ruta_imagen, cv2.IMREAD_GRAYSCALE)

            if imagen is None:
                print(f"  [ERROR] No se pudo leer: {ruta_imagen}")
                continue

            # Asegurar que la imagen tiene el tamaño correcto
            if imagen.shape != (FACE_SIZE[1], FACE_SIZE[0]):
                imagen = cv2.resize(imagen, FACE_SIZE)

            # Agregar a las listas
            rostros.append(imagen)
            etiquetas.append(etiqueta)

        print(f"  [OK] {emocion.capitalize():12s}: {len(imagenes):4d} imágenes cargadas")

    print("-" * 50)
    print(f"  TOTAL: {len(rostros)} imágenes cargadas")

    return rostros, etiquetas, info


def entrenar_modelos(rostros, etiquetas):
    """
    Instancia y entrena los tres modelos de reconocimiento facial
    de OpenCV, midiendo el tiempo de entrenamiento de cada uno.
    
    Args:
        rostros (list): Lista de arrays NumPy con las imágenes de rostros.
        etiquetas (list): Lista de enteros con las etiquetas correspondientes.
        
    Returns:
        dict: Diccionario con nombre_modelo -> (modelo, tiempo_entrenamiento)
    """
    # Convertir listas a arrays de NumPy
    rostros_np = np.array(rostros)
    etiquetas_np = np.array(etiquetas, dtype=np.int32)

    # Crear carpeta de modelos si no existe
    os.makedirs(MODELOS_DIR, exist_ok=True)

    # Definir los tres modelos a entrenar
    modelos = {
        'EigenFace': {
            'instancia': cv2.face.EigenFaceRecognizer_create(),
            'archivo': os.path.join(MODELOS_DIR, 'modelo_eigenface.xml')
        },
        'FisherFace': {
            'instancia': cv2.face.FisherFaceRecognizer_create(),
            'archivo': os.path.join(MODELOS_DIR, 'modelo_fisherface.xml')
        },
        'LBPH': {
            'instancia': cv2.face.LBPHFaceRecognizer_create(),
            'archivo': os.path.join(MODELOS_DIR, 'modelo_lbph.xml')
        }
    }

    resultados = {}

    print("\n" + "=" * 60)
    print("  ENTRENAMIENTO DE MODELOS")
    print("=" * 60)

    for nombre, config in modelos.items():
        modelo = config['instancia']
        archivo = config['archivo']

        print(f"\n[ENTRENANDO] {nombre}...")

        # Medir el tiempo de entrenamiento usando el módulo time
        inicio = time.time()
        modelo.train(rostros_np, etiquetas_np)
        fin = time.time()

        # Calcular tiempo transcurrido
        tiempo = fin - inicio

        # Guardar el modelo entrenado en formato XML
        modelo.write(archivo)

        resultados[nombre] = (modelo, tiempo)

        print(f"  [OK] Modelo {nombre} entrenado en {tiempo:.4f} segundos")
        print(f"  [OK] Guardado en: {archivo}")

    return resultados


def mostrar_resumen(resultados, info):
    """
    Muestra un resumen del entrenamiento con tiempos y datos utilizados.
    
    Args:
        resultados (dict): Resultados del entrenamiento.
        info (dict): Información sobre los datos cargados.
    """
    print("\n" + "=" * 60)
    print("  RESUMEN DE ENTRENAMIENTO")
    print("=" * 60)

    # Datos utilizados
    print("\n  Datos de entrenamiento:")
    for emocion, conteo in info.items():
        print(f"    {emocion.capitalize():12s}: {conteo:4d} imágenes")

    # Tiempos de entrenamiento
    print("\n  Tiempos de entrenamiento:")
    print(f"    {'Modelo':<15s} {'Tiempo (s)':>12s}")
    print(f"    {'-'*15} {'-'*12}")
    for nombre, (_, tiempo) in resultados.items():
        print(f"    {nombre:<15s} {tiempo:>12.4f}")

    # Modelo más rápido
    mas_rapido = min(resultados.items(), key=lambda x: x[1][1])
    print(f"\n  Modelo más rápido: {mas_rapido[0]} ({mas_rapido[1][1]:.4f}s)")

    # Archivos generados
    print("\n  Archivos generados:")
    for archivo in os.listdir(MODELOS_DIR):
        ruta = os.path.join(MODELOS_DIR, archivo)
        tamaño = os.path.getsize(ruta) / 1024  # KB
        print(f"    {archivo}: {tamaño:.1f} KB")

    print("\n" + "=" * 60)
    print("  [OK] Entrenamiento completado exitosamente.")
    print("=" * 60)


# ===================== PUNTO DE ENTRADA =====================
if __name__ == "__main__":
    print("=" * 60)
    print("  SISTEMA DE ENTRENAMIENTO")
    print("  Reconocimiento Facial con Análisis de Emociones")
    print("=" * 60)

    # Cargar datos de entrenamiento
    rostros, etiquetas, info = cargar_datos()

    # Verificar que hay datos suficientes
    if len(rostros) == 0:
        print("\n[ERROR] No se encontraron imágenes de entrenamiento.")
        print("        Ejecuta primero 'captura_rostros.py' para capturar datos.")
        sys.exit(1)

    # Verificar que haya al menos 2 clases (requerido por EigenFace y FisherFace)
    clases_unicas = len(set(etiquetas))
    if clases_unicas < 2:
        print(f"\n[ERROR] Se necesitan al menos 2 emociones diferentes para entrenar.")
        print(f"        Emociones encontradas: {clases_unicas}")
        print("        Captura imágenes para al menos 2 emociones distintas.")
        sys.exit(1)

    print(f"\n[INFO] Clases (emociones) encontradas: {clases_unicas}")
    print(f"[INFO] Total de imágenes: {len(rostros)}")

    # Entrenar los tres modelos
    resultados = entrenar_modelos(rostros, etiquetas)

    # Mostrar resumen
    mostrar_resumen(resultados, info)
