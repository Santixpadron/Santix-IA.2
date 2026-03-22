import os
import cv2
import numpy as np

# Emociones requeridas según entrenamiento.py
emociones = ['felicidad', 'enojo', 'asombro', 'tristeza', 'neutral', 'miedo', 'disgusto']
ruta_base = "data"

print("Iniciando generación de datos sintéticos...")

for emocion in emociones:
    carpeta = os.path.join(ruta_base, emocion)
    os.makedirs(carpeta, exist_ok=True)
    
    # Generar 100 imágenes sintéticas por emoción para asegurar buen volumen
    for i in range(100):
        # Crear una imagen 150x150 con ruido aleatorio
        # Las pequeñas variaciones ayudarán a que los algoritmos matemáticos no fallen
        img = np.random.randint(0, 256, (150, 150), dtype=np.uint8)
        
        # Guardar imagen
        ruta_img = os.path.join(carpeta, f"sintetico_{i}.jpg")
        cv2.imwrite(ruta_img, img)

print("Datos sintéticos generados exitosamente en todas las carpetas.")
