# 🎭 Sistema de Reconocimiento Facial y Análisis de Emociones

Sistema completo de visión artificial para reconocimiento facial y análisis de emociones en tiempo real, desarrollado con Python y OpenCV.

## 📋 Descripción

Este proyecto implementa un sistema de reconocimiento facial con análisis de emociones que utiliza tres algoritmos de OpenCV para la clasificación:

- **EigenFace Recognizer** — Basado en PCA (Análisis de Componentes Principales)
- **FisherFace Recognizer** — Basado en LDA (Análisis Discriminante Lineal)
- **LBPH Recognizer** — Basado en Patrones Binarios Locales

### Emociones Detectadas (7)
😊 Felicidad | 😡 Enojo | 😲 Asombro | 😢 Tristeza | 😐 Neutral | 😨 Miedo | 🤢 Disgusto

## 🏗️ Estructura del Proyecto

```
Santix-IA.2/
├── 📁 data/                    # Imágenes de rostros capturados
│   ├── felicidad/
│   ├── enojo/
│   ├── asombro/
│   ├── tristeza/
│   ├── neutral/
│   ├── miedo/
│   └── disgusto/
├── 📁 haarcascades/            # Clasificadores preentrenados XML
│   └── haarcascade_frontalface_default.xml
├── 📁 modelos/                 # Modelos entrenados (.xml)
│   ├── modelo_eigenface.xml
│   ├── modelo_fisherface.xml
│   └── modelo_lbph.xml
├── 📁 emojis/                  # Imágenes de emojis para feedback visual
├── 📄 captura_rostros.py       # Fase 1: Captura de rostros desde webcam
├── 📄 entrenamiento.py         # Fase 2: Entrenamiento de los 3 modelos
├── 📄 reconocimiento.py        # Fase 3: Reconocimiento en tiempo real
├── 📄 gui_app.py               # Interfaz gráfica Tkinter (3 pantallas)
├── 📄 database.py              # Módulo de base de datos SQLite
├── 📄 requirements.txt         # Dependencias del proyecto
├── 📄 .gitignore               # Archivos ignorados por Git
└── 📄 README.md                # Este archivo
```

## 🚀 Instalación y Configuración

### Requisitos Previos
- Python 3.10+
- Webcam funcional
- Sistema operativo: Windows 10/11

### Pasos de Instalación

```bash
# 1. Clonar el repositorio
git clone https://github.com/Lejayk/Santix-IA.2.git
cd Santix-IA.2

# 2. Crear entorno virtual
python -m venv .venv

# 3. Activar entorno virtual (Windows)
.venv\Scripts\activate

# 4. Instalar dependencias
pip install -r requirements.txt
```

## 📖 Guía de Uso

### Fase 1: Captura de Rostros
```bash
python captura_rostros.py
```
- Ingresa el nombre de la persona
- Usa las teclas **1-7** para seleccionar la emoción
- Presiona **ESPACIO** para iniciar/detener la captura
- Se guardan hasta **200 imágenes** por emoción
- Presiona **ESC** para salir

### Fase 2: Entrenamiento de Modelos
```bash
python entrenamiento.py
```
- Lee automáticamente las imágenes de `/data/`
- Entrena los 3 modelos (EigenFace, FisherFace, LBPH)
- Muestra tiempos de entrenamiento
- Guarda modelos en `/modelos/`
- **Requiere**: Al menos 2 categorías de emociones con imágenes

### Fase 3: Reconocimiento en Tiempo Real
```bash
python reconocimiento.py
```
- Selecciona el modelo a utilizar (1/2/3)
- Detecta rostros y predice emociones en tiempo real
- Interfaz dividida: webcam (izq) + emoji (der)
- Presiona **ESC** para salir

### Interfaz Gráfica Completa
```bash
python gui_app.py
```

La GUI incluye:
- **📝 Registro**: Formulario de persona + captura de rostros con cámara
- **🔍 Detección**: Reconocimiento en tiempo real con overlay informativo
- **📊 Reportes**: Gráficos de emociones, historial, exportación CSV

## 🛠️ Tecnologías

| Tecnología | Versión | Uso |
|---|---|---|
| Python | 3.10+ | Lenguaje principal |
| OpenCV | 4.10+ | Visión artificial |
| OpenCV Contrib | 4.10+ | Módulos de reconocimiento facial |
| NumPy | 1.26+ | Operaciones con arrays |
| Matplotlib | 3.9+ | Gráficos de reportes |
| Pillow | 10.0+ | Procesamiento de imágenes |
| Tkinter | (incluido) | Interfaz gráfica |
| SQLite | (incluido) | Base de datos local |

## 📊 Modelos de Reconocimiento

| Modelo | Método | Velocidad | Precisión |
|---|---|---|---|
| EigenFace | PCA | ⚡ Rápido | Media |
| FisherFace | LDA | ⚡ Rápido | Media-Alta |
| LBPH | Patrones Locales | 🐢 Medio | Alta |

## 👤 Autor

**Lejayk** — [GitHub](https://github.com/Lejayk)

## 📄 Licencia

Este proyecto es de uso educativo y académico.
