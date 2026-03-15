"""
gui_app.py — Interfaz Gráfica Principal (Tkinter)
===================================================
Aplicación de escritorio con tres pantallas principales:
  1. Registro: Formulario + captura de rostros
  2. Detección: Reconocimiento en tiempo real con overlay
  3. Reportes: Gráficos de emociones y estadísticas

Requiere: opencv-python, opencv-contrib-python, numpy, matplotlib, Pillow
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import cv2
import numpy as np
import os
import sys
import time
import threading
from PIL import Image, ImageTk
from datetime import datetime

# Módulos internos del proyecto
from database import Database

# ===================== CONFIGURACIÓN GLOBAL =====================

HAARCASCADE_PATH = os.path.join("haarcascades", "haarcascade_frontalface_default.xml")
DATA_DIR = "data"
MODELOS_DIR = "modelos"
EMOJIS_DIR = "emojis"
FACE_SIZE = (150, 150)

EMOCIONES = {
    0: 'felicidad', 1: 'enojo', 2: 'asombro', 3: 'tristeza',
    4: 'neutral', 5: 'miedo', 6: 'disgusto'
}
EMOCION_A_ETIQUETA = {v: k for k, v in EMOCIONES.items()}

ARCHIVOS_MODELO = {
    'EigenFace': os.path.join(MODELOS_DIR, 'modelo_eigenface.xml'),
    'FisherFace': os.path.join(MODELOS_DIR, 'modelo_fisherface.xml'),
    'LBPH': os.path.join(MODELOS_DIR, 'modelo_lbph.xml')
}

UMBRALES = {
    'EigenFace': 6000.0,
    'FisherFace': 4000.0,
    'LBPH': 120.0
}

# Colores de la interfaz (tema oscuro)
BG_DARK = "#1a1a2e"
BG_CARD = "#16213e"
BG_INPUT = "#0f3460"
FG_TEXT = "#e6e6e6"
FG_ACCENT = "#00d4ff"
FG_SUCCESS = "#00e676"
FG_WARNING = "#ff9100"
FG_DANGER = "#ff1744"
FG_SUBTLE = "#8892b0"

# Colores de emociones para gráficos
COLORES_EMOCION_HEX = {
    'felicidad': '#FFD700',
    'enojo': '#FF4444',
    'asombro': '#00BFFF',
    'tristeza': '#4169E1',
    'neutral': '#AAAAAA',
    'miedo': '#9B59B6',
    'disgusto': '#2ECC71'
}


class AplicacionPrincipal(tk.Tk):
    """
    Ventana principal de la aplicación. Gestiona la navegación
    entre las tres pantallas: Registro, Detección y Reportes.
    """

    def __init__(self):
        super().__init__()

        self.title("Sistema de Reconocimiento Facial y Emociones")
        self.geometry("1100x750")
        self.minsize(1000, 700)
        self.configure(bg=BG_DARK)

        # Base de datos
        self.db = Database()

        # Clasificador Haar Cascade
        self.face_cascade = None
        self._cargar_cascade()

        # Cámara
        self.cap = None
        self.camara_activa = False

        # Crear interfaz
        self._crear_barra_navegacion()
        self._crear_contenedor_principal()

        # Mostrar pantalla de registro por defecto
        self.mostrar_pantalla("registro")

        # Manejar cierre de ventana
        self.protocol("WM_DELETE_WINDOW", self._al_cerrar)

    def _cargar_cascade(self):
        """Carga el clasificador Haar Cascade."""
        if os.path.exists(HAARCASCADE_PATH):
            self.face_cascade = cv2.CascadeClassifier(HAARCASCADE_PATH)
            if self.face_cascade.empty():
                self.face_cascade = None
                print("[WARN] Haar Cascade vacío.")
        else:
            print(f"[WARN] Haar Cascade no encontrado: {HAARCASCADE_PATH}")

    def _crear_barra_navegacion(self):
        """Crea la barra de navegación superior."""
        nav = tk.Frame(self, bg=BG_CARD, height=60)
        nav.pack(fill=tk.X, padx=0, pady=0)
        nav.pack_propagate(False)

        # Logo / Título
        tk.Label(nav, text="🎭 ReconoFace", font=("Segoe UI", 16, "bold"),
                 bg=BG_CARD, fg=FG_ACCENT).pack(side=tk.LEFT, padx=20)

        # Botones de navegación
        self.botones_nav = {}
        for nombre, texto, icono in [
            ("registro", "Registro", "📝"),
            ("deteccion", "Detección", "🔍"),
            ("reportes", "Reportes", "📊")
        ]:
            btn = tk.Button(
                nav, text=f" {icono} {texto} ",
                font=("Segoe UI", 11),
                bg=BG_DARK, fg=FG_TEXT,
                activebackground=FG_ACCENT, activeforeground=BG_DARK,
                relief=tk.FLAT, cursor="hand2",
                command=lambda n=nombre: self.mostrar_pantalla(n)
            )
            btn.pack(side=tk.LEFT, padx=5, pady=10)
            self.botones_nav[nombre] = btn

    def _crear_contenedor_principal(self):
        """Crea el contenedor donde se muestran las pantallas."""
        self.contenedor = tk.Frame(self, bg=BG_DARK)
        self.contenedor.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Crear las tres pantallas
        self.pantallas = {}
        self.pantallas["registro"] = PantallaRegistro(self.contenedor, self)
        self.pantallas["deteccion"] = PantallaDeteccion(self.contenedor, self)
        self.pantallas["reportes"] = PantallaReportes(self.contenedor, self)

        for pantalla in self.pantallas.values():
            pantalla.place(relx=0, rely=0, relwidth=1, relheight=1)

    def mostrar_pantalla(self, nombre):
        """
        Muestra la pantalla seleccionada y actualiza la barra de navegación.
        
        Args:
            nombre (str): Nombre de la pantalla ('registro', 'deteccion', 'reportes').
        """
        # Detener cámara si se cambia de pantalla
        if nombre != "deteccion":
            if hasattr(self.pantallas.get("deteccion", None), 'detener_deteccion'):
                self.pantallas["deteccion"].detener_deteccion()
        if nombre != "registro":
            if hasattr(self.pantallas.get("registro", None), 'detener_camara'):
                self.pantallas["registro"].detener_camara()

        # Actualizar estilo de botones
        for n, btn in self.botones_nav.items():
            if n == nombre:
                btn.configure(bg=FG_ACCENT, fg=BG_DARK)
            else:
                btn.configure(bg=BG_DARK, fg=FG_TEXT)

        # Mostrar pantalla
        self.pantallas[nombre].tkraise()

        # Actualizar datos si es reportes
        if nombre == "reportes":
            self.pantallas["reportes"].actualizar_datos()

    def iniciar_camara(self):
        """Abre la cámara web."""
        if self.cap is None or not self.cap.isOpened():
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                messagebox.showerror("Error", "No se pudo abrir la cámara web.")
                return False
        self.camara_activa = True
        return True

    def detener_camara(self):
        """Detiene y libera la cámara web."""
        self.camara_activa = False
        if self.cap is not None and self.cap.isOpened():
            self.cap.release()
            self.cap = None

    def _al_cerrar(self):
        """Maneja el cierre de la aplicación."""
        self.detener_camara()
        self.db.cerrar()
        self.destroy()


# ===================== PANTALLA DE REGISTRO =====================

class PantallaRegistro(tk.Frame):
    """
    Pantalla de registro de nuevas personas.
    Incluye formulario, vista previa de cámara y captura de imágenes.
    """

    def __init__(self, parent, app):
        super().__init__(parent, bg=BG_DARK)
        self.app = app
        self.capturando = False
        self.emocion_seleccionada = tk.StringVar(value='neutral')
        self.imagenes_capturadas = 0
        self._crear_interfaz()

    def _crear_interfaz(self):
        """Construye la interfaz de la pantalla de registro."""

        # === PANEL IZQUIERDO: Formulario ===
        panel_izq = tk.Frame(self, bg=BG_CARD, width=350)
        panel_izq.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 5))
        panel_izq.pack_propagate(False)

        # Título
        tk.Label(panel_izq, text="📝 Registro de Persona",
                 font=("Segoe UI", 14, "bold"), bg=BG_CARD, fg=FG_ACCENT
                 ).pack(pady=(20, 15))

        # Campos del formulario
        campos_frame = tk.Frame(panel_izq, bg=BG_CARD)
        campos_frame.pack(padx=20, fill=tk.X)

        # Nombre
        tk.Label(campos_frame, text="Nombre:", font=("Segoe UI", 10),
                 bg=BG_CARD, fg=FG_TEXT).pack(anchor=tk.W, pady=(5, 2))
        self.entry_nombre = tk.Entry(campos_frame, font=("Segoe UI", 11),
                                     bg=BG_INPUT, fg=FG_TEXT, insertbackground=FG_TEXT,
                                     relief=tk.FLAT)
        self.entry_nombre.pack(fill=tk.X, ipady=5)

        # Apellido
        tk.Label(campos_frame, text="Apellido:", font=("Segoe UI", 10),
                 bg=BG_CARD, fg=FG_TEXT).pack(anchor=tk.W, pady=(10, 2))
        self.entry_apellido = tk.Entry(campos_frame, font=("Segoe UI", 11),
                                       bg=BG_INPUT, fg=FG_TEXT, insertbackground=FG_TEXT,
                                       relief=tk.FLAT)
        self.entry_apellido.pack(fill=tk.X, ipady=5)

        # Email
        tk.Label(campos_frame, text="Email:", font=("Segoe UI", 10),
                 bg=BG_CARD, fg=FG_TEXT).pack(anchor=tk.W, pady=(10, 2))
        self.entry_email = tk.Entry(campos_frame, font=("Segoe UI", 11),
                                    bg=BG_INPUT, fg=FG_TEXT, insertbackground=FG_TEXT,
                                    relief=tk.FLAT)
        self.entry_email.pack(fill=tk.X, ipady=5)

        # Selector de emoción
        tk.Label(campos_frame, text="Emoción a capturar:", font=("Segoe UI", 10),
                 bg=BG_CARD, fg=FG_TEXT).pack(anchor=tk.W, pady=(15, 2))
        emociones_lista = list(EMOCION_A_ETIQUETA.keys())
        self.combo_emocion = ttk.Combobox(campos_frame, values=emociones_lista,
                                          textvariable=self.emocion_seleccionada,
                                          state="readonly", font=("Segoe UI", 10))
        self.combo_emocion.pack(fill=tk.X, ipady=3)

        # Indicador de calidad
        tk.Label(campos_frame, text="Calidad del registro:", font=("Segoe UI", 10),
                 bg=BG_CARD, fg=FG_TEXT).pack(anchor=tk.W, pady=(15, 2))
        self.barra_calidad = ttk.Progressbar(campos_frame, length=200, mode='determinate')
        self.barra_calidad.pack(fill=tk.X)
        self.label_calidad = tk.Label(campos_frame, text="0/200 imágenes",
                                      font=("Segoe UI", 9), bg=BG_CARD, fg=FG_SUBTLE)
        self.label_calidad.pack(anchor=tk.W)

        # Botones
        botones_frame = tk.Frame(campos_frame, bg=BG_CARD)
        botones_frame.pack(fill=tk.X, pady=20)

        self.btn_registrar = tk.Button(
            botones_frame, text="✅ Registrar Persona",
            font=("Segoe UI", 10, "bold"), bg=FG_ACCENT, fg=BG_DARK,
            activebackground="#00b8d4", relief=tk.FLAT, cursor="hand2",
            command=self._registrar_persona
        )
        self.btn_registrar.pack(fill=tk.X, ipady=8, pady=(0, 5))

        self.btn_capturar = tk.Button(
            botones_frame, text="📸 Iniciar Captura",
            font=("Segoe UI", 10, "bold"), bg=FG_SUCCESS, fg=BG_DARK,
            activebackground="#00c853", relief=tk.FLAT, cursor="hand2",
            command=self._toggle_captura
        )
        self.btn_capturar.pack(fill=tk.X, ipady=8, pady=(0, 5))

        self.btn_camara = tk.Button(
            botones_frame, text="🎥 Encender Cámara",
            font=("Segoe UI", 10), bg=BG_INPUT, fg=FG_TEXT,
            activebackground="#1a237e", relief=tk.FLAT, cursor="hand2",
            command=self._toggle_camara
        )
        self.btn_camara.pack(fill=tk.X, ipady=6)

        # Mensajes de estado
        self.label_status = tk.Label(panel_izq, text="",
                                     font=("Segoe UI", 9), bg=BG_CARD,
                                     fg=FG_SUBTLE, wraplength=300)
        self.label_status.pack(pady=10)

        # Lista de personas registradas
        tk.Label(panel_izq, text="Personas registradas:",
                 font=("Segoe UI", 10, "bold"), bg=BG_CARD, fg=FG_TEXT
                 ).pack(anchor=tk.W, padx=20, pady=(10, 5))
        
        lista_frame = tk.Frame(panel_izq, bg=BG_CARD)
        lista_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=(0, 10))
        
        self.lista_personas = tk.Listbox(lista_frame, font=("Segoe UI", 9),
                                          bg=BG_INPUT, fg=FG_TEXT,
                                          selectbackground=FG_ACCENT,
                                          relief=tk.FLAT)
        self.lista_personas.pack(fill=tk.BOTH, expand=True)
        self._actualizar_lista_personas()

        # === PANEL DERECHO: Vista previa de cámara ===
        self.panel_camara = tk.Frame(self, bg=BG_CARD)
        self.panel_camara.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.label_camara = tk.Label(self.panel_camara, bg=BG_CARD,
                                     text="🎥 Presiona 'Encender Cámara' para iniciar",
                                     font=("Segoe UI", 14), fg=FG_SUBTLE)
        self.label_camara.pack(expand=True)

    def _toggle_camara(self):
        """Enciende o apaga la cámara."""
        if not self.app.camara_activa:
            if self.app.iniciar_camara():
                self.btn_camara.configure(text="⏹ Apagar Cámara", bg=FG_DANGER)
                self._actualizar_camara()
        else:
            self.detener_camara()

    def detener_camara(self):
        """Detiene la cámara y la captura."""
        self.capturando = False
        self.btn_capturar.configure(text="📸 Iniciar Captura", bg=FG_SUCCESS)
        self.app.detener_camara()
        self.btn_camara.configure(text="🎥 Encender Cámara", bg=BG_INPUT)
        self.label_camara.configure(image='', text="🎥 Presiona 'Encender Cámara' para iniciar")

    def _actualizar_camara(self):
        """Actualiza el frame de la cámara en la interfaz."""
        if not self.app.camara_activa or self.app.cap is None:
            return

        ret, frame = self.app.cap.read()
        if ret:
            frame = cv2.flip(frame, 1)
            gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Detectar rostros
            if self.app.face_cascade is not None:
                rostros = self.app.face_cascade.detectMultiScale(
                    gris, scaleFactor=1.3, minNeighbors=5, minSize=(80, 80)
                )
                for (x, y, w, h) in rostros:
                    color = (0, 255, 0) if not self.capturando else (0, 0, 255)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

                    # Capturar si está activado
                    if self.capturando and len(rostros) > 0:
                        self._capturar_frame(gris, rostros[0])

            # Convertir frame para Tkinter
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)

            # Redimensionar para ajustar al panel
            panel_w = self.panel_camara.winfo_width()
            panel_h = self.panel_camara.winfo_height()
            if panel_w > 1 and panel_h > 1:
                ratio = min(panel_w / img.width, panel_h / img.height)
                new_size = (int(img.width * ratio), int(img.height * ratio))
                if new_size[0] > 0 and new_size[1] > 0:
                    img = img.resize(new_size, Image.LANCZOS)

            self._photo = ImageTk.PhotoImage(img)
            self.label_camara.configure(image=self._photo, text="")

        # Repetir cada 30ms (~33 FPS)
        if self.app.camara_activa:
            self.after(30, self._actualizar_camara)

    def _toggle_captura(self):
        """Inicia o detiene la captura de imágenes."""
        if not self.app.camara_activa:
            messagebox.showwarning("Aviso", "Primero enciende la cámara.")
            return

        self.capturando = not self.capturando
        if self.capturando:
            self.imagenes_capturadas = 0
            self.btn_capturar.configure(text="⏹ Detener Captura", bg=FG_DANGER)
            self.label_status.configure(text="Capturando rostros...", fg=FG_WARNING)
        else:
            self.btn_capturar.configure(text="📸 Iniciar Captura", bg=FG_SUCCESS)
            self.label_status.configure(
                text=f"Captura detenida. {self.imagenes_capturadas} imágenes guardadas.",
                fg=FG_SUCCESS
            )

    def _capturar_frame(self, gris, rostro_coords):
        """
        Captura y guarda un frame del rostro detectado.
        
        Args:
            gris: Frame en escala de grises.
            rostro_coords: Coordenadas (x, y, w, h) del rostro.
        """
        x, y, w, h = rostro_coords
        emocion = self.emocion_seleccionada.get()

        # Verificar límite
        ruta_emocion = os.path.join(DATA_DIR, emocion)
        os.makedirs(ruta_emocion, exist_ok=True)
        total = len([f for f in os.listdir(ruta_emocion) if f.endswith(('.jpg', '.png'))])

        if total >= 200:
            self.capturando = False
            self.btn_capturar.configure(text="📸 Iniciar Captura", bg=FG_SUCCESS)
            self.label_status.configure(text=f"Límite de 200 imágenes alcanzado para '{emocion}'.",
                                        fg=FG_WARNING)
            return

        # Recortar y redimensionar
        rostro = gris[y:y + h, x:x + w]
        rostro_redim = cv2.resize(rostro, FACE_SIZE)

        # Guardar
        nombre = self.entry_nombre.get().strip() or "persona"
        archivo = f"{nombre}_{emocion}_{total:04d}.jpg"
        cv2.imwrite(os.path.join(ruta_emocion, archivo), rostro_redim)

        self.imagenes_capturadas += 1

        # Actualizar indicador
        progreso = min(100, (total + 1) / 200 * 100)
        self.barra_calidad['value'] = progreso
        self.label_calidad.configure(text=f"{total + 1}/200 imágenes")

    def _registrar_persona(self):
        """Registra una persona en la base de datos."""
        nombre = self.entry_nombre.get().strip()
        apellido = self.entry_apellido.get().strip()
        email = self.entry_email.get().strip()

        # Validaciones
        if not nombre or not apellido or not email:
            messagebox.showwarning("Campos vacíos", "Completa todos los campos.")
            return

        if '@' not in email:
            messagebox.showwarning("Email inválido", "Ingresa un email válido.")
            return

        # Registrar en BD
        carpeta = os.path.join(DATA_DIR, nombre.lower())
        pid = self.app.db.registrar_persona(nombre, apellido, email, carpeta)

        if pid == -1:
            messagebox.showerror("Error", f"El email '{email}' ya está registrado.")
            self.label_status.configure(text="Error: email duplicado.", fg=FG_DANGER)
        else:
            messagebox.showinfo("Éxito", f"Persona '{nombre} {apellido}' registrada (ID: {pid}).")
            self.label_status.configure(text=f"Registrado: {nombre} {apellido}", fg=FG_SUCCESS)
            # Limpiar campos
            self.entry_nombre.delete(0, tk.END)
            self.entry_apellido.delete(0, tk.END)
            self.entry_email.delete(0, tk.END)
            self._actualizar_lista_personas()

    def _actualizar_lista_personas(self):
        """Actualiza la lista de personas registradas."""
        self.lista_personas.delete(0, tk.END)
        for p in self.app.db.obtener_todas_personas():
            self.lista_personas.insert(tk.END, f"{p['nombre']} {p['apellido']} ({p['email']})")


# ===================== PANTALLA DE DETECCIÓN =====================

class PantallaDeteccion(tk.Frame):
    """
    Pantalla de detección en tiempo real.
    Muestra video con overlay de emoción, nombre e información.
    """

    def __init__(self, parent, app):
        super().__init__(parent, bg=BG_DARK)
        self.app = app
        self.detectando = False
        self.modelo_actual = None
        self.nombre_modelo = tk.StringVar(value='LBPH')
        self.emojis = {}
        self._cargar_emojis()
        self._crear_interfaz()

    def _cargar_emojis(self):
        """Pre-carga las imágenes de emojis."""
        for emocion in EMOCIONES.values():
            for ext in ['.png', '.jpg', '.jpeg']:
                ruta = os.path.join(EMOJIS_DIR, f"{emocion}{ext}")
                if os.path.exists(ruta):
                    try:
                        img = Image.open(ruta).resize((120, 120), Image.LANCZOS)
                        self.emojis[emocion] = ImageTk.PhotoImage(img)
                    except Exception:
                        pass
                    break

    def _crear_interfaz(self):
        """Construye la interfaz de la pantalla de detección."""

        # === PANEL SUPERIOR: Controles ===
        panel_ctrl = tk.Frame(self, bg=BG_CARD, height=60)
        panel_ctrl.pack(fill=tk.X, pady=(0, 5))
        panel_ctrl.pack_propagate(False)

        tk.Label(panel_ctrl, text="🔍 Detección en Tiempo Real",
                 font=("Segoe UI", 14, "bold"), bg=BG_CARD, fg=FG_ACCENT
                 ).pack(side=tk.LEFT, padx=15)

        # Selector de modelo
        modelo_frame = tk.Frame(panel_ctrl, bg=BG_CARD)
        modelo_frame.pack(side=tk.LEFT, padx=20)
        tk.Label(modelo_frame, text="Modelo:", font=("Segoe UI", 10),
                 bg=BG_CARD, fg=FG_TEXT).pack(side=tk.LEFT)
        modelos_disp = list(ARCHIVOS_MODELO.keys())
        self.combo_modelo = ttk.Combobox(modelo_frame, values=modelos_disp,
                                         textvariable=self.nombre_modelo,
                                         state="readonly", width=12)
        self.combo_modelo.pack(side=tk.LEFT, padx=5)

        # Botones
        self.btn_detectar = tk.Button(
            panel_ctrl, text="▶ Iniciar Detección",
            font=("Segoe UI", 10, "bold"), bg=FG_SUCCESS, fg=BG_DARK,
            relief=tk.FLAT, cursor="hand2", command=self._toggle_deteccion
        )
        self.btn_detectar.pack(side=tk.RIGHT, padx=15, pady=10)

        # === PANEL CENTRAL: Video ===
        self.panel_video = tk.Frame(self, bg=BG_CARD)
        self.panel_video.pack(fill=tk.BOTH, expand=True, pady=(0, 5))

        self.label_video = tk.Label(self.panel_video, bg=BG_CARD,
                                    text="▶ Presiona 'Iniciar Detección' para comenzar",
                                    font=("Segoe UI", 16), fg=FG_SUBTLE)
        self.label_video.pack(expand=True)

        # === PANEL INFERIOR: Información ===
        panel_info = tk.Frame(self, bg=BG_CARD, height=120)
        panel_info.pack(fill=tk.X)
        panel_info.pack_propagate(False)

        # Columnas de información
        info_cols = tk.Frame(panel_info, bg=BG_CARD)
        info_cols.pack(fill=tk.BOTH, expand=True, padx=15, pady=10)

        # Emoción detectada
        col1 = tk.Frame(info_cols, bg=BG_CARD)
        col1.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        tk.Label(col1, text="Emoción:", font=("Segoe UI", 9),
                 bg=BG_CARD, fg=FG_SUBTLE).pack(anchor=tk.W)
        self.label_emocion = tk.Label(col1, text="—",
                                      font=("Segoe UI", 20, "bold"),
                                      bg=BG_CARD, fg=FG_ACCENT)
        self.label_emocion.pack(anchor=tk.W)

        # Emoji
        col2 = tk.Frame(info_cols, bg=BG_CARD)
        col2.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.label_emoji = tk.Label(col2, text="🎭", font=("Segoe UI", 40),
                                    bg=BG_CARD)
        self.label_emoji.pack(expand=True)

        # Confianza
        col3 = tk.Frame(info_cols, bg=BG_CARD)
        col3.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        tk.Label(col3, text="Confianza:", font=("Segoe UI", 9),
                 bg=BG_CARD, fg=FG_SUBTLE).pack(anchor=tk.W)
        self.label_confianza = tk.Label(col3, text="—",
                                        font=("Segoe UI", 18, "bold"),
                                        bg=BG_CARD, fg=FG_SUCCESS)
        self.label_confianza.pack(anchor=tk.W)

        # Modelo y FPS
        col4 = tk.Frame(info_cols, bg=BG_CARD)
        col4.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        tk.Label(col4, text="Info:", font=("Segoe UI", 9),
                 bg=BG_CARD, fg=FG_SUBTLE).pack(anchor=tk.W)
        self.label_info = tk.Label(col4, text="—",
                                   font=("Segoe UI", 10),
                                   bg=BG_CARD, fg=FG_TEXT)
        self.label_info.pack(anchor=tk.W)

    def _toggle_deteccion(self):
        """Inicia o detiene la detección."""
        if not self.detectando:
            self._iniciar_deteccion()
        else:
            self.detener_deteccion()

    def _iniciar_deteccion(self):
        """Inicia la detección en tiempo real."""
        nombre = self.nombre_modelo.get()
        ruta = ARCHIVOS_MODELO.get(nombre)

        if not ruta or not os.path.exists(ruta):
            messagebox.showerror("Error",
                                 f"Modelo '{nombre}' no encontrado.\nEjecuta 'entrenamiento.py' primero.")
            return

        if self.app.face_cascade is None:
            messagebox.showerror("Error", "Clasificador Haar Cascade no disponible.")
            return

        # Cargar modelo
        if nombre == 'EigenFace':
            self.modelo_actual = cv2.face.EigenFaceRecognizer_create()
        elif nombre == 'FisherFace':
            self.modelo_actual = cv2.face.FisherFaceRecognizer_create()
        else:
            self.modelo_actual = cv2.face.LBPHFaceRecognizer_create()
        self.modelo_actual.read(ruta)

        # Iniciar cámara
        if not self.app.iniciar_camara():
            return

        self.detectando = True
        self.btn_detectar.configure(text="⏹ Detener Detección", bg=FG_DANGER)
        self.frame_count = 0
        self.fps_inicio = time.time()
        self.fps = 0
        self._actualizar_deteccion()

    def detener_deteccion(self):
        """Detiene la detección."""
        self.detectando = False
        self.app.detener_camara()
        self.btn_detectar.configure(text="▶ Iniciar Detección", bg=FG_SUCCESS)
        self.label_video.configure(image='', text="▶ Presiona 'Iniciar Detección' para comenzar")
        self.label_emocion.configure(text="—")
        self.label_confianza.configure(text="—")
        self.label_info.configure(text="—")

    def _actualizar_deteccion(self):
        """Actualiza la detección frame por frame."""
        if not self.detectando or self.app.cap is None:
            return

        ret, frame = self.app.cap.read()
        if not ret:
            return

        frame = cv2.flip(frame, 1)
        gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # FPS
        self.frame_count += 1
        if self.frame_count % 15 == 0:
            self.fps = 15 / (time.time() - self.fps_inicio + 0.001)
            self.fps_inicio = time.time()

        # Detectar rostros
        rostros = self.app.face_cascade.detectMultiScale(
            gris, scaleFactor=1.3, minNeighbors=5, minSize=(80, 80)
        )

        nombre = self.nombre_modelo.get()
        umbral = UMBRALES.get(nombre, 120.0)

        for (x, y, w, h) in rostros:
            rostro = gris[y:y + h, x:x + w]
            rostro_redim = cv2.resize(rostro, FACE_SIZE)

            etiqueta, confianza = self.modelo_actual.predict(rostro_redim)

            if confianza > umbral:
                emocion = "No identificado"
                conf_pct = 0
                color = (100, 100, 100)
            else:
                emocion = EMOCIONES.get(etiqueta, 'neutral')
                conf_pct = max(0, 100 - (confianza / umbral * 100))
                hex_c = COLORES_EMOCION_HEX.get(emocion, '#FFFFFF')
                # Convertir hex a BGR
                r, g, b = int(hex_c[1:3], 16), int(hex_c[3:5], 16), int(hex_c[5:7], 16)
                color = (b, g, r)

                # Registrar en BD cada 30 frames
                if self.frame_count % 30 == 0:
                    self.app.db.guardar_deteccion(
                        None, "Usuario", emocion, confianza, nombre
                    )

            # Dibujar en el frame
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.rectangle(frame, (x, y - 30), (x + w, y), color, -1)
            cv2.putText(frame, f"{emocion.capitalize()} {conf_pct:.0f}%",
                        (x + 5, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

            # Actualizar panel de información
            self.label_emocion.configure(text=emocion.capitalize())
            self.label_confianza.configure(text=f"{conf_pct:.1f}%")
            self.label_info.configure(
                text=f"Modelo: {nombre}\nFPS: {self.fps:.0f}\n{datetime.now().strftime('%H:%M:%S')}"
            )

            # Mostrar emoji
            if emocion in self.emojis:
                self.label_emoji.configure(image=self.emojis[emocion], text="")
            else:
                emoji_text = {'felicidad': '😊', 'enojo': '😡', 'asombro': '😲',
                              'tristeza': '😢', 'neutral': '😐', 'miedo': '😨',
                              'disgusto': '🤢'}.get(emocion, '🎭')
                self.label_emoji.configure(text=emoji_text, image='')

        # Convertir para Tkinter
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        panel_w = self.panel_video.winfo_width()
        panel_h = self.panel_video.winfo_height()
        if panel_w > 1 and panel_h > 1:
            ratio = min(panel_w / img.width, panel_h / img.height)
            new_size = (max(1, int(img.width * ratio)), max(1, int(img.height * ratio)))
            img = img.resize(new_size, Image.LANCZOS)

        self._photo = ImageTk.PhotoImage(img)
        self.label_video.configure(image=self._photo, text="")

        if self.detectando:
            self.after(30, self._actualizar_deteccion)


# ===================== PANTALLA DE REPORTES =====================

class PantallaReportes(tk.Frame):
    """
    Pantalla de reportes y estadísticas.
    Muestra gráficos, historial y opciones de exportación.
    """

    def __init__(self, parent, app):
        super().__init__(parent, bg=BG_DARK)
        self.app = app
        self._crear_interfaz()

    def _crear_interfaz(self):
        """Construye la interfaz de la pantalla de reportes."""

        # === PANEL SUPERIOR: Título y controles ===
        panel_titulo = tk.Frame(self, bg=BG_CARD, height=60)
        panel_titulo.pack(fill=tk.X, pady=(0, 5))
        panel_titulo.pack_propagate(False)

        tk.Label(panel_titulo, text="📊 Reportes y Estadísticas",
                 font=("Segoe UI", 14, "bold"), bg=BG_CARD, fg=FG_ACCENT
                 ).pack(side=tk.LEFT, padx=15, pady=10)

        # Botones de control
        tk.Button(
            panel_titulo, text="🔄 Actualizar",
            font=("Segoe UI", 10), bg=BG_INPUT, fg=FG_TEXT,
            relief=tk.FLAT, cursor="hand2",
            command=self.actualizar_datos
        ).pack(side=tk.RIGHT, padx=5, pady=10)

        tk.Button(
            panel_titulo, text="📥 Exportar CSV",
            font=("Segoe UI", 10), bg=FG_ACCENT, fg=BG_DARK,
            relief=tk.FLAT, cursor="hand2",
            command=self._exportar_csv
        ).pack(side=tk.RIGHT, padx=5, pady=10)

        # === ZONA CENTRAL: dividida en gráfico + tabla ===
        central = tk.Frame(self, bg=BG_DARK)
        central.pack(fill=tk.BOTH, expand=True)

        # Panel izquierdo: Gráfico
        self.panel_grafico = tk.Frame(central, bg=BG_CARD, width=450)
        self.panel_grafico.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))

        self.label_grafico = tk.Label(self.panel_grafico, bg=BG_CARD,
                                      text="Sin datos para graficar",
                                      font=("Segoe UI", 12), fg=FG_SUBTLE)
        self.label_grafico.pack(expand=True)

        # Panel derecho: Tabla de historial
        panel_tabla = tk.Frame(central, bg=BG_CARD, width=400)
        panel_tabla.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        tk.Label(panel_tabla, text="Historial de Detecciones",
                 font=("Segoe UI", 12, "bold"), bg=BG_CARD, fg=FG_TEXT
                 ).pack(pady=(10, 5))

        # Tabla con scrollbar
        tabla_frame = tk.Frame(panel_tabla, bg=BG_CARD)
        tabla_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))

        cols = ("Fecha", "Emoción", "Confianza", "Modelo")
        self.tabla = ttk.Treeview(tabla_frame, columns=cols, show='headings', height=15)

        for col in cols:
            self.tabla.heading(col, text=col)
            self.tabla.column(col, width=100, anchor=tk.CENTER)

        scroll = ttk.Scrollbar(tabla_frame, orient=tk.VERTICAL, command=self.tabla.yview)
        self.tabla.configure(yscrollcommand=scroll.set)

        self.tabla.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)

        # === PANEL INFERIOR: Estadísticas rápidas ===
        panel_stats = tk.Frame(self, bg=BG_CARD, height=80)
        panel_stats.pack(fill=tk.X, pady=(5, 0))
        panel_stats.pack_propagate(False)

        self.labels_stats = {}
        for i, (titulo, key) in enumerate([
            ("Total Detecciones", "total"),
            ("Emoción Frecuente", "frecuente"),
            ("Personas Registradas", "personas"),
            ("Última Detección", "ultima")
        ]):
            col = tk.Frame(panel_stats, bg=BG_CARD)
            col.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)
            tk.Label(col, text=titulo, font=("Segoe UI", 9),
                     bg=BG_CARD, fg=FG_SUBTLE).pack(anchor=tk.W)
            self.labels_stats[key] = tk.Label(col, text="—",
                                               font=("Segoe UI", 14, "bold"),
                                               bg=BG_CARD, fg=FG_TEXT)
            self.labels_stats[key].pack(anchor=tk.W)

    def actualizar_datos(self):
        """Actualiza todos los datos de la pantalla de reportes."""
        self._actualizar_grafico()
        self._actualizar_tabla()
        self._actualizar_estadisticas()

    def _actualizar_grafico(self):
        """Genera y muestra el gráfico de emociones."""
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

            stats = self.app.db.obtener_estadisticas_emociones()
            if not stats:
                self.label_grafico.configure(text="Sin datos para graficar")
                return

            # Limpiar el panel
            for widget in self.panel_grafico.winfo_children():
                widget.destroy()

            # Crear gráfico de barras
            fig, ax = plt.subplots(figsize=(5, 4))
            fig.patch.set_facecolor('#16213e')
            ax.set_facecolor('#1a1a2e')

            emociones = list(stats.keys())
            conteos = list(stats.values())
            colores = [COLORES_EMOCION_HEX.get(e, '#FFFFFF') for e in emociones]

            bars = ax.bar(emociones, conteos, color=colores, edgecolor='white', linewidth=0.5)
            ax.set_title("Distribución de Emociones Detectadas",
                         color='white', fontsize=12, fontweight='bold')
            ax.set_ylabel("Cantidad", color='white')
            ax.tick_params(colors='white')
            ax.spines['bottom'].set_color('white')
            ax.spines['left'].set_color('white')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

            # Valores sobre las barras
            for bar, val in zip(bars, conteos):
                ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.5,
                        str(val), ha='center', va='bottom', color='white', fontsize=9)

            plt.xticks(rotation=30, ha='right')
            plt.tight_layout()

            # Insertar en Tkinter
            canvas = FigureCanvasTkAgg(fig, self.panel_grafico)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            plt.close(fig)

        except Exception as e:
            self.label_grafico.configure(text=f"Error al generar gráfico:\n{e}")

    def _actualizar_tabla(self):
        """Actualiza la tabla del historial de detecciones."""
        # Limpiar tabla
        for item in self.tabla.get_children():
            self.tabla.delete(item)

        # Obtener historial
        historial = self.app.db.obtener_historial(limite=200)
        for det in historial:
            # Formatear fecha
            try:
                ts = datetime.fromisoformat(det['timestamp'])
                fecha = ts.strftime("%d/%m %H:%M:%S")
            except (ValueError, KeyError):
                fecha = det.get('timestamp', '—')

            self.tabla.insert('', tk.END, values=(
                fecha,
                det.get('emocion', '—'),
                f"{det.get('confianza', 0):.1f}",
                det.get('modelo_usado', '—')
            ))

    def _actualizar_estadisticas(self):
        """Actualiza las estadísticas rápidas."""
        historial = self.app.db.obtener_historial(limite=10000)
        personas = self.app.db.obtener_todas_personas()
        stats = self.app.db.obtener_estadisticas_emociones()

        self.labels_stats["total"].configure(text=str(len(historial)))
        self.labels_stats["personas"].configure(text=str(len(personas)))

        if stats:
            frecuente = max(stats, key=stats.get)
            self.labels_stats["frecuente"].configure(text=frecuente.capitalize())
        else:
            self.labels_stats["frecuente"].configure(text="—")

        if historial:
            try:
                ultima = datetime.fromisoformat(historial[0]['timestamp'])
                self.labels_stats["ultima"].configure(text=ultima.strftime("%H:%M:%S"))
            except (ValueError, KeyError):
                self.labels_stats["ultima"].configure(text="—")
        else:
            self.labels_stats["ultima"].configure(text="—")

    def _exportar_csv(self):
        """Exporta los datos a un archivo CSV."""
        ruta = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV", "*.csv")],
            initialfile="reporte_detecciones.csv"
        )
        if ruta:
            resultado = self.app.db.exportar_a_csv(ruta)
            if resultado:
                messagebox.showinfo("Éxito", f"Reporte exportado a:\n{resultado}")
            else:
                messagebox.showwarning("Aviso", "No hay datos para exportar.")


# ===================== PUNTO DE ENTRADA =====================
if __name__ == "__main__":
    print("=" * 60)
    print("  SISTEMA DE RECONOCIMIENTO FACIAL Y EMOCIONES")
    print("  Interfaz Gráfica - Tkinter")
    print("=" * 60)

    app = AplicacionPrincipal()
    app.mainloop()
