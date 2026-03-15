"""
database.py — Módulo de Base de Datos SQLite
=============================================
Gestiona el almacenamiento de personas registradas y el historial
de detecciones emocionales. Utiliza SQLite para persistencia local.
"""

import sqlite3
import os
from datetime import datetime


class Database:
    """
    Clase principal para gestionar la base de datos del sistema
    de reconocimiento facial y análisis de emociones.
    """

    def __init__(self, db_path="reconocimiento.db"):
        """
        Inicializa la conexión a la base de datos y crea las tablas
        si no existen.
        
        Args:
            db_path (str): Ruta al archivo de la base de datos SQLite.
        """
        self.db_path = db_path
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row  # Permite acceso por nombre de columna
        self.cursor = self.conn.cursor()
        self._crear_tablas()

    def _crear_tablas(self):
        """Crea las tablas necesarias si no existen."""
        # Tabla de personas registradas
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS personas (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                nombre TEXT NOT NULL,
                apellido TEXT NOT NULL,
                email TEXT UNIQUE NOT NULL,
                carpeta_datos TEXT NOT NULL,
                fecha_registro TEXT NOT NULL
            )
        """)

        # Tabla del historial de detecciones emocionales
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS detecciones (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                persona_id INTEGER,
                nombre_persona TEXT,
                emocion TEXT NOT NULL,
                confianza REAL NOT NULL,
                modelo_usado TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                FOREIGN KEY (persona_id) REFERENCES personas(id)
            )
        """)
        self.conn.commit()

    # ===================== CRUD PERSONAS =====================

    def registrar_persona(self, nombre, apellido, email, carpeta_datos):
        """
        Registra una nueva persona en la base de datos.
        
        Args:
            nombre (str): Nombre de la persona.
            apellido (str): Apellido de la persona.
            email (str): Email único de la persona.
            carpeta_datos (str): Ruta a la carpeta con sus imágenes faciales.
            
        Returns:
            int: ID de la persona registrada, o -1 si el email ya existe.
        """
        # Validar que no existan duplicados por email
        if self.buscar_persona_por_email(email):
            return -1

        try:
            self.cursor.execute("""
                INSERT INTO personas (nombre, apellido, email, carpeta_datos, fecha_registro)
                VALUES (?, ?, ?, ?, ?)
            """, (nombre, apellido, email, carpeta_datos, datetime.now().isoformat()))
            self.conn.commit()
            return self.cursor.lastrowid
        except sqlite3.IntegrityError:
            return -1

    def buscar_persona_por_email(self, email):
        """
        Busca una persona por su email.
        
        Args:
            email (str): Email a buscar.
            
        Returns:
            dict | None: Datos de la persona o None si no existe.
        """
        self.cursor.execute("SELECT * FROM personas WHERE email = ?", (email,))
        row = self.cursor.fetchone()
        return dict(row) if row else None

    def buscar_persona_por_id(self, persona_id):
        """
        Busca una persona por su ID.
        
        Args:
            persona_id (int): ID de la persona.
            
        Returns:
            dict | None: Datos de la persona o None si no existe.
        """
        self.cursor.execute("SELECT * FROM personas WHERE id = ?", (persona_id,))
        row = self.cursor.fetchone()
        return dict(row) if row else None

    def obtener_todas_personas(self):
        """
        Obtiene todas las personas registradas.
        
        Returns:
            list[dict]: Lista con los datos de todas las personas.
        """
        self.cursor.execute("SELECT * FROM personas ORDER BY fecha_registro DESC")
        return [dict(row) for row in self.cursor.fetchall()]

    def eliminar_persona(self, persona_id):
        """
        Elimina una persona y su historial de detecciones.
        
        Args:
            persona_id (int): ID de la persona a eliminar.
            
        Returns:
            bool: True si se eliminó correctamente.
        """
        self.cursor.execute("DELETE FROM detecciones WHERE persona_id = ?", (persona_id,))
        self.cursor.execute("DELETE FROM personas WHERE id = ?", (persona_id,))
        self.conn.commit()
        return self.cursor.rowcount > 0

    # ===================== CRUD DETECCIONES =====================

    def guardar_deteccion(self, persona_id, nombre_persona, emocion, confianza, modelo_usado):
        """
        Guarda un registro de detección emocional.
        
        Args:
            persona_id (int | None): ID de la persona (None si no identificada).
            nombre_persona (str): Nombre mostrado.
            emocion (str): Emoción detectada.
            confianza (float): Nivel de confianza de la predicción.
            modelo_usado (str): Nombre del modelo utilizado.
            
        Returns:
            int: ID del registro de detección.
        """
        self.cursor.execute("""
            INSERT INTO detecciones (persona_id, nombre_persona, emocion, confianza, modelo_usado, timestamp)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (persona_id, nombre_persona, emocion, confianza, modelo_usado,
              datetime.now().isoformat()))
        self.conn.commit()
        return self.cursor.lastrowid

    def obtener_historial(self, limite=100):
        """
        Obtiene el historial de detecciones más reciente.
        
        Args:
            limite (int): Número máximo de registros a devolver.
            
        Returns:
            list[dict]: Lista de detecciones ordenadas por fecha descendente.
        """
        self.cursor.execute("""
            SELECT d.*, p.nombre, p.apellido, p.email
            FROM detecciones d
            LEFT JOIN personas p ON d.persona_id = p.id
            ORDER BY d.timestamp DESC
            LIMIT ?
        """, (limite,))
        return [dict(row) for row in self.cursor.fetchall()]

    def obtener_detecciones_por_persona(self, persona_id):
        """
        Obtiene todas las detecciones de una persona específica.
        
        Args:
            persona_id (int): ID de la persona.
            
        Returns:
            list[dict]: Lista de detecciones de la persona.
        """
        self.cursor.execute("""
            SELECT * FROM detecciones
            WHERE persona_id = ?
            ORDER BY timestamp DESC
        """, (persona_id,))
        return [dict(row) for row in self.cursor.fetchall()]

    def obtener_estadisticas_emociones(self, persona_id=None):
        """
        Obtiene estadísticas de emociones (conteo por tipo).
        
        Args:
            persona_id (int | None): Si se especifica, filtra por persona.
            
        Returns:
            dict: Diccionario con emoción -> conteo.
        """
        if persona_id:
            self.cursor.execute("""
                SELECT emocion, COUNT(*) as conteo
                FROM detecciones
                WHERE persona_id = ?
                GROUP BY emocion
                ORDER BY conteo DESC
            """, (persona_id,))
        else:
            self.cursor.execute("""
                SELECT emocion, COUNT(*) as conteo
                FROM detecciones
                GROUP BY emocion
                ORDER BY conteo DESC
            """)
        return {row['emocion']: row['conteo'] for row in self.cursor.fetchall()}

    def exportar_a_csv(self, ruta_archivo="reporte_detecciones.csv"):
        """
        Exporta el historial de detecciones a un archivo CSV.
        
        Args:
            ruta_archivo (str): Ruta del archivo CSV de salida.
            
        Returns:
            str: Ruta del archivo generado.
        """
        import csv
        registros = self.obtener_historial(limite=10000)
        
        if not registros:
            return None

        with open(ruta_archivo, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=registros[0].keys())
            writer.writeheader()
            writer.writerows(registros)
        
        return ruta_archivo

    def cerrar(self):
        """Cierra la conexión a la base de datos."""
        if self.conn:
            self.conn.close()

    def __del__(self):
        """Destructor: cierra la conexión al eliminar la instancia."""
        self.cerrar()


# ===================== PRUEBA RÁPIDA =====================
if __name__ == "__main__":
    db = Database()
    print("[OK] Base de datos inicializada correctamente.")
    print(f"[INFO] Tablas creadas en: {os.path.abspath(db.db_path)}")
    
    # Prueba de registro
    pid = db.registrar_persona("Test", "Usuario", "test@test.com", "data/test")
    if pid > 0:
        print(f"[OK] Persona registrada con ID: {pid}")
        # Limpiar prueba
        db.eliminar_persona(pid)
        print("[OK] Persona de prueba eliminada.")
    else:
        print("[INFO] Email ya registrado (duplicado detectado correctamente).")
    
    db.cerrar()
    print("[OK] Todas las pruebas pasaron.")
