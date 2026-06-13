"""
=============================================================================
 Archivo: main.py
 Sistema Inteligente de Semáforos y Vigilancia para La Paz
 ─────────────────────────────────────────────────────────
 Punto de entrada principal del sistema. Este script:

   1. Configura la aplicación Flask y la base de datos SQLAlchemy.
   2. Carga los tres modelos de IA en memoria global (MLP, LSTM, CNN)
      para evitar recargas costosas en cada request.
   3. Registra los Blueprints de la API.
   4. Levanta el servidor Flask en el puerto 5000.

 EJECUCIÓN
 ─────────
   $ python main.py
   → Servidor disponible en http://localhost:5000

 NOTA SOBRE CARGA DE MODELOS
 ────────────────────────────
 Los modelos se cargan UNA VEZ al inicio y se mantienen en memoria como
 singletons. Esto es crítico para rendimiento, ya que:
   - La carga de ResNet50 (~100MB) tarda ~3-5 segundos.
   - La compilación de grafos de TensorFlow tiene un costo fijo.
   - Mantener los modelos en memoria permite inferencia en ~1ms (MLP)
     y ~200ms (CNN) por request.

 INTEGRACIÓN FUTURA CON MQTT (paho-mqtt)
 ────────────────────────────────────────
 Para conectar sensores físicos en una iteración futura, se añadirá un
 cliente MQTT que se ejecutará en un hilo daemon separado:

   import paho.mqtt.client as mqtt

   def on_message(client, userdata, msg):
       '''Callback que se ejecuta al recibir datos de un sensor.'''
       datos = json.loads(msg.payload)
       interseccion_id = msg.topic.split('/')[2]
       # Llamar al modelo MLP directamente (sin HTTP overhead)
       resultado = modelo_mlp.predecir_fase(np.array(datos['valores']))
       # Publicar la fase decidida
       client.publish(
           f"lapaz/interseccion/{interseccion_id}/fase",
           json.dumps(resultado)
       )

   mqtt_client = mqtt.Client(client_id="controlador_central")
   mqtt_client.on_message = on_message
   mqtt_client.connect("broker.lapaz.gob.bo", 1883, 60)
   # Suscribirse a todos los sensores de todas las intersecciones
   mqtt_client.subscribe("lapaz/interseccion/+/sensores")
   # Iniciar loop de escucha en hilo daemon
   mqtt_client.loop_start()  # No bloquea; corre en background thread

 Esto permite latencias de <50ms desde la lectura del sensor hasta
 el cambio de fase del semáforo, comparado con ~100-200ms vía HTTP.
=============================================================================
"""

from __future__ import annotations

import logging
import os
import sys

import numpy as np
from flask import Flask

# ── Configurar logging antes de cualquier otra importación ───────────────────
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s — %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("main")


def crear_app(config: dict = None) -> Flask:
    """
    Factory function para crear y configurar la aplicación Flask.
    """
    app = Flask(
        __name__,
        static_folder='static',
        template_folder='templates'
    )

    # ── Configuración de la aplicación ───────────────────────────────────
    app.config.update({
        # SQLAlchemy: usar SQLite en desarrollo, PostgreSQL en producción.
        # En producción, configurar:
        #   DATABASE_URL=postgresql://usuario:clave@host:5432/semaforos_lapaz
        "SQLALCHEMY_DATABASE_URI": os.environ.get(
            "DATABASE_URL",
            "sqlite:///semaforos_lapaz.db",
        ),
        "SQLALCHEMY_TRACK_MODIFICATIONS": False,
        "SQLALCHEMY_ENGINE_OPTIONS": {
            "pool_pre_ping": True,      # Verificar conexiones antes de usar
            "pool_recycle": 300,         # Reciclar conexiones cada 5 minutos
        },

        # Configuración general de Flask
        "JSON_SORT_KEYS": False,
        "MAX_CONTENT_LENGTH": 16 * 1024 * 1024,  # 16MB max (para imágenes)
    })

    # Aplicar configuración personalizada si se proporciona
    if config:
        app.config.update(config)

    # ── Inicializar extensiones ──────────────────────────────────────────
    from api.semaforo_api import db, semaforo_bp
    db.init_app(app)

    # ── Registrar Blueprints ─────────────────────────────────────────────
    app.register_blueprint(semaforo_bp)

    # ── Crear tablas de la base de datos e inicializar IA ────────────────
    with app.app_context():
        db.create_all()
        logger.info("Base de datos inicializada correctamente.")
        
        # En producción (Gunicorn), crear_app() es el punto de entrada real.
        # Cargamos los modelos aquí para que cada worker tenga su instancia.
        cargar_modelos(app)

    # ── Endpoint raíz (Dashboard Frontend) ───────────────────────────────
    from flask import render_template

    @app.route("/")
    def raiz():
        return render_template('index.html')

    return app


def cargar_modelos(app: Flask) -> None:
    """
    Carga los tres modelos de IA en memoria global.

    Los modelos se entrenan con datos sintéticos de demostración.
    En producción, se cargarían desde disco con los archivos .keras
    generados durante el entrenamiento offline.

    IMPORTANTE: Esta función se ejecuta UNA SOLA VEZ al inicio del
    servidor. Los modelos permanecen en memoria y se comparten entre
    todas las requests de Flask.

    Parameters
    ----------
    app : Flask
        Aplicación Flask (para contexto de logging).
    """
    from api.semaforo_api import registrar_modelos

    logger.info("=" * 60)
    logger.info("  Cargando modelos de IA en memoria...")
    logger.info("=" * 60)

    # ──────────────────────────────────────────────────────────────────────
    # MODELO 1: MLP para decisión de fase semafórica
    # ──────────────────────────────────────────────────────────────────────
    logger.info("[1/3] Inicializando MLP Semáforo...")
    try:
        from models.mlp_semaforo import SemaforoMLP, generar_datos_sinteticos
        from sklearn.model_selection import train_test_split

        modelo_mlp = SemaforoMLP()

        # Verificar si existe un modelo pre-entrenado en disco
        ruta_mlp = os.path.join(
            os.path.dirname(__file__), "saved_models", "mlp_semaforo.keras"
        )

        if os.path.exists(ruta_mlp):
            modelo_mlp.cargar(ruta_mlp)
            logger.info("  ✓ MLP cargado desde disco: %s", ruta_mlp)
        else:
            # Entrenar con datos sintéticos de demostración
            logger.info("  → No se encontró modelo pre-entrenado.")
            logger.info("  → Entrenando con datos sintéticos de demostración...")

            X, y = generar_datos_sinteticos(n_muestras=3000)
            X_tr, X_val, y_tr, y_val = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            modelo_mlp.entrenar(
                X_tr, y_tr, X_val, y_val, epochs=15, batch_size=64
            )
            logger.info("  ✓ MLP entrenado exitosamente con datos sintéticos.")

    except Exception as e:
        logger.error("  ✗ Error cargando MLP: %s", e, exc_info=True)
        modelo_mlp = None

    # ──────────────────────────────────────────────────────────────────────
    # MODELO 2: LSTM para predicción de congestión
    # ──────────────────────────────────────────────────────────────────────
    logger.info("[2/3] Inicializando LSTM Predicción...")
    try:
        from models.lstm_prediccion import (
            PrediccionLSTM,
            generar_serie_temporal_sintetica,
        )

        modelo_lstm = PrediccionLSTM()

        ruta_lstm = os.path.join(
            os.path.dirname(__file__), "saved_models", "lstm_prediccion.keras"
        )

        if os.path.exists(ruta_lstm):
            modelo_lstm.cargar(ruta_lstm)
            logger.info("  ✓ LSTM cargada desde disco: %s", ruta_lstm)
        else:
            logger.info("  → Entrenando LSTM con datos sintéticos...")

            datos, objetivos = generar_serie_temporal_sintetica(n_pasos=1500)
            X, y = PrediccionLSTM.crear_ventanas(datos, objetivos, ventana=12)

            split = int(len(X) * 0.8)
            X_tr, X_val = X[:split], X[split:]
            y_tr, y_val = y[:split], y[split:]

            modelo_lstm.entrenar(
                X_tr, y_tr, X_val, y_val, epochs=10, batch_size=64
            )
            logger.info("  ✓ LSTM entrenada exitosamente con datos sintéticos.")

    except Exception as e:
        logger.error("  ✗ Error cargando LSTM: %s", e, exc_info=True)
        modelo_lstm = None

    # ──────────────────────────────────────────────────────────────────────
    # MODELO 3: CNN para detección de incidentes
    # ──────────────────────────────────────────────────────────────────────
    logger.info("[3/3] Inicializando CNN Detección...")
    try:
        from models.cnn_deteccion import DeteccionCNN

        modelo_cnn = DeteccionCNN()

        ruta_cnn = os.path.join(
            os.path.dirname(__file__), "saved_models", "cnn_deteccion.keras"
        )

        if os.path.exists(ruta_cnn):
            modelo_cnn.cargar(ruta_cnn)
            logger.info("  ✓ CNN cargada desde disco: %s", ruta_cnn)
        else:
            # La CNN con ResNet50 no se entrena con datos sintéticos al inicio
            # porque requiere imágenes reales y el entrenamiento es costoso.
            # Se deja con los pesos base de ImageNet (transfer learning).
            logger.info(
                "  → CNN inicializada con pesos de ImageNet (sin fine-tuning)."
            )
            logger.info(
                "  → Para detección precisa, entrenar con imágenes reales "
                "de intersecciones de La Paz."
            )

    except Exception as e:
        logger.error("  ✗ Error cargando CNN: %s", e, exc_info=True)
        modelo_cnn = None

    # ── Registrar modelos en la API ──────────────────────────────────────
    registrar_modelos(
        mlp=modelo_mlp,
        lstm=modelo_lstm,
        cnn=modelo_cnn,
    )

    logger.info("=" * 60)
    logger.info("  Todos los modelos cargados en memoria.")
    logger.info("=" * 60)


# =============================================================================
# PUNTO DE ENTRADA
# =============================================================================

if __name__ == "__main__":
    # ── Crear la aplicación Flask e inicializar IA ───────────────────────
    app = crear_app()

    # ── Configuración del servidor ───────────────────────────────────────
    HOST = os.environ.get("FLASK_HOST", "0.0.0.0")
    PORT = int(os.environ.get("FLASK_PORT", "5000"))
    DEBUG = os.environ.get("FLASK_DEBUG", "false").lower() == "true"

    logger.info("")
    logger.info("╔══════════════════════════════════════════════════════════════════════════════╗")
    logger.info("║  Sistema Inteligente De Semáforos y Vigilancia Para La Seguridad Vial       ║")
    logger.info("║  En La Paz                                                                   ║")
    logger.info("║  Desarrollado por: Johan Misme Flores y Eyder Quenta Quispe                  ║")
    logger.info("║                                                                              ║")
    logger.info("║  Servidor Flask iniciando en:                                                ║")
    logger.info("║    → http://%s:%-61s ║", HOST, str(PORT))
    logger.info("║                                                                              ║")
    logger.info("║  Endpoints disponibles:                                                      ║")
    logger.info("║    GET  /api/estado                                                          ║")
    logger.info("║    POST /api/fase_optima                                                     ║")
    logger.info("║    POST /api/predecir_congestion                                             ║")
    logger.info("║    POST /api/detectar_incidente                                              ║")
    logger.info("║    GET  /api/historial                                                       ║")
    logger.info("╚══════════════════════════════════════════════════════════════════════════════╝")
    logger.info("")

    # ── Levantar el servidor ─────────────────────────────────────────────
    # En desarrollo: Flask dev server con auto-reload desactivado
    #   (para evitar recargar los modelos de IA en cada cambio).
    # En producción: usar Gunicorn:
    #   $ gunicorn -w 4 -b 0.0.0.0:5000 "main:crear_app()"
    #   NOTA: Con Gunicorn, los modelos se cargan en cada worker.
    #   Para compartir modelos entre workers, usar TensorFlow Serving
    #   o Redis como caché de modelos.
    app.run(
        host=HOST,
        port=PORT,
        debug=DEBUG,
        use_reloader=False,  # IMPORTANTE: evitar recargar modelos
        threaded=True,       # Permitir requests concurrentes
    )
