"""
=============================================================================
 Módulo: semaforo_api.py
 Sistema Inteligente de Semáforos y Vigilancia para La Paz
 ─────────────────────────────────────────────────────────
 API REST con Flask Blueprints que expone los modelos de IA como servicios
 web para el control de semáforos en intersecciones de La Paz.

 ENDPOINTS
 ─────────
   POST /api/fase_optima
     → Recibe datos de sensores, ejecuta el MLP y devuelve la fase
       óptima del semáforo con nivel de confianza y tiempo de verde.

   POST /api/predecir_congestion
     → Recibe una ventana temporal, ejecuta la LSTM y devuelve
       predicciones de congestión a 15, 20 y 30 minutos.

   POST /api/detectar_incidente
     → Recibe una imagen codificada en base64, ejecuta la CNN (en un
       hilo separado via ThreadPoolExecutor) y devuelve si hay incidente.

   GET /api/estado
     → Endpoint de health-check y estado del sistema.

   GET /api/historial
     → Devuelve los últimos registros de decisiones de la IA.

 DISEÑO DE CONCURRENCIA (Sistemas Operativos)
 ─────────────────────────────────────────────
 La inferencia de la CNN es computacionalmente costosa (~200ms por imagen
 con ResNet50 en CPU). Para evitar que bloquee el hilo principal de Flask
 y degrade la latencia de otros endpoints:

   • Se utiliza `concurrent.futures.ThreadPoolExecutor` con un pool de
     workers dedicados para inferencia de modelos pesados.
   • Las predicciones del MLP (~1ms) se ejecutan en el hilo principal.
   • Las predicciones de la CNN se delegan al ThreadPoolExecutor.
   • Flask en producción debe ejecutarse con Gunicorn (workers=N) para
     paralelismo real a nivel de procesos.

 NOTA: TensorFlow libera el GIL durante operaciones de GPU/CPU intensivas,
 por lo que el ThreadPoolExecutor proporciona paralelismo real en este caso.

 PERSISTENCIA (SQLAlchemy)
 ─────────────────────────
 Cada llamada al endpoint /api/fase_optima registra automáticamente:
   - Timestamp de la decisión
   - ID de la intersección
   - Fase decidida por la IA
   - Nivel de congestión estimado
   - Confianza de la predicción
 en una base de datos SQLite (desarrollo) o PostgreSQL (producción).

 INTEGRACIÓN FUTURA CON MQTT
 ────────────────────────────
 En una iteración futura, esta API se complementará con un cliente MQTT
 (paho-mqtt) que:
   1. Se suscriba a "lapaz/interseccion/+/sensores" para recibir datos
      de sensores en tiempo real.
   2. Llame internamente al endpoint /api/fase_optima.
   3. Publique la fase decidida en "lapaz/interseccion/{id}/fase".
 Esto eliminará la necesidad de polling HTTP y permitirá latencias <100ms.
=============================================================================
"""

from __future__ import annotations

import base64
import logging
import os
from concurrent.futures import ThreadPoolExecutor, Future
from datetime import datetime
from typing import Any, Dict, Optional

import numpy as np
from flask import Blueprint, Flask, jsonify, request
from flask_sqlalchemy import SQLAlchemy


# ── Configuración del Logger ─────────────────────────────────────────────────
logger = logging.getLogger("semaforo_api")
logger.setLevel(logging.INFO)

if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter(
            "[%(asctime)s] %(levelname)s — %(name)s — %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    logger.addHandler(handler)


# ── Base de datos SQLAlchemy ─────────────────────────────────────────────────
# La instancia de SQLAlchemy se crea aquí y se inicializa con la app en main.py
db = SQLAlchemy()


class RegistroDecision(db.Model):
    """
    Modelo SQLAlchemy para persistir cada decisión de la IA.

    Tabla: registro_decisiones
    Columnas:
      - id              : Clave primaria autoincremental
      - timestamp       : Momento de la decisión (UTC)
      - interseccion_id : Identificador de la intersección
      - fase_decidida   : Fase del semáforo seleccionada (0-3)
      - nombre_fase     : Nombre descriptivo de la fase
      - nivel_congestion: Nivel de congestión estimado
      - confianza       : Probabilidad de la predicción
      - tiempo_verde    : Tiempo de verde asignado (segundos)
    """
    __tablename__ = "registro_decisiones"

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    timestamp = db.Column(
        db.DateTime,
        default=datetime.utcnow,
        nullable=False,
        index=True,
    )
    interseccion_id = db.Column(db.String(50), nullable=False, index=True)
    fase_decidida = db.Column(db.Integer, nullable=False)
    nombre_fase = db.Column(db.String(100), nullable=False)
    nivel_congestion = db.Column(db.String(20), nullable=True)
    confianza = db.Column(db.Float, nullable=False)
    tiempo_verde = db.Column(db.Integer, nullable=False)

    def to_dict(self) -> Dict[str, Any]:
        """Serializa el registro a diccionario para respuestas JSON."""
        return {
            "id": self.id,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "interseccion_id": self.interseccion_id,
            "fase_decidida": self.fase_decidida,
            "nombre_fase": self.nombre_fase,
            "nivel_congestion": self.nivel_congestion,
            "confianza": self.confianza,
            "tiempo_verde": self.tiempo_verde,
        }


class RegistroIncidente(db.Model):
    """
    Modelo SQLAlchemy para registrar detecciones de incidentes.

    Tabla: registro_incidentes
    """
    __tablename__ = "registro_incidentes"

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    interseccion_id = db.Column(db.String(50), nullable=False, index=True)
    incidente_detectado = db.Column(db.Boolean, nullable=False)
    probabilidad = db.Column(db.Float, nullable=False)
    requiere_atencion = db.Column(db.Boolean, nullable=False)


# ── ThreadPoolExecutor para inferencia no-bloqueante ─────────────────────────
# Se crea un pool global de hilos para las operaciones de inferencia pesadas.
# El tamaño del pool se configura vía variable de entorno o por defecto a 4.
POOL_SIZE = int(os.environ.get("INFERENCE_POOL_SIZE", "4"))
executor = ThreadPoolExecutor(
    max_workers=POOL_SIZE,
    thread_name_prefix="inferencia",
)

# ── Referencia global a los modelos (se inyectan desde main.py) ──────────────
# Estos se inicializan como None y se cargan al arrancar la aplicación.
_modelo_mlp = None
_modelo_lstm = None
_modelo_cnn = None


def registrar_modelos(mlp=None, lstm=None, cnn=None) -> None:
    """
    Registra las instancias de modelos cargados en memoria para uso
    por los endpoints de la API. Se llama desde main.py al inicio.

    Parameters
    ----------
    mlp : SemaforoMLP, optional
    lstm : PrediccionLSTM, optional
    cnn : DeteccionCNN, optional
    """
    global _modelo_mlp, _modelo_lstm, _modelo_cnn
    _modelo_mlp = mlp
    _modelo_lstm = lstm
    _modelo_cnn = cnn
    logger.info(
        "Modelos registrados: MLP=%s, LSTM=%s, CNN=%s",
        "✓" if mlp else "✗",
        "✓" if lstm else "✗",
        "✓" if cnn else "✗",
    )


# =============================================================================
# BLUEPRINT: API de Semáforos
# =============================================================================
semaforo_bp = Blueprint("semaforo", __name__, url_prefix="/api")


# ── Endpoint: Estado del sistema ─────────────────────────────────────────────

@semaforo_bp.route("/estado", methods=["GET"])
def estado_sistema():
    """
    GET /api/estado

    Retorna el estado de salud del sistema y la disponibilidad de modelos.
    Útil como health-check para balanceadores de carga.
    """
    return jsonify({
        "sistema": "Sistema Inteligente de Semáforos — La Paz",
        "estado": "operativo",
        "version": "1.0.0",
        "modelos": {
            "mlp_semaforo": "cargado" if _modelo_mlp else "no disponible",
            "lstm_prediccion": "cargado" if _modelo_lstm else "no disponible",
            "cnn_deteccion": "cargado" if _modelo_cnn else "no disponible",
        },
        "pool_inferencia": {
            "max_workers": POOL_SIZE,
            "activo": True,
        },
        "timestamp": datetime.utcnow().isoformat(),
    }), 200


# ── Endpoint: Fase Óptima del Semáforo (MLP) ────────────────────────────────

@semaforo_bp.route("/fase_optima", methods=["POST"])
def fase_optima():
    """
    POST /api/fase_optima

    Recibe datos de flujo, velocidad y densidad de una intersección,
    ejecuta el modelo MLP y retorna la fase óptima del semáforo.

    Payload JSON esperado:
    {
        "interseccion_id": "INT-001-PRADO",
        "datos_sensores": {
            "flujo_ns_1": 18, "flujo_ns_2": 22,
            "flujo_eo_1": 12, "flujo_eo_2": 8,
            "velocidad_ns_1": 25.5, "velocidad_ns_2": 30.2,
            "velocidad_eo_1": 40.1, "velocidad_eo_2": 38.7,
            "densidad_ns_1": 45.0, "densidad_ns_2": 38.0,
            "densidad_eo_1": 20.0, "densidad_eo_2": 15.0,
            "hora": 8, "dia_semana": 1, "mes": 6,
            "lluvia_mm": 0.5, "temperatura": 12.0, "visibilidad_km": 8.0,
            "peatones_cruce_1": 15, "peatones_cruce_2": 10,
            "ocupacion_sensor_1": 0.72, "ocupacion_sensor_2": 0.65,
            "ocupacion_sensor_3": 0.30, "ocupacion_sensor_4": 0.25
        }
    }

    Respuesta JSON:
    {
        "interseccion_id": "INT-001-PRADO",
        "fase_recomendada": 0,
        "nombre_fase": "Norte-Sur Verde",
        "confianza": 0.8723,
        "tiempo_verde_segundos": 42,
        "probabilidades": [0.8723, 0.0821, 0.0312, 0.0144],
        "timestamp": "2026-06-12T19:22:14"
    }
    """
    # ── Validar que el modelo MLP esté disponible ──
    if _modelo_mlp is None:
        logger.error("Modelo MLP no cargado")
        return jsonify({
            "error": "Modelo MLP no disponible",
            "detalle": "El modelo no ha sido cargado en memoria. "
                       "Verifique la inicialización en main.py.",
        }), 503

    # ── Parsear y validar el payload ──
    payload = request.get_json(silent=True)
    if payload is None:
        return jsonify({"error": "Payload JSON requerido"}), 400

    interseccion_id = payload.get("interseccion_id", "desconocida")
    datos_sensores = payload.get("datos_sensores")

    if datos_sensores is None:
        return jsonify({
            "error": "Campo 'datos_sensores' requerido",
            "detalle": "Debe incluir un objeto con las 24 variables de sensores.",
        }), 400

    # ── Convertir el dict de sensores a array NumPy de 24 features ──
    try:
        # Orden esperado de las 24 features
        claves_ordenadas = [
            "flujo_ns_1", "flujo_ns_2", "flujo_eo_1", "flujo_eo_2",
            "velocidad_ns_1", "velocidad_ns_2", "velocidad_eo_1", "velocidad_eo_2",
            "densidad_ns_1", "densidad_ns_2", "densidad_eo_1", "densidad_eo_2",
            "hora", "dia_semana", "mes",
            "lluvia_mm", "temperatura", "visibilidad_km",
            "peatones_cruce_1", "peatones_cruce_2",
            "ocupacion_sensor_1", "ocupacion_sensor_2",
            "ocupacion_sensor_3", "ocupacion_sensor_4",
        ]

        # Si los datos vienen como dict, extraer en orden
        if isinstance(datos_sensores, dict):
            valores = []
            for clave in claves_ordenadas:
                valor = datos_sensores.get(clave)
                if valor is None:
                    return jsonify({
                        "error": f"Campo faltante: '{clave}'",
                        "campos_requeridos": claves_ordenadas,
                    }), 400
                valores.append(float(valor))
            array_entrada = np.array(valores, dtype=np.float32).reshape(1, -1)

        # Si los datos vienen como lista directa de 24 valores
        elif isinstance(datos_sensores, list):
            if len(datos_sensores) != 24:
                return jsonify({
                    "error": f"Se esperan 24 valores, se recibieron {len(datos_sensores)}",
                }), 400
            array_entrada = np.array(datos_sensores, dtype=np.float32).reshape(1, -1)

        else:
            return jsonify({
                "error": "'datos_sensores' debe ser un objeto o una lista de 24 valores",
            }), 400

    except (ValueError, TypeError) as e:
        logger.warning("Error procesando datos de sensores: %s", e)
        return jsonify({"error": f"Datos inválidos: {str(e)}"}), 400

    # ── Normalización y predicción con el MLP ──
    try:
        resultado = _modelo_mlp.predecir_fase(array_entrada)
    except Exception as e:
        logger.error("Error en inferencia MLP: %s", e, exc_info=True)
        return jsonify({"error": f"Error de inferencia: {str(e)}"}), 500

    # ── Persistir la decisión en la base de datos ──
    try:
        registro = RegistroDecision(
            interseccion_id=interseccion_id,
            fase_decidida=resultado["fase_recomendada"],
            nombre_fase=resultado["nombre_fase"],
            nivel_congestion="N/A",  # Se llenará si se integra con LSTM
            confianza=resultado["confianza"],
            tiempo_verde=resultado["tiempo_verde_segundos"],
        )
        db.session.add(registro)
        db.session.commit()
        logger.info(
            "Decisión registrada: intersección=%s, fase=%d, confianza=%.2f%%",
            interseccion_id,
            resultado["fase_recomendada"],
            resultado["confianza"] * 100,
        )
    except Exception as e:
        db.session.rollback()
        logger.error("Error guardando en BD: %s", e)
        # No falla la respuesta, solo advierte
        resultado["advertencia_bd"] = "No se pudo guardar el registro en la BD."

    # ── Construir respuesta ──
    respuesta = {
        "interseccion_id": interseccion_id,
        "fase_recomendada": resultado["fase_recomendada"],
        "nombre_fase": resultado["nombre_fase"],
        "confianza": resultado["confianza"],
        "tiempo_verde_segundos": resultado["tiempo_verde_segundos"],
        "probabilidades": resultado["probabilidades"],
        "timestamp": datetime.utcnow().isoformat(),
    }

    return jsonify(respuesta), 200


# ── Endpoint: Predicción de Congestión (LSTM) ───────────────────────────────

@semaforo_bp.route("/predecir_congestion", methods=["POST"])
def predecir_congestion():
    """
    POST /api/predecir_congestion

    Recibe una ventana temporal de datos de sensores y predice la
    congestión para los próximos 15, 20 y 30 minutos.

    Payload JSON:
    {
        "interseccion_id": "INT-001-PRADO",
        "ventana_temporal": [
            [12.0, 35.0, 25.0, 0.6, 0.5, 0.87, 0.0, 8],
            ... (12 pasos temporales, 8 features cada uno)
        ]
    }
    """
    if _modelo_lstm is None:
        return jsonify({
            "error": "Modelo LSTM no disponible",
        }), 503

    payload = request.get_json(silent=True)
    if payload is None:
        return jsonify({"error": "Payload JSON requerido"}), 400

    interseccion_id = payload.get("interseccion_id", "desconocida")
    ventana = payload.get("ventana_temporal")

    if ventana is None:
        return jsonify({
            "error": "Campo 'ventana_temporal' requerido",
            "detalle": "Debe ser una matriz de 12 pasos × 8 features.",
        }), 400

    try:
        array_ventana = np.array(ventana, dtype=np.float32)
        if array_ventana.shape != (12, 8):
            return jsonify({
                "error": f"Dimensiones incorrectas: se esperaba (12, 8), "
                         f"se recibió {array_ventana.shape}",
            }), 400
    except (ValueError, TypeError) as e:
        return jsonify({"error": f"Datos inválidos: {str(e)}"}), 400

    # Ejecutar predicción (LSTM es relativamente rápida, hilo principal)
    try:
        resultado = _modelo_lstm.predecir_congestion(array_ventana)
    except Exception as e:
        logger.error("Error en inferencia LSTM: %s", e, exc_info=True)
        return jsonify({"error": f"Error de inferencia: {str(e)}"}), 500

    resultado["interseccion_id"] = interseccion_id
    resultado["timestamp"] = datetime.utcnow().isoformat()

    return jsonify(resultado), 200


# ── Endpoint: Detección de Incidentes (CNN — ThreadPoolExecutor) ─────────────

@semaforo_bp.route("/detectar_incidente", methods=["POST"])
def detectar_incidente():
    """
    POST /api/detectar_incidente

    Recibe una imagen codificada en base64 y detecta si hay un incidente
    vial. La inferencia se ejecuta en un hilo separado para no bloquear
    el servidor.

    Payload JSON:
    {
        "interseccion_id": "INT-001-PRADO",
        "imagen_base64": "<string base64 de la imagen>"
    }

    NOTA SOBRE CONCURRENCIA:
    La inferencia de la CNN (ResNet50) se delega a un ThreadPoolExecutor
    para no bloquear el hilo principal de Flask. Esto permite que el servidor
    continúe atendiendo otras solicitudes (como /api/fase_optima) mientras
    la CNN procesa la imagen.
    """
    if _modelo_cnn is None:
        return jsonify({
            "error": "Modelo CNN no disponible",
        }), 503

    payload = request.get_json(silent=True)
    if payload is None:
        return jsonify({"error": "Payload JSON requerido"}), 400

    interseccion_id = payload.get("interseccion_id", "desconocida")
    imagen_b64 = payload.get("imagen_base64")

    if imagen_b64 is None:
        return jsonify({
            "error": "Campo 'imagen_base64' requerido",
        }), 400

    # ── Decodificar imagen ──
    try:
        imagen_bytes = base64.b64decode(imagen_b64)
        imagen_array = np.frombuffer(imagen_bytes, dtype=np.uint8)

        # Intentar decodificar con OpenCV
        try:
            import cv2
            imagen = cv2.imdecode(imagen_array, cv2.IMREAD_COLOR)
            if imagen is None:
                raise ValueError("OpenCV no pudo decodificar la imagen")
        except ImportError:
            # Fallback: asumir formato raw (para pruebas)
            logger.warning("OpenCV no disponible, usando datos raw")
            imagen = imagen_array.reshape(224, 224, 3).astype(np.float32)

    except Exception as e:
        return jsonify({
            "error": f"Error decodificando imagen: {str(e)}",
        }), 400

    # ── Ejecutar inferencia en ThreadPoolExecutor (NO bloqueante) ──
    # Esto es crucial desde la perspectiva de Sistemas Operativos:
    # - La CNN con ResNet50 puede tardar ~200ms en CPU.
    # - Sin el ThreadPoolExecutor, ese tiempo bloquearía el hilo de Flask,
    #   impidiendo que se atiendan otras requests simultáneas.
    # - TensorFlow libera el GIL durante operaciones computacionales,
    #   por lo que los threads realmente ejecutan en paralelo.
    try:
        future: Future = executor.submit(
            _modelo_cnn.detectar_incidente, imagen
        )
        # Esperar resultado con timeout de 30 segundos
        resultado = future.result(timeout=30)
    except TimeoutError:
        logger.error("Timeout en inferencia CNN para %s", interseccion_id)
        return jsonify({
            "error": "Timeout en la detección de incidentes",
            "detalle": "La inferencia excedió los 30 segundos.",
        }), 504
    except Exception as e:
        logger.error("Error en inferencia CNN: %s", e, exc_info=True)
        return jsonify({"error": f"Error de inferencia: {str(e)}"}), 500

    # ── Persistir detección de incidente ──
    try:
        registro = RegistroIncidente(
            interseccion_id=interseccion_id,
            incidente_detectado=resultado["incidente_detectado"],
            probabilidad=resultado["probabilidad_incidente"],
            requiere_atencion=resultado["requiere_atencion"],
        )
        db.session.add(registro)
        db.session.commit()
    except Exception as e:
        db.session.rollback()
        logger.error("Error guardando incidente en BD: %s", e)

    resultado["interseccion_id"] = interseccion_id
    resultado["timestamp"] = datetime.utcnow().isoformat()

    return jsonify(resultado), 200


# ── Endpoint: Historial de Decisiones ────────────────────────────────────────

@semaforo_bp.route("/historial", methods=["GET"])
def historial_decisiones():
    """
    GET /api/historial?limite=50&interseccion=INT-001-PRADO

    Retorna los últimos registros de decisiones de la IA.

    Query Parameters:
      - limite (int): Número máximo de registros (default: 50)
      - interseccion (str): Filtrar por ID de intersección (opcional)
    """
    limite = request.args.get("limite", 50, type=int)
    interseccion = request.args.get("interseccion", None, type=str)

    query = RegistroDecision.query.order_by(RegistroDecision.timestamp.desc())

    if interseccion:
        query = query.filter_by(interseccion_id=interseccion)

    registros = query.limit(min(limite, 500)).all()

    return jsonify({
        "total_registros": len(registros),
        "registros": [r.to_dict() for r in registros],
    }), 200


# ── Manejo de errores global del Blueprint ───────────────────────────────────

@semaforo_bp.errorhandler(404)
def no_encontrado(error):
    return jsonify({
        "error": "Endpoint no encontrado",
        "endpoints_disponibles": [
            "POST /api/fase_optima",
            "POST /api/predecir_congestion",
            "POST /api/detectar_incidente",
            "GET  /api/estado",
            "GET  /api/historial",
        ],
    }), 404


@semaforo_bp.errorhandler(500)
def error_interno(error):
    return jsonify({
        "error": "Error interno del servidor",
        "detalle": str(error),
    }), 500
