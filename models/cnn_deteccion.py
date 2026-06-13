"""
=============================================================================
 Módulo: cnn_deteccion.py
 Sistema Inteligente de Semáforos y Vigilancia para La Paz
 ─────────────────────────────────────────────────────────
 Red Neuronal Convolucional (CNN) basada en Transfer Learning con ResNet50
 para detección de incidentes viales en tiempo real a partir de imágenes
 de cámaras de vigilancia instaladas en intersecciones de La Paz.

 ARQUITECTURA
 ────────────
   Base      →  ResNet50 (pre-entrenada en ImageNet, capas congeladas)
   Cabeza    →  GlobalAveragePooling2D
              →  Dense(256, ReLU) + Dropout(0.5)
              →  Dense(128, ReLU) + Dropout(0.3)
              →  Dense(1, Sigmoid)  ← Clasificación binaria

 Clases de salida:
   0 → Tráfico normal
   1 → Incidente detectado (accidente, vehículo detenido, obstrucción)

 CONEXIÓN FUTURA CON CÁMARAS (MQTT + OpenCV)
 ────────────────────────────────────────────
 En producción, las imágenes provendrán de cámaras IP en las intersecciones:
   • Un proceso captura frames cada N segundos usando OpenCV (cv2.VideoCapture)
   • Los frames se pre-procesan (resize a 224x224, normalización) y se envían
     al modelo CNN para clasificación.
   • Si se detecta un incidente (probabilidad > umbral), se publica una alerta
     en MQTT: "lapaz/interseccion/{id}/alerta"
   • El sistema central puede entonces:
     - Activar protocolo de emergencia en el semáforo
     - Notificar a las autoridades de tránsito
     - Registrar el evento en la base de datos

 NOTA SOBRE RENDIMIENTO
 ──────────────────────
 La inferencia de la CNN es computacionalmente costosa. En el diseño del
 sistema, este modelo se ejecuta en un ThreadPoolExecutor separado para
 no bloquear el hilo principal de Flask (ver api/semaforo_api.py).
=============================================================================
"""

from __future__ import annotations

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from typing import Optional, Tuple, Dict, Any

# Intentar importar OpenCV (opcional para pre-procesamiento)
try:
    import cv2
    CV2_DISPONIBLE = True
except ImportError:
    CV2_DISPONIBLE = False


# ── Constantes ───────────────────────────────────────────────────────────────
IMG_SIZE: Tuple[int, int] = (224, 224)  # Tamaño de entrada de ResNet50
IMG_SHAPE: Tuple[int, int, int] = (224, 224, 3)
UMBRAL_INCIDENTE: float = 0.5  # Probabilidad mínima para declarar incidente
ETIQUETAS: Dict[int, str] = {
    0: "Tráfico Normal",
    1: "Incidente Detectado",
}


def construir_cnn_deteccion(
    input_shape: Tuple[int, int, int] = IMG_SHAPE,
    fine_tune_desde: Optional[int] = None,
    learning_rate: float = 0.0001,
) -> keras.Model:
    """
    Construye una CNN con Transfer Learning (ResNet50) para detección
    binaria de incidentes viales.

    Parameters
    ----------
    input_shape : tuple
        Dimensiones de la imagen de entrada (alto, ancho, canales).
    fine_tune_desde : int, optional
        Si se especifica, descongela las capas de ResNet50 desde esta capa
        en adelante para fine-tuning. None = todas congeladas.
    learning_rate : float
        Tasa de aprendizaje para Adam.

    Returns
    -------
    keras.Model
        Modelo compilado para clasificación binaria.
    """
    # ── Base: ResNet50 pre-entrenada ──
    base_model = ResNet50(
        weights="imagenet",
        include_top=False,          # Excluir la cabeza de clasificación original
        input_shape=input_shape,
    )

    # Congelar todas las capas de la base por defecto
    base_model.trainable = False

    # Opcionalmente descongelar capas para fine-tuning
    if fine_tune_desde is not None:
        base_model.trainable = True
        for layer in base_model.layers[:fine_tune_desde]:
            layer.trainable = False

    # ── Cabeza de clasificación personalizada ──
    inputs = keras.Input(shape=input_shape, name="entrada_imagen")

    # Pre-procesamiento integrado (normalización de ResNet50)
    x = keras.applications.resnet50.preprocess_input(inputs)

    # Pasar por la base convolucional
    x = base_model(x, training=False)

    # Pooling global
    x = layers.GlobalAveragePooling2D(name="pooling_global")(x)

    # Capas densas de clasificación
    x = layers.Dense(256, activation="relu", name="densa_1")(x)
    x = layers.Dropout(0.5, name="dropout_1")(x)

    x = layers.Dense(128, activation="relu", name="densa_2")(x)
    x = layers.Dropout(0.3, name="dropout_2")(x)

    # Salida binaria
    outputs = layers.Dense(1, activation="sigmoid", name="salida_incidente")(x)

    modelo = keras.Model(
        inputs=inputs,
        outputs=outputs,
        name="CNN_Deteccion_Incidentes_LaPaz",
    )

    modelo.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy", keras.metrics.AUC(name="auc")],
    )

    return modelo


class DeteccionCNN:
    """
    Wrapper de alto nivel para la CNN de detección de incidentes viales.

    Encapsula el pre-procesamiento de imágenes (con OpenCV), la inferencia
    y el post-procesamiento de resultados.

    Attributes
    ----------
    modelo : keras.Model
        Red CNN compilada.
    umbral : float
        Umbral de probabilidad para clasificar como incidente.
    entrenado : bool
        Indica si el modelo ha sido entrenado o cargado.
    """

    def __init__(
        self,
        input_shape: Tuple[int, int, int] = IMG_SHAPE,
        umbral: float = UMBRAL_INCIDENTE,
        learning_rate: float = 0.0001,
    ) -> None:
        self.input_shape = input_shape
        self.umbral = umbral
        self.modelo = construir_cnn_deteccion(
            input_shape=input_shape,
            learning_rate=learning_rate,
        )
        self.entrenado: bool = False

    # ── Pre-procesamiento de imágenes ────────────────────────────────────

    def preprocesar_imagen(self, imagen: np.ndarray) -> np.ndarray:
        """
        Pre-procesa una imagen para inferencia con ResNet50.

        Parameters
        ----------
        imagen : np.ndarray
            Imagen en formato BGR (OpenCV) o RGB, de cualquier tamaño.

        Returns
        -------
        np.ndarray, shape (1, 224, 224, 3)
            Imagen redimensionada y normalizada lista para predicción.
        """
        if CV2_DISPONIBLE:
            # Redimensionar con OpenCV (más eficiente)
            img = cv2.resize(imagen, IMG_SIZE, interpolation=cv2.INTER_LINEAR)
            # Convertir BGR a RGB si es necesario
            if len(img.shape) == 3 and img.shape[2] == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            # Fallback: resize con TensorFlow
            img = tf.image.resize(imagen, IMG_SIZE).numpy()

        img = img.astype(np.float32)

        # Expandir dimensiones para batch
        if img.ndim == 3:
            img = np.expand_dims(img, axis=0)

        return img

    @staticmethod
    def crear_generador_datos(
        directorio: str,
        batch_size: int = 32,
        modo: str = "training",
    ) -> ImageDataGenerator:
        """
        Crea un generador de datos con data augmentation para entrenamiento.

        En producción, el directorio debe tener la estructura:
            directorio/
            ├── normal/       ← imágenes de tráfico normal
            └── incidente/    ← imágenes de incidentes viales

        Parameters
        ----------
        directorio : str
            Ruta al directorio con subdirectorios de clases.
        batch_size : int
            Tamaño de batch.
        modo : str
            "training" aplica augmentation, "validation" solo rescale.

        Returns
        -------
        ImageDataGenerator
            Generador configurado.
        """
        if modo == "training":
            datagen = ImageDataGenerator(
                rotation_range=20,
                width_shift_range=0.2,
                height_shift_range=0.2,
                horizontal_flip=True,
                zoom_range=0.15,
                brightness_range=[0.8, 1.2],
                fill_mode="nearest",
            )
        else:
            datagen = ImageDataGenerator()

        return datagen

    # ── Entrenamiento ────────────────────────────────────────────────────

    def entrenar(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        epochs: int = 20,
        batch_size: int = 32,
    ) -> keras.callbacks.History:
        """
        Entrena la CNN con imágenes pre-procesadas.

        Parameters
        ----------
        X_train : np.ndarray, shape (n, 224, 224, 3)
        y_train : np.ndarray, shape (n,) con valores 0 o 1
        X_val, y_val : opcionales para validación.
        epochs, batch_size : hiperparámetros.

        Returns
        -------
        History
        """
        validation_data = None
        if X_val is not None and y_val is not None:
            validation_data = (X_val, y_val)

        cbs = [
            callbacks.EarlyStopping(
                monitor="val_loss" if validation_data else "loss",
                patience=5,
                restore_best_weights=True,
            ),
            callbacks.ReduceLROnPlateau(
                monitor="val_loss" if validation_data else "loss",
                factor=0.5,
                patience=3,
                min_lr=1e-7,
            ),
        ]

        history = self.modelo.fit(
            X_train,
            y_train,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=cbs,
            verbose=1,
        )

        self.entrenado = True
        return history

    # ── Predicción ───────────────────────────────────────────────────────

    def detectar_incidente(self, imagen: np.ndarray) -> Dict[str, Any]:
        """
        Analiza una imagen de cámara y determina si hay un incidente vial.

        Parameters
        ----------
        imagen : np.ndarray
            Imagen cruda de la cámara (cualquier tamaño, BGR o RGB).

        Returns
        -------
        dict
            {
                "incidente_detectado": bool,
                "probabilidad_incidente": float,
                "clasificacion": str,
                "confianza": float,
                "requiere_atencion": bool,
            }
        """
        # Pre-procesar
        img_procesada = self.preprocesar_imagen(imagen)

        # Inferencia
        probabilidad = float(self.modelo.predict(img_procesada, verbose=0)[0][0])

        incidente = probabilidad >= self.umbral
        confianza = probabilidad if incidente else (1.0 - probabilidad)

        return {
            "incidente_detectado": incidente,
            "probabilidad_incidente": round(probabilidad, 4),
            "clasificacion": ETIQUETAS[int(incidente)],
            "confianza": round(confianza, 4),
            "requiere_atencion": probabilidad >= 0.8,  # Alta certeza
        }

    def analizar_frame_camara(
        self, frame: np.ndarray, interseccion_id: str = "desconocida"
    ) -> Dict[str, Any]:
        """
        Versión extendida de detección que incluye metadatos de la
        intersección, pensada para integración con el sistema MQTT.

        Parameters
        ----------
        frame : np.ndarray
            Frame capturado de la cámara IP.
        interseccion_id : str
            Identificador de la intersección.

        Returns
        -------
        dict
            Resultado de detección + metadatos.
        """
        from datetime import datetime

        resultado = self.detectar_incidente(frame)
        resultado.update({
            "interseccion_id": interseccion_id,
            "timestamp": datetime.now().isoformat(),
            # En producción, aquí se publicaría vía MQTT:
            # mqtt_client.publish(
            #     f"lapaz/interseccion/{interseccion_id}/alerta",
            #     json.dumps(resultado)
            # )
        })

        return resultado

    # ── Fine-tuning ──────────────────────────────────────────────────────

    def activar_fine_tuning(
        self,
        descongelar_desde: int = 143,
        learning_rate: float = 1e-5,
    ) -> None:
        """
        Descongela las últimas capas de ResNet50 para fine-tuning.

        La ResNet50 tiene 175 capas. Descongelar desde la capa 143 permite
        re-entrenar los últimos bloques residuales manteniendo las features
        de bajo nivel (bordes, texturas) intactas.

        Parameters
        ----------
        descongelar_desde : int
            Índice de capa desde el cual descongelar.
        learning_rate : float
            Tasa de aprendizaje reducida para fine-tuning.
        """
        base = self.modelo.layers[1]  # ResNet50 es la segunda capa
        base.trainable = True
        for layer in base.layers[:descongelar_desde]:
            layer.trainable = False

        # Recompilar con learning rate más bajo
        self.modelo.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss="binary_crossentropy",
            metrics=["accuracy", keras.metrics.AUC(name="auc")],
        )

    # ── Persistencia ─────────────────────────────────────────────────────

    def guardar(self, ruta_modelo: str = "cnn_deteccion.keras") -> None:
        """Guarda el modelo en disco."""
        self.modelo.save(ruta_modelo)

    def cargar(self, ruta_modelo: str = "cnn_deteccion.keras") -> None:
        """Carga el modelo desde disco."""
        self.modelo = keras.models.load_model(ruta_modelo)
        self.entrenado = True


# ── Utilidad: datos sintéticos para pruebas ──────────────────────────────────

def generar_imagenes_sinteticas(
    n_muestras: int = 200,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Genera imágenes sintéticas aleatorias para pruebas del pipeline.

    NOTA: En un proyecto real, estas serían imágenes capturadas de las
    cámaras de vigilancia de las intersecciones de La Paz, etiquetadas
    manualmente como 'normal' o 'incidente'.

    Returns
    -------
    X : np.ndarray, shape (n_muestras, 224, 224, 3)
    y : np.ndarray, shape (n_muestras,) con valores 0 o 1
    """
    rng = np.random.RandomState(seed)

    X = rng.randint(0, 256, size=(n_muestras, *IMG_SHAPE)).astype(np.float32)

    # Simular: 70% tráfico normal, 30% incidentes
    y = (rng.random(n_muestras) > 0.7).astype(np.int32)

    # Añadir "señal" artificial a las imágenes de incidente
    # (parche rojo brillante para simular un vehículo detenido)
    for i in range(n_muestras):
        if y[i] == 1:
            # Añadir un rectángulo rojo como "señal" de incidente
            cx, cy = rng.randint(50, 174, size=2)
            X[i, cy - 20:cy + 20, cx - 30:cx + 30, 0] = 255  # Canal rojo
            X[i, cy - 20:cy + 20, cx - 30:cx + 30, 1] = 50
            X[i, cy - 20:cy + 20, cx - 30:cx + 30, 2] = 50

    return X, y


# ── Ejecución directa para pruebas ──────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 70)
    print("  CNN Detección de Incidentes — Sistema Inteligente La Paz")
    print("=" * 70)

    # 1. Construir modelo
    cnn = DeteccionCNN()
    cnn.modelo.summary()

    # 2. Mostrar arquitectura
    print(f"\nCapas totales del modelo: {len(cnn.modelo.layers)}")
    print(f"Parámetros entrenables: {cnn.modelo.count_params():,}")

    # 3. Predicción de ejemplo con imagen sintética
    img_prueba = np.random.randint(0, 256, size=(480, 640, 3)).astype(np.float32)
    resultado = cnn.detectar_incidente(img_prueba)
    print(f"\n{'─'*50}")
    print(f"  Resultado detección:")
    print(f"    Incidente: {resultado['incidente_detectado']}")
    print(f"    Probabilidad: {resultado['probabilidad_incidente']:.2%}")
    print(f"    Clasificación: {resultado['clasificacion']}")
    print(f"    Confianza: {resultado['confianza']:.2%}")
    print(f"    Requiere atención: {resultado['requiere_atencion']}")
    print(f"{'─'*50}")

    print("\n[INFO] Para entrenamiento completo, proporcionar un directorio")
    print("       con imágenes etiquetadas de intersecciones de La Paz.")
