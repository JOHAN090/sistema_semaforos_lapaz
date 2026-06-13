"""
=============================================================================
 Módulo: mlp_semaforo.py
 Sistema Inteligente de Semáforos y Vigilancia para La Paz
 ─────────────────────────────────────────────────────────
 Perceptrón Multicapa (MLP) para la decisión en tiempo real de la fase
 óptima de un semáforo en una intersección de La Paz.

 ARQUITECTURA
 ────────────
   Entrada  →  24 variables (flujo vehicular, velocidad media, densidad,
                 hora del día, día de la semana, datos meteorológicos,
                 ocupación de sensores inductivos, conteos peatonales, etc.)
   Oculta 1 →  Dense(128, ReLU) → BatchNorm → Dropout(0.3)
   Oculta 2 →  Dense(64,  ReLU) → BatchNorm → Dropout(0.2)
   Oculta 3 →  Dense(32,  ReLU)
   Salida   →  Dense(4,   Softmax)  ← 4 fases del ciclo semafórico

 Las 4 fases de salida representan:
   0 → Norte‑Sur Verde   (Este‑Oeste Rojo)
   1 → Este‑Oeste Verde  (Norte‑Sur Rojo)
   2 → Giro protegido izquierdo
   3 → Todo rojo / peatonal exclusivo

 CONEXIÓN FUTURA CON SENSORES (MQTT)
 ────────────────────────────────────
 En producción, las 24 variables de entrada provendrán de sensores
 físicos (espiras inductivas, cámaras, radares Doppler) publicados en
 tópicos MQTT:
   • tópico: "lapaz/interseccion/{id}/sensores"
   • payload JSON con los 24 campos normalizados.
 Un suscriptor MQTT recibirá los datos, los empaquetará en un array
 NumPy de forma (1, 24) y llamará a `modelo.predict()` para obtener
 la fase recomendada, que luego se publicará en:
   • tópico: "lapaz/interseccion/{id}/fase"
=============================================================================
"""

from __future__ import annotations

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks
from sklearn.preprocessing import StandardScaler
from typing import Optional, Tuple, Dict, Any


# ── Constantes del módulo ────────────────────────────────────────────────────
NUM_FEATURES_ENTRADA: int = 24
NUM_FASES_SALIDA: int = 4
NOMBRES_FASES: Dict[int, str] = {
    0: "Norte-Sur Verde",
    1: "Este-Oeste Verde",
    2: "Giro Protegido Izquierdo",
    3: "Todo Rojo / Peatonal Exclusivo",
}

# Tiempos base de verde por fase (segundos).  Se ajustan dinámicamente
# según el nivel de confianza de la predicción.
TIEMPOS_VERDE_BASE: Dict[int, int] = {
    0: 45,
    1: 40,
    2: 20,
    3: 15,
}


def construir_mlp_semaforo(
    input_dim: int = NUM_FEATURES_ENTRADA,
    num_clases: int = NUM_FASES_SALIDA,
    learning_rate: float = 0.001,
) -> keras.Model:
    """
    Construye y compila el modelo MLP para decisión de fase semafórica.

    Parameters
    ----------
    input_dim : int
        Número de variables de entrada (por defecto 24).
    num_clases : int
        Número de fases de salida (por defecto 4).
    learning_rate : float
        Tasa de aprendizaje del optimizador Adam.

    Returns
    -------
    keras.Model
        Modelo compilado listo para entrenamiento o inferencia.
    """
    modelo = keras.Sequential(
        [
            # --- Capa de entrada ---
            keras.Input(shape=(input_dim,), name="entrada_sensores"),

            # --- Capa Oculta 1: 128 neuronas ---
            layers.Dense(128, activation="relu", name="oculta_1"),
            layers.BatchNormalization(name="batchnorm_1"),
            layers.Dropout(0.3, name="dropout_1"),

            # --- Capa Oculta 2: 64 neuronas ---
            layers.Dense(64, activation="relu", name="oculta_2"),
            layers.BatchNormalization(name="batchnorm_2"),
            layers.Dropout(0.2, name="dropout_2"),

            # --- Capa Oculta 3: 32 neuronas ---
            layers.Dense(32, activation="relu", name="oculta_3"),

            # --- Capa de Salida: Softmax sobre 4 fases ---
            layers.Dense(num_clases, activation="softmax", name="salida_fase"),
        ],
        name="MLP_Semaforo_LaPaz",
    )

    modelo.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return modelo


class SemaforoMLP:
    """
    Wrapper de alto nivel que encapsula el modelo MLP, el escalador de
    características y la lógica de predicción con post‑procesamiento.

    Attributes
    ----------
    modelo : keras.Model
        Red MLP compilada.
    scaler : StandardScaler
        Escalador ajustado a los datos de entrenamiento.
    entrenado : bool
        Indica si el modelo ha sido entrenado o se ha cargado desde disco.
    """

    def __init__(
        self,
        input_dim: int = NUM_FEATURES_ENTRADA,
        num_clases: int = NUM_FASES_SALIDA,
        learning_rate: float = 0.001,
    ) -> None:
        self.input_dim = input_dim
        self.num_clases = num_clases
        self.modelo = construir_mlp_semaforo(input_dim, num_clases, learning_rate)
        self.scaler = StandardScaler()
        self.entrenado: bool = False

    # ── Entrenamiento ────────────────────────────────────────────────────

    def entrenar(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        epochs: int = 100,
        batch_size: int = 32,
    ) -> keras.callbacks.History:
        """
        Entrena el MLP con los datos proporcionados.

        Parameters
        ----------
        X_train : np.ndarray, shape (n_samples, 24)
            Datos de entrenamiento (sin normalizar; el método los normaliza).
        y_train : np.ndarray, shape (n_samples,)
            Etiquetas enteras [0..3].
        X_val, y_val : arrays opcionales para validación.
        epochs, batch_size : hiperparámetros de entrenamiento.

        Returns
        -------
        History
            Historial de entrenamiento de Keras.
        """
        # Ajustar y transformar con el StandardScaler
        X_train_norm = self.scaler.fit_transform(X_train)

        validation_data = None
        if X_val is not None and y_val is not None:
            X_val_norm = self.scaler.transform(X_val)
            validation_data = (X_val_norm, y_val)

        # Callbacks de entrenamiento
        cbs = [
            callbacks.EarlyStopping(
                monitor="val_loss" if validation_data else "loss",
                patience=10,
                restore_best_weights=True,
            ),
            callbacks.ReduceLROnPlateau(
                monitor="val_loss" if validation_data else "loss",
                factor=0.5,
                patience=5,
                min_lr=1e-6,
            ),
        ]

        history = self.modelo.fit(
            X_train_norm,
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

    def predecir_fase(self, datos_sensores: np.ndarray) -> Dict[str, Any]:
        """
        Predice la fase óptima del semáforo a partir de datos de sensores.

        Parameters
        ----------
        datos_sensores : np.ndarray, shape (1, 24) o (24,)
            Vector de 24 features proveniente de los sensores.

        Returns
        -------
        dict
            {
                "fase_recomendada": int,
                "nombre_fase": str,
                "confianza": float,
                "probabilidades": list[float],
                "tiempo_verde_segundos": int,
            }
        """
        datos = np.asarray(datos_sensores, dtype=np.float32)
        if datos.ndim == 1:
            datos = datos.reshape(1, -1)

        # Normalizar con el scaler ajustado en entrenamiento
        datos_norm = self.scaler.transform(datos)

        # Inferencia
        probabilidades = self.modelo.predict(datos_norm, verbose=0)[0]
        fase = int(np.argmax(probabilidades))
        confianza = float(probabilidades[fase])

        # Post-procesamiento: ajustar tiempo de verde según confianza
        tiempo_base = TIEMPOS_VERDE_BASE.get(fase, 30)
        tiempo_verde = int(tiempo_base * (0.8 + 0.4 * confianza))  # rango: 80%-120%

        return {
            "fase_recomendada": fase,
            "nombre_fase": NOMBRES_FASES.get(fase, "Desconocida"),
            "confianza": round(confianza, 4),
            "probabilidades": [round(float(p), 4) for p in probabilidades],
            "tiempo_verde_segundos": tiempo_verde,
        }

    # ── Persistencia ─────────────────────────────────────────────────────

    def guardar(self, ruta_modelo: str = "mlp_semaforo.keras") -> None:
        """Guarda el modelo y el scaler en disco."""
        self.modelo.save(ruta_modelo)
        # Guardar scaler junto al modelo
        import joblib
        ruta_scaler = ruta_modelo.replace(".keras", "_scaler.pkl")
        joblib.dump(self.scaler, ruta_scaler)

    def cargar(self, ruta_modelo: str = "mlp_semaforo.keras") -> None:
        """Carga modelo y scaler desde disco."""
        self.modelo = keras.models.load_model(ruta_modelo)
        import joblib
        ruta_scaler = ruta_modelo.replace(".keras", "_scaler.pkl")
        self.scaler = joblib.load(ruta_scaler)
        self.entrenado = True


# ── Utilidad: datos sintéticos para pruebas rápidas ──────────────────────────

def generar_datos_sinteticos(
    n_muestras: int = 5000,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Genera datos sintéticos que simulan lecturas de sensores en
    intersecciones de La Paz para pruebas del pipeline.

    Los datos simulan:
      - Flujos vehiculares (vehículos/5min) por carril
      - Velocidades medias (km/h)
      - Densidades (veh/km)
      - Variables temporales (hora, día, mes)
      - Indicadores climáticos (lluvia, temperatura, visibilidad)
      - Conteos peatonales
      - Ocupación de sensores inductivos

    Returns
    -------
    X : np.ndarray, shape (n_muestras, 24)
    y : np.ndarray, shape (n_muestras,) con valores en {0, 1, 2, 3}
    """
    rng = np.random.RandomState(seed)

    X = np.zeros((n_muestras, NUM_FEATURES_ENTRADA), dtype=np.float32)

    # Flujos vehiculares (4 carriles principales) — features 0..3
    X[:, 0:4] = rng.poisson(lam=15, size=(n_muestras, 4)).astype(np.float32)

    # Velocidades medias (4 carriles) — features 4..7
    X[:, 4:8] = rng.normal(loc=35.0, scale=10.0, size=(n_muestras, 4)).clip(5, 80)

    # Densidades (4 carriles) — features 8..11
    X[:, 8:12] = rng.exponential(scale=20.0, size=(n_muestras, 4)).clip(0, 100)

    # Hora del día (0-23) — feature 12
    X[:, 12] = rng.randint(0, 24, size=n_muestras)

    # Día de la semana (0=lun, 6=dom) — feature 13
    X[:, 13] = rng.randint(0, 7, size=n_muestras)

    # Mes (1-12) — feature 14
    X[:, 14] = rng.randint(1, 13, size=n_muestras)

    # Lluvia (mm/h) — feature 15
    X[:, 15] = rng.exponential(scale=2.0, size=n_muestras).clip(0, 50)

    # Temperatura (°C, altiplano ~5-20) — feature 16
    X[:, 16] = rng.normal(loc=12.0, scale=5.0, size=n_muestras).clip(-5, 25)

    # Visibilidad (km) — feature 17
    X[:, 17] = rng.normal(loc=8.0, scale=3.0, size=n_muestras).clip(0.5, 15)

    # Conteos peatonales (2 cruces) — features 18..19
    X[:, 18:20] = rng.poisson(lam=8, size=(n_muestras, 2)).astype(np.float32)

    # Ocupación sensores inductivos (4 sensores, 0-1) — features 20..23
    X[:, 20:24] = rng.uniform(0, 1, size=(n_muestras, 4))

    # ── Generar etiquetas con lógica heurística ──
    # Regla simplificada: la fase se decide según cuál dirección tiene más flujo
    flujo_ns = X[:, 0] + X[:, 1]  # flujo Norte-Sur
    flujo_eo = X[:, 2] + X[:, 3]  # flujo Este-Oeste
    peatones = X[:, 18] + X[:, 19]

    y = np.zeros(n_muestras, dtype=np.int32)
    y[flujo_ns > flujo_eo * 1.3] = 0   # Prioridad Norte-Sur
    y[flujo_eo > flujo_ns * 1.3] = 1   # Prioridad Este-Oeste
    y[(np.abs(flujo_ns - flujo_eo) <= flujo_ns * 0.3)] = 2  # Equilibrio → giro protegido
    y[peatones > 20] = 3               # Alta demanda peatonal

    return X, y


# ── Ejecución directa para pruebas ──────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 70)
    print("  MLP Semáforo — Sistema Inteligente para La Paz")
    print("=" * 70)

    # 1. Construir modelo
    mlp = SemaforoMLP()
    mlp.modelo.summary()

    # 2. Generar datos de prueba
    X, y = generar_datos_sinteticos(n_muestras=3000)
    print(f"\nDatos generados: X={X.shape}, y={y.shape}")
    print(f"Distribución de fases: {dict(zip(*np.unique(y, return_counts=True)))}")

    # 3. Entrenar (pocas épocas para demostración)
    from sklearn.model_selection import train_test_split
    X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    mlp.entrenar(X_tr, y_tr, X_val, y_val, epochs=15, batch_size=64)

    # 4. Predicción de ejemplo
    ejemplo = X_val[0:1]
    resultado = mlp.predecir_fase(ejemplo)
    print(f"\n{'─'*50}")
    print(f"  Fase recomendada : {resultado['fase_recomendada']} ({resultado['nombre_fase']})")
    print(f"  Confianza        : {resultado['confianza']:.2%}")
    print(f"  Tiempo verde     : {resultado['tiempo_verde_segundos']}s")
    print(f"  Probabilidades   : {resultado['probabilidades']}")
    print(f"{'─'*50}")
