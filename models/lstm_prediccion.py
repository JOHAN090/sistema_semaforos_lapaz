"""
=============================================================================
 Módulo: lstm_prediccion.py
 Sistema Inteligente de Semáforos y Vigilancia para La Paz
 ─────────────────────────────────────────────────────────
 Red LSTM para predicción de congestión vehicular a corto plazo
 (horizontes de 15, 20 y 30 minutos) en intersecciones de La Paz.

 ARQUITECTURA
 ────────────
   Entrada  →  (batch, 12 pasos temporales, N features)
   LSTM 1   →  100 unidades, return_sequences=True, dropout=0.2
   LSTM 2   →  50 unidades, dropout=0.2
   Dense    →  32 (ReLU)
   Salida   →  Dense(3) → predicción lineal para 3 horizontes

 Cada paso temporal contiene features como:
   - Flujo vehicular agregado por dirección
   - Velocidad media
   - Densidad
   - Variables temporales codificadas (hora, día)
   - Condiciones meteorológicas

 CONEXIÓN FUTURA CON SENSORES (MQTT)
 ────────────────────────────────────
 En producción, los datos de la serie temporal se construirán a partir
 de un buffer circular que acumula las últimas 12 lecturas de sensores
 (cada lectura cada 5 minutos = ventana de 1 hora):
   • Suscripción MQTT: "lapaz/interseccion/{id}/historico"
   • Un proceso en background mantendrá el buffer y, cada 5 minutos,
     alimentará la LSTM para obtener predicciones de congestión a
     15, 20 y 30 minutos.
   • Las predicciones se publicarán en: "lapaz/interseccion/{id}/prediccion"
     para que el controlador del semáforo pueda anticipar fases.
=============================================================================
"""

from __future__ import annotations

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks
from sklearn.preprocessing import MinMaxScaler
from typing import Optional, Tuple, Dict, Any, List


# ── Constantes ───────────────────────────────────────────────────────────────
VENTANA_TEMPORAL: int = 12         # 12 pasos temporales (p.ej. cada 5 min = 1 hora)
NUM_FEATURES_SERIE: int = 8       # Features por paso temporal
HORIZONTES_PREDICCION: int = 3    # 15min, 20min, 30min
NOMBRES_HORIZONTES: List[str] = ["15 min", "20 min", "30 min"]


def construir_lstm_prediccion(
    ventana: int = VENTANA_TEMPORAL,
    num_features: int = NUM_FEATURES_SERIE,
    horizontes: int = HORIZONTES_PREDICCION,
    learning_rate: float = 0.001,
) -> keras.Model:
    """
    Construye y compila la red LSTM para predicción de congestión.

    Parameters
    ----------
    ventana : int
        Número de pasos temporales de la secuencia de entrada.
    num_features : int
        Número de features en cada paso temporal.
    horizontes : int
        Número de valores de salida (horizontes de predicción).
    learning_rate : float
        Tasa de aprendizaje para Adam.

    Returns
    -------
    keras.Model
        Modelo LSTM compilado.
    """
    modelo = keras.Sequential(
        [
            # --- Capa de entrada ---
            keras.Input(
                shape=(ventana, num_features),
                name="entrada_serie_temporal",
            ),

            # --- LSTM Capa 1: 100 unidades, devuelve secuencia completa ---
            layers.LSTM(
                100,
                return_sequences=True,
                dropout=0.2,
                recurrent_dropout=0.1,
                name="lstm_1",
            ),

            # --- LSTM Capa 2: 50 unidades, solo último estado ---
            layers.LSTM(
                50,
                return_sequences=False,
                dropout=0.2,
                recurrent_dropout=0.1,
                name="lstm_2",
            ),

            # --- Capa Densa intermedia ---
            layers.Dense(32, activation="relu", name="densa_intermedia"),

            # --- Salida: 3 valores (predicciones a 15, 20, 30 min) ---
            layers.Dense(horizontes, activation="linear", name="salida_prediccion"),
        ],
        name="LSTM_Prediccion_Congestion_LaPaz",
    )

    modelo.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="mse",
        metrics=["mae"],
    )

    return modelo


class PrediccionLSTM:
    """
    Wrapper de alto nivel para la LSTM de predicción de congestión.

    Gestiona el escalado de datos, el entrenamiento y la inferencia,
    además de proporcionar métodos para construir las ventanas temporales
    a partir de datos crudos de sensores.

    Attributes
    ----------
    modelo : keras.Model
        Red LSTM compilada.
    scaler_X : MinMaxScaler
        Escalador para las features de entrada.
    scaler_y : MinMaxScaler
        Escalador para los valores objetivo.
    entrenado : bool
        Indica si el modelo ha sido entrenado.
    """

    def __init__(
        self,
        ventana: int = VENTANA_TEMPORAL,
        num_features: int = NUM_FEATURES_SERIE,
        horizontes: int = HORIZONTES_PREDICCION,
        learning_rate: float = 0.001,
    ) -> None:
        self.ventana = ventana
        self.num_features = num_features
        self.horizontes = horizontes
        self.modelo = construir_lstm_prediccion(
            ventana, num_features, horizontes, learning_rate
        )
        self.scaler_X = MinMaxScaler(feature_range=(0, 1))
        self.scaler_y = MinMaxScaler(feature_range=(0, 1))
        self.entrenado: bool = False

    # ── Preparación de datos ─────────────────────────────────────────────

    @staticmethod
    def crear_ventanas(
        datos: np.ndarray,
        objetivos: np.ndarray,
        ventana: int = VENTANA_TEMPORAL,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Crea ventanas deslizantes para entrenamiento de la LSTM.

        Parameters
        ----------
        datos : np.ndarray, shape (n_timesteps, num_features)
            Serie temporal completa de features.
        objetivos : np.ndarray, shape (n_timesteps, horizontes)
            Valores objetivo correspondientes a cada timestep.
        ventana : int
            Tamaño de la ventana (pasos hacia atrás).

        Returns
        -------
        X : np.ndarray, shape (n_samples, ventana, num_features)
        y : np.ndarray, shape (n_samples, horizontes)
        """
        X, y = [], []
        for i in range(ventana, len(datos)):
            X.append(datos[i - ventana : i])
            y.append(objetivos[i])
        return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

    # ── Entrenamiento ────────────────────────────────────────────────────

    def entrenar(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        epochs: int = 50,
        batch_size: int = 32,
    ) -> keras.callbacks.History:
        """
        Entrena la LSTM con datos de series temporales ya formateados
        como ventanas (batch, ventana, features).

        Parameters
        ----------
        X_train : np.ndarray, shape (n, ventana, features)
        y_train : np.ndarray, shape (n, horizontes)
        X_val, y_val : opcionales para validación.
        epochs, batch_size : hiperparámetros de entrenamiento.

        Returns
        -------
        History
        """
        # Escalar features: reshape → (n*ventana, features) → escalar → reshape
        n, v, f = X_train.shape
        X_flat = X_train.reshape(-1, f)
        X_flat_norm = self.scaler_X.fit_transform(X_flat)
        X_train_norm = X_flat_norm.reshape(n, v, f)

        # Escalar objetivos
        y_train_norm = self.scaler_y.fit_transform(y_train)

        validation_data = None
        if X_val is not None and y_val is not None:
            n_v = X_val.shape[0]
            X_val_flat = X_val.reshape(-1, f)
            X_val_norm = self.scaler_X.transform(X_val_flat).reshape(n_v, v, f)
            y_val_norm = self.scaler_y.transform(y_val)
            validation_data = (X_val_norm, y_val_norm)

        cbs = [
            callbacks.EarlyStopping(
                monitor="val_loss" if validation_data else "loss",
                patience=8,
                restore_best_weights=True,
            ),
            callbacks.ReduceLROnPlateau(
                monitor="val_loss" if validation_data else "loss",
                factor=0.5,
                patience=4,
                min_lr=1e-6,
            ),
        ]

        history = self.modelo.fit(
            X_train_norm,
            y_train_norm,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=cbs,
            verbose=1,
        )

        self.entrenado = True
        return history

    # ── Predicción ───────────────────────────────────────────────────────

    def predecir_congestion(
        self, ventana_datos: np.ndarray
    ) -> Dict[str, Any]:
        """
        Predice niveles de congestión para los próximos 15, 20 y 30 minutos.

        Parameters
        ----------
        ventana_datos : np.ndarray, shape (ventana, features) o (1, ventana, features)
            Últimos `ventana` pasos temporales de datos de sensores.

        Returns
        -------
        dict
            {
                "predicciones": {
                    "15_min": float,
                    "20_min": float,
                    "30_min": float,
                },
                "nivel_congestion": str,  # "bajo", "medio", "alto", "crítico"
                "indice_congestion": float,  # 0.0 - 1.0
            }
        """
        datos = np.asarray(ventana_datos, dtype=np.float32)
        if datos.ndim == 2:
            datos = datos.reshape(1, *datos.shape)

        # Normalizar
        n, v, f = datos.shape
        datos_flat = datos.reshape(-1, f)
        datos_norm = self.scaler_X.transform(datos_flat).reshape(n, v, f)

        # Inferencia
        pred_norm = self.modelo.predict(datos_norm, verbose=0)
        pred = self.scaler_y.inverse_transform(pred_norm)[0]

        # Calcular índice de congestión agregado (media ponderada)
        pesos = [0.5, 0.3, 0.2]  # Mayor peso al horizonte más cercano
        indice = float(np.clip(np.average(pred, weights=pesos) / 100.0, 0.0, 1.0))

        # Clasificar nivel
        if indice < 0.25:
            nivel = "bajo"
        elif indice < 0.50:
            nivel = "medio"
        elif indice < 0.75:
            nivel = "alto"
        else:
            nivel = "crítico"

        return {
            "predicciones": {
                "15_min": round(float(pred[0]), 2),
                "20_min": round(float(pred[1]), 2),
                "30_min": round(float(pred[2]), 2),
            },
            "nivel_congestion": nivel,
            "indice_congestion": round(indice, 4),
        }

    # ── Persistencia ─────────────────────────────────────────────────────

    def guardar(self, ruta_modelo: str = "lstm_prediccion.keras") -> None:
        """Guarda modelo y escaladores en disco."""
        self.modelo.save(ruta_modelo)
        import joblib
        base = ruta_modelo.replace(".keras", "")
        joblib.dump(self.scaler_X, f"{base}_scaler_X.pkl")
        joblib.dump(self.scaler_y, f"{base}_scaler_y.pkl")

    def cargar(self, ruta_modelo: str = "lstm_prediccion.keras") -> None:
        """Carga modelo y escaladores desde disco."""
        self.modelo = keras.models.load_model(ruta_modelo)
        import joblib
        base = ruta_modelo.replace(".keras", "")
        self.scaler_X = joblib.load(f"{base}_scaler_X.pkl")
        self.scaler_y = joblib.load(f"{base}_scaler_y.pkl")
        self.entrenado = True


# ── Utilidad: datos sintéticos de series temporales ──────────────────────────

def generar_serie_temporal_sintetica(
    n_pasos: int = 2000,
    num_features: int = NUM_FEATURES_SERIE,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Genera una serie temporal sintética que simula el flujo vehicular
    en una intersección de La Paz con patrones diarios y semanales.

    Las features simuladas son:
      0: Flujo vehicular total (veh/5min)
      1: Velocidad media (km/h)
      2: Densidad (veh/km)
      3: Ocupación promedio de sensores (0-1)
      4: Hora del día (0-23, codificada como seno)
      5: Hora del día (coseno)
      6: Indicador de lluvia (0-1)
      7: Conteo peatonal

    Returns
    -------
    datos : np.ndarray, shape (n_pasos, num_features)
    objetivos : np.ndarray, shape (n_pasos, 3)  → congestión a 15/20/30 min
    """
    rng = np.random.RandomState(seed)
    t = np.arange(n_pasos)

    # Patrón diario (288 pasos = 1 día si cada paso = 5 min)
    periodo_diario = 288
    hora_normalizada = (t % periodo_diario) / periodo_diario

    # Pico matutino (~7-9am) y vespertino (~17-19pm)
    pico_am = np.exp(-0.5 * ((hora_normalizada - 0.30) / 0.05) ** 2)
    pico_pm = np.exp(-0.5 * ((hora_normalizada - 0.73) / 0.06) ** 2)
    patron_diario = pico_am + pico_pm * 1.2

    datos = np.zeros((n_pasos, num_features), dtype=np.float32)

    # Feature 0: Flujo vehicular
    datos[:, 0] = 10 + 40 * patron_diario + rng.normal(0, 3, n_pasos)

    # Feature 1: Velocidad (inversamente proporcional al flujo)
    datos[:, 1] = 50 - 25 * patron_diario + rng.normal(0, 5, n_pasos)
    datos[:, 1] = np.clip(datos[:, 1], 5, 60)

    # Feature 2: Densidad
    datos[:, 2] = 5 + 50 * patron_diario + rng.normal(0, 4, n_pasos)
    datos[:, 2] = np.clip(datos[:, 2], 0, 100)

    # Feature 3: Ocupación sensores
    datos[:, 3] = 0.1 + 0.7 * patron_diario + rng.normal(0, 0.05, n_pasos)
    datos[:, 3] = np.clip(datos[:, 3], 0, 1)

    # Features 4-5: Hora codificada (seno/coseno)
    datos[:, 4] = np.sin(2 * np.pi * hora_normalizada)
    datos[:, 5] = np.cos(2 * np.pi * hora_normalizada)

    # Feature 6: Lluvia
    datos[:, 6] = (rng.random(n_pasos) < 0.15).astype(np.float32)

    # Feature 7: Conteo peatonal
    datos[:, 7] = 5 + 15 * patron_diario + rng.poisson(3, n_pasos)

    # ── Objetivos: congestión futura (% de capacidad) ──
    # Simular que la congestión futura es una versión desplazada del patrón actual
    objetivos = np.zeros((n_pasos, 3), dtype=np.float32)
    desplazamientos = [3, 4, 6]  # 15min=3 pasos, 20min=4, 30min=6

    for i, desp in enumerate(desplazamientos):
        congestion_futura = np.roll(datos[:, 2], -desp)  # densidad futura
        objetivos[:, i] = congestion_futura + rng.normal(0, 2, n_pasos)

    objetivos = np.clip(objetivos, 0, 100)

    return datos, objetivos


# ── Ejecución directa para pruebas ──────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 70)
    print("  LSTM Predicción de Congestión — Sistema Inteligente La Paz")
    print("=" * 70)

    # 1. Construir modelo
    lstm = PrediccionLSTM()
    lstm.modelo.summary()

    # 2. Generar datos de prueba
    datos, objetivos = generar_serie_temporal_sintetica(n_pasos=1500)
    print(f"\nSerie temporal generada: datos={datos.shape}, objetivos={objetivos.shape}")

    # 3. Crear ventanas
    X, y = PrediccionLSTM.crear_ventanas(datos, objetivos, ventana=VENTANA_TEMPORAL)
    print(f"Ventanas creadas: X={X.shape}, y={y.shape}")

    # 4. Split train/val
    split = int(len(X) * 0.8)
    X_tr, X_val = X[:split], X[split:]
    y_tr, y_val = y[:split], y[split:]

    # 5. Entrenar
    lstm.entrenar(X_tr, y_tr, X_val, y_val, epochs=10, batch_size=64)

    # 6. Predicción de ejemplo
    ejemplo = X_val[0]
    resultado = lstm.predecir_congestion(ejemplo)
    print(f"\n{'─'*50}")
    print(f"  Predicciones de congestión:")
    for horizonte, valor in resultado["predicciones"].items():
        print(f"    {horizonte}: {valor:.1f}% de capacidad")
    print(f"  Nivel: {resultado['nivel_congestion']}")
    print(f"  Índice: {resultado['indice_congestion']:.2%}")
    print(f"{'─'*50}")
