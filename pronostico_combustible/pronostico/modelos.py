"""
Modelos candidatos.

Todos comparten la misma interfaz, asi el backtesting los trata por igual:

    modelo.predecir(panel, meta, futuros) -> DataFrame (series x meses futuros)

`panel` SIEMPRE viene recortado hasta el ultimo mes conocido, de modo que
ningun modelo puede ver el futuro que se le pide predecir.

Para agregar un modelo propio: crea una subclase de Modelo, registrala en
REGISTRO y activala en config.yaml.
"""

from __future__ import annotations

import warnings
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

from . import caracteristicas as car
from .config import Config


class Modelo(ABC):
    nombre = "base"
    descripcion = ""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.notas: list[str] = []

    @abstractmethod
    def predecir(
        self, panel: pd.DataFrame, meta: pd.DataFrame, futuros: list[pd.Period]
    ) -> pd.DataFrame:
        ...

    @staticmethod
    def _repetir(valores: np.ndarray, futuros: list[pd.Period], indice: pd.Index) -> pd.DataFrame:
        """Aplica el mismo nivel a todos los meses futuros."""
        matriz = np.repeat(valores.reshape(-1, 1), len(futuros), axis=1)
        return pd.DataFrame(matriz, index=indice, columns=futuros)


# ---------------------------------------------------------------------------
# Modelos de referencia (baselines)
# ---------------------------------------------------------------------------
class UltimoValor(Modelo):
    nombre = "ultimo_valor"
    descripcion = "Repite el consumo del ultimo mes conocido."

    def predecir(self, panel, meta, futuros):
        return self._repetir(panel.to_numpy(float)[:, -1], futuros, panel.index)


class MediaMovil(Modelo):
    nombre = "media_3m"
    descripcion = "Promedio simple de los ultimos 3 meses."
    ventana = 3

    def predecir(self, panel, meta, futuros):
        v = panel.to_numpy(float)[:, -self.ventana:]
        return self._repetir(v.mean(axis=1), futuros, panel.index)


class MedianaMovil(Modelo):
    nombre = "mediana_3m"
    descripcion = "Mediana de los ultimos 3 meses (ignora meses atipicos)."
    ventana = 3

    def predecir(self, panel, meta, futuros):
        v = panel.to_numpy(float)[:, -self.ventana:]
        return self._repetir(np.median(v, axis=1), futuros, panel.index)


class MediaPonderada(Modelo):
    nombre = "media_movil_pond"
    descripcion = "Promedio ponderado que pesa mas los meses recientes."

    def predecir(self, panel, meta, futuros):
        pesos = np.asarray(self.cfg.get("modelos.pesos_media_movil") or [0.2, 0.3, 0.5], float)
        pesos = pesos / pesos.sum()
        k = min(len(pesos), panel.shape[1])
        v = panel.to_numpy(float)[:, -k:]
        p = pesos[-k:] / pesos[-k:].sum()
        return self._repetir(v @ p, futuros, panel.index)


class TendenciaRobusta(Modelo):
    nombre = "tendencia_robusta"
    descripcion = "Recta de tendencia por mediana de pendientes (Theil-Sen), amortiguada."
    ventana = 6

    def predecir(self, panel, meta, futuros):
        v = panel.to_numpy(float)[:, -self.ventana:]
        n = v.shape[1]
        phi = float(self.cfg.get("modelos.amortiguacion_tendencia", 0.6) or 0.0)

        if n < 2:
            return self._repetir(v[:, -1], futuros, panel.index)

        # Mediana de las pendientes de todos los pares de puntos.
        i, j = np.triu_indices(n, k=1)
        pendientes = (v[:, j] - v[:, i]) / (j - i)
        pendiente = np.median(pendientes, axis=1)

        # Nivel robusto anclado al centro de la ventana.
        posiciones = np.arange(n, dtype=float)
        intercepto = np.median(v - np.outer(pendiente, posiciones), axis=1)
        nivel = intercepto + pendiente * (n - 1)

        pasos = np.arange(1, len(futuros) + 1, dtype=float)
        amortiguado = np.cumsum(phi ** pasos) if phi > 0 else np.zeros(len(futuros))
        matriz = nivel.reshape(-1, 1) + np.outer(pendiente, amortiguado)
        return pd.DataFrame(matriz, index=panel.index, columns=futuros)


class SuavizadoHolt(Modelo):
    nombre = "suavizado_holt"
    descripcion = "Suavizado exponencial con tendencia amortiguada."
    alpha, beta = 0.5, 0.2

    def predecir(self, panel, meta, futuros):
        v = panel.to_numpy(float)
        phi = float(self.cfg.get("modelos.amortiguacion_tendencia", 0.6) or 0.0)

        nivel = v[:, 0].copy()
        tendencia = (v[:, 1] - v[:, 0]) if v.shape[1] > 1 else np.zeros(v.shape[0])

        for t in range(1, v.shape[1]):
            nivel_prev = nivel
            nivel = self.alpha * v[:, t] + (1 - self.alpha) * (nivel_prev + phi * tendencia)
            tendencia = self.beta * (nivel - nivel_prev) + (1 - self.beta) * phi * tendencia

        pasos = np.arange(1, len(futuros) + 1, dtype=float)
        amortiguado = np.cumsum(phi ** pasos) if phi > 0 else np.zeros(len(futuros))
        matriz = nivel.reshape(-1, 1) + np.outer(tendencia, amortiguado)
        return pd.DataFrame(matriz, index=panel.index, columns=futuros)


class EstacionalNaive(Modelo):
    nombre = "estacional_naive"
    descripcion = "Mismo mes del año anterior (requiere al menos 13 meses de historia)."

    def predecir(self, panel, meta, futuros):
        disponibles = {p: i for i, p in enumerate(panel.columns)}
        v = panel.to_numpy(float)
        respaldo = v[:, -3:].mean(axis=1)

        columnas = {}
        for periodo in futuros:
            hace_un_año = periodo - 12
            if hace_un_año in disponibles:
                columnas[periodo] = v[:, disponibles[hace_un_año]]
            else:
                columnas[periodo] = respaldo
                self.notas.append(f"Sin dato de {hace_un_año}; se usa promedio de 3 meses.")
        return pd.DataFrame(columnas, index=panel.index)


# ---------------------------------------------------------------------------
# Modelo global de Machine Learning
# ---------------------------------------------------------------------------
class MLGlobal(Modelo):
    nombre = "ml_global"
    descripcion = (
        "Un solo modelo entrenado con todos los clientes a la vez. Aprende "
        "patrones compartidos, por eso funciona con historias cortas."
    )

    def predecir(self, panel, meta, futuros):
        ml = self.cfg.get("machine_learning") or {}
        n_lags = int(ml.get("lags", 6))
        min_hist = int(ml.get("minimo_historia", 3))
        objetivo = ml.get("objetivo", "ratio")
        horizonte = len(futuros)
        categoricas = self.cfg.get("variables_categoricas") or []

        respaldo = MediaMovil(self.cfg).predecir(panel, meta, futuros)

        if panel.shape[1] < min_hist + 1:
            self.notas.append(
                f"Historia insuficiente ({panel.shape[1]} meses) para entrenar ML; "
                f"se usa el promedio de 3 meses."
            )
            return respaldo

        entrenamiento = car.construir_muestras(
            panel, meta, n_lags, min_hist, horizonte, categoricas
        )
        prediccion = car.construir_prediccion(
            panel, meta, n_lags, horizonte, futuros, categoricas
        )

        estrategia = ml.get("estrategia", "directa")
        salida = respaldo.copy()

        for h in range(1, horizonte + 1):
            # En modo recursivo hay un unico modelo (h=1) reutilizado.
            h_modelo = 1 if estrategia == "recursiva" else h
            datos = entrenamiento.get(h_modelo)
            if datos is None or datos.empty or len(datos) < 30:
                self.notas.append(
                    f"Pocas muestras para el horizonte {h}; se usa el promedio de 3 meses."
                )
                continue

            modelo, columnas = self._entrenar(datos, objetivo, ml)
            if modelo is None:
                continue

            X = prediccion[h][columnas]
            base = prediccion[h]["base"].to_numpy(float)
            crudo = modelo.predict(X)
            salida[futuros[h - 1]] = self._destransformar(crudo, base, objetivo, ml)

        return salida

    # -- interno ------------------------------------------------------------
    def _entrenar(self, datos: pd.DataFrame, objetivo: str, ml: dict):
        datos = datos.copy()
        base = datos["base"].to_numpy(float)
        y = datos["y"].to_numpy(float)

        if objetivo == "ratio":
            valido = base > 0
            if valido.sum() < 30:
                self.notas.append("Muy pocas series con base positiva; se omite ML.")
                return None, []
            datos, base, y = datos[valido], base[valido], y[valido]
            destino = np.clip(
                y / base,
                float(ml.get("ratio_minimo", 0.0)),
                float(ml.get("ratio_maximo", 3.0)),
            )
        elif objetivo == "log":
            destino = np.log1p(np.maximum(y, 0.0))
        else:
            destino = y

        columnas = car.columnas_modelo(datos)
        X = datos[columnas]

        hp = dict(ml.get("hiperparametros") or {})
        semilla = int(ml.get("semilla", 42))
        algoritmo = ml.get("algoritmo", "lightgbm")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            modelo = self._crear_regresor(algoritmo, hp, semilla)
            modelo.fit(X, destino)
        return modelo, columnas

    def _crear_regresor(self, algoritmo: str, hp: dict, semilla: int):
        if algoritmo == "lightgbm":
            try:
                from lightgbm import LGBMRegressor

                return LGBMRegressor(
                    objective="l1",  # error absoluto: robusto a clientes atipicos
                    random_state=semilla,
                    verbose=-1,
                    n_jobs=-1,
                    **hp,
                )
            except ImportError:
                self.notas.append("LightGBM no esta instalado; se usa scikit-learn.")

        from sklearn.ensemble import HistGradientBoostingRegressor

        return HistGradientBoostingRegressor(
            loss="absolute_error",
            max_iter=int(hp.get("n_estimators", 400)),
            learning_rate=float(hp.get("learning_rate", 0.05)),
            max_leaf_nodes=int(hp.get("num_leaves", 15)),
            min_samples_leaf=int(hp.get("min_child_samples", 20)),
            l2_regularization=float(hp.get("reg_lambda", 1.0)),
            categorical_features="from_dtype",
            random_state=semilla,
        )

    @staticmethod
    def _destransformar(crudo: np.ndarray, base: np.ndarray, objetivo: str, ml: dict) -> np.ndarray:
        if objetivo == "ratio":
            ratio = np.clip(
                crudo,
                float(ml.get("ratio_minimo", 0.0)),
                float(ml.get("ratio_maximo", 3.0)),
            )
            return ratio * base
        if objetivo == "log":
            return np.expm1(np.clip(crudo, 0, 30))
        return crudo


# ---------------------------------------------------------------------------
REGISTRO: dict[str, type[Modelo]] = {
    m.nombre: m
    for m in (
        UltimoValor,
        MediaMovil,
        MedianaMovil,
        MediaPonderada,
        TendenciaRobusta,
        SuavizadoHolt,
        EstacionalNaive,
        MLGlobal,
    )
}


def crear_modelos(cfg: Config) -> dict[str, Modelo]:
    """Instancia los modelos marcados como activos en config.yaml."""
    activos = cfg.get("modelos") or {}
    modelos: dict[str, Modelo] = {}
    for nombre, encendido in activos.items():
        if encendido is not True or nombre not in REGISTRO:
            continue
        modelos[nombre] = REGISTRO[nombre](cfg)
    return modelos
