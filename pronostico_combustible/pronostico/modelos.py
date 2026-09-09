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
from contextlib import contextmanager
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
        valores = np.nan_to_num(valores, nan=0.0, posinf=0.0, neginf=0.0)
        matriz = np.repeat(valores.reshape(-1, 1), len(futuros), axis=1)
        return pd.DataFrame(matriz, index=indice, columns=futuros)

    def _mascara(self, panel: pd.DataFrame, meta: pd.DataFrame | None = None) -> np.ndarray:
        """Meses en que cada serie ya era cliente. Ver 'meses.previo_al_alta'."""
        if self.cfg.get("meses.previo_al_alta", "no_es_cliente") == "cero":
            return np.ones(panel.shape, dtype=bool)
        return car.mascara_actividad(panel, meta)

    def _ventana_activa(
        self, panel: pd.DataFrame, k: int, meta: pd.DataFrame | None = None
    ) -> np.ndarray:
        """
        Ultimos k meses, con NaN en los previos al alta del cliente.

        Sin esto, un cliente dado de alta hace 2 meses ve su promedio de 3 meses
        dividido por 3 en vez de por 2: se subestima sistematicamente a todas
        las altas recientes.
        """
        valores = panel.to_numpy(float)
        activa = self._mascara(panel, meta)
        k = min(k, valores.shape[1])
        return np.where(activa[:, -k:], valores[:, -k:], np.nan)

    @staticmethod
    @contextmanager
    def _sin_avisos_de_nan():
        """Las ventanas enteramente previas al alta dan NaN; es el resultado correcto."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            yield


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
        v = self._ventana_activa(panel, self.ventana, meta)
        with self._sin_avisos_de_nan():
            nivel = np.nanmean(v, axis=1)
        return self._repetir(nivel, futuros, panel.index)


class MedianaMovil(Modelo):
    nombre = "mediana_3m"
    descripcion = "Mediana de los ultimos 3 meses (ignora meses atipicos)."
    ventana = 3

    def predecir(self, panel, meta, futuros):
        v = self._ventana_activa(panel, self.ventana, meta)
        with self._sin_avisos_de_nan():
            nivel = np.nanmedian(v, axis=1)
        return self._repetir(nivel, futuros, panel.index)


class MediaPonderada(Modelo):
    nombre = "media_movil_pond"
    descripcion = "Promedio ponderado que pesa mas los meses recientes."

    def predecir(self, panel, meta, futuros):
        pesos = np.asarray(self.cfg.get("modelos.pesos_media_movil") or [0.2, 0.3, 0.5], float)
        k = min(len(pesos), panel.shape[1])
        p = pesos[-k:]

        v = self._ventana_activa(panel, k, meta)
        presente = ~np.isnan(v)

        # Los pesos se renormalizan sobre los meses efectivamente disponibles,
        # para no diluir el promedio de un cliente de alta reciente.
        numerador = np.nansum(np.where(presente, v, 0.0) * p, axis=1)
        denominador = (presente * p).sum(axis=1)
        nivel = np.divide(
            numerador, denominador,
            out=np.zeros_like(numerador), where=denominador > 0,
        )
        return self._repetir(nivel, futuros, panel.index)


class TendenciaRobusta(Modelo):
    nombre = "tendencia_robusta"
    descripcion = "Recta de tendencia por mediana de pendientes (Theil-Sen), amortiguada."
    ventana = 6

    def predecir(self, panel, meta, futuros):
        v = self._ventana_activa(panel, self.ventana, meta)
        n = v.shape[1]
        phi = float(self.cfg.get("modelos.amortiguacion_tendencia", 0.6) or 0.0)

        if n < 2:
            return self._repetir(np.nan_to_num(v[:, -1]), futuros, panel.index)

        with self._sin_avisos_de_nan():
            # Mediana de las pendientes de todos los pares de puntos. Los pares
            # que tocan un mes previo al alta dan NaN y quedan descartados: si no,
            # un cliente nuevo mostraria una pendiente altisima que solo refleja
            # el salto de "no era cliente" a "es cliente".
            i, j = np.triu_indices(n, k=1)
            pendientes = (v[:, j] - v[:, i]) / (j - i)
            pendiente = np.nanmedian(pendientes, axis=1)
            pendiente = np.nan_to_num(pendiente)

            # Nivel robusto anclado al final de la ventana.
            posiciones = np.arange(n, dtype=float)
            intercepto = np.nanmedian(v - np.outer(pendiente, posiciones), axis=1)

        nivel = np.nan_to_num(intercepto) + pendiente * (n - 1)

        pasos = np.arange(1, len(futuros) + 1, dtype=float)
        amortiguado = np.cumsum(phi ** pasos) if phi > 0 else np.zeros(len(futuros))
        matriz = nivel.reshape(-1, 1) + np.outer(pendiente, amortiguado)
        return pd.DataFrame(
            np.nan_to_num(matriz), index=panel.index, columns=futuros
        )


class SuavizadoHolt(Modelo):
    nombre = "suavizado_holt"
    descripcion = "Suavizado exponencial con tendencia amortiguada."
    alpha, beta = 0.5, 0.2

    def predecir(self, panel, meta, futuros):
        v = panel.to_numpy(float)
        activa = self._mascara(panel, meta)
        phi = float(self.cfg.get("modelos.amortiguacion_tendencia", 0.6) or 0.0)
        n_series, n_meses = v.shape

        # El suavizado arranca en el mes de alta de cada cliente, no en la
        # primera columna del panel: los meses previos no son ceros que suavizar.
        primer_activo = activa.argmax(axis=1)
        nunca_activo = ~activa.any(axis=1)
        nivel = v[np.arange(n_series), primer_activo].copy()
        nivel[nunca_activo] = 0.0
        tendencia = np.zeros(n_series)

        for t in range(1, n_meses):
            # Solo se actualiza en los meses posteriores al alta de cada serie.
            actualizar = activa[:, t] & (t > primer_activo)
            nivel_prev = nivel
            nivel_nuevo = self.alpha * v[:, t] + (1 - self.alpha) * (nivel_prev + phi * tendencia)
            tendencia_nueva = (
                self.beta * (nivel_nuevo - nivel_prev) + (1 - self.beta) * phi * tendencia
            )
            nivel = np.where(actualizar, nivel_nuevo, nivel_prev)
            tendencia = np.where(actualizar, tendencia_nueva, tendencia)

        pasos = np.arange(1, len(futuros) + 1, dtype=float)
        amortiguado = np.cumsum(phi ** pasos) if phi > 0 else np.zeros(len(futuros))
        matriz = nivel.reshape(-1, 1) + np.outer(tendencia, amortiguado)
        return pd.DataFrame(np.nan_to_num(matriz), index=panel.index, columns=futuros)


class EstacionalNaive(Modelo):
    nombre = "estacional_naive"
    descripcion = "Mismo mes del año anterior (requiere al menos 13 meses de historia)."

    def predecir(self, panel, meta, futuros):
        disponibles = {p: i for i, p in enumerate(panel.columns)}
        v = panel.to_numpy(float)
        activa = self._mascara(panel, meta)

        with self._sin_avisos_de_nan():
            respaldo = np.nan_to_num(
                np.nanmean(self._ventana_activa(panel, 3, meta), axis=1)
            )

        columnas = {}
        for periodo in futuros:
            hace_un_año = periodo - 12
            if hace_un_año not in disponibles:
                columnas[periodo] = respaldo
                self.notas.append(f"Sin dato de {hace_un_año}; se usa promedio de 3 meses.")
                continue

            col = disponibles[hace_un_año]
            # Para un cliente que hace un año todavia no existia, ese mes no es
            # un cero estacional: hay que caer al promedio reciente.
            columnas[periodo] = np.where(activa[:, col], v[:, col], respaldo)

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

        usar_mascara = self.cfg.get("meses.previo_al_alta", "no_es_cliente") != "cero"
        entrenamiento = car.construir_muestras(
            panel, meta, n_lags, min_hist, horizonte, categoricas, usar_mascara
        )
        prediccion = car.construir_prediccion(
            panel, meta, n_lags, horizonte, futuros, categoricas, usar_mascara
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
