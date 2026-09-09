"""Construccion de variables predictoras para el modelo global de ML."""

from __future__ import annotations

import numpy as np
import pandas as pd


def a_panel(largo: pd.DataFrame, periodos: list[pd.Period] | None = None) -> pd.DataFrame:
    """Pasa el formato largo a una matriz series x meses (mas comodo para lags)."""
    panel = largo.pivot_table(
        index="serie_id", columns="periodo", values="litros", aggfunc="sum", observed=True
    )
    if periodos is not None:
        panel = panel.reindex(columns=periodos)
    panel = panel.reindex(columns=sorted(panel.columns))
    return panel.fillna(0.0)


def _pendiente(bloque: np.ndarray) -> np.ndarray:
    """Pendiente de la recta de minimos cuadrados de cada fila."""
    n = bloque.shape[1]
    if n < 2:
        return np.zeros(bloque.shape[0])
    x = np.arange(n, dtype=float)
    x_centrado = x - x.mean()
    denominador = (x_centrado ** 2).sum()
    y_centrado = bloque - bloque.mean(axis=1, keepdims=True)
    return (y_centrado * x_centrado).sum(axis=1) / denominador


def _meses_desde_ultimo_consumo(historia: np.ndarray) -> np.ndarray:
    """Cuantos meses hace que la serie no registra litros (0 = consumio el ultimo mes)."""
    n_series, n_meses = historia.shape
    if n_meses == 0:
        return np.full(n_series, 99.0)
    activo = historia > 0
    # Posicion del ultimo mes con consumo; -1 si nunca consumio.
    indices = np.where(activo.any(axis=1), n_meses - 1 - activo[:, ::-1].argmax(axis=1), -1)
    return np.where(indices < 0, 99.0, (n_meses - 1 - indices).astype(float))


def features_en_corte(
    panel_valores: np.ndarray,
    corte: int,
    n_lags: int,
) -> dict[str, np.ndarray]:
    """
    Variables calculadas con la informacion disponible hasta la columna `corte`
    (inclusive). No mira nada posterior: eso es lo que evita el data leakage.
    """
    historia = panel_valores[:, : corte + 1]
    n_disponibles = historia.shape[1]
    f: dict[str, np.ndarray] = {}

    # --- Rezagos (consumo de los meses previos) -------------------------------
    for k in range(1, n_lags + 1):
        col = corte - (k - 1)
        f[f"lag_{k}"] = panel_valores[:, col] if col >= 0 else np.zeros(panel_valores.shape[0])

    # --- Promedios y dispersion de ventanas recientes -------------------------
    for ventana in (3, 6, 12):
        bloque = historia[:, -ventana:]
        f[f"media_{ventana}m"] = bloque.mean(axis=1)
        f[f"mediana_{ventana}m"] = np.median(bloque, axis=1)
        f[f"max_{ventana}m"] = bloque.max(axis=1)
        f[f"min_{ventana}m"] = bloque.min(axis=1)
        f[f"desvio_{ventana}m"] = bloque.std(axis=1)

    # --- Nivel historico completo --------------------------------------------
    f["media_hist"] = historia.mean(axis=1)
    f["max_hist"] = historia.max(axis=1)
    f["suma_hist"] = historia.sum(axis=1)
    f["meses_historia"] = np.full(panel_valores.shape[0], float(n_disponibles))

    # --- Volatilidad relativa -------------------------------------------------
    media3 = np.maximum(f["media_3m"], 1e-9)
    f["cv_3m"] = f["desvio_3m"] / media3
    f["cv_hist"] = historia.std(axis=1) / np.maximum(f["media_hist"], 1e-9)

    # --- Tendencia ------------------------------------------------------------
    f["pendiente_3m"] = _pendiente(historia[:, -3:])
    f["pendiente_6m"] = _pendiente(historia[:, -6:])
    f["pendiente_rel_6m"] = f["pendiente_6m"] / np.maximum(f["media_6m"], 1e-9)

    # --- Momento (aceleracion o freno reciente) -------------------------------
    f["momento_1_3"] = f["lag_1"] / media3
    f["momento_3_6"] = f["media_3m"] / np.maximum(f["media_6m"], 1e-9)

    # --- Actividad / inactividad ---------------------------------------------
    f["meses_activos"] = (historia > 0).sum(axis=1).astype(float)
    f["prop_ceros"] = 1.0 - f["meses_activos"] / max(n_disponibles, 1)
    f["meses_sin_consumo"] = _meses_desde_ultimo_consumo(historia)

    return f


def base_normalizacion(features: dict[str, np.ndarray]) -> np.ndarray:
    """
    Escala de referencia del cliente, usada cuando el modelo predice un ratio.
    Combina el promedio de 3 meses con el ultimo valor para no depender de un
    unico mes atipico.
    """
    base = 0.7 * features["media_3m"] + 0.3 * features["lag_1"]
    respaldo = np.maximum(features["media_hist"], 0.0)
    return np.where(base > 0, base, respaldo)


def construir_muestras(
    panel: pd.DataFrame,
    meta: pd.DataFrame,
    n_lags: int,
    minimo_historia: int,
    horizonte: int,
    cols_categoricas: list[str],
) -> dict[int, pd.DataFrame]:
    """
    Arma el set de entrenamiento para cada horizonte h (1..horizonte).

    Para cada mes objetivo t y cada h, las variables se calculan en el corte
    t-h. El resultado es un dict {h: DataFrame con X, y, base y serie_id}.
    """
    valores = panel.to_numpy(dtype=float)
    periodos = list(panel.columns)
    n_meses = len(periodos)
    series = panel.index.to_numpy()

    cat = _preparar_categoricas(meta, panel.index, cols_categoricas)
    muestras: dict[int, list[pd.DataFrame]] = {h: [] for h in range(1, horizonte + 1)}

    for h in range(1, horizonte + 1):
        for corte in range(minimo_historia - 1, n_meses - h):
            destino = corte + h
            f = features_en_corte(valores, corte, n_lags)
            df = pd.DataFrame(f)
            df["horizonte"] = float(h)
            df["mes_objetivo"] = float(periodos[destino].month)
            df["mes_corte"] = float(periodos[corte].month)
            df["serie_id"] = series
            df["base"] = base_normalizacion(f)
            df["y"] = valores[:, destino]
            for nombre, serie_cat in cat.items():
                df[nombre] = serie_cat
            muestras[h].append(df)

    return {
        h: (pd.concat(partes, ignore_index=True) if partes else pd.DataFrame())
        for h, partes in muestras.items()
    }


def construir_prediccion(
    panel: pd.DataFrame,
    meta: pd.DataFrame,
    n_lags: int,
    horizonte: int,
    periodos_futuros: list[pd.Period],
    cols_categoricas: list[str],
) -> dict[int, pd.DataFrame]:
    """Variables para predecir cada mes futuro, calculadas en el ultimo mes real."""
    valores = panel.to_numpy(dtype=float)
    corte = valores.shape[1] - 1
    cat = _preparar_categoricas(meta, panel.index, cols_categoricas)
    mes_corte = float(panel.columns[-1].month)

    f = features_en_corte(valores, corte, n_lags)
    base = base_normalizacion(f)

    salida: dict[int, pd.DataFrame] = {}
    for h in range(1, horizonte + 1):
        df = pd.DataFrame(f)
        df["horizonte"] = float(h)
        df["mes_objetivo"] = float(periodos_futuros[h - 1].month)
        df["mes_corte"] = mes_corte
        df["serie_id"] = panel.index.to_numpy()
        df["base"] = base
        for nombre, serie_cat in cat.items():
            df[nombre] = serie_cat
        salida[h] = df
    return salida


def _preparar_categoricas(
    meta: pd.DataFrame, indice: pd.Index, columnas: list[str]
) -> dict[str, pd.Categorical]:
    """
    Alinea las columnas categoricas con el panel.

    Las categorias se derivan de `meta` COMPLETO (no del subconjunto alineado),
    para que entrenamiento y prediccion compartan exactamente el mismo
    vocabulario. Si no, el modelo recibe codigos que significan cosas distintas.
    """
    resultado: dict[str, pd.Categorical] = {}
    for col in columnas:
        if col not in meta.columns:
            continue
        completa = meta[col].astype("string").fillna("(sin dato)")
        categorias = pd.Index(sorted(completa.unique()))
        valores = completa.reindex(indice).fillna("(sin dato)")
        resultado[f"cat_{col}"] = pd.Categorical(valores, categories=categorias)
    return resultado


def columnas_modelo(df: pd.DataFrame) -> list[str]:
    """Columnas que entran al modelo (todo menos identificadores y objetivo)."""
    excluir = {"serie_id", "y", "base"}
    return [c for c in df.columns if c not in excluir]
