"""
Backtesting: simula el pronostico en el pasado para medir cuanto se erra.

La logica es "origen movil": se corta la historia en un mes anterior, se
pronostican los meses siguientes (que si conocemos) y se compara. Ningun
modelo ve datos posteriores al corte.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import Config
from .modelos import Modelo

_MIN_MESES_ENTRENAMIENTO = 3


def generar_ventanas(n_meses: int, horizonte: int, ventanas: int) -> list[int]:
    """Devuelve los indices de corte utilizables, del mas reciente al mas viejo."""
    cortes = []
    for k in range(ventanas):
        corte = n_meses - horizonte - k
        if corte < _MIN_MESES_ENTRENAMIENTO:
            break
        cortes.append(corte)
    return cortes


def ejecutar_backtest(
    panel: pd.DataFrame,
    meta: pd.DataFrame,
    modelos: dict[str, Modelo],
    cfg: Config,
) -> pd.DataFrame:
    """Devuelve un DataFrame largo: serie_id | modelo | ventana | periodo | real | pred."""
    horizonte = int(cfg.get("meses.horizonte", 4))
    n_ventanas = int(cfg.get("validacion.ventanas", 2))
    periodos = list(panel.columns)
    cortes = generar_ventanas(len(periodos), horizonte, n_ventanas)

    if not cortes:
        return pd.DataFrame(
            columns=["serie_id", "modelo", "ventana", "periodo", "real", "pred"]
        )

    filas = []
    for n_ventana, corte in enumerate(cortes, start=1):
        historia = panel.iloc[:, :corte]
        objetivo_periodos = periodos[corte : corte + horizonte]
        reales = panel[objetivo_periodos]

        for nombre, modelo in modelos.items():
            pred = modelo.predecir(historia, meta, objetivo_periodos)
            pred = pred.reindex(index=panel.index, columns=objetivo_periodos)

            bloque = pd.DataFrame(
                {
                    "serie_id": np.repeat(panel.index.to_numpy(), len(objetivo_periodos)),
                    "modelo": nombre,
                    "ventana": n_ventana,
                    "periodo": np.tile(objetivo_periodos, len(panel.index)),
                    "real": reales.to_numpy(float).ravel(),
                    "pred": pred.to_numpy(float).ravel(),
                }
            )
            filas.append(bloque)

    resultado = pd.concat(filas, ignore_index=True)
    resultado["pred"] = resultado["pred"].fillna(0.0)
    if cfg.get("reglas.no_negativos", True):
        resultado["pred"] = resultado["pred"].clip(lower=0.0)
    return resultado


def calcular_metricas(bt: pd.DataFrame, por: list[str]) -> pd.DataFrame:
    """WAPE, MAE, RMSE y sesgo agregados por las columnas indicadas."""
    if bt.empty:
        return pd.DataFrame(columns=por + ["wape", "mae", "rmse", "sesgo", "n_obs"])

    d = bt.copy()
    d["error"] = d["pred"] - d["real"]
    d["abs_error"] = d["error"].abs()
    d["sq_error"] = d["error"] ** 2

    g = d.groupby(por, observed=True)
    m = g.agg(
        suma_abs=("abs_error", "sum"),
        suma_real=("real", "sum"),
        suma_error=("error", "sum"),
        mae=("abs_error", "mean"),
        mse=("sq_error", "mean"),
        n_obs=("real", "size"),
    ).reset_index()

    # WAPE: error absoluto total sobre litros totales. Es la metrica que le
    # importa al negocio (un 5% de error en un cliente grande pesa mas).
    m["wape"] = np.where(m["suma_real"] > 0, m["suma_abs"] / m["suma_real"], np.nan)
    m["rmse"] = np.sqrt(m["mse"])
    m["sesgo"] = np.where(m["suma_real"] > 0, m["suma_error"] / m["suma_real"], np.nan)

    return m[por + ["wape", "mae", "rmse", "sesgo", "n_obs"]]


def elegir_modelos(
    bt: pd.DataFrame, cfg: Config, series: pd.Index
) -> tuple[pd.Series, pd.DataFrame]:
    """
    Decide que modelo usa cada serie.

    Devuelve (serie con el modelo elegido por serie_id, tabla de metricas).
    """
    respaldo = cfg.get("validacion.modelo_respaldo", "media_3m")
    modo = cfg.get("validacion.seleccion", "por_cliente")
    metrica = cfg.get("validacion.metrica", "wape")
    margen = float(cfg.get("validacion.margen_mejora", 0.05) or 0.0)

    if modo == "fijo" or bt.empty:
        fijo = cfg.get("validacion.modelo_fijo", respaldo) if modo == "fijo" else respaldo
        return pd.Series(fijo, index=series, name="modelo_elegido"), pd.DataFrame()

    por_serie = calcular_metricas(bt, ["serie_id", "modelo"])

    if modo == "global":
        globales = calcular_metricas(bt, ["modelo"]).dropna(subset=[metrica])
        if globales.empty:
            ganador = respaldo
        else:
            ganador = globales.loc[globales[metrica].idxmin(), "modelo"]
        return pd.Series(ganador, index=series, name="modelo_elegido"), por_serie

    # --- Seleccion por cliente ------------------------------------------------
    tabla = por_serie.pivot(index="serie_id", columns="modelo", values=metrica)
    tabla = tabla.reindex(series)

    if respaldo not in tabla.columns:
        tabla[respaldo] = np.nan

    error_respaldo = tabla[respaldo]

    # Las series sin litros reales en la ventana de validacion (clientes dados
    # de baja, por ejemplo) tienen WAPE indefinido en TODOS los modelos. Hay que
    # excluirlas antes de buscar el minimo: idxmin sobre una fila toda-NA falla.
    mejor_modelo = pd.Series(index=tabla.index, dtype=object)
    mejor_error = pd.Series(np.nan, index=tabla.index, dtype=float)
    evaluables = ~tabla.isna().all(axis=1)
    if evaluables.any():
        comparables = tabla.loc[evaluables]
        mejor_modelo.loc[evaluables] = comparables.idxmin(axis=1)
        mejor_error.loc[evaluables] = comparables.min(axis=1)

    # Solo se cambia de modelo si la mejora supera el margen exigido.
    # Con historias cortas, una mejora chica suele ser ruido, no señal.
    umbral = error_respaldo * (1 - margen)
    conviene_cambiar = mejor_error.notna() & error_respaldo.notna() & (mejor_error < umbral)

    elegido = np.where(conviene_cambiar, mejor_modelo, respaldo)
    # Si el respaldo no tiene metrica (serie sin datos reales), usa el mejor disponible.
    sin_referencia = error_respaldo.isna() & mejor_modelo.notna()
    elegido = np.where(sin_referencia, mejor_modelo, elegido)

    serie_elegida = pd.Series(elegido, index=tabla.index, name="modelo_elegido")
    return serie_elegida.reindex(series).fillna(respaldo), por_serie


def resumen_por_modelo(bt: pd.DataFrame) -> pd.DataFrame:
    """Ranking global de modelos, para el reporte en consola."""
    if bt.empty:
        return pd.DataFrame()
    return calcular_metricas(bt, ["modelo"]).sort_values("wape").reset_index(drop=True)
