"""Orquestador: carga, valida, entrena, pronostica y aplica reglas de negocio."""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import caracteristicas as car
from . import validacion as val
from .config import Config
from .datos import DatosCargados, cargar_datos
from .modelos import crear_modelos


@dataclass
class Resultado:
    datos: DatosCargados
    pronostico: pd.DataFrame           # series x meses futuros (litros finales)
    pronostico_crudo: pd.DataFrame     # antes de las reglas de negocio
    diagnostico: pd.DataFrame          # una fila por serie: modelo, error, clasificacion
    backtest: pd.DataFrame
    ranking_modelos: pd.DataFrame
    avisos: list[str] = field(default_factory=list)


def ejecutar(cfg: Config, verboso: bool = True) -> Resultado:
    log = _crear_log(cfg, verboso)

    log("Leyendo datos...")
    datos = cargar_datos(cfg)
    avisos = list(datos.avisos)

    panel = car.a_panel(datos.largo)
    n_series, n_meses = panel.shape
    log(f"   {n_series:,} series | {n_meses} meses de historia "
        f"({panel.columns[0]} a {panel.columns[-1]})")
    log(f"   Ultimo mes real: {datos.ultimo_real}")
    log(f"   A pronosticar:   {', '.join(str(p) for p in datos.futuros)}")

    modelos = crear_modelos(cfg)
    log(f"\nModelos en competencia: {', '.join(modelos)}")

    log("\nValidando (backtesting)...")
    backtest = val.ejecutar_backtest(panel, datos.meta, modelos, cfg)
    ranking = val.resumen_por_modelo(backtest)

    if backtest.empty:
        avisos.append(
            "No hubo historia suficiente para validar. Se usa el modelo de respaldo "
            "sin comparacion. Considera reducir 'meses.horizonte' o 'validacion.ventanas'."
        )
        log("   ! Historia insuficiente para validar; se usa el modelo de respaldo.")
    else:
        for _, fila in ranking.iterrows():
            log(f"   {fila['modelo']:<20} WAPE {fila['wape']:6.1%}   sesgo {fila['sesgo']:+6.1%}")

    elegidos, metricas_serie = val.elegir_modelos(backtest, cfg, panel.index)
    log(f"\nModelos elegidos: " + ", ".join(
        f"{k} ({v})" for k, v in elegidos.value_counts().items()
    ))

    log("\nGenerando pronostico...")
    predicciones = {
        nombre: modelo.predecir(panel, datos.meta, datos.futuros).reindex(
            index=panel.index, columns=datos.futuros
        )
        for nombre, modelo in modelos.items()
    }
    for nombre, modelo in modelos.items():
        for nota in dict.fromkeys(modelo.notas):
            avisos.append(f"[{nombre}] {nota}")

    crudo = _ensamblar(predicciones, elegidos, panel.index, datos.futuros)

    log("Aplicando reglas de negocio...")
    final, diagnostico = _aplicar_reglas(crudo, panel, datos, cfg, elegidos, avisos)

    diagnostico = _completar_diagnostico(
        diagnostico, metricas_serie, elegidos, panel, final, datos
    )

    log(f"\nTotal pronosticado: {final.to_numpy().sum():,.0f} litros "
        f"en {len(datos.futuros)} meses")
    return Resultado(datos, final, crudo, diagnostico, backtest, ranking, avisos)


# ---------------------------------------------------------------------------
def _ensamblar(
    predicciones: dict[str, pd.DataFrame],
    elegidos: pd.Series,
    indice: pd.Index,
    futuros: list[pd.Period],
) -> pd.DataFrame:
    """Toma de cada serie la fila del modelo que gano su validacion."""
    salida = pd.DataFrame(0.0, index=indice, columns=futuros)
    for nombre, pred in predicciones.items():
        mascara = (elegidos == nombre).reindex(indice, fill_value=False)
        if mascara.any():
            salida.loc[mascara] = pred.loc[mascara].to_numpy(float)
    return salida


def _aplicar_reglas(
    crudo: pd.DataFrame,
    panel: pd.DataFrame,
    datos: DatosCargados,
    cfg: Config,
    elegidos: pd.Series,
    avisos: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    reglas = cfg.get("reglas") or {}
    valores = panel.to_numpy(float)
    activa = _mascara_actividad(panel, cfg, datos.meta)
    final = crudo.copy()

    diag = pd.DataFrame(index=panel.index)
    diag["meses_con_consumo"] = (valores > 0).sum(axis=1)
    diag["consumo_total_hist"] = valores.sum(axis=1)
    # El promedio se calcula solo sobre los meses en que ya era cliente.
    # Una serie sin ningun mes activo da una ventana vacia: el NaN es correcto
    # y se convierte a 0, no es un error que haya que mostrar.
    ventana = np.where(activa[:, -3:], valores[:, -3:], np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        diag["promedio_3m"] = np.nan_to_num(np.nanmean(ventana, axis=1))
    diag["maximo_hist"] = valores.max(axis=1)
    diag["clasificacion"] = "normal"

    # --- Clientes inactivos ---------------------------------------------------
    n_cero = int(reglas.get("meses_cero_para_inactivo", 0) or 0)
    if n_cero > 0 and valores.shape[1] >= n_cero:
        inactivos = (valores[:, -n_cero:] == 0).all(axis=1)
        diag.loc[inactivos, "clasificacion"] = "inactivo"
        final.loc[inactivos] = 0.0

    # --- Clientes nuevos (poca historia) --------------------------------------
    cfg_nuevos = reglas.get("clientes_nuevos") or {}
    min_meses = int(cfg_nuevos.get("minimo_meses", 0) or 0)
    if min_meses > 0:
        nuevos = (diag["meses_con_consumo"] < min_meses) & (diag["clasificacion"] != "inactivo")
        if nuevos.any():
            mapa, etiquetas = _perfil_de_grupo(
                panel, datos, cfg_nuevos.get("agrupar_por") or [], activa
            )
            diag.loc[nuevos, "clasificacion"] = "nuevo"
            final.loc[nuevos] = _pronostico_para_nuevos(
                panel.loc[nuevos], mapa, etiquetas, datos
            )
            avisos.append(
                f"{int(nuevos.sum())} series con menos de {min_meses} meses de consumo "
                f"se pronosticaron con el perfil promedio de su grupo."
            )

    # --- Topes de crecimiento y caida ----------------------------------------
    tope = reglas.get("tope_multiplo_maximo")
    if tope:
        limite = np.maximum(diag["maximo_hist"].to_numpy(float) * float(tope), 0.0)
        normales = (diag["clasificacion"] == "normal").to_numpy()
        recorte = np.minimum(final.to_numpy(float), limite.reshape(-1, 1))
        final.loc[normales] = recorte[normales]

    piso = reglas.get("piso_multiplo_minimo")
    if piso:
        positivos = np.where(valores > 0, valores, np.nan)
        with np.errstate(all="ignore"):
            minimo = np.nanmin(positivos, axis=1)
        minimo = np.nan_to_num(minimo, nan=0.0) * float(piso)
        normales = (diag["clasificacion"] == "normal").to_numpy()
        elevado = np.maximum(final.to_numpy(float), minimo.reshape(-1, 1))
        final.loc[normales] = elevado[normales]

    # --- Factores estacionales manuales --------------------------------------
    factores = reglas.get("factores_estacionales") or {}
    for periodo in final.columns:
        factor = factores.get(f"{periodo.month:02d}")
        if factor is not None and float(factor) != 1.0:
            final[periodo] = final[periodo] * float(factor)

    # --- Ajuste global --------------------------------------------------------
    ajuste = float(reglas.get("ajuste_global", 1.0) or 1.0)
    if ajuste != 1.0:
        final = final * ajuste

    # --- Ajustes manuales por cliente ----------------------------------------
    final = _aplicar_ajustes_manuales(final, cfg, datos, diag, avisos)

    # --- Saneamiento final ----------------------------------------------------
    if reglas.get("no_negativos", True):
        final = final.clip(lower=0.0)

    decimales = reglas.get("redondeo")
    if decimales is not None:
        final = final.round(int(decimales))

    return final, diag


def _mascara_actividad(
    panel: pd.DataFrame, cfg: Config, meta: pd.DataFrame | None = None
) -> np.ndarray:
    """Meses en que cada serie ya era cliente. Ver 'meses.previo_al_alta'."""
    if cfg.get("meses.previo_al_alta", "no_es_cliente") == "cero":
        return np.ones(panel.shape, dtype=bool)
    return car.mascara_actividad(panel, meta)


def _etiquetas_grupo(
    meta: pd.DataFrame, indice: pd.Index, agrupar_por: list[str]
) -> pd.Series:
    """
    Etiqueta de grupo para cada serie. Es la MISMA funcion que usan el calculo
    del perfil y su aplicacion: si difirieran, las claves no coincidirian y
    todos los factores caerian silenciosamente a 1.0.
    """
    columnas = [c for c in agrupar_por if c in meta.columns]
    if not columnas:
        return pd.Series("(todos)", index=indice, dtype=object)
    etiquetas = (
        meta.reindex(indice)[columnas]
        .astype("string")
        .fillna("(sin dato)")
        .agg(" | ".join, axis=1)
    )
    return etiquetas.astype(object)


def _perfil_de_grupo(
    panel: pd.DataFrame, datos: DatosCargados, agrupar_por: list[str],
    activa: np.ndarray | None = None,
) -> tuple[dict[tuple[str, int], float], pd.Series]:
    """
    Indice estacional por grupo: cuanto consume el grupo en cada mes calendario
    respecto de su propio promedio anual. Sirve para darle forma al pronostico
    de un cliente que todavia no tiene historia propia.

    Devuelve (mapa {(grupo, mes) -> factor}, etiquetas por serie).
    """
    etiquetas = _etiquetas_grupo(datos.meta, panel.index, agrupar_por)

    largo = panel.stack().rename("litros").reset_index()
    largo.columns = ["serie_id", "periodo", "litros"]

    if activa is not None:
        # Los meses previos al alta no describen la estacionalidad del grupo:
        # incluirlos como ceros aplanaria el perfil de las industrias con muchas
        # altas recientes.
        vigente = pd.DataFrame(activa, index=panel.index, columns=panel.columns)
        largo = largo[vigente.stack().to_numpy()]

    # Cada serie se normaliza por su propio promedio: asi un cliente grande y uno
    # chico aportan por igual a la FORMA estacional del grupo.
    promedio = largo.groupby("serie_id", observed=True)["litros"].transform("mean")
    largo["relativo"] = np.where(promedio > 0, largo["litros"] / promedio, np.nan)
    largo["mes"] = largo["periodo"].apply(lambda p: p.month)
    largo["_grupo"] = largo["serie_id"].map(etiquetas)

    perfil = (
        largo.dropna(subset=["relativo"])
        .groupby(["_grupo", "mes"], observed=True)["relativo"]
        .median()
    )
    mapa = {(g, int(m)): float(v) for (g, m), v in perfil.items() if np.isfinite(v) and v > 0}
    return mapa, etiquetas


def _pronostico_para_nuevos(
    panel_nuevos: pd.DataFrame,
    mapa: dict[tuple[str, int], float],
    etiquetas: pd.Series,
    datos: DatosCargados,
) -> np.ndarray:
    """
    Nivel propio del cliente (lo poco que tenga) llevado a base anual y luego
    proyectado con la forma estacional de su grupo.

    Corregir por la estacionalidad de los meses observados importa: si un cliente
    se dio de alta en un mes pico, tomar su promedio crudo como nivel anual
    sobreestima todo el resto del año.
    """
    valores = panel_nuevos.to_numpy(float)
    meses_hist = [p.month for p in panel_nuevos.columns]
    etiquetas_nuevos = etiquetas.reindex(panel_nuevos.index).fillna("(todos)")

    salida = np.zeros((len(panel_nuevos), len(datos.futuros)))

    for i, (serie, grupo) in enumerate(zip(panel_nuevos.index, etiquetas_nuevos)):
        observados = valores[i] > 0
        if not observados.any():
            continue

        factores_obs = np.array(
            [mapa.get((grupo, meses_hist[j]), 1.0) for j in np.flatnonzero(observados)],
            dtype=float,
        )
        nivel_bruto = valores[i][observados].mean()
        ajuste = factores_obs.mean()
        nivel_anual = nivel_bruto / ajuste if ajuste > 0 else nivel_bruto

        for j, periodo in enumerate(datos.futuros):
            salida[i, j] = nivel_anual * mapa.get((grupo, periodo.month), 1.0)

    return salida


def _aplicar_ajustes_manuales(
    final: pd.DataFrame,
    cfg: Config,
    datos: DatosCargados,
    diag: pd.DataFrame,
    avisos: list[str],
) -> pd.DataFrame:
    conf = (cfg.get("reglas.ajustes_manuales") or {})
    hoja = conf.get("hoja")
    if not hoja:
        return final

    try:
        ajustes = pd.read_excel(cfg.ruta("archivos.entrada"), sheet_name=hoja)
    except Exception as exc:  # hoja inexistente o ilegible
        avisos.append(f"No pude leer la hoja de ajustes manuales '{hoja}': {exc}")
        return final

    cols_cfg = cfg.get("columnas") or {}
    clave = cfg.get("clave_serie") or ["id_cliente"]
    faltantes = [cols_cfg.get(c) for c in clave if str(cols_cfg.get(c)) not in ajustes.columns]
    if faltantes or "Mes" not in ajustes.columns:
        avisos.append(
            f"La hoja '{hoja}' debe tener las columnas "
            f"{[cols_cfg.get(c) for c in clave]} + 'Mes' (+ 'Factor' y/o 'Litros')."
        )
        return final

    serie_id = ajustes[str(cols_cfg[clave[0]])].astype(str).str.strip()
    for campo in clave[1:]:
        serie_id = serie_id + " | " + ajustes[str(cols_cfg[campo])].astype(str).str.strip()
    ajustes["serie_id"] = serie_id
    ajustes["periodo"] = ajustes["Mes"].apply(lambda v: pd.Period(str(v), freq="M"))

    aplicados = 0
    for _, fila in ajustes.iterrows():
        sid, periodo = fila["serie_id"], fila["periodo"]
        if sid not in final.index or periodo not in final.columns:
            avisos.append(f"Ajuste manual ignorado (serie o mes inexistente): {sid} / {periodo}")
            continue
        litros = fila.get("Litros")
        factor = fila.get("Factor")
        if pd.notna(litros):
            final.loc[sid, periodo] = float(litros)
        elif pd.notna(factor):
            final.loc[sid, periodo] *= float(factor)
        else:
            continue
        aplicados += 1

    if aplicados:
        avisos.append(f"Se aplicaron {aplicados} ajustes manuales desde la hoja '{hoja}'.")
    return final


def _completar_diagnostico(
    diag: pd.DataFrame,
    metricas_serie: pd.DataFrame,
    elegidos: pd.Series,
    panel: pd.DataFrame,
    final: pd.DataFrame,
    datos: DatosCargados,
) -> pd.DataFrame:
    diag = diag.copy()
    diag.insert(0, "modelo_elegido", elegidos.reindex(diag.index))

    if not metricas_serie.empty:
        clave = pd.MultiIndex.from_arrays(
            [diag.index, diag["modelo_elegido"].astype(str)]
        )
        m = metricas_serie.set_index(["serie_id", "modelo"])
        diag["error_validacion_wape"] = m["wape"].reindex(clave).to_numpy()
        diag["sesgo_validacion"] = m["sesgo"].reindex(clave).to_numpy()
    else:
        diag["error_validacion_wape"] = np.nan
        diag["sesgo_validacion"] = np.nan

    diag["total_pronosticado"] = final.sum(axis=1)
    ultimos = panel.to_numpy(float)[:, -len(datos.futuros):]
    diag["total_ultimos_meses"] = ultimos.sum(axis=1)
    with np.errstate(all="ignore"):
        diag["variacion_vs_ultimos"] = np.where(
            diag["total_ultimos_meses"] > 0,
            diag["total_pronosticado"] / diag["total_ultimos_meses"] - 1,
            np.nan,
        )

    return diag.join(datos.meta, how="left")


def _crear_log(cfg: Config, verboso: bool):
    nivel = cfg.get("ejecucion.verbosidad", "normal")
    activo = verboso and nivel != "silencioso"

    def log(mensaje: str = "") -> None:
        if activo:
            print(mensaje)

    return log
