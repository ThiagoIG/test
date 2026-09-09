"""Escritura del Excel de salida, con formato y hojas de control."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from .config import Config
from .motor import Resultado

_AZUL = "#1F4E79"
_VERDE_SUAVE = "#E2EFDA"
_AMARILLO = "#FFF2CC"


def escribir_excel(res: Resultado, cfg: Config) -> Path:
    destino = cfg.ruta("archivos.salida")
    destino.parent.mkdir(parents=True, exist_ok=True)

    hojas = cfg.get("salida.hojas") or {}
    fmt_num = cfg.get("salida.formato_numero", "#,##0")

    with pd.ExcelWriter(destino, engine="xlsxwriter") as writer:
        libro = writer.book
        estilos = _crear_estilos(libro, fmt_num)

        if hojas.get("pronostico", True):
            _hoja_pronostico(writer, estilos, res, cfg)
        if hojas.get("detalle_largo", True):
            _hoja_detalle_largo(writer, estilos, res, cfg)
        if hojas.get("validacion", True):
            _hoja_validacion(writer, estilos, res)
        if hojas.get("resumen_mensual", True):
            _hoja_resumen_mensual(writer, estilos, res)
        if hojas.get("resumen_dimensiones", True):
            _hoja_resumen_dimensiones(writer, estilos, res, cfg)
        if hojas.get("configuracion", True):
            _hoja_configuracion(writer, estilos, res, cfg)

    return destino


# ---------------------------------------------------------------------------
def _crear_estilos(libro, fmt_num: str) -> dict:
    return {
        "encabezado": libro.add_format({
            "bold": True, "bg_color": _AZUL, "font_color": "white",
            "border": 1, "align": "center", "valign": "vcenter", "text_wrap": True,
        }),
        "encabezado_pron": libro.add_format({
            "bold": True, "bg_color": "#2E7D32", "font_color": "white",
            "border": 1, "align": "center", "valign": "vcenter", "text_wrap": True,
        }),
        "numero": libro.add_format({"num_format": fmt_num}),
        "numero_pron": libro.add_format({"num_format": fmt_num, "bg_color": _VERDE_SUAVE}),
        "porcentaje": libro.add_format({"num_format": "0.0%"}),
        "texto": libro.add_format({}),
        "titulo": libro.add_format({"bold": True, "font_size": 13, "font_color": _AZUL}),
        "nota": libro.add_format({"italic": True, "font_color": "#666666", "text_wrap": True}),
        "destacado": libro.add_format({"bg_color": _AMARILLO, "num_format": fmt_num}),
    }


def _escribir(writer, nombre_hoja: str, df: pd.DataFrame, estilos: dict, fila_inicio: int = 0):
    df.to_excel(writer, sheet_name=nombre_hoja, index=False, startrow=fila_inicio)
    hoja = writer.sheets[nombre_hoja]
    for col, titulo in enumerate(df.columns):
        hoja.write(fila_inicio, col, str(titulo), estilos["encabezado"])
    return hoja


def _ancho_columnas(hoja, df: pd.DataFrame, estilos: dict, cols_pronostico: set[str] | None = None):
    cols_pronostico = cols_pronostico or set()
    for i, col in enumerate(df.columns):
        muestra = df[col].astype(str).head(200)
        ancho = max(len(str(col)), int(muestra.str.len().max() or 0)) + 2
        ancho = min(max(ancho, 9), 32)

        if pd.api.types.is_numeric_dtype(df[col]):
            if str(col).lower().startswith(("error", "sesgo", "variacion", "wape")):
                formato = estilos["porcentaje"]
            elif str(col) in cols_pronostico:
                formato = estilos["numero_pron"]
            else:
                formato = estilos["numero"]
        else:
            formato = estilos["texto"]
        hoja.set_column(i, i, ancho, formato)


# ---------------------------------------------------------------------------
def _hoja_pronostico(writer, estilos, res: Resultado, cfg: Config):
    """Formato ancho, igual al Excel de entrada, con los meses nuevos agregados."""
    datos = res.datos
    cols_cfg = cfg.get("columnas") or {}

    descriptivas = [
        str(v) for v in cols_cfg.values()
        if v is not None and str(v) in datos.ancho.columns
    ]
    meses_reales = sorted(datos.cols_mes, key=lambda c: datos.cols_mes[c])
    meses_reales = [c for c in meses_reales if datos.cols_mes[c] <= datos.ultimo_real]

    df = datos.ancho[["serie_id"] + descriptivas + meses_reales].copy()

    # Agrega (o pisa) las columnas de los meses pronosticados.
    nuevas: list[str] = []
    for periodo in datos.futuros:
        nombre = datos.formato.formatear(periodo)
        while nombre in df.columns and nombre not in nuevas:
            nombre = f"{nombre} (pron)"
        df[nombre] = df["serie_id"].map(res.pronostico[periodo]).astype(float)
        nuevas.append(nombre)

    df["Total pronosticado"] = df[nuevas].sum(axis=1)
    nuevas_y_total = nuevas + ["Total pronosticado"]

    if cfg.get("salida.incluir_intervalo", True):
        error = res.diagnostico["error_validacion_wape"].reindex(df["serie_id"]).to_numpy()
        error = np.nan_to_num(error, nan=0.0)
        df["Rango minimo"] = (df["Total pronosticado"] * (1 - error)).clip(lower=0)
        df["Rango maximo"] = df["Total pronosticado"] * (1 + error)

    df["Modelo usado"] = df["serie_id"].map(res.diagnostico["modelo_elegido"])
    df["Clasificacion"] = df["serie_id"].map(res.diagnostico["clasificacion"])
    df = df.drop(columns=["serie_id"])

    hoja = _escribir(writer, "Pronostico", df, estilos)
    _ancho_columnas(hoja, df, estilos, set(nuevas_y_total) if cfg.get("salida.colorear_pronostico", True) else set())

    if cfg.get("salida.colorear_pronostico", True):
        for nombre in nuevas_y_total:
            hoja.write(0, df.columns.get_loc(nombre), nombre, estilos["encabezado_pron"])

    hoja.freeze_panes(1, min(len(descriptivas), 4))
    hoja.autofilter(0, 0, len(df), len(df.columns) - 1)


def _hoja_detalle_largo(writer, estilos, res: Resultado, cfg: Config):
    """Una fila por cliente-mes. Es la hoja para tablas dinamicas y Power BI."""
    historico = res.datos.largo.copy()
    historico["tipo"] = "Real"

    futuro = (
        res.pronostico.stack().rename("litros").reset_index()
    )
    futuro.columns = ["serie_id", "periodo", "litros"]
    futuro["tipo"] = "Pronostico"

    df = pd.concat([historico, futuro], ignore_index=True)
    df["mes"] = df["periodo"].astype(str)
    df["año"] = df["periodo"].apply(lambda p: p.year)
    df["numero_mes"] = df["periodo"].apply(lambda p: p.month)
    df = df.drop(columns=["periodo"])

    df = df.merge(res.datos.meta, left_on="serie_id", right_index=True, how="left")
    df = df.merge(
        res.diagnostico[["modelo_elegido", "clasificacion"]],
        left_on="serie_id", right_index=True, how="left",
    )

    orden = ["serie_id", "año", "numero_mes", "mes", "tipo", "litros"]
    df = df[orden + [c for c in df.columns if c not in orden]]
    df = df.sort_values(["serie_id", "año", "numero_mes"]).reset_index(drop=True)

    hoja = _escribir(writer, "Detalle_Largo", df, estilos)
    _ancho_columnas(hoja, df, estilos)
    hoja.freeze_panes(1, 0)
    hoja.autofilter(0, 0, len(df), len(df.columns) - 1)


def _hoja_validacion(writer, estilos, res: Resultado):
    """Cuanto se equivoco cada modelo, en total y por cliente."""
    hoja_nombre = "Validacion"

    ranking = res.ranking_modelos.copy()
    if not ranking.empty:
        ranking = ranking.rename(columns={
            "modelo": "Modelo", "wape": "Error (WAPE)", "mae": "MAE (litros)",
            "rmse": "RMSE", "sesgo": "Sesgo", "n_obs": "Observaciones",
        })

    detalle = res.diagnostico.reset_index().rename(columns={
        "serie_id": "Serie",
        "modelo_elegido": "Modelo elegido",
        "clasificacion": "Clasificacion",
        "error_validacion_wape": "Error validacion (WAPE)",
        "sesgo_validacion": "Sesgo validacion",
        "meses_con_consumo": "Meses con consumo",
        "promedio_3m": "Promedio 3m",
        "maximo_hist": "Maximo historico",
        "total_pronosticado": "Total pronosticado",
        "total_ultimos_meses": "Total ultimos meses reales",
        "variacion_vs_ultimos": "Variacion vs ultimos meses",
    })

    fila = 0
    libro = writer.book
    hoja = libro.add_worksheet(hoja_nombre)
    writer.sheets[hoja_nombre] = hoja

    hoja.write(fila, 0, "Comparativa global de modelos", estilos["titulo"])
    fila += 1
    hoja.write(
        fila, 0,
        "WAPE = litros de error / litros reales. Mas bajo es mejor. "
        "Sesgo positivo = el modelo sobreestima.",
        estilos["nota"],
    )
    fila += 2

    if not ranking.empty:
        ranking.to_excel(writer, sheet_name=hoja_nombre, index=False, startrow=fila)
        for col, titulo in enumerate(ranking.columns):
            hoja.write(fila, col, str(titulo), estilos["encabezado"])
        fila += len(ranking) + 3
    else:
        hoja.write(fila, 0, "Sin historia suficiente para validar.", estilos["nota"])
        fila += 2

    hoja.write(fila, 0, "Detalle por cliente", estilos["titulo"])
    fila += 2
    detalle.to_excel(writer, sheet_name=hoja_nombre, index=False, startrow=fila)
    for col, titulo in enumerate(detalle.columns):
        hoja.write(fila, col, str(titulo), estilos["encabezado"])

    _ancho_columnas(hoja, detalle, estilos)
    hoja.autofilter(fila, 0, fila + len(detalle), len(detalle.columns) - 1)
    hoja.freeze_panes(fila + 1, 1)


def _hoja_resumen_mensual(writer, estilos, res: Resultado):
    """Totales de la cartera mes a mes: lo real y lo proyectado."""
    real = res.datos.largo.groupby("periodo", observed=True)["litros"].sum()
    pron = res.pronostico.sum(axis=0)

    df = pd.DataFrame({
        "Mes": [str(p) for p in list(real.index) + list(pron.index)],
        "Tipo": ["Real"] * len(real) + ["Pronostico"] * len(pron),
        "Litros": list(real.to_numpy()) + list(pron.to_numpy()),
    })
    df["Clientes con consumo"] = (
        list(
            res.datos.largo[res.datos.largo["litros"] > 0]
            .groupby("periodo", observed=True)["serie_id"].nunique()
            .reindex(real.index, fill_value=0).to_numpy()
        )
        + list((res.pronostico > 0).sum(axis=0).to_numpy())
    )
    df["Variacion mensual"] = df["Litros"].pct_change()

    hoja = _escribir(writer, "Resumen_Mensual", df, estilos)
    _ancho_columnas(hoja, df, estilos)

    grafico = writer.book.add_chart({"type": "column"})
    grafico.add_series({
        "name": "Litros",
        "categories": ["Resumen_Mensual", 1, 0, len(df), 0],
        "values": ["Resumen_Mensual", 1, 2, len(df), 2],
    })
    grafico.set_title({"name": "Consumo mensual: real y pronosticado"})
    grafico.set_y_axis({"name": "Litros"})
    grafico.set_size({"width": 900, "height": 380})
    hoja.insert_chart(2, len(df.columns) + 1, grafico)


def _hoja_resumen_dimensiones(writer, estilos, res: Resultado, cfg: Config):
    """Totales pronosticados abiertos por cada dimension del negocio."""
    hoja_nombre = "Resumen_Dimensiones"
    libro = writer.book
    hoja = libro.add_worksheet(hoja_nombre)
    writer.sheets[hoja_nombre] = hoja

    dimensiones = [
        d for d in ("solucion", "segmento", "industria", "portfolio", "grupo_cliente")
        if d in res.datos.meta.columns
    ]

    total = res.pronostico.sum(axis=1).rename("Total pronosticado")
    ultimos = res.diagnostico["total_ultimos_meses"]

    fila = 0
    hoja.write(fila, 0, "Pronostico por dimension", estilos["titulo"])
    fila += 2

    for dim in dimensiones:
        etiquetas = res.datos.meta[dim].reindex(total.index).astype("string").fillna("(sin dato)")
        agrupado = pd.DataFrame({
            dim.capitalize(): etiquetas,
            "Total pronosticado": total,
            "Total ultimos meses reales": ultimos.reindex(total.index),
        }).groupby(dim.capitalize(), observed=True).agg(
            Clientes=("Total pronosticado", "size"),
            Total_pronosticado=("Total pronosticado", "sum"),
            Total_ultimos_reales=("Total ultimos meses reales", "sum"),
        ).reset_index()

        agrupado["Variacion"] = np.where(
            agrupado["Total_ultimos_reales"] > 0,
            agrupado["Total_pronosticado"] / agrupado["Total_ultimos_reales"] - 1,
            np.nan,
        )
        agrupado = agrupado.rename(columns={
            "Total_pronosticado": "Total pronosticado",
            "Total_ultimos_reales": "Total ultimos meses reales",
        }).sort_values("Total pronosticado", ascending=False)

        agrupado.to_excel(writer, sheet_name=hoja_nombre, index=False, startrow=fila)
        for col, titulo in enumerate(agrupado.columns):
            hoja.write(fila, col, str(titulo), estilos["encabezado"])
        fila += len(agrupado) + 3

    hoja.set_column(0, 0, 28)
    hoja.set_column(1, 1, 10, estilos["numero"])
    hoja.set_column(2, 3, 20, estilos["numero"])
    hoja.set_column(4, 4, 12, estilos["porcentaje"])


def _hoja_configuracion(writer, estilos, res: Resultado, cfg: Config):
    """Deja registro de con que parametros se genero este archivo."""
    hoja_nombre = "Configuracion"
    libro = writer.book
    hoja = libro.add_worksheet(hoja_nombre)
    writer.sheets[hoja_nombre] = hoja

    hoja.set_column(0, 0, 110)
    fila = 0
    hoja.write(fila, 0, "Parametros usados en esta corrida", estilos["titulo"])
    fila += 2

    hoja.write(fila, 0, f"Ultimo mes real: {res.datos.ultimo_real}")
    fila += 1
    hoja.write(fila, 0, f"Meses pronosticados: {', '.join(str(p) for p in res.datos.futuros)}")
    fila += 1
    hoja.write(fila, 0, f"Series procesadas: {len(res.pronostico):,}")
    fila += 2

    if res.avisos:
        hoja.write(fila, 0, "Avisos", estilos["titulo"])
        fila += 1
        for aviso in dict.fromkeys(res.avisos):
            hoja.write(fila, 0, f"- {aviso}", estilos["nota"])
            fila += 1
        fila += 1

    hoja.write(fila, 0, "config.yaml", estilos["titulo"])
    fila += 1
    texto = yaml.safe_dump(cfg.dict, allow_unicode=True, sort_keys=False, default_flow_style=False)
    for linea in texto.splitlines():
        hoja.write(fila, 0, linea)
        fila += 1
