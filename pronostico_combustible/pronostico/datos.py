"""Lectura del Excel, deteccion de columnas de meses y paso a formato largo."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

import pandas as pd

from .config import Config

# --------------------------------------------------------------------------
# Diccionario de meses (español e ingles, con y sin abreviatura)
# --------------------------------------------------------------------------
_MESES = {
    "ene": 1, "enero": 1, "jan": 1, "january": 1,
    "feb": 2, "febrero": 2, "february": 2,
    "mar": 3, "marzo": 3, "march": 3,
    "abr": 4, "abril": 4, "apr": 4, "april": 4,
    "may": 5, "mayo": 5,
    "jun": 6, "junio": 6, "june": 6,
    "jul": 7, "julio": 7, "july": 7,
    "ago": 8, "agosto": 8, "aug": 8, "august": 8,
    "sep": 9, "sept": 9, "set": 9, "setiembre": 9, "septiembre": 9, "september": 9,
    "oct": 10, "octubre": 10, "october": 10,
    "nov": 11, "noviembre": 11, "november": 11,
    "dic": 12, "diciembre": 12, "dec": 12, "december": 12,
}

_ABREV_ES = {
    1: "Ene", 2: "Feb", 3: "Mar", 4: "Abr", 5: "May", 6: "Jun",
    7: "Jul", 8: "Ago", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dic",
}
_COMPLETO_ES = {
    1: "Enero", 2: "Febrero", 3: "Marzo", 4: "Abril", 5: "Mayo", 6: "Junio",
    7: "Julio", 8: "Agosto", 9: "Septiembre", 10: "Octubre", 11: "Noviembre",
    12: "Diciembre",
}

_NOMBRES_MES = "|".join(sorted(_MESES, key=len, reverse=True))
_SEP = r"[\s\-_/.]*"

_RE_NOMBRE_AÑO = re.compile(rf"^({_NOMBRES_MES}){_SEP}(\d{{2}}|\d{{4}})$")
_RE_AÑO_NOMBRE = re.compile(rf"^(\d{{4}}){_SEP}({_NOMBRES_MES})$")
_RE_AÑO_NUM = re.compile(r"^(\d{4})[\s\-_/.](\d{1,2})$")
_RE_NUM_AÑO = re.compile(r"^(\d{1,2})[\s\-_/.](\d{2}|\d{4})$")
_RE_COMPACTO = re.compile(r"^(\d{4})(\d{2})$")


def _normalizar(texto: str) -> str:
    """Minusculas, sin acentos, sin espacios sobrantes."""
    t = unicodedata.normalize("NFKD", str(texto))
    t = "".join(c for c in t if not unicodedata.combining(c))
    return t.strip().lower()


def _expandir_año(y: int) -> int:
    """25 -> 2025. Ventana 1970-2069, suficiente para datos comerciales."""
    if y >= 100:
        return y
    return 2000 + y if y < 70 else 1900 + y


def interpretar_mes(etiqueta) -> pd.Period | None:
    """Convierte el encabezado de una columna en un periodo mensual, o None."""
    if etiqueta is None or (isinstance(etiqueta, float) and pd.isna(etiqueta)):
        return None

    if isinstance(etiqueta, (datetime, date, pd.Timestamp)):
        return pd.Period(year=etiqueta.year, month=etiqueta.month, freq="M")

    txt = _normalizar(etiqueta)
    if not txt:
        return None

    if m := _RE_NOMBRE_AÑO.match(txt):
        return pd.Period(year=_expandir_año(int(m.group(2))), month=_MESES[m.group(1)], freq="M")

    if m := _RE_AÑO_NOMBRE.match(txt):
        return pd.Period(year=int(m.group(1)), month=_MESES[m.group(2)], freq="M")

    if m := _RE_AÑO_NUM.match(txt):
        mes = int(m.group(2))
        if 1 <= mes <= 12:
            return pd.Period(year=int(m.group(1)), month=mes, freq="M")

    if m := _RE_NUM_AÑO.match(txt):
        mes = int(m.group(1))
        if 1 <= mes <= 12:
            return pd.Period(year=_expandir_año(int(m.group(2))), month=mes, freq="M")

    if m := _RE_COMPACTO.match(txt):
        mes = int(m.group(2))
        if 1 <= mes <= 12:
            return pd.Period(year=int(m.group(1)), month=mes, freq="M")

    return None


# --------------------------------------------------------------------------
# Reproduccion del estilo de encabezado para las columnas nuevas
# --------------------------------------------------------------------------
@dataclass
class FormatoMes:
    """Recuerda como se ven tus columnas para escribir las nuevas igual."""

    estilo: str = "abrev"      # "abrev" | "completo" | "numerico"
    separador: str = "-"
    digitos_año: int = 2
    mayuscula: str = "titulo"  # "titulo" | "alta" | "baja"
    año_primero: bool = False

    @classmethod
    def inferir(cls, etiqueta) -> "FormatoMes":
        if isinstance(etiqueta, (datetime, date, pd.Timestamp)):
            return cls(estilo="numerico", separador="-", digitos_año=4, año_primero=True)

        crudo = str(etiqueta).strip()
        txt = _normalizar(crudo)

        sep_hallado = re.search(r"[\s\-_/.]", crudo)
        separador = sep_hallado.group(0) if sep_hallado else ""

        if m := _RE_AÑO_NUM.match(txt):
            return cls("numerico", separador or "-", len(m.group(1)), año_primero=True)
        if m := _RE_NUM_AÑO.match(txt):
            return cls("numerico", separador or "-", len(m.group(2)), año_primero=False)
        if _RE_COMPACTO.match(txt):
            return cls("numerico", "", 4, año_primero=True)

        m = _RE_NOMBRE_AÑO.match(txt) or _RE_AÑO_NOMBRE.match(txt)
        if m:
            año_primero = bool(_RE_AÑO_NOMBRE.match(txt))
            token = m.group(2) if año_primero else m.group(1)
            año = m.group(1) if año_primero else m.group(2)
            estilo = "completo" if len(token) > 4 else "abrev"

            # Capitalizacion tal como aparece en el archivo original.
            pos = crudo.lower().find(token[:3])
            fragmento = crudo[pos:pos + len(token)] if pos >= 0 else token
            if fragmento.isupper():
                mayuscula = "alta"
            elif fragmento.islower():
                mayuscula = "baja"
            else:
                mayuscula = "titulo"

            return cls(estilo, separador or "-", len(año), mayuscula, año_primero)

        return cls()

    def formatear(self, periodo: pd.Period) -> str:
        año = f"{periodo.year % 100:02d}" if self.digitos_año == 2 else f"{periodo.year}"

        if self.estilo == "numerico":
            mes = f"{periodo.month:02d}"
            return f"{año}{self.separador}{mes}" if self.año_primero else f"{mes}{self.separador}{año}"

        nombre = (_COMPLETO_ES if self.estilo == "completo" else _ABREV_ES)[periodo.month]
        if self.mayuscula == "alta":
            nombre = nombre.upper()
        elif self.mayuscula == "baja":
            nombre = nombre.lower()

        return f"{año}{self.separador}{nombre}" if self.año_primero else f"{nombre}{self.separador}{año}"


# --------------------------------------------------------------------------
# Resultado de la carga
# --------------------------------------------------------------------------
@dataclass
class DatosCargados:
    ancho: pd.DataFrame                    # tal cual el Excel, con clave_serie agregada
    largo: pd.DataFrame                    # serie_id | periodo | litros
    meta: pd.DataFrame                     # serie_id + columnas descriptivas
    cols_mes: dict[str, pd.Period]         # nombre columna -> periodo
    formato: FormatoMes
    ultimo_real: pd.Period
    futuros: list[pd.Period]
    avisos: list[str]


def cargar_datos(cfg: Config) -> DatosCargados:
    ruta = cfg.ruta("archivos.entrada")
    if not ruta.exists():
        raise FileNotFoundError(
            f"No encuentro el Excel de entrada: {ruta}\n"
            f"   Revisa 'archivos.entrada' en config.yaml, o genera datos de "
            f"ejemplo con:  python datos/generar_ejemplo.py"
        )

    hoja = cfg.get("archivos.hoja", "auto")
    fila = int(cfg.get("archivos.fila_encabezado", 1) or 1)
    df = pd.read_excel(
        ruta,
        sheet_name=0 if hoja in (None, "auto") else hoja,
        header=fila - 1,
    )
    df.columns = [str(c).strip() if not isinstance(c, (datetime, date, pd.Timestamp)) else c
                  for c in df.columns]

    avisos: list[str] = []
    cols_mes = _detectar_meses(df, cfg, avisos)
    if not cols_mes:
        raise ValueError(
            "No pude identificar ninguna columna de mes.\n"
            "   Pon 'meses.deteccion: manual' en config.yaml y completa 'mapa_manual'.\n"
            f"   Columnas encontradas: {list(df.columns)}"
        )

    formato = FormatoMes.inferir(max(cols_mes, key=lambda c: cols_mes[c]))
    meta, df = _construir_meta(df, cfg, avisos)
    largo = _a_formato_largo(df, cols_mes, cfg)

    ultimo_real = _resolver_ultimo_real(cfg, largo, cols_mes, avisos)
    largo = largo[largo["periodo"] <= ultimo_real].copy()

    horizonte = int(cfg.get("meses.horizonte", 4))
    futuros = [ultimo_real + i for i in range(1, horizonte + 1)]

    return DatosCargados(df, largo, meta, cols_mes, formato, ultimo_real, futuros, avisos)


def _detectar_meses(df: pd.DataFrame, cfg: Config, avisos: list[str]) -> dict[str, pd.Period]:
    if cfg.get("meses.deteccion") == "manual":
        mapa = cfg.get("meses.mapa_manual") or {}
        if not mapa:
            raise ValueError(
                "'meses.deteccion' es 'manual' pero 'meses.mapa_manual' esta vacio."
            )
        resultado: dict[str, pd.Period] = {}
        for col, periodo in mapa.items():
            if col not in df.columns:
                avisos.append(f"'mapa_manual' menciona la columna '{col}', que no existe en el Excel.")
                continue
            resultado[col] = pd.Period(str(periodo), freq="M")
        return resultado

    # Deteccion automatica: solo columnas que ademas sean numericas.
    descriptivas = {
        str(v) for v in (cfg.get("columnas") or {}).values() if v is not None
    }
    resultado = {}
    for col in df.columns:
        if str(col) in descriptivas:
            continue
        periodo = interpretar_mes(col)
        if periodo is None:
            continue
        if not pd.api.types.is_numeric_dtype(df[col]):
            convertida = pd.to_numeric(df[col], errors="coerce")
            if convertida.notna().sum() == 0 and df[col].notna().sum() > 0:
                avisos.append(f"La columna '{col}' parece un mes pero no tiene valores numericos; se ignora.")
                continue
        resultado[str(col)] = periodo

    duplicados = pd.Series(list(resultado.values())).duplicated(keep=False)
    if duplicados.any():
        repes = sorted({str(p) for p, d in zip(resultado.values(), duplicados) if d})
        avisos.append(f"Hay columnas distintas que apuntan al mismo mes: {repes}. Se suman.")

    return resultado


def _construir_meta(df: pd.DataFrame, cfg: Config, avisos: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    cols_cfg = cfg.get("columnas") or {}
    clave = cfg.get("clave_serie") or ["id_cliente"]

    faltantes = [c for c in clave if str(cols_cfg.get(c)) not in df.columns]
    if faltantes:
        raise ValueError(
            f"Faltan en el Excel columnas de 'clave_serie': "
            f"{[cols_cfg.get(c) for c in faltantes]}\n"
            f"   Columnas disponibles: {list(df.columns)}"
        )

    df = df.copy()
    partes = [df[str(cols_cfg[c])].astype(str).str.strip() for c in clave]
    df["serie_id"] = partes[0]
    for p in partes[1:]:
        df["serie_id"] = df["serie_id"] + " | " + p

    cols_meta = ["serie_id"]
    for logico, real in cols_cfg.items():
        if real is not None and str(real) in df.columns:
            df[f"_{logico}"] = df[str(real)]
            cols_meta.append(f"_{logico}")

    meta = df[cols_meta].drop_duplicates(subset="serie_id").set_index("serie_id")
    meta.columns = [c.lstrip("_") for c in meta.columns]

    # La fecha de primer consumo se normaliza a periodo mensual. Es el dato
    # autoritativo del alta: evita tener que inferirla del primer mes con litros.
    if "fecha_primer_consumo" in meta.columns:
        meta["fecha_primer_consumo"] = meta["fecha_primer_consumo"].map(interpretar_mes)
        sin_fecha = int(meta["fecha_primer_consumo"].isna().sum())
        if sin_fecha:
            avisos.append(
                f"{sin_fecha} series sin 'fecha primer consumo' legible; "
                f"para esas se infiere el alta del primer mes con litros."
            )

    dups = df["serie_id"].duplicated().sum()
    if dups:
        avisos.append(
            f"Hay {dups} filas con la misma clave de serie ({' + '.join(clave)}). "
            f"Se suman sus litros. Revisa si es lo esperado."
        )
    return meta, df


def _a_formato_largo(df: pd.DataFrame, cols_mes: dict[str, pd.Period], cfg: Config) -> pd.DataFrame:
    largo = df.melt(
        id_vars=["serie_id"],
        value_vars=list(cols_mes),
        var_name="columna",
        value_name="litros",
    )
    largo["periodo"] = largo["columna"].map(cols_mes)
    largo["litros"] = pd.to_numeric(largo["litros"], errors="coerce")

    if cfg.get("meses.celdas_vacias", "cero") == "cero":
        largo["litros"] = largo["litros"].fillna(0.0)
    else:
        largo = largo.dropna(subset=["litros"])

    largo = (
        largo.groupby(["serie_id", "periodo"], as_index=False, observed=True)["litros"]
        .sum()
        .sort_values(["serie_id", "periodo"])
        .reset_index(drop=True)
    )
    return largo


def _resolver_ultimo_real(
    cfg: Config, largo: pd.DataFrame, cols_mes: dict[str, pd.Period], avisos: list[str]
) -> pd.Period:
    configurado = cfg.get("meses.ultimo_mes_real", "auto")

    if configurado not in (None, "auto"):
        periodo = pd.Period(str(configurado), freq="M")
        if periodo not in set(cols_mes.values()):
            avisos.append(
                f"'ultimo_mes_real' = {periodo} no corresponde a ninguna columna del Excel."
            )
        return periodo

    # Auto: ultimo mes en el que ALGUN cliente tiene consumo distinto de cero.
    con_datos = largo[largo["litros"].fillna(0) != 0]
    if con_datos.empty:
        return max(cols_mes.values())
    return con_datos["periodo"].max()


def escribir_plantilla_ejemplo(destino: Path) -> None:  # pragma: no cover - utilitario
    destino.parent.mkdir(parents=True, exist_ok=True)
