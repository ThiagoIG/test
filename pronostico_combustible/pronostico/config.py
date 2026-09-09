"""Carga y validacion de config.yaml."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


class Config:
    """Acceso comodo a la configuracion con rutas tipo 'modelos.ml_global'."""

    def __init__(self, datos: dict[str, Any], ruta_base: Path):
        self._d = datos
        self.ruta_base = ruta_base

    def __getitem__(self, ruta: str) -> Any:
        return self.get(ruta)

    def get(self, ruta: str, defecto: Any = None) -> Any:
        nodo: Any = self._d
        for parte in ruta.split("."):
            if not isinstance(nodo, dict) or parte not in nodo:
                return defecto
            nodo = nodo[parte]
        return nodo

    def ruta(self, clave: str) -> Path:
        """Resuelve una ruta de archivo relativa a la ubicacion del config."""
        valor = self.get(clave)
        if valor is None:
            raise ValueError(f"Falta la ruta '{clave}' en config.yaml")
        p = Path(valor)
        return p if p.is_absolute() else (self.ruta_base / p)

    @property
    def dict(self) -> dict[str, Any]:
        return self._d


# Claves obligatorias y sus valores admitidos (validacion temprana = errores claros).
_OPCIONES_VALIDAS = {
    "meses.deteccion": {"auto", "manual"},
    "meses.celdas_vacias": {"cero", "nulo"},
    "machine_learning.algoritmo": {"lightgbm", "sklearn"},
    "machine_learning.objetivo": {"ratio", "log", "absoluto"},
    "machine_learning.estrategia": {"directa", "recursiva"},
    "validacion.metrica": {"wape", "mae", "rmse"},
    "validacion.seleccion": {"por_cliente", "global", "fijo"},
    "ejecucion.verbosidad": {"silencioso", "normal", "detallado"},
}


def cargar_config(ruta: str | Path) -> Config:
    ruta = Path(ruta).expanduser().resolve()
    if not ruta.exists():
        raise FileNotFoundError(f"No encuentro el archivo de configuracion: {ruta}")

    with open(ruta, "r", encoding="utf-8") as fh:
        datos = yaml.safe_load(fh) or {}

    cfg = Config(datos, ruta_base=ruta.parent)
    _validar(cfg)
    return cfg


def _validar(cfg: Config) -> None:
    errores: list[str] = []

    for clave, validas in _OPCIONES_VALIDAS.items():
        valor = cfg.get(clave)
        if valor is not None and valor not in validas:
            errores.append(
                f"config.yaml -> '{clave}' = '{valor}' no es valido. "
                f"Opciones: {sorted(validas)}"
            )

    if not cfg.get("columnas.id_cliente"):
        errores.append("config.yaml -> 'columnas.id_cliente' es obligatorio.")

    clave_serie = cfg.get("clave_serie") or []
    if not clave_serie:
        errores.append("config.yaml -> 'clave_serie' no puede estar vacia.")
    for campo in clave_serie:
        if cfg.get(f"columnas.{campo}") is None:
            errores.append(
                f"config.yaml -> 'clave_serie' incluye '{campo}' pero "
                f"'columnas.{campo}' no esta definido."
            )

    horizonte = cfg.get("meses.horizonte")
    if not isinstance(horizonte, int) or horizonte < 1:
        errores.append("config.yaml -> 'meses.horizonte' debe ser un entero >= 1.")

    activos = [n for n, on in (cfg.get("modelos") or {}).items() if on is True]
    if not activos:
        errores.append("config.yaml -> no hay ningun modelo activo en 'modelos'.")

    respaldo = cfg.get("validacion.modelo_respaldo")
    if respaldo and cfg.get(f"modelos.{respaldo}") is not True:
        errores.append(
            f"config.yaml -> 'validacion.modelo_respaldo' = '{respaldo}' "
            f"pero ese modelo no esta activo en 'modelos'."
        )

    if errores:
        raise ValueError("Errores en la configuracion:\n  - " + "\n  - ".join(errores))
