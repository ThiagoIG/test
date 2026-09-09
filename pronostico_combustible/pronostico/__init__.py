"""
Pronostico de consumo de combustible (litros/mes) por cliente.

Uso rapido:
    from pronostico import cargar_config, ejecutar, escribir_excel

    cfg = cargar_config("config.yaml")
    resultado = ejecutar(cfg)
    escribir_excel(resultado, cfg)
"""

from .config import Config, cargar_config
from .motor import Resultado, ejecutar
from .salida import escribir_excel

__all__ = ["Config", "cargar_config", "Resultado", "ejecutar", "escribir_excel"]
__version__ = "1.0.0"
