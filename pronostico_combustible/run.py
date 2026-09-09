#!/usr/bin/env python3
"""
Punto de entrada del pronostico de consumo de combustible.

Uso tipico:
    python run.py                          # usa config.yaml
    python run.py --config otro.yaml       # otra configuracion
    python run.py --entrada datos/x.xlsx   # pisa el Excel de entrada
    python run.py --horizonte 6            # pronostica 6 meses en vez de 4
    python run.py --solo-validar           # mide errores sin generar el Excel
"""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

RAIZ = Path(__file__).resolve().parent
sys.path.insert(0, str(RAIZ))

from pronostico import cargar_config, ejecutar, escribir_excel  # noqa: E402


def parsear_argumentos() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Pronostico mensual de consumo de combustible por cliente.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--config", default="config.yaml", help="Archivo de configuracion (default: config.yaml)")
    p.add_argument("--entrada", help="Excel de entrada (pisa lo del config)")
    p.add_argument("--salida", help="Excel de salida (pisa lo del config)")
    p.add_argument("--horizonte", type=int, help="Cuantos meses pronosticar")
    p.add_argument("--ultimo-mes-real", help="Ultimo mes con datos reales, formato AAAA-MM")
    p.add_argument("--solo-validar", action="store_true", help="Solo mide errores, no escribe el Excel")
    p.add_argument("--silencioso", action="store_true", help="Sin salida en consola")
    return p.parse_args()


def aplicar_overrides(cfg, args) -> None:
    """Los argumentos de linea de comandos tienen prioridad sobre el config."""
    if args.entrada:
        cfg.dict["archivos"]["entrada"] = args.entrada
    if args.salida:
        cfg.dict["archivos"]["salida"] = args.salida
    if args.horizonte:
        cfg.dict["meses"]["horizonte"] = args.horizonte
    if args.ultimo_mes_real:
        cfg.dict["meses"]["ultimo_mes_real"] = args.ultimo_mes_real
    if args.silencioso:
        cfg.dict.setdefault("ejecucion", {})["verbosidad"] = "silencioso"


def main() -> int:
    args = parsear_argumentos()
    ruta_config = Path(args.config)
    if not ruta_config.is_absolute():
        ruta_config = RAIZ / ruta_config

    try:
        cfg = cargar_config(ruta_config)
        aplicar_overrides(cfg, args)

        verboso = not args.silencioso
        if verboso:
            print("=" * 70)
            print("  PRONOSTICO DE CONSUMO DE COMBUSTIBLE")
            print("=" * 70)

        resultado = ejecutar(cfg, verboso=verboso)

        if args.solo_validar:
            if verboso:
                print("\nModo --solo-validar: no se escribio el Excel.")
        else:
            destino = escribir_excel(resultado, cfg)
            if verboso:
                print(f"\nExcel generado: {destino}")

        if resultado.avisos and verboso:
            print("\nAvisos:")
            for aviso in dict.fromkeys(resultado.avisos):
                print(f"   - {aviso}")

        if verboso:
            print("\nListo.")
        return 0

    except (FileNotFoundError, ValueError) as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        return 1
    except Exception:
        print("\nERROR inesperado:\n", file=sys.stderr)
        traceback.print_exc()
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
