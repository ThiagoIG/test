#!/usr/bin/env python3
"""
Genera un Excel de ejemplo con la MISMA estructura de columnas que tu archivo
real, para poder probar todo el circuito antes de enchufar los datos de
produccion.

    python datos/generar_ejemplo.py
    python datos/generar_ejemplo.py --clientes 800 --semilla 7

Los patrones son sinteticos pero realistas: estacionalidad por industria,
tendencias, clientes nuevos, clientes que se dan de baja y meses atipicos.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

AQUI = Path(__file__).resolve().parent

SOLUCIONES = ["Combustible Edenred", "Shell Flota"]

SEGMENTOS = {  # nombre -> (litros/mes tipicos minimo, maximo)
    "Micro":    (200, 1_200),
    "Pequeño":  (1_200, 5_000),
    "Mediano":  (5_000, 20_000),
    "Grande":   (20_000, 80_000),
    "Corporate": (80_000, 300_000),
}
PESOS_SEGMENTO = [0.30, 0.32, 0.22, 0.12, 0.04]

# Perfil estacional por industria (indice 0 = enero). Hemisferio sur.
INDUSTRIAS = {
    "Transporte de carga": [0.92, 0.95, 1.05, 1.02, 1.00, 0.98, 1.00, 1.02, 1.03, 1.06, 1.08, 1.05],
    "Agro":                [0.85, 0.90, 1.15, 1.25, 1.20, 0.85, 0.75, 0.80, 1.10, 1.25, 1.20, 0.95],
    "Construccion":        [1.05, 1.08, 1.10, 1.02, 0.95, 0.85, 0.82, 0.88, 1.00, 1.08, 1.12, 1.10],
    "Mineria":             [1.00, 1.00, 1.02, 1.01, 1.00, 0.99, 0.98, 1.00, 1.01, 1.02, 1.01, 0.98],
    "Servicios":           [0.90, 0.98, 1.04, 1.03, 1.02, 1.00, 1.00, 1.02, 1.03, 1.04, 1.05, 0.92],
    "Gobierno":            [0.80, 0.92, 1.05, 1.05, 1.03, 1.02, 1.00, 1.03, 1.05, 1.06, 1.05, 0.88],
    "Retail y distribucion": [0.88, 0.92, 1.00, 0.98, 0.98, 0.97, 0.99, 1.01, 1.03, 1.08, 1.15, 1.20],
}

GRUPOS = [
    "Gobierno de Mendoza", "Grupo Andes", "Grupo Cuyo Logistica", "Corporacion del Valle",
    "Grupo San Rafael", "Holding Patagonia", "Sin grupo", "Sin grupo", "Sin grupo",
]

ABREV = {1: "Ene", 2: "Feb", 3: "Mar", 4: "Abr", 5: "May", 6: "Jun",
         7: "Jul", 8: "Ago", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dic"}


def generar(n_clientes: int, semilla: int, ultimo_mes: str, meses_historia: int) -> pd.DataFrame:
    rng = np.random.default_rng(semilla)
    fin = pd.Period(ultimo_mes, freq="M")
    periodos = [fin - i for i in range(meses_historia - 1, -1, -1)]

    filas = []
    for i in range(n_clientes):
        id_cliente = f"C{10000 + i}"
        nombre = f"Cliente {id_cliente}"
        industria = rng.choice(list(INDUSTRIAS))
        segmento = rng.choice(list(SEGMENTOS), p=PESOS_SEGMENTO)
        grupo = rng.choice(GRUPOS)

        # Un mismo cliente puede tener 1 o 2 soluciones contratadas.
        n_soluciones = 1 if rng.random() < 0.78 else 2
        soluciones = rng.choice(SOLUCIONES, size=n_soluciones, replace=False)

        for solucion in soluciones:
            filas.append(
                _generar_serie(rng, id_cliente, nombre, grupo, segmento, industria,
                               solucion, periodos, fin)
            )

    df = pd.DataFrame(filas)
    orden = ["Solucion", "ID", "Cliente", "Cliente Grupo", "Segmento", "Industria",
             "Portfolio", "Fecha primer consumo"]
    meses = [f"{ABREV[p.month]}-{p.year % 100:02d}" for p in periodos]
    return df[orden + meses]


def _generar_serie(rng, id_cliente, nombre, grupo, segmento, industria, solucion,
                   periodos, fin) -> dict:
    minimo, maximo = SEGMENTOS[segmento]
    nivel = float(rng.uniform(minimo, maximo))
    estacional = INDUSTRIAS[industria]

    # Antiguedad -> Portfolio, y desde que mes empieza a consumir.
    dado = rng.random()
    if dado < 0.12:                                   # alta este año
        portfolio = "NS N"
        inicio = int(rng.integers(max(len(periodos) - 8, 0), len(periodos) - 1))
    elif dado < 0.28:                                 # alta el año pasado
        portfolio = "NS N-1"
        inicio = int(rng.integers(max(len(periodos) - 20, 0), max(len(periodos) - 9, 1)))
    else:
        portfolio = "Portfolio"
        inicio = 0

    tendencia_mensual = float(rng.normal(0.004, 0.012))   # crecimiento/caida suave
    volatilidad = float(rng.uniform(0.06, 0.22))

    # Algunos clientes dejan de operar en algun momento.
    baja = len(periodos)
    if rng.random() < 0.06:
        baja = int(rng.integers(len(periodos) - 6, len(periodos)))

    alta = periodos[min(inicio, len(periodos) - 1)]
    fila = {
        "Solucion": solucion,
        "ID": id_cliente,
        "Cliente": nombre,
        "Cliente Grupo": grupo,
        "Segmento": segmento,
        "Industria": industria,
        "Portfolio": portfolio,
        "Fecha primer consumo": f"{alta.year}-{alta.month:02d}",
    }

    for t, periodo in enumerate(periodos):
        etiqueta = f"{ABREV[periodo.month]}-{periodo.year % 100:02d}"

        if t < inicio:
            fila[etiqueta] = np.nan          # todavia no era cliente
            continue
        if t >= baja:
            fila[etiqueta] = 0.0             # cliente dado de baja
            continue

        valor = (
            nivel
            * estacional[periodo.month - 1]
            * (1 + tendencia_mensual) ** (t - inicio)
            * rng.lognormal(0, volatilidad)
        )
        if rng.random() < 0.03:              # mes atipico (parada, siniestro, pico)
            valor *= rng.choice([0.25, 0.4, 1.8, 2.3])
        if rng.random() < 0.02:              # mes sin operacion
            valor = 0.0

        fila[etiqueta] = round(max(valor, 0.0), 1)

    return fila


def main() -> None:
    p = argparse.ArgumentParser(description="Genera un Excel de ejemplo.")
    p.add_argument("--clientes", type=int, default=400, help="Cantidad de clientes (default 400)")
    p.add_argument("--semilla", type=int, default=42)
    p.add_argument("--ultimo-mes", default="2026-08", help="Ultimo mes con datos reales (AAAA-MM)")
    p.add_argument("--meses", type=int, default=32, help="Meses de historia a generar")
    p.add_argument("--salida", default=str(AQUI / "consumo_clientes.xlsx"))
    args = p.parse_args()

    df = generar(args.clientes, args.semilla, args.ultimo_mes, args.meses)
    destino = Path(args.salida)
    destino.parent.mkdir(parents=True, exist_ok=True)

    with pd.ExcelWriter(destino, engine="xlsxwriter") as writer:
        df.to_excel(writer, sheet_name="Consumo", index=False)
        hoja = writer.sheets["Consumo"]
        hoja.set_column(0, 7, 22)
        hoja.set_column(8, len(df.columns) - 1, 11)
        hoja.freeze_panes(1, 8)

    print(f"Excel de ejemplo generado: {destino}")
    print(f"   {len(df):,} filas (cliente + solucion)")
    print(f"   {len(df.columns) - 8} columnas de meses: "
          f"{df.columns[8]} ... {df.columns[-1]}")


if __name__ == "__main__":
    main()
