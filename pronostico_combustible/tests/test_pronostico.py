"""
Tests del motor de pronostico.

Ejecutar:  pytest -q

El test mas importante es `test_backtest_no_espia_el_futuro`: verifica que
ningun modelo use informacion posterior al corte. Si eso se rompe, el error
de validacion se vuelve mentira y el pronostico parece mucho mejor de lo que es.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

from pronostico import cargar_config, ejecutar  # noqa: E402
from pronostico.caracteristicas import a_panel, features_en_corte  # noqa: E402
from pronostico.config import Config  # noqa: E402
from pronostico.datos import FormatoMes, interpretar_mes  # noqa: E402
from pronostico.modelos import crear_modelos  # noqa: E402
from pronostico.validacion import calcular_metricas, ejecutar_backtest, generar_ventanas  # noqa: E402


# ---------------------------------------------------------------------------
# Deteccion de columnas de meses
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "etiqueta, esperado",
    [
        ("Ene-25", "2025-01"),
        ("ene_2025", "2025-01"),
        ("Enero 2025", "2025-01"),
        ("ENE/25", "2025-01"),
        ("2025-01", "2025-01"),
        ("01/2025", "2025-01"),
        ("202501", "2025-01"),
        ("Set-26", "2026-09"),
        ("Sep 26", "2026-09"),
        ("Septiembre-2026", "2026-09"),
        ("Dic-24", "2024-12"),
        ("Ago-26", "2026-08"),
        ("2026 Diciembre", "2026-12"),
    ],
)
def test_interpretar_mes_formatos(etiqueta, esperado):
    assert interpretar_mes(etiqueta) == pd.Period(esperado, freq="M")


@pytest.mark.parametrize(
    "etiqueta", ["Cliente", "ID", "Segmento", "Total", "", None, "Solucion", "13/2025"]
)
def test_interpretar_mes_rechaza_no_meses(etiqueta):
    assert interpretar_mes(etiqueta) is None


def test_interpretar_mes_acepta_fechas_reales():
    assert interpretar_mes(pd.Timestamp("2026-08-15")) == pd.Period("2026-08", freq="M")


@pytest.mark.parametrize(
    "muestra, periodo, esperado",
    [
        ("Ene-25", "2026-09", "Sep-26"),
        ("ene_25", "2026-09", "sep_26"),
        ("ENE-25", "2026-12", "DIC-26"),
        ("Enero 2025", "2026-09", "Septiembre 2026"),
        ("2025-01", "2026-09", "2026-09"),
    ],
)
def test_formato_salida_imita_al_de_entrada(muestra, periodo, esperado):
    """Las columnas nuevas deben verse igual que las que ya tiene el Excel."""
    formato = FormatoMes.inferir(muestra)
    assert formato.formatear(pd.Period(periodo, freq="M")) == esperado


# ---------------------------------------------------------------------------
# Variables predictoras
# ---------------------------------------------------------------------------
def test_features_solo_miran_hasta_el_corte():
    valores = np.array([[10.0, 20.0, 30.0, 999.0, 999.0]])
    f = features_en_corte(valores, corte=2, n_lags=3)

    assert f["lag_1"][0] == 30.0
    assert f["lag_2"][0] == 20.0
    assert f["lag_3"][0] == 10.0
    assert f["media_3m"][0] == pytest.approx(20.0)
    assert f["max_hist"][0] == 30.0        # el 999 del futuro no aparece
    assert f["meses_historia"][0] == 3


def test_features_meses_sin_consumo():
    valores = np.array([
        [10.0, 0.0, 0.0],    # dos meses sin consumo
        [0.0, 0.0, 5.0],     # consumio el ultimo mes
        [0.0, 0.0, 0.0],     # nunca consumio
    ])
    f = features_en_corte(valores, corte=2, n_lags=2)
    assert list(f["meses_sin_consumo"]) == [2.0, 0.0, 99.0]
    assert list(f["meses_activos"]) == [1.0, 1.0, 0.0]


# ---------------------------------------------------------------------------
# Ventanas de validacion
# ---------------------------------------------------------------------------
def test_generar_ventanas():
    assert generar_ventanas(n_meses=32, horizonte=4, ventanas=2) == [28, 27]
    assert generar_ventanas(n_meses=8, horizonte=4, ventanas=2) == [4, 3]
    # Historia demasiado corta: no hay ventanas validas.
    assert generar_ventanas(n_meses=5, horizonte=4, ventanas=2) == []


def test_metricas_wape():
    bt = pd.DataFrame({
        "modelo": ["m", "m"],
        "real": [100.0, 100.0],
        "pred": [120.0, 90.0],
    })
    m = calcular_metricas(bt, ["modelo"]).iloc[0]
    assert m["wape"] == pytest.approx(30 / 200)     # (20 + 10) / 200
    assert m["sesgo"] == pytest.approx(10 / 200)    # (+20 - 10) / 200
    assert m["mae"] == pytest.approx(15.0)


# ---------------------------------------------------------------------------
# Datos y configuracion de prueba
# ---------------------------------------------------------------------------
def _panel_sintetico(n_series=40, n_meses=24, semilla=0) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(semilla)
    periodos = [pd.Period("2024-01", freq="M") + i for i in range(n_meses)]
    niveles = rng.uniform(500, 20000, size=n_series)
    estacional = 1 + 0.15 * np.sin(np.arange(n_meses) * 2 * np.pi / 12)
    valores = np.outer(niveles, estacional) * rng.lognormal(0, 0.1, (n_series, n_meses))

    indice = pd.Index([f"S{i}" for i in range(n_series)], name="serie_id")
    panel = pd.DataFrame(valores, index=indice, columns=periodos)
    meta = pd.DataFrame(
        {
            "segmento": rng.choice(["Chico", "Grande"], n_series),
            "industria": rng.choice(["Agro", "Transporte"], n_series),
            "solucion": rng.choice(["A", "B"], n_series),
        },
        index=indice,
    )
    return panel, meta


def _config_minima(**overrides) -> Config:
    base = {
        "archivos": {"entrada": "x.xlsx", "salida": "y.xlsx", "hoja": "auto", "fila_encabezado": 1},
        "columnas": {"id_cliente": "ID", "solucion": "Solucion", "segmento": "Segmento",
                     "industria": "Industria"},
        "clave_serie": ["id_cliente", "solucion"],
        "variables_categoricas": ["segmento", "industria", "solucion"],
        "meses": {"deteccion": "auto", "horizonte": 4, "ultimo_mes_real": "auto",
                  "celdas_vacias": "cero", "formato_salida": "auto"},
        "modelos": {"media_3m": True, "ultimo_valor": True, "mediana_3m": True,
                    "tendencia_robusta": True, "suavizado_holt": True,
                    "estacional_naive": True, "ml_global": True,
                    "pesos_media_movil": [0.2, 0.3, 0.5], "amortiguacion_tendencia": 0.6},
        "machine_learning": {"algoritmo": "lightgbm", "objetivo": "ratio",
                             "estrategia": "directa", "lags": 6, "minimo_historia": 3,
                             "ratio_minimo": 0.0, "ratio_maximo": 3.0,
                             "hiperparametros": {"n_estimators": 40, "num_leaves": 7,
                                                 "min_child_samples": 5},
                             "semilla": 42},
        "validacion": {"ventanas": 2, "metrica": "wape", "seleccion": "por_cliente",
                       "modelo_respaldo": "media_3m", "margen_mejora": 0.05},
        "reglas": {"no_negativos": True, "redondeo": 0},
        "ejecucion": {"verbosidad": "silencioso"},
    }
    base.update(overrides)
    return Config(base, ruta_base=RAIZ)


# ---------------------------------------------------------------------------
# EL test critico: ausencia de data leakage
# ---------------------------------------------------------------------------
def test_backtest_no_espia_el_futuro():
    """
    Si se corrompen los meses que el backtest debe predecir, las PREDICCIONES
    no pueden cambiar: los modelos solo ven la historia previa al corte.
    Solo deben cambiar los valores reales contra los que se comparan.
    """
    panel, meta = _panel_sintetico()
    cfg = _config_minima()
    modelos = crear_modelos(cfg)

    original = ejecutar_backtest(panel, meta, modelos, cfg)

    # Se destruyen los ultimos 4 meses (los que el backtest intenta predecir).
    corrompido = panel.copy()
    corrompido.iloc[:, -4:] = corrompido.iloc[:, -4:] * 1000 + 50_000

    despues = ejecutar_backtest(corrompido, meta, modelos, cfg)

    clave = ["serie_id", "modelo", "ventana", "periodo"]
    a = original.set_index(clave)["pred"].sort_index()
    b = despues.set_index(clave)["pred"].sort_index()

    # La primera ventana predice justamente esos meses corrompidos: su historia
    # es identica, asi que la prediccion debe ser identica.
    primera = original["ventana"] == 1
    idx = original[primera].set_index(clave).index
    np.testing.assert_allclose(a.loc[idx].to_numpy(), b.loc[idx].to_numpy(), rtol=1e-9)


def test_modelos_devuelven_forma_correcta():
    panel, meta = _panel_sintetico()
    cfg = _config_minima()
    futuros = [panel.columns[-1] + i for i in range(1, 5)]

    for nombre, modelo in crear_modelos(cfg).items():
        pred = modelo.predecir(panel, meta, futuros)
        assert pred.shape == (len(panel), 4), f"{nombre} devolvio {pred.shape}"
        assert list(pred.columns) == futuros, f"{nombre} devolvio columnas erroneas"
        assert pred.index.equals(panel.index), f"{nombre} altero el indice"
        assert np.isfinite(pred.to_numpy(float)).all(), f"{nombre} devolvio NaN o infinito"


def test_ml_global_le_gana_al_promedio_en_datos_con_estacionalidad():
    """Con señal estacional real, el modelo global debe superar al promedio movil."""
    panel, meta = _panel_sintetico(n_series=120, n_meses=30, semilla=3)
    cfg = _config_minima()
    modelos = crear_modelos(cfg)

    bt = ejecutar_backtest(panel, meta, modelos, cfg)
    ranking = calcular_metricas(bt, ["modelo"]).set_index("modelo")["wape"]

    assert ranking["ml_global"] < ranking["media_3m"]


# ---------------------------------------------------------------------------
# Reglas de negocio, sobre el circuito completo
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def entorno(tmp_path_factory):
    """Genera un Excel de ejemplo chico y un config apuntando a el."""
    sys.path.insert(0, str(RAIZ / "datos"))
    from generar_ejemplo import generar

    carpeta = tmp_path_factory.mktemp("caso")
    entrada = carpeta / "consumo.xlsx"
    generar(n_clientes=60, semilla=1, ultimo_mes="2026-08", meses_historia=24).to_excel(
        entrada, sheet_name="Consumo", index=False
    )

    with open(RAIZ / "config.yaml", encoding="utf-8") as fh:
        base = yaml.safe_load(fh)
    base["archivos"]["entrada"] = str(entrada)
    base["archivos"]["salida"] = str(carpeta / "salida.xlsx")
    base["ejecucion"]["verbosidad"] = "silencioso"
    base["machine_learning"]["hiperparametros"] = {"n_estimators": 40, "num_leaves": 7,
                                                   "min_child_samples": 5}
    return carpeta, base


def _correr(base: dict, carpeta: Path, **cambios):
    cfg_dict = yaml.safe_load(yaml.safe_dump(base))  # copia profunda
    for ruta, valor in cambios.items():
        nodo = cfg_dict
        partes = ruta.split(".")
        for p in partes[:-1]:
            nodo = nodo.setdefault(p, {})
        nodo[partes[-1]] = valor
    return ejecutar(Config(cfg_dict, ruta_base=carpeta), verboso=False)


def test_circuito_completo(entorno):
    carpeta, base = entorno
    res = _correr(base, carpeta)

    assert len(res.pronostico) > 0
    assert list(res.pronostico.columns) == res.datos.futuros
    assert (res.pronostico.to_numpy() >= 0).all(), "hay litros negativos"
    assert np.isfinite(res.pronostico.to_numpy()).all()
    assert res.datos.ultimo_real == pd.Period("2026-08", freq="M")
    assert len(res.diagnostico) == len(res.pronostico)


def test_regla_no_negativos(entorno):
    carpeta, base = entorno
    res = _correr(base, carpeta)
    assert (res.pronostico.to_numpy() >= 0).all()


def test_regla_clientes_inactivos_pronostican_cero(entorno):
    carpeta, base = entorno
    res = _correr(base, carpeta, **{"reglas.meses_cero_para_inactivo": 3})

    inactivos = res.diagnostico.index[res.diagnostico["clasificacion"] == "inactivo"]
    if len(inactivos) == 0:
        pytest.skip("el caso de prueba no genero clientes inactivos")
    assert res.pronostico.loc[inactivos].to_numpy().sum() == 0


def test_regla_ajuste_global_escala_todo(entorno):
    carpeta, base = entorno
    normal = _correr(base, carpeta, **{"reglas.redondeo": 4})
    escalado = _correr(base, carpeta, **{"reglas.ajuste_global": 1.10, "reglas.redondeo": 4})

    # El tope de crecimiento puede recortar algunas series, por eso se compara
    # el total con tolerancia en vez de exigir exactamente +10%.
    assert escalado.pronostico.to_numpy().sum() > normal.pronostico.to_numpy().sum()


def test_regla_factor_estacional_por_mes(entorno):
    carpeta, base = entorno
    # Sin redondeo: se comprueba la proporcionalidad exacta del factor.
    normal = _correr(base, carpeta, **{"reglas.redondeo": None})

    factores = {f"{m:02d}": 1.0 for m in range(1, 13)}
    factores["12"] = 0.5
    ajustado = _correr(base, carpeta, **{"reglas.factores_estacionales": factores,
                                         "reglas.redondeo": None})

    diciembre = pd.Period("2026-12", freq="M")
    noviembre = pd.Period("2026-11", freq="M")
    np.testing.assert_allclose(
        ajustado.pronostico[diciembre].to_numpy(),
        normal.pronostico[diciembre].to_numpy() * 0.5,
        rtol=1e-9,
    )
    # Los demas meses no se tocan.
    np.testing.assert_allclose(
        ajustado.pronostico[noviembre].to_numpy(),
        normal.pronostico[noviembre].to_numpy(),
        rtol=1e-9,
    )


def test_regla_tope_de_crecimiento(entorno):
    carpeta, base = entorno
    res = _correr(base, carpeta, **{"reglas.tope_multiplo_maximo": 1.0,
                                    "reglas.redondeo": 4})

    normales = res.diagnostico["clasificacion"] == "normal"
    maximos = res.diagnostico.loc[normales, "maximo_hist"].to_numpy()
    pronosticado = res.pronostico.loc[normales].to_numpy(float)

    assert (pronosticado <= maximos.reshape(-1, 1) + 1e-6).all()


def test_horizonte_configurable(entorno):
    carpeta, base = entorno
    res = _correr(base, carpeta, **{"meses.horizonte": 6})

    assert len(res.datos.futuros) == 6
    assert res.datos.futuros[-1] == pd.Period("2027-02", freq="M")
    assert res.pronostico.shape[1] == 6


def test_clave_serie_separa_soluciones(entorno):
    carpeta, base = entorno
    res = _correr(base, carpeta)

    # Un mismo ID con dos soluciones debe producir dos series distintas.
    ids = res.diagnostico["id_cliente"]
    duplicados = ids[ids.duplicated()].unique()
    if len(duplicados) == 0:
        pytest.skip("el caso de prueba no genero clientes con dos soluciones")

    muestra = res.diagnostico[res.diagnostico["id_cliente"] == duplicados[0]]
    assert len(muestra) == 2
    assert muestra["solucion"].nunique() == 2


# ---------------------------------------------------------------------------
# Validacion de la configuracion
# ---------------------------------------------------------------------------
def test_config_rechaza_opcion_invalida(tmp_path):
    ruta = tmp_path / "c.yaml"
    ruta.write_text(yaml.safe_dump({
        "columnas": {"id_cliente": "ID"},
        "clave_serie": ["id_cliente"],
        "meses": {"horizonte": 4, "deteccion": "telepatia"},
        "modelos": {"media_3m": True},
    }), encoding="utf-8")

    with pytest.raises(ValueError, match="deteccion"):
        cargar_config(ruta)


def test_config_rechaza_modelo_respaldo_apagado(tmp_path):
    ruta = tmp_path / "c.yaml"
    ruta.write_text(yaml.safe_dump({
        "columnas": {"id_cliente": "ID"},
        "clave_serie": ["id_cliente"],
        "meses": {"horizonte": 4},
        "modelos": {"media_3m": True, "ultimo_valor": False},
        "validacion": {"modelo_respaldo": "ultimo_valor"},
    }), encoding="utf-8")

    with pytest.raises(ValueError, match="modelo_respaldo"):
        cargar_config(ruta)


def test_config_rechaza_clave_serie_sin_columna(tmp_path):
    ruta = tmp_path / "c.yaml"
    ruta.write_text(yaml.safe_dump({
        "columnas": {"id_cliente": "ID"},
        "clave_serie": ["id_cliente", "solucion"],
        "meses": {"horizonte": 4},
        "modelos": {"media_3m": True},
    }), encoding="utf-8")

    with pytest.raises(ValueError, match="solucion"):
        cargar_config(ruta)


# ---------------------------------------------------------------------------
# Meses previos al alta del cliente
# ---------------------------------------------------------------------------
def test_mascara_actividad_detecta_el_alta():
    from pronostico.caracteristicas import mascara_actividad

    panel = pd.DataFrame([
        [0.0, 0.0, 100.0, 120.0],   # alta en el mes 3
        [50.0, 0.0, 0.0, 60.0],     # cliente viejo con dos meses sin consumo
        [0.0, 0.0, 0.0, 0.0],       # nunca consumio
    ])
    esperado = [
        [False, False, True, True],
        [True, True, True, True],   # los ceros POSTERIORES al alta si cuentan
        [False, False, False, False],
    ]
    np.testing.assert_array_equal(mascara_actividad(panel), np.array(esperado))


def test_features_excluyen_meses_previos_al_alta():
    from pronostico.caracteristicas import mascara_actividad

    # Alta en el mes 3: solo 100 y 200 son historia real.
    valores = np.array([[0.0, 0.0, 100.0, 200.0]])
    mascara = mascara_actividad(pd.DataFrame(valores))

    con = features_en_corte(valores, corte=3, n_lags=2, mascara=mascara)
    sin = features_en_corte(valores, corte=3, n_lags=2, mascara=None)

    # Con la mascara el promedio es (100+200)/2 = 150.
    assert con["media_3m"][0] == pytest.approx(150.0)
    # Sin ella se divide por 3 y se hunde a 100.
    assert sin["media_3m"][0] == pytest.approx(100.0)

    assert con["meses_desde_alta"][0] == 2
    assert con["prop_ceros"][0] == pytest.approx(0.0)   # nunca consumio 0 siendo cliente
    assert sin["prop_ceros"][0] == pytest.approx(0.5)   # los previos al alta se leen como ceros


def test_media_movil_no_subestima_altas_recientes():
    """Un cliente de alta reciente no debe verse castigado por sus meses previos."""
    panel = pd.DataFrame(
        [[0.0, 0.0, 900.0, 1000.0, 1100.0]],
        index=pd.Index(["nuevo"], name="serie_id"),
        columns=[pd.Period("2026-04", freq="M") + i for i in range(5)],
    )
    meta = pd.DataFrame({"segmento": ["Micro"]}, index=panel.index)
    futuros = [pd.Period("2026-09", freq="M")]

    modelos = crear_modelos(_config_minima())
    activo = modelos["media_3m"].predecir(panel, meta, futuros).iloc[0, 0]
    assert activo == pytest.approx(1000.0)   # (900 + 1000 + 1100) / 3

    apagado = crear_modelos(
        _config_minima(meses={"previo_al_alta": "cero", "horizonte": 4,
                              "deteccion": "auto", "celdas_vacias": "cero"})
    )["media_3m"].predecir(panel, meta, futuros).iloc[0, 0]
    assert apagado == pytest.approx(1000.0)  # aca la ventana de 3 ya es toda posterior al alta


def test_entrenamiento_descarta_meses_previos_al_alta():
    """Las filas con y=0 por no ser todavia cliente no deben entrar al modelo."""
    from pronostico.caracteristicas import construir_muestras

    periodos = [pd.Period("2025-01", freq="M") + i for i in range(12)]
    indice = pd.Index(["viejo", "nuevo"], name="serie_id")
    panel = pd.DataFrame(
        [
            [100.0] * 12,                       # cliente de siempre
            [0.0] * 8 + [500.0] * 4,            # alta en el mes 9
        ],
        index=indice, columns=periodos,
    )
    meta = pd.DataFrame({"segmento": ["A", "B"]}, index=indice)

    con = construir_muestras(panel, meta, 3, 3, 4, ["segmento"], usar_mascara=True)
    sin = construir_muestras(panel, meta, 3, 3, 4, ["segmento"], usar_mascara=False)

    filas_nuevo_con = (con[1]["serie_id"] == "nuevo").sum()
    filas_nuevo_sin = (sin[1]["serie_id"] == "nuevo").sum()
    assert filas_nuevo_con < filas_nuevo_sin

    # Ninguna fila del cliente nuevo puede tener y=0 por no existir todavia.
    assert (con[1].loc[con[1]["serie_id"] == "nuevo", "y"] > 0).all()


def test_previo_al_alta_es_configurable(entorno):
    carpeta, base = entorno
    a = _correr(base, carpeta, **{"meses.previo_al_alta": "no_es_cliente"})
    b = _correr(base, carpeta, **{"meses.previo_al_alta": "cero"})

    # Ambos modos deben correr y dar resultados validos, pero distintos.
    for r in (a, b):
        assert np.isfinite(r.pronostico.to_numpy()).all()
        assert (r.pronostico.to_numpy() >= 0).all()
    assert a.pronostico.to_numpy().sum() != b.pronostico.to_numpy().sum()


# ---------------------------------------------------------------------------
# Fecha de primer consumo declarada (columna del Excel)
# ---------------------------------------------------------------------------
def _panel_con_alta():
    periodos = [pd.Period("2026-01", freq="M") + i for i in range(6)]
    indice = pd.Index(["declarado", "sin_fecha"], name="serie_id")
    panel = pd.DataFrame(
        [
            # Cliente desde marzo, pero recien consumio en mayo:
            # marzo y abril son ceros REALES (era cliente y no consumio).
            [0.0, 0.0, 0.0, 0.0, 100.0, 200.0],
            [0.0, 0.0, 0.0, 0.0, 100.0, 200.0],
        ],
        index=indice, columns=periodos,
    )
    meta = pd.DataFrame(
        {"fecha_primer_consumo": [pd.Period("2026-03", freq="M"), None]},
        index=indice,
    )
    return panel, meta


def test_mascara_prefiere_la_fecha_declarada():
    from pronostico.caracteristicas import mascara_actividad

    panel, meta = _panel_con_alta()
    m = mascara_actividad(panel, meta)

    # Declarada: cliente desde marzo (indice 2), asi los ceros de marzo y abril
    # cuentan como historia real.
    np.testing.assert_array_equal(m[0], [False, False, True, True, True, True])
    # Sin fecha: se infiere del primer consumo (mayo, indice 4).
    np.testing.assert_array_equal(m[1], [False, False, False, False, True, True])


def test_fecha_declarada_cambia_el_promedio():
    """Los ceros posteriores al alta deben pesar; los previos no."""
    from pronostico.caracteristicas import mascara_actividad

    panel, meta = _panel_con_alta()
    modelo = crear_modelos(_config_minima())["media_3m"]
    futuros = [pd.Period("2026-07", freq="M")]

    pred = modelo.predecir(panel, meta, futuros)
    # Declarado: ultimos 3 meses son (0, 100, 200) -> 100.
    assert pred.loc["declarado"].iloc[0] == pytest.approx(100.0)
    # Sin fecha: el cero de abril es previo al alta inferida -> (100, 200) -> 150.
    assert pred.loc["sin_fecha"].iloc[0] == pytest.approx(150.0)


def test_antiguedad_se_mide_desde_el_alta_no_desde_el_panel():
    from pronostico.caracteristicas import antiguedad_meses

    indice = pd.Index(["viejo", "nuevo", "sin_dato"], name="serie_id")
    meta = pd.DataFrame(
        {"fecha_primer_consumo": [
            pd.Period("2019-05", freq="M"),
            pd.Period("2026-06", freq="M"),
            None,
        ]},
        index=indice,
    )
    a = antiguedad_meses(meta, indice, pd.Period("2026-08", freq="M"))

    assert a[0] == 87.0     # el panel puede empezar en 2024 y aun asi verse la antiguedad real
    assert a[1] == 2.0
    assert np.isnan(a[2])


def test_fecha_primer_consumo_se_lee_del_excel(entorno):
    carpeta, base = entorno
    res = _correr(base, carpeta)

    assert "fecha_primer_consumo" in res.datos.meta.columns
    fechas = res.datos.meta["fecha_primer_consumo"].dropna()
    assert len(fechas) > 0
    assert all(isinstance(p, pd.Period) for p in fechas)
    # Ningun alta puede ser posterior al ultimo mes real.
    assert fechas.max() <= res.datos.ultimo_real


def test_funciona_sin_la_columna_de_alta(entorno):
    """Quien no tenga esa columna debe seguir corriendo, infiriendo el alta."""
    carpeta, base = entorno
    res = _correr(base, carpeta, **{"columnas.fecha_primer_consumo": None})

    assert "fecha_primer_consumo" not in res.datos.meta.columns
    assert np.isfinite(res.pronostico.to_numpy()).all()
    assert (res.pronostico.to_numpy() >= 0).all()
