# Pronóstico de consumo de combustible

Proyecta el consumo mensual en **litros** de cada cliente usando su historia y
el comportamiento de clientes similares. Pensado para el caso: tenés enero-agosto
real y necesitás septiembre-diciembre.

Entra un Excel con clientes en filas y meses en columnas. Sale otro Excel con los
meses nuevos agregados, más las hojas de control para saber cuánto podés confiar
en cada número.

---

## Arranque rápido

```bash
pip install -r requirements.txt

python datos/generar_ejemplo.py     # crea un Excel de prueba
python run.py                       # genera el pronóstico
```

El resultado queda en `salidas/pronostico_combustible.xlsx`.

Cuando quieras usar tus datos reales, apuntá `archivos.entrada` en `config.yaml`
a tu archivo y volvé a correr `python run.py`.

---

## Cómo funciona

El programa **no** usa un solo modelo. Corre ocho en paralelo, mide cuál acierta
más en **cada cliente** y usa el ganador de cada uno.

### 1. Los modelos que compiten

| Modelo | Qué hace |
|---|---|
| `ultimo_valor` | Repite el último mes. Es la vara mínima a superar. |
| `media_3m` | Promedio de los últimos 3 meses. Difícil de ganar. |
| `mediana_3m` | Igual, pero ignora meses atípicos. |
| `media_movil_pond` | Promedio que pesa más los meses recientes. |
| `tendencia_robusta` | Recta de tendencia por mediana de pendientes (Theil-Sen). |
| `suavizado_holt` | Suavizado exponencial con tendencia amortiguada. |
| `estacional_naive` | Mismo mes del año pasado. Necesita 13+ meses. |
| `ml_global` | **Machine Learning.** Ver abajo. |

### 2. Por qué el modelo de ML es "global"

Con 8 meses por cliente, entrenar un modelo por cliente no funciona: son
demasiados pocos puntos. La solución estándar es al revés — **un solo modelo
entrenado con todos los clientes juntos**.

Así, un cliente de Agro con 4 meses de historia hereda el patrón estacional de
los demás clientes de Agro. De ahí que `Segmento`, `Industria` y `Portfolio`
entren al modelo: son el puente entre clientes.

El modelo (LightGBM con función de pérdida absoluta, robusta a clientes atípicos)
no predice litros crudos, sino un **ratio contra el nivel reciente del cliente**.
Es lo que permite que un cliente de 500 litros/mes y uno de 300.000 entrenen
juntos sin que el grande domine todo.

Variables que usa: rezagos de 6 meses, promedios y medianas móviles, desvíos,
pendientes, momento, meses sin consumo, mes calendario objetivo, y las
categóricas del cliente.

Se entrena un modelo separado por cada mes futuro (`estrategia: directa`), que es
más robusto que encadenar predicciones sobre predicciones.

### 3. Validación honesta (backtesting)

Antes de pronosticar, el programa **se pone a prueba en el pasado**: corta la
historia en un mes anterior, pronostica lo que sigue, y compara contra lo que
realmente pasó. La métrica principal es **WAPE** (litros de error sobre litros
reales), porque es la que le importa al negocio: errarle 5% a un cliente grande
pesa más que errarle 30% a uno chico.

Ese error es el que aparece en la hoja `Validacion` y el que arma las columnas
`Rango mínimo` / `Rango máximo`. **No es un adorno**: es tu medida de cuánta
confianza darle a cada cliente.

Hay un test (`test_backtest_no_espia_el_futuro`) que verifica que ningún modelo
use información posterior al corte. Sin eso, el error reportado sería mentira.

### 4. Clientes que no existían en 2024

Si tenés historia desde 2024 pero muchos clientes son altas recientes, sus filas
arrancan vacías. Esas celdas **no son "consumió 0"**: son "no era cliente".

Tratarlas como ceros hunde los promedios históricos de toda alta nueva y hace que
el modelo la lea como un cliente errático o en caída. El programa detecta el mes
de alta (primer mes con consumo) y excluye lo anterior de promedios, tendencias y
del entrenamiento. Los ceros **posteriores** al alta sí se toman como reales:
esos son bajas o paradas, e importan.

Se controla con `meses.previo_al_alta` (`no_es_cliente` por defecto, `cero` para
volver al comportamiento ingenuo y comparar).

### 5. Reglas de negocio

Después del modelo se aplica lo que el modelo no puede saber, todo configurable:

- Clientes sin consumo en los últimos N meses → se pronostica 0.
- Clientes nuevos → se usa el perfil estacional de su segmento e industria,
  corregido por la estacionalidad de los meses que sí observaste.
- Topes de crecimiento y caída, para que ninguna extrapolación se vaya de rango.
- Factores estacionales manuales por mes.
- Ajuste global (escenarios, metas comerciales).
- Ajustes manuales por cliente desde una hoja del propio Excel.

---

## El Excel de salida

| Hoja | Para qué sirve |
|---|---|
| **Pronostico** | Igual a tu archivo, con los meses nuevos en verde. Es la hoja para circular. |
| **Detalle_Largo** | Una fila por cliente-mes, real y pronóstico juntos. Para tablas dinámicas y Power BI. |
| **Validacion** | Ranking de modelos y el error de cada cliente. **Miralo antes de creerle a un número.** |
| **Resumen_Mensual** | Totales de la cartera mes a mes, con gráfico. |
| **Resumen_Dimensiones** | Aperturas por Solución, Segmento, Industria, Portfolio y Grupo. |
| **Configuracion** | Con qué parámetros se generó, más los avisos. Trazabilidad. |

---

## Configuración

Todo se toca en `config.yaml`, que está comentado línea por línea. Lo que más vas
a usar:

```yaml
archivos:
  entrada: "datos/consumo_clientes.xlsx"   # tu archivo

meses:
  ultimo_mes_real: "auto"   # o fijalo: "2026-08"
  horizonte: 4              # Sep, Oct, Nov, Dic

reglas:
  ajuste_global: 1.00       # 1.05 = +5% en toda la cartera
  factores_estacionales:
    "12": 1.00              # bajalo si diciembre siempre cae
```

Las columnas descriptivas se mapean acá, así que si en tu Excel se llaman
distinto no hace falta tocar código:

```yaml
columnas:
  id_cliente: "ID"
  solucion:   "Solucion"
  segmento:   "Segmento"
```

**La serie se identifica por `ID + Solución`**, no por ID solo. Un cliente con
dos soluciones contratadas son dos series independientes, que es lo correcto:
tienen consumos y estacionalidades distintas. Si alguna vez querés el ID
consolidado, dejá `clave_serie: ["id_cliente"]`.

### Desde la línea de comandos

```bash
python run.py --entrada datos/octubre.xlsx   # otro archivo
python run.py --horizonte 6                  # 6 meses en vez de 4
python run.py --ultimo-mes-real 2026-09      # cuando cierres septiembre
python run.py --solo-validar                 # mide el error sin escribir el Excel
```

---

## El uso mes a mes

Cuando cierra septiembre:

1. Agregá la columna `Sep-26` con el consumo real a tu Excel.
2. `python run.py`

El programa detecta solo que ahora hay un mes más de historia y reproyecta
octubre-diciembre. No hay que tocar nada.

Con `--solo-validar` podés comparar el error de este mes contra el anterior y ver
si el modelo se está degradando.

---

## Formatos de columna de mes que reconoce

`Ene-25`, `ene_2025`, `Enero 2025`, `ENE/25`, `2025-01`, `01/2025`, `202501`,
`Set-26`, `Sep 26`, y fechas reales de Excel. Las columnas nuevas se escriben
imitando el estilo de las tuyas.

Si tu formato no está, poné `meses.deteccion: manual` y completá `mapa_manual`.

---

## Estructura

```
pronostico_combustible/
├── config.yaml                    # <- lo único que tocás normalmente
├── run.py                         # punto de entrada
├── pronostico/
│   ├── config.py                  # carga y valida config.yaml
│   ├── datos.py                   # lectura Excel, detección de meses
│   ├── caracteristicas.py         # variables predictoras
│   ├── modelos.py                 # los 8 modelos
│   ├── validacion.py              # backtesting y métricas
│   ├── motor.py                   # orquestador y reglas de negocio
│   └── salida.py                  # escritura del Excel
├── datos/generar_ejemplo.py       # datos sintéticos de prueba
└── tests/test_pronostico.py       # 50 tests
```

### Agregar un modelo propio

En `modelos.py`, heredá de `Modelo`, implementá `predecir()`, registralo en
`REGISTRO` y activalo en `config.yaml`. Entra a competir solo:

```python
class MiModelo(Modelo):
    nombre = "mi_modelo"

    def predecir(self, panel, meta, futuros):
        # panel: series x meses, ya recortado hasta el último mes conocido
        nivel = panel.to_numpy(float)[:, -4:].mean(axis=1)
        return self._repetir(nivel, futuros, panel.index)
```

---

## Tests

```bash
pip install pytest
python -m pytest tests/ -q
```

Cubren la detección de meses, la ausencia de data leakage, el tratamiento de los
meses previos al alta, la forma de salida de cada modelo, cada regla de negocio y
la validación de la configuración.

---

## Qué esperar del error

En los datos de ejemplo (484 series, 32 meses), el ranking a 4 meses vista:

```
ml_global            WAPE  23.7%   sesgo  -0.0%
mediana_3m           WAPE  26.2%   sesgo  -3.7%
media_3m             WAPE  27.6%   sesgo  -4.6%
ultimo_valor         WAPE  31.0%   sesgo  -0.6%
```

El ML gana, pero por unos 4 puntos sobre un promedio móvil de 3 meses. **Eso es
lo normal y conviene decirlo de entrada**: en pronóstico de demanda los baselines
son fuertes, y cualquiera que prometa mejoras enormes sobre ellos con 8 meses de
historia está midiendo mal. Por eso el programa deja que compitan de igual a
igual y elige por cliente.

Dos advertencias honestas:

- **Con solo 8 meses no hay ciclo estacional completo.** El modelo no puede saber
  que diciembre cae si nunca vio un diciembre. Si tenés columnas de años
  anteriores, incluilas: es lo que más mejora el resultado. Si no, usá
  `factores_estacionales` para meter esa información a mano.
- **El error a 4 meses vista es sustancialmente mayor que a 1 mes.** Mirá la
  hoja `Validacion` por cliente antes de comprometer un número de diciembre.
