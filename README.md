# Taller B5-T1 · Generación de datos financieros sintéticos

¿Pueden los modelos generativos mejorar el rendimiento de un modelo de machine
learning cuando los datos reales escasean? Este repositorio responde a esa
pregunta sobre un problema en el que la escasez no es una limitación
circunstancial sino estructural: **la detección anticipada de regímenes de
crisis en los mercados financieros**.

## Grupo

- Oscar Romero Quincoces
- Fernando Dapena Tauste
- Daniel García López

Máster MIAX · Bloque 5 · Entrega: 3 de septiembre de 2026

---

## 1. El problema y por qué necesita datos sintéticos

Un modelo que anticipe si el mercado entrará en tensión durante el próximo mes
tiene un valor evidente para la gestión de riesgo. El obstáculo no es la
arquitectura: es que **las crisis son raras**. En las dos décadas que cubre
nuestro panel (2003-2026) hay del orden de diez episodios de estrés, y aunque el
dataset contenga cientos de ventanas etiquetadas como crisis, el número de
eventos **independientes** es de una decena. Un clasificador entrenado sobre ese
desbalance aprende a no predecir nunca la clase que importa: acierta el 90 % de
las veces y es inútil.

Ese es exactamente el escenario en el que los datos sintéticos deberían aportar
valor, y por eso lo hemos elegido.

Se resuelven **dos tareas** sobre la misma ventana de entrada, para contrastar
si el beneficio del dato sintético depende de la naturaleza del problema:

| | Tarea principal | Tarea de control |
|---|---|---|
| **Objetivo** | Régimen dominante a 21 días | Volatilidad realizada a 21 días |
| **Tipo** | Clasificación (3 clases) | Regresión |
| **Métrica** | F1-macro y recall de crisis | MAE y R² |
| **Desbalance** | Severo (crisis ≈ 16 %) | No aplica |

Si los sintéticos ayudan en la primera y no en la segunda, el mecanismo es el
rebalanceo de clases. Si ayudan en ambas, es enriquecimiento general de la
distribución. Ambos resultados son informativos.

## 2. Datos

Panel híbrido descargado de yfinance, sin claves de API: cualquier integrante
reconstruye los datos brutos con una sola llamada.

- **Universo (15 activos):** S&P 500, los nueve ETF sectoriales SPDR, VIX,
  tesoro a 20 y 10 años, crédito grado de inversión e índice dólar.
- **Canales (20):** once de retornos (índice, nueve sectores y dólar) y nueve
  derivados —nivel y variación del VIX, volatilidad realizada, drawdown,
  momento, spread de crédito, pendiente de curva, correlación acción-bono y
  dispersión sectorial.
- **Ventana:** `X` de 60 días × 20 canales → `Y` sobre los 21 días siguientes.

Todas las transformaciones son **causales**: en el instante `t` solo usan
información de `t` o anterior. Los z-score se calculan con media y desviación
acumuladas, nunca sobre la muestra completa.

El etiquetado de regímenes usa un HMM gaussiano de tres estados ajustado **solo
con el tramo de entrenamiento**, con los estados ordenados por volatilidad
creciente (`0 = calma`, `2 = crisis`) para que la numeración sea estable entre
ejecuciones.

Toda la configuración vive en [`data/catalog.yaml`](data/catalog.yaml), que es
la fuente de verdad del proyecto: ningún módulo lleva tickers, fechas ni
longitudes de ventana escritos a mano.

## 3. Los generadores

El enunciado pide tres modelos generativos de los vistos en clase más un modelo
simple. Implementamos **siete**, para poder comparar familias enteras y no solo
instancias:

| Generador | Familia | Condicional | Convergencia se lee en |
|---|---|---|---|
| `jitter` | Reales + ruido (modelo simple) | Por muestreo | Desviación relativa |
| `gaussiano` | Estadístico | Un modelo por régimen | Número de condición |
| `cvae` | Latente variacional | Etiqueta en encoder y decoder | Recon / KL / unidades activas |
| `cgan` | Adversarial | Etiqueta en G y D | Equilibrio en torno a log 2 |
| `rbig` | Normalizing flow | Un modelo por régimen | Reducción de información por capa |
| `flow_matching` | Flujo continuo (ODE) | Embedding de clase | Pérdida de regresión |
| `difusion` | Difusión (DDIM) | Classifier-free guidance | MSE de predicción de ruido |

Todos comparten la interfaz de
[`src/generadores/base.py`](src/generadores/base.py), lo que permite que los
notebooks de mezcla y análisis los traten por igual.

Los cuatro que exige el enunciado son `jitter` (el simple) más tres cualesquiera
de los generativos; los demás son ampliación.

## 4. Diseño experimental

El análisis recorre **dos ejes**, no uno:

1. **Proporción de sintéticos:** 0, 0.5×, 1×, 2× y 5× respecto al número de
   muestras reales.
2. **Escasez de datos reales:** 250, 500, 1000, 2000 y todas las ventanas.

El segundo eje es imprescindible. El material del taller muestra que el
beneficio de los sintéticos se concentra cuando hay pocos datos reales y se
desvanece —o se invierte— cuando ya hay muchos. Barrer solo el porcentaje de
sintéticos con el dataset completo produciría una línea plana y la conclusión
errónea de que los generadores no sirven.

Se prueban además dos políticas de reparto por clase: `proporcional` (el
sintético replica el desbalance) y `equilibrado` (el sintético se concentra en
la clase minoritaria). La comparación entre ambas aísla cuánto del efecto es
"más datos" y cuánto es "más datos donde hacen falta".

**El modelo downstream es el mismo en todas las versiones**: una CNN 1D fijada
en el notebook 03 con datos reales y congelada a partir de ahí. Lo único que
cambia entre experimentos son los datos de entrenamiento.

### Integridad del experimento

Las ventanas de 60 días se solapan en 59 observaciones con su vecina. Un split
aleatorio colocaría ventanas casi idénticas a ambos lados del corte y el modelo
obtendría métricas excelentes por memorizar. Por eso:

- El split es **temporal**, con un **embargo de 85 sesiones de mercado** en cada frontera, verificado por conteo de posiciones.
- Generadores, etiquetado de regímenes y escalado se ajustan **solo con train**.
- Validación y test son **siempre 100 % reales**.

Los notebooks guiados del máster usan `train_test_split` aleatorio sobre estas
mismas ventanas; apartarnos de ahí es una decisión deliberada, documentada en
[`docs/DECISIONES.md`](docs/DECISIONES.md) y justificada en
[`docs/metodologia/riesgos_datos_sinteticos.md`](docs/metodologia/riesgos_datos_sinteticos.md).

## 5. Estructura del repositorio

```
TALLER_GENERATIVOS/
├── data/
│   ├── catalog.yaml            fuente de verdad: universo, canales, ventanas, splits
│   ├── raw/                    precios descargados de yfinance (no versionado)
│   ├── processed/              X, Y, splits y escalador  ← permite saltarse los notebooks 00-02
│   └── synthetic/              banco de muestras de cada generador (.npz)
├── models/
│   ├── generadores/            un subdirectorio por generador: pesos + historial + metadatos
│   └── downstream/             modelo de referencia de la arquitectura fijada
├── src/
│   ├── config.py               lectura del catálogo y semillas
│   ├── datos.py                descarga y alineado del panel
│   ├── features.py             primitivas causales y construcción de canales
│   ├── regimenes.py            HMM, canonicalización económica y agregación al horizonte
│   ├── ventanas.py             ventanas deslizantes, split temporal con embargo, escalado
│   ├── generadores/            base.py (contrato) + un módulo por generador
│   ├── downstream.py           la arquitectura CNN 1D, común a todos los experimentos
│   ├── mezclas.py              rejilla experimental y montaje de datasets mixtos
│   ├── evaluacion.py           métricas downstream y de calidad sintética
│   └── viz.py                  estilo y figuras del informe
├── notebooks/                  00 a 14, numerados y encadenados
├── docs/
│   ├── INDEX.md                índice de la documentación
│   ├── DECISIONES.md           decisiones de diseño y su justificación
│   ├── teoria/                 un documento por familia de modelos generativos
│   ├── metodologia/            etiquetado, métricas de calidad y riesgos
│   ├── decisions/              registros de decisión de arquitectura (ADR)
│   ├── enunciado/              enunciado oficial del taller
│   └── material_clase/         material del profesor (no versionado)
├── results/
│   ├── historiales/            curvas de entrenamiento en CSV, una por experimento
│   ├── metricas/               tabla maestra de resultados
│   └── figures/                figuras del informe
└── report/                     presentación en PDF
```

## 6. Arranque rápido

```bash
pip install -r requirements.txt
```

Después, desde `notebooks/`, hay **dos caminos**:

**Reproducir todo desde cero** (varias horas en CPU): ejecutar los notebooks 00
a 14 en orden.

**Empezar directamente** (recomendado): `data/processed/`, `data/synthetic/` y
`models/generadores/` están versionados, así que se puede abrir cualquier
notebook del 11 en adelante y trabajar con los generadores ya entrenados. Solo
si `data/raw/` está vacío hace falta ejecutar el 00 una vez para descargar los
precios.

### Encadenado de los notebooks

Los notebooks 04 a 10 **no dependen entre sí**: todos leen `data/processed/` y
escriben en su propio subdirectorio, así que se pueden entrenar en paralelo sin
bloquearse.

| Notebook | Contenido |
|---|---|---|
| `00_datos_y_features` | Descarga y construcción de canales |
| `01_etiquetado_regimenes` | HMM y canonicalización |
| `02_ventanas_y_splits` | Ventanas, split temporal, escalado |
| `03_downstream_baseline` | Arquitectura CNN 1D de referencia |
| `04_gen_ruido` | Generador jitter |
| `05_gen_gaussiano` | Gaussiano multivariante |
| `06_gen_cvae` | VAE condicional |
| `07_gen_cgan` | GAN condicional |
| `08_gen_flow_matching` | Flow matching condicional |
| `09_gen_rbig` | Normalizing flow RBIG |
| `10_gen_difusion` | Difusión DDIM condicional |
| `11_mezclas_datasets` | Rejilla de datasets mixtos |
| `12_barrido_entrenamiento` | Entrenamiento de todas las versiones |
| `13_analisis_resultados` | Figuras y tablas del informe |
| `14_calidad_sinteticos` | Métricas de calidad de los sintéticos |

### Nota sobre el entorno

Todo está dimensionado para **CPU**: no se ha usado GPU en ningún momento. Las
arquitecturas son deliberadamente pequeñas y el notebook 12 es **reanudable**
—salta las recetas que ya tengan resultado en el CSV—, porque el barrido
completo son varios cientos de entrenamientos.

## 7. Resultados

_Pendiente de completar tras ejecutar el barrido._

## 8. Estado del proyecto

| Fase | Estado |
|---|---|
| Definición del problema y diseño experimental | ✅ |
| Documentación teórica y metodológica | ✅ |
| Infraestructura de código (`src/`) | ✅ |
| Esqueleto de notebooks | ✅ |
| Descarga de datos y construcción de canales (notebook 00) | ✅ |
| Etiquetado de regímenes (notebook 01) | 🔜 |
| Arquitectura downstream de referencia | 🔜 |
| Entrenamiento de los siete generadores | 🔜 |
| Barrido experimental | 🔜 |
| Análisis y figuras | 🔜 |
| Presentación en PDF | 🔜 |
