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
| **Desbalance** | Severo (crisis 15,9 % en train) | No aplica |

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

**El modelo downstream es el mismo en todas las versiones**: la arquitectura se
**busca** en el notebook 03 entre seis candidatas con datos reales, y la que gana
—`lineal`, una multinomial sobre la ventana aplanada, sin ninguna capa
convolucional— queda congelada a partir de ahí. Lo único que cambia entre
experimentos son los datos de entrenamiento.

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
│   ├── calidad.py              los once controles de integridad del panel: cinco lanzan, seis avisan
│   ├── regimenes.py            HMM, canonicalización económica y agregación al horizonte
│   ├── ventanas.py             ventanas deslizantes, split temporal con embargo, escalado
│   ├── generadores/            base.py (contrato) + un módulo por generador
│   ├── downstream.py           las seis candidatas y la arquitectura congelada, común a todos
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

**Empezar directamente** (recomendado): lo que está versionado hoy es
`data/processed/` —`canales.parquet`, `objetivos.parquet`, `regimenes.parquet`,
`ventanas.npz` y `etiquetador_regimenes.pkl`—, es decir los canales, el
etiquetado de regímenes, las ventanas ya partidas con embargo y el escalador. Con
eso, **el punto de entrada de un tercero es el notebook 03**. `data/synthetic/` y
`models/generadores/` solo contienen un `.gitkeep` mientras no se ejecute el
bloque de generadores, así que los notebooks 11 en adelante todavía no se pueden
abrir en frío.

**Qué no es reproducible bit a bit.** `data/raw/` no se versiona (D18) y **no se
reconstruye idéntico**: yfinance revisa su histórico y reajusta los dividendos
hacia atrás, de modo que dos descargas separadas en el tiempo no devuelven los
mismos precios —`src/datos.py` lo documenta en `huella()`, y por eso los precios
se cachean una sola vez—. Lo reproducible es el pipeline **desde
`data/processed/`**. Para saber si dos personas trabajan sobre los mismos
números, el notebook 00 imprime dos huellas de contenido, que en la ejecución
versionada valen:

| Objeto | Huella |
|---|---|
| panel de precios (5.942 × 15) | `95170c3447ad` |
| `canales.parquet` (5.670 × 20) | `f43482436dcd` |

Si esas dos cadenas coinciden, cualquier diferencia de métricas viene del código
y no de los datos.

### Encadenado de los notebooks

Los notebooks 04 a 10 **no dependen entre sí**: todos leen `data/processed/` y
escriben en su propio subdirectorio, así que se pueden entrenar en paralelo sin
bloquearse.

| Notebook | Contenido |
|---|---|
| `00_datos_y_features` | Descarga y construcción de canales |
| `01_etiquetado_regimenes` | HMM y canonicalización |
| `02_ventanas_y_splits` | Ventanas, split temporal, escalado |
| `03_downstream_baseline` | Búsqueda entre seis candidatas y arquitectura congelada |
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

Los cuadernos 00 a 03 están ejecutados y sus cifras son reproducibles desde
`data/processed/`. El barrido de generadores (cuadernos 04 a 14) está pendiente.

### 7.1 · Lo que ya está medido

**Los canales llevan señal, y es de los derivados.** Los 20 canales pasan el
contraste de causalidad 20 de 20. Seis superan un AUC univariante de 0,70 contra el
decil superior de volatilidad futura —`vix_nivel_z` 0,90 a la cabeza— y **los seis
son features derivadas**; el mejor retorno crudo se queda en 0,55.

**El mercado no cabe en los generadores baratos, y es falsable.** La curtosis de los
residuos estandarizados vale **7,2** con banda bootstrap **[5,6 – 10,3]**, y los
cuatro procesos de referencia —gaussiano, t de Student, GARCH(1,1) y mezcla por
régimen— caen fuera. Esa es la hipótesis H1 del trabajo y está enunciada de forma
que los cuadernos 05 a 10 puedan refutarla.

**El etiquetado hubo que corregirlo, y la comparación es el argumento.** Con las
cinco features iniciales el HMM suspende el control bloqueante: la cobertura del
episodio de Inflación 2022 se queda en el 44,0 % frente al 50 % exigido, porque
`spread_credito_z` deriva 1,17 sigmas entre las dos mitades de la muestra y
`drawdown_sp500` satura. Con tres features el control pasa y la clase de crisis
queda en el **15,8 %**.

**El embargo estaba contado en días naturales y era una fuga real.** 85 días
naturales son 59 sesiones y hacen falta 81: **22 sesiones estaban a la vez en train
y validación**, y otras 22 entre validación y test. Contado en sesiones, el hueco es
de 86 y las sesiones compartidas son **0**.

**El tamaño muestral honesto no son las ventanas.** Las 3.696 ventanas de train son
**45 bloques disjuntos** de 81 sesiones y 8 rachas de crisis; test son 12 bloques y
3 rachas. Por eso el recall de crisis de la persistencia, 0,800, viene con un
intervalo por bootstrap de bloques de **[0,560 – 1,000]**, tres veces más ancho que
el binomial ingenuo.

### 7.2 · La arquitectura, y el control que suspende

De seis candidatas gana `lineal`, una multinomial de **3.603 parámetros** sobre la
ventana aplanada, empatada dentro de una desviación entre semillas con
`cnn_kernel7`, que tiene 271.315. Con 45 bloques independientes la convolución no
aporta nada medible.

El control bloqueante **suspende**. Sobre test:

| | balanced accuracy | f1-macro | recall crisis |
|---|---|---|---|
| persistencia causal (la barra) | **0,777** | 0,754 | 0,800 |
| arquitectura congelada | 0,575 | 0,499 | 0,591 |
| con `class_weight` | 0,556 | 0,464 | 0,673 |

**El diagnóstico es lo más útil del cuaderno 03**, porque señala qué hay que
decidir antes de entrenar un solo generador. Cambiando solo el objetivo, con la
misma entrada y la misma arquitectura:

| tarea (con `cnn_ancha`) | balanced accuracy en test |
|---|---|
| ventana a régimen de **hoy** | **0,808** |
| ventana a régimen a 21 días, **directo** (con `lineal`, la congelada) | 0,575 |
| ventana a régimen de hoy y luego **persistir** | **0,719** |

La información está en la ventana: el encadenado recupera 14 de los 20 puntos que
separan al modelo directo de la barra, aunque no la alcanza. Y en la tarea auxiliar
la convolución **sí** paga —`cnn_ancha` 0,808 frente a 0,682 de `lineal`—, justo lo
contrario que en la directa: que el orden entre arquitecturas se invierta al cambiar
el objetivo señala que el problema está en cómo se plantea la tarea, no en los datos.
La formulación directa además sobreajusta esos 45 bloques, aunque despacio: con la
arquitectura congelada la pérdida de train cae de 1,156 a 0,496 en diez épocas y sigue
bajando hasta 0,314 en la 60, mientras la de validación toca su mínimo en 0,766 en la
época 38 y a partir de ahí sube; la brecha entre ambas pasa de 0,40 a 0,54
(`results/historiales/baseline_regimen.csv`). Adoptar el encadenado obligaría a que
los generadores produzcan también el régimen de hoy, lo que toca D8, y es una
decisión de diseño pendiente.

La tarea de control sí funciona: la regresión de volatilidad da **MAE 0,046 frente
a 0,050** de su línea base trivial, con R² de 0,164 contra la media de train. Que
una tarea sin desbalance mejore y la desbalanceada no es, en sí, un dato sobre
dónde está el problema.

### 7.3 · Lo que cuesta el barrido

Con la arquitectura congelada, 0,43 s por época y **51,0 épocas efectivas medias**
—las suyas en la búsqueda, no la media de las seis candidatas, que baja a 24,2
porque las CNN paran mucho antes—: **4,4 h para los siete generadores** y 2,5 h para
los cuatro del núcleo mínimo. Los siete siguen siendo asumibles.

## 8. Estado del proyecto

| Fase | Estado |
|---|---|
| Definición del problema y diseño experimental | ✅ |
| Documentación teórica y metodológica | ✅ |
| Infraestructura de código (`src/`) | ✅ |
| Esqueleto de notebooks | ✅ |
| Descarga de datos y construcción de canales (notebook 00) | ✅ |
| Etiquetado de regímenes (notebook 01) | ✅ |
| Ventanas, split temporal con embargo y escalado (notebook 02) | ✅ |
| Arquitectura downstream congelada (notebook 03) | ✅ · control bloqueante suspendido, ver 7.2 |
| Entrenamiento de los siete generadores (notebooks 04 a 10) | 🔜 |
| Mezclas y barrido experimental (notebooks 11 y 12) | 🔜 |
| Análisis, figuras y calidad de los sintéticos (notebooks 13 y 14) | 🔜 |
| Presentación en PDF | 🔜 |
