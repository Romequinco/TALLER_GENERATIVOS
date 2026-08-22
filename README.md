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

**Convención de nombres de figura.** Las figuras creadas en la reescritura de los
cuadernos 00 a 03 llevan el número de su cuaderno como prefijo (`00_`, `01_`, `02_`,
`03_`); las anteriores conservan su nombre heredado porque hay documentación que las
referencia por él —`docs/metodologia/etiquetado_regimenes.md` cita
`control_etiquetado.png` y `linea_base_persistencia.png`—, y por eso el notebook 01
guarda la comparación causal-frente-a-Viterbi dos veces, con el nombre nuevo y con el
heredado. Hoy hay 44 figuras: 35 de los cuadernos 00 a 03 y 9 de los generadores ya
entrenados.

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
`models/generadores/` ya no están vacíos: traen los bancos y los modelos de
`jitter`, `gaussiano` y `cvae` —`jitter.npz` y `gaussiano.npz` con 12.000 muestras de
1.201 columnas, 4.000 por régimen, y `cvae.npz` con 9.000, 3.000 por régimen—. Lo que
falta son los cuatro generadores restantes (`cgan`, `flow_matching`, `rbig` y
`difusion`), así que el notebook 11 todavía no puede montar la rejilla completa.

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

**Aviso de trazabilidad sobre `data/processed/ventanas.npz`.** Ese riesgo ya se
materializó una vez: al re-ejecutar la cadena 00→02 contra el `precios.parquet`
cacheado actual, el fichero se regeneró con valores distintos de los que tenía. Las
etiquetas —`y_reg` y `y_vol` de las tres particiones— son **idénticas bit a bit**; lo
único que se movió es `X`, y en unidades ya escaladas: la diferencia máxima es de
**2,5·10⁻⁴** en train, y el origen es el reajuste retroactivo de dividendos de
yfinance sobre los ETF sectoriales. Ninguna cifra de este README depende de eso.
Lo que sí depende es la trazabilidad de los generadores: los tres bancos de
`data/synthetic/` y los tres modelos de `models/generadores/` se ajustaron sobre la
versión anterior del `.npz` —se ve en la primera celda de los notebooks 04, 05 y 06,
que imprimen `1970-01-01 a 1970-01-01` porque aquella versión guardaba las fechas en
milisegundos y la actual en nanosegundos—, de modo que **hay que re-ejecutar 04, 05 y
06 antes del barrido del notebook 12**. No cambia ninguna conclusión de signo, pero el
barrido tiene que salir de un único `ventanas.npz`. Desde esa
regeneración el fichero es estable: el notebook 02 comprueba array a array, antes de
escribir, que reproduce el que ya había en disco —esa comprobación es contra la versión
ya regenerada, no contra la que leyeron 04, 05 y 06, y por eso el cuaderno 02 da por
buenos los bancos sintéticos y esta sección no—.

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
`data/processed/`. También lo están los tres primeros generadores —04, 05 y 06—, con
sus bancos y sus modelos versionados. Pendientes quedan los cuadernos 07 a 14: los
cuatro generadores restantes, las mezclas, el barrido y el análisis.

### 7.1 · Lo que ya está medido

**La escasez es estructural, y se cuenta.** Las 5.590 ventanas nominales del panel son
solo **70 bloques disjuntos** de 81 sesiones —79,9 ventanas por bloque, que es el
factor con el que el dataset se infla a sí mismo— y contienen **11 episodios de
estrés**, de los que 7 caen en train y 5 duran una sola sesión. De los de train, los
que superan el horizonte de 21 días son **2**: esas son las crisis entrenables, y ese
es el tamaño muestral real del problema. El conteo depende del
umbral pero no el orden de magnitud: con el VIX en |z| > 1,5 salen 13 episodios, con
2,0 salen 11 y con 2,5 salen 5.

**Los canales llevan señal, y es de los derivados.** Los 20 canales pasan el
contraste de causalidad 20 de 20. Seis superan un AUC univariante de 0,70 contra el
decil superior de volatilidad futura —`vix_nivel_z` 0,90 a la cabeza— y **los seis
son features derivadas**; el mejor retorno crudo se queda en 0,55, y el suelo de ruido
de la métrica, medido permutando el objetivo, está en 0,52.

**El mercado no cabe en los generadores baratos, y es falsable.** La curtosis de los
residuos estandarizados vale **7,2** con banda bootstrap **[5,6 – 10,3]**, y los
cuatro procesos de referencia —gaussiano, t de Student, GARCH(1,1) y mezcla por
régimen— caen fuera. Esa es la hipótesis H1 del trabajo y está enunciada de forma
que los cuadernos 05 a 10 puedan refutarla.

**Lo que un generador tiene que reproducir no está en la marginal.** La
autocorrelación de |r| en el retardo 1 vale 0,313 sobre la serie agrupada y 0,026
ventana a ventana: un factor **12** que solo existe si se respeta el orden temporal.
Y la correlación media entre los nueve sectores pasa de **0,615 en calma a 0,786 en
estrés**, con los 36 pares subiendo sin excepción; en el 5 % de peores sesiones del
índice los nueve sectores caen a la vez el 79 % de las veces y el VIX sube el 97 %,
un +16,9 % de media.

**El etiquetado hubo que corregirlo, y la comparación es el argumento.** Con las
cinco features iniciales el HMM suspende el control bloqueante: la cobertura del
episodio de Inflación 2022 se queda en el 44,0 % frente al 50 % exigido, porque
`spread_credito_z` deriva 1,17 sigmas entre las dos mitades de la muestra y
`drawdown_sp500` satura —el 47 % de las sesiones caen en su 5 % superior—. Se ve año
a año: esas cinco features encienden la crisis en el **40 % de las sesiones de 2021**,
un año que cerró +26,9 % con volatilidad del 13 %, y solo cubren el 44,0 % de 2022.
Con tres features el control pasa, 2022 sube al 56,5 % y la clase de crisis queda en
el **15,8 %**.

**Entre semillas no converge la verosimilitud: converge la etiqueta.** Las cinco
semillas del catálogo caen en dos óptimos locales distintos —9.418,7 de
log-verosimilitud las semillas 42 y 45 y en torno a 9.417,5 las otras tres, 1,26 nats
de recorrido sobre 3.755 observaciones—. Lo estable es la salida: con tres features
dos semillas cualesquiera coinciden en al menos el **98,9 %** de las sesiones, frente al **52,8 %**
con las cinco iniciales. Gana la semilla 45, con 163 iteraciones del EM.

**Las alternativas al etiquetado están medidas, no descartadas de oídas.** Un HMM de
**dos** estados deja la clase extrema en el **37,4 %** y suspende 1 de los 4 controles,
el del peso de la clase de crisis, que exige la banda [3 %, 20 %]. Y agregar los 21
días por máximo en vez de por moda subiría la crisis del 15,8 % al **22,2 %**: son dos
preguntas distintas —el régimen dominante del mes frente a hubo crisis en algún
momento del mes— y el catálogo se queda con la primera.

**El embargo estaba contado en días naturales y era una fuga real.** 85 días
naturales son 59 sesiones y hacen falta 81: **22 sesiones estaban a la vez en train
y validación**, y otras 22 entre validación y test —con split aleatorio habrían sido
80—. Contado en sesiones, el hueco es de 86 y las sesiones compartidas son **0**.

**El tamaño muestral honesto no son las ventanas.** Las 3.696 ventanas de train son
**45 bloques disjuntos** de 81 sesiones —82,1 ventanas por bloque— y 8 rachas de
crisis; test son 12 bloques y 3 rachas. Por eso el recall de crisis de la
persistencia, 0,800, viene con un intervalo por bootstrap de bloques de
**[0,560 – 1,000]**, tres veces más ancho que el binomial ingenuo. Las rachas también
se cuentan dos veces: 18 en el etiquetado diario y **14** en la etiqueta a 21 días,
que es la que se predice. Y los bloques admiten dos convenios: **45 / 8 / 12** contando
ventanas, que es **el oficial** y está protegido con un `assert` contra
`ventanas.tamano_muestral_efectivo`, y 46 / 9 / 13 contando las sesiones de calendario
que cubre la huella. El notebook 02 imprime los dos para que nadie los reconcilie
creyendo que hay un error de conteo: no lo hay, son dos preguntas distintas.

### 7.2 · La arquitectura, y el control que suspende

De seis candidatas gana `lineal`, una multinomial de **3.603 parámetros** sobre la
ventana aplanada, empatada dentro de una desviación entre semillas con
`cnn_kernel7`, que tiene 271.315 —0,6956 frente a 0,6917, con dispersión de 0,0081—.
Con 45 bloques independientes la convolución no aporta nada medible, y `lineal`
entrena con un factor **22** menos de tiempo por época que `cnn_ancha`: 0,353 s frente
a 7,603 s.

El control bloqueante **suspende**. Sobre test:

| | balanced accuracy | f1-macro | recall crisis |
|---|---|---|---|
| persistencia causal (la barra) | **0,777** | 0,754 | 0,800 |
| arquitectura congelada | 0,575 | 0,499 | 0,591 |
| con `class_weight` | 0,556 | 0,464 | 0,673 |

El suspenso es nítido en la métrica y honesto en la incertidumbre. El modelo congelado
señala **351 crisis donde hay 110**; pero las bandas por bootstrap de bloques **se
solapan** en las dos métricas —balanced accuracy 0,777 [0,681 – 0,860] frente a 0,575
[0,479 – 0,686], 0,005 de solape, y recall de crisis 0,800 [0,560 – 1,000] frente a
0,591 [0,328 – 0,865], 0,305 de solape—. Con 12 bloques de test no hay resolución para
separar a las dos: el control suspende, y publicar la banda al lado es lo que impide
leerlo como una catástrofe y lo que fija el listón que los sintéticos tendrán que
superar de forma medible.

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
bajando hasta 0,314 en la 60, mientras la de validación entra en meseta desde la época
30 —desviación 0,0289 y **seis épocas dentro de una desviación de su mínimo**, el de la
época 38: su argmin no es una época distinguible— y la brecha entre ambas pasa de 0,40
a 0,54 (`results/historiales/baseline_regimen.csv`). Adoptar el encadenado obligaría a que
los generadores produzcan también el régimen de hoy, lo que toca D8, y es una
decisión de diseño pendiente.

La tarea de control sí funciona: la regresión de volatilidad da **MAE 0,046 frente
a 0,050** de su línea base trivial, con R² de 0,164 contra la media de train, QLIKE
0,0716 frente a 0,0826 y **0 de 1.052** predicciones recortadas al piso —D24 exige
mirar ese contador antes de leer el QLIKE—. Bate a la trivial en el 52 % de las
ventanas de test. Que una tarea sin desbalance mejore y la desbalanceada no es, en sí,
un dato sobre dónde está el problema.

### 7.3 · Lo que cuesta el barrido

Con la arquitectura congelada, **0,35 s por época** y **51,0 épocas efectivas medias**
—las suyas en la búsqueda, no la media de las seis candidatas, que baja a 24,2 porque
las CNN paran mucho antes; usar esa media subestimaría el coste por un factor 2,11—:
**3,55 h para los siete generadores** (285 recetas, 570 entrenamientos) y **2,04 h**
para los cuatro del núcleo mínimo (165 recetas, 330 entrenamientos). Los siete siguen
siendo asumibles. Los segundos por época son **tiempo de reloj de la máquina que
ejecutó el cuaderno 03**: esa cifra y `results/metricas/coste_barrido.csv` cambian
entre equipos; ninguna otra cifra de este README lo hace.

El presupuesto que se congela con la arquitectura es 60 épocas, lote 256, paciencia 12
y **parada por balanced accuracy**, no por `val_loss`. Sobre los 22 entrenamientos con
historial completo, haber parado por `val_loss` habría costado 1,8 puntos de media y
hasta 6,6 en el peor caso; pero lo que decide es el reparto: **10 entrenamientos pagan
más de 2 puntos y 5 no pagan nada**, exactamente cero. La media es un sesgo, no una
pérdida esperada.

### 7.4 · Los tres generadores ya entrenados

Los tres se ajustan sobre el mismo bloque de train, **3.696 × 1.201** (60 × 20 + 1),
con el reparto 2.019 / 1.090 / 587 por régimen = 54,63 / 29,49 / 15,88 %.

| Generador | Lo que dice su convergencia | Banco |
|---|---|---|
| `jitter` | σ = 0,1 con desviación relativa 3,466 y **0 duplicados exactos** en todo el barrido de σ | 12.000 × 1.201, 4.000 por régimen |
| `gaussiano` | número de condición con Ledoit-Wolf 6.998 / 4.316 / 1.429 por régimen frente a 1,4 / 1,8 / 1,7 ·10⁶ con la covarianza muestral, y **0 saltos de Cholesky** | 12.000 × 1.201, 4.000 por régimen |
| `cvae` | 65 épocas, mejor validación en la 64 con pérdida total 1.152,15, y **32 de 32 dimensiones latentes activas** | 9.000 × 1.201, 3.000 por régimen |

El `cvae` publica además el dato incómodo que hace falta para el cuaderno 14: su
dispersión sintética se queda corta justo donde importa —ratio 0,700 en calma, 0,588 en
intermedio y **0,355 en crisis**—, y el percentil 99 de volatilidad de crisis baja de
0,830 real a 0,354 sintético. Es exactamente el tipo de fallo que H1 anticipa y que el
barrido tiene que penalizar.

### 7.5 · Las cuatro figuras que sostienen la presentación

Cada uno de los cuatro cuadernos designa una figura de diapositiva. Son estas, y cada
una afirma un hecho con su cifra:

| Figura | Cuaderno | Lo que afirma |
|---|---|---|
| `00_escasez_estructural.png` | 00 | el dataset dice 5.590 ventanas; las crisis entrenables de train son **2** |
| `01_etiquetado_por_anio.png` | 01 | las cinco features encienden la crisis en el **40 %** de 2021 y cubren solo el **44 %** de 2022; con tres, 2022 sube al **56,5 %** |
| `02_frontera_sin_fuga.png` | 02 | sesiones compartidas en la frontera: **80** con split aleatorio, **22** con 85 días naturales, **0** con 85 sesiones |
| `03_veredicto.png` | 03 | el control **suspende** en las dos métricas, y las bandas de bloques se solapan |

Están en `results/figures/` junto a las otras 40, y se leen en ese orden: el problema
es escaso, la etiqueta hubo que ganársela, el corte no filtra y el modelo real no llega.

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
| Entrenamiento de `jitter`, `gaussiano` y `cvae` (notebooks 04, 05 y 06) | ✅ · tres bancos y tres modelos versionados, ver 7.4 |
| Re-ejecución de 04, 05 y 06 sobre el `ventanas.npz` vigente | 🔜 · requisito del barrido, ver 6 |
| Entrenamiento de `cgan`, `flow_matching`, `rbig` y `difusion` (notebooks 07 a 10) | 🔜 |
| Mezclas y barrido experimental (notebooks 11 y 12) | 🔜 |
| Análisis, figuras y calidad de los sintéticos (notebooks 13 y 14) | 🔜 |
| Presentación en PDF | 🔜 |
