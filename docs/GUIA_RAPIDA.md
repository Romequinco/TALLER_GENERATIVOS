# Guía rápida · Taller B5-T1 · Generación de datos financieros sintéticos

Contexto operativo del repositorio. Corto y operativo: lo que hay que saber antes de tocar
nada. La explicación larga está en `README.md` y en `docs/`.

## Qué es este proyecto

Trabajo de grupo del máster MIAX (Oscar Romero Quincoces, Fernando Dapena Tauste y Daniel
García López). La pregunta es si los datos sintéticos mejoran un modelo cuando los datos
reales escasean, sobre un problema donde la escasez es **estructural**: predecir el régimen
de mercado a **21 días** vista, con la clase de crisis en el **15,9 %** de las ventanas de
train y sólo **2 crisis entrenables** por encima del horizonte en todo el tramo de
entrenamiento (`results/metricas/reparto_particiones.csv`; notebook 00).

Hay **dos tareas** sobre la misma ventana de 60 días × 20 canales: clasificación de régimen
a 21 días (principal, desbalanceada) y regresión de volatilidad realizada a 21 días
(control, sin desbalance). Comparar las dos es lo que separa "los sintéticos rebalancean
clases" de "los sintéticos enriquecen la distribución". Siete generadores producen los
datos sintéticos y el barrido mide su efecto sobre esas dos tareas.

## Cómo se arranca

```bash
pip install -r requirements.txt          # Python 3.13, CPU, sin CUDA
```

Los cuadernos usan el kernel **`taller-generativos`** (nombre visible:
`Python (.venv Taller Generativos)`) y **se ejecutan desde `notebooks/`**: la primera celda
hace `sys.path.insert(0, "..")` e importa `src`, que fija el backend de Keras a PyTorch
antes de que nadie importe `keras`. Ejecutarlos desde la raíz rompe los dos supuestos.

Ejecución headless, desde `notebooks/`:

```bash
jupyter nbconvert --to notebook --execute --inplace \
    --ExecutePreprocessor.timeout=7200 03_downstream_baseline.ipynb
```

El tope de tiempo por celda hay que subirlo: el valor por defecto de nbconvert son 30 s y la
búsqueda de arquitectura del cuaderno 03 son 18 entrenamientos en una sola celda. El kernel
sale de la metadata del propio `.ipynb`, no hace falta pasarlo.

Todo está dimensionado para **CPU** (D19): no se ha usado GPU en ningún momento.

`config.cargar_catalogo` está cacheado por sesión. Si se edita `data/catalog.yaml` con el
kernel vivo hay que reiniciarlo o llamar a `config.cargar_catalogo.cache_clear()`, como hace
la segunda celda del cuaderno 00; si no, se regenera todo con la versión antigua y en
silencio.

## Reglas duras

Son las que más se incumplen. Cada una tiene su decisión en `docs/DECISIONES.md`.

- **`data/catalog.yaml` es la fuente de verdad.** Universo, canales, geometría de ventanas,
  splits, semillas y política de huecos viven ahí. **Ningún módulo de `src/` lleva tickers,
  fechas ni longitudes de ventana escritos a mano**, y ningún cuaderno tampoco. Cambiar un
  parámetro del catálogo y re-ejecutar 00 y 02 basta para regenerar el dataset.
- **`data/raw/` no se toca y no se versiona** (D18). yfinance reajusta dividendos hacia
  atrás, así que dos descargas separadas no devuelven los mismos precios: `precios.parquet`
  se cachea una sola vez y lo reproducible es el pipeline **desde `data/processed/`**. Las
  dos huellas de contenido que imprime el cuaderno 00 son el mecanismo para comprobar que
  dos personas trabajan sobre los mismos números.
- **Los datos se auditan, no se limpian** (D23). Nada de imputación, winsorización, recorte
  de atípicos, suavizado ni remuestreo. `src/calidad.py` aplica once controles de
  integridad: cinco invariantes que lanzan excepción y seis avisos que nunca abortan.
- **El split es temporal y el embargo se cuenta en SESIONES de mercado, nunca en días
  naturales** (D7). Son 85 sesiones. Contarlas como días naturales fue una fuga real: 85
  días naturales son 59 sesiones donde hacen falta 81, y dejaban 22 sesiones a la vez en
  train y validación.
- **Generadores, etiquetado de regímenes y escalado se ajustan sólo con train.**
  **Validación y test son siempre 100 % reales**, sin una sola muestra sintética.

## La convención de cifras

**Las celdas markdown de Jupyter no interpolan variables de Python**, y en este repositorio
no hay ningún mecanismo que lo haga. De ahí la regla que evita que la documentación se
desincronice de la ejecución:

- **Toda cifra medida vive en un f-string del código**: título de figura, anotación sobre el
  eje o `print()`. Ahí se interpola desde la variable viva y no puede quedarse obsoleta.
- **El markdown afirma la conclusión sin el número**, o remite a la salida de la celda
  siguiente. Nada de transcribir cifras a mano.
- Excepción única: los **parámetros de diseño** del catálogo —60 días de ventana, horizonte
  21, tres estados, embargo 85— pueden ir en markdown, porque no son mediciones.
- No se usa `display(Markdown(f"..."))`: el visor de `.ipynb` de GitHub no garantiza el
  renderizado de salidas `text/markdown`, y el 20 % de la nota es que el repositorio se lea
  en GitHub.

Lo mismo vale para la documentación: **ninguna cifra sin origen**. Toda cifra de `README.md`
o de `docs/` tiene que poder trazarse a la salida de un cuaderno ejecutado, a un CSV de
`results/metricas/` o a un historial de `results/historiales/`.

## Las figuras las genera el código

El enunciado exige que el código genere todas las gráficas y tablas reportadas. Por eso toda
figura sale de `src/viz.py` —sus funciones de figura, o código inline que usa su paleta
pública: `viz.aplicar_estilo()`, `viz.PALETA`, `viz.rampa_secuencial(n)`, `viz.guardar(...)`—
y se escribe en `results/figures/` con `viz.guardar`. Nada de imágenes montadas a mano.

Convenciones vigentes: los tres regímenes se pintan siempre con `viz.rampa_secuencial(3)`
(calma claro → crisis oscuro) y las particiones con `viz.PALETA[0..2]`. Las figuras creadas
en la reescritura de los cuadernos 00 a 03 llevan el número de su cuaderno como prefijo
(`00_`, `01_`, `02_`, `03_`); las anteriores conservan su nombre heredado porque hay
documentación que las referencia por él, y por eso el cuaderno 01 guarda la comparación
causal-frente-a-Viterbi dos veces, con el nombre nuevo y con el heredado.

Cada uno de los cuadernos 00 a 03 designa **una** figura candidata a diapositiva
(`00_escasez_estructural`, `01_etiquetado_por_anio`, `02_frontera_sin_fuga`,
`03_veredicto`): el 80 % de la nota es una presentación de cinco minutos.

## Dónde está cada cosa

```
data/catalog.yaml   fuente de verdad      src/          diez módulos + src/generadores/
data/processed/     punto de entrada real notebooks/    00 a 14, encadenados
data/synthetic/     bancos por generador  results/      figures/ metricas/ historiales/
models/             generadores/ downstream/            docs/  teoría, metodología, decisiones
```

- **Por qué el proyecto es así** → `docs/DECISIONES.md` (D1 a D24). Léelo antes de tocar nada.
- **Cómo se define y se estima el régimen** → `docs/metodologia/etiquetado_regimenes.md`.
- **Cómo se juzga un dataset sintético y qué puede salir mal** →
  `docs/metodologia/metricas_calidad_sintetica.md` y `riesgos_datos_sinteticos.md`, que
  incluye el checklist anti-fuga.
- **Teoría de cada familia generativa** → `docs/teoria/`, un documento por familia.
- **Estado, cifras medidas y arranque detallado** → `README.md`, secciones 6 a 8.
- **El mapa completo de la documentación** → `docs/INDEX.md`.

## Estado y trabajo de terceros

Ejecutados: cuadernos **00 a 06**. Sin ejecutar: **07 a 14**. `data/synthetic/` trae los
bancos de `jitter`, `gaussiano` y `cvae`, y `models/generadores/` sus tres modelos.

**Este repositorio es de tres personas.** Cada bloque de cuadernos tiene su responsable dentro
del grupo: los cuadernos 04 en adelante y sus artefactos —`data/synthetic/`,
`models/generadores/` y sus figuras— no se editan ni se re-ejecutan sin acordarlo antes.

Apunte de trazabilidad abierto: los tres bancos se ajustaron sobre una versión anterior de
`data/processed/ventanas.npz`. Las etiquetas no se movieron ni un bit y sólo cambió `X` en
el cuarto decimal de unidades ya escaladas, pero el barrido del cuaderno 12 tiene que salir
de un único `ventanas.npz`, así que 04, 05 y 06 hay que re-ejecutarlos antes. Está contado
en `README.md` § 6.
