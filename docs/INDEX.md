# Índice de la documentación

Toda la documentación del taller, organizada por para qué sirve. La teoría se
escribió antes que el código y es la referencia técnica del equipo; la
metodología fija cómo se mide; las decisiones explican por qué el proyecto es
como es.

## Punto de partida

| Documento | Para qué |
|---|---|
| [`GUIA_RAPIDA.md`](GUIA_RAPIDA.md) | Contexto operativo en una página: qué es el proyecto, cómo se arranca, las reglas duras y dónde está cada cosa. **Es lo primero que hay que leer** |
| [`../README.md`](../README.md) | Visión general, arranque rápido, cifras medidas y estado del proyecto |
| [`DECISIONES.md`](DECISIONES.md) | Todas las decisiones de diseño, numeradas D1 en adelante, con su justificación. **Léelo antes de tocar nada** |
| [`enunciado/Taller_B5_T1.pdf`](enunciado/Taller_B5_T1.pdf) | Enunciado oficial: requisitos de la entrega y criterios de evaluación |

## Teoría de modelos generativos

Un documento por familia. Todos siguen la misma estructura —intuición,
formulación, arquitectura, pérdida, **diagnóstico de convergencia**, patologías,
aplicación a nuestro problema, implementación de referencia— para poder
compararlos sección a sección.

| Documento | Familia | Generador asociado |
|---|---|---|
| [`teoria/00_fundamentos.md`](teoria/00_fundamentos.md) | Marco general: densidad explícita vs implícita, cambio de variable, cómo se evalúa | — |
| [`teoria/01_baselines_generativos.md`](teoria/01_baselines_generativos.md) | Ruido, gaussiano multivariante y bootstrap por bloques | `jitter`, `gaussiano` |
| [`teoria/02_gans.md`](teoria/02_gans.md) | Adversarial, incluida la variante condicional | `cgan` |
| [`teoria/03_vae.md`](teoria/03_vae.md) | Latente variacional | `cvae` |
| [`teoria/04_normalizing_flows_rbig.md`](teoria/04_normalizing_flows_rbig.md) | Flujos normalizadores y gaussianización iterativa | `rbig` |
| [`teoria/05_difusion.md`](teoria/05_difusion.md) | Difusión, DDPM y muestreo DDIM | `difusion` |
| [`teoria/06_flow_matching.md`](teoria/06_flow_matching.md) | Transporte continuo por ODE | `flow_matching` |
| [`teoria/07_autoregresivos.md`](teoria/07_autoregresivos.md) | Factorización por regla de la cadena | ampliación opcional |
| [`teoria/08_sota_series_financieras.md`](teoria/08_sota_series_financieras.md) | Estado del arte específico de series financieras y hechos estilizados | — |

## Metodología

| Documento | Qué fija |
|---|---|
| [`metodologia/etiquetado_regimenes.md`](metodologia/etiquetado_regimenes.md) | Cómo se define y se estima el régimen: HMM, número de estados, canonicalización, agregación al horizonte y controles de aceptación |
| [`metodologia/metricas_calidad_sintetica.md`](metodologia/metricas_calidad_sintetica.md) | Cómo se juzga un dataset sintético: utilidad (TSTR), fidelidad, diversidad y detección de memorización |
| [`metodologia/riesgos_datos_sinteticos.md`](metodologia/riesgos_datos_sinteticos.md) | Qué puede salir mal: colapso, memorización, sesgos amplificados y, sobre todo, el **checklist anti-fuga** del diseño experimental |

## Registros de decisión

`decisions/` recoge las decisiones estructurales que cambian el rumbo del
proyecto, en formato ADR: contexto, problema, opciones consideradas y decisión
adoptada. `DECISIONES.md` es el resumen operativo; los ADR guardan el
razonamiento largo.

Hoy hay uno:

| Documento | Qué fija |
|---|---|
| [`decisions/ADR-001-problema-y-diseno-experimental.md`](decisions/ADR-001-problema-y-diseno-experimental.md) | Por qué el problema es la predicción de régimen y no el ejemplo del material guiado, y qué diseño experimental se deriva de esa elección. De aquí salen D1, D2, D3, D5 y D13 |

## Material de clase

`material_clase/` contiene las diapositivas y notebooks originales del máster.
**No está versionado**: es material ajeno y ocupa unos 90 MB. Quien lo necesite
lo descarga del aula virtual y lo coloca ahí.

- `material_clase/slides/` — los seis PDF de teoría más la presentación del taller
- `material_clase/notebooks/` — los 24 notebooks de clase

El enunciado sí se versiona, en `enunciado/`, porque es la especificación de lo
que hay que entregar.

## Cómo leer esto según lo que vayas a hacer

**Abres el repositorio por primera vez.** Lee [`GUIA_RAPIDA.md`](GUIA_RAPIDA.md) —una
página: el problema, el arranque, las reglas duras y el mapa— y después las
secciones 5 a 8 de [`../README.md`](../README.md), que dan la estructura, el
punto de entrada real (`data/processed/`, notebook 03), las cifras ya medidas y
el estado por fases.

**Vas a escribir o re-ejecutar un cuaderno.** La convención que hay que respetar
está en `GUIA_RAPIDA.md`: las celdas markdown de Jupyter no interpolan variables, así
que toda cifra medida vive en un f-string del código —título de figura,
anotación o `print()`— y el markdown afirma la conclusión sin el número. Las
figuras salen de `src/viz.py` o de código inline que usa su paleta pública, y
las nuevas llevan el número del cuaderno como prefijo.

**Vas a tocar los datos.** Lee **D23** de `DECISIONES.md` —los datos se auditan,
no se limpian— y `src/calidad.py`, que es el módulo que gobierna qué se le hace y
qué no se le hace al panel: once controles de integridad, cinco **invariantes**
que lanzan excepción y seis **avisos** que nunca abortan, y la misma auditoría
sirve para un panel sintético en el notebook 14. Ninguna técnica de limpieza
—imputación, winsorización, recorte de atípicos, suavizado, remuestreo— entra en
este repositorio: D23 mide, una por una, qué destruyen de lo que el trabajo
pretende medir. La política de huecos de `data/catalog.yaml` es la otra mitad de
la misma historia.

**Vas a entrenar un generador.** Lee `DECISIONES.md`, el documento de teoría de
tu familia, y `src/generadores/base.py`. Las secciones de "diagnóstico de
convergencia" y "aplicación a nuestro problema" son las que necesitas. `jitter`,
`gaussiano` y `cvae` ya están entrenados y sus cifras de convergencia están en
`../README.md` § 7.4: sirven de referencia de qué se publica al cerrar un
generador. Antes del barrido hay que re-ejecutar los notebooks 04, 05 y 06 sobre
el `ventanas.npz` vigente, por el motivo que explica `../README.md` § 6.

**Vas a montar los datasets o el barrido.** Lee `DECISIONES.md` (en especial D7,
D13 y D14 para montarlo, y D21 y D24 para saber cómo se leen las cifras que
salen) y el checklist anti-fuga de `metodologia/riesgos_datos_sinteticos.md`. El
coste medido del barrido está en `../README.md` § 7.3 y en
`results/metricas/coste_barrido.csv`; los segundos por época son tiempo de reloj
de la máquina que ejecutó el notebook 03 y cambian entre equipos.

**Vas a preparar la presentación.** `DECISIONES.md` tiene la justificación de
cada elección, y las secciones de aplicación de cada documento de teoría tienen
los argumentos técnicos para la defensa. `teoria/08_sota_series_financieras.md`
sitúa el trabajo frente al estado del arte. Cada uno de los notebooks 00 a 03
designa una figura de diapositiva —las cuatro, con el hecho que sostiene cada
una, en `../README.md` § 7.5—, y el veredicto que hay que saber defender, con su
banda de incertidumbre, está en § 7.2.
