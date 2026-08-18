# Índice de la documentación

Toda la documentación del taller, organizada por para qué sirve. La teoría se
escribió antes que el código y es la referencia técnica del equipo; la
metodología fija cómo se mide; las decisiones explican por qué el proyecto es
como es.

## Punto de partida

| Documento | Para qué |
|---|---|
| [`../README.md`](../README.md) | Visión general, arranque rápido y reparto del trabajo |
| [`DECISIONES.md`](DECISIONES.md) | Las 22 decisiones de diseño con su justificación. **Léelo antes de tocar nada** |
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

## Material de clase

`material_clase/` contiene las diapositivas y notebooks originales del máster.
**No está versionado**: es material ajeno y ocupa unos 90 MB. Quien lo necesite
lo descarga del aula virtual y lo coloca ahí.

- `material_clase/slides/` — los seis PDF de teoría más la presentación del taller
- `material_clase/notebooks/` — los 24 notebooks de clase

El enunciado sí se versiona, en `enunciado/`, porque es la especificación de lo
que hay que entregar.

## Cómo leer esto según lo que vayas a hacer

**Vas a entrenar un generador.** Lee `DECISIONES.md`, el documento de teoría de
tu familia, y `src/generadores/base.py`. Las secciones de "diagnóstico de
convergencia" y "aplicación a nuestro problema" son las que necesitas.

**Vas a montar los datasets o el barrido.** Lee `DECISIONES.md` (en especial D7,
D13 y D14) y el checklist anti-fuga de `metodologia/riesgos_datos_sinteticos.md`.

**Vas a preparar la presentación.** `DECISIONES.md` tiene la justificación de
cada elección, y las secciones de aplicación de cada documento de teoría tienen
los argumentos técnicos para la defensa. `teoria/08_sota_series_financieras.md`
sitúa el trabajo frente al estado del arte.
