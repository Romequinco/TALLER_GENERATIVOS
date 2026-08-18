# Decisiones de diseño

Registro de las decisiones que condicionan el resto del proyecto, con su
justificación. Sirve para dos cosas: que los tres integrantes trabajen sobre los
mismos supuestos, y tener preparadas las respuestas de la defensa técnica.

Cada decisión indica si nos apartamos del material guiado del máster y por qué.

---

## D1 · El problema: predicción de régimen, no detección

**Decisión.** La variable objetivo es el régimen que **dominará los próximos 21
días**, no el régimen actual.

**Por qué.** Detectar el régimen presente con un HMM es un problema resuelto y
casi tautológico: las features de entrada y las de etiquetado serían las mismas.
Predecir el régimen futuro a partir de la ventana pasada es un problema real,
difícil, y con valor práctico para la gestión de riesgo.

**Implicación.** La etiqueta mira al futuro y eso es legítimo: es el objetivo,
no una entrada. Lo que nunca puede mirar al futuro son las features de `X`.

**Límite que hay que reconocer.** El generador aprende la distribución conjunta
de `(ventana pasada, régimen futuro)` del histórico. No es un simulador
prospectivo de mercado y no debe presentarse como tal.

---

## D2 · Dos tareas downstream en lugar de una

**Decisión.** Además de la clasificación de régimen se resuelve una regresión de
volatilidad realizada, sobre la misma ventana y con la misma troncal.

**Por qué.** Actúa como control experimental. La hipótesis del trabajo es que
los sintéticos ayudan **porque compensan el desbalance de clases**. La tarea de
volatilidad no tiene desbalance. Si los sintéticos ayudan solo en la
clasificación, la hipótesis se sostiene; si ayudan en ambas, el mecanismo es
otro y hay que decirlo.

**Coste.** Bajo: comparten pipeline de datos, generadores y bancos de muestras.
Solo se duplica el barrido de entrenamiento.

---

## D3 · Panel híbrido de 20 canales, no 23 tickers en crudo

**Decisión.** `X` combina once canales de retorno —índice, nueve sectores y
dólar— con nueve features derivadas de estrés (nivel y variación del VIX,
volatilidad realizada, drawdown, momento, spread de crédito, pendiente de curva,
correlación acción-bono y dispersión sectorial).

**Por qué.** Los notebooks guiados usan 23 tickers en crudo, lo que da un bloque
de 1.380 dimensiones donde la GAN densa del ejemplo se atasca visiblemente. Las
features derivadas concentran la señal de régimen y hacen el problema aprendible
con menos dimensiones.

**Medido sobre este panel**, no heredado de otro: el notebook 00 calcula el AUC
univariante de cada canal contra el decil superior de volatilidad futura, sobre
el tramo de entrenamiento. Seis canales superan 0,70 —`vix_nivel_z` 0,90,
`vol_realizada_z` y `drawdown_sp500` 0,88, `spread_credito_z` 0,86,
`dispersion_sectorial` 0,82 y `corr_accion_bono` 0,73— y **son todos derivados**;
el mejor retorno se queda en 0,55. La banda bootstrap de estos AUC ronda ±0,15,
así que lo defendible es la separación entre los dos bloques, no el orden entre
canales vecinos.

Una versión anterior de esta decisión citaba `MOVE_level_z` con AUC 0,80. Esa
cifra procede del panel del TFM de regímenes y **`MOVE` no está en el universo de
este taller**, de modo que se retira y se sustituye por la medición de arriba.

**Alternativa descartada.** Solo features derivadas (sin sectores): pierde la
estructura transversal, que es informativa en las rotaciones sectoriales previas
a las crisis.

---

## D4 · Solo yfinance, sin claves de API

**Decisión.** Todas las series se descargan de yfinance. Se renuncia a FRED, que
daría mejores proxies de crédito y curva.

**Por qué.** Cualquiera de los tres puede reconstruir los datos brutos sin
configurar credenciales. En un trabajo de dos semanas a tres manos, esa fricción
cuesta más de lo que aportaría la calidad extra de las series.

**Consecuencia.** El spread de crédito es un proxy (comportamiento relativo de
`LQD` frente a `IEF`) y no el diferencial BAA-10Y directo.

---

## D5 · Ventana 2003-2026

**Decisión.** El periodo arranca en 2003, gobernado por los ETF de renta fija
(`TLT`, `IEF`, `LQD`, disponibles desde julio de 2002).

**Por qué.** Es la "pista B" de nuestro TFM: panel rico a cambio de menos
historia. Cubre 2008, 2011, 2020, 2022 y 2023, que son episodios de naturaleza
distinta entre sí.

**Alternativa descartada.** Serie profunda desde 1927 con solo el índice: más
crisis (22) pero sin sectores, VIX ni crédito, que son justo los canales con más
poder discriminante.

---

## D6 · Etiquetado con HMM gaussiano de tres estados

**Decisión.** HMM gaussiano (`hmmlearn`), tres estados, ajustado solo con train,
cinco semillas y se conserva el de mayor log-verosimilitud.

**Por qué tres y no dos.** Con dos estados el corte calma/crisis deja una clase
minoritaria de en torno al 37 %, demasiado poblada para que el desbalance sea el
problema central. Con tres, el estado extremo queda en el **16,1 %** y separa la
tensión genuina de la simple transición.

**Tres features de etiquetado, no cinco.** La versión inicial usaba `ret_sp500`,
`vol_realizada_z`, `vix_nivel_z`, `drawdown_sp500` y `spread_credito_z`. **Suspende
el control bloqueante del notebook 01**: la clase de crisis sale al 39 % y el
modelo marca **2021 entero** como crisis, el año en que el S&P 500 subió un 26,9 %
con volatilidad del 13 %.

La causa es que un HMM gaussiano supone emisiones normales y estacionarias dentro
de cada estado, y dos de las cinco lo incumplen: `spread_credito_z` **deriva**
—1,17 sigmas de salto entre las dos mitades de la muestra, porque el z-score
expanding no vuelve nunca de 2008— y `drawdown_sp500` **satura**, con el 47 % de
las sesiones en el 5 % superior de su recorrido. Con ellas dentro, el estado
extremo deja de significar "mercado en tensión" y pasa a significar "estamos
después de 2020".

Retirándolas, la crisis baja al 16,1 %, el control pasa, y —lo que pesa más— el
ajuste deja de depender de la inicialización: las 20 semillas probadas convergen a
la misma solución, mientras que con las cinco el segundo óptimo local daba un
21,2 % y habría suspendido. Un etiquetado que cambia de significado según la
semilla no sirve como variable objetivo. Las dos descartadas siguen en el catálogo
bajo `features_descartadas`, porque el notebook 01 reconstruye con ellas el ajuste
rechazado: la comparación **es** el argumento.

**Nota de alcance.** Que una feature no sirva para *etiquetar* no dice nada sobre
si sirve como *entrada*. `drawdown_sp500` y `spread_credito_z` siguen siendo dos
de los seis canales con AUC por encima de 0,70 y se quedan en `X`. Lo que las
descalifica es el supuesto gaussiano del HMM, no su contenido informativo.

**Canonicalización.** Los estados se ordenan por volatilidad creciente, con el
retorno medio solo como desempate dentro de bandas de anchura `0.15 × vol
media`. Criterio heredado del TFM, donde se comprobó que ordenar por retorno
medio intercambia calma y crisis de forma errática entre ejecuciones.

---

## D7 · Split temporal con embargo de 85 sesiones

**Decisión.** Particiones por fecha, con 85 **sesiones de mercado** descartadas
en cada frontera. Nunca `train_test_split` aleatorio.

**Corrección (notebook 02).** El código aplicaba el embargo con
`pd.Timedelta(days=85)`, es decir en **días naturales**, mientras el catálogo y
esta decisión decían "sesiones". No es lo mismo: 85 días naturales son 59
sesiones y hacen falta 81. Medido: **22 sesiones quedaban simultáneamente en
train y en validación**, y otras 22 entre validación y test —enero y febrero de
2022, el episodio principal del tramo de test—. Era una fuga real que habría
invalidado el experimento.

Se corrige contando el embargo en **posiciones del índice de mercado**. El valor
85 no cambia; cambia la unidad en que se lee, y el campo pasa a llamarse
`embargo_sesiones`. La alternativa —subir los días naturales hasta que el hueco
salga— se descarta con datos: 119 días **no llegan al mínimo en el 2,5 % de las
fechas de corte posibles** del panel y 125 descartan más ventanas que la opción
en sesiones. Un embargo que funciona según dónde caigan los festivos es una
coincidencia, no una garantía. `partir` ahora lanza si el embargo es menor que el
solape, que es la comprobación que habría impedido el fallo.

**Por qué.** Las ventanas se desplazan un día y comparten 59 de 60 observaciones
con su vecina. Con split aleatorio, prácticamente toda muestra de test tiene un
cuasi-duplicado en train, y las métricas miden memorización.

**Nos apartamos del material guiado.** `Taller_Gaussian_solution.ipynb` y
`Taller_GANs.ipynb` usan `train_test_split(..., random_state=42)` con `shuffle`
por defecto sobre estas mismas ventanas solapadas. Es un punto que conviene
mencionar en la defensa: no es una crítica al ejemplo, que persigue ilustrar el
mecanismo, sino la razón por la que nuestras cifras no son comparables con las
suyas y sí son defendibles.

**Valor del embargo.** El mínimo teórico es `60 + 21 = 81`; se adopta 85 por
margen.

---

## D8 · Generación del bloque conjunto, no solo de las entradas

**Decisión.** Los generadores modelan `[X aplanada ; y_vol]` condicionado a
`y_reg`, no solo `X`.

**Por qué.** Es la opción OPT2 del planteamiento del taller: cada muestra
sintética viene con su objetivo coherente, sin necesidad de etiquetarla después
con un modelo auxiliar (que introduciría su propio error).

**Por qué el régimen va como condición y no dentro del bloque.** Si la etiqueta
fuera una dimensión más del vector generado, la proporción de clases sintéticas
replicaría el desbalance del train y **no se podría sobre-generar la clase de
crisis**, que es el objetivo del trabajo. Al condicionar, la etiqueta es exacta
por construcción y su distribución es una decisión de diseño.

Nota: el notebook `GAN_1_..._etiquetas_y_balanceo.ipynb` del material de clase,
pese al título, anexa la etiqueta como columna del vector de datos. No es una
GAN condicional en sentido estricto y no sirve para nuestro propósito.

---

## D9 · Siete generadores en lugar de cuatro

**Decisión.** Se implementan `jitter`, `gaussiano`, `cvae`, `cgan`, `rbig`,
`flow_matching` y `difusion`.

**Por qué.** El enunciado pide tres generativos más uno simple. Con siete se
comparan familias enteras (estadística, latente, adversarial, flujo, difusión) y
la conclusión deja de depender de qué tan bien se ajustó una instancia concreta.

**Prioridad si el calendario aprieta.** El núcleo mínimo que cumple el enunciado
es `jitter` + `gaussiano` + `cvae` + `cgan`. `rbig` es barato y añade la familia
de flujos. `flow_matching` y `difusion` son los candidatos a caer.

**Expectativa documentada.** La literatura sugiere que en nuestro régimen —pocos
datos, CPU, dimensión moderada— el cVAE es más fiable que el cGAN, y así se
plantea: el cVAE como caballo principal y el cGAN como contraste, no al revés.

---

## D10 · Un generador por régimen en los modelos no condicionables

**Decisión.** `gaussiano` y `rbig` no admiten condicionamiento nativo: se ajusta
un modelo independiente por régimen.

**Riesgo asumido.** En la clase de crisis quedan pocos cientos de ventanas para
un bloque de ~1.200 dimensiones. Ambos módulos avisan por consola cuando el
número de muestras es insuficiente, y `rbig` reduce dimensión con PCA previa.

**Consecuencia que hay que tener presente al juzgar el resultado.** Un modelo por
régimen no es una gaussiana: el banco de muestras que produce es una **mezcla de
escala**, porque el régimen es constante dentro de cada ventana y cambia entre
ventanas. Y una mezcla sí tiene colas gruesas y, bajo el protocolo de
autocorrelación agrupada, sí tiene agrupamiento de volatilidad. Medido en el
notebook 00 con el reparto de regímenes que espera D6: curtosis en torno a 8 y
ACF de |r| en el primer retardo de 0,31, frente a 16,5 y 0,31 reales.

Por eso sería un error enunciar "el generador gaussiano no puede reproducir las
colas ni el agrupamiento": eso vale para una gaussiana única, que no es lo que
este repositorio implementa, y el notebook 05 nos refutaría. El estadístico que
**sí** separa a la mezcla del mercado es la curtosis de los residuos
estandarizados: 7,3 reales con banda [5,7 – 11,2], frente a 3,4 de la mezcla. Es
sobre esa fila sobre la que el notebook 00 enuncia su hipótesis falsable.

---

## D11 · Shrinkage obligatorio en el generador gaussiano

**Decisión.** La covarianza se estima por defecto con Ledoit-Wolf, y la
factorización de Cholesky actúa como puerta dura.

**Por qué.** Con `n < d` la covarianza muestral es singular por construcción.
`numpy.random.multivariate_normal` **no advierte de nada** en ese caso: devuelve
muestras que son combinaciones afines exactas de los datos de entrenamiento, sin
energía fuera de su span. Serían copias disfrazadas, no muestras nuevas.
`np.linalg.cholesky` sí lanza excepción, y por eso se usa como control.

**Efecto secundario buscado.** El coeficiente de shrinkage decrece
monótonamente con el número de muestras, lo que lo convierte en el diagnóstico
de convergencia de este generador, que no tiene curva de pérdida.

---

## D12 · Amplitud del ruido del jitter relativa, no absoluta

**Decisión.** `sigma` se expresa como fracción de la desviación típica de cada
dimensión. Valor por defecto 0.1.

**Por qué.** El `sig = 0.01` del notebook de clase es un valor absoluto sobre
log-retornos sin estandarizar. Nuestro bloque mezcla retornos diarios y
z-scores, cuyas escalas difieren en órdenes de magnitud: un sigma absoluto
ahogaría unos canales y no tocaría otros.

**Rango defendible.** Entre 0.05 y 0.15, acotado por la atenuación de las
correlaciones, la dilución de la curtosis y la distancia al vecino más cercano
(por debajo del rango se generan duplicados).

---

## D13 · Dos ejes de barrido, no uno

**Decisión.** Se barre la proporción de sintéticos **y** el número de datos
reales.

**Por qué.** El resultado central del material del taller es que el beneficio
del sintético depende del volumen de datos reales: grande con 500, nulo o
negativo con 20.000. Con un solo eje y el dataset completo, el resultado sería
una línea plana y la conclusión, equivocada.

---

## D14 · Dos políticas de reparto por clase

**Decisión.** `proporcional` (replica el desbalance) y `equilibrado` (concentra
el sintético en la clase minoritaria).

**Por qué.** Aísla el mecanismo. `proporcional` mide el efecto de "más datos";
`equilibrado`, el de "más datos donde hacen falta". La diferencia entre ambas es
la contribución atribuible al rebalanceo.

---

## D15 · Comparación obligatoria contra reponderación de clases

**Decisión.** El barrido incluye una versión con `class_weight` inverso a la
frecuencia y sin ningún dato sintético.

**Por qué.** Es la alternativa clásica y gratuita al desbalance. Si reponderar
la pérdida iguala al mejor generador, los datos sintéticos no aportan nada que
no se consiguiera con una línea de código, y el trabajo honesto es decirlo.

---

## D16 · Test de memorización obligatorio

**Decisión.** Cada generador se somete a un test de distancia al vecino más
cercano frente al conjunto de entrenamiento.

**Por qué.** Los episodios de crisis independientes son del orden de una decena.
Un generador puede aparentar buen rendimiento simplemente reproduciendo ventanas
de 2008 y 2020. Si el cociente entre la distancia sintético-real y la distancia
real-real cae muy por debajo de 1, está copiando, y cualquier mejora downstream
es un artefacto.

**Espacio de comparación.** Las distancias se calculan sobre una proyección PCA
de 50 componentes ajustada solo con los reales de entrenamiento. En las ~1.200
dimensiones originales todas las distancias entre pares convergen al mismo valor
y el vecino más cercano deja de significar nada.

**Comportamiento esperado del jitter.** Por construcción perturba muestras
reales, así que suspenderá este test: su cociente de distancias estará muy por
debajo de 1 aunque su fidelidad distribucional sea excelente. Eso no es un fallo
del test sino su demostración, y es el argumento que explica por qué un baseline
puede ganar en las métricas de parecido y aun así no aportar información nueva.

---

## D17 · La métrica primaria es el recall de crisis, no el accuracy

**Decisión.** Se reporta F1-macro, balanced accuracy y recall de la clase de
crisis. El accuracy se incluye solo por completitud.

**Por qué.** Con la clase de crisis en torno al 10 %, un modelo que nunca la
prediga acierta el 90 %. El accuracy no distingue ese modelo inútil de uno
valioso.

---

## D18 · Persistencia de todo lo caro

**Decisión.** `data/processed/`, `data/synthetic/` y `models/generadores/` se
versionan. Los historiales de entrenamiento se guardan en CSV.

**Por qué.** Es lo que permite que cada integrante entrene solo su generador y
que cualquiera regenere una figura del informe sin reentrenar. Con varios
cientos de entrenamientos en CPU, reentrenar para rehacer un gráfico no es
viable.

**Excepción.** `data/raw/` no se versiona: se reconstruye con una llamada.

---

## D19 · Todo dimensionado para CPU

**Decisión.** No se asume GPU en ningún punto. Las arquitecturas son MLP
pequeños; la difusión usa DDIM con pocos pasos y no una U-Net.

**Por qué.** El equipo no dispone de CUDA. Diseñar para GPU y descubrirlo a
mitad de camino habría costado el calendario entero.

**Consecuencia asumida.** Los generadores no están en el estado del arte. El
trabajo compara familias de modelos en igualdad de condiciones, no busca el
mejor generador posible de cada familia.

---

## D20 · Una sola arquitectura downstream, congelada

**Decisión.** La CNN 1D se busca en el notebook 03 con datos reales y no se
vuelve a tocar.

**Por qué.** Lo exige el enunciado y es lo único que hace honesta la
comparación: si la arquitectura cambiara entre versiones, no se sabría si la
diferencia viene de los datos o del modelo.

**Control adicional.** El presupuesto de optimización (épocas y criterio de
parada temprana) también se fija, para no confundir "más entrenamiento" con
"mejores datos".

---

## D21 · Línea base de persistencia

**Decisión.** Toda métrica de la tarea de régimen se compara contra el
predictor trivial "el régimen futuro será el mismo que el actual".

**Por qué.** Los regímenes son muy persistentes: las probabilidades de
permanencia en la diagonal de la matriz de transición del HMM rondan 0.94-0.98.
Eso convierte a la persistencia en una línea base fortísima. Un clasificador que
no la bata no está prediciendo nada: se está limitando a leer el estado
presente, que es un problema distinto y mucho más fácil.

**Consecuencia para el análisis.** Las mejoras por datos sintéticos se reportan
sobre el margen respecto a la persistencia, no sobre el cero absoluto.

---

## D22 · Lotes grandes en todos los entrenamientos

**Decisión.** Los generadores neuronales usan lotes de 256-512, no los 32-128
habituales. No se bajan sin medir antes.

**Por qué.** En CPU el coste por iteración está dominado por sobrecarga fija por
operación, no por el número de operaciones aritméticas. Con lotes pequeños las
rutinas de álgebra lineal no llegan a llenarse y la máquina pasa más tiempo
gestionando que calculando. Dos mediciones independientes en esta misma máquina:

| Modelo | Lote 128 | Lote 512-1024 |
|---|---|---|
| Difusión | 76 muestras/s | 1.095 muestras/s (×14) |
| cGAN | 137 ms/iteración | 195 ms/iteración (×2,6 en muestras/s) |

En la difusión, la primera versión con lote 128 daba 59 s por época, es decir
3,3 horas de entrenamiento. Con lote 512 y el mismo modelo son 20-35 minutos.

**Efecto secundario.** Un lote grande da gradientes menos ruidosos, lo que en
una GAN reduce la inestabilidad del juego adversarial. Aquí no hay que
compensar la pérdida de ruido de gradiente: con estos tamaños de dataset no
estamos en el régimen donde ese ruido regulariza.

**Aviso relacionado.** Subir el número de hilos de PyTorch por encima del número
de núcleos **físicos** empeora los tiempos. En esta máquina, `set_num_threads(4)`
sobre 2 núcleos físicos es más lento que dejarlo por defecto.

---

## D23 · Los datos se auditan, no se limpian

**Decisión.** El panel pasa diez controles de integridad en `src/calidad.py`, y
**no se le aplica ninguna técnica de limpieza**: ni imputación, ni winsorización,
ni recorte de valores atípicos, ni suavizado, ni remuestreo.

**Por qué.** No es purismo: las técnicas habituales destruyen justo lo que el
trabajo mide. Medido sobre este panel:

| técnica | efecto medido |
|---|---|
| `ffill` **(A)** sobre rejilla hábil | +222 sesiones inventadas (+3,7 %), los retornos nulos de `sp500` pasan de 3 a 225 y la volatilidad realizada se sesga −2,0 % |
| winsorizar al 1 % | curtosis 16,5 → **6,4**, por debajo de la banda bootstrap del propio panel [7,3 – 21,5] **y por debajo del 8,0 que produce nuestro propio generador gaussiano** |
| winsorizar al 5 % | `curtosis_residuos` 7,3 → **5,3**, por debajo de la banda [5,7 – 11,2]: **H1 se vuelve falsa sobre los datos reales y todo generador queda automáticamente no refutado** |
| recortar el 1 % extremo | 67 % de las 58 bajas caen en 2008 y 2020, y abre un hueco de 13 días naturales en marzo de 2020 que dispara el control de contigüidad ya existente |
| media móvil centrada de 3 | `ac1_retorno` −0,117 → **+0,610**: fabrica predictibilidad del retorno que no existe, y además no es causal |
| remuestreo semanal | `ac100_absoluto` 0,083 → −0,034 y los bloques disjuntos caen de 69 a ~14 |

El caso de la winsorización al 1 % es el decisivo y el menos intuitivo: tras
aplicarla, **el mercado tendría colas más finas que el modelo baratísimo contra el
que se supone que hay que compararlo**, y el notebook 05 "refutaría" la premisa del
trabajo por un artefacto de preprocesado. Además, ajustar los cortes sobre la
muestra completa sería fuga; ajustados solo con train seguirían censurando 18
observaciones de test, que D7 obliga a mantener 100 % reales.

**Corrección de alcance: "rellenar" son dos operaciones, no una.** La tabla de
arriba medía una sola, y el nombre compartido tapaba la otra:

- **(A) Reindexar a una rejilla de días hábiles y rellenar.** **Inventa** sesiones
  que no existieron. Es lo que mide la tabla y **sigue prohibido**.
- **(B) Rellenar por activo dentro del calendario del índice, antes de la
  intersección.** **Recupera** sesiones reales de la NYSE que se descartaban
  porque un activo concreto no cotizó ese día. **Se adopta.**

Lo que separa (B) de (A) es el **anclaje**, y es la parte que se puede hacer mal.
La unión de calendarios que devuelve yfinance ya contiene los días en que
`DX-Y.NYB` cotiza en ICE con la NYSE cerrada; rellenar sobre esa unión, sin
anclar, **es (A)**: mete 18 festivos inventados —el 4 de julio de 2003, los
funerales de Reagan, Ford y Carter—, sube los retornos nulos de 834 a 1.091 y
borra el hueco de calendario del 2 de enero de 2007, que era un cierre real. Por
eso `datos.alinear()` ancla primero al calendario del activo de rol `indice` y
solo después rellena.

**Qué hace (B) sobre este panel.** De las 20 sesiones que el recorte descartaba,
**18 eran festivos verificables de la NYSE** —bien descartados— y **2 eran
sesiones reales** perdidas porque `dolar_indice` no cotizó: el 10 de octubre de
2016 (Columbus Day) y el 11 de noviembre de 2016 (Veterans Day). El panel pasa de
5.940 a 5.942 sesiones y `canales` de 5.668 a 5.670 filas; `curtosis_residuos`
va de 7,29 a 7,23 con banda [5,6 – 10,3], de modo que **H1 sobrevive**; el AUC de
los 20 canales se mueve menos de 0,0001; y la causalidad sigue en 20/20.

**Sobre el tope de relleno.** El hueco más largo del universo dentro del
calendario del índice mide **una sola sesión**, así que 1, 2, 5 o sin tope dan hoy
exactamente el mismo panel: el número no elige un resultado, elige lo que se
autorizará mañana. Se adopta el **5** que fija el catálogo, con una advertencia
medida: borrando y rellenando k cierres consecutivos del índice,
`curtosis_residuos` da 7,29 (k=1), 7,30 (k=3) y **14,88 (k=5)** —fuera de la banda
de H1—, y `vol_futura` se desvía un 2,2 %, 6,8 % y **72 %**. Es decir, un solo
relleno que agote el tope refutaría la hipótesis del trabajo por un artefacto de
preprocesado. Por eso `calidad.auditar_panel` tiene una fila `relleno` que **avisa
de cualquier relleno de más de una sesión**: el tope es un seguro, y el aviso es
lo que impide que se use sin verlo.

**Dos naturalezas, no dos grados.** Los cinco primeros controles —tipos, índice,
finitud, positividad, degeneradas— son **invariantes**: cosas imposibles en datos
correctos, silenciosas aguas abajo, y lanzan excepción desde `datos.alinear()`. Un
solo cierre puesto a cero evapora 61 ventanas y deja `drawdown_sp500` en −1,0 sin
que nada avise. Los otros cinco —congelados, saltos, calendario, densidad,
cobertura— son **avisos** y nunca abortan, porque lo inverosímil aquí a veces es
cierto: el mayor movimiento del panel es un VIX de +115,6 % el 5 de febrero de
2018, y es un dato bueno. Un control que aborta sobre datos buenos enseña al
equipo a subir el umbral.

**Lo que no se comprueba, y por qué.** Nada de tests de normalidad ni de
estacionariedad como puerta de calidad: rechazarían con p≈0 por construcción,
porque la no normalidad **es el hallazgo del proyecto**, no el defecto. Tampoco
umbral porcentual fijo para los saltos —836 observaciones superan el 10 % y casi
todas son correctas—, sino z robusto por columna, que se autoescala al VIX.

**Excepción declarada.** La única aparición legítima de la winsorización en este
repositorio es como *baseline de comparación* del jitter en el notebook 12, y allí
los cortes se ajustan solo con train.
