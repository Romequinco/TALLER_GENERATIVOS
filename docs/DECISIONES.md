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
minoritaria demasiado poblada para que el desbalance sea el problema central, y el
notebook 01 **ya no lo cita, lo mide**: ajusta el contraste de dos estados sobre el
mismo train y las mismas tres features, y el estado extremo se va al **37,4 %** de
las sesiones frente al **15,8 %** de la versión de tres. Con esa prevalencia el
diseño de dos estados **suspende 1 de los 4 controles bloqueantes** —el del peso de
la clase de crisis, cuya banda de aceptación es [3 %, 20 %]—. El **25,3 %** que
`metodologia/etiquetado_regimenes.md` §7.5 recoge del HMM de dos estados del TFM
—otro panel, 2007-2026 y siete features— sigue siendo un anclaje externo que este
repositorio no reproduce, pero ha dejado de ser la única cifra de la que dispone el
proyecto, y la medición propia es **más adversa** al diseño de dos estados que el
anclaje, no menos. Con tres, el estado extremo queda en el 15,8 % y separa la
tensión genuina de la simple transición.

**Tres features de etiquetado, no cinco.** La versión inicial usaba `ret_sp500`,
`vol_realizada_z`, `vix_nivel_z`, `drawdown_sp500` y `spread_credito_z`. **Suspende
el control bloqueante del notebook 01**, y conviene ser exacto sobre por dónde,
porque no es por donde se esperaría:

- El **peso de la clase de crisis sale al 19,0 %**, dentro de la banda de
  aceptación [3 %, 20 %]: **ese control pasa**. Lo hace pegado al techo, que ya es
  un aviso, pero pasa.
- Lo que suspende es la **cobertura del episodio de Inflación 2022: 44,0 % frente
  al 50 % exigido**. Con las tres features se va al 56,5 %. GFC 2008 (100 %) y
  COVID 2020 (96 %) los cubren las dos versiones.
- Y el estado extremo se enciende donde no debe: el **40 % de las sesiones de
  2021** quedan marcadas como crisis, el año en que el S&P 500 subió un 26,9 % con
  volatilidad del 13 %. No es el año entero, pero son cuatro de cada diez sesiones
  de un mercado alcista y tranquilo.

Los tres hechos apuntan en la misma dirección, y ninguno de ellos lo habría
delatado el peso de la clase mirado por su cuenta: el control que suspende es el
de cobertura.

La causa es que un HMM gaussiano supone emisiones normales y estacionarias dentro
de cada estado, y dos de las cinco lo incumplen: `spread_credito_z` **deriva**
—1,17 sigmas de salto entre las dos mitades de la muestra, porque el z-score
expanding no vuelve nunca de 2008— y `drawdown_sp500` **satura**, con el 47 % de
las sesiones en el 5 % superior de su recorrido. Con ellas dentro, el estado
extremo deja de significar "mercado en tensión" y pasa a significar "estamos
después de 2020".

Retirándolas, la crisis queda en el 15,8 %, la cobertura de 2022 sube al 56,5 % y
los cuatro controles pasan.

**El ajuste tampoco depende de la inicialización, y esto también está medido en
lugar de afirmado.** El notebook 01 recorre las **cinco semillas del catálogo**
(`semillas: [42, 43, 44, 45, 46]`, de las que se conserva la de mayor
log-verosimilitud) y publica dos hechos que conviene no confundir:

- **Las verosimilitudes no coinciden.** Con tres features el EM cae en **dos
  óptimos locales**: 9.418,73 y 9.418,75 (semillas 42 y 45) frente a 9.417,49,
  9.417,49 y 9.417,57 (semillas 43, 44 y 46). El recorrido entre el mejor y el peor
  es de **1,26 nats sobre 3.755 observaciones**. Gana la semilla 45, con
  log-verosimilitud **9.418,747** y **163 iteraciones del EM**.
- **Las etiquetas sí coinciden, y es lo que decide.** El **acuerdo mínimo entre los
  etiquetados de dos semillas cualesquiera es del 98,9 %** con tres features, frente
  al **52,8 %** con las cinco iniciales. La superficie de verosimilitud tiene más de
  un máximo, pero con tres features todos nombran el mismo régimen casi todos los
  días.

Lo que un etiquetado necesita para servir como variable objetivo no es que el EM
converja siempre al mismo punto, sino que el estado extremo signifique lo mismo se
parta de donde se parta. Con las cinco features no lo cumplía —dos semillas podían
discrepar en casi la mitad de la muestra—; con tres, sí. Una versión anterior de
esta decisión daba la convergencia por sentada; la medición la sustituye y la
refuerza, porque separa lo que converge de lo que no.

Una versión anterior de esta decisión hablaba de "20 semillas probadas" y de un
segundo óptimo local que con las cinco features daba un 21,2 %. Ni una ni otra
cifra las reproduce ningún notebook de este repositorio —el catálogo ejecuta
cinco semillas, no veinte—, así que se retiran en vez de dejarlas como argumento
no verificable. Las dos descartadas siguen en el catálogo
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
salga— se descarta con datos: 119 días **no llegan al mínimo en el 2,0 % de las
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

**Valor del embargo.** La huella de una ventana son `60 + 21 = 81` sesiones, pero
el mínimo del embargo es **80**, no 81: una ventana que empieza en `t` usa datos
hasta `t + pasado + horizonte - 1`, así que el solape máximo entre dos ventanas es
`60 + 21 - 1 = 80`. Es lo que devuelve `ventanas.solape_maximo` y es contra lo que
valida `ventanas.partir`. Los dos números conviven sin contradicción porque
descartar E ventanas abre un hueco de E+1 sesiones entre particiones, así que
exigir un hueco de 81 sesiones o más es exigir un embargo de 80 o más. Se adopta
85, cinco de margen.

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
estandarizados: 7,2 reales con banda [5,6 – 10,3], frente a 3,4 de la mezcla. Es
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

**Dos fallos silenciosos que ahora lanzan.** La comparación solo vale si de
verdad se aplicó lo que se dice haber aplicado, y había dos maneras de creer que
sí sin que nada avisara:

- `pesos_por_clase` devolvía peso **`0.0`** para una clase sin muestras. Un `0.0`
  se lee como "esta clase no importa" cuando lo que significa es "esta clase no
  está", y Keras lo acepta sin decir nada: se entrenaba, sin un solo aviso, un
  modelo que nunca predice crisis, que es exactamente el fallo que mide el
  trabajo. Una clase vacía es un defecto de la receta de mezcla, no un caso de
  uso, y ahora lanza `ValueError`.
- `usar_pesos=True` sobre la tarea de **volatilidad** era un **no-op silencioso**:
  la regresión no tiene `class_weight`, así que el argumento se ignoraba y quien
  escribiera la variante reponderada de la tarea de control creería haberla
  aplicado y habría publicado dos veces el mismo experimento. Ahora lanza.

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

**Por qué.** La clase de crisis pesa un **15,9 % en train y un 10,5 % en test**,
así que un modelo que nunca la prediga acierta en torno al 90 % del test. El
accuracy no distingue ese modelo inútil de uno valioso.

**Y la que selecciona no es la que se reporta primero.** La prevalencia de crisis
en **validación es del 22,0 %, más del doble que en test (10,5 %)**, porque el
tramo de validación contiene el COVID. Ese desplazamiento de prevalencia es la
razón por la que la métrica que **elige** la arquitectura es la **balanced
accuracy** y no el f1-macro: medido sobre la rejilla de candidatas, al pasar de la
prevalencia de validación a la de test el f1-macro se mueve entre **-0,013 y
+0,044** según lo liberal que sea el modelo con la clase de crisis, e **invierte
el orden del 8,1 % de los pares** de candidatas; la balanced accuracy es
exactamente invariante, porque es la media de los recalls por clase y un recall no
depende de cuántas muestras haya de esa clase. Elegir con f1-macro sobre una
validación con doble prevalencia de crisis que el test es elegir con una regla que
sabemos que se rompe al cambiar de tramo. El f1-macro se sigue reportando en la
tabla; simplemente no decide.

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

## D20 · Una sola arquitectura downstream, congelada, y gana la que no convoluciona

**Decisión.** La arquitectura se busca en el notebook 03 con datos reales entre
**seis candidatas**, se elige sobre **validación** y no se vuelve a tocar. Queda
escrita, con su presupuesto y su huella, en `models/downstream/arquitectura.json`.

**Por qué.** Lo exige el enunciado y es lo único que hace honesta la
comparación: si la arquitectura cambiara entre versiones, no se sabría si la
diferencia viene de los datos o del modelo.

**Cómo se busca.** Cada candidata mueve **un solo eje** respecto a `cnn_base`, para
que el resultado se pueda defender: `lineal` quita la convolución entera y es el
control de si aporta algo; `cnn_pequena` y `cnn_ancha` mueven la capacidad;
`cnn_kernel7` el campo receptivo; `cnn_pool_global` la agregación temporal. Tres
semillas por candidata, porque con una sola corrida la dispersión no se puede medir
y sin dispersión no se puede afirmar que una gane.

**Criterio de selección: `balanced_accuracy`, no f1-macro.** Ver D17: validación
tiene un 22,0 % de crisis y test un 10,5 %, y bajo ese salto de prevalencia el
f1-macro invierte el orden del 8,1 % de los pares mientras la balanced accuracy es
exactamente invariante. Desempate por `recall_crisis` con margen 0,005 y, si tampoco
separa, gana la más barata (D19).

**Resultado medido, y es incómodo.** Balanced accuracy sobre validación:

| candidata | bal. acc. | desv. entre semillas | parámetros | s/época |
|---|---|---|---|---|
| **`lineal`** | **0,6956** | 0,0081 | **3.603** | **0,35** |
| `cnn_kernel7` | 0,6917 | 0,0122 | 271.315 | 4,08 |
| `cnn_base` | 0,6849 | 0,0391 | 167.891 | 2,52 |
| `cnn_ancha` | 0,6845 | 0,0092 | 533.123 | 7,60 |
| `cnn_pequena` | 0,6681 | 0,0168 | 69.859 | 0,94 |
| `cnn_pool_global` | 0,6617 | 0,0203 | 91.091 | 3,08 |

Las cuatro columnas salen de `results/metricas/busqueda_arquitectura.csv`. **La
última es tiempo de reloj de la máquina que ejecuta el notebook 03** y cambia entre
equipos; las otras tres, no.

**`lineal` y `cnn_kernel7` caen dentro de una desviación entre semillas: no están
separadas por nada**, y las dos siguientes se quedan a 0,0107 y 0,0111. La que
encabeza es una multinomial sobre la ventana aplanada, con 148 veces menos
parámetros que `cnn_ancha` y **veintidós veces** menos tiempo por época —0,35 s
frente a 7,60 s, medido en esta máquina—. No es una
anomalía: con **45 bloques independientes** en train (ver D21) la convolución no
tiene de dónde sacar ventaja. Se congela `lineal`.

**Y el empate depende de la tarea, no de los datos.** Reentrenando las mismas
candidatas para predecir el régimen de **hoy** en vez del dominante a 21 días, la
comparación se invierte: `cnn_ancha` llega a 0,808 de balanced accuracy y `lineal`
se queda en 0,682, doce puntos por debajo. Es decir, la convolución sí sirve cuando
el objetivo está bien condicionado; lo que no sirve es contra una etiqueta a 21 días
sobre 45 observaciones independientes. Ese contraste es el argumento de por qué el
suspenso del control bloqueante (D21) es un problema de formulación y no de datos.

**El presupuesto también se congela**, y vive en `downstream.Presupuesto`: 60
épocas, lote 256 (D22), paciencia 12 y **parada por `balanced_accuracy` de
validación**, no por `val_loss`. Ese último punto no es cosmético y está medido
sobre los **22 entrenamientos** que el cuaderno 03 deja en
`results/historiales/*.csv`, y que su última celda recorre: parar por `val_loss`
en vez de por la métrica que decide cuesta **1,8 puntos de balanced accuracy de
media** y hasta **6,6 puntos** en el peor caso (`cnn_kernel7`, semilla 2). En la
arquitectura congelada, el mínimo de `val_loss` cae en la época 38 y el máximo de
balanced accuracy en la 55: **2,6 puntos**. La causa es que la entropía cruzada de
validación está dominada por el desajuste de prevalencias entre train (54,6 %
calma) y validación (59,7 % transición), no por lo que el modelo aprende. La
métrica que decide tiene que ser la métrica que para.

Una versión anterior de esta decisión publicaba un mínimo de `val_loss` en la
**época 1** frente a un máximo de balanced accuracy en la **5**, con un coste de
**8,5 puntos**. Esa medición sale de un experimento auxiliar con `cnn_ancha` y
pérdida reponderada que **no está en el repositorio** y **no es reproducible desde
los historiales**, así que se retira y se sustituye por la de arriba, que sí sale
de CSV versionados. El argumento de fondo no cambia: lo que hace inservible a
`val_loss` es el desajuste de prevalencias, y eso se mide igual en los 22
historiales.

**Por qué se persiste en disco y no solo en el código.** El notebook 12 llamaba a
`construir()` sin argumentos y heredaba el valor por defecto del **código fuente**:
si alguien editaba el módulo entre el 03 y el 12, nada lo detectaba y los 570
entrenamientos quedaban contaminados en silencio. Ahora `cargar_congelada()` lee el
JSON y valida su huella.

**Consecuencia sobre el coste.** Que la ganadora sea la barata cambia el
dimensionado del barrido, y las dos magnitudes hay que tomarlas **de la ganadora**,
no una de ella y otra de la media de las seis: a **0,35 s por época** y **51,0
épocas efectivas medias** —las de `lineal` en la búsqueda; consumió 60 en su
entrenamiento real—, los siete generadores del notebook 12 salen por **3,55 h** y
los cuatro del núcleo mínimo por **2,04 h**
(`results/metricas/coste_barrido.csv`). La media de las seis candidatas, 24,2
épocas, está dominada por las CNN, que paran entre la 15 y la 22, y usarla con los
segundos de `lineal` **subestimaría el coste por un factor 2,11**, que es el número
medido y no "a la mitad". Con `cnn_ancha` habrían sido **veintidós** veces más los
segundos por época, y la decisión de recortar generadores habría sido inevitable.

**Estas tres cifras, y sólo estas tres, son tiempo de reloj.** Los 0,35 s por
época, las 3,55 h y las 2,04 h se miden en la máquina que ejecuta el notebook 03 y
cambian entre equipos, igual que la columna `s/época` de la tabla de arriba y todo
`coste_barrido.csv`; la celda de cierre del cuaderno lo escribe para que nadie las
lea como una constante del proyecto. El orden de las candidatas, las balanced
accuracies, las desviaciones entre semillas, los recuentos de parámetros y las
épocas efectivas no dependen de la máquina.

---

## D21 · Líneas base: la persistencia causal es la barra, y viene con banda

**Decisión.** Toda métrica de la tarea de régimen se compara contra el predictor
trivial "el régimen futuro será el mismo que el actual", **decodificado con el
filtro causal**, y se publica con un intervalo de confianza por **bootstrap de
bloques**, no con la cifra suelta.

**Por qué la persistencia es una barra alta.** Los regímenes son muy
persistentes: la diagonal de la matriz de transición del HMM ajustado vale
**0,987 · 0,970 · 0,986** (calma, transición, crisis). Con esas probabilidades de
permanencia, "mañana como hoy" acierta casi siempre, y un clasificador que no la
bata no está prediciendo nada: se está limitando a leer el estado presente, que es
un problema distinto y mucho más fácil.

**Sobre qué muestra se mide.** El notebook 01 publica la persistencia sobre la
**muestra completa**, y desde su última ejecución la publica **con las dos
decodificaciones al lado**, que es lo que esta decisión pedía y no se estaba
haciendo: la **causal** da accuracy **0,7775** y recall de crisis **0,7850**; la de
Viterbi, **0,8210** y **0,8208**. El 0,821 que circulaba como "la persistencia" era
la fila de Viterbi. Ninguna de las dos es la barra del experimento: la muestra
completa está dominada por train, y esa cifra es un diagnóstico del etiquetado. La
barra se mide donde se evalúa el modelo, que es **test**. Confundir las muestras
—o citar Viterbi sin nombrarlo— es el error fácil de esta decisión, porque la cifra
bonita está a la vez en la muestra que no toca y en el decodificador que no vale.

**Con qué decodificador, que es donde estaba el problema de fondo.**
`hmmlearn.predict` es **Viterbi**: nombra el régimen de hoy buscando la secuencia
de estados más probable **mirando la serie entera, futuro incluido**. Para
construir la etiqueta objetivo eso es legítimo —la etiqueta *es* el futuro—, pero
como predictor no lo es: no es un modelo, es un oráculo sobre el proceso que
fabrica la etiqueta. El decodificador honesto es el **filtro forward**, que usa
información hasta `t` y ni un día más; está implementado en
`EtiquetadorRegimenes.predict_causal`. Sobre test:

| línea base (test) | accuracy | f1-macro | recall crisis | precisión crisis |
|---|---|---|---|---|
| `persistencia_causal` (la barra) | 0,772 | 0,754 | 0,800 | 0,638 |
| `persistencia_viterbi` (usa el futuro) | 0,805 | 0,790 | 0,800 | 0,715 |

La balanced accuracy de la causal, que es la métrica con la que D17 selecciona
arquitectura, vale **0,777** sobre test: ese es el número contra el que se lee el
notebook 03.

Las dos decodificaciones difieren en **321 de 5.670 sesiones (5,66 %)**, y el
filtro causal marca 924 sesiones de crisis frente a las 901 de Viterbi. Sobre la
muestra completa la diferencia son **4,35 puntos de accuracy**, y es lo que titula
`results/figures/01_persistencia_causal_vs_viterbi.png`, que enfrenta las dos
matrices de confusión y trama la de Viterbi por usar el futuro. El detalle
que hay que saber decir en la defensa es que **el recall de crisis es 0,800 con
las dos**: toda la ventaja de Viterbi está en la **precisión** (0,715 frente a
0,638), porque el futuro le dice cuáles de las alarmas eran falsas. Publicar
Viterbi como barra regalaría 3,6 puntos de f1-macro a cualquier modelo que se
compare con ella.

**Cuál se publica.** La oficial es la **causal**. La de Viterbi se publica al lado,
marcada como no alcanzable por un modelo causal: `evaluacion.lineas_base` devuelve
una columna booleana `usa_futuro` que existe exactamente para eso.

**Las otras tres líneas base y por qué están.** `evaluacion.lineas_base` produce
además `mayoritaria_train` (accuracy 0,330, recall de crisis 0,000),
`azar_estratificado` (0,364 / 0,160) y `siempre_crisis` (accuracy 0,105, recall de
crisis **1,000**). Las dos primeras acotan el suelo. La tercera está por una razón
distinta y deliberada: impedir que un recall de crisis alto se presente como un
logro sin mirar la precisión. "Predice siempre crisis" tiene el recall máximo
posible y es inútil. Cualquier lectura de `recall_crisis` que no mire a la vez
`precision_crisis` y `f1_macro` está reproduciendo esa trampa.

**La banda, que es la parte que decide qué comparaciones son legítimas.** Las 110
ventanas de crisis del test **no son 110 observaciones independientes**: son **3
rachas contiguas** (`regimenes.tramos_contiguos` las cuenta; en train son 587
ventanas en 8 rachas y en validación 148 en 2). Tratarlas como independientes
estrecha el intervalo de forma artificial. `evaluacion.banda_bloques` mide el
recall de crisis de 0,800 con **IC 95 % [0,560 – 1,000]** por bootstrap de bloques
circular de longitud 81 —la huella exacta de una ventana, el mismo 81 del que sale
el embargo de D7—, frente al **[0,716 – 0,864]** del intervalo binomial de Wilson
que las trataría como independientes: el honesto es **3,0 veces más ancho**. El
límite superior satura en 1,000 porque hay una racha entera con recall perfecto. El
notebook 03 publica también la banda de la métrica de selección: la balanced
accuracy de 0,777 viene con **[0,681 – 0,860]**, de ancho 0,178. Las dos bandas son
las que fijan la barra en la figura `results/figures/03_veredicto.png`.

**Consecuencia operativa, que hay que escribir en el informe y no solo aquí:
ninguna comparación de recall de crisis entre dos recetas del notebook 12 que
difiera en menos de unos 20 puntos es distinguible del ruido.** Ordenar la tabla
de resultados por esa columna y quedarse con la primera fila es quedarse con el
ruido más afortunado.

**Consecuencia para el análisis.** Las mejoras por datos sintéticos se reportan
como margen respecto a la persistencia causal, no sobre el cero absoluto, y
siempre con la banda al lado.

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

**El modelo downstream también, y con su propia medición.** La regla anterior se
midió sobre los generadores; el downstream faltaba, y es donde se gastan las horas
del barrido. Sobre las 3.696 ventanas de train, con validación: lote 64 cuesta
**3,30 s/época**, lote 256 **2,43** y lote 512 **2,34**. **Se adopta 256**, y no
512, porque ya recoge la mayor parte de la mejora sin dividir aún más el número de
actualizaciones de gradiente por época.

**Sobre los hilos: no se toca, y no por lo que decía antes.** Una versión anterior
afirmaba que `set_num_threads(4)` sobre 2 núcleos físicos es *más lento* que el
valor por defecto. Lo medido en `src/config.py`, sobre el modelo downstream con
lote 256, es **2,61 s/época con el valor por defecto frente a 2,54 s/época
forzando 4**: está dentro del ruido entre repeticiones, no es más lento. La razón
real para no tocar `torch.set_num_threads` es otra y es doble: forzarlo **no mejora
de forma medible**, y **no sería portable**, porque `os.cpu_count()` devuelve los
núcleos lógicos y detectar los físicos exigiría una dependencia que no está en
`requirements.txt`. Añadir un paquete para no ganar nada medible es un mal
cambio.

---

## D23 · Los datos se auditan, no se limpian

**Decisión.** El panel pasa once controles de integridad en `src/calidad.py`, y
**no se le aplica ninguna técnica de limpieza**: ni imputación, ni winsorización,
ni recorte de valores atípicos, ni suavizado, ni remuestreo.

**Por qué.** No es purismo: las técnicas habituales destruyen justo lo que el
trabajo mide. Medido sobre este panel en un banco de pruebas aparte, con la
salvedad que va al pie de la tabla:

| técnica | efecto medido |
|---|---|
| `ffill` **(A)** sobre rejilla hábil | +222 sesiones inventadas (+3,7 %), los retornos nulos de `sp500` pasan de 3 a 225 y la volatilidad realizada se sesga −2,0 % |
| winsorizar al 1 % | curtosis 16,5 → **6,4**, por debajo de la banda bootstrap del propio panel [7,3 – 21,5] **y por debajo del 8,0 que produce nuestro propio generador gaussiano** |
| winsorizar al 5 % | `curtosis_residuos` 7,3 → **5,3**, por debajo de la banda [5,6 – 10,3]: **H1 se vuelve falsa sobre los datos reales y todo generador queda automáticamente no refutado** |
| recortar el 1 % extremo | 67 % de las 58 bajas caen en 2008 y 2020, y abre un hueco de 13 días naturales en marzo de 2020 que dispara el control de contigüidad ya existente |
| media móvil centrada de 3 | `ac1_retorno` −0,117 → **+0,610**: fabrica predictibilidad del retorno que no existe, y además no es causal |
| remuestreo semanal | `ac100_absoluto` 0,083 → −0,034 y los bloques disjuntos caen de 70 a ~14 |

**De dónde sale esta tabla, y qué no es.** Las seis filas se midieron sobre este
panel en un banco de pruebas que **no se conserva en el repositorio**: **ningún
cuaderno del 00 al 14 las recalcula**, así que **no están reproducidas aquí** y no
deben citarse como salida de una celda. Lo que sí es reproducible es el punto de
partida contra el que se leen: el cuaderno 00 publica la curtosis de los residuos
estandarizados del panel vigente, **7,2** con banda **[5,6 – 10,3]**, mientras la
fila de la winsorización al 5 % arrastra el 7,3 de una ejecución anterior a la
corrección (B) de más abajo. El argumento se mantiene entero —cada una de estas
técnicas destruye justamente lo que el trabajo mide, y ninguna decisión del
proyecto depende de su tercera cifra—, pero si alguna hiciera falta en la defensa
hay que rehacerla en un cuaderno antes de enseñarla.

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
que nada avise. Los otros seis —congelados, saltos, calendario, densidad,
cobertura y el `relleno` que añade esta decisión— son **avisos** y nunca abortan,
porque lo inverosímil aquí a veces es
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

---

## D24 · Política de indefinidos, piso del QLIKE y referencia del R2

**Decisión.** Tres reglas sobre cómo se calculan las métricas que llenan la tabla
del notebook 12. Estaban solo en los docstrings de `src/evaluacion.py`, y afectan
a todas las filas de la tabla de resultados, así que se registran aquí.

**1. Lo que no se puede medir es NaN, no cero.** Antes, `zero_division=0` hacía
que el recall de crisis valiera 0,0 cuando el conjunto evaluado no contenía
ninguna ventana de crisis. Un 0,0 así es indistinguible del 0,0 de un modelo que
las falla todas, que es un juicio completamente distinto; y además contradecía al
propio control de integridad del notebook 12, que busca NaN y por tanto **no podía
dispararse nunca**. Ahora:

- el recall de crisis es **NaN** si el conjunto no tiene crisis,
- la precisión de crisis es **NaN** si el modelo no predijo ninguna,
- y se emite `soporte_crisis`, el número de ventanas de crisis del conjunto
  evaluado, que no es una métrica sino lo que dice si las demás significan algo.

Las dos métricas macro promedian sobre las clases presentes en la verdad, que es
la convención de `balanced_accuracy_score`. Esto importa en el eje de escasez de
D13: con submuestras de 250 reales es perfectamente posible quedarse sin crisis en
un conjunto, y ahí la diferencia entre "no había nada que medir" y "el modelo
falla todo" es la diferencia entre una fila descartable y una conclusión.

**2. El QLIKE es frágil y hay que decirlo.** Es la pérdida estándar en previsión
de volatilidad, y se usa porque penaliza más infraestimar el riesgo que
sobreestimarlo, que es la asimetría correcta en gestión de riesgo. Pero diverge
cuando la predicción se acerca a cero, y con el recorte anterior a `1e-6` eso lo
volvía inservible para rankear: con `y_real` en torno a 0,151 y 1.052 muestras de
test, un sesgo del −10 % en **todas** las predicciones daba QLIKE **0,0058**,
mientras que **una sola** predicción negativa daba **143,5**. Un único valor
recortado pesaba unas 25.000 veces más que equivocarse un 10 % en todo el
conjunto: el ranking de generadores por QLIKE habría sido el ranking de cuántos
negativos escupió cada red, que no es lo que se quiere comparar. Peor todavía, el
contador de diagnóstico miraba `y_pred <= 0` y **no contaba las predicciones entre
0 y 1e-6**, que se recortan igual, de modo que el caso más dañino se producía en
silencio. Ahora el piso es **relativo a la escala de la variable**
(`FRACCION_PISO_QLIKE * mediana(y_real)`) y el contador cuenta exactamente lo que
toca el piso. Sobre el test real, una predicción recortada pasa de aportar 267,1 a
aportar 0,0163, el mismo orden que el sesgo del 10 %. El piso hace la métrica
comparable entre recetas; **no** borra el problema, así que hay que mirar el
contador antes de leer la columna.

**3. El R2 tenía un oráculo dentro.** Se calculaba contra la media del propio
conjunto de test. Esa media nadie la conoce al desplegar, y además no es
comparable entre recetas, porque cada una cambia el denominador con el que se
juzga. Ahora se puede pasar la **media de train** como referencia, que es el R2
honesto: el único que mide contra lo que se podía saber de antemano.

**Por qué está aquí y no solo en el código.** Las tres reglas cambian números que
van a la tabla de resultados y a la presentación. Un tribunal que pregunte "¿por
qué esta celda está vacía?" o "¿por qué el QLIKE de este generador es enorme?"
tiene aquí la respuesta, y las tres respuestas son del mismo tipo: la métrica dice
lo que puede decir y calla lo que no.
