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

**Decisión.** `X` combina retornos del índice y de nueve sectores con diez
features derivadas de estrés (VIX, spread de crédito, pendiente de curva,
drawdown, correlación acción-bono, dispersión sectorial).

**Por qué.** Los notebooks guiados usan 23 tickers en crudo, lo que da un bloque
de 1.380 dimensiones donde la GAN densa del ejemplo se atasca visiblemente. Las
features derivadas concentran la señal de régimen —en nuestro TFM de detección
de regímenes, `VIX_level_z` alcanza AUC 0.81 y `MOVE_level_z` 0.80— y hacen el
problema aprendible con menos dimensiones.

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
minoritaria de en torno al 25 %, demasiado poblada para que el desbalance sea el
problema central. Con tres, el estado extremo queda en torno al 10 % y separa la
tensión genuina de la simple transición.

**Canonicalización.** Los estados se ordenan por volatilidad creciente, con el
retorno medio solo como desempate dentro de bandas de anchura `0.15 × vol
media`. Criterio heredado del TFM, donde se comprobó que ordenar por retorno
medio intercambia calma y crisis de forma errática entre ejecuciones.

---

## D7 · Split temporal con embargo de 85 sesiones

**Decisión.** Particiones por fecha, con 85 sesiones descartadas en cada
frontera. Nunca `train_test_split` aleatorio.

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
