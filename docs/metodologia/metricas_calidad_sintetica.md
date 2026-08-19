# Métricas de calidad de datos sintéticos

Documento metodológico del taller B5-T1. Fija el protocolo con el que se comparan los siete generadores (jitter, gaussiano multivariante, cGAN, cVAE, RBIG, flow matching, DDIM) sobre el mismo bloque de datos.

**Objeto evaluado.** Bloque conjunto $[\mathbf{X}\ \text{aplanada};\ y_{\text{vol}}]$ **condicionado a** $y_{\text{reg}}$, con $\mathbf{X} \in \mathbb{R}^{60 \times 20}$ (60 días × 20 canales de un panel híbrido de mercado). El bloque tiene por tanto $60 \times 20 + 1 = \mathbf{1.201}$ dimensiones, con $n \approx 4.000$–$5.000$ muestras reales (3.696 ventanas en train). El régimen va como **condición** y no como una dimensión más del vector generado: si fuera una componente del bloque, la proporción de clases sintéticas replicaría el desbalance del train y no se podría sobre-generar la clase de crisis, que es justamente el objetivo del trabajo (D8). Tarea downstream principal: clasificación del régimen de mercado a 21 días (3 clases; la de crisis pesa un 15,9 % en train y un 10,5 % en test). Secundaria: regresión de volatilidad realizada.

**Restricción dura.** Todo el cómputo es CPU-only. Cualquier métrica cuyo coste crezca peor que $O(n^2)$ en el espacio original de 1.201 dimensiones queda fuera o se calcula sobre una proyección reducida.

## 1. Los tres ejes: fidelidad, utilidad y privacidad/diversidad

No existe una métrica escalar que ordene generadores. La evaluación se descompone en ejes que pueden moverse en direcciones opuestas:

| Eje | Pregunta | Fallo que detecta |
|---|---|---|
| **Fidelidad** | ¿Las muestras son plausibles bajo $P_{\text{real}}$? | Artefactos, marginales deformadas, correlaciones rotas |
| **Diversidad / cobertura** | ¿El sintético cubre todo el soporte real? | Mode collapse, pérdida de la clase minoritaria |
| **Utilidad** | ¿Sirve para entrenar el modelo downstream? | Datos plausibles pero sin señal predictiva |
| **Generalización** | ¿El generador copia el entrenamiento? | Memorización, sobreajuste del generador |

Alaa et al. (ICML 2022) formalizan esta descomposición en tres cantidades muestrales — $\alpha$-precision (fidelidad), $\beta$-recall (diversidad) y *authenticity* (generalización) — y argumentan que la generalización es una dimensión **independiente** del compromiso fidelidad–diversidad: un generador que memoriza obtiene fidelidad y diversidad perfectas y sigue siendo inútil.[^alaa] Du y Li (arXiv:2402.06806, CCS 2025) llegan a la misma tripartición auditando 8 sintetizadores sobre 12 datasets, y documentan que las comparaciones publicadas suelen fallar por métricas mal elegidas más que por generadores malos.[^duli]

La tensión central del taller es exactamente ésa: un generador que reproduzca literalmente el conjunto de entrenamiento maximiza cualquier métrica de fidelidad y obtiene un TSTR indistinguible del baseline real. Por eso la sección 4 no es un apéndice de privacidad, sino **la condición de validez de la sección 2**.

**Protocolo de particiones (aplica a todo el documento).** Tres conjuntos disjuntos: `real_train` (lo único que ve el generador), `real_val` (hiperparámetros del modelo downstream) y `real_test` (holdout final, nunca visto por el generador ni por el selector de modelos). Como el panel es temporal y las ventanas de 60 días se solapan, los cortes deben ser **cronológicos con purga y embargo**: un split aleatorio sobre ventanas solapadas filtra información y sobreestima todas las métricas de utilidad.[^prado]

## 2. Utilidad: TSTR y TRTS

### 2.1 Origen y protocolo

TSTR (*Train on Synthetic, Test on Real*) y TRTS (*Train on Real, Test on Synthetic*) se formalizan en Esteban, Hyland y Rätsch (2017), en el contexto de un RCGAN para series temporales médicas.[^esteban] Si el sintético captura la estructura condicional $P(y \mid \mathbf{X})$, un modelo entrenado solo con sintético debe rendir sobre datos reales de forma comparable a uno entrenado con reales.

Se definen cuatro entrenamientos con el **mismo** modelo downstream, la misma parrilla de hiperparámetros y la misma semilla:

| Etiqueta | Entrena en | Evalúa en | Interpretación |
|---|---|---|---|
| **TRTR** | `real_train` | `real_test` | Baseline. Techo de referencia |
| **TSTR** | sintético | `real_test` | Utilidad: ¿sustituye al real? |
| **TRTS** | `real_train` | sintético | ¿El sintético es "demasiado fácil"? |
| **TSTS** | sintético | sintético | Diagnóstico de sobreajuste del generador |

La métrica primaria es el **ratio de utilidad** $\rho = \text{TSTR}/\text{TRTR}$: $\rho \geq 0{,}9$ indica sustituibilidad práctica; $\rho \in [0{,}7 , 0{,}9)$ utilidad parcial (aumentación sí, sustitución no); $\rho < 0{,}7$ el generador no captura la relación condicional.

Lecturas cruzadas: **TSTR bajo + TRTS alto** · el sintético es un subconjunto simplificado del real (mode collapse); el modelo real lo clasifica sin esfuerzo pero no enseña las regiones difíciles. **TSTS ≫ TSTR** · el sintético es internamente consistente pero desplazado respecto al real.

### 2.2 El eje central del taller: la curva de mezcla

El experimento principal no es TSTR puro sino la curva de rendimiento downstream frente a la cantidad de sintético **añadida** al conjunto real. Es una **aumentación**, no una sustitución: el conjunto real no se toca y el tamaño total crece con el ratio.

$$\mathcal{D}_{\text{train}}(n_r, \rho) \;=\; n_r \ \text{reales} \ \cup \ \lfloor \rho\, n_r \rceil \ \text{sintéticos}$$

La rejilla la genera `mezclas.rejilla` y tiene **tres** ejes, no uno:

| Eje | Valores | Constante en `src/mezclas.py` |
|---|---|---|
| Reales disponibles $n_r$ | 250, 500, 1.000, 2.000 y **todos** (3.696 ventanas de train) | `NIVELES_REALES` |
| Ratio sintético/real $\rho$ | 0, 0,5, 1, 2 y 5 | `RATIOS_SINTETICOS` |
| Política de reparto por clase | `proporcional`, `equilibrado` | `POLITICAS` |

**Por qué se añade en vez de sustituir.** La variante de tamaño total constante mide sustituibilidad, y es una pregunta legítima, pero deja el efecto del tamaño muestral confundido con el de la mezcla. Aquí ese efecto se aísla con el **segundo eje**, que es el que de verdad importa: el material del taller muestra que el beneficio del sintético se concentra en el régimen de pocos datos y se desvanece —o se vuelve negativo— cuando ya hay muchos reales. Barrer solo la proporción de sintéticos con todos los reales disponibles produciría una línea plana y la conclusión errónea de que los generadores no sirven. TSTR puro sigue siendo interpretable como el caso límite y se reporta aparte.

**Las dos políticas de reparto** son lo que separa dos hipótesis distintas: `proporcional` replica el desbalance real y mide el efecto de "más datos" sin más; `equilibrado` concentra el sintético en las clases minoritarias hasta acercarlas a la mayoritaria y mide el efecto de "más datos donde hacen falta". La diferencia entre ambas es la contribución atribuible al rebalanceo (D14). El reparto por clase lo calcula `mezclas.repartir` con el método del mayor resto, para que la suma sea exactamente el número pedido.

Las combinaciones con $\rho = 0$ no dependen del generador ni de la política —son solo datos reales— y `mezclas.rejilla` las emite una sola vez bajo el generador nominal `solo_real`, en vez de reentrenar el mismo modelo catorce veces por nivel de reales. Cada punto se repite con al menos 3 semillas y se reporta media ± desviación: con una clase de crisis en torno al 16 % en train, la varianza entre semillas es del mismo orden que las diferencias entre generadores. El comportamiento típico es no monótono: sube con poco sintético y se degrada cuando el sintético domina el entrenamiento.

### 2.3 Métricas downstream con clases desbalanceadas

**El accuracy simple no se reporta como métrica principal.** Con 3 clases y la de crisis al 10,5 % en test, un clasificador que nunca prediga crisis pierde como máximo 10,5 puntos de accuracy y puede rondar 0,90 siendo inservible para el único caso que importa. Peor: el modo de fallo más probable de un generador — colapso de la moda minoritaria — es **invisible** en accuracy y catastrófico en cualquier métrica por clase. Un TSTR con accuracy 0,88 frente a un TRTR de 0,91 parece excelente y puede esconder un recall de crisis que cae de 0,55 a 0,04.

Hay además un motivo específico de este panel para desconfiar de las métricas sensibles a la prevalencia: la clase de crisis pesa un **22,0 % en validación** y un **10,5 % en test**, más del doble. Cualquier métrica cuyo valor dependa de cuántas muestras hay de cada clase cambia al pasar de una partición a otra sin que el modelo haya cambiado.

Métricas obligatorias, en este orden: (1) **F1-macro**, media no ponderada de los $F_1$ por clase, donde cada clase pesa igual con independencia de su prevalencia — es la métrica de titular; (2) **balanced accuracy**, media de los recalls por clase, equivalente al accuracy bajo reponderación a clases uniformes;[^brodersen] (3) **recall de la clase crisis**, la métrica operativa, que un generador debe preservar o mejorar para ser aceptable; (4) **matriz de confusión completa** en el informe, no agregada; y (5) **accuracy** como columna de contexto, nunca como criterio de selección.

Para la tarea secundaria (volatilidad realizada) se usa MAE y $R^2$ sobre `real_test` con el mismo esquema TSTR/TRTR, más el MAE restringido al decil superior de volatilidad, que es donde el sintético suele fallar.

**Baselines contra los que se compara la aumentación.** El sintético no compite contra "no hacer nada", sino contra las formas baratas de tratar el desbalance: reponderación por clase y sobremuestreo por interpolación (SMOTE). Si la curva de mezcla no supera a ambos en recall de crisis, el generador no aporta nada operativo.

**Sobre la reponderación, con un matiz que importa.** El criterio **por defecto** es entrenar el modelo downstream **sin** reponderar, porque reponderar enmascara justo el efecto que se quiere medir: si la pérdida ya compensa el desbalance, el sintético de crisis no tiene margen que ocupar y la comparación entre generadores mide otra cosa. Pero eso no convierte la reponderación en una rama opcional: **D15 la hace obligatoria como comparación**. El barrido incluye una versión con `class_weight` inverso a la frecuencia de clase y **sin ningún dato sintético**, que es lo que implementa `downstream.pesos_por_clase` y activa `downstream.entrenar(..., usar_pesos=True)`. La razón es directa: si reponderar la pérdida iguala al mejor generador, los datos sintéticos no aportan nada que no se consiguiera con una línea de código, y el trabajo honesto es decirlo. Las dos ramas conviven sin contradicción: la principal va sin pesos, y la reponderada es el listón contra el que se lee su resultado.

**Elección del modelo downstream.** No se decide en este documento. La fija el enunciado del taller —se busca una arquitectura válida con datos reales y después se entrena esa misma arquitectura, sin tocarla, sobre cada dataset— y la materializa D20: se **busca** en el cuaderno 03 entre las seis candidatas de `downstream.CANDIDATAS`, gana **`lineal`** —una multinomial de 3.603 parámetros sobre la ventana aplanada, **sin ninguna capa convolucional**— y es esa la que queda **congelada** en `models/downstream/arquitectura.json` (`"filtros": []`, `"unidades_densa": 0`) junto con su presupuesto de optimización (60 épocas, lote 256, paciencia 12). Congelar el presupuesto no es un adorno: si cambiara entre versiones no se sabría si la diferencia viene de los datos o de haber entrenado más.

La regresión logística **no está descartada**: es una de esas seis candidatas, la llamada `lineal`, y existe con un papel preciso. `Arquitectura(filtros=(), unidades_densa=0)` deja el modelo sin convoluciones, es decir, una regresión logística multinomial sobre la ventana aplanada, con el mismo dropout que las demás para que la comparación no le regale ventaja. Es el **control de si la convolución aporta algo**: si empata con las convolucionales, toda la troncal es coste sin contrapartida, y eso es un resultado publicable. Las otras cinco mueven **un** eje cada una respecto a `cnn_base` —capacidad arriba (`cnn_ancha`) y abajo (`cnn_pequena`), campo receptivo (`cnn_kernel7`) y agregación temporal (`cnn_pool_global`)—, de modo que la diferencia de puntuación se pueda atribuir a ese eje y no a tres cambios simultáneos.

La selección se hace **sobre validación**, con tres semillas por candidata, y decide `balanced_accuracy` con `recall_crisis` como desempate, no el F1 macro. La razón es medible y es la prevalencia: con un 22,0 % de crisis en validación y un 10,5 % en test, el F1 macro invierte el orden del 8,1 % de los pares de candidatas, mientras que la balanced accuracy —media de los recalls por clase— es exactamente invariante. El F1 macro se sigue reportando; simplemente no decide.

**El argumento de coste que sostenía la elección anterior ya no se sostiene.** Una versión previa de esta sección fijaba la logística como modelo principal alegando que un modelo más caro situaría la comparativa en varias horas. La medición sobre esta máquina, con las 3.696 ventanas de train y validación real, es de **unos 2,4 segundos por época** con lote 256 —3,30 s con lote 64 y 2,34 s con lote 512—, y la parada temprana corta muy por debajo de las 60 épocas del presupuesto. Cualquiera de las cinco CNN cabe holgadamente en el presupuesto de CPU del taller, así que elegir el modelo por su coste habría sido elegirlo por la razón equivocada. Que la búsqueda del cuaderno 03 acabara eligiendo `lineal` no rehabilita aquel argumento: gana por balanced accuracy sobre validación, no por barata, y el desempate por coste de D20 ni siquiera llega a aplicarse.

**Etiquetas del sintético.** Los generadores condicionales (cGAN, cVAE) permiten fijar la proporción de clases. Los que no admiten condicionamiento nativo (gaussiano multivariante, RBIG) se ajustan con un modelo independiente por régimen, de modo que la etiqueta de cada muestra la fija el modelo que la produjo. En todos los casos hay que verificar la proporción efectiva del banco generado y reportarla: el reparto por clase que pide cada receta lo decide `mezclas.repartir`, y si el banco no tiene muestras de un régimen, `mezclas.montar` avisa y omite la cuota en vez de rellenarla en silencio.

### 2.4 Limitaciones de TSTR

- **Depende del modelo downstream.** El ranking puede invertirse al cambiar de modelo: de la `lineal` congelada a cualquiera de las cinco CNN de la búsqueda, o a un gradient boosting. Por eso se fija **un** modelo para toda la comparativa y se congela en disco (D20); las cinco convolucionales del cuaderno 03 quedan disponibles en `downstream.CANDIDATAS` como control de robustez.
- **No detecta memorización.** Un generador que devuelva copias de `real_train` obtiene el TSTR máximo posible. TSTR solo es interpretable junto al test de la sección 4.
- **Alto TSTR no implica estructura correcta.** El TSTR puede mantenerse alto mientras las importancias de variables divergen de las reales; conviene reportar la correlación de rangos entre importancias de permutación del modelo TRTR y del TSTR.
- **No detecta varianza deflactada.** van Breugel et al. (ICML 2023) muestran que los análisis hechos sobre sintéticos subestiman la incertidumbre y producen intervalos de confianza inválidos.[^vanbreugel]

## 3. Fidelidad de distribución

### 3.1 El espacio de trabajo

Con $d = 1.201$ y $n \approx 5.000$ se está en régimen $n \lesssim 5d$: las covarianzas son singulares, las distancias euclídeas se concentran y cualquier métrica basada en vecinos pierde discriminación. Todas las métricas multivariantes se calculan sobre una **proyección PCA de 50 componentes ajustada exclusivamente sobre `real_train`** y aplicada después a real y sintético. Esto elimina la singularidad, reduce el coste de $O(n^2 d)$ a $O(n^2 \cdot 50)$ (factor ~22) y sigue la recomendación estándar de preprocesado para métodos basados en vecinos y para t-SNE.[^bhsne] Las componentes retenidas y la varianza explicada acumulada forman parte del informe. Las métricas univariantes (marginales, ACF) se calculan siempre en el **espacio original**, donde son interpretables.

### 3.2 Discriminative score / Classifier Two-Sample Test

Lopez-Paz y Oquab (ICLR 2017) formalizan el C2ST: se etiquetan los reales como 0 y los sintéticos como 1, se entrena un clasificador binario con validación cruzada y se mide su separación fuera de muestra. Bajo $H_0: P_{\text{real}} = P_{\text{sint}}$ el clasificador rinde a nivel de azar.[^c2st] TimeGAN usa la misma idea como *discriminative score*, definido como $|0{,}5 - a|$ con $a$ el accuracy del discriminador post-hoc.[^timegan] Aquí se reporta **AUC** en lugar de accuracy: es insensible al desbalance entre tamaños de muestra y no depende del umbral.

$$\text{AUC} \approx 0{,}5 \Rightarrow \text{indistinguibles}; \qquad \text{AUC} \to 1{,}0 \Rightarrow \text{trivialmente separables}$$

Precauciones: (i) **validación cruzada obligatoria** (5 folds estratificados) — el AUC en entrenamiento es siempre ~1,0 y no significa nada; (ii) **capacidad controlada** — con 1.201 dimensiones y 5.000 muestras un modelo suficientemente flexible separa cualquier par de muestras, por lo que se usa `HistGradientBoostingClassifier` sobre PCA-50 y, como control, una regresión logística regularizada; si ambos dan 0,5 la conclusión es sólida, y si solo el boosting separa, la diferencia es no lineal y localizada; (iii) **intervalo de confianza** — un AUC de 0,55 ± 0,04 no es distinguible de 0,5; (iv) **diagnóstico** — las importancias de permutación del discriminador indican *qué* delata al sintético, y son la información más accionable de toda la batería.

Umbrales: AUC < 0,60 excelente; 0,60–0,75 aceptable; > 0,85 rechazo.

### 3.3 Maximum Mean Discrepancy

MMD compara las medias de dos muestras en el RKHS inducido por un kernel $k$. El estimador insesgado de Gretton et al. (JMLR 2012) es:[^gretton]

$$\widehat{\text{MMD}}_u^2 = \frac{1}{m(m-1)}\sum_{i \neq i'} k(x_i, x_{i'}) + \frac{1}{n(n-1)}\sum_{j \neq j'} k(y_j, y_{j'}) - \frac{2}{mn}\sum_{i,j} k(x_i, y_j)$$

**Kernel.** RBF gaussiano $k(x,y)=\exp(-\gamma\lVert x-y\rVert^2)$ con $\gamma = 1/(2\sigma^2)$ y $\sigma$ fijado por la **heurística de la mediana** ($\sigma$ = mediana de las distancias euclídeas por pares sobre la muestra conjunta): es la práctica estándar y la potencia del test es estable en torno a ese valor. Conviene promediar sobre una rejilla multiescala $\sigma \in \{\sigma_{\text{med}}/4, \sigma_{\text{med}}/2, \sigma_{\text{med}}, 2\sigma_{\text{med}}, 4\sigma_{\text{med}}\}$ para no depender de una única escala.

**Interpretación.** El valor absoluto de $\widehat{\text{MMD}}_u^2$ no es interpretable por sí solo (depende del kernel y de la escala). Se acompaña siempre de un **test de permutación** (200–500 permutaciones) que da p-valor, y se reporta el estadístico normalizado por su desviación bajo la nula.

**Coste.** $O(n^2)$ en tiempo y memoria: con $n=5.000$ la Gram conjunta es $10^4\times10^4$ ≈ 800 MB en float64, así que se submuestrea a 2.000 por lado. Con 200 permutaciones se reutiliza la Gram permutando índices, sin recalcular el kernel.

### 3.4 Distancia de Wasserstein

En 1D la $W_1$ tiene forma cerrada como área entre las funciones de distribución empíricas, y `scipy.stats.wasserstein_distance` la calcula en $O(n\log n)$.[^scipy] Se aplica **por marginal**, normalizando por la desviación típica real para hacerla adimensional: $\tilde{W}_1(j) = W_1(P^{(j)}_{\text{real}}, P^{(j)}_{\text{sint}}) / \hat{\sigma}^{(j)}_{\text{real}}$.

La $W_2$ multivariante exacta requiere transporte óptimo de coste $O(n^3\log n)$ — inviable aquí. La alternativa es la **Sliced Wasserstein**: se proyecta sobre $L$ direcciones aleatorias de la esfera unidad y se promedian las $W_1$ unidimensionales, con coste $O(L \cdot n\log n)$. Con $L=200$ sobre PCA-50 es barata, pero aporta poco por encima de MMD + C2ST y es más difícil de interpretar: métrica de segundo nivel.

### 3.5 Marginales

Por canal (mejor que por cada una de las 1.201 dimensiones aplanadas): **estadístico KS** $D=\sup_x|F_{\text{real}}(x)-F_{\text{sint}}(x)|$, que SDMetrics reporta como `KSComplement` $=1-D$ con 1,0 óptimo;[^sdmetrics] **momentos** (media, desviación, asimetría, curtosis) — en datos financieros la curtosis es el momento que más generadores fallan, y un gaussiano multivariante producirá por construcción curtosis ≈ 3 frente a valores reales de 5–20; y **colas** (cuantiles 1%, 5%, 95%, 99%), que es donde vive la clase crisis.

**Comparaciones múltiples.** Con 1.201 tests marginales al 5% se esperan ~60 rechazos por puro azar. Se aplica Benjamini–Hochberg (FDR 5%) y se reporta la **fracción de marginales rechazadas**, no la lista de p-valores. Resumen operativo: KS medio, percentil 95 y fracción de canales con $D > 0{,}1$.

### 3.6 Matrices de correlación

**No se calcula la matriz $1.201 \times 1.201$ del vector aplanado**: con $n=5.000$ es una estimación de $\sim7{,}2\cdot10^5$ entradas independientes con 5.000 observaciones, es decir, ruido, y su norma mediría el error de estimación más que el del generador. La comparación se hace a dos niveles interpretables:

1. **Correlación contemporánea entre canales** ($20\times20$), apilando todos los pasos temporales de todas las ventanas. Es lo que implementa `evaluacion.error_correlaciones`, que además descarta la última columna del bloque —la volatilidad futura, que no forma parte de la ventana— antes de deshacer el aplanado. Métrica: error de Frobenius relativo $\lVert C_r - C_s\rVert_F / \lVert C_r\rVert_F$ y máximo error absoluto elemento a elemento.
2. **Correlación cruzada con retardo** entre pares de canales para $\ell = 1,\dots,10$, que captura estructura *lead-lag* que la matriz contemporánea ignora.

Se reportan Pearson y Spearman: la primera captura la estructura lineal, la segunda es robusta a colas pesadas y detecta si el generador ha "gaussianizado" las dependencias. SDMetrics lo implementa como `CorrelationSimilarity`.[^sdmetrics]

## 4. Diversidad y detección de memorización

Esta sección da validez al resto: un generador que copia muestras reales supera todas las métricas de las secciones 2 y 3.

### 4.1 Precision y recall para modelos generativos

Kynkäänniemi et al. (NeurIPS 2019) sustituyen el escalar único por dos cantidades basadas en manifolds no paramétricos.[^kyn] La pertenencia al manifold de un conjunto $\Phi$ se define con hiperesferas centradas en cada punto y radio igual a su distancia al $k$-ésimo vecino más cercano:

$$f(\phi, \Phi) = \begin{cases} 1 & \text{si } \lVert \phi - \phi' \rVert_2 \leq \lVert \phi' - \text{NN}_k(\phi', \Phi) \rVert_2 \text{ para algún } \phi' \in \Phi \\ 0 & \text{en otro caso}\end{cases}$$

**Precision** = fracción de sintéticos dentro del manifold real (fidelidad); **recall** = fracción de reales dentro del manifold sintético (cobertura). Los autores usan $k=3$, valor que describen como robusto y que evita la saturación de la métrica en la mayoría de los casos.

### 4.2 Density y coverage

Naeem et al. (ICML 2020) muestran que precision y recall fallan en tres puntos: no detectan la coincidencia entre distribuciones idénticas, no son robustas a outliers (un solo real atípico infla su hiperesfera y admite cualquier sintético cercano) y tienen hiperparámetros arbitrarios. Proponen dos sustitutos:[^naeem]

$$\text{density} := \frac{1}{kM}\sum_{j=1}^{M}\sum_{i=1}^{N} \mathbb{1}\left[Y_j \in B\left(X_i, \text{NND}_k(X_i)\right)\right] \qquad \text{coverage} := \frac{1}{N}\sum_{i=1}^{N} \mathbb{1}\left[\exists j : Y_j \in B\left(X_i, \text{NND}_k(X_i)\right)\right]$$

`density` cuenta cuántas hiperesferas reales contiene en promedio cada sintético (no se satura en 1; valores > 1 indican concentración excesiva en zonas densas del real); `coverage` mide qué fracción de reales tiene al menos un sintético cerca, y es **la** métrica de mode collapse. Los autores usan $k=5$ con $M=N=10.000$, calibrado para que la coverage esperada supere 0,95 cuando las distribuciones coinciden. Implementación de referencia: `clovaai/generative-evaluation-prdc`.[^prdc]

**Adaptación al taller.** Estas métricas se definieron sobre embeddings de Inception. Aquí no hay red preentrenada relevante: el espacio es la **proyección PCA-50 sobre datos estandarizados**, y esa elección debe declararse porque los valores absolutos no son comparables con los de la literatura de imagen. Lo comparable es el **ranking entre los siete generadores evaluados en el mismo espacio**.

**Diversidad por clase.** Con la crisis en torno al 16 % en train, la `coverage` global puede ser 0,95 mientras la restringida a reales de clase crisis es 0,4. Se calcula `coverage` **estratificada por clase** y se reporta la de crisis por separado: es el diagnóstico de mode collapse más informativo del taller.

### 4.3 Test de memorización por vecino más cercano

La distancia al registro más cercano (DCR) de cada sintético respecto al conjunto de entrenamiento del generador es el test básico; un DCR de cero es copia literal. **El valor absoluto de DCR no dice nada**: depende de la escala, la dimensión y la densidad local. Lo informativo es el diseño **con holdout** de Platzer y Reutterer (2021), que compara las distancias de los sintéticos a `real_train` contra las de un conjunto real de holdout a ese mismo `real_train`.[^platzer] El holdout viene de la misma distribución y por construcción no está memorizado, así que fija la escala de referencia.

| Estadístico | Definición | Objetivo |
|---|---|---|
| `ratio_dcr_mediana` | $\text{med}(d_{\text{sint}\to\text{train}}) / \text{med}(d_{\text{hold}\to\text{train}})$ | ≈ 1,0 (aceptable ≥ 0,9) |
| `ratio_dcr_p5` | mismo cociente en el percentil 5 | ≥ 0,8 — captura la cola de copias |
| `frac_mas_cerca` | fracción de sintéticos con DCR menor que la mediana del holdout | ≈ 0,5 |
| `n_duplicados` | sintéticos con DCR < $10^{-8}$ | **0** |

Un ratio de 0,4 con `frac_mas_cerca` de 0,85 es memorización aunque el DCR absoluto parezca grande. Un ratio > 1,2 tampoco es bueno: indica distribución desplazada. Complemento: `authenticity` de Alaa et al. (2022), fracción de sintéticos que no son "ruido añadido a un punto real", estimada con un clasificador a nivel de muestra e implementada en synthcity.[^alaa]

**Advertencia obligatoria.** Yao et al. (2025) demuestran que DCR y métricas de distancia similares son insuficientes como garantía de privacidad: datasets considerados privados por DCR resultan vulnerables a ataques de inferencia de pertenencia, y recomiendan MIAs como estándar riguroso.[^dcrdelusion] En este taller el objetivo no es una garantía de privacidad sino **descartar la memorización que invalidaría el TSTR**; para eso el test con holdout es adecuado, siempre que se enuncie con esa limitación explícita.

### 4.4 Diversidad intra-conjunto

Dos comprobaciones baratas que detectan colapso sin comparar con el real: **ratio de dispersión** (distancia media entre pares del sintético dividida por la del real; objetivo ≈ 1,0, valores < 0,7 indican concentración) y **fracción de vecinos sintético–sintético a distancia casi nula**, que detecta que el generador emite un puñado de prototipos repetidos.

## 5. Métricas específicas de series temporales

Las métricas de las secciones 3 y 4 tratan cada ventana como un vector y son ciegas al orden temporal: una permutación de los 60 días deja invariantes marginales, correlaciones entre canales y PRDC. Esta sección cubre ese hueco.

### 5.1 Predictive score

TimeGAN entrena un predictor secuencial sobre el sintético y lo evalúa sobre el real: una GRU que, a partir de las primeras $d-1$ variables en $t_1..t_{N-1}$, predice la variable restante en $t_2..t_N$, reportando el MAE sobre el real.[^timegan] Es un TSTR aplicado a predicción a un paso y captura si las dependencias temporales son aprendibles.

**Adaptación CPU-only.** Se sustituye la GRU por una regresión ridge (o un gradient boosting pequeño) sobre los canales en $t$ para predecir el canal objetivo en $t+1$. Se pierde capacidad de modelar no linealidades largas, pero el coste baja de minutos de GPU a segundos de CPU y la comparación entre los siete generadores sigue siendo válida porque el predictor es idéntico para todos. Se reporta el ratio $\text{MAE}_{\text{TSTR}}/\text{MAE}_{\text{TRTR}}$, objetivo $\leq 1{,}15$.

### 5.2 Autocorrelación y hechos estilizados

En series financieras los hechos estilizados de Cont (2001) son el criterio cualitativo de aceptación: ausencia de autocorrelación en los retornos, colas pesadas, agrupamiento de volatilidad y efecto apalancamiento.[^cont] Quant GANs (Wiese et al., 2020) evalúa explícitamente autocorrelaciones seriales, clusters de volatilidad y efecto apalancamiento como criterio de fidelidad.[^quantgans] Se comparan tres funciones, promediadas sobre ventanas y por canal, para retardos $\ell=1,\dots,20$:

| Función | Qué captura | Comportamiento real esperado |
|---|---|---|
| $\text{ACF}(r_t, \ell)$ | Autocorrelación de retornos | ≈ 0 para $\ell \geq 1$ |
| $\text{ACF}(\lvert r_t\rvert, \ell)$ | Agrupamiento de volatilidad | Positiva, decaimiento lento |
| $\text{corr}(r_t, \lvert r_{t+\ell}\rvert)$ | Efecto apalancamiento | Negativa a retardos cortos |

Métrica escalar: MAE entre la ACF real y la sintética sobre los 20 retardos, por función y por canal. La segunda fila es la discriminante: casi todos los generadores reproducen la primera (es trivial, el ruido blanco la satisface) y casi ninguno la segunda. Un generador jitter o gaussiano multivariante fallará sistemáticamente en $\text{ACF}(\lvert r_t\rvert)$. Complemento barato: **densidad espectral** media por canal (periodograma de Welch) comparada en escala log, que detecta periodicidades artificiales.

### 5.3 Visualización: PCA y t-SNE

**PCA (obligatoria, coste despreciable).** Ajustada sobre `real_train`, se proyectan real y sintético sobre las 2 primeras componentes en el mismo gráfico. Es determinista, reproducible e interpretable. Se acompaña de la comparación de las curvas de **varianza explicada acumulada** de PCA ajustada por separado sobre cada conjunto (si el sintético concentra mucha más varianza en pocas componentes, hay colapso de rango) y de los **ángulos principales** entre los subespacios de las 10 primeras componentes: un coseno medio > 0,95 indica que ambos ocupan el mismo subespacio dominante.

**t-SNE (opcional, con restricciones estrictas).** TimeGAN la usa para inspección visual. Su coste es $O(n^2)$ en la versión original y $O(n\log n)$ con la aproximación Barnes-Hut, que en la práctica funciona bien hasta ~100.000 puntos pero se degrada antes; la recomendación original de los autores es preprocesar con PCA a ~50 dimensiones.[^bhsne] Regla operativa:

> **t-SNE solo sobre un submuestreo de ≤ 2.000 reales + 2.000 sintéticos, siempre después de PCA-50, con `perplexity=30` y semilla fija.** Aplicarla sobre 50.000 puntos de 1.201 dimensiones en CPU es inviable (horas por generador × 7 generadores) e innecesario: el gráfico resultante no es cuantitativo.

t-SNE es una herramienta de inspección, no una métrica. No se reporta ningún número derivado de ella y las distancias en el plano t-SNE no tienen interpretación métrica.

## 6. Métricas que no aplican a este problema

**Fréchet Inception Distance.** FID (Heusel et al., 2017) calcula la distancia de Fréchet entre dos gaussianas ajustadas a las activaciones de la capa `pool3` de Inception-v3.[^fid] No es trasladable por cuatro razones independientes: (i) **la red no existe para este dominio** — Inception-v3 se entrena sobre ImageNet, imágenes RGB de 299×299 de escenas naturales, y sus activaciones sobre un panel financiero de 60×20 son la respuesta de detectores de texturas y objetos a una entrada fuera de distribución, sin significado; (ii) **no hay embedding sustituto validado** — reemplazar Inception por un autoencoder propio da un número calculable pero incomparable entre trabajos y dependiente del entrenamiento del encoder; (iii) **supuesto de gaussianidad** en el espacio de características, que en datos financieros de colas pesadas es precisamente lo que se quiere testar, no lo que se puede asumir; (iv) **requisito de muestra** — una estimación estable necesita del orden de 10.000 muestras por lado y el estimador está sesgado con muestras pequeñas, mientras aquí hay ~5.000 reales en total.

> Nota práctica: synthcity expone una métrica `fid` documentada como aplicable únicamente a datos de imagen.[^synthcity] No usarla sobre este dataset.

Lo legítimo es transportar la *idea*, no la métrica: una distancia de Fréchet entre gaussianas ajustadas en el espacio PCA-50 es calculable y comparable *dentro* de este taller, pero entonces **no es FID**, no debe llamarse así y aporta poco por encima de MMD, que no requiere el supuesto gaussiano. Queda fuera de la batería.

**Inception Score.** IS (Salimans et al., 2016) mide $\exp(\mathbb{E}_x \text{KL}(p(y|x)\Vert p(y)))$ sobre las posteriores de las 1.000 clases de ImageNet.[^is] Tiene un defecto adicional decisivo: **no usa los datos reales en ningún punto del cálculo**, de modo que un generador puede obtener un IS alto produciendo muestras nítidas de una distribución completamente distinta de la real. Aquí ni siquiera está definido: no hay taxonomía de clases naturales sobre la que evaluar posteriores.

**Otras.** SSIM, PSNR y LPIPS son métricas de similitud perceptual pareada entre imágenes: requieren correspondencia uno a uno entre muestras (inexistente entre real y sintético) y un modelo de percepción visual humana (irrelevante para series de precios). $k$-anonimato, $l$-diversidad y $\delta$-presencia, disponibles en synthcity, están definidas sobre tablas con cuasi-identificadores categóricos; sobre un panel continuo de 1.201 dimensiones cada registro es único y todas devuelven el valor degenerado.

## 7. Batería de métricas elegida para este taller

P0 = obligatoria en el informe final; P1 = recomendada; P2 = opcional si sobra presupuesto. Los costes son órdenes de magnitud por generador para $n=5.000$ y $d=1.201$, medidos sobre una CPU de 4 núcleos; las métricas multivariantes se calculan en PCA-50, con el proyector que ajusta `evaluacion.proyector` una sola vez sobre los reales, de modo que todos los generadores se juzguen en el mismo espacio.

La columna **Implementación** dice qué existe hoy en `src/evaluacion.py` y qué no. Las filas marcadas como **pendiente** no tienen función en el módulo: se enuncian aquí como especificación de lo que habría que escribir, y ninguna cifra del informe puede citarlas hasta que exista el código. `evaluacion.bateria_calidad` encadena todas las implementadas y devuelve una fila de resultados por generador.

| # | Prio | Qué mide | Implementación | Valor "bueno" | Coste |
|---|---|---|---|---|---|
| 1 | P0 | Utilidad downstream, como ratio frente al baseline TRTR | `evaluacion.evaluar` sobre el modelo congelado, más `metricas_regimen` / `metricas_volatilidad`; el ratio se calcula en el cuaderno | ratio ≥ 0,90 excelente, ≥ 0,80 aceptable; recall de crisis ≥ 0,8 × TRTR | 4 entrenamientos de la arquitectura congelada |
| 2 | P0 | Rendimiento downstream frente al ratio sintético/real, por nivel de reales y política | no es una función: es el barrido del cuaderno 12, `mezclas.rejilla` + `mezclas.montar` + `downstream.entrenar` + `evaluacion.evaluar`, persistido con `evaluacion.acumular` | Máximo en $\rho>0$ · el sintético aporta; caída monótona · no aporta | 285 recetas por tarea, 570 entrenamientos en total |
| 3 | P0 | C2ST: separabilidad real vs. sintético | **`puntuacion_discriminativa`** — `HistGradientBoostingClassifier` sobre PCA-50, partición 70/30 estratificada. Devuelve `auc_discriminativo` y `distancia_a_indistinguible` | < 0,60 excelente; 0,60–0,75 aceptable; > 0,85 rechazo | 30 s – 3 min |
| 4 | P0 | Memorización: distancia al vecino real más cercano | **`distancia_vecino_mas_cercano`** — devuelve `cociente_dvmc`, `frac_mas_cerca` y `duplicados_exactos`. **Ojo**: la escala de referencia son las distancias **real-real** (segundo vecino), no un holdout | cociente ≈ 1,0, aceptable ≥ 0,9; `frac_mas_cerca` ≈ 0,5; duplicados = 0 | 5–20 s |
| 5 | P0 | Cobertura del soporte real; mode collapse | **pendiente** — `coverage` de Naeem ($k=5$), global y estratificada por clase, no está implementada | ≥ 0,80 global y ≥ 0,70 en clase crisis | 30–120 s |
| 6 | P0 | Marginales: momentos y colas | **`comparar_momentos`** da media, desviación, asimetría y curtosis, y `bateria_calidad` publica `dif_curtosis` y `dif_asimetria`. El resumen KS por canal (medio, p95, fracción con $D>0{,}1$) está **pendiente** | curtosis dentro de ±30 % de la real; KS medio < 0,05 cuando exista | 5–15 s |
| 7 | P0 | Frobenius relativo de la matriz 20×20 entre canales | **`error_correlaciones`** — solo Pearson; la variante Spearman y el máximo absoluto elemento a elemento están **pendientes** | < 0,10 excelente; < 0,20 aceptable | < 1 s |
| 8 | P1 | Estructura temporal y agrupamiento de volatilidad, retardos 1–20 | **`error_autocorrelacion`** — devuelve `error_acf_retornos` y `error_acf_absolutos`, sobre el canal 0 (retorno del índice) | MAE < 0,05 en ambas; la de $\lvert r_t\rvert$ es la discriminante | 2–10 s |
| 9 | P1 | Fidelidad robusta a outliers (`density`, Naeem $k=5$) | **pendiente** | ∈ [0,8 ; 1,2] | incluido en (5) |
| 10 | P1 | Discrepancia distribucional global con p-valor (MMD² RBF + permutaciones) | **pendiente** | $p>0{,}05$, o MMD² a menos de 2 desviaciones de la nula | 1–4 min (2.000/lado) |
| 11 | P1 | Aprendibilidad de la dinámica temporal (predictive score, ridge AR a 1 paso) | **pendiente** | ratio MAE ≤ 1,15 | 5–20 s |
| 12 | P1 | Inspección visual y colapso de rango (PCA overlay, varianza explicada, ángulos principales) | **pendiente** como figura; el proyector ya lo da `evaluacion.proyector` | Solapamiento visual; coseno medio de los 10 primeros > 0,95 | < 5 s |
| 13 | P2 | Precision/recall de Kynkäänniemi ($k=3$) | **pendiente** | ambas ≥ 0,7 | incluido en (5) |
| 14 | P2 | Sliced Wasserstein (200 proyecciones) | **pendiente** | Menor es mejor, solo comparación relativa | 10–30 s |
| 15 | P2 | Inspección cualitativa t-SNE (≤ 2.000 + 2.000 puntos, tras PCA-50) | **pendiente** | Nubes superpuestas, sin islas puramente sintéticas | 1–3 min |
| 16 | P2 | Auditoría a nivel de muestra (synthcity: `alpha_precision`, `beta_recall`, `authenticity`) | **pendiente**, dependencia externa | `authenticity` ≥ 0,7 | 2–10 min |

**Criterio de aceptación de un generador**, con lo que hoy se puede medir: `distancia_vecino_mas_cercano` con `cociente_dvmc` ≥ 0,9 y cero duplicados exactos — **si falla esto, el resto de métricas no se interpreta**; `auc_discriminativo` < 0,85; y ratio de utilidad downstream ≥ 0,70 en F1 macro. El cuarto criterio histórico —`coverage` en clase crisis ≥ 0,70— **no es exigible mientras la métrica siga pendiente**, y no se puede dar por cumplido: hasta que se implemente, el diagnóstico de mode collapse en la clase de crisis descansa en `dif_curtosis` y en la inspección del banco por régimen, que son más débiles.

El ranking final se ordena por el resultado del barrido del cuaderno 12: mejor F1 macro alcanzado en cualquier $\rho>0$, con su desviación entre semillas, y leído siempre contra la línea base de persistencia causal (D21) y no contra el cero absoluto.

**Presupuesto estimado** de la parte de calidad sintética: las cuatro métricas implementadas cuestan del orden de un minuto por generador, dominadas por el C2ST, único punto donde se usa boosting; si el presupuesto aprieta, reducir `max_iter` a 50 con `early_stopping=True` antes que recortar cualquier otra. El proyector PCA se ajusta una sola vez y se reutiliza en las filas 3 y 4. La partida realmente cara del taller no es esta batería sino el barrido de la fila 2.

## 8. Implementación de referencia

> **No vinculante.** Lo que se ejecuta es `src/evaluacion.py`, y la tabla de la sección 7 dice qué hay allí y qué no. El código que sigue es una implementación de referencia, escrita antes que el módulo, que conserva su valor como especificación de las métricas **pendientes** —MMD, PRDC, marginales KS, predictive score— y como documentación de los supuestos de cada fórmula. Donde discrepe del módulo, manda el módulo. En particular: el test de memorización de aquí usa un **holdout real** como escala de referencia y el implementado usa las distancias **real-real**; y el modelo downstream de aquí es una logística de scikit-learn, mientras que el del taller es la candidata `lineal` congelada de la sección 2.3, que es la misma multinomial sobre la ventana aplanada pero entrenada en Keras, con dropout y con el presupuesto de D20.

```python
"""Batería de métricas de calidad para datos sintéticos (taller B5-T1).

Convenciones:
  X_* : matrices aplanadas (n, d) con d = 1201
  P_* : paneles sin aplanar (n, T, C) con T=60, C=20
  Z_* : representaciones proyectadas a PCA-50 (n, 50)
"""
import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import ks_2samp, wasserstein_distance
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (balanced_accuracy_score, f1_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# --- 0. Espacio de evaluación común ---------------------------------------
def espacio_evaluacion(X_real_train, n_componentes=50, semilla=0):
    """Estandarización + PCA ajustadas SOLO sobre el real de entrenamiento.

    Ajustar la PCA incluyendo el sintético contaminaría la comparación: el
    espacio lo define el real, no lo que el generador haya producido.
    """
    escalador = StandardScaler().fit(X_real_train)
    pca = PCA(n_components=n_componentes, random_state=semilla)
    pca.fit(escalador.transform(X_real_train))
    return (lambda X: pca.transform(escalador.transform(X))), pca

# --- 1. Fidelidad: discriminative score (C2ST) ----------------------------
def discriminative_auc(Z_real, Z_sint, semilla=0, n_folds=5):
    """AUC fuera de muestra de un clasificador real-vs-sintético.

    0.5 = indistinguibles (ideal). Se reporta media y desviación entre folds:
    un 0.55 +- 0.04 no es estadísticamente distinto de 0.5.
    """
    X = np.vstack([Z_real, Z_sint])
    y = np.concatenate([np.zeros(len(Z_real)), np.ones(len(Z_sint))])
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=semilla)
    aucs = []
    for idx_tr, idx_te in cv.split(X, y):
        clf = HistGradientBoostingClassifier(max_iter=100, early_stopping=True,
                                             random_state=semilla)
        clf.fit(X[idx_tr], y[idx_tr])
        aucs.append(roc_auc_score(y[idx_te], clf.predict_proba(X[idx_te])[:, 1]))
    return {"auc_media": float(np.mean(aucs)), "auc_std": float(np.std(aucs))}

# --- 2. Fidelidad: MMD con kernel RBF y test de permutación ---------------
def _mmd2_desde_gram(K, m, n):
    """MMD^2 insesgado a partir de la Gram conjunta (m+n, m+n)."""
    Kxx, Kyy, Kxy = K[:m, :m], K[m:, m:], K[:m, m:]
    return ((Kxx.sum() - np.trace(Kxx)) / (m * (m - 1))
            + (Kyy.sum() - np.trace(Kyy)) / (n * (n - 1))
            - 2.0 * Kxy.mean())

def mmd2_rbf(Z_real, Z_sint, semilla=0, n_permutaciones=200, n_max=2000):
    """MMD^2 insesgado (Gretton et al., 2012), ancho por heurística de la mediana.

    Submuestrea a n_max por lado: la Gram es O(n^2) en memoria. El test de
    permutación reutiliza la Gram ya calculada, permutando índices.
    """
    rng = np.random.default_rng(semilla)
    if len(Z_real) > n_max:
        Z_real = Z_real[rng.choice(len(Z_real), n_max, replace=False)]
    if len(Z_sint) > n_max:
        Z_sint = Z_sint[rng.choice(len(Z_sint), n_max, replace=False)]
    Z = np.vstack([Z_real, Z_sint])
    m, n = len(Z_real), len(Z_sint)

    sub = Z[rng.choice(len(Z), min(1000, len(Z)), replace=False)]
    d = cdist(sub, sub)
    mediana = np.median(d[np.triu_indices(len(sub), k=1)])
    gamma = 1.0 / (2.0 * mediana ** 2)

    K = np.exp(-gamma * cdist(Z, Z, "sqeuclidean"))
    estadistico = _mmd2_desde_gram(K, m, n)
    nulos = np.array([_mmd2_desde_gram(K[np.ix_(p, p)], m, n)
                      for p in (rng.permutation(m + n)
                                for _ in range(n_permutaciones))])
    return {"mmd2": float(estadistico),
            "p_valor": float((1 + np.sum(nulos >= estadistico)) / (n_permutaciones + 1)),
            "z_score": float((estadistico - nulos.mean()) / (nulos.std() + 1e-12)),
            "gamma": float(gamma)}

# --- 3. Fidelidad: marginales y correlaciones -----------------------------
def _curtosis(X):
    z = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-12)
    return (z ** 4).mean(axis=0)

def resumen_marginales(X_real, X_sint, umbral_ks=0.10):
    """KS y W1 normalizada por columna. Se resume: no se listan 1201 p-valores."""
    d = X_real.shape[1]
    ks, w1 = np.empty(d), np.empty(d)
    for j in range(d):
        a, b = X_real[:, j], X_sint[:, j]
        ks[j] = ks_2samp(a, b).statistic
        w1[j] = wasserstein_distance(a, b) / (a.std() + 1e-12)
    return {"ks_medio": float(ks.mean()),
            "ks_p95": float(np.percentile(ks, 95)),
            "frac_ks_alto": float((ks > umbral_ks).mean()),
            "w1_norm_media": float(w1.mean()),
            "curtosis_real": float(_curtosis(X_real).mean()),
            "curtosis_sint": float(_curtosis(X_sint).mean())}

def error_correlacion(P_real, P_sint, metodo="pearson"):
    """Compara la matriz C x C de correlaciones entre canales.

    NO se compara la matriz 1201 x 1201 del vector aplanado: con n~5000 sería
    una estimación de ~7,2e5 entradas con 5e3 observaciones, es decir, ruido.
    """
    def matriz(P):
        Z = P.reshape(-1, P.shape[2])
        if metodo == "spearman":                      # rangos por columna
            Z = np.argsort(np.argsort(Z, axis=0), axis=0).astype(float)
        return np.corrcoef(Z, rowvar=False)

    dif = matriz(P_real) - matriz(P_sint)
    return {"frobenius_relativo": float(np.linalg.norm(dif)
                                        / np.linalg.norm(matriz(P_real))),
            "max_abs": float(np.abs(dif).max())}

# --- 4. Diversidad y memorización -----------------------------------------
def prdc(Z_real, Z_sint, k=5):
    """Precision/recall (Kynkäänniemi 2019) y density/coverage (Naeem 2020).

    k=5 sigue a Naeem et al.; para replicar Kynkäänniemi usar k=3.
    Coste O(n^2 * dim): usar SIEMPRE el espacio PCA-50, no las 1201 dims.
    """
    d_rr = cdist(Z_real, Z_real); np.fill_diagonal(d_rr, np.inf)
    radio_r = np.sort(d_rr, axis=1)[:, k - 1]          # NND_k de cada real
    d_ss = cdist(Z_sint, Z_sint); np.fill_diagonal(d_ss, np.inf)
    radio_s = np.sort(d_ss, axis=1)[:, k - 1]          # NND_k de cada sintético

    d_rs = cdist(Z_real, Z_sint)                       # (N_real, N_sint)
    en_bola_real = d_rs <= radio_r[:, None]            # sintético en bola real
    en_bola_sint = d_rs <= radio_s[None, :]            # real en bola sintética
    return {"precision": float(en_bola_real.any(axis=0).mean()),
            "recall": float(en_bola_sint.any(axis=1).mean()),
            "density": float(en_bola_real.sum(axis=0).mean() / k),
            "coverage": float(en_bola_real.any(axis=1).mean())}

def coverage_por_clase(Z_real, y_real, Z_sint, y_sint, k=5):
    """Coverage estratificada: con la crisis al 10-16%, la global la enmascara."""
    salida = {}
    for c in np.unique(y_real):
        zr, zs = Z_real[y_real == c], Z_sint[y_sint == c]
        salida[int(c)] = (float("nan") if min(len(zr), len(zs)) <= k
                          else prdc(zr, zs, k=k)["coverage"])
    return salida

def memorizacion_dcr(Z_train, Z_holdout, Z_sint, bloque=512):
    """Test de memorización con holdout (Platzer y Reutterer, 2021).

    El holdout real fija la escala: sus distancias a Z_train son las de datos
    NO memorizados procedentes de la misma distribución.
    """
    def dcr(Z):
        out = np.empty(len(Z))
        for i in range(0, len(Z), bloque):
            out[i:i + bloque] = cdist(Z[i:i + bloque], Z_train).min(axis=1)
        return out

    d_sint, d_hold = dcr(Z_sint), dcr(Z_holdout)
    return {"ratio_dcr_mediana": float(np.median(d_sint) / (np.median(d_hold) + 1e-12)),
            "ratio_dcr_p5": float(np.percentile(d_sint, 5)
                                  / (np.percentile(d_hold, 5) + 1e-12)),
            "frac_mas_cerca": float((d_sint < np.median(d_hold)).mean()),
            "n_duplicados": int((d_sint < 1e-8).sum())}

# --- 5. Series temporales --------------------------------------------------
def acf_panel(P, max_lag=20, sobre_abs=False):
    """ACF media por canal. P: (n, T, C) -> (C, max_lag). Vectorizada.

    sobre_abs=True da la ACF de |r_t|, que mide agrupamiento de volatilidad:
    es la función que separa a los generadores buenos de los malos.
    """
    S = np.abs(P) if sobre_abs else P
    S = S - S.mean(axis=1, keepdims=True)
    var = (S ** 2).sum(axis=1) + 1e-12                        # (n, C)
    out = np.empty((P.shape[2], max_lag))
    for l in range(1, max_lag + 1):
        out[:, l - 1] = ((S[:, :-l, :] * S[:, l:, :]).sum(axis=1) / var).mean(axis=0)
    return out

def error_acf(P_real, P_sint, max_lag=20):
    """MAE entre ACF real y sintética, para retornos y para |retornos|."""
    a = np.abs(acf_panel(P_real, max_lag) - acf_panel(P_sint, max_lag))
    b = np.abs(acf_panel(P_real, max_lag, True) - acf_panel(P_sint, max_lag, True))
    return {"mae_acf_retornos": float(a.mean()),
            "mae_acf_abs": float(b.mean()),
            "mae_acf_abs_por_canal": b.mean(axis=1)}

def predictive_score(P_entrena, P_evalua, canal_objetivo=-1):
    """Predictive score de TimeGAN con predictor lineal (viable en CPU).

    Predice el canal objetivo en t+1 a partir de todos los canales en t.
    Entrenar con sintético y evaluar sobre real es la variante TSTR.
    """
    def matriz(P):
        return (P[:, :-1, :].reshape(-1, P.shape[2]),
                P[:, 1:, canal_objetivo].reshape(-1))

    X_tr, y_tr = matriz(P_entrena)
    X_te, y_te = matriz(P_evalua)
    return float(np.abs(Ridge(alpha=1.0).fit(X_tr, y_tr).predict(X_te) - y_te).mean())

# --- 6. Utilidad: TSTR / TRTS / TRTR y curva de mezcla ---------------------
def modelo_downstream(tipo="logistica", semilla=0):
    """Sustituto ligero del downstream, SOLO para prototipar esta bateria.

    El modelo downstream del taller NO es exactamente este: es la candidata
    `lineal` congelada en models/downstream/arquitectura.json (D20), construida por
    downstream.construir y entrenada por downstream.entrenar. Esa candidata gano la
    busqueda entre las seis y tampoco tiene convoluciones: es una multinomial sobre
    la ventana aplanada, asi que la logistica de aqui es su equivalente barato en
    scikit-learn, sin dropout ni el presupuesto de D20, y sirve para iterar sobre el
    codigo de las metricas sin pagar un entrenamiento completo por prueba.
    """
    if tipo == "boosting":
        return HistGradientBoostingClassifier(
            max_iter=150, learning_rate=0.08, early_stopping=True,
            l2_regularization=1.0, random_state=semilla)
    return make_pipeline(StandardScaler(),
                         LogisticRegression(C=0.1, max_iter=2000,
                                            random_state=semilla))

def entrena_evalua(X_tr, y_tr, X_te, y_te, clase_minoritaria=2, semilla=0,
                   tipo="logistica"):
    """Métricas para clasificación desbalanceada. El accuracy es contexto, no
    criterio: con la crisis al 10,5% en test, un modelo que nunca la prediga
    puede superar 0.85 de accuracy con recall de crisis igual a cero.

    Sin `class_weight="balanced"` a propósito en la rama principal: reponderar
    enmascararía el efecto del sintético sobre la clase minoritaria, que es lo
    que se mide. La rama reponderada existe aparte y es obligatoria (D15).
    """
    y_pred = modelo_downstream(tipo, semilla).fit(X_tr, y_tr).predict(X_te)
    return {"f1_macro": float(f1_score(y_te, y_pred, average="macro")),
            "balanced_accuracy": float(balanced_accuracy_score(y_te, y_pred)),
            "recall_crisis": float(recall_score(y_te, y_pred, average="macro",
                                                labels=[clase_minoritaria],
                                                zero_division=0)),
            "accuracy": float((y_pred == y_te).mean())}

def tstr_trts(X_real_tr, y_real_tr, X_real_te, y_real_te, X_sint, y_sint, **kw):
    """Los cuatro entrenamientos con el mismo modelo y la misma semilla."""
    trtr = entrena_evalua(X_real_tr, y_real_tr, X_real_te, y_real_te, **kw)
    tstr = entrena_evalua(X_sint, y_sint, X_real_te, y_real_te, **kw)
    return {"TRTR": trtr, "TSTR": tstr,
            "TRTS": entrena_evalua(X_real_tr, y_real_tr, X_sint, y_sint, **kw),
            "TSTS": entrena_evalua(X_sint, y_sint, X_sint, y_sint, **kw),
            "ratio_TSTR_TRTR": {k: tstr[k] / (trtr[k] + 1e-12) for k in trtr}}

def curva_mezcla(X_real_tr, y_real_tr, X_sint, y_sint, X_real_te, y_real_te,
                 ratios=(0.0, 0.5, 1.0, 2.0, 5.0), semillas=(0, 1, 2), **kw):
    """F1-macro downstream frente al ratio sintético/real. Eje central del taller.

    AUMENTACIÓN, no sustitución: el sintético se AÑADE al real y el tamaño total
    crece con el ratio. El efecto del tamaño muestral se aísla con el segundo eje
    del barrido —el número de reales disponibles—, no manteniendo el total fijo.

    Versión reducida: la vigente es src/mezclas.py, que además cruza con los
    niveles de reales (250, 500, 1000, 2000, todos) y reparte los sintéticos por
    clase según la política (`proporcional` o `equilibrado`).
    """
    n_real, filas = len(X_real_tr), []
    for r in ratios:
        for s in semillas:
            rng = np.random.default_rng(s)
            n_s = int(round(r * n_real))
            if n_s == 0:
                X, y = X_real_tr, y_real_tr
            else:
                idx_s = rng.choice(len(X_sint), n_s, replace=n_s > len(X_sint))
                X = np.vstack([X_real_tr, X_sint[idx_s]])
                y = np.concatenate([y_real_tr, y_sint[idx_s]])
            filas.append({"ratio_sintetico": r, "semilla": s,
                          **entrena_evalua(X, y, X_real_te, y_real_te,
                                           semilla=s, **kw)})
    return filas

# --- 7. Driver: batería P0 completa para un generador ---------------------
def evaluar_generador(nombre, P_real_tr, y_real_tr, P_real_te, y_real_te,
                      P_holdout, P_sint, y_sint, semilla=0):
    """Ejecuta la batería P0 y devuelve un diccionario plano de resultados."""
    aplanar = lambda P: P.reshape(len(P), -1)
    X_real_tr, X_real_te = aplanar(P_real_tr), aplanar(P_real_te)
    X_holdout, X_sint = aplanar(P_holdout), aplanar(P_sint)

    proyectar, pca = espacio_evaluacion(X_real_tr, 50, semilla)
    Z_tr, Z_hold, Z_sint = proyectar(X_real_tr), proyectar(X_holdout), proyectar(X_sint)

    res = {"generador": nombre,
           "var_explicada_pca50": float(pca.explained_variance_ratio_.sum())}
    res.update(memorizacion_dcr(Z_tr, Z_hold, Z_sint))       # (4) va PRIMERO
    res.update(discriminative_auc(Z_tr, Z_sint, semilla))    # (3)
    res.update(prdc(Z_tr, Z_sint, k=5))                      # (5), (9), (13)
    res["coverage_por_clase"] = coverage_por_clase(Z_tr, y_real_tr, Z_sint, y_sint)
    res.update(resumen_marginales(X_real_tr, X_sint))        # (6)
    res.update(error_correlacion(P_real_tr, P_sint))         # (7)
    res.update(error_acf(P_real_tr, P_sint))                 # (8)
    res["utilidad"] = tstr_trts(X_real_tr, y_real_tr, X_real_te, y_real_te,
                                X_sint, y_sint, semilla=semilla)          # (1)
    res["curva_mezcla"] = curva_mezcla(X_real_tr, y_real_tr, X_sint, y_sint,
                                       X_real_te, y_real_te)              # (2)
    # si hay memorización, el resto de métricas no es interpretable
    res["memorizacion_ok"] = (res["ratio_dcr_mediana"] >= 0.9
                              and res["n_duplicados"] == 0)
    return res
```

### 8.1 Frameworks externos

| Framework | Instalable | Útil aquí | Comentario |
|---|---|---|---|
| **SDMetrics** (SDV) | `pip install sdmetrics`, puro Python | Parcial | Diseñado para tablas mixtas fila a fila. Aporta `KSComplement`, `TVComplement`, `CorrelationSimilarity`, `RangeCoverage` y los informes de calidad. Sobre un panel aplanado de 1.201 columnas los informes son lentos y poco legibles: usarlo solo sobre los 20 canales agregados.[^sdmetrics] |
| **synthcity** | `pip install synthcity`; Python 3.7–3.10 | Sí, con cuidado | Aporta `prdc`, `alpha_precision` (α-precision, β-recall, authenticity), `max_mean_discrepancy`, `wasserstein_dist`, `detection_xgb` y métricas de rendimiento downstream. Su `fid` es solo para imagen. Las métricas de privacidad formal (`k_anonymization`, `l_diversity`) son degeneradas sobre datos continuos de alta dimensión.[^synthcity] |
| **generative-evaluation-prdc** | `pip install prdc` (repo Clova AI) | Sí | Implementación canónica de precision/recall/density/coverage en una función, `compute_prdc(real_features, fake_features, nearest_k)`. Ligera y sin dependencias pesadas.[^prdc] |
| **POT** (Python Optimal Transport) | `pip install pot` | Opcional | Solo necesario para Sliced Wasserstein multivariante; para $W_1$ marginal basta `scipy`. |

**Recomendación**: implementar la batería P0 con `numpy`/`scipy`/`sklearn` (código anterior) para controlar el espacio de evaluación y los splits, y usar synthcity solo como contraste de sanidad sobre las métricas P2. Depender de un framework externo para las métricas de titular introduce supuestos de preprocesado que no quedan visibles en el informe.

## 9. Referencias

[^esteban]: Esteban, C., Hyland, S. L., Rätsch, G. (2017). *Real-valued (Medical) Time Series Generation with Recurrent Conditional GANs*. arXiv:1706.02633. https://arxiv.org/abs/1706.02633
[^c2st]: Lopez-Paz, D., Oquab, M. (2017). *Revisiting Classifier Two-Sample Tests*. ICLR 2017. arXiv:1610.06545. https://arxiv.org/abs/1610.06545
[^gretton]: Gretton, A., Borgwardt, K. M., Rasch, M. J., Schölkopf, B., Smola, A. (2012). *A Kernel Two-Sample Test*. JMLR 13:723–773. https://www.jmlr.org/papers/volume13/gretton12a/gretton12a.pdf
[^kyn]: Kynkäänniemi, T., Karras, T., Laine, S., Lehtinen, J., Aila, T. (2019). *Improved Precision and Recall Metric for Assessing Generative Models*. NeurIPS 2019. arXiv:1904.06991. https://arxiv.org/abs/1904.06991
[^naeem]: Naeem, M. F., Oh, S. J., Uh, Y., Choi, Y., Yoo, J. (2020). *Reliable Fidelity and Diversity Metrics for Generative Models*. ICML 2020. arXiv:2002.09797. https://arxiv.org/abs/2002.09797
[^prdc]: Clova AI Research (2020). *generative-evaluation-prdc*. https://github.com/clovaai/generative-evaluation-prdc
[^alaa]: Alaa, A., van Breugel, B., Saveliev, E. S., van der Schaar, M. (2022). *How Faithful is your Synthetic Data? Sample-level Metrics for Evaluating and Auditing Generative Models*. ICML 2022, PMLR 162. https://proceedings.mlr.press/v162/alaa22a.html
[^timegan]: Yoon, J., Jarrett, D., van der Schaar, M. (2019). *Time-series Generative Adversarial Networks*. NeurIPS 2019. https://www.vanderschaar-lab.com/papers/NIPS2019_TGAN_Main.pdf · Código y métricas: https://github.com/jsyoon0823/TimeGAN
[^platzer]: Platzer, M., Reutterer, T. (2021). *Holdout-Based Empirical Assessment of Mixed-Type Synthetic Data*. Frontiers in Big Data 4:679939. https://www.frontiersin.org/journals/big-data/articles/10.3389/fdata.2021.679939/full · arXiv:2104.00635
[^dcrdelusion]: Yao, Z., Krčo, N., Ganev, G., de Montjoye, Y.-A. (2025). *The DCR Delusion: Measuring the Privacy Risk of Synthetic Data*. arXiv:2505.01524. https://arxiv.org/abs/2505.01524
[^duli]: Du, Y., Li, N. (2024). *Systematic Assessment of Tabular Data Synthesis*. arXiv:2402.06806 (ACM CCS 2025). https://arxiv.org/abs/2402.06806
[^vanbreugel]: van Breugel, B., Qian, Z., van der Schaar, M. (2023). *Synthetic Data, Real Errors: How (Not) to Publish and Use Synthetic Data*. ICML 2023, PMLR 202. https://proceedings.mlr.press/v202/van-breugel23a/van-breugel23a.pdf
[^synthcity]: Qian, Z., Cebere, B.-C., van der Schaar, M. (2023). *Synthcity: facilitating innovative use cases of synthetic data in different data modalities*. arXiv:2301.07573; NeurIPS 2023 Datasets & Benchmarks. https://arxiv.org/abs/2301.07573 · https://github.com/vanderschaarlab/synthcity
[^sdmetrics]: SDV Team. *SDMetrics: Metrics to evaluate quality and efficacy of synthetic datasets*. https://github.com/sdv-dev/SDMetrics · Documentación: https://docs.sdv.dev/sdmetrics
[^fid]: Heusel, M., Ramsauer, H., Unterthiner, T., Nessler, B., Hochreiter, S. (2017). *GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium*. NeurIPS 2017. arXiv:1706.08500. https://arxiv.org/abs/1706.08500
[^is]: Salimans, T., Goodfellow, I., Zaremba, W., Cheung, V., Radford, A., Chen, X. (2016). *Improved Techniques for Training GANs*. NeurIPS 2016. arXiv:1606.03498. https://arxiv.org/abs/1606.03498
[^bhsne]: van der Maaten, L. (2013). *Barnes-Hut-SNE*. ICLR 2013. https://lvdmaaten.github.io/publications/papers/ICLR_2013.pdf
[^cont]: Cont, R. (2001). *Empirical properties of asset returns: stylized facts and statistical issues*. Quantitative Finance 1(2):223–236. https://doi.org/10.1080/713665670
[^quantgans]: Wiese, M., Knobloch, R., Korn, R., Kretschmer, P. (2020). *Quant GANs: deep generation of financial time series*. Quantitative Finance 20(9):1419–1440. https://www.tandfonline.com/doi/abs/10.1080/14697688.2020.1730426
[^brodersen]: Brodersen, K. H., Ong, C. S., Stephan, K. E., Buhmann, J. M. (2010). *The Balanced Accuracy and Its Posterior Distribution*. ICPR 2010, 3121–3124. https://doi.org/10.1109/ICPR.2010.764
[^prado]: López de Prado, M. (2018). *Advances in Financial Machine Learning*. Wiley, cap. 7 (purged k-fold CV y embargo). https://www.wiley.com/en-us/Advances+in+Financial+Machine+Learning-p-9781119482086
[^scipy]: SciPy Developers. *scipy.stats*: `wasserstein_distance`, `ks_2samp`, `energy_distance`. https://docs.scipy.org/doc/scipy/reference/stats.html
