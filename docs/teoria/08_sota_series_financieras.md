# Estado del arte: generación de series financieras sintéticas

Documento de referencia bibliográfica del taller B5-T1. Cubre los modelos generativos publicados para series temporales financieras, los criterios de validación específicos del dominio y la delimitación explícita de qué es alcanzable bajo la restricción operativa del taller (CPU, 16 días).

Notación coherente con `00_fundamentos.md`: $x \in \mathbb{R}^d$ es el bloque conjunto $[\,X ; y_{\text{vol}}\,]$, con $X$ ventana de $60 \times 20$ y $y_{\text{reg}}$ aparte como condición. Los log-retornos se denotan $r_t = \log(P_t/P_{t-1})$.

---

## 1. Panorama y taxonomía

La literatura de generación de series financieras se organiza en tres ejes que conviene no confundir, porque un modelo puede ser excelente en uno e irrelevante en otro.

**Eje 1 — objeto modelado.**

| Objeto | Qué se aprende | Ejemplos |
|---|---|---|
| Marginal de retornos | $p(r)$, sin dinámica | Ajuste $t$-Student, kernel |
| Ley del camino completo | $p(r_1,\dots,r_T)$ | QuantGAN, TimeGAN, Diffusion-TS |
| Ley condicional al pasado | $p(r_{t+1:t+h} \mid r_{1:t})$ | SigCWGAN, CSDI |
| Ley condicional a un estado | $p(x \mid c)$ con $c$ = régimen | RSQGAN, CoFinDiff, cGAN/cVAE |

El taller trabaja en la tercera y cuarta filas: ventanas de longitud fija con etiqueta de régimen, no una simulación de horizonte arbitrario.

**Eje 2 — familia generativa.** Adversarial (GAN), variable latente (VAE), difusión/score, flujos normalizantes, autorregresivos, y no-neuronales (RBM, cópulas, gaussianización). La correspondencia con densidad explícita/implícita está en `00_fundamentos.md` §2.

**Eje 3 — criterio de éxito.** Y aquí está el problema central del campo. Un generador puede pasar tests de fidelidad distribucional y fracasar aguas abajo. Zhang et al. (2026, preprint) documentan precisamente esa disociación: modelos que producen series visualmente convincentes y con momentos correctos rinden sistemáticamente peor en tareas financieras posteriores, lo que sugiere que capturan estructura superficial y no las dependencias que importan para decidir ([arXiv:2601.12990](https://arxiv.org/abs/2601.12990)). En la misma línea, Kwon y Lee (ICAIF '24) encuentran que la capacidad de un GAN de reproducir hechos estilizados **varía drásticamente según la arquitectura del generador**, y concluyen que aplicar GANs a este dominio exige selección y validación arquitectónica explícita, no implementación ingenua ([arXiv:2410.09850](https://arxiv.org/abs/2410.09850), [DOI ACM](https://dl.acm.org/doi/10.1145/3677052.3698661)).

Consecuencia de diseño para el taller: la métrica primaria es la utilidad downstream (§7), los hechos estilizados son el filtro de fidelidad de dominio, y las métricas genéricas de dos muestras son secundarias.

**Marco de referencia sectorial.** Assefa et al. (ICAIF 2020, J.P. Morgan AI Research) sistematizan oportunidades, retos y trampas de los datos sintéticos en finanzas: silos de datos, restricciones regulatorias, y la dificultad de validar sin una definición de utilidad acordada ([DOI](https://dl.acm.org/doi/10.1145/3383455.3422554), [SSRN 3634235](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3634235)).

---

## 2. Modelos específicos de series temporales

### 2.1 TimeGAN (Yoon, Jarrett y van der Schaar, NeurIPS 2019)

**Idea central.** Combinar el objetivo adversarial no supervisado con una pérdida **supervisada paso a paso** en un espacio latente aprendido. La arquitectura tiene cuatro redes: *embedder* (datos → latente), *recovery* (latente → datos), generador y discriminador; y se optimizan pérdidas de reconstrucción, adversarial y supervisada sobre transiciones $h_t \to h_{t+1}$.

**Problema que resuelve.** Un GAN estándar sobre la secuencia aplanada no impone que la dinámica condicional $p(x_t \mid x_{1:t-1})$ sea correcta; solo que la marginal conjunta se parezca. La pérdida supervisada fuerza explícitamente las transiciones.

**Resultados y disponibilidad.** Evaluado en series sinusoidales, precios bursátiles (Yahoo Finance) y consumo energético (UCI), con dos métricas que se han vuelto estándar de facto: *discriminative score* $|\text{acc} - 0.5|$ de un RNN post-hoc que intenta separar real de sintético, y *predictive score* (error de un RNN entrenado en sintético y evaluado en real). Código oficial en [github.com/jsyoon0823/TimeGAN](https://github.com/jsyoon0823/TimeGAN), TensorFlow 1.x. Paper: [vanderschaar-lab, NeurIPS 2019](https://www.vanderschaar-lab.com/papers/NIPS2019_TGAN_Main.pdf).

**Limitaciones reportadas.** Inestabilidad de entrenamiento y varianza entre ejecuciones son críticas recurrentes en trabajos posteriores que proponen alternativas ([SeriesGAN, arXiv:2410.21203](https://arxiv.org/abs/2410.21203); [PCF-GAN, arXiv:2305.12511](https://arxiv.org/abs/2305.12511), ambos preprints en el momento de consulta). Coste: en un estudio comparativo sobre log-retornos diarios del S&P 500 (2000–2024), TimeGAN requirió **4,5 h de entrenamiento en una NVIDIA RTX 3090**, frente a 1,5 h de un VAE y <0,1 h de ARIMA–GARCH ([Hounwanou, Ntakirutimana y Gaba, 2025, preprint, arXiv:2512.21791](https://arxiv.org/abs/2512.21791)). Ese mismo trabajo describe un **trilema estructural** entre realismo, interpretabilidad y eficiencia computacional. Es el dato más relevante de todo este documento para la §7.

### 2.2 QuantGAN (Wiese, Knobloch, Korn y Kretschmer, 2020)

**Idea central.** Generador y discriminador construidos con **redes convolucionales temporales** (TCN, convoluciones causales dilatadas) en vez de recurrentes. Las dilataciones dan un campo receptivo exponencial en la profundidad, lo que permite capturar dependencias de largo alcance —los clústeres de volatilidad— sin el coste secuencial de un RNN.

**Problema que resuelve.** Sustituir el modelado paramétrico por procesos estocásticos (GARCH, volatilidad estocástica) por un modelo data-driven que igualmente admita transición a la distribución riesgo-neutral, requisito para uso en pricing.

**Resultados.** Los autores reportan concordancia de propiedades distribucionales para lags pequeños y grandes, y reproducción de clústeres de volatilidad, efecto apalancamiento y autocorrelaciones seriales. Publicado en *Quantitative Finance* 20(9):1419–1440 ([DOI](https://www.tandfonline.com/doi/abs/10.1080/14697688.2020.1730426); preprint [arXiv:1907.06673](https://arxiv.org/abs/1907.06673)).

**Lo aprovechable.** La elección arquitectónica —TCN sobre RNN— es transferible y barata: las convoluciones dilatadas paralelizan sobre el eje temporal, lo que en CPU importa mucho más que en GPU.

### 2.3 Fin-GAN (Vuletić et al., 2024)

**Idea central.** Introducir una **función de pérdida guiada por economía** en el generador: además del término adversarial, un término de PnL que premia muestras alineadas con el PnL realizado y un término de desviación típica que penaliza PnL volátil.

**Problema que resuelve.** Que el objetivo de entrenamiento sea la verosimilitud o la indistinguibilidad estadística no garantiza rentabilidad. Fin-GAN optimiza directamente aquello que se va a evaluar.

**Resultados reportados.** Los autores reportan los mayores ratios de Sharpe medio, mediano y de cartera entre los modelos comparados, con Sharpe mediano cercano al doble del siguiente mejor (LSTM), menor varianza del PnL, y resultados competitivos en acciones no vistas en entrenamiento. Publicado en *Quantitative Finance* ([DOI 10.1080/14697688.2023.2299466](https://www.tandfonline.com/doi/full/10.1080/14697688.2023.2299466)).

**Lo aprovechable.** El principio —añadir al objetivo un término alineado con la tarea downstream— es de coste casi nulo y directamente trasladable: en nuestro caso, un término que penalice la mala clasificación de régimen en las muestras generadas.

### 2.4 Métodos basados en firma (signature)

**SigCWGAN / Sig-Wasserstein GAN.** Liao, Ni, Szpruch, Wiese, Sabate-Vidales y Xiao sustituyen el discriminador entrenado por una métrica analítica sobre el espacio de características de la **firma del camino** (rough paths). La firma truncada es un mapa de características universal para caminos; la métrica $\text{Sig-}W_1$ condicional admite fórmula explícita, lo que **elimina la necesidad de entrenar el discriminador** y con ello el cuello de botella computacional y la inestabilidad min-max ([arXiv:2006.05421](https://arxiv.org/abs/2006.05421), 2020; versión revisada en *Mathematical Finance*, [DOI 10.1111/mafi.12423](https://doi.org/10.1111/mafi.12423), [PDF abierto UCL](https://discovery.ucl.ac.uk/id/eprint/10180853/7/Ni_Mathematical%20Finance%20-%202023%20-%20Liao%20-%20Sig%E2%80%90Wasserstein%20GANs%20for%20conditional%20time%20series%20generation.pdf)).

**Generador de mercados con firmas.** Buehler, Horvath, Lyons, Perez Arribas y Wood (2020) usan un **CVAE sobre características de firma** y muestran que supera a la generación basada en retornos, tanto numérica como teóricamente, y que converge con conjuntos de entrenamiento pequeños — el régimen habitual en finanzas ([SSRN 3657366](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3657366)).

**Coste a tener en cuenta.** La dimensión de la firma truncada a profundidad $M$ con $c$ canales es $\sum_{k=1}^{M} c^k$. Con $c=20$ canales y $M=3$ son $20 + 400 + 8000 = 8420$ términos. Sobre 3 o 4 canales seleccionados es perfectamente manejable en CPU; sobre los 20 no lo es.

### 2.5 TimeVAE (Desai, Freeman, Wang y Beaver, 2021, preprint)

**Idea central.** VAE convolucional para series multivariantes con un decodificador que admite **bloques interpretables** —tendencia polinómica y estacionalidad— sumados a un bloque residual libre. Permite inyectar conocimiento de dominio en la estructura del decodificador.

**Problema que resuelve.** El coste y la fragilidad del entrenamiento adversarial. Los autores reportan tiempos de entrenamiento reducidos (hasta un orden de magnitud menos según el conjunto) y robustez bajo escasez de datos, con rendimiento igual o superior a los métodos GAN de referencia en métricas de similitud y predictibilidad ([arXiv:2111.08095](https://arxiv.org/abs/2111.08095)).

**Relevancia.** Es la referencia natural del cVAE del taller. La conclusión operativa —VAE gana a GAN cuando hay pocos datos y poco cómputo— es exactamente nuestra situación.

### 2.6 Difusión sobre series temporales

**Diffusion-TS (Yuan y Qiao, ICLR 2024).** Transformer encoder-decoder con **descomposición tendencia/estacionalidad desacoplada**, entrenado para reconstruir directamente la muestra en lugar del ruido en cada paso, con un término de pérdida en el dominio de Fourier. Se extiende a generación condicional (predicción, imputación) sin cambiar el modelo ([arXiv:2403.01742](https://arxiv.org/abs/2403.01742), [OpenReview](https://openreview.net/forum?id=4h1apFjO99), código en [github.com/Y-debug-sys/Diffusion-TS](https://github.com/Y-debug-sys/Diffusion-TS)).

**CSDI (Tashiro, Song, Song y Ermon, NeurIPS 2021).** Difusión basada en score **condicionada a las observaciones disponibles**, entrenada explícitamente para imputación. Reporta mejoras del 40–65 % sobre métodos probabilísticos de imputación previos y reducción del error del 5–20 % frente a los métodos deterministas del estado del arte; se extiende a interpolación y predicción probabilística ([NeurIPS 2021](https://proceedings.neurips.cc/paper/2021/hash/cfe8504bda37b575c70ee1a8276f3486-Abstract.html), código en [github.com/ermongroup/CSDI](https://github.com/ermongroup/CSDI)).

**CoFinDiff (Tanaka et al., 2025, preprint).** Difusión condicional para series financieras con condiciones derivadas de los precios (tendencia, volatilidad) inyectadas por *cross-attention*. Reporta reproducción de hechos estilizados, satisfacción de las condiciones especificadas, mayor diversidad que las líneas base, y mejora en una tarea de *deep hedging* entrenada con sus datos ([arXiv:2503.04164](https://arxiv.org/abs/2503.04164)).

**Coste.** El muestreo por difusión es el punto crítico: un DDPM con $T=1000$ pasos requiere 1000 pasadas hacia delante por muestra. DDIM (Song, Meng y Ermon, ICLR 2021) generaliza el DDPM a procesos no markovianos con **el mismo objetivo de entrenamiento** pero muestreo determinista en muchos menos pasos, lo que reduce el coste de inferencia en uno o dos órdenes de magnitud ([ICLR 2021](https://iclr.cc/virtual/2021/poster/2804)). Es la razón por la que DDIM, y no DDPM, es el candidato viable del taller.

### 2.7 Escenarios de cola y generadores no neuronales

**Tail-GAN (Cont, Cucuringu, Xu y Zhang).** Explota la propiedad de **elicitabilidad conjunta de VaR y ES** para diseñar la pérdida del GAN, de modo que el generador aprende a preservar el riesgo de cola de una clase de estrategias estáticas y dinámicas. Publicado en *Management Science* ([DOI 10.1287/mnsc.2023.00936](https://pubsonline.informs.org/doi/abs/10.1287/mnsc.2023.00936); preprint [arXiv:2203.01664](https://arxiv.org/abs/2203.01664); código en [github.com/chaozhang-ox/Tail-GAN](https://github.com/chaozhang-ox/Tail-GAN)).

**The Market Generator (Kondratyev y Schwarz, 2019).** Máquina de Boltzmann restringida de Bernoulli sobre datos reales transformados y binarizados, capaz de replicar la estructura de dependencia entre factores de riesgo —incluida su ruptura en eventos de cola— sin ser paramétrica ([SSRN 3384948](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3384948)).

### 2.8 Benchmarks

**TSGBench** (Ang, Huang, Bao, Tung y Huang, *PVLDB* vol. 17, 2023) evalúa diez métodos de generación de series sobre diez conjuntos reales con medidas estadísticas, basadas en distancia, visualización y **generalización por adaptación de dominio**, y publica un análisis estadístico de rankings ([PDF](https://www.vldb.org/pvldb/vol17/p305-huang.pdf)). La conclusión transversal es que el ranking depende fuertemente del conjunto y de la medida: no hay un ganador único.

---

## 3. Modelos tabulares aplicables

Una ventana $60 \times 20$ aplanada es un vector de 1.200 componentes; formalmente, una fila tabular. Los generadores tabulares son por tanto aplicables, con reservas importantes.

**CTGAN y TVAE** (Xu, Skoularidou, Cuesta-Infante y Veeramachaneni, NeurIPS 2019). CTGAN resuelve dos patologías del dato tabular: distribuciones numéricas multimodales y sesgadas, mediante **normalización específica por modo** (mezcla de gaussianas variacional + codificación one-hot del modo), y desequilibrio de categorías, mediante **muestreo condicional y training-by-sampling** que fuerza la aparición de categorías raras. TVAE aplica la misma normalización dentro de un VAE optimizando el ELBO ([arXiv:1907.00503](https://arxiv.org/abs/1907.00503), código en [github.com/sdv-dev/CTGAN](https://github.com/sdv-dev/CTGAN)).

**TabDDPM** (Kotelnikov, Baranchuk, Rubachev y Babenko, ICML 2023). Difusión gaussiana para las columnas continuas y difusión multinomial para las categóricas, en un único proceso. Los autores reportan superioridad sobre las alternativas GAN/VAE, coherente con la ventaja de los modelos de difusión en otros dominios ([código](https://github.com/yandex-research/tab-ddpm), [ficha ICML](https://dblp.org/rec/conf/icml/KotelnikovBRB23.html)).

**GReaT** (Borisov, Seßler, Leemann, Pawelczyk y Kasneci, ICLR 2023). Serializa cada fila a texto (`col1 is v1, col2 is v2, ...`), afina un GPT-2 y muestrea. La propiedad interesante es que permite **condicionar sobre cualquier subconjunto arbitrario de columnas** sin coste adicional, porque el condicionamiento es simplemente el prefijo del prompt ([arXiv:2210.06280](https://arxiv.org/abs/2210.06280), código en [github.com/tabularis-ai/be_great](https://github.com/tabularis-ai/be_great)).

**Las reservas, que son serias.**

1. **Intercambiabilidad de filas.** Los tres asumen filas i.i.d. sin orden. Al aplanar la ventana, el orden temporal solo sobrevive como correlación entre columnas que el modelo debe redescubrir sin ningún sesgo inductivo que lo favorezca. Una convolución causal lo obtiene gratis.
2. **Dimensionalidad.** Estos modelos se validan típicamente sobre decenas de columnas. $d = 1.201$ está fuera del régimen probado, y en CTGAN la normalización por modo ajusta una mezcla por columna: 1.201 mezclas.
3. **Fuga temporal.** Un modelo tabular no distingue "pasado" de "futuro" dentro de la ventana; nada le impide generar dependencias anticausales.

**Uso legítimo.** Como línea base de contraste sobre un vector de **características agregadas** por ventana (media, volatilidad realizada, asimetría, drawdown, pendiente por canal) en lugar de la serie completa. Ahí el supuesto de intercambiabilidad es razonable y la dimensión baja a decenas.

---

## 4. Hechos estilizados: qué debe reproducir un generador financiero

La lista de referencia es la de Cont (2001), *Quantitative Finance* 1(2):223–236, once propiedades comunes a una gran variedad de mercados e instrumentos ([ficha en Taylor & Francis](https://www.tandfonline.com/doi/abs/10.1080/713665670); PDF alojado por Rice University: [stat.rice.edu](https://www.stat.rice.edu/~dobelman/courses/texts/stylized.cont.2001.pdf)).

| # | Hecho | Contraste empírico |
|---|---|---|
| 1 | Ausencia de autocorrelación en retornos | ACF de $r_t$ con bandas $\pm 1.96/\sqrt{T}$; Ljung–Box |
| 2 | Colas pesadas | Estimador de Hill $\hat\alpha$; QQ-plot; curtosis (con cautela) |
| 3 | Asimetría ganancia/pérdida | Asimetría muestral; distancia de Kuiper entre $P^+$ y $P^-$ |
| 4 | Gaussianidad por agregación | Curtosis en función del horizonte de agregación $h$ |
| 5 | Intermitencia | Dispersión de estimadores de volatilidad locales |
| 6 | Clustering de volatilidad | ACF de $r_t^2$; test ARCH-LM; $\hat\alpha+\hat\beta$ de un GARCH(1,1) |
| 7 | Colas pesadas condicionales | Curtosis de los residuos estandarizados post-GARCH |
| 8 | Decaimiento lento de la ACF de $\lvert r_t \rvert$ | Regresión log-log de $\rho_{|r|}(\tau)$ sobre $\tau$ |
| 9 | Efecto apalancamiento | Correlación cruzada $\text{corr}(r_t, \sigma^2_{t+\tau})$ para $\tau>0$ |
| 10 | Correlación volumen/volatilidad | Correlación entre volumen y estimadores de volatilidad |
| 11 | Asimetría en escalas temporales | Predictibilidad asimétrica volatilidad gruesa → fina |

Detalle de los que el taller sí puede contrastar.

**Ausencia de autocorrelación (1).** Cont: *"las autocorrelaciones (lineales) de los retornos son a menudo insignificantes, excepto en escalas intradía muy pequeñas ($\approx$ 20 minutos) donde entran en juego efectos de microestructura"*. En datos diarios se contrasta con la ACF de $r_t$ y bandas de Bartlett. Precisión práctica de Davies y Krämer (2016, preprint): las autocorrelaciones no son estrictamente nulas sino pequeñas — reportan un primer lag de la ACF de los **signos** del S&P 500 de 0,0577, estadísticamente significativo pero sin relevancia práctica ([arXiv:1612.05229](https://arxiv.org/abs/1612.05229)). El criterio de validación debe ser reproducir la magnitud, no exigir cero.

**Colas pesadas (2).** Cont describe una cola tipo ley de potencia con **índice de cola finito, mayor que 2 y menor que 5** en la mayoría de conjuntos estudiados, lo que excluye simultáneamente las leyes estables de varianza infinita ($\alpha<2$) y la normal. Estimación: Hill sobre los $k$ estadísticos de orden superiores,

$$\hat\alpha_H(k) = \left[\frac{1}{k}\sum_{i=1}^{k} \log \frac{X_{(n-i+1)}}{X_{(n-k)}}\right]^{-1}$$

con inspección del *Hill plot* frente a $k$, porque el estimador es sensible a la elección del umbral.

**Advertencia sobre la curtosis.** Davies y Krämer documentan que en su serie del S&P 500 la curtosis cae de **21,51 a 15,37 al eliminar una sola observación** (el retorno de $-0{,}229$). Una métrica de fidelidad basada en igualar la curtosis es, por tanto, casi una métrica de igualar el máximo muestral. Los autores la descartan y usan en su lugar la media de la diferencia entre cuantiles absolutos empíricos normalizados por su mediana y los correspondientes cuantiles normales — estadístico mucho más estable. Para el taller: reportar curtosis, pero **no usarla como criterio de aceptación**; usar un contraste de cuantiles de cola.

**Clustering de volatilidad (6) y decaimiento lento (8).** Son la misma observación medida de dos formas. Cont cuantifica (8) como *"la función de autocorrelación de los retornos absolutos decae lentamente en función del lag, aproximadamente como una ley de potencia con exponente $\beta \in [0{,}2\,,\,0{,}4]$"*. El contraste directo es una regresión

$$\log \rho_{|r|}(\tau) = c - \beta \log \tau + \varepsilon_\tau,\qquad \tau = 1,\dots,100$$

y comparar $\hat\beta$ real frente a sintético. Es el hecho estilizado que más generadores fallan: un modelo que genera cada ventana independientemente y con ruido i.i.d. produce $\rho_{|r|}(\tau)\approx 0$ y $\hat\beta$ indefinido. Davies y Krämer advierten además de que la variabilidad muestral de la ACF de retornos absolutos es grande incluso en series muy largas, lo que obliga a comparar contra bandas de remuestreo y no contra un valor puntual.

**Colas pesadas condicionales (7).** Ajustar un GARCH(1,1) a la serie, estandarizar $z_t = r_t/\hat\sigma_t$ y volver a medir la curtosis de $z_t$. Si el clustering fuera la única fuente de leptocurtosis, $z_t$ sería aproximadamente normal; empíricamente no lo es. Este test separa dos causas que se confunden y es de los más informativos para diagnosticar un generador: si el sintético pasa (2) pero falla (7), está generando colas pesadas por mezcla de escalas y no por saltos condicionales.

**Gaussianidad por agregación (4).** Calcular retornos agregados $r_t^{(h)} = \sum_{i=0}^{h-1} r_{t+i}$ para $h \in \{1,5,21,63\}$ y verificar que la curtosis decrece monótonamente hacia 3. Un generador que produce ruido i.i.d. leptocúrtico también satisface esto por el TCL, así que (4) por sí solo discrimina poco: solo tiene valor combinado con (6) y (8).

**Efecto apalancamiento (9).** Cont: *"la mayoría de medidas de volatilidad de un activo están negativamente correlacionadas con los retornos de ese activo"*. Se contrasta con la correlación cruzada

$$L(\tau) = \frac{\text{corr}\!\left(r_t,\; |r_{t+\tau}|^2\right)}{\text{Var}(r_t)\,},\qquad \tau > 0$$

que debe ser negativa y decaer con $\tau$. Es el hecho que QuantGAN reporta explícitamente reproducir.

**Asimetría ganancia/pérdida (3).** Cont: *"se observan grandes caídas en precios de acciones e índices, pero no movimientos al alza igualmente grandes"*. Contraste: asimetría muestral, y comparación de las distribuciones de retornos positivos y negativos con la distancia de Kuiper (variante bilateral de Kolmogorov–Smirnov). Davies y Krämer reportan valores de Kuiper de 0,0412 (S&P 500) y 0,0290 (DAX) con $p$-valores 0,000 y 0,060 — es decir, el efecto es real pero no enorme, y **no universal**: en un valor individual (Heidelberger Zement) encuentran el signo contrario. Consecuencia: exigir asimetría negativa en cada uno de los 20 canales sería un criterio equivocado; el hecho aplica a índices, no necesariamente a cada activo.

**Los que quedan fuera.** (5) intermitencia y (11) asimetría en escalas temporales requieren alta frecuencia. (10) volumen/volatilidad requiere volumen, que no está en el panel. No se contrastan.

**Batería mínima de validación.** Una implementación compacta del contraste:

```python
import numpy as np
from scipy import stats

def hechos_estilizados(r, max_lag=100):
    """Batería de hechos estilizados sobre una serie de log-retornos 1D.

    Devuelve el vector de estadísticos que se compara real vs sintetico.
    Todos los estadísticos son escalares para permitir bootstrap de bandas.
    """
    r = np.asarray(r, dtype=float)
    r = r - r.mean()

    # (1) autocorrelacion de retornos: primer lag
    ac1 = np.corrcoef(r[:-1], r[1:])[0, 1]

    # (2) cola: cuantiles absolutos normalizados por la mediana,
    #     mas robusto que la curtosis (Davies y Kramer, 2016)
    a = np.abs(r) / np.median(np.abs(r))
    n = len(a)
    q_norm = stats.norm.ppf(np.arange(1, n + 1) / (n + 1))
    q_norm = np.abs(q_norm) / np.median(np.abs(q_norm))
    peso_cola = np.mean(np.sort(a) - np.sort(q_norm))

    # (2b) indice de Hill sobre el 5% superior de |r|.
    #      Sesgado a la baja con este umbral: usar solo comparativamente.
    k = max(int(0.05 * n), 20)
    orden = np.sort(a)[::-1]
    hill = 1.0 / np.mean(np.log(orden[:k] / orden[k]))

    # (8) decaimiento de la ACF de |r|: pendiente log-log
    absr = np.abs(r)
    lags = np.arange(1, max_lag + 1)
    acf = np.array([np.corrcoef(absr[:-l], absr[l:])[0, 1] for l in lags])
    mask = acf > 0                      # el log exige positividad
    beta = -np.polyfit(np.log(lags[mask]), np.log(acf[mask]), 1)[0]

    # (9) apalancamiento: correlacion cruzada r_t con |r_{t+tau}|^2
    apal = np.mean([np.corrcoef(r[:-t], absr[t:] ** 2)[0, 1] for t in (1, 2, 3, 4, 5)])

    # (3) asimetria
    asim = stats.skew(r)

    return dict(ac1=ac1, peso_cola=peso_cola, hill=hill,
                beta_acf_abs=beta, apalancamiento=apal, asimetria=asim)
```

**Calibración sobre procesos conocidos.** Antes de aplicar la batería a un generador conviene fijar los valores de referencia sobre procesos cuya respuesta se conoce. Con $n=20\,000$:

| Proceso | `peso_cola` | `hill` | `beta_acf_abs` | `apalancamiento` |
|---|---|---|---|---|
| Gaussiano i.i.d. | 0,00 | 5,8 | 0,10 | 0,00 |
| $t(4)$ i.i.d. | 0,18 | 3,2 | 0,23 | 0,00 |
| GARCH(1,1) $\alpha=0{,}08$, $\beta=0{,}90$ | 0,06 | 4,9 | 0,66 | 0,01 |

Tres lecturas. Primera: `peso_cola` ordena correctamente las colas y es consistente con los valores que Davies y Krämer reportan para $t(2)$ y $t(3)$ con $n\approx 23\,000$ (0,451 y 0,226) — el 0,18 de $t(4)$ continúa la serie decreciente. Segunda: `beta_acf_abs` es el discriminante decisivo, 0,10 en ruido i.i.d. frente a 0,66 en GARCH; un generador que produzca ventanas independientes con ruido sin memoria caerá cerca de 0,10 y quedará expuesto de inmediato. Tercera: ninguno de los tres procesos reproduce apalancamiento — hace falta asimetría explícita en la dinámica, y ese es exactamente el hecho que un cGAN condicionado solo a la etiqueta de régimen puede no capturar.

El uso correcto no es comparar valores puntuales sino generar bandas por *bootstrap* de bloques sobre la serie real y comprobar si el estadístico del sintético cae dentro. Es el protocolo de $p$-valor por simulación de Davies y Krämer.

---

## 5. Generación condicionada a régimen de mercado

Sí existe literatura específica, y es directamente aplicable al taller.

### 5.1 RSQGAN: cGAN condicionado a régimen

Huang, Khushi y Suleiman (2023) proponen **RSQGAN**, un cGAN semi-supervisado que genera retornos sintéticos condicionados a la clase de régimen. Las etiquetas se obtienen segmentando la serie con un **algoritmo de puntos de ruptura estructurales**. El resultado clave: RSQGAN simula comportamiento consistente con regímenes empíricos concretos y **supera a un GAN incondicional configurado de forma equivalente entrenado solo con datos del régimen de crisis**. Los autores proponen cuatro métricas sensibles a comportamiento dependiente del camino y accionables en entorno de crisis, e incorporan técnicas de GANs de imagen para regular el compromiso fidelidad/variedad ([*Applied Sciences* 13(19):10639, DOI 10.3390/app131910639](https://doi.org/10.3390/app131910639); [texto completo abierto](https://bura.brunel.ac.uk/bitstream/2438/27486/3/FullText.pdf)).

Ese resultado —condicionar gana a filtrar el conjunto de entrenamiento— es el argumento cuantitativo a favor de un cGAN/cVAE frente a entrenar un generador por régimen. Con crisis $\approx$ 16 % de las ventanas de train (10,5 % en test) repartidas en 8 rachas contiguas, un generador entrenado solo con crisis dispone de una fracción mínima de datos; el condicional comparte representación con los otros dos regímenes.

### 5.2 Otras formas de condicionamiento

- **Condicionamiento al pasado.** SigCWGAN modela $p(r_{t+1:t+h} \mid r_{1:t})$ directamente; el "estado" es el camino reciente, no una etiqueta discreta (§2.4).
- **Condicionamiento a atributos continuos.** CoFinDiff condiciona por tendencia y volatilidad vía *cross-attention* (§2.6). Más expresivo que una etiqueta de tres clases y compatible con la variable secundaria $y_{\text{vol}}$ del taller.
- **Mecanismo genérico.** El condicionamiento por etiqueta se remonta a Mirza y Osindero ([arXiv:1411.1784](https://arxiv.org/abs/1411.1784), 2014), alimentando $c$ tanto al generador como al discriminador; y a Sohn et al. (2015) para el CVAE. Es lo que ya cubren los notebooks de clase (`GAN_1_Really_Simple_GAN_MNIST_etiquetas_y_balanceo.ipynb`).

### 5.3 Regime-switching + generativos

La combinación explícita es la de detectar regímenes con un modelo de cambio de estado y usar el estado como condición.

- **HMM gaussianos.** Un HMM jerárquico adaptativo sobre retornos semanales de S&P 500, EURO STOXX 50 y DAX 40 (2000–2024) identifica meta-regímenes de alta incertidumbre y estados dominados por estrés que se alinean con la Crisis Financiera Global, el desplome de la COVID y el ciclo de endurecimiento monetario 2022–2023; los autores reportan mayor verosimilitud dentro de muestra y cobertura de VaR al menos tan buena como un HMM estándar ([*JRFM* 19(1):15](https://www.mdpi.com/1911-8074/19/1/15)). Un HMM multivariante sobre un índice sectorial identifica regímenes de riesgo con volatilidades marcadamente distintas entre estados ([*Risks* 13(7):135](https://www.mdpi.com/2227-9091/13/7/135)).
- **RBM condicionales para detección de régimen** ([arXiv:2512.21823](https://arxiv.org/abs/2512.21823), preprint), en la línea del *market generator* de Kondratyev y Schwarz.

### 5.4 El problema metodológico que hay que declarar

En el taller la etiqueta $y_{\text{reg}}$ se define sobre una **ventana futura de 21 días**. Modelar $p(X \mid y_{\text{reg}})$ significa condicionar a información posterior al final de $X$. Esto es legítimo para aumento de datos —se generan pares $(\tilde X, \tilde y)$ coherentes, que es exactamente lo que necesita el clasificador— pero **invalida cualquier interpretación del generador como simulador de mercado en tiempo real**. La distinción hay que dejarla escrita en la memoria, y la evaluación final debe hacerse sobre datos reales fuera de muestra temporal, nunca sobre sintéticos.

Segunda cautela: modelar la conjunta $p(X, y_{\text{reg}}, y_{\text{vol}})$ reproduce la frecuencia natural de crisis ($\approx$ 16 % en train, 10,5 % en test); modelar la condicional permite fijarla. Son diseños distintos con consecuencias distintas para el balanceo (§6).

---

## 6. Aumento de datos para eventos raros en finanzas

### 6.1 Línea base clásica

**SMOTE** (Chawla, Bowyer, Hall y Kegelmeyer, *JAIR* 16:321–357, 2002) interpola linealmente entre una instancia minoritaria y uno de sus $k$ vecinos más próximos ($k=5$ típico). Es la referencia obligada porque es barata, determinista y sorprendentemente difícil de batir. Sus variantes híbridas —SMOTE-ENN, que combina sobremuestreo con eliminación de ruido por vecinos editados— siguen siendo estándar en riesgo de crédito ([ejemplo aplicado](https://www.sciencedirect.com/science/article/pii/S2666827025000751)).

**Aumento clásico de series temporales.** Wen et al. (IJCAI 2021) sistematizan el catálogo: *jittering* (ruido aditivo), escalado, rotación, *time warping* y *window warping* (remuestreo de un subintervalo manteniendo el resto). Reportan que **combinar varios métodos básicos supera a usar uno solo** en clasificación de series ([arXiv:2002.12478](https://arxiv.org/abs/2002.12478), [IJCAI](https://www.ijcai.org/proceedings/2021/0631.pdf)).

Matiz crítico para datos financieros: el *jittering* con ruido i.i.d. destruye la autocorrelación de $|r_t|$ (hecho 8). Si se usa jitter, el ruido debe tener estructura temporal — por ejemplo ruido escalado por la volatilidad local estimada, o ruido en bloques.

### 6.2 Generativos profundos para clases minoritarias

**Fraude con tarjeta.** Existe una encuesta dedicada a técnicas GAN para aumento de datos en detección de fraude con desequilibrio de clases ([*MAKE* 5(1):19, 2023](https://www.mdpi.com/2504-4990/5/1/19)). Los enfoques híbridos SMOTE-GAN se documentan en [*JRFM* 11(3):110 (2023)](https://www.mdpi.com/2227-7072/11/3/110). Hay resultados con GAN condicional consciente de *embeddings* sobre transacciones fraudulentas ([ScienceDirect, 2025](https://www.sciencedirect.com/science/article/abs/pii/S2214579625000528)) y con GAN potenciado por transformer ([arXiv:2509.19032](https://arxiv.org/abs/2509.19032), preprint). La conclusión mayoritaria reportada es que el sobremuestreo generativo supera a los métodos basados en densidad, pero **la comparabilidad entre estudios es baja**: distintos conjuntos, distintos clasificadores, distintas métricas.

**Escenarios de estrés y cola.** Es el caso de uso más maduro en la industria. Tail-GAN (§2.7) ataca directamente la preservación de VaR/ES. Los generadores condicionales a régimen (§5.1) permiten sobremuestrear el régimen de crisis manteniendo estructura de camino. El informe del CFA Institute sobre datos sintéticos en gestión de inversiones (julio 2025) recoge el estado de adopción sectorial ([PDF](https://rpc.cfainstitute.org/sites/default/files/docs/research-reports/tait_syntheticdataininvestmentmanagement_online.pdf)).

### 6.3 Las trampas metodológicas, que en este contexto son decisivas

1. **Sintetizar antes de partir.** Generar sobre el conjunto completo y después dividir en entrenamiento/test filtra información de test al entrenamiento. El generador debe entrenarse **solo con el fold de entrenamiento**, y en series temporales la partición debe ser temporal, no aleatoria.
2. **Métricas engañosas.** Con crisis al ~16 % en train y al 10,5 % en test, la exactitud y hasta el ROC-AUC son poco informativos. La métrica debe ser PR-AUC, recall de la clase crisis a precisión fijada, o F1 macro.
3. **Tamaño muestral efectivo.** Es el punto más importante y el menos mencionado. Con ventanas solapadas de 60 días, dos ventanas consecutivas comparten 59 días. El número de **episodios de crisis independientes** en un panel S&P 500 de dos décadas es del orden de la decena (2008, 2011, 2015, 2018, 2020, 2022), no de los miles de ventanas etiquetadas como crisis. Un generador entrenado sobre esa clase tiene un riesgo alto de memorizar episodios concretos y presentarlos como diversidad. **Hay que medirlo**, no suponerlo: distancia al vecino más próximo del conjunto de entrenamiento para cada muestra sintética, comparada con la distribución de distancias entre muestras reales de entrenamiento. Si la distribución sintética está desplazada hacia cero, hay memorización.
4. **Utilidad, no realismo.** Es la conclusión de §1: el criterio de aceptación es que el clasificador entrenado con datos aumentados mejore sobre datos reales fuera de muestra.

---

## 7. Qué aplicamos en este taller y qué dejamos fuera (y por qué)

Restricción operativa: **CPU exclusivamente, sin GPU, y 16 días de calendario** para siete generadores más el pipeline de evaluación. Esto no es un detalle, es la restricción dominante de todo el diseño.

### 7.1 Presupuesto de cómputo

El dato de referencia es el de Hounwanou et al. (§2.1): TimeGAN necesitó **4,5 h en una RTX 3090** para log-retornos **univariantes** del S&P 500. Nuestro objeto es $60 \times 20$ multivariante, entre uno y dos órdenes de magnitud más de trabajo por paso. La aceleración típica de una GPU de esa gama frente a una CPU de escritorio en entrenamiento de redes densas y convolucionales pequeñas está en el rango de uno a dos órdenes de magnitud. Multiplicando ambos factores, un TimeGAN completo en nuestro problema es de días de CPU **por semilla** — y la varianza entre ejecuciones documentada en la literatura exige varias semillas. Es inviable, y decirlo es más útil que intentarlo y entregar un resultado con una sola semilla y sin diagnóstico.

### 7.2 Decisiones

| Elemento del estado del arte | Decisión | Razón |
|---|---|---|
| TimeGAN completo (4 redes, 5 pérdidas) | **Fuera** | Coste (§7.1), TF1, varianza entre semillas |
| Métricas de TimeGAN (*discriminative* / *predictive score*) | **Dentro** | Coste marginal; son el estándar de facto |
| QuantGAN como reimplementación | Fuera | Reproducir el paper no cabe en 16 días |
| TCN / convoluciones causales dilatadas | **Dentro** | Barato en CPU, paraleliza en el eje temporal, mejor que RNN |
| Pérdida guiada por tarea (Fin-GAN) | **Dentro**, versión reducida | Añadir un término de clasificación de régimen al generador es casi gratis |
| SigCWGAN | Fuera como generador | Entrenar el generador sigue requiriendo optimización cara |
| Firma truncada como métrica | **Dentro**, condicionado | Solo profundidad 2 sobre 3–4 canales; con 20 canales la dimensión explota (§2.4) |
| TimeVAE | **Dentro** en espíritu | El cVAE convolucional es la misma familia; VAE gana a GAN con pocos datos y poco cómputo |
| Decodificador interpretable de TimeVAE | Fuera | Días de implementación, beneficio incierto en la tarea downstream |
| Diffusion-TS / CSDI | Fuera | Transformer + cientos de pasos de difusión; no cabe en CPU |
| DDIM sobre U-Net 1D pequeña | **Dentro** | 20–50 pasos de muestreo en lugar de 1000; ya cubierto en clase |
| Flow matching | **Dentro** | Entrenamiento sin simulación (regresión de campo vectorial), estable, sin min-max |
| RBIG | **Dentro**, con reducción previa | Sin gradientes ni red; pero la rotación es $O(d^3)$: aplicar tras PCA a $\sim$50–100 componentes |
| Gaussiano multivariante | **Dentro** | Línea base honesta; con $d = 1.201$ la covarianza es singular → usar contracción (Ledoit–Wolf) o estructura factorial |
| Jitter | **Dentro**, con ruido estructurado | Ruido i.i.d. destruye el hecho estilizado 8; escalar por volatilidad local |
| cGAN / cVAE condicionados a régimen | **Dentro** — núcleo del trabajo | Es exactamente el diseño de RSQGAN (§5.1), validado frente a entrenar solo con crisis |
| CTGAN / TVAE / TabDDPM / GReaT | Fuera como generadores primarios | Intercambiabilidad de filas, $d = 1.201$ fuera de régimen probado (§3) |
| Modelos tabulares sobre características agregadas | Opcional, si sobra tiempo | Ahí el supuesto sí se sostiene |
| Tail-GAN (pérdida elicitable VaR/ES) | Fuera | Requiere definir una clase de estrategias; fuera de alcance |
| Calibración riesgo-neutral (QuantGAN) | Fuera | No hay tarea de pricing en el taller |
| Análisis de privacidad / *membership inference* | Fuera, salvo vecino más próximo | El chequeo de memorización sí es obligatorio (§6.3) |

### 7.3 Protocolo de evaluación adoptado

Jerarquía de criterios, en orden de prioridad:

1. **Utilidad downstream (primaria).** PR-AUC y recall de la clase crisis del clasificador de régimen a 21 días, entrenado con real, real+sintético y solo sintético (TSTR), evaluado siempre sobre un test **temporalmente posterior y real**. Es la respuesta a la crítica de Zhang et al. (2026) y al protocolo del taller.
2. **Fidelidad de dominio.** La batería de hechos estilizados de §4 sobre el canal del S&P 500 y sobre al menos tres sectoriales, con bandas por *bootstrap* de bloques. Criterio de aceptación por banda, no por igualdad puntual.
3. **Fidelidad genérica.** *Discriminative score* y *predictive score* de TimeGAN, más MMD y Kolmogorov–Smirnov sobre marginales. Secundarias.
4. **Guardarraíl de memorización.** Distribución de distancias al vecino más próximo del conjunto de entrenamiento (§6.3). Un generador que falle esto queda descartado aunque gane en 1–3.

### 7.4 Lo que no se afirmará

No se afirmará que los generadores reproducen el mercado, que las muestras son utilizables para backtesting de estrategias, ni que el modelo condicional es un simulador prospectivo — por el problema de la etiqueta futura descrito en §5.4. El alcance declarado es: **aumento de datos etiquetados para una tarea de clasificación con clase minoritaria escasa**, evaluado por utilidad.

---

## 8. Referencias

**Hechos estilizados**

- Cont, R. (2001). *Empirical properties of asset returns: stylized facts and statistical issues*. Quantitative Finance 1(2):223–236. https://www.tandfonline.com/doi/abs/10.1080/713665670
- Davies, L. y Krämer, W. (2016). *Stylized Facts and Simulating Long Range Financial Data*. Preprint. https://arxiv.org/abs/1612.05229
- Ratliff-Crain, E., Van Oort, C. M., Bagrow, J., Koehler, M. T. K. y Tivnan, B. F. (2023). *Revisiting Stylized Facts for Modern Stock Markets*. IEEE BigData 2023. https://arxiv.org/abs/2311.07738

**Generadores de series temporales**

- Yoon, J., Jarrett, D. y van der Schaar, M. (2019). *Time-series Generative Adversarial Networks*. NeurIPS 2019. https://www.vanderschaar-lab.com/papers/NIPS2019_TGAN_Main.pdf · Código: https://github.com/jsyoon0823/TimeGAN
- Wiese, M., Knobloch, R., Korn, R. y Kretschmer, P. (2020). *Quant GANs: deep generation of financial time series*. Quantitative Finance 20(9):1419–1440. https://www.tandfonline.com/doi/abs/10.1080/14697688.2020.1730426 · Preprint: https://arxiv.org/abs/1907.06673
- Vuletić, M. et al. (2024). *Fin-GAN: forecasting and classifying financial time series via generative adversarial networks*. Quantitative Finance. https://www.tandfonline.com/doi/full/10.1080/14697688.2023.2299466
- Liao, S., Ni, H., Szpruch, L., Wiese, M., Sabate-Vidales, M. y Xiao, B. (2020/2024). *Sig-Wasserstein GANs for conditional time series generation*. Mathematical Finance. https://doi.org/10.1111/mafi.12423 · Preprint: https://arxiv.org/abs/2006.05421
- Buehler, H., Horvath, B., Lyons, T., Perez Arribas, I. y Wood, B. (2020). *Generating Financial Markets With Signatures*. SSRN 3657366. https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3657366
- Desai, A., Freeman, C., Wang, Z. y Beaver, I. (2021). *TimeVAE: A Variational Auto-Encoder for Multivariate Time Series Generation*. Preprint. https://arxiv.org/abs/2111.08095
- Yuan, X. y Qiao, Y. (2024). *Diffusion-TS: Interpretable Diffusion for General Time Series Generation*. ICLR 2024. https://arxiv.org/abs/2403.01742 · Código: https://github.com/Y-debug-sys/Diffusion-TS
- Tashiro, Y., Song, J., Song, Y. y Ermon, S. (2021). *CSDI: Conditional Score-based Diffusion Models for Probabilistic Time Series Imputation*. NeurIPS 2021. https://proceedings.neurips.cc/paper/2021/hash/cfe8504bda37b575c70ee1a8276f3486-Abstract.html · Código: https://github.com/ermongroup/CSDI
- Cont, R., Cucuringu, M., Xu, R. y Zhang, C. (2025). *Tail-GAN: Learning to Simulate Tail Risk Scenarios*. Management Science. https://pubsonline.informs.org/doi/abs/10.1287/mnsc.2023.00936 · Código: https://github.com/chaozhang-ox/Tail-GAN
- Kondratyev, A. y Schwarz, C. (2019). *The Market Generator*. SSRN 3384948. https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3384948

**Condicionamiento y régimen**

- Huang, A., Khushi, M. y Suleiman, B. (2023). *Regime-Specific Quant Generative Adversarial Network*. Applied Sciences 13(19):10639. https://doi.org/10.3390/app131910639
- Tanaka, Y. et al. (2025). *CoFinDiff: Controllable Financial Diffusion Model for Time Series Generation*. Preprint. https://arxiv.org/abs/2503.04164
- Mirza, M. y Osindero, S. (2014). *Conditional Generative Adversarial Nets*. Preprint. https://arxiv.org/abs/1411.1784
- *Adaptive Hierarchical Hidden Markov Models for Structural Market Change*. JRFM 19(1):15. https://www.mdpi.com/1911-8074/19/1/15
- *Identifying Risk Regimes in a Sectoral Stock Index Through a Multivariate Hidden Markov Framework*. Risks 13(7):135. https://www.mdpi.com/2227-9091/13/7/135

**Modelos tabulares**

- Xu, L., Skoularidou, M., Cuesta-Infante, A. y Veeramachaneni, K. (2019). *Modeling Tabular Data using Conditional GAN*. NeurIPS 2019. https://arxiv.org/abs/1907.00503 · Código: https://github.com/sdv-dev/CTGAN
- Kotelnikov, A., Baranchuk, D., Rubachev, I. y Babenko, A. (2023). *TabDDPM: Modelling Tabular Data with Diffusion Models*. ICML 2023. https://github.com/yandex-research/tab-ddpm
- Borisov, V., Seßler, K., Leemann, T., Pawelczyk, M. y Kasneci, G. (2023). *Language Models are Realistic Tabular Data Generators*. ICLR 2023. https://arxiv.org/abs/2210.06280 · Código: https://github.com/tabularis-ai/be_great

**Métodos base y evaluación**

- Song, J., Meng, C. y Ermon, S. (2021). *Denoising Diffusion Implicit Models*. ICLR 2021. https://iclr.cc/virtual/2021/poster/2804
- Lipman, Y. et al. (2023). *Flow Matching for Generative Modeling*. ICLR 2023. https://iclr.cc/virtual/2023/poster/11309
- Laparra, V., Camps-Valls, G. y Malo, J. (2011). *Iterative Gaussianization: from ICA to Random Rotations* (RBIG). IEEE TNN 22(4). https://arxiv.org/abs/1602.00229
- Chawla, N. V., Bowyer, K. W., Hall, L. O. y Kegelmeyer, W. P. (2002). *SMOTE: Synthetic Minority Over-sampling Technique*. JAIR 16:321–357.
- Wen, Q. et al. (2021). *Time Series Data Augmentation for Deep Learning: A Survey*. IJCAI 2021. https://www.ijcai.org/proceedings/2021/0631.pdf
- Ang, Y., Huang, Q., Bao, Y., Tung, A. K. H. y Huang, Z. (2023). *TSGBench: Time Series Generation Benchmark*. PVLDB vol. 17. https://www.vldb.org/pvldb/vol17/p305-huang.pdf
- Kwon, S. y Lee, Y. (2024). *Can GANs Learn the Stylized Facts of Financial Time Series?* ICAIF '24. https://arxiv.org/abs/2410.09850
- Zhang, F. et al. (2026). *Beyond Visual Realism: Toward Reliable Financial Time Series Generation*. Preprint. https://arxiv.org/abs/2601.12990
- Hounwanou, C. D., Ntakirutimana, P. y Gaba, Y. U. (2025). *Evaluating generative models for synthetic financial data*. Preprint. https://arxiv.org/abs/2512.21791
- Assefa, S. et al. (2020). *Generating synthetic data in finance: opportunities, challenges and pitfalls*. ICAIF 2020. https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3634235
- CFA Institute (2025). *Synthetic Data in Investment Management*. https://rpc.cfainstitute.org/sites/default/files/docs/research-reports/tait_syntheticdataininvestmentmanagement_online.pdf
