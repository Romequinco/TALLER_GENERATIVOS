# Riesgos y límites de entrenar con datos sintéticos

Documento metodológico del Taller B5-T1. Fija el marco crítico bajo el que se interpretan los resultados de los siete generadores (jitter, gaussiano, cGAN, cVAE, RBIG, flow matching, DDIM) sobre la tarea downstream de clasificación de régimen de mercado a 21 días y regresión de volatilidad. Las secciones 6 y 7 son normativas: definen qué se puede y qué no se puede hacer en el código del repositorio.

---

## 1. El sintético no crea información

Un generador se ajusta sobre un conjunto real `D` de tamaño `n`. Toda muestra que emite es función determinista de los parámetros aprendidos de `D` más ruido latente independiente de la distribución objetivo `p`. La desigualdad de procesamiento de datos impone el límite: ninguna transformación de `D` puede aumentar la información mutua entre la muestra y `p`. El sintético redistribuye la información que ya está en `D`; no la amplía.

Formulaciones recientes lo hacen explícito para bucles de síntesis: el sintético mejora un modelo solo cuando el bucle generación-entrenamiento es *abierto en información*, es decir, cuando entra señal externa (un verificador, un entorno, una restricción física, una etiqueta nueva). Si el bucle es cerrado —el generador consume únicamente su propia distribución aprendida— la información relevante para la tarea solo puede decrecer ([arXiv:2605.16379](https://arxiv.org/abs/2605.16379)).

¿Por qué funciona a veces, entonces? Porque el sintético no actúa como fuente de información sino como **regularizador**. Tres mecanismos:

1. **Codificación de invarianzas conocidas a priori.** El jitter gaussiano sobre una ventana de retornos afirma que la etiqueta de régimen no debería cambiar ante una perturbación pequeña. Esa afirmación es conocimiento de dominio inyectado por nosotros, no información extraída de los datos. Si la invarianza es cierta, el clasificador gana; si es falsa (por ejemplo, si la perturbación cruza el umbral de volatilidad que define la clase), inyecta ruido de etiqueta.
2. **Suavizado de la frontera de decisión.** Rellenar el espacio entre observaciones reduce la varianza del estimador: el clasificador ve una superficie más densa y depende menos de la posición exacta de cada muestra.
3. **Reponderación de clases.** Generar más ventanas de crisis modifica la pérdida efectiva. Es equivalente, en primera aproximación, a un `class_weight`, con la diferencia de que además introduce el sesgo del generador.

El balance es una descomposición sesgo-varianza. Sea `p̂_θ` la distribución del generador. Entrenar con muestras de `p̂_θ` introduce un sesgo asintótico proporcional a la discrepancia `d(p̂_θ, p)` y reduce la varianza en un factor que decrece con `n`. La utilidad neta es positiva solo mientras la reducción de varianza domine al sesgo introducido. Como la varianza decae con `n` y el sesgo del generador **no** decae (está acotado por su fidelidad, que a su vez está limitada por el mismo `n`), existe necesariamente un tamaño real a partir del cual el sintético deja de compensar. Es el fenómeno que documenta la sección 3.

Corolario operativo: **toda ganancia atribuida al sintético debe compararse contra los regularizadores baratos que persiguen el mismo objetivo** —weight decay, dropout, early stopping, `class_weight`, reducción de capacidad—. Si un `class_weight` bien calibrado iguala la ganancia de una cGAN entrenada tres horas, el resultado del taller es ese, y es un resultado válido.

---

## 2. Model collapse y modelos autofágicos

### 2.1 Qué dice la literatura

**Shumailov et al., Nature 631:755–759 (2024)** ([enlace](https://www.nature.com/articles/s41586-024-07566-y); preprint previo *The Curse of Recursion*, [arXiv:2305.17493](https://arxiv.org/abs/2305.17493)). Entrenar una generación de modelos sobre las salidas de la anterior produce un proceso degenerativo en dos fases: **colapso temprano**, en el que se pierde progresivamente la información de las colas y los eventos de baja frecuencia se subrepresentan; y **colapso tardío**, en el que la distribución converge a algo de varianza muy reducida y poco parecido al original.

Identifican tres fuentes de error que se acumulan: error de muestreo estadístico (finitud de las muestras de cada generación), error de expresividad funcional (el modelo no puede representar `p` exactamente) y error de aproximación funcional (la optimización no alcanza el óptimo). Lo demuestran sobre GMM, VAE y modelos de lenguaje OPT. Conclusión práctica de los propios autores: hay que reinyectar datos humanos frescos periódicamente y filtrar seriamente el sintético.

**Alemohammad et al., *Self-Consuming Generative Models Go MAD*, ICLR 2024** ([arXiv:2307.01850](https://arxiv.org/abs/2307.01850)). Definen *Model Autophagy Disorder* (MAD), por analogía con la encefalopatía espongiforme bovina, y estudian tres bucles: *fully synthetic* (cada generación se entrena solo con sintético de la anterior), *synthetic augmentation* (cada generación ve el mismo conjunto real fijo más sintético) y *fresh data* (llegan datos reales nuevos). Resultado central: **sin suficiente dato real fresco en cada generación, la calidad (precisión) o la diversidad (recall) decrecen progresivamente**.

Crítico para nosotros: el bucle de *aumentación sintética* con conjunto real **fijo** —el más parecido a lo que hace cualquier taller de augmentación— también degrada, aunque más despacio que el bucle puramente sintético. Observan además que sesgar el muestreo hacia calidad (truncamiento, temperatura baja, cherry-picking de muestras "bonitas") acelera la pérdida de diversidad.

### 2.2 Matizaciones importantes

El resultado no es universal ni condena al sintético.

- **Acumular en lugar de sustituir evita el colapso.** Gerstgrasser et al. ([arXiv:2404.01413](https://arxiv.org/abs/2404.01413)) muestran que si cada generación *acumula* el sintético junto al real original en lugar de reemplazarlo, el error de test queda acotado y no diverge, en modelos de lenguaje, difusión sobre conformaciones moleculares y VAE. El escenario catastrófico de Shumailov asume reemplazo, el supuesto más pesimista.
- **Hay condiciones de estabilidad demostrables.** Bertrand et al., ICLR 2024 ([arXiv:2310.00429](https://arxiv.org/abs/2310.00429)), prueban que el reentrenamiento iterativo es estable si (i) el modelo inicial está suficientemente cerca de la distribución real y (ii) la proporción de dato real en cada iteración es suficientemente alta. La fracción real es un parámetro de control, no un detalle.
- **Pero fracciones mínimas pueden romper el escalado.** Dohmatob y Feng, *Strong Model Collapse*, ICLR 2025 ([arXiv:2410.04840](https://arxiv.org/abs/2410.04840)), demuestran en regresión supervisada con proyecciones aleatorias que *"even the smallest fraction of synthetic data (e.g., as little as 1% of the total training dataset) can still lead to model collapse"*, entendido como que aumentar el conjunto de entrenamiento deja de mejorar el rendimiento. Encuentran además que los modelos más grandes amplifican el colapso por debajo del umbral de interpolación y solo lo mitigan parcialmente por encima.
- **La extrapolación del experimento de Nature ha sido cuestionada.** Borji ([arXiv:2410.12954](https://arxiv.org/abs/2410.12954)) discute los supuestos del montaje y hasta qué punto las conclusiones se transfieren a escenarios realistas de curación de datos.

### 2.3 Aplicabilidad a este taller

Honestidad metodológica: **este taller no ejecuta un bucle autofágico**. Cada generador se ajusta una sola vez sobre datos reales y produce un único lote sintético; ninguno se reentrena sobre su propia salida. El model collapse recursivo de Shumailov no aplica directamente y sería incorrecto invocarlo como riesgo principal.

Lo que sí aplica:

- **El fenómeno de una sola generación** ya está presente: el colapso de modos en cGAN, el suavizado excesivo de cVAE y la pérdida de masa en las colas son la versión de una iteración del mismo mecanismo. La primera generación del bucle MAD ya muestra pérdida de diversidad.
- **El bucle de aumentación con real fijo** de Alemohammad es exactamente nuestra configuración, y es la referencia adecuada.
- **Existe un bucle blando encubierto** si seleccionamos hiperparámetros del generador optimizando la métrica downstream: el clasificador retroalimenta al generador a través de nuestras decisiones. Se mitiga fijando la selección del generador con criterios de fidelidad intrínsecos, separados de la evaluación downstream.

Diagnóstico obligatorio por generador, en una sola generación: cobertura (fracción de ventanas reales con alguna sintética cerca), diversidad efectiva (número de modos, distancia media al vecino sintético más próximo) y comparación de colas (sección 5).

---

## 3. Cuándo ayuda y cuándo perjudica: el papel del tamaño del conjunto real

La presentación del taller muestra el patrón clave: con ~500 muestras reales el sintético mejora sustancialmente el downstream; con ~20.000 deja de aportar o empeora. No es una anomalía del ejemplo: es el comportamiento esperado y tiene respaldo bibliográfico convergente.

### 3.1 Mecanismo

Reformulando la sección 1: el sintético compra reducción de varianza pagando con el sesgo del generador.

- **Pocos datos reales.** La varianza del clasificador domina el error. El sesgo del generador, aunque grande, es menor que la varianza que elimina. Ganancia neta positiva.
- **Muchos datos reales.** La varianza ya es pequeña. El sesgo del generador —que no desaparece añadiendo más muestras sintéticas, porque todas provienen del mismo `p̂_θ`— pasa a dominar. Ganancia neta negativa.
- Con más datos reales el generador también se ajusta mejor y su sesgo baja, pero más despacio de lo que baja la varianza del clasificador: estimar una densidad es estadísticamente más duro que clasificar.

### 3.2 Evidencia

- **He et al., ICLR 2023**, *Is synthetic data from generative models ready for image recognition?* ([openreview](https://openreview.net/pdf?id=nUmCcZ5RKF)). El beneficio del sintético se concentra en escenarios de escasez (zero-shot y few-shot), y la mezcla real+sintético supera a cualquiera de los dos por separado: el real corrige el desajuste de dominio del sintético y el sintético estabiliza el ajuste con pocas muestras reales.
- **Dohmatob y Feng, ICLR 2025** ([arXiv:2410.04840](https://arxiv.org/abs/2410.04840)). Formalización directa del extremo derecho de la curva: en régimen de conjuntos grandes, una fracción sintética incluso pequeña aplana la ley de escalado y añadir datos deja de mejorar. Es exactamente "con 20.000 reales el sintético ya no aporta o perjudica".
- **Escalado de sintético en preentrenamiento** ([arXiv:2510.01631](https://arxiv.org/abs/2510.01631)): rendimientos decrecientes y en algunos casos negativos al aumentar el volumen sintético en modelos pequeños, mientras que los modelos grandes toleran más. Mismo eje capacidad/varianza.
- **Elor y Averbuch-Elor, *To SMOTE, or not to SMOTE?*, 2022** ([arXiv:2201.08528](https://arxiv.org/abs/2201.08528)). Con clasificadores fuertes bien ajustados, SMOTE y variantes rara vez superan al reponderado de clases o al remuestreo trivial en métricas independientes del umbral. Traducción: si el objetivo del sintético es solo compensar el 15,9 % de crisis del train, hay que demostrar que bate a `class_weight`, y medirlo con PR-AUC, no con accuracy.

### 3.3 Un confound que invalida la mitad de los estudios publicados

Añadir sintético aumenta el tamaño del conjunto de entrenamiento y, con un número fijo de épocas, aumenta el número de pasos de optimización. Parte de la "ganancia del sintético" es simplemente **más entrenamiento**. Un trabajo reciente sobre EHR veterinarios ([arXiv:2601.09756](https://arxiv.org/abs/2601.09756)) documenta que las mejoras observadas con augmentación sintética sobre conjuntos reales pequeños se atenúan o desaparecen al fijar el número de actualizaciones del optimizador, indicando que el efecto era de exposición y no de fidelidad del sintético.

**Consecuencia normativa**: toda comparación con y sin sintético debe fijar el presupuesto de optimización (mismo número de pasos, mismo criterio de early stopping, misma malla de hiperparámetros). Si no se puede fijar, se documenta y el resultado se marca como no concluyente.

### 3.4 Proporción óptima real/sintético

No existe un ratio universal. La evidencia apunta a tres regularidades:

1. **La mezcla bate a los extremos.** Real+sintético supera tanto a solo real como a solo sintético, con una curva en U invertida respecto a la fracción sintética.
2. **El óptimo depende de la tarea y del régimen de datos.** Los rangos reportados en tareas tabulares y de texto se sitúan típicamente entre un 20% y un 50% de sintético en tareas específicas de dominio, y toleran fracciones mayores cuando el objetivo es robustez o diversidad. En el extremo derecho de la curva de datos reales, el óptimo tiende a 0%.
3. **El resultado teórico existe pero es específico del montaje.** Bertrand et al. dan condiciones de estabilidad en función de la proporción real, no una receta numérica transferible.

**Decisión**: no se adopta un ratio de la literatura, se **mide** la curva. Rejilla de ratios sintético/real `{0, 0.5, 1, 2, 5}` cruzada con tamaños reales `{250, 500, 1000, 2000, todo}` y con las dos políticas de reparto por clase (`proporcional`, `equilibrado`), tal y como la genera `mezclas.rejilla`. El sintético se **añade** al real, no lo sustituye. El punto de cruce donde el sintético deja de aportar es un resultado central del taller, no ruido experimental. Se reporta como superficie, con intervalos sobre al menos tres semillas.

---

## 4. Memorización, privacidad y cómo detectarla

### 4.1 Los generativos memorizan

- **Carlini et al., USENIX Security 2023**, *Extracting Training Data from Diffusion Models* ([PDF](https://www.usenix.org/system/files/usenixsecurity23-carlini.pdf)). Extraen imágenes de entrenamiento de modelos de difusión mediante generación masiva seguida de un ataque de inferencia de pertenencia sobre la densidad de generación. Los modelos más grandes son más vulnerables; la duplicación en el conjunto de entrenamiento agrava la memorización.
- **Somepalli et al., CVPR 2023**, *Diffusion Art or Digital Forgery?* ([openaccess](https://openaccess.thecvf.com/content/CVPR2023/html/Somepalli_Diffusion_Art_or_Digital_Forgery_Investigating_Data_Replication_in_Diffusion_CVPR_2023_paper.html)). Documentan replicación directa de contenido de entrenamiento en Stable Diffusion y analizan cómo el tamaño del conjunto modula la tasa: **cuanto menor el conjunto, mayor la replicación**.
- **Inferencia de pertenencia.** LOGAN (Hayes et al., PoPETs 2019, [arXiv:1705.07663](https://arxiv.org/abs/1705.07663)) explota el sobreajuste del discriminador de una GAN para decidir si un registro estuvo en el entrenamiento. GAN-Leaks (Chen et al., ACM CCS 2020, [PDF](https://yangzhangalmo.github.io/papers/CCS20-GAN-Leaks.pdf)) construye la taxonomía completa —caja negra total, caja negra parcial, caja blanca, discriminador accesible— y muestra ataques efectivos basados en la distancia al vecino más cercano en el conjunto sintético.

### 4.2 Por qué importa en nuestro montaje

Nuestro régimen es precisamente el peligroso: **3.696 ventanas de train, cada una de 60 × 20 canales —el bloque que ve el generador tiene 1.201 dimensiones— y generadores con capacidad no trivial (cGAN, cVAE, DDIM)**. Los 20 canales no son 20 activos: el universo son 15 tickers, y los canales combinan once retornos con nueve features derivadas de estrés (D3). Es el escenario de pocos datos y alta capacidad donde memorizar es la solución más fácil para el optimizador. Tres consecuencias:

1. **Contaminación de la evaluación.** Si el generador reproduce literalmente ventanas de entrenamiento, el clasificador no aprende estructura de régimen: reconoce fechas memorizadas. La métrica downstream se infla sin que haya generalización.
2. **Riesgo de licencia.** Las series de proveedores comerciales tienen condiciones de redistribución. Un conjunto sintético que reproduce registros reales no es automáticamente un derivado libre.
3. **Riesgo regulatorio en el caso general.** Con datos de cliente (transacciones, posiciones, scoring), el sintético **no es anónimo por construcción**. La literatura sobre datos sintéticos en finanzas lo señala entre las trampas del enfoque (Assefa et al., ICAIF 2020, [PDF](https://www.jpmorgan.com/content/dam/jpm/cib/complex/content/technology/ai-research-publications/pdf-8.pdf)). En este taller trabajamos con precios públicos y el riesgo práctico es nulo; se documenta porque la técnica se defiende habitualmente con el argumento de privacidad, y ese argumento requiere pruebas que aquí no se aportan.

### 4.3 Cómo detectarlo, y qué no vale como prueba

Diagnósticos baratos que sí ejecutamos:

- **Duplicados exactos y cuasi-exactos.** Para cada ventana sintética, correlación máxima y distancia mínima contra el conjunto de entrenamiento. Se marca cualquier caso con correlación > 0,99 o distancia por debajo del percentil 1 de las distancias entre ventanas reales adyacentes.
- **DCR comparado (*distance to closest record*).** Distribución de la distancia de cada muestra sintética al vecino más próximo en train, frente a la análoga contra un holdout real. Si la primera está sistemáticamente por debajo, hay memorización.
- **NNDR (*nearest neighbour distance ratio*).** Ratio entre la distancia al primer y al segundo vecino real. Valores cercanos a 1 indican que la muestra sintética no imita a ningún registro concreto.

Advertencia necesaria: **pasar el test DCR no demuestra privacidad**. Yao, Krčo, Ganev y de Montjoye, *The DCR Delusion* ([arXiv:2505.01524](https://arxiv.org/abs/2505.01524)), muestran que conjuntos sintéticos que superan los tests basados en DCR siguen siendo vulnerables a ataques de inferencia de pertenencia, con Baynet, CTGAN y modelos de difusión; concluyen que las métricas de distancia son poco informativas sobre el riesgo real y que el estándar riguroso es ejecutar un MIA.

**Postura del repo**: DCR y NNDR se usan como *sanity check de memorización*, útiles para detectar copia burda y para interpretar métricas downstream sospechosamente buenas. No se formula ninguna afirmación de privacidad ni de anonimización en la entrega.

---

## 5. Amplificación de sesgos: el caso de los eventos raros

Sección específica de nuestro problema, y donde está el riesgo real del taller.

### 5.1 El mecanismo general

Un generador ajustado sobre datos sesgados reproduce el sesgo; si hay realimentación, lo amplifica. Wyllie, Shumailov y Papernot, FAccT 2024, *Fairness Feedback Loops* ([arXiv:2403.07857](https://arxiv.org/abs/2403.07857)), formalizan los *model-induced distribution shifts* (MIDS): cuando un modelo induce un desplazamiento de la distribución, codifica sus errores, sesgos e injusticias en la verdad de referencia del ecosistema de datos. A lo largo de las generaciones observan pérdida de rendimiento, de equidad y **de representación de los grupos minorizados**, incluso partiendo de conjuntos inicialmente no sesgados. Sustitúyase "grupo minorizado" por "régimen de crisis" y se tiene el diagnóstico de este taller.

### 5.2 El tamaño muestral efectivo de "crisis" es mucho menor de lo que parece

La clase crisis representa el **15,9 % de las ventanas de train: 587 sobre 3.696**. Ese número es engañoso. Con ventanas de 60 días y desplazamiento de 1 día, dos ventanas consecutivas comparten 59 de 60 días. Un episodio de tres meses genera del orden de 60–90 ventanas **casi idénticas**, no 60–90 observaciones independientes.

Lo que hay que contar son las **rachas contiguas** de ventanas de crisis, que es lo que mide `regimenes.tramos_contiguos`. Medido sobre este panel:

| Partición | Ventanas de crisis | Rachas contiguas |
|---|---|---|
| Train (hasta 2018-12-31) | 587 de 3.696 (15,9 %) | **8** |
| Validación (hasta 2021-12-31) | 22,0 % de 672 | **2** |
| Test (2022 en adelante) | 110 de 1.052 (10,5 %) | **3** |

El panel arranca en **2003**, gobernado por los ETF de renta fija del universo (D5), de modo que los episodios de 1998 (Rusia/LTCM) y de 2000–2002 (tecnológicas) **no están en la muestra** y no se pueden invocar como respaldo. Los que sí están son 2008 (crisis financiera global), 2011 (deuda soberana europea), 2020 (COVID), 2022 (inflación y subidas de tipos) y 2023 (SVB). **El tamaño muestral efectivo de la clase crisis es de 13 rachas en todo el panel, y de 3 en el conjunto de test, no de varios cientos de ventanas.**

Cualquier intervalo de confianza, prueba de significación o afirmación sobre la clase crisis debe construirse sobre ese número. Es lo que hace `evaluacion.banda_bloques`, con un bootstrap de bloques circular de 81 ventanas —la huella de una ventana—: sobre la línea base de persistencia en test, un `recall_crisis` de 0,800 tiene un IC del 95 % de **[0,560 – 1,000]**, **3,0 veces más ancho** que el **[0,716 – 0,864]** que produciría el intervalo binomial de Wilson tratando las 110 ventanas como independientes. La consecuencia operativa es que **ninguna comparación de `recall_crisis` entre dos recetas del barrido que difiera en menos de unos 20 puntos es distinguible del ruido**. Reportar la banda estrecha es un error estadístico grave, y es el tipo de cosa que un tribunal detecta.

### 5.3 Los episodios no son intercambiables

Las crisis difieren en mecanismo, velocidad, magnitud y estructura de correlación. Tres casos canónicos, de los cuales solo los dos últimos están dentro del panel 2003-2026; el primero se cita como ilustración del rango de mecanismos posibles, no como evidencia disponible para el generador:

- **1929–1932** (fuera de la muestra): deflación de deuda, quiebras bancarias en cadena, contracción monetaria. Caída acumulada del orden del 89% en el Dow a lo largo de casi tres años, con dinámica lenta y múltiples rebotes fallidos.
- **2007–2009**: crisis de crédito y apalancamiento originada en titulizaciones. Caída del orden del 57% en el S&P 500 entre el máximo de octubre de 2007 y el mínimo de marzo de 2009, unos 17 meses, con deterioro progresivo y episodios agudos discretos.
- **Febrero–marzo 2020**: shock exógeno sanitario. Caída en torno al 34% en 23 sesiones y recuperación en V impulsada por respuesta fiscal y monetaria masiva. Velocidad sin precedentes y reversión igual de rápida.

La estructura de correlación entre activos, el comportamiento de tipos, crédito y dólar, el nivel y la persistencia de la volatilidad y la asimetría del retorno son distintos en los tres casos. **No son realizaciones de la misma distribución condicionada.**

### 5.4 Qué ocurre al generar sintéticos de "crisis" a partir de ese conjunto

**(a) Interpolación entre mecanismos incompatibles.** Una cGAN o un cVAE condicionados a `clase = crisis` aprenden una única distribución condicional. Con episodios heterogéneos, el modo aprendido es una mezcla o un promedio. Una ventana sintética con la velocidad de 2020 y la estructura de correlación de 2008 **no corresponde a ningún régimen que haya existido**. No es augmentación: es fabricación de un modo inexistente, y el clasificador aprende a reconocer un artefacto.
*Diagnóstico*: proyectar las ventanas sintéticas de crisis sobre el espacio de las reales (PCA o UMAP ajustados solo con train) y verificar si caen sobre los clusters de episodios reales o en el espacio vacío entre ellos.

**(b) Colapso hacia el episodio dominante.** Si un episodio aporta más ventanas que los demás —típicamente 2008 por duración, o 2020 por magnitud— el generador concentra masa ahí y la diversidad intra-clase se pierde. Es el colapso temprano de Shumailov et al. en su forma de una sola generación.
*Diagnóstico*: asignar cada ventana sintética de crisis a su episodio real más próximo y comprobar que la distribución no está concentrada. Si el 80% del sintético se parece a 2008, el generador no ha aprendido "crisis": ha aprendido "2008".

**(c) Subestimación sistemática de las colas.** La patología más peligrosa aquí. Los modelos generativos concentran masa en las regiones de alta densidad y producen muestras típicas; la literatura sobre extremos documenta que **subestiman los eventos más extremos** y generan menos muestras en los percentiles más altos de cada marginal ([arXiv:2311.18521](https://arxiv.org/abs/2311.18521); [arXiv:2506.06380](https://arxiv.org/abs/2506.06380)). Hay además un límite estructural: Wiese et al., *Quant GANs*, Quantitative Finance 20(9), 2020 ([arXiv:1907.06673](https://arxiv.org/abs/1907.06673)), establecen una caracterización `L^p` según la cual, con un proceso latente gaussiano i.i.d. y una red con momentos finitos, la ley generada hereda momentos finitos. No se pueden manufacturar colas más pesadas de lo que la arquitectura y el latente permiten.

Consecuencia: **el sintético de "crisis" será más suave que la crisis real**. Un clasificador entrenado con él subestimará la severidad, justo en el caso de uso donde el coste del error es máximo. Esto invierte el sentido de la mejora aparente: una subida del F1 agregado puede convivir con un empeoramiento en las ventanas más extremas.
*Diagnóstico obligatorio*, real contra sintético: cuantiles 0,1%, 1%, 99% y 99,9% de los retornos; distribución del máximo drawdown por ventana; estimador de Hill del índice de cola; ACF de los retornos absolutos (clustering de volatilidad); efecto apalancamiento.

**(d) Sesgo de muestra macro.** Una muestra que empieza en los años 90 codifica un régimen concreto: desinflación, tipos a la baja, respuesta agresiva de los bancos centrales ante caídas. El generador lo aprende y lo reproduce: las crisis sintéticas heredarán recuperaciones rápidas porque las de la muestra las tuvieron. No es un defecto del generador, es un límite de lo que el dato contiene, y debe declararse en el alcance de las conclusiones.

**(e) El rebalanceo puede no aportar nada.** Antes de atribuir cualquier mejora al realismo del sintético hay que descartar que sea puro efecto de reponderación de clases, que se consigue gratis (Elor y Averbuch-Elor, 2022). El baseline con `class_weight` y ajuste de umbral es obligatorio, no opcional.

### 5.5 Protocolo de evaluación que se deriva

- **Leave-one-crisis-out.** Entrenar excluyendo por completo un episodio (2008, 2020, 2022) y evaluar sobre él. Es la única prueba honesta de si el sintético ayuda a generalizar a una crisis no vista. Si solo ayuda cuando el episodio de test está representado en train, está interpolando dentro del episodio, no generalizando entre episodios.
- **Métricas por episodio**, no solo agregadas.
- **PR-AUC y recall a precisión fijada** para la clase crisis; nunca accuracy sola con una prevalencia del 10,5 % en test.
- **Intervalos de confianza sobre el número de episodios**, no de ventanas.
- **Declaración de alcance**: no se afirma que el sintético "cubre escenarios de crisis"; como mucho, que cubre la vecindad de los episodios observados.

---

## 6. Fuga de información en el diseño experimental

### 6.1 Definición y taxonomía

Fuga es *"la introducción de información sobre el objetivo del análisis que no debería estar legítimamente disponible para modelar"* (Kaufman, Rosset, Perlich y Stitelman, ACM TKDD 6(4), 2012, [DOI](https://dl.acm.org/doi/10.1145/2382577.2382579)). Kapoor y Narayanan, *Patterns* 4(9), 2023 ([enlace](https://www.cell.com/patterns/fulltext/S2666-3899(23)00159-9)), catalogan ocho tipos y documentan 294 artículos afectados en 17 campos, muchos con conclusiones gravemente sobreoptimistas. Los tipos que nos afectan son tres: **preprocesado sobre el conjunto completo**, **solapamiento entre train y test** y **contaminación temporal**.

### 6.2 La aritmética del solapamiento en nuestro montaje

Configuración: ventana de entrada `L = 60` sesiones, horizonte de etiqueta `H = 21` sesiones, desplazamiento de 1 sesión.

- La muestra en el índice `i` usa como entrada `[i-60, i)` y como etiqueta `[i, i+21)`.
- Su **huella informativa** es `[i-60, i+21)`: **81 sesiones**.
- Dos muestras `i` y `j` **no comparten ningún dato bruto** si y solo si `|i - j| ≥ 81`.
- Dos ventanas consecutivas comparten **59 de 60 días de entrada: el 98,3%**.

Aplíquese ahora un split aleatorio con `test_size = 0.1`. Para una muestra de test en `i`, la probabilidad de que su vecino `i-1` caiga en train es 0,9, y lo mismo para `i+1`. La probabilidad de que **al menos un vecino inmediato esté en train** es `1 - 0,1² = 0,99`. Es decir: **alrededor del 99% del conjunto de test tiene un cuasi-duplicado en el conjunto de entrenamiento**.

Ese test no mide generalización fuera de muestra: mide la capacidad de interpolar entre ventanas adyacentes, tarea trivial cuando comparten el 98% de su contenido. El error reportado puede ser arbitrariamente optimista y no acota el error real en producción.

A esto se suma la fuga por etiqueta: la etiqueta de la muestra `i` se calcula sobre sesiones futuras `[i, i+21)`, que son entradas de las muestras `i+1 … i+81`. Si unas están en train y otras en test, el modelo ve en entrenamiento los datos con los que se construyó la etiqueta de test. Es la fuga que motiva el *purging* de López de Prado.

### 6.3 El caso concreto de los notebooks de clase

Se documenta aquí porque justifica técnicamente que el repositorio se aparte del notebook guiado.

Los notebooks `docs/material_clase/notebooks/Taller_Gaussian_solution.ipynb` y `docs/material_clase/notebooks/Taller_GANs.ipynb` construyen el dataset en la celda 13 con desplazamiento de 1 sesión sobre `returns`, `window_size_X = 60` y `window_size_Y = 30` (media de los 30 días siguientes), y a continuación, en la celda 14:

```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X_train_aux, X_test, Y_train_aux, Y_test = train_test_split(X, Y, test_size=0.1, random_state=42)
X_train, X_val, Y_train, Y_val = train_test_split(X_train_aux, Y_train_aux, test_size=0.1, random_state=42)
```

`train_test_split` baraja por defecto (`shuffle=True`). Aplicado sobre ventanas solapadas de una serie temporal produce la situación descrita en 6.2: cuasi-duplicados repartidos entre train y test, y solapamiento entre el horizonte de etiqueta de unas muestras y la entrada de otras. Con `window_size_Y = 30` la huella informativa es de 90 sesiones y el problema es aún mayor. El segundo split reproduce el patrón entre train y validación, de modo que la validación tampoco es independiente y el early stopping se calibra sobre datos contaminados.

Precisión sobre el escalado: en estos dos notebooks `StandardScaler` se importa pero no llega a usarse, de modo que no hay fuga por normalización en ellos. El riesgo es genérico y se evita en nuestro repositorio de todos modos.

Contexto y postura: **estos notebooks son material didáctico** cuyo objetivo es exponer la mecánica de los generadores, no el protocolo de validación; el split no es el objeto de la lección y `train_test_split` es la forma más corta de llegar a la parte que se está enseñando. La situación cambia en una entrega evaluada que reporta métricas downstream como resultado: ahí el diseño del split determina si los números significan algo. Por eso este repositorio adopta un esquema temporal con purga y embargo, y lo documenta. Se reporta además la comparación explícita entre ambos protocolos (salvaguarda S16, §7): la diferencia de métricas **es** la magnitud de la fuga, y constituye por sí misma un resultado interesante para la defensa.

### 6.4 La alternativa correcta

**Split cronológico, sin barajar.** Train = periodo más antiguo, validación = intermedio, test = más reciente. Es el orden en que la información está disponible en la realidad.

**Purga.** Eliminar del entrenamiento toda muestra cuya huella `[i-L, i+H)` solape con la huella de cualquier muestra de test. Con un split cronológico simple equivale a recortar la cola del train.

**Embargo.** Hueco adicional entre el final del train y el inicio del test, para absorber la dependencia serial que el solapamiento estricto no captura: persistencia de la volatilidad, reacción retardada del mercado, efectos de calendario (López de Prado, *Advances in Financial Machine Learning*, Wiley, 2018, cap. 7; resumen del método en [purged cross-validation](https://en.wikipedia.org/wiki/Purged_cross-validation)).

**Hueco mínimo**: la huella de una ventana mide `L + H = 60 + 21 = 81` sesiones, así que dos ventanas dejan de compartir dato bruto cuando distan 81. Hay que distinguir dos números que no son el mismo (D7): el **hueco** exigido entre particiones es de **81 sesiones** —es lo que publica `ventanas.auditar_solape` como `minimo_exigido`, `solape_maximo + 1`—, y el **embargo**, es decir el número de ventanas que hay que **descartar** para abrirlo, es de **80**, porque descartar E ventanas deja un hueco de E+1 sesiones. Se adopta un embargo de **85 sesiones de mercado** (unos cuatro meses) para incluir margen. El embargo se cuenta en sesiones, nunca en días naturales: contarlo en días naturales fue un fallo real y medido, porque 85 días naturales son 59 sesiones y dejaban 22 sesiones simultáneamente en dos particiones (D7).

**Si se hace validación cruzada**: *purged K-fold* o *combinatorial purged CV* con purga y embargo en cada frontera, o *walk-forward* con ventana expansiva. Nunca `KFold` estándar ni `train_test_split` barajado.

### 6.5 Checklist operativo anti-fuga

Cada punto es verificable en el código.

1. **Split temporal, nunca aleatorio.** Prohibido `train_test_split` con `shuffle=True` y `KFold` estándar sobre el dataset de ventanas. El orden cronológico se respeta en train, val y test.
2. **Purga y embargo de al menos 80 sesiones** entre bloques contiguos —el mínimo que neutraliza una huella de `L + H = 81`—; valor adoptado, **85 sesiones de mercado**. Se aplica en cada frontera train/val y val/test.
3. **El generador se ajusta solo con train.** Ninguno de los siete ve una sola observación de validación o de test, ni directamente ni a través de estadísticos agregados.
4. **El etiquetado de regímenes se ajusta solo con train.** Si las clases se definen por cuantiles de volatilidad o retorno futuro, esos cuantiles se calculan sobre train y se **aplican congelados** a val y test. Calcularlos sobre el dataset completo es fuga de etiqueta.
5. **Escalado y toda transformación con parámetros, ajustados solo con train.** Media, desviación, winsorización, PCA, selección de variables: `fit` en train, `transform` en el resto. Nunca `fit_transform` sobre `X` completo.
6. **El conjunto de test es siempre real y nunca contiene sintético.** Tampoco la validación: validar sobre sintético mide fidelidad al generador, no rendimiento.
7. **Ninguna muestra sintética puede derivar de una observación cuya huella toque el test.** Se verifica por índices temporales, no por confianza.
8. **Hiperparámetros y early stopping se seleccionan en validación**, jamás en test. El test se toca una sola vez, al final, y ese hecho se registra.
9. **Presupuesto de optimización fijado** entre condiciones con y sin sintético (mismo número de pasos, mismo criterio de parada), para que la comparación no mida simplemente más entrenamiento.
10. **Controles de detección activos**: prueba de etiquetas permutadas (si el modelo bate al azar con etiquetas barajadas, hay fuga); comparación split aleatorio contra split temporal; alerta automática si el error de test cae por debajo del de validación.

### 6.6 Trampas adicionales propias de datos financieros

- **Sesgo de supervivencia** en la composición del índice: usar la composición actual sobre histórico introduce información futura.
- **Datos no *point-in-time***: series macro revisadas (PIB, IPC) contienen valores que no estaban disponibles en la fecha. Usar *vintages* o prescindir de ellas.
- **Solapamiento entrada/etiqueta del mismo día**: si la etiqueta usa el cierre del día `i` y la entrada también, la etiqueta es parcialmente observable.
- **Redundancia por desplazamiento 1**: con `L = 60` el número de muestras independientes es aproximadamente `N / 81`, no `N`. Considerar desplazamiento mayor y, en todo caso, reportar el tamaño muestral efectivo junto al nominal.

---

## 7. Salvaguardas adoptadas en este taller

Decisiones concretas del repositorio, cada una trazable a la sección que la motiva.

**Sobre la numeración.** Estas salvaguardas se numeran **S1...S20**, en un espacio propio. Una versión anterior las llamaba D1...D20 y colisionaba con el registro de decisiones de `docs/DECISIONES.md`, donde esos mismos identificadores significan otra cosa —allí D1 es la elección del problema y D16 el test de memorización—, de modo que una cita como "(D16)" no se podía resolver sin adivinar a cuál de los dos documentos apuntaba. La última columna traza cada salvaguarda a la decisión de `DECISIONES.md` que la implementa; un guion significa que la salvaguarda es normativa aquí pero **no** tiene todavía una decisión formal que la respalde, y eso es información, no un hueco de maquetación.

| Id | Salvaguarda | Motivación | Decisión en `DECISIONES.md` |
|----|----------|------------|------|
| S1 | Split **cronológico** train/val/test, sin barajar, con purga y embargo de **85 sesiones de mercado** (mínimo 80, huella `L + H = 81`) en cada frontera. | §6.2, §6.4 | D7 |
| S2 | Umbrales del etiquetado de regímenes (3 clases) calculados **solo con train** y aplicados congelados a val y test. | §6.5.4 | D6 |
| S3 | Escalador y cualquier transformación con parámetros: `fit` en train, `transform` en el resto. Prohibido `fit_transform` sobre el dataset completo. | §6.5.5 | — (`catalog.yaml`, bloque `escalado`) |
| S4 | Los siete generadores se ajustan **exclusivamente con la partición de train**. Verificación por índices temporales. | §6.5.3, §6.5.7 | — |
| S5 | **Test y validación son 100% reales.** El sintético solo entra en train. | §6.5.6 | D7 |
| S6 | Presupuesto de optimización fijado entre condiciones con y sin sintético; si no es posible, el resultado se marca como no concluyente. | §3.3 | D20 (y D22) |
| S7 | Curva de utilidad sobre tamaños reales `{250, 500, 1000, 2000, todos}`. El punto de cruce se reporta como resultado principal. | §3.1, §3.2 | D13 |
| S8 | Rejilla de ratios sintético/real `{0, 0.5, 1, 2, 5}`, cruzada con las dos políticas de reparto por clase. No se adopta ningún ratio de la literatura sin medirlo. | §3.4 | D13, D14 |
| S9 | Baselines obligatorios antes de atribuir mérito al sintético: sin augmentación, `class_weight`, ajuste de umbral y jitter (la augmentación más barata). | §1, §3.2 | D15, D12 |
| S10 | Métricas de clase rara: PR-AUC, recall a precisión fija, matriz de confusión completa. Accuracy nunca en solitario. | §5.5 | D17 |
| S11 | Evaluación **leave-one-crisis-out** sobre 2008, 2020 y 2022, con métricas por episodio. | §5.5 | — |
| S12 | Intervalos de confianza calculados sobre el número de **rachas** (8 en train, 2 en validación, 3 en test), no de ventanas. Se reporta el tamaño muestral efectivo junto al nominal. | §5.2 | — (`evaluacion.banda_bloques`) |
| S13 | Diagnóstico de colas real contra sintético por generador: cuantiles 0,1/1/99/99,9%, distribución del máximo drawdown, estimador de Hill, ACF de retornos absolutos, efecto apalancamiento. | §5.4c | D10 |
| S14 | Diagnóstico de diversidad y colapso de modos: asignación de cada ventana sintética de crisis a su episodio real más próximo; proyección PCA/UMAP (ajustada solo con train) para detectar modos fabricados entre clusters. | §5.4a, §5.4b, §2.3 | — |
| S15 | Diagnóstico de memorización: duplicados exactos, correlación máxima contra train, DCR y NNDR. Declarado como sanity check, **no** como garantía de privacidad. | §4.3 | D16 |
| S16 | Control de etiquetas permutadas y comparación explícita **split aleatorio contra split temporal**; la diferencia se reporta como medida de la fuga. | §6.5.10, §6.3 | D7 |
| S17 | Semillas fijas y al menos **3 repeticiones** por configuración; se reportan medias e intervalos, no la mejor ejecución. | §3.4 | — (`catalog.yaml`, `semilla_global`) |
| S18 | **Ningún reentrenamiento del generador sobre su propia salida.** Una sola generación, documentada como frontera deliberada del alcance. | §2.3 | — |
| S19 | Los hiperparámetros del generador se seleccionan con criterios de fidelidad intrínsecos, separados de la evaluación downstream, para no crear un bucle de realimentación blando. | §2.3 | — |
| S20 | Declaraciones de alcance en la entrega: no se afirma privacidad, no se afirma cobertura de "escenarios de crisis", y las conclusiones se acotan al periodo muestral y a su régimen macro. | §4.2, §5.4d, §5.5 | D1 |

Nota sobre el alcance del riesgo de colapso: al ejecutarse una única generación (S18), el model collapse recursivo descrito en §2.1 no aplica a este trabajo. Lo que sí se mide es su manifestación de una sola iteración —pérdida de diversidad y de masa en las colas— mediante S13 y S14. Invocar el colapso recursivo como riesgo principal de este montaje sería incorrecto y se evita deliberadamente.

---

## 8. Referencias

**Colapso de modelos y bucles autofágicos**

- Shumailov, I., Shumaylov, Z., Zhao, Y., Papernot, N., Anderson, R., Gal, Y. (2024). *AI models collapse when trained on recursively generated data*. Nature 631, 755–759. https://www.nature.com/articles/s41586-024-07566-y
- Shumailov, I. et al. (2023). *The Curse of Recursion: Training on Generated Data Makes Models Forget*. arXiv:2305.17493. https://arxiv.org/abs/2305.17493
- Alemohammad, S., Casco-Rodriguez, J., Luzi, L., Humayun, A. I., Babaei, H., LeJeune, D., Siahkoohi, A., Baraniuk, R. G. (2023/2024). *Self-Consuming Generative Models Go MAD*. ICLR 2024. arXiv:2307.01850. https://arxiv.org/abs/2307.01850
- Gerstgrasser, M., Schaeffer, R. et al. (2024). *Is Model Collapse Inevitable? Breaking the Curse of Recursion by Accumulating Real and Synthetic Data*. arXiv:2404.01413. https://arxiv.org/abs/2404.01413
- Bertrand, Q., Bose, A. J., Duplessis, A., Jiralerspong, M., Gidel, G. (2024). *On the Stability of Iterative Retraining of Generative Models on their own Data*. ICLR 2024. arXiv:2310.00429. https://arxiv.org/abs/2310.00429
- Dohmatob, E., Feng, Y. et al. (2024/2025). *Strong Model Collapse*. ICLR 2025. arXiv:2410.04840. https://arxiv.org/abs/2410.04840
- Borji, A. (2024). *A Note on Shumailov et al. (2024): 'AI Models Collapse When Trained on Recursively Generated Data'*. arXiv:2410.12954. https://arxiv.org/abs/2410.12954

**Cuándo ayuda el sintético, proporciones y límites de información**

- He, R., Sun, S., Yu, X., Xue, C., Zhang, W., Torr, P., Bai, S., Qi, X. (2023). *Is synthetic data from generative models ready for image recognition?* ICLR 2023. https://openreview.net/pdf?id=nUmCcZ5RKF
- Elor, Y., Averbuch-Elor, H. (2022). *To SMOTE, or not to SMOTE?* arXiv:2201.08528. https://arxiv.org/abs/2201.08528
- *Demystifying Synthetic Data in LLM Pre-training: A Systematic Study of Scaling Laws, Benefits, and Pitfalls* (2025). arXiv:2510.01631. https://arxiv.org/abs/2510.01631
- *An Information-Theoretic Criterion for Efficient Data Synthesis* (2026). arXiv:2605.16379. https://arxiv.org/abs/2605.16379
- *Synthetic Data for Veterinary EHR De-identification: Benefits, Limits, and Safety Trade-offs Under Fixed Compute* (2026). arXiv:2601.09756. https://arxiv.org/abs/2601.09756

**Memorización y privacidad**

- Carlini, N., Hayes, J., Nasr, M., Jagielski, M., Sehwag, V., Tramèr, F., Balle, B., Ippolito, D., Wallace, E. (2023). *Extracting Training Data from Diffusion Models*. USENIX Security 23. https://www.usenix.org/system/files/usenixsecurity23-carlini.pdf
- Somepalli, G., Singla, V., Goldblum, M., Geiping, J., Goldstein, T. (2023). *Diffusion Art or Digital Forgery? Investigating Data Replication in Diffusion Models*. CVPR 2023, 6048–6058. https://openaccess.thecvf.com/content/CVPR2023/html/Somepalli_Diffusion_Art_or_Digital_Forgery_Investigating_Data_Replication_in_Diffusion_CVPR_2023_paper.html
- Hayes, J., Melis, L., Danezis, G., De Cristofaro, E. (2019). *LOGAN: Membership Inference Attacks Against Generative Models*. PoPETs 2019(1). arXiv:1705.07663. https://arxiv.org/abs/1705.07663
- Chen, D., Yu, N., Zhang, Y., Fritz, M. (2020). *GAN-Leaks: A Taxonomy of Membership Inference Attacks against Generative Models*. ACM CCS 2020. https://yangzhangalmo.github.io/papers/CCS20-GAN-Leaks.pdf
- Yao, Z., Krčo, N., Ganev, G., de Montjoye, Y.-A. (2025). *The DCR Delusion: Measuring the Privacy Risk of Synthetic Data*. arXiv:2505.01524. https://arxiv.org/abs/2505.01524

**Sesgos, eventos raros y colas**

- Wyllie, S., Shumailov, I., Papernot, N. (2024). *Fairness Feedback Loops: Training on Synthetic Data Amplifies Bias*. ACM FAccT 2024. arXiv:2403.07857. https://arxiv.org/abs/2403.07857
- Gu, J., Zhang, X., Wang, G. (2025). *Beyond the Norm: A Survey of Synthetic Data Generation for Rare Events*. arXiv:2506.06380. https://arxiv.org/abs/2506.06380
- *Combining deep generative models with extreme value theory for synthetic hazard simulation: a multivariate and spatially coherent approach* (2023). arXiv:2311.18521. https://arxiv.org/abs/2311.18521
- Wiese, M., Knobloch, R., Korn, R., Kretschmer, P. (2020). *Quant GANs: deep generation of financial time series*. Quantitative Finance 20(9), 1419–1440. arXiv:1907.06673. https://arxiv.org/abs/1907.06673

**Fuga de información y validación**

- Kaufman, S., Rosset, S., Perlich, C., Stitelman, O. (2012). *Leakage in data mining: Formulation, detection, and avoidance*. ACM TKDD 6(4), 1–21. https://dl.acm.org/doi/10.1145/2382577.2382579
- Kapoor, S., Narayanan, A. (2023). *Leakage and the reproducibility crisis in machine-learning-based science*. Patterns 4(9), 100804. https://www.cell.com/patterns/fulltext/S2666-3899(23)00159-9
- López de Prado, M. (2018). *Advances in Financial Machine Learning*. Wiley. Capítulo 7 (purga y embargo en validación cruzada). Resumen del método: https://en.wikipedia.org/wiki/Purged_cross-validation

**Datos sintéticos en finanzas**

- Assefa, S. A., Dervovic, D., Mahfouz, M., Tillman, R. E., Reddy, P., Veloso, M. (2020). *Generating synthetic data in finance: opportunities, challenges and pitfalls*. ICAIF '20. https://www.jpmorgan.com/content/dam/jpm/cib/complex/content/technology/ai-research-publications/pdf-8.pdf
- Esteban, C., Hyland, S. L., Rätsch, G. (2017). *Real-valued (Medical) Time Series Generation with Recurrent Conditional GANs*. arXiv:1706.02633. https://arxiv.org/abs/1706.02633 (protocolo TSTR: entrenar con sintético, evaluar con real).

**Material de clase analizado**

- `docs/material_clase/notebooks/Taller_Gaussian_solution.ipynb` (celdas 11, 13, 14).
- `docs/material_clase/notebooks/Taller_GANs.ipynb` (celdas 11, 13, 14).
