# Modelos generativos autoregresivos

> **Coste real de este generador dentro del taller.** Está clasificado como **opcional (nivel 3)**: se aborda solo si el calendario lo permite. El entrenamiento es *más barato* que el de una GAN (loss estable, una sola red, sin equilibrio adversario) y la loss es directamente interpretable, pero el **muestreo es 2-3 órdenes de magnitud más lento**: 60 pasadas secuenciales por ventana frente a una única pasada de un generador GAN o VAE. El trabajo adicional real está en (a) diseñar una cabeza de salida *distribucional* —sin ella no hay generador, ver §5— y (b) vectorizar el muestreo. Presupuesto estimado: 4-6 h de trabajo neto sobre una implementación ya existente de carga de datos, de las cuales ~1 h es entrenamiento en CPU y ~2 h es depurar el bucle de generación.

Material base: `docs/material_clase/slides/AR Generative Models.pdf` (Valero Laparra, 9 diapositivas), `docs/material_clase/notebooks/Taller_AR.ipynb` y `docs/material_clase/slides/2026_Intro_Generative_Models.pdf` (diap. 38-41, taxonomía de modelos *explicit density*).

---

## 1. Intuición: factorizar la densidad conjunta por la regla de la cadena

Un modelo generativo necesita representar $p(\mathbf{x})$ sobre un objeto de alta dimensión: una imagen, una frase, o —en nuestro caso— una ventana de 60 días × 20 canales. Modelar esa densidad conjunta de golpe es inviable. La idea autoregresiva consiste en **descomponerla en un producto de condicionales de una sola dimensión** aplicando la regla de la cadena de la probabilidad (diap. 3):

$$p(\mathbf{x}) = p(x_1, x_2, \dots, x_T) = \prod_{t=1}^{T} p(x_t \mid x_1, \dots, x_{t-1}) = \prod_{t=1}^{T} p(x_t \mid x_{<t})$$

Dos observaciones que conviene fijar desde el principio:

1. **La factorización es exacta, no una aproximación.** A diferencia del ELBO de un VAE (cota inferior) o del juego minimax de una GAN (sin verosimilitud), aquí no se pierde nada al descomponer. Toda la aproximación se concentra en la capacidad de la red para representar cada condicional $p(x_t \mid x_{<t})$. Por eso las diapositivas de introducción clasifican los autoregresivos como **modelos de densidad explícita y tratable** (`2026_Intro_Generative_Models.pdf`, diap. 38-39).

2. **La factorización exige un orden.** En imágenes ese orden es artificial: PixelCNN recorre la imagen en *raster scan*, izquierda-derecha y arriba-abajo (diap. 8), lo que impone una asimetría que no existe en el objeto real. En **series temporales financieras el orden viene dado por el tiempo**, y además coincide con la causalidad física del fenómeno: el precio de mañana depende del de hoy, no al revés. Esta es la razón principal por la que el enfoque autoregresivo es el más natural para nuestro problema y por la que su sesgo inductivo es aquí una ventaja y no un artefacto.

La intuición operativa es la del modelo de lenguaje carácter a carácter que se usa en clase como ejemplo (diap. 4-5, con la demo interactiva de Karpathy): el modelo lee el prefijo y emite una **distribución** sobre el siguiente símbolo. Generar consiste en muestrear de esa distribución, añadir el símbolo al prefijo y repetir. Sustituyendo "carácter" por "vector de 20 canales del día $t$" tenemos exactamente nuestro generador.

---

## 2. Formulación y verosimilitud exacta

Tomando logaritmos, el producto se convierte en suma y el entrenamiento es un problema de máxima verosimilitud estándar:

$$\log p_\theta(\mathbf{x}) = \sum_{t=1}^{T} \log p_\theta(x_t \mid x_{<t})
\qquad\Longrightarrow\qquad
\mathcal{L}(\theta) = -\frac{1}{N}\sum_{n=1}^{N}\sum_{t=1}^{T} \log p_\theta\!\left(x^{(n)}_t \mid x^{(n)}_{<t}\right)$$

Esta suma es la **log-verosimilitud negativa (NLL)** y es *toda* la función de pérdida: no hay término de regularización adversario, ni KL, ni pesos que ajustar entre objetivos. Las diapositivas la presentan en su versión de PLN (diap. 6): la métrica **intrínseca** es la *perplejidad*, "cuánto sorprende al modelo el siguiente token", frente a las métricas **extrínsecas** de tarea (por ejemplo BLEU en traducción). En nuestro caso la métrica extrínseca es el rendimiento del modelo *downstream* entrenado con los datos sintéticos.

Unidades en las que se debe reportar la pérdida: **nats por dimensión** $\mathcal{L}/(TC)$ por defecto en salidas continuas; **bits por dimensión** $\mathcal{L}/(TC\ln 2)$ para comparar con la literatura; **perplejidad** $\exp(\mathcal{L}/T)$ solo si la salida está discretizada.

**Aviso crítico sobre densidades continuas.** Si $x_t$ es continuo, $p_\theta$ es una *densidad*, no una probabilidad: puede ser mayor que 1 y la NLL puede ser **negativa**. Además **depende de la escala**. Si se estandariza $z = (x-m)/s$ por canal, la densidad en el espacio original se recupera con el jacobiano del cambio de variable:

$$\log p_X(\mathbf{x}) = \log p_Z(\mathbf{z}) - \sum_{j=1}^{C} \log s_j$$

Consecuencia práctica: **dos modelos solo son comparables por NLL si comparten exactamente el mismo preprocesado**, o si se corrige por el jacobiano. Este punto se vuelve central en §6, donde se compara el autoregresivo con un *normalizing flow*.

### Factorización dentro del paso temporal

Con $x_t \in \mathbb{R}^{C}$ ($C = 20$ canales) hay que decidir cómo se modela el vector completo:

- **Independencia condicional**: $p(x_t\mid x_{<t}) = \prod_c p(x_{t,c}\mid x_{<t})$. Es lo más simple y lo que sale por defecto de una cabeza con $2C$ salidas. **Destruye la correlación contemporánea** entre el S&P y los sectoriales, que es fortísima ($\rho > 0{,}8$ en muchos pares). Inaceptable sin comprobación explícita.
- **Cadena también entre canales**: $p(x_t\mid x_{<t}) = \prod_c p(x_{t,c}\mid x_{<t}, x_{t,<c})$. Es exactamente la solución de PixelCNN para los canales RGB, implementada con máscaras de tipo A y B (diap. 8-9). Correcta pero multiplica el coste de muestreo por $C$: $60\times20 = 1200$ pasadas.
- **Gaussiana multivariante con covarianza**: la red emite $\mu_t\in\mathbb{R}^{C}$ y un factor de Cholesky $L_t$ triangular inferior ($C(C+1)/2 = 210$ parámetros para $C=20$). Un único paso de muestreo por instante temporal y correlaciones contemporáneas capturadas. **Es el mejor compromiso para este taller.**

> Ampliación (no cubierto en clase): la parametrización por Cholesky con diagonal positiva (`softplus` o exponencial sobre la diagonal) garantiza que $\Sigma_t = L_tL_t^\top$ sea definida positiva y permite calcular $\log\det\Sigma_t = 2\sum_j \log L_{t,jj}$ en $O(C)$.

---

## 3. Arquitecturas: máscaras causales, convoluciones dilatadas (WaveNet), transformers

Cualquier arquitectura vale mientras respete una única restricción: **la salida en el instante $t$ no puede depender de $x_{\ge t}$**. Todo lo demás son variantes de cómo comprimir el prefijo.

### 3.1 Redes recurrentes (diap. 4)

El estado oculto $h_t = \sigma(W_{hh}h_{t-1} + W_{xh}x_t)$ resume todo el pasado y $W_{hy}h_t$ parametriza la distribución del siguiente paso; la causalidad es automática. Coste: el entrenamiento es **secuencial en $t$** y no se paraleliza. Para $T=60$ el desvanecimiento del gradiente es un problema menor (y lo mitiga una LSTM); el problema real es la lentitud en CPU.

### 3.2 Convoluciones enmascaradas (diap. 8-9)

PixelCNN sustituye la recurrencia por convoluciones cuyo kernel está **enmascarado** para no ver el futuro. Las diapositivas distinguen los dos tipos clásicos (diap. 9):

- **Máscara tipo A**: anula también el píxel central. Se usa en la **primera capa**, para que la predicción de $x_i$ no dependa de $x_i$.
- **Máscara tipo B**: el centro es visible. Se usa en las **capas siguientes**, donde el centro ya representa una función solo del pasado.

En 1D esto se implementa de forma más limpia con **padding asimétrico**: se rellena la secuencia solo por la izquierda con $(k-1)d$ ceros y se recorta la cola. Es la construcción que se usa en §9.

Ventaja decisiva frente a la RNN: en **entrenamiento** todas las posiciones $t$ se calculan en paralelo con una sola pasada (*teacher forcing*), porque los targets son conocidos. La secuencialidad solo reaparece en el muestreo.

### 3.3 Convoluciones causales dilatadas (WaveNet)

> Ampliación (no cubierto en clase): las diapositivas describen convoluciones enmascaradas pero no las dilataciones ni WaveNet.

El problema de apilar convoluciones causales con kernel $k$ es que el campo receptivo crece linealmente: $L(k-1)+1$ capas. Con **dilatación** $d_\ell = 2^{\ell-1}$ crece exponencialmente:

$$\text{RF} = 1 + (k-1)\sum_{\ell=1}^{L} d_\ell = 1 + (k-1)(2^{L}-1)$$

Con $k=2$ y $L=6$ (dilataciones $1,2,4,8,16,32$) se obtiene $\text{RF}=64 \ge 60$: **seis capas bastan para que el último día vea toda la ventana**. Es el motivo por el que esta es la arquitectura de referencia del documento: cubre el contexto completo con ~90k parámetros y entrena en minutos en CPU.

### 3.4 Transformers causales (diap. 7)

Las diapositivas listan los componentes: *embeddings* de token más codificación posicional, **atención multi-cabeza causal** (la máscara triangular es el equivalente exacto de la máscara tipo A/B en el dominio de la atención), *layer norm* con conexiones residuales y MLP de expansión. El coste de atención es $O(T^2)$ en tiempo y memoria por capa.

### 3.5 Comparativa para nuestro caso ($T=60$, $C=20$, CPU)

| Arquitectura | Entrenamiento | Muestreo | Parámetros típicos | Veredicto CPU |
|---|---|---|---|---|
| LSTM 2×64 | secuencial, $O(T)$ | $O(T)$, estado cacheable | ~60k | viable, alternativa |
| CNN causal dilatada | paralelo en $T$ | $O(T)$ pasadas | ~90k | **recomendada** |
| Transformer causal 4 capas | paralelo, $O(T^2)$ | $O(T)$ con caché KV | ~400k+ | sobredimensionado |

Con 60 pasos y 3.696 ventanas de entrenamiento, un transformer no aporta nada que la CNN dilatada no capture, y multiplica por 4-5 el tiempo en CPU.

---

## 4. Muestreo secuencial y acumulación de error (exposure bias)

Existe una asimetría estructural entre cómo se entrena y cómo se genera:

- **Entrenamiento (*teacher forcing*)**: el condicionante $x_{<t}$ es siempre **historia real**. Una sola pasada por ventana.
- **Generación (*free running*)**: el condicionante son las **muestras que el propio modelo ha generado**. $T$ pasadas encadenadas.

La distribución de entradas que ve el modelo en generación no es la que vio en entrenamiento. A esto se le llama **exposure bias**: en cuanto el modelo comete una desviación, entra en una región del espacio de entradas donde nunca fue entrenado, y el error se **compone** paso a paso. Formalmente, si $\hat{x}_{<t}$ se aleja del soporte de entrenamiento, la calidad de $p_\theta(\cdot\mid\hat{x}_{<t})$ deja de estar garantizada, y el efecto se acumula multiplicativamente a lo largo de los 60 pasos.

Manifestaciones típicas en series financieras: **deriva de la volatilidad** (la desviación típica de la trayectoria crece o decrece monótonamente con $t$ en vez de mantenerse estacionaria), **explosión** (si $\sigma$ se realimenta al alza, la trayectoria diverge en pocas decenas de pasos) y **colapso**, el caso opuesto y el que efectivamente ocurre en el notebook del taller (§5).

Mitigaciones, de menor a mayor coste:
1. **Sembrar con contexto real** (*warm start*): dar al modelo 10-15 días reales y generar solo los restantes. Reduce mucho la deriva. Coste: las muestras son menos novedosas y el contexto debe salir **siempre del conjunto de entrenamiento**, nunca de test.
2. **Diagnóstico por *rollout* durante el entrenamiento**: cada $N$ épocas, generar 200 trayectorias completas y comparar sus estadísticos con los reales. Es la única forma de detectar deriva, porque la NLL con *teacher forcing* **no la ve**.
3. **Truncar por cordura**: recortar cada muestra a $\pm 6\sigma$ del canal. Feo, pero evita que una sola muestra tóxica arruine el lote.

> Ampliación (no cubierto en clase): *scheduled sampling* (Bengio et al., 2015) mezcla progresivamente entradas reales y generadas durante el entrenamiento; y los enfoques de *professor forcing* añaden un discriminador sobre los estados ocultos. Ambos complican el entrenamiento y no se justifican en un taller de este alcance.

---

## 5. Modelado de la incertidumbre: salida determinista vs distribucional

**Esta es la sección más importante del documento.** El notebook `Taller_AR.ipynb` contiene un error conceptual que invalida su uso directo como generador, y entenderlo es la diferencia entre implementar un modelo generativo y implementar un regresor disfrazado.

### 5.1 Qué hace exactamente el notebook

Celda 17: una CNN de `Conv1D` + `MaxPooling1D` + `Flatten` + `Dense`, que mapea una ventana $(60, 23)$ a un vector de 23 valores, compilada con `loss='mse'`. Celda 20, el bucle de generación:

```python
generated = X_train[1:2,:,:].copy()
for i in range(LEN):
    preds = cnn_model_2.predict(generated, verbose=0)[0]
    generated[:,0:-1,:] = generated[:,1:,:]   # desplaza la ventana
    generated[:,-1,:]   = preds               # realimenta la PREDICCIÓN PUNTUAL
```

Se desplaza la ventana un paso y se inserta la salida de la red como si fuera el nuevo día.

### 5.2 Tres defectos, en orden de gravedad

**(1) La salida es una media condicional, no una distribución.** Minimizar el ECM tiene una solución óptima conocida:

$$f^\star(x_{<t}) = \arg\min_f \mathbb{E}\left[(x_t - f(x_{<t}))^2\right] = \mathbb{E}[x_t \mid x_{<t}]$$

La red aprende, en el mejor de los casos, la **media condicional**. No hay verosimilitud, no hay $p_\theta(x_t\mid x_{<t})$, y por tanto **no hay nada de lo descrito en §2**: ni NLL, ni perplejidad, ni comparabilidad. Es un regresor.

**(2) Realimentar la media colapsa la trayectoria.** Sea el proceso real $x_t = \mu(x_{<t}) + \varepsilon_t$ con $\operatorname{Var}(\varepsilon_t)=\sigma^2 > 0$. El generador determinista produce $\hat{x}_t = \mu(\hat{x}_{<t})$, es decir, **elimina $\varepsilon_t$ en cada paso**. La recursión resultante es determinista y, en rendimientos diarios —donde la parte predecible de la media es prácticamente nula, $R^2 \approx 0$— es además fuertemente contractiva: tras unos pocos pasos la ventana se llena de valores próximos a la media incondicional y la trayectoria se aplana. La varianza de la muestra generada tiende a cero:

$$\operatorname{Var}(\hat{x}_t) \longrightarrow 0 \qquad \text{mientras que} \qquad \operatorname{Var}(x_t) = \sigma^2$$

Es exactamente lo que muestra la figura de la celda 21: curvas planas. **Un generador cuyas muestras tienen volatilidad casi nula es inservible para un taller cuyo objetivo es predecir volatilidad.** Los sintéticos no solo no ayudan al modelo *downstream*: rompen activamente la relación $X \to y_{\text{vol}}$ que ese modelo debe aprender.

**(3) Desajuste de target adicional.** El target `Y` del notebook (celda 13) es la **media de los 30 días siguientes**, no el rendimiento del día siguiente. Al realimentarlo como un día individual se inyecta una serie cuya escala está mal por construcción: si los rendimientos fueran i.i.d., la desviación típica de una media de 30 sería $\sigma/\sqrt{30} \approx \sigma/5{,}5$. Se está realimentando una señal ~5,5 veces menos volátil que la que el modelo espera en su entrada. Esto agrava el colapso del punto (2) y hace que la primera iteración ya sea inconsistente con la distribución de entrada.

**Regla práctica.** *Si el bucle de generación no consume números aleatorios en cada paso, no es un modelo generativo.* En el bucle de la celda 20 no aparece ninguna llamada a un generador de números aleatorios.

### 5.3 La solución correcta: predecir una distribución y muestrear de ella

La cabeza de la red debe emitir **parámetros de una distribución**, y el bucle debe **muestrear**:

```python
# INCORRECTO (notebook): realimenta la media condicional
x_next = model(ventana)

# CORRECTO: la red parametriza una densidad y se extrae una realización
mu, log_sigma = model(ventana)
x_next = mu + torch.exp(log_sigma) * torch.randn_like(mu)   # el ruido es lo que genera
```

Tres opciones de cabeza, por orden creciente de expresividad:

**(a) Gaussiana heterocedástica.** La red emite $\mu_\theta$ y $\log\sigma_\theta$ por canal y paso. La pérdida es la NLL gaussiana:

$$-\log p_\theta(x_t\mid x_{<t}) = \log\sigma_\theta + \tfrac{1}{2}\log(2\pi) + \frac{(x_t-\mu_\theta)^2}{2\sigma_\theta^2}$$

Sale gratis respecto al ECM (una cabeza más) y ya captura lo esencial de los rendimientos financieros: la **agrupación de volatilidad**, porque $\sigma_\theta$ depende del contexto. Limitación: colas gaussianas, sin curtosis ni asimetría. Precaución obligatoria: acotar $\log\sigma$ por abajo (p. ej. $\ge -6$), porque si $\sigma\to 0$ sobre una muestra memorizada la NLL diverge a $-\infty$ y el entrenamiento explota.

**(b) Mixture Density Network (MDN).** $K$ componentes gaussianas con pesos $\pi_k$ dependientes del contexto:

$$p_\theta(x_t\mid x_{<t}) = \sum_{k=1}^{K}\pi_k(x_{<t})\,\mathcal{N}\!\left(x_t;\mu_k(x_{<t}),\sigma_k^2(x_{<t})\right),\qquad \sum_k\pi_k = 1$$

Se implementa con `logsumexp` sobre $\log\pi_k + \log\mathcal{N}_k$ por estabilidad numérica. Con $K=3{-}5$ captura **colas gruesas y asimetría** (una componente estrecha para el régimen normal, otra ancha para los saltos). Es la opción con mejor relación calidad/coste para datos financieros.

**(c) Discretización en bins + softmax.** Es literalmente lo que hace PixelCNN con los valores 0-255 (diap. 8: *"Values: Discrete 0-255 (Softmax) or Logistic Mixture"*). Se cuantiza cada canal en $K$ bins definidos por **cuantiles empíricos** (bins finos en el centro, anchos en las colas), y se entrena con entropía cruzada. Ventajas: ninguna hipótesis paramétrica, colas y multimodalidad capturadas automáticamente, y la pérdida es una perplejidad interpretable (diap. 6). Inconvenientes: se pierde la ordinalidad (el modelo no sabe que el bin 12 está junto al 13, mitigable con *label smoothing* sobre bins vecinos), la resolución está limitada por $K$, y la salida crece a $K\times C$ (con $K=50$ y $C=20$, 1.000 logits).

> Ampliación (no cubierto en clase): la *mixture of discretized logistics* de PixelCNN++, mencionada de pasada en la diap. 8, es el compromiso entre (b) y (c): distribución discreta pero parametrizada de forma continua, con muchos menos parámetros de salida que un softmax de 256 clases.

### 5.4 Temperatura: no la uses para "arreglar" las muestras

Es habitual muestrear con $x_{t} = \mu + \tau\sigma\varepsilon$ y $\tau<1$ para obtener trayectorias "más limpias". **Con $\tau<1$ se está reintroduciendo deliberadamente el colapso a la media descrito en §5.2.** Para generar datos sintéticos destinados a entrenar otro modelo, $\tau=1$ es la única opción defendible; cualquier otro valor debe justificarse y reportarse, porque sesga sistemáticamente $y_{\text{vol}}$ a la baja.

### 5.5 Comprobación mínima antes de dar el generador por bueno

Tabla real vs sintético con cuatro filas: $\sigma$ anualizada por canal (tolerancia ±15 %), curtosis del canal S&P (mismo orden de magnitud), autocorrelación de $\lvert r_t\rvert$ en los lags 1-10 (mismo signo y decaimiento, es la firma de la agrupación de volatilidad) y correlación media entre S&P y sectoriales (±0,1). Si la columna sintética da $\sigma \approx 0$, se está reproduciendo el fallo del notebook.

---

## 6. Diagnóstico de convergencia

El enunciado (`docs/enunciado/Taller_B5_T1.pdf`, sección 5) exige literalmente: *"Para cada entrenamiento, incluir las curvas de loss donde se vea que el modelo ha convergido"*. El autoregresivo es el generador que **mejor cumple ese requisito**, y hay que explotarlo explícitamente en la presentación.

### 6.1 Por qué aquí la loss sí significa algo

En una GAN la pérdida del generador y la del discriminador miden el estado de un juego, no la calidad de las muestras: pueden oscilar indefinidamente con un generador excelente, o quedarse planas con uno colapsado. En un autoregresivo, **la loss es la NLL exacta del modelo sobre los datos**: es la misma cantidad que se optimiza, la que mide la calidad del ajuste y la que permite comparar dos modelos. Una curva de NLL de validación que baja y se estabiliza es una demostración de convergencia en sentido estricto, no una heurística.

### 6.2 Qué graficar

1. **NLL de entrenamiento y validación por época, en el mismo eje**, en nats por dimensión (dividir por $T\cdot C$ para que el número no dependa del tamaño de la ventana).
2. **Líneas horizontales de referencia** que hagan legible el valor absoluto. Sobre datos estandarizados por canal, el baseline gaussiano i.i.d. es una constante conocida:

$$\mathrm{NLL}_{\text{i.i.d.}} = \tfrac{1}{2}\log(2\pi\sigma^2) + \tfrac{1}{2} \;\xrightarrow{\ \sigma=1\ }\; \tfrac{1}{2}\log(2\pi) + \tfrac{1}{2} = 1{,}4189\ \text{nats/dim} = 2{,}047\ \text{bits/dim}$$

   Cualquier modelo que no baje de 1,4189 **no ha aprendido nada**. Es la prueba de humo obligatoria.
3. **Baseline fuerte**: una gaussiana con volatilidad condicional EWMA o GARCH(1,1) por canal, evaluada con la misma NLL. Si el autoregresivo no lo bate en validación, no compensa su coste. La mejora esperable es de **décimas de nat**, porque en rendimientos diarios lo predecible es la volatilidad, no la media; una mejora grande sobre este baseline es señal de fuga de información, no de talento.
4. **NLL desagregada por posición $t$** dentro de la ventana. Los primeros pasos tienen menos contexto y su NLL debe ser mayor; una NLL que **crece** con $t$ delata un fallo de causalidad o de normalización.

### 6.3 Lectura de la curva

| Patrón observado | Diagnóstico | Acción |
|---|---|---|
| Train y val bajan juntas y se aplanan | convergencia sana | parar; es la figura que pide el enunciado |
| Val se aplana y luego sube; train sigue bajando | sobreajuste | *early stopping*, *dropout*, reducir anchura |
| Plana desde la época 1 en $\approx 1{,}42$ | no aprende: lr muy bajo, o la cabeza solo ve ruido | subir lr, revisar el desplazamiento entrada/target |
| NLL cae a $-\infty$ o aparece `nan` | $\sigma_\theta\to 0$ sobre muestras memorizadas | acotar $\log\sigma \ge -6$, bajar lr |
| Escalones bruscos | efecto del *scheduler* de lr | normal si es intencionado |
| Val **por debajo** de train de forma persistente | fuga de información entre particiones | revisar el *split* (ver 6.4) |

### 6.4 El error de partición que hay que corregir

El notebook usa `train_test_split(X, Y, test_size=0.1, random_state=42)` (celda 14) sobre ventanas **solapadas** con paso 1 día. Dos ventanas consecutivas comparten 59 de sus 60 días: al barajar aleatoriamente, el conjunto de validación contiene casi la misma información que el de entrenamiento. La NLL de validación resultante es optimista y **no demuestra convergencia, sino memorización**. Corrección obligatoria:

- **Partición temporal** en bloques contiguos: entrenamiento hasta 2018, validación 2019-2021, test 2022 en adelante.
- **Purga y embargo** contado en sesiones de mercado: la huella de una ventana es $60 + H = 81$ sesiones, con $H=21$ el horizonte usado para $y_{\text{vol}}$, el mínimo que neutraliza el solape es 80 y el valor adoptado es **85 sesiones**.
- Estadísticos de normalización calculados **solo con entrenamiento**.

### 6.5 Convergencia no es correctitud: cuatro comprobaciones adicionales

Una NLL baja garantiza buen ajuste bajo *teacher forcing*, pero no dice nada sobre las trayectorias generadas libremente (§4). Añadir:

1. **Calibración por PIT.** Con $u_t = F_\theta(x_t\mid x_{<t})$, si el modelo está bien calibrado los $u_t$ son uniformes en $[0,1]$. Histograma con forma de U ⇒ el modelo es **sobreconfiado** ($\sigma$ demasiado pequeña); forma de campana ⇒ **infraconfiado**. Contrastar con Kolmogórov-Smirnov.
2. **Cobertura de intervalos.** El intervalo predicho al 90 % debe contener el 90 % de las observaciones de validación (±2 puntos).
3. **Estadísticos de *rollout*.** La tabla de §5.5, calculada sobre 1.000 trayectorias generadas.
4. **Distancia al vecino más próximo.** Para cada ventana sintética, distancia mínima a las ventanas reales de entrenamiento. Si esa distribución está desplazada muy a la izquierda respecto de la distancia real-a-real, el modelo está copiando.

### 6.6 Comparabilidad entre los generadores del taller

- **AR vs *normalizing flow***: comparables por NLL **solo** si ambos reportan log-densidad en el mismo espacio y con el mismo preprocesado, corrigiendo el jacobiano de §2. Documentarlo explícitamente en el informe.
- **AR vs GAN**: no comparables por loss, porque la GAN no define verosimilitud. La comparación debe hacerse por los estadísticos de §5.5 y por la métrica *downstream*, que es la métrica extrínseca del enunciado.
- **AR vs baseline "datos + ruido"**: el baseline simple exigido por el enunciado también admite NLL si se formula como una gaussiana centrada en el dato real, y resulta un punto de referencia sorprendentemente duro.

---

## 7. Patologías y limitaciones

- **Muestreo $O(T)$ estrictamente secuencial.** No hay forma de paralelizar en el eje temporal. Es la limitación estructural del enfoque y la que decide si merece la pena (§9.4).
- **Exposure bias y deriva** en horizontes largos (§4).
- **Colapso a la media** si la cabeza es determinista (§5). Es el fallo más frecuente y el más difícil de detectar mirando solo la loss.
- **Sensibilidad al orden de factorización.** En series temporales el orden es natural, pero para variables tabulares no ordenadas el orden elegido sesga el modelo.
- **Estructura global débil.** Modelan muy bien la dependencia local y peor las propiedades globales de la ventana (*drawdown* acumulado, forma del episodio), porque nada en la pérdida penaliza el resultado agregado a 60 días.
- **Memorización.** Con verosimilitud exacta, 3.696 ventanas de train muy solapadas —sólo 70 bloques disjuntos en todo el panel— y 90k parámetros, memorizar es más fácil que generalizar. Comprobación obligatoria en 6.5.4.
- **Régimen minoritario.** La clase "crisis" es ~16 % de las muestras de train (10,5 % en test), y sólo 8 rachas contiguas. El condicional $p(x_t\mid x_{<t}, y_{\text{reg}}=\text{crisis})$ se estima con muy pocos datos y tiende a colapsar hacia el régimen mayoritario: se generan "crisis" que son en realidad mercados normales. Mitigación: sobremuestrear la clase crisis en el entrenamiento del generador y **verificar** que la volatilidad de los sintéticos condicionados a crisis es efectivamente superior.
- **Sin espacio latente.** No hay representación comprimida: no se pueden hacer interpolaciones ni aritmética en el latente, a diferencia de un VAE o una GAN.
- **NLL baja no implica muestras buenas.** Son objetivos relacionados pero no equivalentes; por eso §6.5 no es opcional.

> Ampliación (no cubierto en clase): la disociación entre verosimilitud y calidad perceptual de las muestras está analizada en Theis, van den Oord & Bethge, *A note on the evaluation of generative models* (ICLR 2016).

---

## 8. Aplicación a nuestro problema

### 8.1 Objeto a generar

El taller genera el **bloque conjunto** $[\,X;\ y_{\text{vol}}\,]$, con $X\in\mathbb{R}^{60\times 20}$ procedente de un panel híbrido de 15 activos (S&P 500, 9 SPDR sectoriales, VIX, tesoro a 20 y a 10 años, crédito grado de inversión e índice dólar): once canales de retornos (índice, nueve sectores y dólar) y nueve derivados (nivel y variación del VIX, volatilidad realizada, *drawdown*, momento, spread de crédito, pendiente de curva, correlación acción-bono y dispersión sectorial). $y_{\text{vol}}\in\mathbb{R}$ es la volatilidad futura y $y_{\text{reg}}\in\{0,1,2\}$ el régimen, que va aparte como condición.

### 8.2 Factorización elegida

$$p(X, y_{\text{reg}}, y_{\text{vol}}) \;=\; \underbrace{p(y_{\text{reg}})}_{\text{se fija a mano}}\;\cdot\;\underbrace{\prod_{t=1}^{60} p_\theta(x_t\mid x_{<t}, y_{\text{reg}})}_{\text{el modelo autoregresivo}}\;\cdot\;\underbrace{\delta\!\left(y_{\text{vol}} - g(x_{61:60+H})\right)}_{\text{derivada, no modelada}}$$

Tres decisiones incorporadas en esa expresión:

**(a) $y_{\text{reg}}$ no se muestrea de la distribución empírica: se impone.** Si el ~16 % de los datos reales de train son crisis, muestrear $p(y_{\text{reg}})$ reproduce ese ~16 % y el sintético no aporta nada donde más falta hace. La justificación de negocio de este generador es precisamente **rellenar la clase minoritaria**: se generan lotes con la proporción de crisis que se decida (30 %, 50 %) y se mide el efecto en el modelo *downstream*.

**(b) El condicionamiento al régimen es global.** Se aprende un *embedding* $e(y_{\text{reg}})\in\mathbb{R}^{48}$ que se suma a la representación de **todos** los pasos temporales, análogo al *global conditioning* de WaveNet. Coste: 144 parámetros. Alternativa más expresiva: modulación FiLM ($\gamma,\beta$ por bloque residual), que multiplica y desplaza las activaciones; solo merece la pena si el condicionamiento aditivo se ignora, lo que se detecta comparando la volatilidad generada por clase.

**(c) $y_{\text{vol}}$ se deriva, no se genera.** Es la decisión más importante de esta sección. Dos opciones:

- **Recomendada**: generar $60+H$ pasos, usar los 60 primeros como $X$ y calcular $y_{\text{vol}}$ sobre los $H$ siguientes **con exactamente la misma función** que se usa sobre datos reales. Coherencia garantizada por construcción: la relación $X\to y_{\text{vol}}$ que el sintético enseña al modelo *downstream* es la que el propio generador ha producido, no una etiqueta pegada a posteriori.
- **Descartada**: apilar $y$ como paso temporal 61, que es lo que hace el notebook para la GAN (celda 24, `XY_aux[:, 60, :] = Y_train_aux`). Más simple, pero nada garantiza que la etiqueta generada sea consistente con la trayectoria generada, y ese desajuste se transmite directamente al modelo *downstream*.

### 8.3 Preprocesado

Cada canal exige un tratamiento distinto antes de entrar en el modelo, y todo lo que se aplique debe ser **invertible y documentado** para poder reportar NLL comparable (§2): rendimientos logarítmicos diarios para el S&P, los sectoriales y el dólar; logaritmo y luego diferencias para el VIX; diferencias para el spread de crédito y la pendiente de curva; logaritmo para la volatilidad realizada; el *drawdown* ($\in[-1,0]$) puede quedarse en nivel o pasarse por un *logit* reescalado.

Después: *winsorizar* al 0,1 %/99,9 % (con los umbrales de entrenamiento) y estandarizar por canal con $\mu,\sigma$ **de entrenamiento**. Sin la estandarización, los canales de mayor varianza dominan la NLL y el modelo ignora los demás.

### 8.4 Protocolo de evaluación

Idéntico al del resto de generadores para que la comparación del enunciado sea válida: mismos porcentajes de mezcla real/sintético (0 / 25 / 50 / 75 / 100 %), misma arquitectura *downstream*, mismas semillas, y **test 100 % real** con partición temporal. Se reportan además las curvas de NLL del generador (§6) y la tabla de estadísticos de §5.5.

---

## 9. Implementación de referencia (CPU)

Entorno: `torch 2.11.0+cpu`, Python 3.13.7, sin CUDA. Todo lo que sigue está dimensionado para ese entorno.

### 9.1 Arquitectura

WaveNet-lite: proyección de entrada $20\to 48$, seis bloques residuales con convolución causal $k=2$ y dilataciones $1,2,4,8,16,32$ (campo receptivo 64 ≥ 60), activación con puerta, conexiones residuales y *skip*, y cabeza gaussiana heterocedástica de $2\times 20$ salidas.

Recuento de parámetros: 1.008 (conv 1×1 de entrada $20\to48$) + 144 (*embedding* de régimen) + 84.096 (6 bloques × [9.312 de la conv dilatada con puerta + 2.352 residual + 2.352 *skip*]) + 4.312 (cabeza $48\to48\to40$) = **≈ 89.600 parámetros, ~0,36 MB en float32**.

Es un modelo deliberadamente pequeño: con 3.696 ventanas de entrenamiento, subir de ~150k parámetros lleva directamente a memorización.

### 9.2 Código

```python
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_num_threads(8)          # ajustar al número de núcleos físicos

class CausalConv1d(nn.Module):
    """Convolución causal dilatada: rellena solo por la izquierda para no ver el futuro."""
    def __init__(self, in_ch, out_ch, kernel_size=2, dilation=1):
        super().__init__()
        self.pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, dilation=dilation)

    def forward(self, x):                       # x: (B, C, T)
        return self.conv(F.pad(x, (self.pad, 0)))


class WaveNetLite(nn.Module):
    """Generador autoregresivo con cabeza gaussiana heterocedastica."""
    def __init__(self, n_ch=20, n_reg=3, width=48, dilations=(1, 2, 4, 8, 16, 32)):
        super().__init__()
        self.inp  = nn.Conv1d(n_ch, width, 1)
        self.emb  = nn.Embedding(n_reg, width)          # condicionamiento global al regimen
        self.filt = nn.ModuleList([CausalConv1d(width, 2 * width, 2, d) for d in dilations])
        self.res  = nn.ModuleList([nn.Conv1d(width, width, 1) for _ in dilations])
        self.skip = nn.ModuleList([nn.Conv1d(width, width, 1) for _ in dilations])
        self.h_mu = nn.Conv1d(width, n_ch, 1)
        self.h_ls = nn.Conv1d(width, n_ch, 1)

    def forward(self, x, reg):                          # x: (B, C, T);  reg: (B,)
        h = self.inp(x) + self.emb(reg).unsqueeze(-1)   # el embedding se difunde a todos los pasos
        s = 0.0
        for f, r, k in zip(self.filt, self.res, self.skip):
            a, b = f(h).chunk(2, dim=1)
            z = torch.tanh(a) * torch.sigmoid(b)        # activacion con puerta (WaveNet)
            s = s + k(z)
            h = h + r(z)                                # conexion residual
        s = F.relu(s)
        mu = self.h_mu(s)
        log_sigma = self.h_ls(s).clamp(-6.0, 3.0)       # suelo de sigma: evita NLL -> -inf
        return mu, log_sigma


def nll_gaussiana(mu, log_sigma, objetivo):
    """NLL media en nats por (paso, canal). Es la loss que se grafica en la seccion 6."""
    return (log_sigma
            + 0.5 * math.log(2 * math.pi)
            + 0.5 * (objetivo - mu) ** 2 * torch.exp(-2.0 * log_sigma)).mean()
```

Entrenamiento con *teacher forcing* (una sola pasada por ventana, todas las posiciones en paralelo):

```python
modelo = WaveNetLite()
opt = torch.optim.Adam(modelo.parameters(), lr=1e-3)
hist = {"train": [], "val": []}

for epoca in range(120):
    modelo.train()
    acum = 0.0
    for ventana, reg in cargador_train:                 # ventana: (B, 20, 60)
        entrada, objetivo = ventana[:, :, :-1], ventana[:, :, 1:]   # desplazamiento de un paso
        mu, ls = modelo(entrada, reg)
        loss = nll_gaussiana(mu, ls, objetivo)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(modelo.parameters(), 5.0)    # estabiliza la cabeza de sigma
        opt.step()
        acum += loss.item()
    hist["train"].append(acum / len(cargador_train))
    hist["val"].append(evaluar(modelo, cargador_val))    # misma NLL, sin gradientes
    # criterio de parada: NLL de validacion sin mejorar en 15 epocas -> restaurar el mejor estado
```

Muestreo **vectorizado en el eje de muestras** (el punto crítico de rendimiento):

```python
@torch.inference_mode()
def muestrear(modelo, n, T=60, H=21, n_ch=20, regimen=2, temp=1.0):
    """Genera n trayectorias de T+H pasos en paralelo. temp=1.0 es la unica opcion no sesgada."""
    modelo.eval()
    reg = torch.full((n,), regimen, dtype=torch.long)
    x = torch.zeros(n, n_ch, 1)                          # token de inicio: contexto cero
    for _ in range(T + H):
        mu, ls = modelo(x, reg)
        mu_t, ls_t = mu[:, :, -1], ls[:, :, -1]          # solo interesa el ultimo paso
        ruido = torch.randn_like(mu_t)                   # <-- sin esto no hay modelo generativo
        x_next = mu_t + temp * torch.exp(ls_t) * ruido
        x = torch.cat([x, x_next.unsqueeze(-1)], dim=-1)
    return x[:, :, 1:]                                   # (n, 20, T+H)


def derivar_y_vol(traj, idx_sp500=0, T=60, H=21):
    """y_vol de la trayectoria generada, con la MISMA formula usada sobre datos reales."""
    futuro = traj[:, idx_sp500, T:T + H]
    return futuro.std(dim=1) * math.sqrt(252)
```

### 9.3 Tiempos estimados en CPU

Supuestos: 3.696 ventanas de train de $60\times20$, lote 128 (29 lotes/época), 8 hilos.

| Fase | Coste | Estimación |
|---|---|---|
| Entrenamiento, una época | 29 pasadas paralelas en $t$ | **1-2,5 s** |
| Entrenamiento, 120 épocas | — | **2-5 min** |
| Muestreo, lote de 512 en paralelo | 81 pasadas de red | **2-5 s** |
| Muestreo de 5.000 muestras (10 lotes) | — | **30-60 s** |
| Muestreo **muestra a muestra** (bucle ingenuo) | 5.000 × 81 pasadas | **varias horas** |

La última fila es el error que hay que evitar y es exactamente el patrón del notebook (`predict` sobre un único ejemplo dentro del bucle, celda 20). Con la generación vectorizada en el eje de muestras, el coste total es perfectamente asumible.

### 9.4 El cuello de botella, en contexto

Generar 5.000 muestras cuesta **81 pasadas de red secuenciales**, frente a **1 pasada** de una GAN o un VAE, que producen la ventana completa de un tiro. Ese factor ~80-100× es intrínseco al enfoque y no se elimina; solo se amortigua paralelizando el eje de muestras. En términos absolutos, y para el tamaño de este taller, sigue siendo un minuto de cómputo: **el coste de muestreo no es el argumento para descartar el autoregresivo**; el argumento es el tiempo de implementación de la cabeza distribucional y del bucle de generación correcto.

> Ampliación (no cubierto en clase): el muestreo ingenuo recalcula todo el prefijo en cada paso, con coste $O(T^2)$. La técnica de *fast WaveNet* cachea las activaciones de cada capa dilatada en colas circulares y reduce el coste a $O(T)$. Con $T=81$ la ganancia no compensa la complejidad.

### 9.5 Entregables mínimos de este generador

Curvas de NLL de entrenamiento y validación con las líneas de referencia i.i.d. y GARCH (§6.2); histograma PIT y tabla de cobertura (§6.5); panel de 9 trayectorias reales frente a 9 sintéticas con el mismo eje vertical; tabla de estadísticos real vs sintético (§5.5) desglosada por régimen; y curva de métrica *downstream* frente a porcentaje de sintético, comparable con el resto de generadores.

---

## 10. Referencias

**Material de clase**

- `docs/material_clase/slides/AR Generative Models.pdf` — Valero Laparra, *Autoregressive Generative Models*, Deep Learning & Generative AI. Diap. 3 (regla de la cadena y probabilidad conjunta), diap. 4-5 (RNN y demo de Karpathy), diap. 6 (funciones de pérdida, perplejidad, intrínseca vs extrínseca), diap. 7 (transformers y atención causal), diap. 8 (PixelCNN, imágenes como secuencias, salida softmax discreta o mixtura logística), diap. 9 (convoluciones enmascaradas, máscaras tipo A y B).
- `docs/material_clase/slides/2026_Intro_Generative_Models.pdf` — diap. 8 (taxonomía de modelos generativos profundos), diap. 38-41 (autoregresivos como modelo de densidad explícita, más enlaces recomendados).
- `docs/material_clase/notebooks/Taller_AR.ipynb` — celdas 13-14 (construcción de ventanas y partición), celda 17 (CNN con pérdida ECM), celda 20 (bucle de generación por realimentación de la predicción puntual, analizado en §5), celda 24 (apilado de $Y$ como paso extra).
- `docs/enunciado/Taller_B5_T1.pdf` — sección 4 (tareas: tres generadores distintos más un baseline con ruido) y sección 5 (entregables, incluida la exigencia de curvas de loss que muestren convergencia).

**Enlaces citados en las diapositivas**: curso CSC421 de Toronto, *Autoregressive and Reversible Models* (`cs.toronto.edu/~rgrosse/courses/csc421_2019/readings/`); ejemplo oficial de PixelCNN en Keras (`keras.io/examples/generative/pixelcnn/`); visualizador de RNN de Karpathy (`cs.stanford.edu/people/karpathy/recurrentjs/`); visualizador de LLM de Bycroft (`bbycroft.net/llm`).

**Ampliación (no cubierto en clase)**

- van den Oord et al. (2016), *WaveNet: A Generative Model for Raw Audio* — convoluciones causales dilatadas, activación con puerta, condicionamiento global.
- van den Oord et al. (2016), *Pixel Recurrent Neural Networks* — PixelRNN y PixelCNN.
- Salimans et al. (2017), *PixelCNN++* — mixtura de logísticas discretizadas.
- Bishop (1994), *Mixture Density Networks* — cabeza de mixtura, base de §5.3(b).
- Bengio et al. (2015), *Scheduled Sampling for Sequence Prediction with RNNs* — mitigación del exposure bias.
- Theis, van den Oord & Bethge (2016), *A note on the evaluation of generative models* — por qué una NLL baja no garantiza buenas muestras.
