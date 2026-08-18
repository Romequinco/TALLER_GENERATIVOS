# Modelos de difusión

Material de clase: `docs/material_clase/slides/Diffusion_Models_DM_2026.pdf` (Valero Laparra, 33 diapositivas), `docs/material_clase/notebooks/5_Difussion_Models_Keras_DDIM_Jordi_mod.ipynb` (implementación DDIM continua en Keras, fuente principal) y `docs/material_clase/notebooks/6_HF_stable_diffusion_VAL.ipynb` (difusión latente y CFG a escala industrial). Detalle de citas en la sección 11.

## 1. Intuición: destruir con ruido y aprender a deshacerlo

El hilo conductor de clase (diapositivas 2-5, «DIFFUSION ROADMAP») es corto y vale la pena tomarlo en serio porque explica *por qué* la difusión funciona sin trucos adversarios:

```
Probabilidad → Aumentar probabilidad
Autoencoder → Denoising Autoencoder → Aumentar probabilidad
= Diffusion Models
```

La cadena de razonamiento es la siguiente:

1. **Un *denoiser* óptimo en mínimos cuadrados no es un objeto arbitrario.** Dado $y = x + \sigma\varepsilon$ con $\varepsilon\sim\mathcal N(0,I)$, el estimador que minimiza $\mathbb E\|x-\hat x(y)\|^2$ es la media posterior $\hat x(y)=\mathbb E[x\mid y]$, y satisface la identidad de Miyasawa (1961) (diapositiva 10):

   $$\hat x(y) \;=\; y + \sigma^2\,\nabla_y \log p_\sigma(y)$$

   Es decir: **el residuo de un denoiser entrenado con MSE es, salvo escala, el gradiente de la log-densidad de los datos ruidosos** (el *score*). Entrenar un denoiser es, sin pretenderlo, estimar $\nabla\log p$.

2. **Si tengo el gradiente de la log-densidad, puedo subir por él para generar muestras.** Diapositiva 11: «aplicar denoising paso a paso» + «se mete ruido para no caer en máximo local (Langevin dynamics)». Sin el ruido inyectado, el ascenso por gradiente convergería a modas y devolvería siempre lo mismo; con él se obtiene un muestreador de $p$, no un maximizador.

3. **Un denoiser a un solo nivel de ruido no basta** (diapositiva 12: «Un paso» vs «Varios pasos»). Con $\sigma$ pequeño el score sólo es fiable cerca del *manifold* de datos; con $\sigma$ grande la densidad es casi gaussiana y no informa del detalle. La solución es entrenar **un único denoiser condicionado al nivel de ruido** y recorrer $\sigma$ de grande a pequeño.

Eso es exactamente un modelo de difusión (diapositivas 18-20, DDPM, Ho et al. 2020). Frente al resto de generativos del taller la diferencia práctica es enorme: la GAN optimiza un *minimax* con objetivo móvil (loss no interpretable), el VAE un ELBO con dos términos que compiten, el flow una verosimilitud exacta a costa de restringir la arquitectura. La difusión, en cambio, entrena una **regresión supervisada corriente**: dado un dato contaminado y el nivel de contaminación, predecir el ruido que se le añadió. El objetivo (`eps`) es conocido y fijo, no lo produce otra red que a su vez está aprendiendo. De aquí sale la propiedad que explotamos en la sección 7: **la curva de loss baja de forma limpia y monótona**, cosa que ni la GAN ni (a veces) el VAE ofrecen.

El precio es que el muestreo es iterativo y caro. La sección 8 lo cuantifica.

## 2. Proceso directo (forward): cadena de ruido y forma cerrada

El proceso directo es una cadena de Markov **fija, sin parámetros aprendidos**, que degrada progresivamente el dato hasta ruido blanco. Diapositiva 22: «De la imagen al ruido es fácil (conozco la imagen)».

Con $T$ pasos y una planificación $\{\beta_t\}_{t=1}^T$ con $0<\beta_t<1$:

$$q(x_t \mid x_{t-1}) \;=\; \mathcal N\!\left(x_t;\ \sqrt{1-\beta_t}\,x_{t-1},\ \beta_t I\right)$$

El factor $\sqrt{1-\beta_t}$ en la media es lo que mantiene la varianza acotada: si $x_{t-1}$ tiene varianza 1, entonces $x_t$ también, porque $(1-\beta_t)\cdot 1 + \beta_t = 1$. Esto obliga a **normalizar los datos a media 0 y varianza 1** antes de entrenar (el notebook lo hace con `layers.Normalization()` y lo justifica en las conclusiones: «aquí tiene más sentido normalizar para obtener imágenes con media 0 y varianza 1, igual que los ruidos añadidos»).

**Forma cerrada.** La propiedad clave, y la razón por la que la difusión es entrenable: **no hace falta simular la cadena**. Definiendo $\alpha_t = 1-\beta_t$ y $\bar\alpha_t = \prod_{s=1}^{t}\alpha_s$, la composición de gaussianas da directamente

$$\boxed{\;q(x_t\mid x_0)=\mathcal N\!\left(x_t;\ \sqrt{\bar\alpha_t}\,x_0,\ (1-\bar\alpha_t)I\right)\quad\Longleftrightarrow\quad x_t=\sqrt{\bar\alpha_t}\,x_0+\sqrt{1-\bar\alpha_t}\,\varepsilon\;}$$

con $\varepsilon\sim\mathcal N(0,I)$. Se salta a cualquier $t$ en una línea de código, lo que convierte el entrenamiento en un muestreo i.i.d. de tripletas $(x_0, t, \varepsilon)$. Se cumple $\bar\alpha_0\approx 1$ (dato limpio) y $\bar\alpha_T\approx 0$ (ruido puro), de modo que $q(x_T)\approx\mathcal N(0,I)$: el punto de partida del muestreo es tratable.

**Reparametrización señal/ruido del notebook.** El notebook de clase usa una versión **continua** con $t\in[0,1]$ y una notación equivalente pero más legible (celda 13):

$$x_t \;=\; \underbrace{s(t)}_{\texttt{signal\_rate}}\,x_0 \;+\; \underbrace{n(t)}_{\texttt{noise\_rate}}\,\varepsilon, \qquad s(t)^2+n(t)^2=1$$

con $s(t)=\sqrt{\bar\alpha}$ y $n(t)=\sqrt{1-\bar\alpha}$ («En el artículo DDIM estas se corresponden con $\sqrt{\alpha}$ y $\sqrt{1-\alpha}$»). La restricción $s^2+n^2=1$ garantiza varianza unidad en todo el recorrido: en términos de procesado de señal, **potencia constante**, sólo cambia el reparto entre señal y ruido. La relación señal-ruido $\mathrm{SNR}(t) = \bar\alpha_t/(1-\bar\alpha_t)$ es la variable que gobierna la dificultad de la tarea en cada $t$ y la que hay que mirar al diseñar el schedule (sección 4).

## 3. Proceso inverso: parametrización y objetivo de entrenamiento

El proceso inverso $q(x_{t-1}\mid x_t)$ es intratable, pero para $\beta_t$ pequeño es aproximadamente gaussiano, así que se modela como

$$p_\theta(x_{t-1}\mid x_t)=\mathcal N\!\left(x_{t-1};\ \mu_\theta(x_t,t),\ \sigma_t^2 I\right)$$

Maximizar la verosimilitud por ELBO conduce, tras álgebra, a una suma de términos KL entre $p_\theta$ y el posterior exacto $q(x_{t-1}\mid x_t,x_0)$, que sí es gaussiano y conocido.

**Las tres parametrizaciones.** La red puede predecir tres cosas equivalentes entre sí, ligadas por la forma cerrada:

| Parametrización | Salida | Conversión |
|---|---|---|
| $\varepsilon$-pred (DDPM, estándar) | $\varepsilon_\theta(x_t,t)$ | $\hat x_0=(x_t-\sqrt{1-\bar\alpha_t}\,\varepsilon_\theta)/\sqrt{\bar\alpha_t}$ |
| $x_0$-pred | $\hat x_{0,\theta}(x_t,t)$ | $\hat\varepsilon=(x_t-\sqrt{\bar\alpha_t}\hat x_0)/\sqrt{1-\bar\alpha_t}$ |
| $v$-pred | $v=\sqrt{\bar\alpha_t}\varepsilon-\sqrt{1-\bar\alpha_t}x_0$ | combinación de las dos |

La opción por defecto es **$\varepsilon$-pred** y la razón es puramente numérica: el objetivo $\varepsilon\sim\mathcal N(0,I)$ tiene **escala constante 1 para todo $t$**, mientras que $x_0$ tiene la escala de los datos y su predicción es trivial cuando $t$ es pequeño e imposible cuando $t$ es grande. Con $\varepsilon$-pred el problema está bien condicionado en todo el rango y, como veremos, la loss tiene un valor de referencia interpretable.

> Ampliación (no cubierto en clase): la parametrización $v$ («velocity») se menciona de pasada en el notebook (celda 28, *«more network output types: predicting image or velocity instead of noise»*, Salimans & Ho 2022) pero no se implementa. Es preferible cuando se usan schedules con SNR terminal cero o muy pocos pasos de muestreo; para nuestro caso $\varepsilon$-pred es suficiente.

**Media del proceso inverso.** Con $\varepsilon$-pred,

$$\mu_\theta(x_t,t)=\frac{1}{\sqrt{\alpha_t}}\left(x_t-\frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\,\varepsilon_\theta(x_t,t)\right)$$

y la varianza se fija (no se aprende) a $\sigma_t^2=\beta_t$ o $\sigma_t^2=\tilde\beta_t=\frac{1-\bar\alpha_{t-1}}{1-\bar\alpha_t}\beta_t$.

**Objetivo simplificado.** El ELBO completo lleva pesos $\lambda_t$ por término. La aportación práctica de DDPM fue observar que **descartar esos pesos** (equivalente a ponderar cada $t$ igual) funciona mejor:

$$\boxed{\;\mathcal L_{\text{simple}}=\mathbb E_{x_0,\;t\sim\mathcal U\{1..T\},\;\varepsilon\sim\mathcal N(0,I)}\Big[\big\|\varepsilon-\varepsilon_\theta\big(\sqrt{\bar\alpha_t}x_0+\sqrt{1-\bar\alpha_t}\varepsilon,\ t\big)\big\|^2\Big]\;}$$

Esto es todo el entrenamiento. Un MSE. El bucle completo cabe en seis líneas (sección 10).

El notebook (celda 16) añade un matiz operativo importante: mantiene **dos métricas**, `n_loss` (error sobre el ruido) e `i_loss` (error sobre la imagen reconstruida), y subraya que **sólo `n_loss` se usa para entrenar**; `i_loss` es diagnóstico. Reproduciremos esa separación porque es útil (sección 7).

Sobre MSE vs MAE, las conclusiones del notebook son explícitas: *«Normalmente se usa MSE, que genera imágenes más diversas, pero "alucina" más. MAE genera imágenes más "suaves". Lo mejor es probar con ambos y comparar.»* Para datos financieros esto no es una anécdota estética: MAE suaviza, y suavizar retornos significa **matar las colas**, que es justo lo que queremos preservar. Usaremos MSE.

## 4. Planificación del ruido (noise schedule) y su impacto

Diapositiva 23: «Parametrización del *schedule*». El schedule es la elección de diseño con más impacto después de la arquitectura, y la que más se subestima.

**Schedule lineal (DDPM original).** $\beta_t$ crece linealmente de $10^{-4}$ a $0.02$ con $T=1000$. Problema conocido: destruye la información **demasiado pronto**; buena parte de los pasos finales trabajan sobre datos ya indistinguibles de ruido y no aportan gradiente útil.

**Schedule coseno.** Nichol & Dhariwal (2021), citado en la celda 13 del notebook como *«versión simplificada y continua de un [cosine scheduler (Section 3.2)]»*:

$$\bar\alpha_t=\frac{f(t)}{f(0)},\qquad f(t)=\cos^2\!\left(\frac{t/T+s}{1+s}\cdot\frac{\pi}{2}\right),\quad s=0.008$$

Mantiene señal durante más tiempo en la parte central del recorrido y degrada suavemente en los extremos.

**La versión continua del notebook**, que es la que adoptaremos, porque **desacopla el schedule de entrenamiento del de muestreo**:

```python
def diffusion_schedule(t, min_signal_rate=0.02, max_signal_rate=0.95):
    """t en [0,1] -> (noise_rate, signal_rate), con noise^2 + signal^2 = 1."""
    start_angle = np.arccos(max_signal_rate)   # t=0 -> casi toda señal
    end_angle   = np.arccos(min_signal_rate)   # t=1 -> casi todo ruido
    angles = start_angle + t * (end_angle - start_angle)
    return np.sin(angles), np.cos(angles)      # noise_rate, signal_rate
```

Convertir el tiempo en un ángulo entre dos límites y devolver seno y coseno garantiza la restricción $s^2+n^2=1$ por construcción. Dos observaciones del notebook que importan:

- **Los límites son hiperparámetros sensibles.** Conclusiones: *«El `min_signal_rate` es importante, si es muy bajo las imágenes salen sobresaturadas, si es muy alto al revés. Si se pone a 0 se produce un error de división por cero.»* La división por cero viene de $\hat x_0=(x_t - n\,\varepsilon_\theta)/s$: con $s\to 0$ el estimador explota.
- **La red recibe el nivel de ruido, no el índice temporal.** Celda 9: *«Los modelos de difusión usan un índice temporal en lugar de la varianza del ruido. Aquí usamos lo segundo (...) Esto permite cambiar el sampling schedule en inferencia sin tener que reentrenar la red.»* Esto es una ventaja operativa de primer orden para nosotros: entrenamos una vez y podemos barrer el número de pasos de muestreo gratis.

**Embedding sinusoidal del nivel de ruido.** El nivel de ruido entra a la red mediante una codificación en frecuencia, análoga a la codificación posicional de los transformers (celda 9): `emb = concat([sin(2*pi*f*x), cos(2*pi*f*x)])` con `f = exp(linspace(log(f_min), log(f_max), d_emb//2))`. No es un adorno: *«Esto es crucial para obtener un buen rendimiento»* (conclusiones). Una red que recibe $t$ como escalar plano es poco sensible a variaciones finas del nivel de ruido y funciona mal. El notebook usa `embedding_dims = 32`, `embedding_max_frequency = 1000.0` y recomienda como frecuencia mínima la inversa del rango de la entrada.

**Impacto en datos de baja dimensión intrínseca.**

> Ampliación (no cubierto en clase): los schedules estándar se calibraron para imágenes de $64\times64\times3$ (12.288 dimensiones). En dimensión efectiva mucho más baja —y un panel financiero de 60×18, aunque tenga 1.080 números, tiene una dimensión intrínseca muy inferior por la fuerte correlación entre sectores y días— el mismo schedule **destruye la señal demasiado tarde**: en pasos donde para una imagen ya no quedaría información, en nuestro panel todavía se puede reconstruir casi todo. El síntoma es una loss engañosamente baja acompañada de muestras poco diversas. El remedio habitual es reescalar la entrada por un factor $<1$ (equivale a adelantar el schedule) o desplazar el schedule hacia SNR más bajos. Merece la pena tratarlo como hiperparámetro y verificar visualmente que a $t=1$ el dato ruidoso es indistinguible de $\mathcal N(0,I)$.

## 5. Muestreo: DDPM vs DDIM, número de pasos

**DDPM: muestreo ancestral.** Se recorre la cadena completa, paso a paso, inyectando ruido en cada uno:

$$x_{t-1}=\frac{1}{\sqrt{\alpha_t}}\left(x_t-\frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\varepsilon_\theta(x_t,t)\right)+\sigma_t z,\qquad z\sim\mathcal N(0,I)$$

Es **estocástico** y requiere los $T$ pasos con los que se entrenó (típicamente 1000). Coste: 1000 pasadas forward de la red por cada lote de muestras.

**DDIM: proceso no markoviano.** DDIM (Song et al. 2021; diapositiva 24: *«Mete el "tiempo" (alpha) como variable»*) define una familia de procesos inversos **no markovianos** que comparten exactamente el mismo marginal $q(x_t\mid x_0)$ y, por tanto, **el mismo modelo entrenado**. No hay que reentrenar nada. El paso genérico, sobre una subsecuencia arbitraria de tiempos $\tau_1<\dots<\tau_S$:

$$x_{\tau_{i-1}}=\sqrt{\bar\alpha_{\tau_{i-1}}}\;\underbrace{\left(\frac{x_{\tau_i}-\sqrt{1-\bar\alpha_{\tau_i}}\,\varepsilon_\theta(x_{\tau_i},\tau_i)}{\sqrt{\bar\alpha_{\tau_i}}}\right)}_{\hat x_0\ \text{(predicción del dato limpio)}}+\underbrace{\sqrt{1-\bar\alpha_{\tau_{i-1}}-\sigma^2}\;\varepsilon_\theta(x_{\tau_i},\tau_i)}_{\text{dirección hacia }x_{\tau_{i-1}}}+\;\sigma z$$

con $\sigma=\eta\sqrt{\tfrac{1-\bar\alpha_{\tau_{i-1}}}{1-\bar\alpha_{\tau_i}}}\sqrt{1-\tfrac{\bar\alpha_{\tau_i}}{\bar\alpha_{\tau_{i-1}}}}$. Dos casos:

- $\eta=1$ → se recupera DDPM.
- $\eta=0$ → **muestreo determinista**: dado el ruido inicial, la muestra queda unívocamente determinada. Es el que implementa el notebook de clase.

El código de clase (celda 17, `reverse_diffusion`) es DDIM determinista escrito en la parametrización señal/ruido, y resulta transparente:

```python
# separar la señal ruidosa en sus dos componentes
pred_noises, pred_images = self.denoise(noisy_images, noise_rates, signal_rates, ...)
# recombinar con los ratios del SIGUIENTE paso  <-- esto es DDIM con eta=0
next_noisy = next_signal_rates * pred_images + next_noise_rates * pred_noises
```

En cada paso: descomponer, y volver a mezclar con los coeficientes del siguiente instante. Nada más.

**Número de pasos: qué se gana y qué se pierde.** El notebook usa `plot_diffusion_steps = 20` por defecto y muestra en resultados imágenes generadas «usando entre 1 y 20 pasos de muestreo partiendo desde el mismo ruido».

| Pasos | Coste | Efecto |
|---|---|---|
| 1 | mínimo | La muestra es $\hat x_0$ directo: la media condicional. Sobre-suavizado severo. |
| 10-20 | muy bajo | Ya reconocible. Detalle fino y colas todavía comprometidos. |
| 50 | bajo | Buen compromiso; en la práctica indistinguible de 200 para datos tabulares. |
| 200-1000 | alto | Rendimientos decrecientes. Necesario sólo con $\eta>0$. |

Lo que se **pierde** al bajar de pasos:

1. **Error de discretización.** DDIM con $\eta=0$ es un integrador de Euler de primer orden sobre una EDO. Menos pasos = paso de integración mayor = sesgo sistemático en la trayectoria, que se traduce en un desplazamiento de la distribución generada (típicamente hacia menos varianza).
2. **Diversidad.** Con $\eta=0$ el mapa ruido→muestra es determinista y suave: la variabilidad proviene *sólo* del ruido inicial. Con $\eta=1$ hay inyección adicional de ruido en cada paso y mayor diversidad, pero se necesitan muchos más pasos.
3. **Colas.** Los eventos extremos se construyen en los últimos pasos, con SNR alto. Cortar pasos ahí es lo que produce el aplanamiento de la curtosis (sección 8).

> Ampliación (no cubierto en clase): existen *solvers* de orden superior (Heun, DPM-Solver, PNDM) que reducen el error de discretización a igualdad de evaluaciones de la red. El notebook 6 (celdas 34 y 42) los menciona y usa `LMSDiscreteScheduler` en vez del PNDM por defecto, señalando que DPM-Solver «is able to achieve great quality in less steps... try with 25 instead of the default 50». Para nuestro caso DDIM-Euler basta; no justifica la complejidad extra.

## 6. Condicionamiento (embedding de clase, classifier-free guidance)

Diapositiva 26 enumera las tres vías vistas en clase para el modelo condicional:

- Con un clasificador que guíe el entrenamiento: $f(y\mid x_t)$ — *classifier guidance*.
- Entrenando directamente en pares $(x,y)$.
- Condicionado a texto vía CLIP.

**Condicionamiento directo por embedding.** La forma más simple, y la que usaremos: la red pasa de $\varepsilon_\theta(x_t,t)$ a $\varepsilon_\theta(x_t,t,c)$, con $c$ la etiqueta de clase. Se implementa con una tabla de embeddings del mismo ancho que el embedding temporal, y **se suman**: `h = proyeccion_entrada(x) + mlp_tiempo(emb_sinusoidal(t)) + emb_clase(c)`. Es barato (una tabla de $K\times H$ parámetros) y suficiente cuando el número de clases es pequeño, que es nuestro caso (3 regímenes).

**Classifier guidance.** Requiere entrenar un clasificador auxiliar $p_\phi(y\mid x_t)$ **sobre datos ruidosos** (a todos los niveles de $t$) y desplazar el score en muestreo: $\tilde\varepsilon_\theta = \varepsilon_\theta(x_t,t) - w\sqrt{1-\bar\alpha_t}\,\nabla_{x_t}\log p_\phi(y\mid x_t)$. Funciona, pero obliga a mantener y entrenar un segundo modelo sobre entradas ruidosas. En un presupuesto de CPU no compensa.

**Classifier-free guidance (CFG).** La alternativa que se impuso, y la que aparece implementada literalmente en el notebook 6 (celda 61):

```python
noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
```

Mecánica completa:

1. **En entrenamiento**: con probabilidad $p_{\text{uncond}}\approx 0.1$ se sustituye la etiqueta por un **token nulo** $\varnothing$ (por eso la tabla de embeddings tiene $K+1$ entradas). La misma red aprende así el modelo condicional y el incondicional.
2. **En muestreo**: dos pasadas (o una con el lote duplicado, como hace el notebook 6 en la celda 51-52 «we can concatenate both into a single batch to avoid doing two forward passes») y extrapolación:

$$\tilde\varepsilon = \varepsilon_\theta(x_t,t,\varnothing) + w\big(\varepsilon_\theta(x_t,t,c)-\varepsilon_\theta(x_t,t,\varnothing)\big)$$

- $w=0$ → incondicional.
- $w=1$ → condicional puro, sin guía.
- $w>1$ → se exagera la diferencia entre condicional e incondicional: **más adherencia a la clase, menos diversidad**.

Sobre el valor de $w$, el notebook 6 (celda 16) es directo: *«Numbers like 7 or 8.5 give good results, if you use a very large number the images might look good, but will be less diverse.»*

**Advertencia para nuestro caso**: $w\approx 7.5$ es un valor calibrado para texto→imagen, donde la condición es altísimamente informativa y el espacio enorme. Con 3 clases y datos tabulares, valores altos de $w$ **colapsan la diversidad**, y precisamente en la clase minoritaria («crisis», ~10% de los datos) el colapso es catastrófico: se generarían 5.000 copias casi idénticas de las pocas crisis del conjunto de entrenamiento. Rango razonable a barrer: $w\in\{1.0,\,1.5,\,2.0,\,3.0\}$, midiendo diversidad explícitamente (sección 7), no sólo pureza de clase.

CFG cuesta el **doble** de forward passes en muestreo. Es el único sitio donde importa; el entrenamiento no se encarece.

## 7. Diagnóstico de convergencia

**Esta es la sección que resuelve el requisito del enunciado** (`docs/enunciado/Taller_B5_T1.pdf`, apartado 5): *«Para cada entrenamiento, incluir las curvas de loss donde se vea que el modelo ha convergido.»*

De los siete generadores del taller, la difusión es **el que mejor cumple ese requisito**. Conviene aprovecharlo y presentarlo como tal.

### 7.1 Por qué la loss de difusión sí converge de forma limpia

Tres razones, todas estructurales:

1. **El objetivo es fijo.** $\varepsilon$ es un vector muestreado de $\mathcal N(0,I)$, no la salida de otra red que está aprendiendo simultáneamente. No hay equilibrio de Nash que perseguir, no hay *moving target*. Es exactamente el mismo régimen de optimización que una regresión supervisada.
2. **Es un único término.** No hay reconstrucción compitiendo con KL como en el VAE; no hay que interpretar si una subida de loss del generador es buena o mala como en la GAN. La loss baja $\Rightarrow$ el modelo mejora en la tarea que se le pide.
3. **El problema está bien condicionado en todo $t$.** Gracias a $\varepsilon$-pred, el objetivo tiene escala 1 uniformemente, sin importar el nivel de ruido.

Consecuencia práctica: se espera una curva **monótona decreciente, con ruido de muestreo pero sin oscilaciones estructurales**, saturando en una asíntota. Si tu curva no tiene esa forma, algo está mal en la implementación (normalización, schedule, learning rate), no en el método.

### 7.2 El ancla numérica: la loss arranca exactamente en 1

Esta es la propiedad más útil del método y está señalada en el notebook (celda 9):

> *«El kernel de la última capa de convolución se inicializa a cero para que la red prediga ceros al principio, que es el valor medio. Esto mejora el entrenamiento al inicio y **hace que el MSE valga justo 1 al principio**.»*

La razón es inmediata: si $\varepsilon_\theta\equiv 0$, entonces $\mathcal L=\mathbb E\|\varepsilon\|^2/D = 1$ porque $\varepsilon\sim\mathcal N(0,I)$.

**Esto te da una escala absoluta y comparable entre ejecuciones**, algo que no tiene ningún otro generador del taller:

| Loss (MSE, por dimensión) | Lectura |
|---|---|
| $\approx 1.00$ | La red no ha aprendido nada (predice la media). Es el punto de partida. |
| $> 1$ de forma sostenida | Divergencia. Learning rate demasiado alto o normalización rota. |
| $0.3 - 0.6$ | Aprendiendo, lejos de converger. |
| $0.05 - 0.25$ | Rango típico de convergencia con schedule coseno y $T=1000$. |
| $< 0.01$ | Sospechoso: o el schedule deja demasiada señal en $t$ altos, o hay memorización. |

Inicializa la última capa a cero (`kernel_initializer="zeros"` en Keras, `nn.init.zeros_` en torch) para que ese 1.0 inicial aparezca en tu gráfica. Es medio segundo de trabajo y convierte la curva en autoexplicativa.

Nota si usas MAE en lugar de MSE (como hace el notebook con flores): la referencia inicial es $\mathbb E|\varepsilon| = \sqrt{2/\pi}\approx 0.798$, no 1.

### 7.3 La loss NO puede llegar a cero, y eso es correcto

Existe un **suelo irreducible**. Para $t$ grande, $x_t$ es casi ruido puro y la información sobre $\varepsilon$ que contiene es mínima: el mejor predictor posible sigue cometiendo un error grande. Formalmente, el error óptimo es la varianza posterior $\mathbb E\|\varepsilon-\mathbb E[\varepsilon\mid x_t]\|^2$, que tiende a $1$ cuando $\bar\alpha_t\to 0$. Por tanto **el valor absoluto de la loss no es interpretable en sí mismo**: es una media sobre todo el rango de $t$, dominada por los $t$ donde la tarea es casi imposible. Un modelo perfecto tampoco daría 0.

### 7.4 El diagnóstico que de verdad informa: loss estratificada por $t$

Promediar sobre $t$ oculta información. **Descompón la loss en bandas de $t$** —por ejemplo `[(0,.2),(.2,.4),(.4,.6),(.6,.8),(.8,1.)]`, con $\varepsilon$ y $t$ fijos— y grafícala. Perfil esperado en un modelo sano: en $t\in[0,0.2]$ (poco ruido) loss **muy baja**, $\lesssim 0.05$; transición suave en los $t$ intermedios; en $t\in[0.8,1.0]$ (casi ruido puro) loss **cercana a 1**, porque la tarea es casi imposible y debe serlo.

Patologías que sólo se ven aquí:

- **Loss baja también en $t$ altos** → el schedule no destruye lo suficiente; a $t=1$ todavía queda señal reconocible. El muestreo partirá de $\mathcal N(0,I)$, que el modelo nunca vio, y generará basura pese a una loss excelente. Es el fallo silencioso más común en datos de baja dimensión intrínseca.
- **Loss alta en $t$ bajos** → la red no tiene capacidad para el detalle fino, o el embedding de nivel de ruido no está funcionando (revisa las frecuencias del embedding sinusoidal).

### 7.5 Cómo quitarle el ruido a la curva de validación

La loss de entrenamiento es intrínsecamente ruidosa porque en cada lote se muestrean $t$ y $\varepsilon$ al azar. Buena parte de la varianza visible en la curva **no es del modelo, es del estimador**. Solución obligatoria para la figura de la entrega:

```python
# Conjunto de validacion FIJO: mismas ventanas, mismos t (estratificados) y mismo eps.
# Reduce la varianza del estimador ~10x y hace la curva legible.
rng = torch.Generator().manual_seed(1234)
t_val   = torch.linspace(0, 1, len(X_val))          # t estratificado, no aleatorio
eps_val = torch.randn(X_val.shape, generator=rng)   # ruido congelado
```

Con esto la curva de validación queda suave y la convergencia es visualmente incontestable. Sin esto, con lotes pequeños en CPU, la curva parece ruido y no demuestra nada. Como complemento, aplicar **media móvil exponencial a la loss** de entrenamiento ($\beta\approx 0.98$) y **EMA a los pesos** (`ema = 0.999` en el notebook, celda 6): la red que se usa en inferencia es la promediada, tal como explica la celda 14, lo que suaviza el entrenamiento y mejora las muestras sin coste apreciable.

### 7.6 Criterios operativos de parada

1. **Umbral relativo**: la loss de validación estratificada mejora menos de un **1% en 50 épocas**.
2. **Forma**: la curva ha entrado visiblemente en meseta, no sólo bajado.
3. **Separación train/val**: si la validación se despega al alza mientras el entrenamiento sigue bajando, hay **memorización** (riesgo alto en nuestro caso, sección 8). Parar y reducir capacidad.
4. **Perfil por bandas de $t$** estabilizado (7.4).
5. **Métricas de muestra** en meseta (7.7) — este es el criterio que manda.

### 7.7 Por qué una loss baja NO garantiza buenas muestras

Es el punto crítico y hay que decirlo explícitamente en la entrega. Razones concretas:

1. **La loss mide un paso; el muestreo encadena decenas.** El error de denoising de un solo paso se compone a lo largo de la trayectoria. Un sesgo pequeño y sistemático, invisible en el MSE, puede acumularse y desplazar la distribución generada.
2. **Desajuste de exposición (*exposure bias*).** Durante el entrenamiento la red siempre ve $x_t$ construidos a partir de datos **reales**. Durante el muestreo ve $x_t$ construidos a partir de sus **propias predicciones**, que están ligeramente fuera de la distribución vista. La loss no mide esto en absoluto.
3. **La loss está dominada por los $t$ que menos importan perceptualmente.** El grueso de la media viene de los $t$ altos, donde la tarea es casi imposible; la calidad estadística de la muestra se decide en los $t$ bajos, que aportan poco a la media.
4. **La loss no mide diversidad.** Un modelo que memoriza el conjunto de entrenamiento tiene una loss excelente. Un modelo con CFG demasiado agresivo produce muestras casi idénticas y su loss de entrenamiento no cambia (CFG sólo actúa en muestreo).
5. **Hiperparámetros de muestreo no aparecen en la loss.** El notebook lo dice sin ambages en las conclusiones: con `min_signal_rate` mal ajustado *«las imágenes salen sobresaturadas»* — y la loss de entrenamiento es exactamente la misma, porque el problema está en el muestreo.

### 7.8 Métricas de muestra a lo largo del entrenamiento (obligatorio)

> Ampliación (no cubierto en clase): el notebook evalúa la calidad mirando las imágenes generadas en cada época (`LambdaCallback(on_epoch_end=model.plot_images)`, celda 22) y menciona el KID, comentado en el código por coste. Para datos financieros no hay inspección visual equivalente, así que hay que sustituir el ojo por métricas cuantitativas. Las siguientes son las adecuadas para nuestro panel.

**Protocolo**: cada 25 épocas, generar 512 muestras con DDIM-50 (coste medido: ~1-3 s, sección 10) y registrar:

| Métrica | Qué detecta | Valor objetivo |
|---|---|---|
| Media y desviación por canal vs real | Sesgo de escala, sobre-suavizado | error relativo < 5% |
| **Curtosis** de los retornos generados | Colas aplastadas (patología nº 1 aquí) | ≥ 70% de la real |
| ACF de retornos (lags 1-10) | Estructura temporal espuria | ≈ 0, como en el real |
| **ACF de $|$retornos$|$** | Clustering de volatilidad | debe reproducir el decaimiento lento real |
| Error de Frobenius de la matriz de correlación entre los 18 canales | Estructura sectorial | mínimo |
| **AUC de un discriminador post-hoc** (GBM real vs sintético) | Realismo global, en un número | **0.5 = indistinguible; > 0.8 = malo** |
| Distancia al vecino más cercano real | **Memorización** | debe ser comparable a la distancia real-real |
| **TSTR** (entrenar en sintético, testear en real) | Utilidad, que es el objetivo del taller | maximizar |

**La figura que hay que entregar** es un panel de dos ejes: eje X = época; eje Y izquierdo = loss de validación estratificada; eje Y derecho = AUC del discriminador y curtosis relativa. El mensaje que debe transmitir: *la loss converge limpiamente **y** las métricas de muestra se estabilizan en el mismo punto*. Si convergen a la vez, la convergencia está demostrada de verdad. Si la loss satura pero la AUC del discriminador sigue en 0.95, la conclusión es que el problema está en el muestreo o el schedule, no en el entrenamiento — y eso también es un resultado presentable.

## 8. Patologías y coste computacional

**8.1 Patologías específicas de datos financieros**

- **Sobre-suavizado y colas aplastadas.** El $\varepsilon$-denoiser aproxima una media condicional. Con pocos pasos de muestreo, la muestra se acerca a $\hat x_0$, que es literalmente esa media, y los retornos generados tienen **curtosis muy inferior a la real**. Para un problema donde la clase minoritaria son crisis, esto ataca justo lo que queremos generar. Mitigaciones: MSE (no MAE), suficientes pasos de muestreo (≥ 50), evitar $w$ de CFG alto, y **medir la curtosis explícitamente** (7.8).
- **Memorización.** Con ~6.000 ventanas de 60 días **solapadas**, el número de sucesos de mercado independientes es de unas pocas decenas. Un modelo de 2,5M de parámetros puede memorizar sin dificultad. En la clase «crisis» (~10%, ~600 ventanas, y en la práctica 3-4 episodios históricos) el riesgo es máximo. Métrica de control: distancia al vecino más cercano real (7.8). Mitigaciones: reducir capacidad, *weight decay* (el notebook usa AdamW con `weight_decay=1e-4` y lo justifica: *«hace más estable el entrenamiento»*), y no exprimir hasta la última décima de loss.
- **Colapso de diversidad por CFG.** Ver sección 6. Es un fallo de muestreo, invisible en la loss.
- **Escalas heterogéneas.** Los 18 canales mezclan retornos ($\sigma\sim 1\%$ diario), VIX (nivel ~15-80), MOVE, *spreads* de crédito y pendiente de curva. Sin normalización **por canal**, el ruido gaussiano isótropo destruye antes los canales de menor varianza: el modelo entrena de facto con schedules distintos por canal. **Obligatorio**: z-score por canal con estadísticos calculados **sólo en train**.
- **Fuga temporal.** Ventanas solapadas + etiqueta a 21 días vista. Un *shuffle* aleatorio pone en validación ventanas que comparten días con las de entrenamiento y la loss de validación queda inflada de optimismo.
- **Inestabilidad numérica.** $\hat x_0=(x_t-n\varepsilon_\theta)/s$ explota si $s\to 0$. De ahí `min_signal_rate = 0.02` en el notebook. No lo bajes a 0.

**8.2 Coste computacional.** El coste de **entrenamiento** es comparable al de cualquier regresor: una pasada forward+backward por lote. El coste de **muestreo** es el problema: $S$ evaluaciones completas de la red por muestra, $\times 2$ si se usa CFG. Medido en la máquina objetivo (Intel Skylake móvil, 2 núcleos físicos / 4 hilos, torch 2.11.0+cpu, sin CUDA), MLP de 0,92M parámetros, lote de 512 muestras:

| Muestreador | Pasos | Coste por lote de 512 | 30.000 muestras | 30.000 con CFG |
|---|---|---|---|---|
| **DDIM** | 20 | ~0,4 s | ~25 s | ~50 s |
| **DDIM** | 50 | ~1,0 s | ~1 min | ~2 min |
| DDPM | 1000 | ~20 s | **~20 min** | **~40 min** |

Y este coste **se paga cada vez**: por cada punto de la curva de diagnóstico, por cada valor de $w$ del barrido, por cada proporción de mezcla real/sintético que exige el enunciado. Con DDPM-1000 el ciclo de iteración se rompe.

**Conclusión operativa: DDIM con 20-50 pasos no es una optimización, es un requisito de viabilidad del taller en CPU.** Y es gratis, porque el mismo modelo entrenado sirve para ambos muestreadores.

Aviso de hardware medido: en un portátil de 2 núcleos, bajo carga sostenida aparece *throttling* térmico. En los benchmarks se observaron épocas puntuales 3-8× más lentas que la mediana. **Multiplica cualquier estimación por 1,5-2 para planificar tiempo de pared real** y ejecuta los entrenamientos largos sin nada más en la máquina.

## 9. Aplicación a nuestro problema

### 9.1 Qué se genera exactamente

El objeto a modelar es el bloque conjunto $[\,X\;;\;y_{\text{reg}}\;;\;y_{\text{vol}}\,]$, con $X$ de forma $(60,18)$ (60 días × ~18 canales del panel híbrido: S&P500, 9 SPDR sectoriales, VIX, MOVE, *spreads* de crédito, pendiente de curva, *drawdown*, volatilidad realizada).

**Decisión recomendada: no generar $y_{\text{reg}}$; condicionarse a ella.**

$$\text{difusión sobre } [\,X_{\text{aplanado}} \;;\; y_{\text{vol}}\,] \in \mathbb R^{1081}, \quad \text{condicionada a } y_{\text{reg}}\in\{0,1,2\}$$

1. $y_{\text{reg}}$ es categórica. La difusión gaussiana modela variables continuas; meter una etiqueta discreta en el vector la trata como continua y luego hay que redondear, con etiquetas ambiguas como resultado.
2. **Resuelve el desbalance por construcción.** Si la clase es una condición y no una salida, se puede muestrear la clase «crisis» en la proporción que se quiera: 10%, 33% o 100%. Ese es precisamente el uso que el enunciado busca para los datos sintéticos.
3. La etiqueta de cada muestra sintética es exacta por definición, sin necesidad de un clasificador que la infiera.

Con CFG, la tabla de embeddings tiene 4 entradas: 3 regímenes + token nulo $\varnothing$.

### 9.2 Preprocesado

- **Z-score por canal**, con $\mu,\sigma$ calculados **sólo sobre el tramo de entrenamiento**. Guardar los estadísticos para desnormalizar al generar.
- **No winsorizar** antes de entrenar: eliminaría exactamente los eventos extremos que queremos que el modelo aprenda a generar.
- **Partición cronológica**, no aleatoria: train / val / test en bloques temporales.

> Ampliación (no cubierto en clase): dado que las ventanas de 60 días se solapan y la etiqueta mira 21 días hacia delante, entre bloques hay que dejar un **embargo de al menos 60 + 21 = 81 días hábiles** para eliminar el solape de información. Sin ese embargo, la loss de validación está contaminada y no demuestra nada.

### 9.3 Representación: secuencia vs aplanado

| Opción | Arquitectura | Parámetros | Coste medido (época) | Valoración |
|---|---|---|---|---|
| Aplanado $\mathbb R^{1081}$ | MLP + emb. tiempo + emb. clase | 0,9M (H=256) | **~1,7 s** | **Recomendada.** Barata, estable, sin sesgo inductivo pero suficiente para 60 pasos. |
| Secuencia $(60,18)$ | U-Net 1D ligera | 0,17M | ~9,3 s (**5,5× más lenta**) | Mejor sesgo inductivo temporal, pero las convoluciones 1D en CPU son ineficientes. Sólo si sobra tiempo. |

Contraintuitivo pero medido: la U-Net 1D tiene **5× menos parámetros y es 5× más lenta** que el MLP en esta CPU. Las convoluciones sobre secuencias cortas no aprovechan bien las rutinas BLAS; el MLP es todo GEMM denso, que es donde la CPU rinde. **Empezar por el MLP aplanado.**

### 9.4 ¿Difusión en el espacio original o sobre PCA?

> Ampliación (no cubierto en clase): la idea de difundir en un espacio latente reducido es exactamente la de los *Latent Diffusion Models* (diapositiva 29 y notebook 6, celda 28: *«latent diffusion can reduce the memory and compute complexity by applying the diffusion process over a lower dimensional latent space»*), sustituyendo el VAE por una PCA lineal.

Propuesta: ajustar PCA sobre el bloque aplanado (**sólo con datos de train**), conservar $k=64$-$128$ componentes (típicamente ≥95% de varianza en un panel tan correlacionado), difundir en ese espacio y reconstruir al generar.

**A favor**: (a) *coste* — 1081 → 64 dimensiones, medido **~1,0 s/época frente a ~1,7 s** y el modelo baja a 0,12M parámetros, con margen para muchas más épocas y barridos; (b) *menos memorización* — menos parámetros y menos dimensiones donde memorizar 600 ventanas de crisis; (c) *blanqueado natural* — si se escalan las componentes a varianza unidad, el supuesto de ruido isótropo se cumple *exactamente* en lugar de aproximadamente, lo que resuelve de raíz el problema de escalas heterogéneas del apartado 8.1; (d) la dimensión intrínseca real del panel es baja (1081 números con correlaciones sectoriales fuertes y suavidad temporal).

**En contra**: (a) la PCA es **lineal**, y toda estructura no lineal (clustering de volatilidad, asimetría, dependencia en colas) que no viva en el subespacio principal se pierde antes de que el generador la vea; (b) la reconstrucción **suaviza**, sumándose al sobre-suavizado propio de la difusión, doble penalización sobre la curtosis; (c) las componentes de baja varianza descartadas pueden ser justo las que codifican los eventos raros.

**Recomendación**: implementar **ambas** y compararlas con las métricas de 7.8, principalmente curtosis y AUC del discriminador. La variante PCA es la **configuración mínima viable garantizada** en CPU; la variante en espacio original es la preferible si el tiempo lo permite. El contraste entre ambas es además un resultado presentable de por sí.

### 9.5 Encaje en el protocolo del taller

Una vez entrenado, generar el número de muestras que exija cada proporción de mezcla real/sintético (0%, 25%, 50%, 100%, sólo sintético), reentrenar el modelo *downstream* con la **misma arquitectura** en cada caso y comparar. La difusión es el generador más caro de los siete, pero el coste está en el entrenamiento (una sola vez); con DDIM el muestreo es prácticamente gratis, así que el barrido de proporciones no añade coste apreciable.

## 10. Implementación de referencia 1D en CPU

Objetivo: una implementación deliberadamente pequeña, con embedding sinusoidal del paso temporal y embedding de clase, entrenable en esta máquina en un tiempo razonable.

### 10.1 Configuración, schedule y proceso directo

```python
import math, torch, torch.nn as nn

# ---------------- Configuracion de referencia (CPU) ----------------
D_DATOS   = 1081     # 60 dias x 18 canales aplanado + y_vol
N_CLASES  = 3        # regimenes de mercado (+1 token nulo para CFG)
T         = 1000     # pasos de difusion en entrenamiento
H         = 256      # ancho oculto -> ~0.9M parametros
D_EMB     = 128      # dimension del embedding sinusoidal del tiempo
BATCH     = 128
LR        = 2e-4     # el notebook usa 1e-3; en datos tabulares conviene mas bajo
WD        = 1e-4     # AdamW: "hace mas estable el entrenamiento" (notebook)
EMA_DECAY = 0.999    # media movil de pesos; la red EMA es la que se usa al muestrear
P_UNCOND  = 0.1      # probabilidad de sustituir la clase por el token nulo (CFG)
EPOCAS    = 400
PASOS_DDIM = 50      # muestreo

def schedule_coseno(T, s=0.008):
    """Schedule coseno (Nichol & Dhariwal). Devuelve alpha_barra[0..T]."""
    t = torch.linspace(0, T, T + 1) / T
    f = torch.cos((t + s) / (1 + s) * math.pi / 2) ** 2
    ab = f / f[0]
    return torch.clip(ab, 1e-5, 0.9999)   # evita divisiones por cero al reconstruir x0

ALPHA_BARRA = schedule_coseno(T)

def difundir(x0, t, eps):
    """Proceso directo en forma cerrada: x_t = sqrt(ab)*x0 + sqrt(1-ab)*eps."""
    ab = ALPHA_BARRA[t].view(-1, 1)
    return ab.sqrt() * x0 + (1 - ab).sqrt() * eps
```

### 10.2 Red

```python
class EmbSinusoidal(nn.Module):
    """Codificacion en frecuencia del paso temporal. Critica para el rendimiento."""
    def __init__(self, dim):
        super().__init__(); self.dim = dim
    def forward(self, t):
        mitad = self.dim // 2
        frec = torch.exp(-math.log(10000.0) * torch.arange(mitad, dtype=torch.float32) / mitad)
        ang = t.float()[:, None] * frec[None, :]
        return torch.cat([torch.sin(ang), torch.cos(ang)], dim=1)

class DifusionMLP(nn.Module):
    """Denoiser condicional: predice el ruido eps a partir de (x_t, t, clase)."""
    def __init__(self, D=D_DATOS, H=H, d_emb=D_EMB, n_clases=N_CLASES):
        super().__init__()
        self.emb_t = EmbSinusoidal(d_emb)
        self.mlp_t = nn.Sequential(nn.Linear(d_emb, H), nn.SiLU(), nn.Linear(H, H))
        # n_clases + 1: la ultima entrada es el token nulo para classifier-free guidance
        self.emb_c = nn.Embedding(n_clases + 1, H)
        self.entrada = nn.Linear(D, H)
        self.bloque1 = nn.Sequential(nn.SiLU(), nn.Linear(H, H), nn.SiLU(), nn.Linear(H, H))
        self.bloque2 = nn.Sequential(nn.SiLU(), nn.Linear(H, H), nn.SiLU(), nn.Linear(H, H))
        self.salida  = nn.Sequential(nn.SiLU(), nn.Linear(H, D))
        # Ultima capa a cero => la red predice 0 al inicio => la loss arranca EXACTAMENTE en 1.0
        nn.init.zeros_(self.salida[-1].weight); nn.init.zeros_(self.salida[-1].bias)

    def forward(self, x, t, c):
        h = self.entrada(x) + self.mlp_t(self.emb_t(t)) + self.emb_c(c)
        h = h + self.bloque1(h)      # conexiones residuales: imprescindibles
        h = h + self.bloque2(h)
        return self.salida(h)
```

### 10.3 Bucle de entrenamiento

```python
modelo = DifusionMLP()
ema    = DifusionMLP(); ema.load_state_dict(modelo.state_dict())
opt    = torch.optim.AdamW(modelo.parameters(), lr=LR, weight_decay=WD)

for epoca in range(EPOCAS):
    for xb, cb in cargador_train:                       # xb: (B, D) normalizado; cb: (B,) clase
        t   = torch.randint(0, T, (xb.shape[0],))       # muestreo uniforme del paso
        eps = torch.randn_like(xb)
        xt  = difundir(xb, t, eps)

        # dropout de la condicion -> la misma red aprende el modelo incondicional
        c = cb.clone()
        c[torch.rand(c.shape[0]) < P_UNCOND] = N_CLASES  # indice del token nulo

        loss = ((modelo(xt, t, c) - eps) ** 2).mean()    # MSE: NO usar MAE (aplana colas)
        opt.zero_grad(); loss.backward(); opt.step()

        # media movil exponencial de los pesos
        with torch.no_grad():
            for p, pe in zip(modelo.parameters(), ema.parameters()):
                pe.mul_(EMA_DECAY).add_(p, alpha=1 - EMA_DECAY)
```

### 10.4 Muestreo DDIM determinista con CFG

```python
@torch.no_grad()
def muestrear_ddim(red, n, clase, pasos=PASOS_DDIM, w=1.5):
    """DDIM con eta=0 (determinista) y classifier-free guidance."""
    x  = torch.randn(n, D_DATOS)
    ts = torch.linspace(T - 1, 0, pasos).long()          # subsecuencia de tiempos
    c_cond   = torch.full((n,), clase, dtype=torch.long)
    c_uncond = torch.full((n,), N_CLASES, dtype=torch.long)

    for i, t in enumerate(ts):
        tb = torch.full((n,), t)
        # classifier-free guidance: dos pasadas, extrapolacion entre ambas
        e_u = red(x, tb, c_uncond)
        e_c = red(x, tb, c_cond)
        eps = e_u + w * (e_c - e_u)

        ab  = ALPHA_BARRA[t]
        ab_prev = ALPHA_BARRA[ts[i + 1]] if i + 1 < len(ts) else torch.tensor(1.0)

        x0 = (x - (1 - ab).sqrt() * eps) / ab.sqrt()     # prediccion del dato limpio
        x  = ab_prev.sqrt() * x0 + (1 - ab_prev).sqrt() * eps   # recombinar (eta = 0)
    return x0                                            # se devuelve la ultima estimacion limpia
```

Nótese que las dos últimas líneas son exactamente la lógica del notebook de clase (celda 17): descomponer en señal y ruido, y recombinar con los coeficientes del siguiente paso.

### 10.5 Tiempos medidos en la máquina objetivo

Máquina: Intel Skylake móvil, 2 núcleos físicos / 4 hilos, sin CUDA. torch 2.11.0+cpu, Python 3.13.7. $N=6.000$ ventanas, `batch=128` → 46 pasos por época.

| Configuración | $D$ | $H$ | Parámetros | s/época | 400 épocas | 1.000 épocas |
|---|---|---|---|---|---|---|
| **MLP (recomendada)** | 1081 | 256 | **0,92M** | **~1,7 s** | **~11 min** | ~28 min |
| MLP amplia | 1081 | 512 | 2,49M | ~3,3-4,4 s | ~25 min | ~60 min |
| **MLP sobre PCA-64 (mínima)** | 64 | 128 | **0,12M** | **~1,0 s** | **~7 min** | ~17 min |
| U-Net 1D `base=32` | (60,18) | — | 0,17M | ~9,3 s | ~60 min | ~155 min |

Muestreo (lote de 512, DDIM-50, sin CFG): **~1,0 s** con H=256, ~3 s con H=512. Con CFG, el doble.

**Estimación realista de tiempo de pared, incluyendo diagnósticos periódicos y con el margen del 1,5-2× por *throttling*:**

- Configuración recomendada (MLP H=256, 400 épocas, generación de 512 muestras cada 25 épocas): **20-30 minutos**.
- Configuración mínima viable (PCA-64, MLP H=128, 400 épocas): **10-15 minutos**.
- Barrido completo (2 representaciones × 3 valores de $w$ de CFG × generación de todos los datasets de mezcla): **1,5-2,5 horas**.

**Configuración mínima viable si el tiempo aprieta**: PCA a 64 componentes, MLP con $H=128$ (~0,12M parámetros), $T=1000$, `batch=128`, 300 épocas, muestreo DDIM-20, CFG con $w=1.5$. Entrena en **menos de 10 minutos** y produce curvas de convergencia presentables.

**Lo que NO hay que intentar en esta máquina**: replicar el notebook de clase. Sus propios resultados (celda 26) lo dicen: *«Con 50 épocas (2 horas en una GPU T4 como las del colab) salen imágenes de calidad.»* Dos horas de T4 son del orden de días de esta CPU. La U-Net 2D sobre imágenes de 64×64 queda descartada por completo; la reducción a 1D no es una simplificación cosmética, es la condición de existencia del experimento.

## 11. Referencias

**Material de clase**

- `docs/material_clase/slides/Diffusion_Models_DM_2026.pdf` (Valero Laparra) — dias. 2-5: roadmap AE → Denoising AE → Diffusion; dia. 10: *Least Squares Denoising*, Miyasawa (1961); dias. 11-12: denoising iterativo y Langevin; dias. 20-22: DDPM y proceso directo; dia. 23: parametrización del *schedule*; dia. 24: DDIM; dias. 26-27: modelo condicional (clasificador guía, pares $(x,y)$, CLIP); dia. 29: *Latent Diffusion Models*; dia. 33: cronología 2020-2026.
- `docs/material_clase/notebooks/5_Difussion_Models_Keras_DDIM_Jordi_mod.ipynb` — celda 6: hiperparámetros; celda 9: arquitectura y embedding sinusoidal; celda 13: schedule continuo; celda 16: `n_loss` vs `i_loss`; celda 17: `DiffusionModel` completo; celda 26: resultados y coste en GPU; celda 27: consejos prácticos.
- `docs/material_clase/notebooks/6_HF_stable_diffusion_VAL.ipynb` — celdas 28-32: difusión latente; celdas 16 y 49-61: classifier-free guidance.
- `docs/enunciado/Taller_B5_T1.pdf` — enunciado del taller (requisito de curvas de convergencia en el apartado 5).

**Artículos citados en el material**

- Ho, Jain, Abbeel (2020). *Denoising Diffusion Probabilistic Models*. arXiv:2006.11239 — dia. 20.
- Song, Meng, Ermon (2021). *Denoising Diffusion Implicit Models*. arXiv:2010.02502 — dia. 24.
- Nichol, Dhariwal (2021). *Improved DDPM* (schedule coseno, sec. 3.2). arXiv:2102.09672 — celda 13.
- Rombach et al. (2022). *High-Resolution Image Synthesis with Latent Diffusion Models*. arXiv:2112.10752 — dia. 29.
- Karras et al. (2022). *Elucidating the Design Space of Diffusion-Based Generative Models*. arXiv:2206.00364 — celda 9.
- Salimans, Ho (2022). *Progressive Distillation* (parametrización $v$, ap. D). arXiv:2202.00512 — celda 28.
- Kadkhodaie, Simoncelli (2021). *Solving Linear Inverse Problems Using the Prior Implicit in a Denoiser*. arXiv:2007.13640 — dia. 9.
- Song, Y., *Score-Based Generative Modeling* (`yang-song.net/blog/2021/score/`) — dia. 11; Weng, L., *What are diffusion models?* (`lilianweng.github.io`) — dia. 22.
