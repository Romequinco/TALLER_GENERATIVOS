# Fundamentos de modelos generativos

Documento base del taller B5-T1 (generación de datos financieros sintéticos). Fija vocabulario, notación y criterios de decisión para el resto de `docs/teoria/`.

Notación común a todo el repositorio:

- $x \in \mathbb{R}^d$: muestra del espacio de datos. Aquí, el bloque conjunto $[\,X ; y_{\text{reg}} ; y_{\text{vol}}\,]$ con $X$ una ventana de $60 \times 18$ (días × canales), es decir $d \approx 1082$.
- $p_{\text{data}}(x)$: densidad real desconocida. $p_\theta(x)$: densidad del modelo.
- $z \in \mathbb{R}^k$: variable latente con densidad base $p(z)$ conocida y fácil de muestrear.
- $J_f(x) = \partial f/\partial x$: matriz jacobiana de $f$ evaluada en $x$.

Los números de diapositiva citados corresponden a la **página del PDF**, que no coincide con el número impreso en la esquina de la lámina.

---

## 1. Qué es un modelo generativo

El material de clase no da una definición sino tres, y las tres conviven (`docs/material_clase/slides/2026_Intro_Generative_Models.pdf`, diap. 2-4).

**Definición 1 — por lo que modela (diap. 2):** generativo estima $p(X,Y)$; condicional (a veces llamado discriminativo) estima $p(Y \mid X)$; discriminativo aprende $Y = f(X)$.

**Definición 2 — por la dirección del condicionamiento (diap. 3):** generativo es $p(X \mid Y)$, discriminativo es $p(Y \mid X)$. Es la lectura clásica de Naive Bayes frente a regresión logística.

**Definición 3 — pragmática (diap. 4):** *un modelo generativo es un modelo que se usa para generar datos*. Es la definición operativa de este taller: si el objeto entrenado produce muestras nuevas plausibles, es generativo, con independencia de si la densidad es accesible.

Las tres no son equivalentes. Un GAN genera datos (Def. 3) pero no da acceso a $p(X,Y)$ (Def. 1); un modelo autorregresivo cumple las tres. Esa tensión estructura la sección 2.

Las diap. 10-11 sitúan estos modelos dentro del **entrenamiento no supervisado**: no hay etiqueta objetivo, el supervisor es la estructura de los propios datos. Es literal aquí: al modelar el bloque conjunto no distinguimos entrada de salida, tratamos todo como un vector cuya densidad queremos aprender.

**Qué significa generar.** Generar es muestrear de $p_\theta$. La consecuencia importante es que las etiquetas se generan también: cada muestra sintética trae su propio $(\tilde{X}, \tilde{y}_{\text{reg}}, \tilde{y}_{\text{vol}})$ coherente entre sí. Es el esquema del ejemplo del profesor sobre temperatura superficial, donde el generador produce pares $\{y_g, X_g\}$ conjuntos y no solo entradas (`docs/material_clase/slides/2026_Taller_Generativos.pdf`, diap. 18-26).

Modelar la conjunta $p(X, y_{\text{reg}}, y_{\text{vol}})$ o la condicional $p(X, y_{\text{vol}} \mid y_{\text{reg}})$ son diseños distintos: la conjunta reproduce la frecuencia natural de regímenes (crisis ~10%), la condicional permite fijarla a voluntad. Se discute en la sección 7.

---

## 2. Taxonomía: densidad explícita vs implícita

El eje que más decisiones prácticas determina es si el modelo da acceso a $p_\theta(x)$ o solo a un muestreador.

**Densidad explícita y tratable** — $\log p_\theta(x)$ se calcula exactamente:

- *Autorregresivos*: factorizan por regla de la cadena, $p_\theta(x) = \prod_{i=1}^{d} p_\theta(x_i \mid x_{<i})$. Las diap. 38-39 los etiquetan literalmente como *"Explicit model"*.
- *Normalizing flows*: aplican la fórmula de cambio de variable (diap. 28: *"Use the change of variables formula to calculate probability estimates"*). Ver sección 5.
- *Gaussiana multivariante y mezclas* (diap. 9); *RBIG* (diap. 31), un flow sin red neuronal.

**Densidad explícita pero aproximada** — existe $p_\theta(x)$ pero es intratable y se optimiza una cota: *VAE* vía ELBO (diap. 22, Kingma & Welling 2013) y *modelos de difusión*, cuyo objetivo es también una cota variacional (diap. 34-35).

**Densidad implícita** — solo hay muestreador: *GAN*, un generador $G: z \mapsto x$ entrenado contra un discriminador (diap. 16-19, Goodfellow et al. 2014). No hay verosimilitud.

> Ampliación (no cubierto en clase): la jerarquía explícita-tratable / explícita-aproximada / implícita procede del tutorial de Goodfellow en NIPS 2016. Las diapositivas listan las familias (diap. 8, 14, 42) pero no las agrupan bajo este árbol. La asignación de cada familia a su casilla sí se apoya en el material: "Explicit model" para autorregresivos (diap. 38) y cambio de variable para flows (diap. 28).

Por qué importa, en concreto para el taller: con densidad explícita se puede (a) calcular la log-verosimilitud de un test **real** bajo el modelo, métrica de ajuste independiente de la tarea downstream, y (b) detectar régimen anómalo, porque una ventana con $\log p_\theta(x)$ muy bajo es atípica respecto a lo aprendido. Con densidad implícita toda la evaluación pasa por muestras (sección 6): no es peor, obliga a métricas de dos muestras y a validar por utilidad.

| Generador del taller | Familia | Densidad | $\log p_\theta(x)$ accesible |
|---|---|---|---|
| Jitter (ruido sobre reales) | Aumento de datos | Implícita (kernel empírico) | No en la práctica |
| Gaussiano multivariante | Paramétrico clásico | Explícita tratable | Sí, forma cerrada |
| cGAN | Adversarial | Implícita | No |
| cVAE | Variable latente | Explícita aproximada (ELBO) | Solo cota |
| RBIG | Flow (gaussianización) | Explícita tratable | Sí |
| Flow Matching | Flow continuo (ODE) | Explícita, costosa | Sí, integrando la traza |
| DDIM | Difusión (muestreo determinista) | Explícita aproximada | Solo cota |

El jitter merece un matiz: sumar ruido gaussiano a datos reales equivale a muestrear de una estimación de densidad por kernel (KDE) con ancho de banda igual a la desviación del ruido. Es un modelo generativo legítimo bajo la Definición 3, y es el baseline que exige el enunciado (`docs/enunciado/Taller_B5_T1.pdf`, tarea 2: *"se debe incluir un cuarto modelo simple, por ejemplo que coja datos originales y les añada ruido"*).

---

## 3. Estimación de máxima verosimilitud y sus límites

Dado $\{x^{(i)}\}_{i=1}^{N}$ supuestamente i.i.d. de $p_{\text{data}}$:

$$\hat{\theta} = \arg\max_{\theta} \sum_{i=1}^{N} \log p_\theta\big(x^{(i)}\big).$$

Cuando $N \to \infty$ esto equivale a minimizar la divergencia de Kullback-Leibler directa
$\mathrm{KL}(p_{\text{data}} \| p_\theta) = \mathbb{E}_{p_{\text{data}}}[\log (p_{\text{data}}(x) / p_\theta(x))]$.
La asimetría de la KL tiene consecuencias operativas:

- La **KL directa** (la que optimiza MLE) penaliza infinitamente que $p_\theta(x) \to 0$ donde $p_{\text{data}}(x) > 0$: el modelo se ve forzado a **cubrir todos los modos**, aunque ponga masa donde no hay datos. Resultado típico: muestras sobre-dispersas y promediadas. Es el sesgo del cVAE y de la gaussiana multivariante.
- La **KL inversa** $\mathrm{KL}(p_\theta \| p_{\text{data}})$ penaliza poner masa donde no hay datos y no penaliza ignorar modos: muestras nítidas pero incompletas. Es el sesgo del entrenamiento adversarial y la raíz del *mode collapse*.

> Ampliación (no cubierto en clase): la lectura *mass covering* / *mode seeking* de la asimetría de la KL, y la conexión del objetivo original del GAN con la divergencia de Jensen-Shannon, no aparecen en las diapositivas. Se incluyen porque explican por qué cVAE y cGAN fallan de formas opuestas sobre los mismos datos.

Caso explícito y tratable, útil como baseline del taller:

```python
import numpy as np
from scipy.linalg import cholesky, solve_triangular

def log_verosimilitud_gaussiana(X, mu, Sigma):
    """Log-verosimilitud media por muestra de una gaussiana multivariante.

    X : (N, d) datos reales sobre los que evaluamos el ajuste.
    mu, Sigma : parametros estimados SOLO con el tramo de entrenamiento.
    """
    N, d = X.shape
    L = cholesky(Sigma, lower=True)          # evita invertir Sigma explicitamente
    sol = solve_triangular(L, (X - mu).T, lower=True)
    log_det = 2.0 * np.sum(np.log(np.diag(L)))   # log|Sigma| = 2*sum(log diag(L))
    cuad = np.sum(sol**2, axis=0)                # Mahalanobis al cuadrado
    return float(np.mean(-0.5 * (d * np.log(2*np.pi) + log_det + cuad)))
```

### Límites de MLE

**(a) Maldición de la dimensionalidad.** La diap. 7 la enuncia como obstáculo central y el PDF de flows la cuantifica (`docs/material_clase/slides/Normalizing Flows_2026.pdf`, diap. 3): con un histograma de 7 bins por dimensión, $\text{n.º bins} = 7^{d}$ y el número de muestras necesarias crece como $(\text{n.º bins})^2$. Para $d \approx 1082$ la cifra es astronómica. La estimación no paramétrica es inviable; toda familia utilizable impone estructura (factorización, invertibilidad, latente de baja dimensión).

**(b) Muestras efectivas, no nominales.** Las ventanas de 60 días con solape diario están fuertemente correlacionadas. Con $\sim 6\,500$ días hay $\sim 6\,400$ ventanas nominales pero solo $\sim 108$ bloques disjuntos, y el número efectivo de observaciones independientes está cerca del segundo. Toda estimación de covarianza $d \times d$ con $d \approx 1082$ es singular por construcción y exige regularización (shrinkage tipo Ledoit-Wolf, o reducción previa por PCA).

**(c) Verosimilitud alta no implica muestras buenas.** En dimensión alta ambas magnitudes se desacoplan: un modelo puede alcanzar buena log-verosimilitud y producir muestras inservibles, y al revés.

> Ampliación (no cubierto en clase): el desacoplamiento está formalizado en Theis, van den Oord & Bethge (2016). Es la razón por la que este taller se evalúa por utilidad downstream y no por verosimilitud.

**(d) Desbalanceo.** MLE pondera cada muestra por igual, luego dedica capacidad proporcional a la frecuencia. Con la clase *crisis* al ~10%, el generador aprenderá sobre todo regímenes normales y suavizará precisamente la cola que interesa. Mitigaciones: condicionar por $y_{\text{reg}}$ (cGAN/cVAE), reponderar la verosimilitud, o entrenar un generador por régimen.

**(e) No siempre es aplicable.** MLE no sirve para un GAN: su entrenamiento se formula como juego minimax entre generador y discriminador, no como maximización de verosimilitud.

---

## 4. Variable latente y muestreo

Estrategia común a casi todas las familias profundas: definir una densidad base $p(z)$ trivial (típicamente $\mathcal{N}(0,I)$) y aprender una transformación que la transporte hasta los datos.

$$p_\theta(x) = \int p_\theta(x \mid z)\, p(z) \, dz.$$

Muestrear son dos pasos: $z \sim p(z)$, luego $x \sim p_\theta(x \mid z)$ (o $x = G_\theta(z)$ si el decodificador es determinista).

### El caso 1D: CDF inversa

`docs/material_clase/notebooks/My_first_generative_model.ipynb` construye el modelo generativo más simple posible y expone el mecanismo sin red neuronal de por medio, sobre una mezcla trimodal (un $\chi^2_1$ y dos gaussianas). Corresponde a las diap. 5-6 del PDF de intro (*"Modelo Básico: histograma or model (+CDF)"*):

```python
import numpy as np
from scipy.stats import norm, kde

# 1) Datos: mezcla de tres componentes, claramente no gaussiana.
x1 = norm.rvs(size=1000, loc=0,  scale=1)**2   # componente asimetrica (chi-cuadrado)
x2 = norm.rvs(size=500,  loc=8,  scale=0.3)    # modo estrecho
x3 = norm.rvs(size=500,  loc=12, scale=3)      # modo ancho
x = np.r_[x1, x2, x3]

# 2) Densidad por kernel (alternativa no parametrica al histograma).
density = kde.gaussian_kde(x)
xgrid = np.linspace(x.min(), x.max(), 100)
d = density(xgrid)

# 3) CDF: integracion numerica de la densidad, normalizada a 1.
cdf = np.cumsum(d) / np.sum(d)

# 4) Muestreo por CDF inversa: u ~ U(0,1) invertida por interpolacion.
#    np.interp(u, cdf, xgrid) evalua CDF^{-1}(u) sin resolverla analiticamente.
u = np.random.rand(100_000)
muestras = np.interp(u, cdf, xgrid)
```

El histograma de `muestras` reproduce la mezcla trimodal original. La variable latente es $u \sim \mathcal{U}(0,1)$ y la transformación es $F^{-1}$: el mismo esquema conceptual que un flow, con $d = 1$. No escala, porque requiere estimar la densidad sobre una rejilla cuyo tamaño crece exponencialmente con $d$ (diap. 7). Las familias profundas son, en esencia, respuestas distintas a "cómo construir $F^{-1}$ en dimensión alta sin rejilla".

### El papel del latente en cada familia

| Familia | Latente $z$ | Relación $z \leftrightarrow x$ |
|---|---|---|
| GAN | $\mathcal{N}(0,I)$, $k \ll d$ | $x = G(z)$, no invertible, sin densidad |
| VAE | $q_\phi(z\mid x)$ al entrenar, $\mathcal{N}(0,I)$ al generar, $k \ll d$ | Codificador + decodificador estocásticos |
| Flow / RBIG | $z = f(x)$, **$k = d$ obligatorio** | Biyección exacta, invertible y diferenciable |
| Difusión / DDIM | $z = x_T$, ruido puro, $k = d$ | Cadena de desruido iterativa |
| Flow Matching | $z = x_0 \sim \mathcal{N}(0,I)$, $k = d$ | ODE continua, $x_{t+1} = x_t + v$ con $v = M(x_t,t)$ |

La última fila reproduce la formulación de las diapositivas de flows (diap. 20): *"la red predice cómo hay que cambiar el dato en el instante $t$"*.

**Truco de la reparametrización (VAE).** Para retropropagar a través de una muestra estocástica se escribe $z = \mu_\phi(x) + \sigma_\phi(x) \odot \epsilon$ con $\epsilon \sim \mathcal{N}(0,I)$: el gradiente fluye por $\mu$ y $\sigma$, y la aleatoriedad queda aislada en $\epsilon$, que no depende de los parámetros.

**Condicionamiento.** Para generar bajo régimen fijado, la etiqueta $y_{\text{reg}}$ se concatena al latente y a la entrada del discriminador o decodificador. Es lo que convierte un GAN en cGAN (diap. 19, *"GANs / Conditional GANs / Coupled GANs"*) y lo que permite sobremuestrear la clase minoritaria.

---

## 5. Cambio de variable y jacobianos

Es la maquinaria que convierte "transformar muestras" en "transformar densidades", y el fundamento de la familia de flows.

### Formulación

En 1D, si $y = g(x)$ con $g$ monótona y diferenciable, la conservación de masa $p_Y(y)|dy| = p_X(x)|dx|$ da

$$p_Y(y) = p_X(x) \left| \frac{dx}{dy} \right| = \frac{p_X\big(g^{-1}(y)\big)}{\big| g'(g^{-1}(y)) \big|}.$$

En dimensión $d$, con $f: \mathbb{R}^d \to \mathbb{R}^d$ invertible y diferenciable, $z = f(x)$:

$$p_X(x) = p_Z\big(f(x)\big)\,\big|\det J_f(x)\big|, \qquad \log p_X(x) = \log p_Z\big(f(x)\big) + \log\big|\det J_f(x)\big|.$$

El determinante jacobiano *"nos dice cuánto se estira el dominio $dx$ respecto a $dz$"* (`docs/material_clase/slides/Normalizing Flows_2026.pdf`, diap. 7-8) y el objetivo de entrenamiento es la log-verosimilitud en el dominio transformado (diap. 9). Requisitos, textualmente de la diap. 13: **obligatorio** ser invertible y diferenciable y ser suficientemente expresivo; **deseable** ser computacionalmente eficiente al calcular la transformación, su inversa y el jacobiano. Componer flows aumenta la expresividad (diap. 10-12): si $f = f_L \circ \cdots \circ f_1$, los log-determinantes se suman, $\log|\det J_f| = \sum_l \log|\det J_{f_l}|$.

### Estructuras de jacobiano

La diap. 15 del PDF de flows enumera las familias por la estructura que imponen a $J$ para abaratar el determinante:

| Estructura | Jacobiano | Coste del determinante |
|---|---|---|
| Elementwise (independiente) | Diagonal | $\mathcal{O}(d)$ |
| Autorregresivo | Triangular inferior | $\mathcal{O}(d)$ |
| Coupling | Bloques con dispersión | $\mathcal{O}(d)$ |
| Determinant identity | Bajo rango | Lema del determinante |
| Unbiased | Libre | Estimación estocástica |

**RBIG** (diap. 16 del PDF de flows; diap. 31 del de intro) alterna dos operaciones triviales en jacobiano: gaussianización marginal canal a canal (jacobiano diagonal; es la CDF inversa de la sección 4 aplicada por dimensión) y una rotación, por ejemplo PCA (matriz ortogonal, $|\det| = 1$, log-determinante nulo). Repetir el par converge a una gaussiana. Es un flow sin entrenamiento por gradiente, lo que lo hace robusto con muestras escasas.

### Deformación de la incertidumbre: el caso $y = x^2$

`docs/material_clase/notebooks/Incertidumbre_bajo_transformaciones.ipynb` construye la intuición geométrica: inyecta ruido gaussiano de $\sigma = 0.05$ alrededor de $x_0 \in \{0,1,2,4\}$ y observa la distribución de salida bajo $y = x^2$.

```python
import numpy as np

NN, sig_n = 10_000, 0.05
for x_0 in [0.0, 1.0, 2.0, 4.0]:
    xx = x_0 + sig_n * np.random.randn(NN)   # ruido gaussiano identico en la entrada
    yy = xx**2                               # transformacion no lineal conocida
    # El factor de estiramiento local es |dy/dx| = |2*x_0|.
    print(x_0, yy.std(), abs(2*x_0)*sig_n)   # coinciden salvo en x_0 = 0
```

Tres lecturas que hay que retener:

1. **El mismo ruido de entrada produce incertidumbres de salida distintas**, escaladas por el jacobiano local $|dy/dx| = |2x_0|$: en $x_0 = 4$ la desviación de salida es ocho veces la de entrada, en $x_0 = 1$ el doble.
2. **La forma cambia, no solo la escala.** En $x_0 = 0$ el jacobiano se anula, la linealización falla y la transformación pliega el eje: la salida deja de ser gaussiana y pasa a ser $\chi^2_1$ escalada, asimétrica y acumulada en cero. El notebook lo muestra normalizando los cuatro histogramas y superponiéndolos: solo el de $x_0 = 0$ tiene forma distinta. Donde $\det J = 0$ la densidad transformada diverge, que es exactamente donde una transformación deja de ser invertible y, por tanto, deja de ser un flow válido.
3. **Una red aprende la transformación y su derivada, pero solo dentro del soporte.** La segunda parte del notebook entrena una MLP (dos capas de 40 unidades, ReLU) sobre $y = x^2$ y extrae $\partial y/\partial x$ con `tf.GradientTape`, comparándola con la derivada analítica $2x$:

```python
import tensorflow as tf

def derivada_de_la_red(X):
    """Jacobiano de la salida respecto a la entrada, por diferenciacion automatica."""
    with tf.GradientTape() as tape:
        prediccion = model(X)
    return tape.gradient(prediccion, X)

XX = np.expand_dims(np.arange(-3, 3, 0.01), axis=1)
d_y_XX = derivada_de_la_red(tf.Variable(XX))   # se compara contra 2*XX
```

La derivada aprendida sigue a $2x$ en la región cubierta por datos y se desvía fuera de ella. Traducción al taller: un generador entrenado con datos de 1999-2019 tiene jacobiano arbitrario en regiones de mercado no observadas, y ahí sus muestras no son fiables.

### Consecuencia práctica para el preprocesado

Estandarizar cada canal, $\tilde{x} = (x - \mu)/\sigma$, es un cambio de variable afín con jacobiano diagonal constante y log-determinante $-\sum_j \log \sigma_j$. Por tanto: no altera el ordenamiento entre modelos comparados **en el mismo espacio**, pero sí altera el valor absoluto de la log-verosimilitud, de modo que las NLL solo son comparables si todos los modelos se evalúan sobre la misma escala; hay que fijar el preprocesado una vez y documentarlo. Además, $\mu$ y $\sigma$ deben estimarse **solo con el tramo de entrenamiento**: hacerlo sobre el panel completo introduce fuga de información del futuro.

---

## 6. Cómo se evalúa un modelo generativo

No hay métrica única. Conviene separar tres ejes y no confundirlos.

### Eje 1 — Utilidad downstream (criterio primario del taller)

Es el enfoque del profesor y lo que exige el enunciado. En su problema real de estimación de temperatura superficial (LST) a partir de observaciones IASI, la evaluación es directamente el error del modelo predictivo (`docs/material_clase/slides/2026_Taller_Generativos.pdf`, diap. 27-28): solo reales, 4.81 K; reales + GAN, 3.46 K; reales + RBIG, 3.62 K. El protocolo replicable es fijar una arquitectura downstream válida con datos reales y reentrenarla **sin cambiarla** sobre datasets con proporciones crecientes de sintético.

Reglas no negociables del protocolo:

1. El test es **siempre real**. Nunca se evalúa sobre sintético.
2. El generador se entrena **solo con el tramo de entrenamiento**. Si ve el test, el sintético lo filtra al downstream y el resultado es una fuga.
3. Con series temporales el corte es cronológico y con embargo: hay que purgar al menos $60 + 21 = 81$ días entre train y test para que ninguna ventana de entrenamiento solape con el horizonte de una etiqueta de test.
4. Varias semillas por configuración, con media ± desviación. Las diferencias entre generadores suelen ser del orden de la varianza entre semillas.

Sobre las métricas downstream: con $y_{\text{reg}}$ de 3 clases y crisis al ~10%, la *accuracy* es engañosa (un clasificador que nunca prediga crisis alcanza ~90%). Hay que usar *balanced accuracy*, macro-F1 y sobre todo el F1 o el recall de la clase crisis. Para $y_{\text{vol}}$, RMSE sobre volatilidad realizada.

> Ampliación (no cubierto en clase): el protocolo TSTR (*Train on Synthetic, Test on Real*) está formalizado en Esteban, Hyland & Rätsch (2017) para series temporales. Las métricas de desbalanceo y el embargo temporal tampoco aparecen en las diapositivas; sin ellas los resultados de este taller no serían interpretables.

### Eje 2 — Fidelidad distribucional

Mide si $p_\theta \approx p_{\text{data}}$, con independencia de la tarea.

- **NLL sobre test real**, si el modelo es explícito (gaussiana, RBIG; con reservas cVAE/DDIM vía cota). Comparable solo dentro del mismo espacio y preprocesado.
- **Classifier two-sample test (C2ST)**: entrenar un clasificador para distinguir real de sintético. AUC $\approx 0.5$ significa muestras indistinguibles; $\approx 1.0$, generador fallido. Es la métrica más informativa para modelos implícitos, y es el papel del discriminador de un GAN reutilizado como métrica.

```python
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import numpy as np

def c2st(X_real, X_sint, semilla=0):
    """AUC real-vs-sintetico. 0.5 = indistinguibles, 1.0 = trivialmente separables."""
    X = np.vstack([X_real, X_sint])
    y = np.r_[np.ones(len(X_real)), np.zeros(len(X_sint))]
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=semilla)
    clf = HistGradientBoostingClassifier(random_state=semilla).fit(X_tr, y_tr)
    return roc_auc_score(y_te, clf.predict_proba(X_te)[:, 1])
```

- **Estadísticos estilizados de series financieras**: autocorrelación de retornos $\approx 0$; autocorrelación de $|$retornos$|$ con decaimiento lento (agrupamiento de volatilidad); curtosis elevada (colas pesadas); matriz de correlación entre los 9 sectores y el índice; efecto apalancamiento. Un generador que reproduzca medias y varianzas pero destruya estos estadísticos produce datos inútiles para el problema.

> Ampliación (no cubierto en clase): la lista de *stylized facts* no está en el material del taller. Es el conjunto estándar en econometría financiera y la comprobación mínima para validar sintético de mercado.

### Eje 3 — Diversidad y memorización

Dos fallos opuestos que la utilidad downstream no siempre distingue. **Mode collapse** (típico de GAN): el generador produce pocas configuraciones distintas; se detecta por la distancia media entre pares de muestras sintéticas frente a la misma medida entre reales, o por la cobertura de las 3 clases de régimen. **Memorización**: el generador copia el entrenamiento; se detecta comparando la distancia al vecino más próximo del *train* para muestras sintéticas frente a la misma distancia para muestras de *test* reales. Si el sintético está sistemáticamente más cerca del train, no generaliza.

Advertencia sobre las curvas de pérdida, que el enunciado exige para demostrar convergencia: para cVAE, RBIG, Flow Matching y DDIM son interpretables; para el cGAN no lo son. En el ejemplo del profesor (diap. 18-24 de las slides del taller) la pérdida del discriminador baja de 0.75 a 0.41 mientras la del generador **sube** de 0.73 a 1.37, y aun así las muestras mejoran visiblemente. En un GAN la evidencia de convergencia son las muestras y el C2ST, no la loss.

---

## 7. Mapa de decisión: qué familia para qué problema

### Restricciones concretas del taller

| Restricción | Valor | Implicación |
|---|---|---|
| Dimensión del bloque | $d \approx 1082$ ($60 \times 18 + 2$) | Descarta densidad no paramétrica; obliga a estructura |
| Muestras nominales | $\sim 6\,000$ ventanas | Insuficiente para modelos *data hungry* |
| Muestras efectivas | $\sim 100$ bloques disjuntos | Riesgo severo de sobreajuste del generador |
| Objeto a generar | Bloque conjunto $[X ; y_{\text{reg}} ; y_{\text{vol}}]$ | Debe preservar la dependencia $X \leftrightarrow y$ |
| Clase crisis | ~10% | Favorece generadores condicionables |
| Estructura de $X$ | Temporal (60 pasos) + cross-sectional (18 canales) | Favorece arquitecturas que la exploten |

### Tabla de decisión

| Generador | Coste | Estabilidad | Fuerte en | Riesgo dominante | Prioridad |
|---|---|---|---|---|---|
| **Jitter** | Nulo | Total | Baseline obligatorio | No crea diversidad; solo suaviza la frontera | 1 (listón mínimo) |
| **Gaussiano multivariante** | Bajo | Total | Medias y covarianzas exactas | Covarianza singular con $d \gg N$; ignora colas y no linealidad | 2 (segundo listón) |
| **RBIG** | Bajo-medio | Alta | Pocas muestras, sin gradientes, da $\log p$ | Rotación $d \times d$ costosa; requiere PCA previa | 3 |
| **cVAE** | Medio | Alta | Latente estructurado, condicionamiento limpio | Sobre-suavizado: subestima colas y picos de volatilidad | 4 |
| **Flow Matching** | Medio | Alta | Objetivo acotado, muestreo en 10-25 pasos | Requiere red suficientemente expresiva | 5 |
| **DDIM** | Alto | Media | Máxima expresividad | *Data hungry*: puede ser inviable con $N$ efectivo bajo | 6 |
| **cGAN** | Alto | Baja | Muestras nítidas, control directo de clase | *Mode collapse*, inestabilidad, loss no interpretable | 7 |

Coste y estabilidad de los tres últimos siguen el cuadro comparativo de `docs/material_clase/slides/Normalizing Flows_2026.pdf` (diap. 20): el flow clásico sufre la maldición de la dimensión en el cálculo del jacobiano pero infiere en un solo paso; la difusión escala muy bien y es *data hungry*, con 20-100 pasos de trayectorias curvas; Flow Matching tiene la misma escalabilidad que la difusión con objetivo acotado ($x_1 - x_0$), pérdida que baja más rápido y trayectorias casi rectas que permiten 10-25 pasos.

### Reglas de decisión

**Para tener un listón honesto**: gaussiano multivariante con shrinkage. Preserva la estructura de covarianza completa del panel, que es parte sustancial de la señal en datos financieros. Un generador profundo que no lo bata no aporta nada, y ese resultado negativo es tan reportable como uno positivo.

**Si el cuello de botella es el número de muestras**: RBIG. No entrena por gradiente, no tiene hiperparámetros de arquitectura y funciona con muestras escasas. En el ejemplo del profesor obtiene 3.62 K frente a 3.46 K del GAN (slides del taller, diap. 28): mismo orden de resultado con una fracción del coste y del riesgo.

**Si el objetivo es corregir el desbalanceo de la clase crisis**: cGAN o cVAE condicionados por $y_{\text{reg}}$, que permiten fijar la proporción de crisis en lugar de heredar el ~10% natural. Es la aplicación donde el condicionamiento aporta valor que la conjunta no puede dar.

**Si se busca el mejor compromiso calidad/estabilidad con cómputo medio**: Flow Matching, por las razones del cuadro comparativo citado.

### Conjunta o condicional

- **Conjunta** $p(X, y_{\text{reg}}, y_{\text{vol}})$: se genera el bloque completo de una vez, como hace el profesor con los pares $\{y_g, X_g\}$ (slides del taller, diap. 18-26). Preserva por construcción la coherencia entre ventana y etiquetas y reproduce la frecuencia natural de regímenes.
- **Condicional** $p(X, y_{\text{vol}} \mid y_{\text{reg}})$: se fija el régimen y se genera el resto, lo que permite balancear clases. Coste: hay que decidir a mano la mezcla de regímenes, una decisión de diseño que hay que justificar y que afecta al resultado downstream.

Recomendación operativa: la conjunta como configuración base para todos los generadores que la admitan, y la condicional solo en cGAN/cVAE, para aislar el efecto del condicionamiento.

### Advertencia sobre el diseño experimental

El hallazgo más accionable del material no está en la tabla de errores sino en las diap. 30-32 de `docs/material_clase/slides/2026_Taller_Generativos.pdf`: la comparación GAN vs RBIG se repite con 500, 1000, 3000, 7000, 20000 y todos los datos reales. **La ganancia del sintético es grande cuando hay pocos datos reales y se desvanece a medida que crecen.** Consecuencia directa: no basta con barrer el porcentaje de sintético a $N_{\text{real}}$ fijo, hay que barrer también $N_{\text{real}}$. Un resultado plano al 100% de los datos reales no significa que los generadores no funcionen; puede significar que el régimen de datos ya es suficiente.

---

## 8. Referencias

### Material de clase

- `docs/material_clase/slides/2026_Intro_Generative_Models.pdf` (Valero Laparra) — definiciones formales (diap. 2-4), histograma + CDF (5-6), maldición de la dimensionalidad (7), catálogo de familias (8-9), relación con no supervisado (10-11), GANs (16-20), VAEs (22-24), flows y cambio de variable (26-32), difusión (34-36), autorregresivos (38-41), style transfer (43-48).
- `docs/material_clase/slides/Normalizing Flows_2026.pdf` — cambio de variable y determinante jacobiano (diap. 5-8), objetivo de log-verosimilitud (9), composición de flows (10-12), requisitos de un flow (13), estructuras de jacobiano (15), RBIG (16), Flow Matching (17-18), cuadro comparativo flow / difusión / flow matching (20).
- `docs/material_clase/slides/2026_Taller_Generativos.pdf` — planteamiento (diap. 3-7), procedimiento con GANs y generación conjunta $\{y,X\}$ (14-26), resultados cuantitativos (27-28), dependencia con el tamaño del conjunto real (30-32).
- `docs/material_clase/notebooks/My_first_generative_model.ipynb` — KDE, CDF y muestreo por CDF inversa.
- `docs/material_clase/notebooks/Incertidumbre_bajo_transformaciones.ipynb` — deformación de la incertidumbre bajo $y = x^2$ y cálculo del jacobiano por diferenciación automática.
- `docs/enunciado/Taller_B5_T1.pdf` — 3 generativos + 1 baseline simple, barrido de proporción sintético/real, análisis comparativo, curvas de pérdida obligatorias.

### Artículos citados en las diapositivas

Goodfellow et al. (2014), *Generative Adversarial Networks*, arXiv:1406.2661 · Kingma & Welling (2013), *Auto-Encoding Variational Bayes*, arXiv:1312.6114 · Isola et al. (2016), *pix2pix*, arXiv:1611.07004 · Zhu et al. (2017), *CycleGAN*, arXiv:1703.10593 · Laparra, Camps-Valls & Malo (2011), *Iterative Gaussianization: from ICA to Random Rotations*, IEEE TNN (RBIG) · Chen & Gopinath (2000), *Gaussianization*, NeurIPS · Meng et al. (2020), *Gaussianization Flows*, AISTATS · Inouye & Ravikumar (2018), *Deep Density Destructors*, ICML · Kobyzev et al. (2019), *Normalizing Flows: An Introduction and Review of Current Methods* · Papamakarios et al. (2019), *Normalizing Flows for Probabilistic Modeling and Inference* · Lipman et al. (2022), *Flow Matching for Generative Modeling*, arXiv:2210.02747 · Gatys et al. (2016), *Image Style Transfer Using CNNs*, CVPR.

### Referencias de ampliación (fuera del material de clase)

Goodfellow (2016), *NIPS 2016 Tutorial: GANs*, arXiv:1701.00160 — taxonomía explícita/implícita (§2) · Theis, van den Oord & Bethge (2016), *A Note on the Evaluation of Generative Models*, ICLR — desacoplamiento verosimilitud/calidad (§3) · Ho, Jain & Abbeel (2020), *DDPM*, NeurIPS · Song, Meng & Ermon (2021), *DDIM*, ICLR · Esteban, Hyland & Rätsch (2017), *Real-valued (Medical) Time Series Generation with RCGANs*, arXiv:1706.02633 — protocolo TSTR (§6) · Ledoit & Wolf (2004), *A well-conditioned estimator for large-dimensional covariance matrices*, JMVA — shrinkage del baseline gaussiano (§3, §7) · Cont (2001), *Empirical properties of asset returns: stylized facts and statistical issues*, Quantitative Finance — estadísticos estilizados (§6).
