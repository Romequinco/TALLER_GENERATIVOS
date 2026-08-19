# Generadores de referencia

Fundamento teórico de los tres generadores no neuronales del taller B5-T1: perturbación por ruido (*jitter*), gaussiano multivariante y bootstrap por bloques.

## 1. Por qué los baselines importan

El enunciado (`docs/enunciado/Taller_B5_T1.pdf`, tarea 2) exige tres generativos neuronales **y** un cuarto modelo simple: *"se debe incluir un cuarto modelo simple, por ejemplo que coja datos originales y les añada ruido"*. La tarea 5 cierra el círculo: *"se comparará entre los distintos tipos de modelos generativos usados (incluyendo el modelo simple)"*.

El baseline no es un trámite. Cumple tres funciones:

1. **Cota inferior de utilidad.** La métrica que importa no es la calidad visual de las muestras sino la mejora del modelo *downstream*. La presentación del taller (`docs/material_clase/slides/2026_Taller_Generativos.pdf`, láminas 30–32) muestra que la ventaja de un generativo sofisticado depende críticamente del número de datos reales: con 500 reales la mejora es enorme; con 20 000 reales GAN y RBIG son indistinguibles entre sí y la ganancia se agota. El eje relevante es $n_{\text{real}}$, no la arquitectura.
2. **Control de sanidad.** Si una GAN o un difusor no baten a añadir ruido gaussiano, hay dos lecturas: el generativo está mal entrenado, o el problema no admite ganancia por datos sintéticos. Sin baseline no se pueden distinguir.
3. **Descomposición del mecanismo de mejora.** Cada baseline aísla una hipótesis sobre *por qué* ayudan los datos sintéticos: el jitter aísla la regularización local, el gaussiano la estructura de segundo orden, el bootstrap la dependencia temporal. Si el gaussiano recupera el 90 % de la ganancia de la GAN, la conclusión defendible es que la ganancia venía de la covarianza, no del modelado no lineal.

**Un resultado negativo bien medido es un resultado.** La lámina 29 concluye "yes", pero las 31–32 la matizan. Reportar que ningún generador bate al ruido en nuestro régimen de datos es una conclusión válida siempre que el protocolo de medida sea idéntico para todos.

---

## 2. Jitter: datos reales más ruido

### 2.1 Formulación y elección de sigma

Es el modelo de referencia explícito del profesor, en `docs/material_clase/notebooks/Taller_GANs.ipynb`, sección *"Ejemplo muy tonto (datos con ruido)"* (celda 46):

```python
n_muestras = 500
sig = 0.01
Ruido_x = rng.normal(0, sig, X_train[0:n_muestras].shape)
Ruido_y = rng.normal(0, sig, Y_train[0:n_muestras].shape)
X_train_ext = np.concatenate((X_train[0:500], X_train[0:n_muestras] + Ruido_x), axis=0)
Y_train_ext = np.concatenate((Y_train[0:500], Y_train[0:n_muestras] + Ruido_y), axis=0)
```

Formalmente, dada una muestra real $x^{(i)} \in \mathbb{R}^d$:

$$\tilde{x}^{(i)} = x^{(i)} + \varepsilon, \qquad \varepsilon \sim \mathcal{N}(0, \Sigma_\varepsilon).$$

Con $\Sigma_\varepsilon = \sigma^2 I$ y eligiendo $i$ uniformemente al azar, el generador es exactamente un muestreador de un **estimador de densidad de núcleo (KDE) gaussiano** con ancho de banda $\sigma$: la densidad implícita es $\hat p(x) = \frac{1}{n}\sum_i \mathcal{N}(x; x^{(i)}, \sigma^2 I)$. Elegir $\sigma$ es un problema clásico de selección de ancho de banda.

**El valor `sig = 0.01` del profesor no es transferible.** En su notebook los datos son log-retornos diarios sin estandarizar, con desviación típica del orden de $0{,}01$–$0{,}02$: su ruido tiene magnitud **comparable a la de la señal** ($\alpha \approx 0{,}5$–$1{,}0$ abajo). Nuestro bloque mezcla canales de escalas radicalmente distintas (retornos $\sim 10^{-3}$, VIX $\sim 20$, spreads en puntos básicos), donde un $\sigma$ escalar único es inaplicable.

**Criterio propuesto: sigma como fracción de la desviación típica por canal**, con un único hiperparámetro adimensional $\alpha$:

$$\sigma_j = \alpha \cdot \hat\sigma_j, \qquad \hat\sigma_j = \text{desv. típica del canal } j \text{ en entrenamiento}.$$

Si los datos ya están estandarizados por canal (§7), esto se reduce a $\Sigma_\varepsilon = \alpha^2 I$. El compromiso: con $\alpha \to 0$ las muestras son duplicados y no aportan información; con $\alpha \to \infty$ se destruye la estructura y se inyecta ruido blanco con etiquetas reales, degradando el modelo.

> Ampliación (no cubierto en clase): los tres criterios cuantitativos siguientes.

**Criterio 1 — atenuación de correlaciones.** Con ruido independiente por canal de varianza relativa $\alpha^2$, la correlación entre dos canales se atenúa de forma exacta (atenuación clásica por error de medida):

$$\rho'_{jk} = \frac{\rho_{jk}}{1+\alpha^2}.$$

Para $\alpha = 0{,}1$ la atenuación es del $1\,\%$; para $\alpha = 0{,}3$, del $8{,}3\,\%$. Si el modelo *downstream* explota correlaciones entre canales, $\alpha \le 0{,}15$ mantiene el sesgo por debajo del $2\,\%$.

**Criterio 2 — dilución de las colas.** Para $X$ real y $\varepsilon$ gaussiano independiente:

$$\mathrm{Kurt}(\tilde{X}) = \frac{\mathrm{Kurt}(X) + 6\alpha^2 + 3\alpha^4}{(1+\alpha^2)^2}.$$

Con $\mathrm{Kurt}(X) = 10$: $\alpha = 0{,}05 \Rightarrow 9{,}97$ (irrelevante); $\alpha = 0{,}3 \Rightarrow 8{,}89$ (pérdida del $11\,\%$). El jitter **gaussianiza progresivamente** las colas, justo lo que queremos evitar en datos financieros.

**Criterio 3 — escala del vecino más próximo (el operativo).** Los dos anteriores acotan $\alpha$ por arriba; éste por abajo. La perturbación tiene norma esperada $\mathbb{E}\|\varepsilon\| \approx \alpha\sqrt{d}$ en unidades estandarizadas. Si es mucho menor que la distancia típica al vecino más próximo $\delta_{NN}$, la muestra es un duplicado; si es mucho mayor, sale de la variedad de datos. Regla: elegir $\alpha$ tal que $\alpha\sqrt{d} \approx (0{,}25 - 0{,}5)\cdot \mathrm{mediana}(\delta_{NN})$.

```python
import numpy as np
from sklearn.neighbors import NearestNeighbors

def alpha_por_vecino_mas_proximo(Z, fraccion=0.35):
    """Z: (n, d) estandarizado por canal. Devuelve alpha y la distancia mediana al 1-NN."""
    dist, _ = NearestNeighbors(n_neighbors=2).fit(Z).kneighbors(Z)
    delta_nn = np.median(dist[:, 1])          # col 0 = uno mismo (distancia 0)
    return fraccion * delta_nn / np.sqrt(Z.shape[1]), delta_nn

def generar_jitter(Z, y_reg, y_vol, n_gen, alpha, rng):
    """KDE gaussiano isotropo de ancho alpha. La etiqueta se copia del punto base,
    de modo que el par (X, y) permanece coherente."""
    idx = rng.integers(0, Z.shape[0], size=n_gen)      # con reemplazo
    return Z[idx] + alpha * rng.standard_normal((n_gen, Z.shape[1])), y_reg[idx], y_vol[idx]
```

En la práctica se barre $\alpha \in \{0{,}01,\ 0{,}05,\ 0{,}1,\ 0{,}25,\ 0{,}5\}$ y se reporta la métrica *downstream* frente a $\alpha$. Ese barrido **es** la curva de sensibilidad del baseline y sustituye a la curva de loss (§6).

### 2.2 Qué preserva y qué destruye

**Preserva.** El emparejamiento $(X, y)$: la etiqueta se copia del punto base, así que la relación entrada-salida no se corrompe (no es trivial: un generativo conjunto mal ajustado sí puede romperla). Preserva también, hasta $O(\alpha^2)$, todos los momentos y las estructuras temporal y transversal.

**Destruye.** Nada, estrictamente: *añade*. Y ahí está el problema. El soporte de la densidad generada son bolas de radio $\sim\alpha\sqrt{d}$ centradas en las muestras reales: **no hay extrapolación**. El generador no puede producir un régimen de mercado que no esté ya en entrenamiento. Para la clase "crisis", el jitter multiplica filas, no episodios.

**Patología en alta dimensión.** Con $d = 1\,201$ y dimensión intrínseca $k \ll d$, una perturbación isótropa reparte su energía uniformemente: sólo una fracción $k/d$ cae dentro de la variedad de datos y $1 - k/d$ cae fuera. **La mayor parte del ruido empuja las muestras fuera de la variedad**, generando configuraciones dinámicamente imposibles (por ejemplo, un VIX que salta sin que se muevan los retornos).

> Ampliación (no cubierto en clase): mitigación por jitter anisótropo. Proyectar el ruido sobre las $k$ primeras componentes principales y escalarlo por $\sqrt{\lambda_i}$ mantiene las muestras sobre la variedad: $\tilde{x} = x + \alpha\, U_k \Lambda_k^{1/2} z$ con $z \sim \mathcal{N}(0, I_k)$. Es una ablación barata que aísla el efecto "salir de la variedad".

---

## 3. Gaussiano multivariante

### 3.1 Formulación: media, covarianza y muestreo

Es el generador de `docs/material_clase/notebooks/Taller_Gaussian_solution.ipynb`. La idea clave del profesor es **aplanar el bloque conjunto $[X; Y]$ y ajustar un único gaussiano sobre él**: así las correlaciones entrada-salida quedan capturadas y las muestras son pares $(X_g, Y_g)$ coherentes (opción OPT2 de la lámina 8):

```python
# Taller_Gaussian_solution.ipynb, celdas 16-19 (reproducido)
XY_aux = np.zeros((X_train_aux.shape[0], X_train_aux.shape[1] + 1, X_train_aux.shape[2]))
XY_aux[:, 0:X_train_aux.shape[1], :] = X_train_aux
XY_aux[:, X_train_aux.shape[1], :] = Y_train_aux
XY_aux_flat = XY_aux.reshape(XY_aux.shape[0], -1)

mean_XY = np.mean(XY_aux_flat, axis=0)
cov_XY = np.cov(XY_aux_flat.T)
XY_synth_flat = rng.multivariate_normal(mean_XY, cov_XY, size=10000)
```

El modelo es $p(x) = \mathcal{N}(x; \hat\mu, \hat\Sigma)$ con los estimadores de máxima verosimilitud

$$\hat\mu = \frac{1}{n}\sum_{i=1}^{n} x^{(i)}, \qquad \hat\Sigma = \frac{1}{n-1}\sum_{i=1}^{n} (x^{(i)} - \hat\mu)(x^{(i)} - \hat\mu)^\top,$$

y el muestreo se hace por factorización $\hat\Sigma = LL^\top$, $x = \hat\mu + Lz$ con $z \sim \mathcal{N}(0, I_d)$. **No hay entrenamiento iterativo**: dos pasadas sobre los datos y una factorización. Ese es su atractivo en un entorno CPU-only, y también la razón de que §6 sea necesaria.

**Coste real medido** en un banco de pruebas a $d = 1\,100$ y $n = 4\,500$ (CPU, numpy 2.3.4), muy cerca de nuestro bloque real ($d = 1\,201$, $n = 3\,696$ ventanas de train; a esa dimensión la matriz $d\times d$ ocupa 11,5 MB en vez de 9,7):

| Operación | Tiempo | Memoria |
|---|---|---|
| `np.cov(X.T)` | 0,48 s | 9,7 MB (matriz $d\times d$) |
| `LedoitWolf().fit(X)` | 14,0 s | — |
| `np.linalg.cholesky` | 0,62 s | — |
| `rng.multivariate_normal(mu, C, size=5000)` | **10,1 s** | 44 MB |
| Cholesky cacheado + `@` (5000 muestras) | **0,88 s** | 44 MB |

`multivariate_normal` refactoriza la covarianza (SVD) **en cada llamada**. Para generar varios tamaños de *dataset* sintético, factorizar una vez y reutilizar es 11× más rápido:

```python
def ajustar_gaussiano(Z):
    """Factorizar UNA vez y reutilizar en todas las generaciones."""
    C, info = estimar_covarianza(Z)         # ver 3.2: nunca np.cov a secas
    return Z.mean(axis=0), np.linalg.cholesky(C), info

def muestrear_gaussiano(mu, L, n_gen, rng):
    return mu + rng.standard_normal((n_gen, mu.size)) @ L.T
```

El fallo de `np.linalg.cholesky` no es un inconveniente: es la **verificación** de que la covarianza es utilizable. Se usa como puerta dura.

### 3.2 Estimación de la covarianza en alta dimensión

Nuestro bloque tiene $d = 1\,201$ dimensiones y disponemos de $n = 3\,696$ ventanas de entrenamiento, de las que sólo **587 son de la clase minoritaria**. La covarianza muestral tiene $d(d+1)/2 = 721\,801$ parámetros libres estimados con $n\cdot d \approx 4{,}4\cdot10^6$ números: apenas seis números por parámetro.

> Ampliación (no cubierto en clase): toda esta subsección. El notebook del profesor usa `np.cov` directamente porque su bloque tiene $61\times23 = 1403$ dimensiones con $\sim14\,000$ muestras y canales homogéneos (todos log-retornos). Nuestro caso es más adverso.

**Problema 1: rango deficiente.** $\mathrm{rank}(\hat\Sigma) \le \min(n-1, d)$. Para la clase crisis con $n_c = 587 < d = 1\,201$, $\hat\Sigma$ es **singular por construcción**, con al menos $615$ autovalores nulos.

**Problema 2: sesgo espectral aunque $n > d$.** Con $n = 3\,696 > d = 1\,201$, el ratio $q = d/n \approx 0{,}32$ distorsiona el espectro: para datos blancos los autovalores muestrales se dispersan sobre $[(1-\sqrt{q})^2, (1+\sqrt{q})^2] \approx [0{,}18,\ 2{,}47]$ en vez de concentrarse en 1. El autovalor mínimo se subestima en un factor 5,4 y el máximo se sobreestima en un factor 2,5. Los autovalores pequeños están sistemáticamente **sesgados hacia abajo**, la dirección peligrosa para muestrear.

**Problema 3: qué hace realmente `np.random.multivariate_normal` con una covarianza no definida positiva.** Comprobado empíricamente con numpy 2.3.4:

| Situación | Comportamiento observado |
|---|---|
| $\hat\Sigma$ de rango deficiente ($n<d$), autovalores $\sim -3\cdot10^{-15}$ | **Ninguna advertencia.** Muestrea en silencio. |
| Autovalor claramente negativo ($\lambda=-1$) | `RuntimeWarning: covariance is not symmetric positive-semidefinite`; **no lanza excepción**, devuelve muestras. |
| `method='cholesky'` | `numpy.linalg.LinAlgError: Matrix is not positive definite`. |
| `method='eigh'` con autovalor negativo | Advierte, pero también muestrea. |

Dos hallazgos que hay que interiorizar:

1. **Con un autovalor negativo $-|\lambda|$, la varianza generada en esa dirección es $+|\lambda|$.** Verificado: una covarianza con $\lambda=-1$ produce muestras con varianza medida $1{,}004$ en esa dirección. El método por defecto (SVD) devuelve valores singulares $\ge 0$, de modo que se muestrea efectivamente con $|\hat\Sigma|$ en la base propia: se **inyecta varianza** donde el estimador decía que no había ninguna.
2. **Con rango deficiente, el fallo es silencioso y total.** Midiendo la energía de las muestras generadas fuera del subespacio generado por los datos de entrenamiento se obtiene $2{,}6\cdot10^{-8}$, es decir, **cero**. Las "muestras sintéticas" son combinaciones afines exactas de las muestras reales, confinadas a un subespacio de dimensión $n_c-1$. Para la clase crisis, el gaussiano no genera nada nuevo y lo hace sin emitir una sola advertencia. Este es el modo de fallo que hay que detectar antes de la defensa oral, no durante.

**Solución 1: shrinkage de Ledoit-Wolf.** Se combina la covarianza muestral con un objetivo bien condicionado:

$$\hat\Sigma_{\text{LW}} = (1-\rho)\,\hat\Sigma + \rho\,\mu I, \qquad \mu = \frac{\mathrm{tr}(\hat\Sigma)}{d},$$

donde $\rho \in [0,1]$ se elige de forma analítica (sin validación cruzada) minimizando $\mathbb{E}\|\hat\Sigma_{\text{LW}} - \Sigma\|_F^2$. El resultado es **siempre definido positivo** para $\rho>0$, con $\lambda_{\min} \ge \rho\mu$. Comportamiento medido sobre una covarianza estructurada de tipo factorial ($d=300$):

| $n$ | $\rho$ Ledoit-Wolf | $\mathrm{cond}(\hat\Sigma)$ | $\|\hat\Sigma - \Sigma\|_F/\|\Sigma\|_F$ |
|---|---|---|---|
| 100 | 0,1113 | $\infty$ | 0,341 |
| 300 | 0,0422 | $8{,}3\cdot10^{18}$ | 0,208 |
| 1000 | 0,0127 | $2{,}5\cdot10^{3}$ | 0,126 |
| 3000 | 0,0044 | $1{,}1\cdot10^{3}$ | 0,062 |
| 10000 | 0,0013 | $8{,}7\cdot10^{2}$ | 0,030 |

$\rho$ decrece monótonamente con $n$: **es un diagnóstico de convergencia legible** (§6). Un $\rho$ alto indica que el estimador está dominado por la regularización, no por los datos.

**Solución 2: regularización diagonal (ridge).** $\hat\Sigma_\epsilon = \hat\Sigma + \epsilon I$. Más burda que LW, pero admite un criterio cerrado: con autovalores $\lambda_1 \ge \dots \ge \lambda_d$, exigir $\mathrm{cond}(\hat\Sigma_\epsilon) \le \kappa_{\max}$ da

$$\epsilon \ge \frac{\lambda_1 - \kappa_{\max}\lambda_d}{\kappa_{\max}-1}.$$

Con $\kappa_{\max}=10^6$ en float64 quedan ~10 dígitos significativos: objetivo seguro.

**Solución 3: factorización por componentes principales (modelo factorial).** Se aproxima

$$\hat\Sigma \approx W W^\top + \Psi, \qquad W = U_k \Lambda_k^{1/2} \in \mathbb{R}^{d\times k}, \qquad \Psi = \mathrm{diag}(\psi_1,\dots,\psi_d) \succ 0,$$

y se muestrea sin formar nunca la matriz $d\times d$:

$$x = \hat\mu + W z + \Psi^{1/2}\varepsilon, \qquad z\sim\mathcal{N}(0,I_k),\ \varepsilon\sim\mathcal{N}(0,I_d).$$

Coste $O(dk)$ por muestra en vez de $O(d^2)$, y definida positiva por construcción si $\psi_j>0$. **Es la opción recomendada para la clase minoritaria**, imponiendo $k \le n_c-1$ y en la práctica $k \sim 20$–$50$ (por varianza explicada del 90–95 %).

```python
import numpy as np
from sklearn.covariance import LedoitWolf

def estimar_covarianza(Z, metodo="auto", k=None, kappa_max=1e6):
    """Estimador robusto en alta dimension. Z: (n, d) estandarizado por canal."""
    n, d = Z.shape
    if metodo == "auto":
        # Con menos de 2 muestras por dimension, la covarianza muestral no es utilizable
        metodo = "pca" if n < 2 * d else "ledoit_wolf"

    if metodo == "ledoit_wolf":
        lw = LedoitWolf(block_size=512).fit(Z)     # block_size acota la memoria si d es grande
        C, info = lw.covariance_, {"shrinkage": lw.shrinkage_}

    elif metodo == "ridge":
        C = np.cov(Z.T)
        w = np.linalg.eigvalsh(C)
        eps = max((w[-1] - kappa_max * w[0]) / (kappa_max - 1.0), 0.0)
        C, info = C + eps * np.eye(d), {"epsilon": eps}

    elif metodo == "pca":
        # Modelo factorial: k factores + residuo diagonal (siempre definido positivo)
        Zc = Z - Z.mean(0)
        _, s, Vt = np.linalg.svd(Zc, full_matrices=False)
        k = k or min(50, n - 1)
        lam = s[:k] ** 2 / (n - 1)
        W = Vt[:k].T * np.sqrt(lam)                                   # (d, k)
        psi = np.maximum(Zc.var(0, ddof=1) - (W ** 2).sum(1), 1e-6)   # residuo positivo
        C = W @ W.T + np.diag(psi)
        info = {"k": k, "var_explicada": lam.sum() / Zc.var(0, ddof=1).sum()}
    else:
        raise ValueError(metodo)
    return C, info

def verificar_covarianza(C):
    """Puerta dura antes de generar: falla pronto y ruidosamente."""
    assert np.allclose(C, C.T, atol=1e-10), "covarianza no simetrica"
    w = np.linalg.eigvalsh(C)
    assert w[0] > 0, f"covarianza no definida positiva: lambda_min={w[0]:.3e}"
    np.linalg.cholesky(C)                          # LinAlgError si no es definida positiva
    return {"lambda_min": w[0], "lambda_max": w[-1], "cond": w[-1] / w[0]}
```

**Criterio de decisión resumido:**

| Situación | Método | Justificación |
|---|---|---|
| $n > 4d$ | Ledoit-Wolf | Sesgo espectral moderado; $\rho$ pequeño y coste asumible |
| $2d < n \le 4d$ | Ledoit-Wolf | $\rho$ crecerá; reportarlo explícitamente |
| $n \le 2d$ (clase crisis) | PCA / factorial con $k \le n-1$ | La muestral es singular; el factorial acota los parámetros a $O(dk)$ |
| Cualquiera | + `verificar_covarianza` | Nunca confiar en el silencio de numpy |

Nunca usar `np.cov` seguido de `multivariate_normal` sin verificación: es el camino directo al fallo silencioso descrito arriba.

### 3.3 Versión condicional: un gaussiano por régimen

$y_{\text{reg}}$ es una etiqueta **categórica** de 3 clases, y un gaussiano no puede generarla: codificarla *one-hot* dentro del bloque aplanado y aplicar `argmax` a la salida produce etiquetas incoherentes con la parte $X$ generada y distorsiona la proporción de clases. La solución correcta es ajustar $p(x \mid y_{\text{reg}}=c) = \mathcal{N}(\hat\mu_c, \hat\Sigma_c)$ por clase y muestrear con la etiqueta fijada.

```python
def ajustar_gaussiano_condicional(Z, y_reg, metodo="auto"):
    """Un gaussiano por regimen. Devuelve dict clase -> (mu, L, info)."""
    modelos = {}
    for c in np.unique(y_reg):
        Zc = Z[y_reg == c]
        C, info = estimar_covarianza(Zc, metodo=metodo)
        info |= verificar_covarianza(C) | {"n_c": Zc.shape[0]}
        modelos[c] = (Zc.mean(0), np.linalg.cholesky(C), info)
    return modelos

def muestrear_condicional(modelos, n_por_clase, rng):
    """n_por_clase: dict clase -> nº de muestras. Permite reequilibrar la clase crisis."""
    Zs, ys = [], []
    for c, (mu, L, _) in modelos.items():
        m = n_por_clase[c]
        Zs.append(mu + rng.standard_normal((m, mu.size)) @ L.T)
        ys.append(np.full(m, c))
    return np.concatenate(Zs), np.concatenate(ys)
```

Ventajas: las etiquetas son exactas por construcción; permite **sobremuestrear la clase crisis** de forma controlada, que es la aplicación más plausible de datos sintéticos aquí; y captura que la matriz de correlaciones cambia entre regímenes (las correlaciones entre activos aumentan en crisis), algo que el gaussiano único promedia y pierde. Coste: la clase minoritaria tiene $n_c = 587 \ll d$, lo que fuerza el método factorial. Es justo donde el modelo es más frágil y más útil, así que hay que reportar $k$, $\rho$ o $\epsilon$ **por clase**, no sólo globalmente.

---

## 4. Bootstrap por bloques estacionario

> Ampliación (no cubierto en clase): toda la sección 4. No aparece en el material del taller. Se incluye porque es el único de los tres baselines que preserva simultáneamente la distribución marginal exacta y la dependencia temporal, y porque su comparación con el gaussiano aísla limpiamente el efecto "colas y no linealidad" frente al efecto "estructura de segundo orden".

### 4.1 Motivación: preservar la dependencia temporal

Los dos baselines anteriores tratan cada ventana de 60 días como un vector intercambiable: el jitter no crea dinámica nueva y el gaussiano crea dinámica nueva pero sólo de segundo orden. El bootstrap por bloques ocupa el hueco: **recombina trozos de historia real**.

La clave es no remuestrear ventanas ya construidas (eso sería duplicación pura), sino **remuestrear bloques de días del panel diario y recortar las ventanas después**. Dos consecuencias: si se remuestrean **fechas completas** (todos los canales a la vez), la estructura transversal se preserva **exactamente**, porque cada día sintético es un día real con todas sus relaciones contemporáneas intactas; y recalculando $y_{\text{reg}}$ y $y_{\text{vol}}$ **sobre la trayectoria remuestreada**, el par $(X,y)$ es coherente por construcción, sin modelar $p(y\mid x)$.

El bootstrap i.i.d. clásico (remuestrear días sueltos) destruiría el clustering de volatilidad y toda la autocorrelación. El de bloques la conserva dentro de cada bloque y sólo la rompe en las uniones.

### 4.2 Formulación y elección del tamaño de bloque

**Bootstrap por bloques móviles (MBB, Künsch 1989).** Se eligen $\lceil T/b\rceil$ bloques de longitud fija $b$ con inicios uniformes en $\{1,\dots,T-b+1\}$ y se concatenan. Inconveniente: la serie resultante no es estacionaria (los índices extremos se muestrean menos).

**Bootstrap estacionario (Politis y Romano 1994).** Las longitudes de bloque son aleatorias, $L\sim\mathrm{Geom}(p)$ con $\mathbb{E}[L]=1/p=b$, y la serie se envuelve circularmente. La serie remuestreada **sí es estacionaria** y su autocovarianza cumple aproximadamente

$$\hat\gamma^*(k) \approx (1-p)^{|k|}\,\hat\gamma(k),$$

es decir, la dependencia se atenúa geométricamente con escala característica $b$. Toda dependencia a plazo mucho mayor que $b$ se pierde.

**Elección de $b$.** Dos criterios en tensión. El estadístico clásico (Politis y White 2004) da un óptimo del orden de $b \propto T^{1/3}$ para estimar medias: con $T\approx5\,700$ sesiones son $\sim18$ días, demasiado corto. El estructural de nuestro problema: cada muestra abarca $60+21=81$ días, y con bloques de longitud media $b$ cada ventana contiene en promedio $81/b$ uniones artificiales — con $b=20$ hay 4 discontinuidades por ventana; con $b=81$, una; con $b=252$, un tercio. Prevalece el estructural: **$b \in [81, 252]$**, entre la longitud completa de la muestra y un año bursátil. Se reporta la sensibilidad a $b$ igual que la sensibilidad a $\alpha$ en el jitter.

```python
def bootstrap_estacionario(panel, T_out, b, rng):
    """Politis-Romano sobre el panel diario. panel: (T, n_canales).
    Remuestrea FECHAS completas -> la estructura transversal se preserva exacta."""
    T, p = panel.shape[0], 1.0 / b
    idx = np.empty(T_out, dtype=np.int64)
    i = rng.integers(0, T)
    for t in range(T_out):
        idx[t] = i
        i = rng.integers(0, T) if rng.random() < p else (i + 1) % T   # nuevo bloque o continua
    return panel[idx], idx

def generar_bootstrap(panel, n_gen, b, w_x, h, rng, construir_ventanas):
    """Recorta ventanas de una trayectoria remuestreada; las etiquetas se RECALCULAN
    sobre esa trayectoria con la misma funcion usada para los datos reales."""
    panel_bs, _ = bootstrap_estacionario(panel, n_gen + w_x + h, b, rng)
    return construir_ventanas(panel_bs)
```

**Versión condicional.** Para sobremuestrear la clase crisis se restringen los inicios de bloque a fechas etiquetadas como crisis. Advertencia honesta que conviene adelantar al tribunal: los períodos de crisis son **contiguos**, de modo que el tamaño muestral efectivo no es el número de días de crisis sino el **número de episodios independientes**, del orden de una decena en el histórico disponible. El bootstrap condicional multiplica filas, no episodios. Conviene reportar ese recuento explícitamente.

---

## 5. Qué hechos estilizados financieros captura cada uno

> Ampliación (no cubierto en clase): esta sección. Los hechos estilizados de retornos no forman parte del material del taller; se incorporan porque son el criterio natural para juzgar el realismo de datos sintéticos financieros más allá de la métrica *downstream*.

| Hecho estilizado | Jitter | Gaussiano MV | Bootstrap por bloques |
|---|---|---|---|
| Colas gruesas (curtosis > 3) | **Sí**, diluidas por $(1+\alpha^2)^{-2}$ | **No, imposible** | **Sí, exactas** |
| Ausencia de autocorrelación en $r_t$ | Sí | Sí (la estima) | Sí |
| Autocorrelación en $\lvert r_t\rvert$ (clustering) | Sí, diluida | **No** (ver abajo) | Sí hasta lag $\sim b$ |
| Correlación transversal entre canales | Sí, atenuada $\rho/(1+\alpha^2)$ | **Sí, exacta** (es lo que ajusta) | **Sí, exacta** |
| Cambio de correlaciones en crisis | Sí (hereda) | Sólo en versión condicional | Sí (hereda) |
| Efecto apalancamiento | Sí, diluido | Parcial (ver abajo) | Sí, dentro del bloque |
| Extrapolación fuera de las muestras | **No** | **Sí** | **No** |

**Gaussiano y colas gruesas: imposibilidad estructural, no error de ajuste.** Toda combinación lineal de un vector gaussiano es exactamente gaussiana. La curtosis marginal generada es exactamente 3, con independencia de la calidad de $\hat\Sigma$ y del número de muestras. Ningún ajuste mejora esto. Es el argumento más fuerte a favor de los generativos neuronales y hay que enunciarlo así.

**Gaussiano y clustering de volatilidad: también imposible, y demostrable.** Para un vector gaussiano centrado, el teorema de Isserlis da

$$\mathrm{Cov}(r_i^2, r_j^2) = 2\,\Sigma_{ij}^2.$$

Como la autocovarianza de los retornos es prácticamente nula ($\Sigma_{ij}\approx 0$ para $i\ne j$ en el eje temporal), el modelo produce $\mathrm{Cov}(r_i^2, r_j^2)\approx 0$: **cero clustering de volatilidad**. El clustering es una dependencia entre *magnitudes*, que la covarianza no puede representar. Es una predicción verificable y da la gráfica de diagnóstico más rentable del trabajo: autocorrelograma de $\lvert r\rvert$ para datos reales, gaussianos y bootstrap superpuestos.

**Matiz importante en nuestro panel.** Nuestro bloque incluye canales explícitos de volatilidad (nivel y variación del VIX, volatilidad realizada, drawdown). El gaussiano **sí** estima y reproduce sus autocovarianzas, de modo que generará trayectorias de VIX persistentes y suaves. Lo que no puede capturar es el **acoplamiento** entre el nivel del VIX y la *amplitud* de los retornos, porque es una dependencia entre un nivel y una magnitud. Modo de fallo concreto y observable: ventanas sintéticas con VIX alto y sostenido pero retornos de amplitud normal, y viceversa. Merece una figura.

Por la misma razón, el **efecto apalancamiento** queda parcialmente capturado: la covarianza cruzada $\mathrm{Cov}(r_t, \mathrm{RV}_{t+k})$ entre el canal de retornos y el de volatilidad realizada **sí** se estima y se reproduce, porque ahí la volatilidad es una variable explícita del panel y no una magnitud implícita. Es un caso donde la ingeniería de características rescata parcialmente al modelo.

**Jitter: fidelidad alta, novedad nula.** Preserva todo hasta $O(\alpha^2)$ precisamente porque casi no cambia nada. Su aportación es regularización (suavizado del modelo alrededor de cada muestra), no información nueva.

**Bootstrap: fidelidad marginal exacta, dinámica de largo plazo rota.** La distribución marginal de cada canal es exactamente la empírica, con sus colas y su asimetría reales. El límite es el largo plazo: la persistencia de la volatilidad decae hiperbólicamente (memoria larga) mientras que la del bootstrap estacionario decae geométricamente como $(1-p)^k$. Ningún $b$ finito reproduce la memoria larga. Y no hay extrapolación: no genera una crisis peor que la peor observada.

**Síntesis defendible.** Los tres baselines cubren un espacio complementario: el jitter aporta regularización local sin extrapolación; el gaussiano aporta extrapolación pero con dependencia estrictamente de segundo orden y colas gaussianas; el bootstrap aporta marginales y dependencia local exactas sin extrapolación. **Ninguno tiene simultáneamente colas gruesas, clustering de volatilidad y extrapolación.** Ese hueco es exactamente lo que un generativo neuronal debe justificar que llena; si no lo llena, no aporta.

---

## 6. Diagnóstico de "convergencia" en modelos sin entrenamiento iterativo

El enunciado exige *"para cada entrenamiento, incluir las curvas de loss donde se vea que el modelo ha convergido"* (`docs/enunciado/Taller_B5_T1.pdf`, entregables). Los tres baselines **no tienen función de pérdida ni bucle de optimización**. La respuesta no es omitir la evidencia sino sustituirla por diagnósticos con la misma función: demostrar que el modelo ha extraído toda la información disponible y que más datos o más cómputo no lo cambian.

> Ampliación (no cubierto en clase): esta sección completa.

**Diagnóstico 1 — estabilidad del estimador frente al número de muestras (sustituto directo de la curva de loss).** Para $m = 100, 200, 500, 1000, \dots, n$ se ajusta el modelo con una submuestra de tamaño $m$ y se mide la distancia a los estimadores con los $n$ datos completos:

$$e_\mu(m) = \frac{\|\hat\mu_m - \hat\mu_n\|_2}{\|\hat\mu_n\|_2}, \qquad e_\Sigma(m) = \frac{\|\hat\Sigma_m - \hat\Sigma_n\|_F}{\|\hat\Sigma_n\|_F}.$$

Se grafica con el eje $x$ logarítmico (nº de muestras, en lugar de época) y el eje $y$ en error relativo. Debe decaer como $O(m^{-1/2})$; **si la curva se ha aplanado, el estimador ha convergido en el mismo sentido operativo que una loss plana**. Repitiendo con varias submuestras por cada $m$ se obtiene una banda, análoga a la varianza entre corridas de entrenamiento.

**Diagnóstico 2 — condicionamiento e intensidad de regularización.** Frente a $m$ se reportan $\lambda_{\min}(\hat\Sigma_m)$, $\mathrm{cond}(\hat\Sigma_m)$ y el coeficiente de shrinkage $\rho_m$ (o $\epsilon_m$, o $k_m$). La tabla de §3.2 muestra la forma esperada: ambos decrecientes. **Un $\rho$ que no baja al añadir datos indica que el modelo está dominado por el prior y no convergerá con los datos disponibles** — que es exactamente lo que hay que reportar para la clase crisis.

**Diagnóstico 3 — convergencia de los momentos de las muestras generadas.** Frente a $n_{\text{gen}} = 100, 300, 1000, 3000, 10000$ se mide la distancia entre los momentos empíricos de las muestras sintéticas y los de las reales (media, desviación típica, asimetría, curtosis, matriz de correlaciones). Aquí aparece lo más informativo: para el gaussiano, el error en media y covarianza tiende a cero, pero **el error en curtosis converge a un suelo distinto de cero**. Ese suelo es el sesgo asintótico del modelo, no ruido de muestreo, y es la demostración numérica del argumento estructural de §5.

**Diagnóstico 4 — estabilidad frente a la semilla.** Repetir la generación completa con $S\ge 5$ semillas y reportar media ± desviación típica de la métrica *downstream*. Sin esto no se puede afirmar que una diferencia entre generadores sea real. Aplica igual a los generadores neuronales.

**Diagnóstico 5 — test de dos muestras con clasificador (C2ST), el puente con las GAN.** Se entrena un clasificador binario real-vs-sintético y se reporta el AUC sobre un conjunto reservado: $\approx 0{,}5$ significa muestras indistinguibles de las reales; $\approx 1{,}0$, trivialmente distinguibles. Esta métrica **es directamente comparable con la loss del discriminador de una GAN**, que mide lo mismo. Permite poner los cuatro generadores en un único eje y es la figura que unifica la comparación.

```python
def curva_convergencia_estimador(Z, tamanos, n_rep, rng, metodo="ledoit_wolf"):
    """Sustituto de la curva de loss: error del estimador frente al nº de muestras."""
    C_ref, _ = estimar_covarianza(Z, metodo=metodo)
    mu_ref, filas = Z.mean(0), []
    for m in tamanos:
        for _ in range(n_rep):                        # repeticiones -> banda de incertidumbre
            sub = Z[rng.choice(Z.shape[0], m, replace=False)]
            C_m, info = estimar_covarianza(sub, metodo=metodo)
            filas.append({
                "m": m,
                "err_mu": np.linalg.norm(sub.mean(0) - mu_ref) / np.linalg.norm(mu_ref),
                "err_cov": np.linalg.norm(C_m - C_ref) / np.linalg.norm(C_ref),
                "cond": np.linalg.cond(C_m),
                "shrinkage": info.get("shrinkage", np.nan),
            })
    return filas
```

**Cómo graficarlo para que sea comparable con los generadores neuronales.** Una rejilla de paneles con formato idéntico para los cuatro modelos: eje $x$ común de *presupuesto* en escala logarítmica (época para los neuronales, nº de muestras de ajuste para los baselines) y eje $y$ común de error normalizado a su valor inicial, para que curvas de magnitudes distintas sean superponibles. Más una figura final común, que es la de las láminas 30–32 del taller: **métrica *downstream* frente a nº de muestras sintéticas añadidas**, una serie por generador. Esa gráfica responde a la pregunta del taller y pone baselines y neuronales a competir en igualdad de condiciones.

En el README debe constar explícitamente que los baselines no tienen curva de loss porque su ajuste es de forma cerrada, y que los diagnósticos 1–5 son la evidencia equivalente de convergencia. Es un punto que el tribunal preguntará.

---

## 7. Aplicación a nuestro problema

**Objeto a generar.** El bloque conjunto $[X; y_{\text{vol}}]$ —con $y_{\text{reg}}$ aparte, como condición—, siguiendo la estrategia de aplanado de `docs/material_clase/notebooks/Taller_Gaussian_solution.ipynb` y la opción OPT2 de la lámina 8 (generar pares entrada-salida, no sólo entradas). $X$ son 60 días × 20 canales derivados de un panel híbrido de 15 activos (S&P 500, 9 SPDR sectoriales, VIX, tesoro a 20 y a 10 años, crédito grado de inversión e índice dólar): once canales de retornos (índice, nueve sectores y dólar) y nueve derivados (nivel y variación del VIX, volatilidad realizada, drawdown, momento, spread de crédito, pendiente de curva, correlación acción-bono y dispersión sectorial). Son 1.200 dimensiones aplanadas más la etiqueta de regresión, $d = 1\,201$.

**Paso obligatorio previo: estandarización por canal.** Es la diferencia principal con el notebook del profesor, donde todos los canales son log-retornos de la misma escala. Sin estandarizar, la covarianza queda dominada por los canales de mayor escala y el shrinkage hacia $\mu I$ carece de sentido (el objetivo presupone escalas comparables), y un $\sigma$ único de jitter es ruido despreciable para unos canales y destructivo para otros. El `StandardScaler` se ajusta **sólo con el conjunto de entrenamiento**; la inversión se hace después de generar. Ajustarlo sobre validación o test es fuga de información y anula la comparación.

| Generador | Configuración | Justificación |
|---|---|---|
| Jitter | $\sigma_j=\alpha\hat\sigma_j$; barrido $\alpha\in\{0{,}01, 0{,}05, 0{,}1, 0{,}25, 0{,}5\}$; punto de operación $\alpha\in[0{,}05, 0{,}15]$ | §2.1: atenuación de correlación $\le 2\,\%$ y dilución de curtosis $<1\,\%$; validar con el criterio del vecino más próximo |
| Gaussiano global | Ledoit-Wolf ($n=3\,696\approx3d$), Cholesky cacheado | §3.2: $n>2d$, LW aplicable; reportar $\rho$ |
| Gaussiano condicional | Un modelo por régimen. Mayoritarias: LW. Crisis ($n_c=587<d$): factorial con $k\in[20,50]$ | §3.2: la muestral es singular; sin esto el "gaussiano" devuelve combinaciones afines de los datos reales sin avisar |
| Bootstrap por bloques | Estacionario, $b\in\{81, 126, 252\}$ días, remuestreo de fechas completas, etiquetas recalculadas | §4.2: $b\ge81$ para limitar a $\le1$ unión por ventana |

**Manejo de las etiquetas.** $y_{\text{reg}}$ es categórica: se genera siempre con modelos condicionales, nunca por `argmax` de una codificación *one-hot* generada. $y_{\text{vol}}$ es continua y positiva: conviene generarla en escala logarítmica y deshacer la transformación al final, para que el gaussiano no produzca volatilidades negativas. El jitter y el bootstrap no tienen este problema (copian o recalculan valores reales); el gaussiano sí, y hay que verificarlo contando cuántas muestras generadas violan restricciones físicas (volatilidad negativa, VIX negativo). **Ese recuento es en sí mismo una métrica de calidad reportable.**

**Presupuesto computacional.** Entorno CPU-only, Python 3.13.7, numpy 2.3.4, scikit-learn 1.8.0. Con los tiempos medidos en §3.1, el coste total de los tres baselines (ajuste más generación de varios *datasets* de hasta 10 000 muestras, incluidas las versiones condicionales) es del orden de **minutos**, frente a horas de un generativo neuronal en CPU. Dos consecuencias prácticas: se pueden permitir barridos completos de $\alpha$, $b$ y $k$ y varias semillas por configuración, algo probablemente inviable para los neuronales. Conviene decirlo en la presentación: parte de la ventaja de los baselines es que **su coste permite cuantificar su propia incertidumbre**, y los neuronales deberían compararse contra la banda, no contra un punto.

**Protocolo de comparación.** Idéntico para los cuatro generadores, según las láminas 14–17 del taller: (1) ajustar el generador **sólo con el conjunto de entrenamiento**; (2) generar $n_g$ muestras para varios $n_g$; (3) entrenar el modelo *downstream* con la **misma arquitectura** y los mismos hiperparámetros sobre cada mezcla real+sintético; (4) evaluar sobre el mismo conjunto de test **exclusivamente real**. Cualquier desviación de este protocolo entre generadores invalida la comparación.

---

## 8. Referencias

**Material del taller (fuente primaria):**

- `docs/enunciado/Taller_B5_T1.pdf` — enunciado B5-T1. Tarea 2 (exigencia del cuarto modelo simple), tarea 5 (comparación incluyendo el modelo simple), entregables (curvas de convergencia).
- `docs/material_clase/notebooks/Taller_Gaussian_solution.ipynb` — generador gaussiano multivariante sobre el bloque aplanado $[X;Y]$ del S&P500. Celdas 16–24.
- `docs/material_clase/notebooks/Taller_GANs.ipynb` — sección *"Ejemplo muy tonto (datos con ruido)"*, celdas 45–50. Modelo simple de referencia del profesor, `sig = 0.01`.
- `docs/material_clase/slides/2026_Taller_Generativos.pdf` — marco de comparación. Láminas 8–9 (opciones OPT1–OPT4), 14–17 (protocolo de 4 pasos), 27–28 (comparación de errores), 30–32 (error frente a nº de reales y de sintéticos).

**Herramientas:** `sklearn.covariance.LedoitWolf` y `sklearn.covariance.OAS` (shrinkage analítico, scikit-learn 1.8.0); `numpy.random.Generator.multivariate_normal`, parámetro `method` (`'svd'` por defecto, `'cholesky'`, `'eigh'`) — comportamiento ante covarianzas no definidas positivas verificado empíricamente en §3.2 con numpy 2.3.4.

> Ampliación (no cubierto en clase): referencias bibliográficas de apoyo para las secciones 3.2, 4 y 5.

- Ledoit, O. y Wolf, M. (2004). *A well-conditioned estimator for large-dimensional covariance matrices*. Journal of Multivariate Analysis, 88(2), 365–411.
- Künsch, H. R. (1989). *The jackknife and the bootstrap for general stationary observations*. Annals of Statistics, 17(3), 1217–1241.
- Politis, D. N. y Romano, J. P. (1994). *The stationary bootstrap*. JASA, 89(428), 1303–1313.
- Politis, D. N. y White, H. (2004). *Automatic block-length selection for the dependent bootstrap*. Econometric Reviews, 23(1), 53–70.
- Cont, R. (2001). *Empirical properties of asset returns: stylized facts and statistical issues*. Quantitative Finance, 1(2), 223–236.
- Lopez-Paz, D. y Oquab, M. (2017). *Revisiting classifier two-sample tests*. ICLR. (Base del diagnóstico C2ST de §6.)
- Tipping, M. E. y Bishop, C. M. (1999). *Probabilistic principal component analysis*. JRSS-B, 61(3), 611–622. (Base del muestreo factorial de §3.2.)
