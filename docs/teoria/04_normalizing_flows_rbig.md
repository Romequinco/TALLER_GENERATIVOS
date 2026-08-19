# Normalizing flows y RBIG

## 1. Intuición: transformar una distribución compleja en una gaussiana

Estimar $p(x)$ por histograma multidimensional es inviable: con 7 bins por dimensión hacen
falta $7^{N_{dim}}$ celdas, y poblarlas exige del orden del cuadrado de ese número de muestras
(`docs/material_clase/slides/Normalizing Flows_2026.pdf`, diap. 3).

Los *flows* no modelan $p(x)$ directamente: buscan una **transformación invertible $f$ que
lleve los datos a una distribución conocida y sencilla**, típicamente $p(z)=\mathcal{N}(0,I)$
(diap. 4). Con $f$, estimar densidad se reduce a evaluar la gaussiana en $z=f(x)$ corrigiendo
por la deformación de volumen, y generar se reduce a muestrear $z\sim\mathcal{N}(0,I)$ y
aplicar $f^{-1}$ (portada del deck: *Flow* / *Flow Inverse*). Toda la dificultad se traslada a
construir $f$. Esto los distingue del resto de generativos del taller: una GAN genera pero no
evalúa $p(x)$ y un VAE solo evalúa una cota inferior, mientras que un flow da **verosimilitud
exacta y muestreo exacto** con el mismo objeto, a cambio de imponer que $f$ sea invertible.

## 2. Cambio de variable y log-determinante del jacobiano

Dada $f$ invertible y diferenciable con $z=f(x)$, la fórmula de cambio de variable (diap. 5):

$$p_x(x) = p_z\big(f(x)\big)\left|\det \frac{\partial f(x)}{\partial x}\right|$$

El determinante del jacobiano mide **cuánto estira o comprime el dominio $dx$ al transformarse
en $dz$** (diap. 7-8): es el factor que conserva la masa de probabilidad. En logaritmos,
$\log p_x(x) = \log p_z(f(x)) + \log\left|\det J_f(x)\right|$. Entrenar un flow paramétrico es
**maximizar la log-verosimilitud en el dominio transformado** (diap. 9), es decir minimizar

$$\mathcal{L}(\theta) = -\frac{1}{N}\sum_{n}\left[\log p_z\big(f_\theta(x_n)\big) + \log\left|\det J_{f_\theta}(x_n)\right|\right]$$

El deck apunta que también caben pérdidas adversariales, aunque no es lo habitual (diap. 9).

**Composición.** Un único $f$ expresivo es difícil de diseñar; se encadenan transformaciones
sencillas $f = f_L\circ\cdots\circ f_1$ (diap. 10-12). Como el determinante de una composición
es el producto de determinantes, en logaritmos los términos se suman:

$$\log\left|\det J_f(x)\right| = \sum_{\ell=1}^{L}\log\left|\det J_{f_\ell}(x^{(\ell-1)})\right|$$

Basta con que cada capa sea invertible y de jacobiano barato; la profundidad aporta la
expresividad. **Requisitos de una capa** (diap. 13) — *obligatorio*: invertible, diferenciable
y suficientemente expresiva; *deseable*: eficiente en las tres operaciones (directa, inversa y
jacobiano). La tensión entre esos tres costes genera las familias de la sección 3.

## 3. Familias de flujos (planar, coupling/RealNVP, autoregresivos)

El deck organiza las familias según **la estructura impuesta al jacobiano** para que su
determinante sea calculable (diap. 15):

| Estructura del jacobiano | Familia | Coste del determinante |
|---|---|---|
| Diagonal | Element-wise (independiente) | $O(D)$ |
| Identidad + rango bajo | Planar / Sylvester | $O(D)$ (lema del determinante) |
| Dispersa por bloques | Coupling (RealNVP, Glow) | $O(D)$ |
| Triangular inferior | Autoregresivos | $O(D)$, producto de la diagonal |
| Libre (free-form) | Estimación insesgada | Aproximado |

**Element-wise.** Una función escalar por dimensión, $z_i=\psi_i(x_i)$. Jacobiano diagonal,
$\log|\det J| = \sum_i \log|\psi_i'(x_i)|$. Es la capa más barata posible y por sí sola solo
remodela las marginales: no crea ni destruye dependencia. Es el primer paso de RBIG.

**Planar.** Perturba la identidad con una corrección de rango uno.

> Ampliación (no cubierto en clase): $f(x)=x+u\,h(w^\top x+b)$, con
> $\log|\det J| = \log\left|1 + h'(w^\top x+b)\,u^\top w\right|$ por el lema del determinante
> matricial, coste $O(D)$. Su inversa no tiene forma cerrada general, lo que la relega a
> inferencia variacional en vez de generación.

**Coupling (RealNVP, Glow).** El deck los nombra en la tabla resumen (diap. 20) y los sitúa
como jacobiano disperso (diap. 15).

> Ampliación (no cubierto en clase): se deja un bloque intacto y el otro se transforma
> condicionado a él, $y_{d+1:D}=x_{d+1:D}\odot\exp\big(s(x_{1:d})\big)+t(x_{1:d})$, de donde
> $\log|\det J|=\sum_j s_j(x_{1:d})$ sea cual sea la complejidad de $s$ y $t$. La inversa es
> explícita y del mismo coste que la directa: única familia con muestreo y densidad igual de
> baratos. Se alternan las particiones entre capas.

**Autoregresivos.** Cada dimensión se transforma condicionada a las anteriores.

> Ampliación (no cubierto en clase): $y_i=\mu_i(x_{<i})+\sigma_i(x_{<i})\,x_i$ da jacobiano
> triangular inferior y $\log|\det J|=\sum_i\log\sigma_i(x_{<i})$. Más expresivos que los
> coupling pero asimétricos: una dirección exige $D$ pasadas secuenciales (MAF es rápido
> evaluando densidad y lento generando; IAF al revés).

**Flow matching.** El deck cierra con esta variante (diap. 17-19): en lugar de aprender la
transformación de golpe se aprende **la velocidad del cambio**, $v=M(x_t,t)$, integrando
$x_{t+1}=x_t+v$; "normalizing flows pero en suave". La tabla comparativa (diap. 20) explica su
popularidad: los flows clásicos sufren la maldición de la dimensión en memoria y coste del
jacobiano y son inestables de entrenar, mientras flow matching escala como difusión con un
objetivo acotado. Esa limitación — dimensionalidad baja/media — es el eje de la sección 8.

## 4. RBIG: gaussianización marginal + rotación

RBIG (*Rotation-Based Iterative Gaussianization*) es un método del grupo IPL-UV de la
Universitat de València, el mismo grupo del profesor del taller. El deck introductorio lo lista
entre los generativos anteriores al deep learning
(`docs/material_clase/slides/2026_Intro_Generative_Models.pdf`, diap. 9) y el deck de flows lo
presenta en la diap. 16 con dos referencias: Chen & Gopinath (NeurIPS 2000) y Laparra et al.
(2011). La receta de esa diapositiva es engañosamente simple: **gaussianización marginal por
dimensión (histograma) + rotación (PCA)**, iterado.

### 4.1 El algoritmo

Sea $x^{(\ell)}$ la entrada de la capa $\ell$. Cada capa encadena tres bijectores (verificado
en `rbig/_src/training.py`):

**(a) Uniformización marginal.** Por dimensión se estima la CDF empírica $\hat F_i$ con
histograma y se aplica $u_i=\hat F_i(x_i^{(\ell)})\in[0,1]$, aproximadamente uniforme sea cual
sea la marginal original. El soporte se extiende un 30 % a cada lado del rango observado
(`bound_ext=0.3`) y se suma $\alpha=10^{-10}$ a la pdf para que la CDF sea estrictamente
creciente e invertible.

**(b) Probit.** $g_i=\Phi^{-1}(u_i)$. Tras (a)+(b) **cada marginal es exactamente
$\mathcal{N}(0,1)$**, pero las variables siguen siendo dependientes. Juntos, (a) y (b) son una
capa *element-wise* de la sección 3: jacobiano diagonal, sin efecto sobre la dependencia.

**(c) Rotación.** $x^{(\ell+1)}=R\,g$ con $R$ ortonormal, obtenida por PCA (por defecto), ICA
(vía `python-picard`) o aleatoria. Al ser ortonormal, $\det R=\pm1$ y su contribución al
log-determinante es **cero**: el método `gradient()` de las clases de rotación devuelve
literalmente un vector de ceros.

### 4.2 Por qué funciona

El reparto de papeles es la clave. El paso (a)+(b) elimina toda la no-gaussianidad *marginal*
pero no toca la dependencia. El paso (c) no cambia la dependencia total, pero la
**redistribuye**: la rotación mezcla las variables, de modo que lo que era dependencia
puramente conjunta reaparece como no-gaussianidad marginal en el nuevo sistema de coordenadas,
y la capa siguiente ya puede eliminarla. Iterando, cada capa extrae algo más de estructura. El
resultado de Laparra et al. (2011) es que el procedimiento **converge para cualquier rotación**,
incluso aleatoria; PCA o ICA afectan a la velocidad, no a que converja.

### 4.3 No hay descenso de gradiente

Este es el punto que más lo separa del resto de modelos del taller y el que condiciona la
sección 6. **Cada capa se ajusta en forma cerrada** con los datos que recibe: el histograma se
calcula y la PCA se resuelve por descomposición espectral, sin parámetros que se actualicen.
**No hay pérdida que minimizar, ni épocas, ni learning rate, ni backpropagation, ni pesos
inicializados al azar.** Las capas se ajustan **secuencialmente y una sola vez** — la capa
$\ell$ ve la salida ya fija de la $\ell-1$ y nunca se revisita — y su número no se busca por
validación: **se determina solo** (sección 6.3).

Por tanto no aparecen las patologías del entrenamiento adversarial o variacional: ni colapso de
modos por dinámica, ni divergencia, ni sensibilidad a la semilla de optimización. Las
limitaciones de RBIG son de otra naturaleza (sección 7).

## 5. Invertibilidad, muestreo y estimación de densidad

Los tres bloques son invertibles explícitamente:

| Paso | Directo | Inverso |
|---|---|---|
| Uniformización | $\hat F_i(x_i)$ | $\hat F_i^{-1}(u_i)$ (ppf del histograma) |
| Probit | $\Phi^{-1}(u_i)$ | $\Phi(g_i)$ |
| Rotación | $R\,g$ | $R^\top x$ |

Invertir el modelo es recorrer las capas al revés. El notebook lo comprueba explícitamente
(`docs/material_clase/notebooks/2_RBIG_demo_2D.ipynb`, celda 9) con
`np.testing.assert_array_almost_equal(data, data_approx)`. Reproducido en el entorno del
proyecto (Python 3.13.7, numpy 2.3.4), el error máximo sobre 10.000 puntos es
$6{,}1\times10^{-13}$: invertibilidad numéricamente exacta.

**Muestreo.** Es el uso que interesa para el taller: el dominio transformado es gaussiano
factorizado, así que se muestrea ahí y se invierte (celdas 14-16):

```python
# 1. muestrear en el dominio gaussiano (trivial: normales independientes)
z_sintetico = rng.randn(n_muestras, n_dimensiones)
# 2. devolver al dominio de los datos con el flujo inverso
x_sintetico = rbig_model.inverse_transform(z_sintetico)
```

La librería expone además `rbig_model.sample(n)`, que encapsula ambos pasos.

**Densidad.** `predict_proba(X)` evalúa $p(x)$ aplicando el cambio de variable de la sección 2;
`log_det_jacobian(X)` devuelve solo el log-determinante. El notebook lo usa para colorear los
datos por densidad (celdas 20-21) y, más interesante, para evaluar puntos que **no** provienen
de la distribución: en `2_RBIG_demo_2D_clase.ipynb` (celdas 22-27) se evalúan muestras frescas
del mismo generador y luego ruido uniforme, comprobando que el modelo asigna densidad alta a la
variedad de los datos y baja fuera de ella. Es una comprobación de generalización que
reutilizamos en la sección 6.

## 6. Diagnóstico de convergencia

El enunciado exige, para cada entrenamiento, "las curvas de loss donde se vea que el modelo ha
convergido" (`docs/enunciado/Taller_B5_T1.pdf`, pág. 2). **En RBIG no existe curva de loss**,
porque no hay optimización (sección 4.3). El equivalente legítimo, y el que usa el notebook de
clase, es la **curva de reducción de multi-información acumulada**.

### 6.1 Qué mide la curva

La magnitud relevante es la **multi-información** o **correlación total**, que cuantifica
cuánta dependencia estadística hay entre las componentes:

$$T(x) = \sum_{i=1}^{D} h(x_i) - h(x) = D_{KL}\Big(p(x)\ \Big\|\ \prod_{i=1}^{D}p(x_i)\Big)$$

Es no negativa y vale cero si y solo si las componentes son independientes. Como el objetivo de
RBIG es llegar a una gaussiana **factorizada**, el trabajo pendiente en cualquier momento es
exactamente $T$ del estado actual.

> Ampliación (no cubierto en clase): la reducción por capa admite forma cerrada. Las
> transformaciones dimensión a dimensión (a y b) **no alteran** $T$, porque el cambio en la
> entropía conjunta iguala la suma de los cambios marginales; la rotación es ortonormal y
> preserva la entropía conjunta. Si $g$ es el estado tras el probit (marginales exactamente
> $\mathcal{N}(0,1)$, entropía $h_\mathcal{N}=\tfrac12\log 2\pi e$ cada una) y $y=Rg$:
> $$\Delta T_\ell = T(x^{(\ell)}) - T(x^{(\ell+1)}) = D\,h_\mathcal{N} - \sum_{i=1}^{D} h(y_i)$$
> no negativa porque la gaussiana maximiza la entropía a varianza dada. La reducción de cada
> capa es, literalmente, cuánta no-gaussianidad marginal ha logrado destapar la rotación.

En la implementación, `information_reduction` (`rbig/_src/total_corr.py`) estima esa cantidad
por capa a partir de entropías marginales por histograma corregidas de sesgo (Miller-Madow) y
la guarda en el atributo `info_loss`. La curva del notebook (celda 12) es su suma acumulada:

```python
fig, ax = plt.subplots()
ax.plot(np.cumsum(rbig_model.info_loss), 'o-')
ax.set_title('Information Reduction')
plt.show()
```

### 6.2 Cómo se lee

`np.cumsum(info_loss)` es la **información total eliminada hasta la capa $L$**, en nats. Sus
propiedades la hacen un diagnóstico mucho más limpio que una loss adversarial:

- **Monótona no decreciente** por construcción: ninguna capa puede añadir dependencia.
- **Acotada superiormente** por $T(x)$, la dependencia realmente presente. El plateau es por
  tanto una estimación de $T(x)$: `total_correlation()` devuelve exactamente `info_loss.sum()`.
- **Converge cuando se aplana.** El plateau significa que las capas adicionales ya no
  encuentran estructura: los datos transformados son gaussianos factorizados.

Ejecutando el notebook en el entorno del proyecto:

```
capas retenidas : 14
info_loss : [0.3965 0.1082 0.1306 0.0391 0.0155 0. 0. 0. 0. 0. 0. 0. 0. 0.0134]
cumsum    : [0.3965 0.5046 0.6353 0.6744 0.6899 ... 0.6899 0.7033]
correlacion total estimada : 0.7033 nats
```

Lectura inmediata: **las tres primeras capas capturan el 90 % de la dependencia**, desde la
quinta la curva es plana y el modelo efectivo tiene 14 capas. El valor no nulo aislado en la
capa 14 (0,0134) tras una racha de ceros es ruido del estimador de entropía, no estructura
real: conviene saberlo para no sobreinterpretar la cola de la curva.

Una advertencia importante: lo que se interpreta con seguridad es **la forma** de la curva
(dónde satura). Su **altura** solo es comparable entre modelos ajustados con el mismo número de
muestras, porque el estimador de entropía por histograma se sesga al alza cuando $n$ baja
(cifras medidas en 9.4).

### 6.3 Cuándo dejar de añadir capas

El criterio está codificado en `train_rbig_info_loss` y es explícito:

```python
if ilayer > zero_tolerance:
    if np.sum(np.abs(info_losses[-zero_tolerance:])) == 0:
        info_losses = info_losses[:-zero_tolerance]          # descarta capas inutiles
        transformations = transformations[:-3*zero_tolerance]
        break
```

Se para cuando **las últimas `zero_tolerance` capas (60 por defecto) han aportado exactamente
cero**, y esas 60 se descartan. "Exactamente cero" no es literal: `information_reduction`
**umbraliza a cero** toda reducción por debajo de una tolerancia que depende del número de
muestras (interpolada de una tabla en función de $n$), es decir, cuando la reducción medida ya
no se distingue estadísticamente de cero con el tamaño muestral disponible.

Dos consecuencias prácticas:

1. `max_layers=1000` no implica ajustar 1000 capas: en el ejemplo se ajustaron 74 y se
   retuvieron 14.
2. Pero el modelo **siempre ajusta al menos 61 capas** antes de poder parar. Con
   `zero_tolerance=60` ese es el suelo de coste, y en dimensión alta domina el presupuesto
   (sección 8).

### 6.4 Contraste con el criterio de convergencia de una GAN o un VAE

| | GAN | VAE | RBIG |
|---|---|---|---|
| Curva | Loss de $D$ y de $G$ | ELBO (recons. + KL) | Multi-información acumulada |
| Monotonía | No; oscilan por construcción | Sí (a la baja) | Sí (al alza) |
| ¿Loss baja = buen modelo? | **No** | No necesariamente | Sí, el plateau es interpretable |
| Óptimo esperado | Equilibrio, $D$ acierta ~50 % | Cota inferior, sin valor absoluto | Plateau $=T(x)$, valor absoluto |
| Parada | Heurística / inspección visual | Early stopping en validación | Automática y determinista |
| Fallos típicos | Colapso de modos, divergencia | Colapso posterior | Ninguno *de entrenamiento* |

El contraste está en el propio material. Las diapositivas 18-24 de
`docs/material_clase/slides/2026_Taller_Generativos.pdf` muestran la traza de la GAN del
profesor: la loss del discriminador baja de 0,748 a 0,407 mientras su *accuracy* sube del 42 %
al 80 %, y la loss del generador **sube** de 0,731 a 1,365 con su accuracy cayendo al 13 %.
Ninguna de esas cifras dice por sí sola si el modelo genera bien: hay que mirar los scatter de
reales frente a generados (diap. 25-26) y, en última instancia, el rendimiento downstream
(diap. 27-28). En una GAN la loss **no** es prueba de convergencia.

En RBIG ocurre lo contrario: la curva de información acumulada es en sí misma la prueba,
porque su plateau tiene significado absoluto (toda la dependencia extraíble ha sido extraída) y
no relativo a un adversario que también se mueve.

**Recomendación para el entregable.** Presentar `np.cumsum(info_loss)` como curva de
convergencia de RBIG, con una nota explicando la equivalencia, y acompañarla de dos
diagnósticos comparables entre modelos: (1) **gaussianidad del dominio transformado** —
comprobar que $Z=f(X)$ tiene media $\approx0$, desviación $\approx1$, covarianza $\approx I$ y
marginales normales (Q-Q plot), que es la verificación directa del objetivo del método; y
(2) **log-verosimilitud sobre validación** vía `predict_proba`, siguiendo el patrón de las
celdas 22-27 de `2_RBIG_demo_2D_clase.ipynb`, que detecta sobreajuste — el riesgo real de RBIG
(sección 7) — y es la única de las tres métricas comparable contra un VAE o un flow entrenado
por gradiente.

## 7. Limitaciones y patologías

**Soporte acotado.** El histograma cubre el rango observado extendido un 30 %
(`bound_ext=0.3`); fuera de ahí la CDF es constante y la inversa satura. Un RBIG **no puede
generar valores más extremos que los observados más ese margen**. Para datos financieros es una
limitación de fondo: el modelo no inventará una caída peor que la peor del histórico. Sirve
para densificar el interior de la distribución, no para explorar la cola.

**Estimación por histograma con pocas muestras.** Toda la capacidad reside en $D$ histogramas
univariantes por capa. Con $n$ pequeño los bins son ruidosos, la CDF reproduce los accidentes
de la muestra y el modelo **memoriza**: como la transformación es invertible y determinista, un
RBIG sobreajustado devuelve muestras que son los datos originales suavizados por la anchura del
bin. Es el fallo dominante en la clase minoritaria (sección 8). El mismo sesgo contamina la
métrica de convergencia, que se infla al reducirse $n$ (cifras medidas en 9.4): la curva no
avisa del sobreajuste, lo disfraza de buen ajuste.

**Maldición de la dimensión.** Coincide con lo que el deck señala para los flows clásicos
(diap. 20: "Dimensionalidad Baja / Media"). Además del coste, aparece un problema de rango: la
rotación PCA necesita $n>D$ para estar bien determinada; con $n<D$ es degenerada y las
direcciones de varianza nula introducen inestabilidad numérica.

**Truncamiento en las colas.** `InverseGaussCDF` recorta $u$ a $[\epsilon,1-\epsilon]$ con
$\epsilon=10^{-10}$ antes del probit; los puntos más extremos saturan al mismo $z$,
comprimiendo artificialmente las colas.

**Ruido del estimador de información.** Como se vio en 6.2, `info_loss` puede dar valores no
nulos aislados después de haberse aplanado. La ventana de 60 capas del criterio de parada es
robusta a eso, a costa de ajustar 60 capas de más.

**Elección de la rotación.** PCA es determinista y barata pero puede estancarse cuando la
estructura remanente no está alineada con las direcciones de máxima varianza; ICA converge en
menos capas pero cada capa es mucho más cara; las rotaciones aleatorias convergen igualmente
(Laparra et al., 2011) y son las más baratas por capa, a cambio de más capas.

Además: **no es condicional de forma nativa** (sección 8.4); **la memoria crece linealmente con
las capas** (cada una guarda $D$ histogramas y una matriz $D\times D$); y **el software de
referencia no recibe commits desde marzo de 2023** (sección 9.1).

## 8. Aplicación a nuestro problema

### 8.1 Encaje en el taller

El taller B5-T1 pide tres modelos generativos distintos más un modelo simple de referencia, y
medir si los datos sintéticos mejoran un modelo downstream. RBIG está especialmente
justificado:

- Es un generativo **no basado en redes ni en gradiente**: aporta contraste metodológico real
  frente a GAN, VAE o difusión.
- Es **rápido en dimensión moderada** y determinista, lo que facilita la reproducibilidad.
- Da **densidad exacta**, útil para diagnósticos que los demás modelos no permiten.
- Es el método del grupo del profesor y su resultado de referencia está en el material: en el
  problema de estimación de temperatura superficial a partir de observaciones IASI
  (`docs/material_clase/slides/2026_Taller_Generativos.pdf`, diap. 27-28) el error downstream es
  **4,81 K solo con datos reales, 3,46 K añadiendo datos de GAN y 3,62 K añadiendo datos de
  RBIG**. RBIG queda a un 5 % de la GAN sin entrenamiento adversarial. Las diapositivas 30-32
  repiten la comparación barriendo la cantidad de datos reales (500, 1.000, 3.000, 7.000,
  20.000 y todos), que es exactamente el eje de análisis que pide el enunciado.

### 8.2 El problema de escalabilidad

El bloque conjunto es $[X; y_{vol}]$ —$y_{reg}$ va aparte, como condición— con $X$ de 60 días
$\times$ 20 canales de un panel híbrido de 15 activos (S&P 500, 9 SPDR sectoriales, VIX, tesoro
a 20 y a 10 años, crédito grado de inversión e índice dólar): once canales de retornos (índice,
nueve sectores y dólar) y nueve derivados (nivel y variación del VIX, volatilidad realizada,
drawdown, momento, spread de crédito, pendiente de curva, correlación acción-bono y dispersión
sectorial). Aplanado, **$D = 1.201$**, muy por encima del régimen cómodo de RBIG.

El coste por capa suma: $D$ histogramas univariantes, $D$ evaluaciones de CDF/ppf, una PCA
$D\times D$ y $2D$ estimaciones de entropía. Medido directamente en el entorno del proyecto
(CPU, sin CUDA, rotación PCA); las dos filas de dimensión completa están escaladas desde la
medida inmediatamente inferior con el exponente empírico $\approx D^{1{,}3}$ que dan las cuatro
primeras:

| $D$ | $n$ | s/capa | Mínimo 61 capas |
|---:|---:|---:|---:|
| 32 | 1.500 | 0,34 | 0,3 min |
| 64 | 1.500 | 0,64 | 0,6 min |
| 128 | 1.500 | 1,26 | 1,3 min |
| 512 | 1.500 | 8,47 | 8,6 min |
| **1.201** | **1.500** | **~25** | **~25 min** |
| 1.201 | 3.000 | ~40 | ~41 min |

Como la estrategia condicional exige **un RBIG por régimen** (3 modelos), el presupuesto real a
dimensión completa es de **75 a 120 minutos por experimento**, repetido cada vez que se cambie
una decisión de diseño. A eso se suma la memoria: solo las rotaciones ocupan
$1201^2\times8$ bytes $=11{,}5$ MB por capa, es decir **700 MB para 61 capas y por régimen**, más
73.261 objetos `rv_histogram` de scipy.

El problema de rango es más serio que el de tiempo. Para que los $D$ histogramas y la PCA estén
bien estimados hace falta, como regla de trabajo, $n\gtrsim10D$: **12.000 ventanas por
régimen**. Con panel diario y ventanas de 60 días no hay tantas ventanas ni en el total —son
5.590 en todo el panel y 3.696 en train—, y menos en el régimen de crisis (~16 % en train,
10,5 % en test). A dimensión completa el RBIG de crisis tendría 587 ventanas para 1.201
dimensiones: PCA degenerada e histogramas que memorizan.
**Ajustar RBIG sobre las 1.201 dimensiones crudas no es defendible.**

### 8.3 Mitigaciones

Por relación beneficio/coste:

1. **PCA previa (la medida decisiva).** Proyectar a $k\approx32$–$64$ componentes reteniendo el
   95-99 % de la varianza, ajustar RBIG ahí y deshacer la proyección al generar. El panel es
   masivamente redundante — ventanas solapadas, sectores muy correlacionados con el índice — así
   que la pérdida es pequeña. El ahorro es un factor $\approx40$ (~25 a 0,64 s/capa) y,
   sobre todo, cambia el requisito muestral de 12.000 a unas 300-600 ventanas por régimen, que
   sí es alcanzable. Es coherente con RBIG, que ya usa PCA internamente.
2. **Reducir `zero_tolerance`.** El suelo de 61 capas es puro coste: con la curva saturando en
   menos de 10 capas, una ventana de 10 en lugar de 60 divide el tiempo por ~5 sin afectar al
   modelo retenido. Fijar también `max_layers` en 30-50.
3. **Submuestrear la ventana temporal.** 60 días diarios son muy redundantes: 1 de cada 3 días,
   o una ventana de 20, baja $D$ de 1.200 a 400 o menos antes de cualquier otra medida.
4. **Ajuste por bloques.** RBIG por grupos de canales con sentido económico (renta variable,
   volatilidad, crédito/curva) y composición posterior. Rompe la dependencia entre bloques, así
   que solo vale si esa dependencia se modela aparte; peor que la PCA previa, pero útil si se
   quiere preservar la interpretabilidad de los canales.
5. **Rotación aleatoria en vez de PCA**, que evita la descomposición espectral por capa con
   convergencia garantizada (Laparra et al., 2011) a cambio de más capas; y **`float32`**, que
   halva la memoria pero exige la implementación propia de la sección 9 (la librería usa
   `float64`).

### 8.4 Estrategia condicional: un RBIG por régimen

$y_{reg}$ es el régimen de mercado a 21 días, 3 clases, con "crisis" minoritaria (~16 % en
train, 10,5 % en test). RBIG
no admite condicionamiento nativo, así que se ajusta **un modelo independiente por clase**:

$$\text{RBIG}_k\ \text{ajustado sobre}\ \{\,[X_i;y_{vol,i}]\ :\ y_{reg,i}=k\,\},\quad k\in\{0,1,2\}$$

y se muestrea de cada uno etiquetando con su $k$. Implicaciones:

- **$y_{reg}$ sale del bloque.** Al condicionar por clase, la etiqueta es el índice del modelo,
  no una dimensión a generar; se evita que RBIG produzca valores continuos donde debería haber
  un one-hot. $y_{vol}$, continua, permanece en el bloque y se genera conjuntamente con $X$,
  preservando por construcción la coherencia entre serie y objetivo, que es la razón de generar
  el bloque conjunto.
- **Control explícito de la mezcla.** Se genera la proporción que se quiera de cada clase. Eso
  convierte la limitación en ventaja: permite **rebalancear la clase de crisis**, justo donde el
  modelo downstream tiene menos datos y más margen de mejora.
- **El requisito muestral se aplica por clase, no al total.** Es la restricción que manda: con
  $N$ ventanas de train, crisis aporta $N_{crisis}\approx0{,}16N$, y ese es el $n$ del histograma. Refuerza
  la mitigación 1: sin reducción de dimensión previa, el modelo de crisis es inservible.
- **Riesgo de sobreajuste en la clase rara.** Con pocos cientos de muestras RBIG reproducirá la
  muestra de crisis casi literalmente y los "sintéticos" serán copias suavizadas: el downstream
  no gana información y la validación puede parecer buena por fuga. Contramedidas: dimensión
  reducida agresiva ($k\approx16$–$32$ para esta clase), histogramas con menos bins (`bins=20`
  en vez de `'auto'`), limitar capas y **evaluar con log-verosimilitud sobre ventanas de crisis
  no vistas**, además del downstream.
- **Alternativa descartada.** Un único RBIG con $y_{reg}$ incluido y condicionamiento por
  rechazo es ineficiente para una clase del 16 % en train (10,5 % en test) y no garantiza
  etiquetas válidas.

Como control conviene reportar además un RBIG único no condicional: si el condicional por
régimen no mejora al downstream frente al único, la complejidad añadida no se justifica.

## 9. Implementación de referencia (CPU)

### 9.1 Viabilidad de la librería en Python 3.13.7

El notebook de clase instala `python-picard` y
`pip install "git+https://github.com/IPL-UV/rbig.git"` (celda 1).

**Verificado en el entorno del proyecto** (Python 3.13.7, numpy 2.3.4, scikit-learn 1.8.0,
scipy 1.16.3, Windows, sin CUDA): **la librería funciona**, pero la instalación falla en Windows
por un motivo ajeno al algoritmo. `setup.py` abre `README.md` sin especificar codificación, así
que Python usa el códec local (cp1252) y la construcción del wheel aborta con
`UnicodeDecodeError: 'charmap' codec can't decode byte 0x9d`. La solución es forzar UTF-8:

```bash
PYTHONUTF8=1 pip install "git+https://github.com/IPL-UV/rbig.git"
```

```powershell
$env:PYTHONUTF8 = "1"; pip install "git+https://github.com/IPL-UV/rbig.git"
```

Con esa variable el wheel (`py_rbig-0.0.1`) se construye y todo funciona: `fit_transform`,
`inverse_transform` (error máximo $6{,}1\times10^{-13}$), `predict_proba`, `log_det_jacobian`,
`sample` y la rotación ICA vía `python-picard`. No hay uso de alias eliminados en numpy 2
(`np.trapz`, `np.float_`…), que era el riesgo principal.

*Advertencias:* el repositorio **no recibe commits desde el 18 de marzo de 2023** y acumula 16
issues abiertos (no está archivado, pero tampoco mantenido); la versión es `0.0.1` y el paquete
se llama `py_rbig` (se importa como `rbig`); no declara `python_requires` ni fija versiones en
`requirements.txt`, de modo que una instalación limpia arrastra numpy, scipy, scikit-learn,
astropy, statsmodels, matplotlib, seaborn e ipykernel en sus últimas versiones — **conviene
fijar versiones**. `astropy` y `statsmodels` solo hacen falta para opciones que no usamos.

**Veredicto: viable con el workaround `PYTHONUTF8=1`.** Aun así conviene disponer de la
implementación propia de 9.3, porque necesitamos controlar `zero_tolerance`, `float32` y el
número de bins con más finura de la que ofrece la API, y porque elimina la dependencia de un
repositorio sin mantenimiento.

### 9.2 Uso de la librería

```python
import numpy as np, matplotlib.pyplot as plt
from rbig import RBIG

# ajuste: no hay epocas ni learning rate, solo capas
modelo = RBIG(rotation="pca", max_layers=1000, zero_tolerance=60)
Z = modelo.fit_transform(X)                      # X: (n_muestras, n_dimensiones)

# curva de convergencia (equivalente a la curva de loss)
plt.plot(np.cumsum(modelo.info_loss), "o-")
plt.xlabel("capa"); plt.ylabel("multi-informacion eliminada (nats)")
print("capas retenidas: %d, correlacion total: %.4f nats"
      % (len(modelo.info_loss), modelo.total_correlation()))

# comprobaciones y generacion
print("error de inversion: %.2e" % np.abs(X - modelo.inverse_transform(Z)).max())
print("gaussianidad de Z: media %.3f, std %.3f" % (Z.mean(), Z.std()))
X_sintetico = modelo.inverse_transform(np.random.randn(5000, X.shape[1]))
```

### 9.3 Plan B: implementación propia

RBIG es corto de implementar y no depende de nada exótico: CDF empírica por histograma, probit
y rotación PCA, iterado. Esta versión se ha verificado en el entorno del proyecto (error de
inversión $1{,}4\times10^{-13}$; sobre los datos del notebook converge en 4 capas reproduciendo
media, desviación y correlación de los originales).

```python
import numpy as np
from scipy.special import ndtri, ndtr
from sklearn.decomposition import PCA

H_NORMAL = 0.5 * np.log(2 * np.pi * np.e)      # entropia de una N(0,1)


def _entropia_hist(col, n_bins=128):
    """Entropia diferencial por histograma con correccion de sesgo Miller-Madow."""
    cuentas, bordes = np.histogram(col, bins=n_bins)
    n = cuentas.sum(); p = cuentas / n; nz = p > 0
    H = -np.sum(p[nz] * np.log(p[nz])) + np.log(bordes[1] - bordes[0])
    return H + 0.5 * (nz.sum() - 1) / n


def _tolerancia(n_muestras, d):
    """Umbral por debajo del cual la reduccion no se distingue de cero (depende de n)."""
    xxx = np.logspace(2, 8, 7)
    yyy = [0.1571, 0.0468, 0.0145, 0.0046, 0.0014, 0.0001, 0.00001]
    return np.sqrt(d * 0.25) * np.interp(n_muestras, xxx, yyy)


class RBIGManual:
    """RBIG minimo: gaussianizacion marginal + rotacion PCA, iterado."""

    def __init__(self, n_capas=30, n_bins=128, ext=0.3, eps=1e-6, paciencia=3, semilla=0):
        self.n_capas, self.n_bins = n_capas, n_bins
        self.ext, self.eps = ext, eps            # soporte extendido y recorte pre-probit
        self.paciencia, self.semilla = paciencia, semilla

    def _ajusta_marginal(self, col):
        lo, hi = col.min(), col.max()
        margen = self.ext * (hi - lo + 1e-12)
        cuentas, bordes = np.histogram(col, bins=self.n_bins,
                                       range=(lo - margen, hi + margen))
        pdf = cuentas.astype(float) + 1e-3       # regularizacion: CDF estrictamente creciente
        cdf = np.concatenate([[0.0], np.cumsum(pdf)])
        return bordes, cdf / cdf[-1]

    def _marg_fwd(self, col, bordes, cdf):
        return ndtri(np.clip(np.interp(col, bordes, cdf), self.eps, 1 - self.eps))

    def _marg_inv(self, col, bordes, cdf):
        return np.interp(np.clip(ndtr(col), cdf[0], cdf[-1]), cdf, bordes)

    def fit(self, X):
        Z = np.asarray(X, dtype=float).copy()
        n, d = Z.shape
        umbral = _tolerancia(n, d)
        self.capas_, info = [], []
        for _ in range(self.n_capas):
            # (a)+(b) gaussianizacion marginal dimension a dimension
            params, G = [], np.empty_like(Z)
            for j in range(d):
                bordes, cdf = self._ajusta_marginal(Z[:, j])
                params.append((bordes, cdf))
                G[:, j] = self._marg_fwd(Z[:, j], bordes, cdf)
            # (c) rotacion ortonormal
            pca = PCA(random_state=self.semilla).fit(G)
            Y = pca.transform(G)
            # informacion eliminada por esta capa
            dT = d * H_NORMAL - sum(_entropia_hist(Y[:, j], self.n_bins) for j in range(d))
            info.append(float(dT) if dT > umbral else 0.0)
            self.capas_.append((params, pca))
            Z = Y
            # parada: varias capas seguidas sin extraer informacion
            if len(info) >= self.paciencia and max(info[-self.paciencia:]) == 0.0:
                self.capas_ = self.capas_[:-self.paciencia]   # descarta capas inutiles
                info = info[:-self.paciencia]
                break
        self.info_loss_ = np.array(info)
        return self

    def transform(self, X):
        Z = np.asarray(X, dtype=float).copy()
        for params, pca in self.capas_:
            G = np.empty_like(Z)
            for j, (bordes, cdf) in enumerate(params):
                G[:, j] = self._marg_fwd(Z[:, j], bordes, cdf)
            Z = pca.transform(G)
        return Z

    def inverse_transform(self, Z):
        X = np.asarray(Z, dtype=float).copy()
        for params, pca in reversed(self.capas_):
            G = pca.inverse_transform(X)
            X = np.empty_like(G)
            for j, (bordes, cdf) in enumerate(params):
                X[:, j] = self._marg_inv(G[:, j], bordes, cdf)
        return X

    def sample(self, n, rng=None):
        rng = np.random.default_rng(self.semilla if rng is None else rng)
        d = self.capas_[0][1].n_features_in_
        return self.inverse_transform(rng.standard_normal((n, d)))
```

### 9.4 Pipeline para el taller: PCA previa + un RBIG por régimen

Combina las dos decisiones de la sección 8. Verificado a dimensión completa: con PCA previa a 64
componentes (98,5 % de varianza retenida) el ajuste completo baja de decenas de minutos a
segundos.

```python
from sklearn.decomposition import PCA

def ajusta_rbig_por_regimen(bloque, regimen, n_comp=64, semilla=0):
    """Un RBIG por clase de regimen, cada uno en su propio subespacio PCA.

    bloque : (n_ventanas, 1201) array aplanado [X ; y_vol]
    regimen: (n_ventanas,) etiqueta entera del regimen de mercado
    """
    modelos = {}
    for k in np.unique(regimen):
        Xk = bloque[regimen == k]
        # la clase rara manda: nunca mas componentes que muestras/10
        k_comp = int(min(n_comp, Xk.shape[0] // 10, Xk.shape[1]))
        pca = PCA(n_components=k_comp, random_state=semilla).fit(Xk)
        rbig = RBIGManual(n_capas=30, semilla=semilla).fit(pca.transform(Xk))
        modelos[k] = (pca, rbig)
        print("regimen %s: n=%d, comp=%d, var=%.3f, capas=%d, info=%.3f nats"
              % (k, Xk.shape[0], k_comp, pca.explained_variance_ratio_.sum(),
                 len(rbig.capas_), rbig.info_loss_.sum()))
    return modelos


def genera_sintetico(modelos, n_por_regimen, semilla=0):
    """Genera muestras con la mezcla de regimenes que se le pida."""
    bloques, etiquetas = [], []
    for k, (pca, rbig) in modelos.items():
        n = n_por_regimen[k]
        if n:
            # muestrear en el espacio reducido y deshacer la PCA
            bloques.append(pca.inverse_transform(rbig.sample(n, rng=semilla + int(k))))
            etiquetas.append(np.full(n, k))
    return np.vstack(bloques), np.concatenate(etiquetas)
```

Para el entregable, guardar por régimen la curva `np.cumsum(rbig.info_loss_)` y el valor del
plateau: son la evidencia de convergencia que pide el enunciado.

**Cuidado al comparar el plateau entre regímenes.** El estimador de entropía por histograma
está sesgado al alza cuando hay pocas muestras, de modo que un régimen con menos datos reporta
*más* información acumulada aunque el proceso generador sea idéntico. Medido en el entorno del
proyecto con el mismo proceso generador variando solo $n$:

| $n$ | 4.000 | 2.000 | 1.000 | 500 | 200 | 100 |
|---|---:|---:|---:|---:|---:|---:|
| info acumulada (nats) | 10,1 | 15,0 | 22,3 | 40,4 | 117,1 | 293,9 |

El plateau solo es comparable **a igual $n$**. Un valor anómalamente alto en el régimen de
crisis no indica más estructura: es la firma del sobreajuste descrito en 8.4. Para comparar
regímenes hay que submuestrear todos al $n$ de la clase minoritaria, o interpretar únicamente
la *forma* de la curva (dónde satura) y no su altura.

## 10. Referencias

**Material de clase** (la numeración corresponde al orden de página del PDF):

- `docs/material_clase/slides/Normalizing Flows_2026.pdf` — Valero Laparra. Maldición de la
  dimensión (diap. 3); modelo de flujo (4); cambio de variable (5); determinante del jacobiano
  (7-8); log-verosimilitud (9); composición (10-12); requisitos de una capa (13); taxonomía por
  estructura del jacobiano (15); **Gaussianización / RBIG (16)**; flow matching (17-19); tabla
  comparativa (20); librerías (21-22).
- `docs/material_clase/slides/2026_Taller_Generativos.pdf` — Valero Laparra. Planteamiento
  (2-7); trazas de entrenamiento de la GAN (14-24); scatter reales vs generados (25-26);
  **resultado de referencia: solo reales 4,81 K / + GANs 3,46 K / + RBIG 3,62 K (27-28)**;
  comparación GANs vs RBIG variando el volumen de datos reales (30-32).
- `docs/material_clase/slides/2026_Intro_Generative_Models.pdf` — RBIG entre los generativos
  previos al deep learning (diap. 9).
- `docs/material_clase/notebooks/2_RBIG_demo_2D.ipynb` — instalación (celda 1); ajuste (5);
  invertibilidad (9); **curva `np.cumsum(rbig_model.info_loss)` (12)**; generación (14-16);
  densidad (18-21). Y `2_RBIG_demo_2D_clase.ipynb` — densidad sobre datos no vistos y sobre
  ruido uniforme (celdas 22-27).
- `docs/enunciado/Taller_B5_T1.pdf` — requisito de curvas de convergencia y de comparación entre
  proporciones de datos reales y sintéticos (pág. 2).

**Bibliografía citada en las diapositivas:** Chen & Gopinath (2000), *Gaussianization*, NeurIPS
(origen del método); **Laparra, Camps-Valls & Malo (2011), *Iterative Gaussianization: from ICA
to Random Rotations*, IEEE TNN, https://www.uv.es/lapeva/papers/Laparra11.pdf** (referencia de
RBIG, con la prueba de convergencia para rotaciones arbitrarias); Meng et al. (2020),
*Gaussianization Flows*, AISTATS (versión diferenciable); Inouye & Ravikumar (2018), *Deep
Density Destructors*, ICML; Kobyzev, Prince & Brubaker (2019) y Papamakarios et al. (2019),
revisiones de normalizing flows, arXiv; Lipman et al. (2022), *Flow Matching for Generative
Modeling*, arXiv:2210.02747.

**Código:** https://github.com/IPL-UV/rbig (implementación de referencia, `py_rbig` v0.0.1,
último commit 18/03/2023); https://github.com/IPL-UV/gaussflow (Gaussianization Flows del mismo
grupo); https://github.com/bayesiains/nflows; https://github.com/janosh/awesome-normalizing-flows.
