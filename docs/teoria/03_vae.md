# Autoencoders variacionales

## 1. Del autoencoder determinista al variacional

Un autoencoder clásico aprende dos funciones deterministas, $f_\phi:\mathbb{R}^D\to\mathbb{R}^J$ (encoder) y $g_\theta:\mathbb{R}^J\to\mathbb{R}^D$ (decoder), minimizando el error de reconstrucción

$$\mathcal{L}_{AE}(\theta,\phi)=\frac{1}{N}\sum_{i=1}^{N}\lVert x_i - g_\theta(f_\phi(x_i))\rVert^2 .$$

Con $J\ll D$ esto produce una compresión útil, pero **no es un modelo generativo**. El código $z=f_\phi(x)$ solo está definido sobre los puntos que el encoder ha visto; el resto del espacio latente son "agujeros" donde el decoder nunca ha sido entrenado. Si se muestrea $z$ al azar y se decodifica, la salida no se parece a nada del dominio. Falta lo esencial: una distribución conocida sobre $z$ de la que se pueda muestrear.

El VAE (Kingma y Welling, 2013 — citado en `docs/material_clase/slides/2026_Intro_Generative_Models.pdf`, diapositiva 50) resuelve esto planteando un **modelo de variable latente** explícito:

$$p_\theta(x)=\int p_\theta(x\mid z)\,p(z)\,dz,\qquad p(z)=\mathcal{N}(0,I_J).$$

El prior es fijo y muestreable. Entrenar por máxima verosimilitud requeriría esa integral sobre $\mathbb{R}^J$, que es intratable, y el posterior $p_\theta(z\mid x)=p_\theta(x\mid z)p(z)/p_\theta(x)$ tampoco tiene forma cerrada. La solución es **inferencia variacional amortizada**: se introduce una familia aproximada $q_\phi(z\mid x)=\mathcal{N}(\mu_\phi(x),\operatorname{diag}\sigma^2_\phi(x))$ cuyos parámetros los produce una red neuronal (el encoder). "Amortizada" significa que no se optimiza un $q$ por dato, sino una única red que mapea cualquier $x$ a los parámetros de su posterior aproximado.

El cambio conceptual respecto al AE es que el encoder deja de emitir un punto y pasa a emitir una **distribución**. Eso obliga al decoder a ser robusto en un entorno de cada código, y el término KL empuja todos esos entornos hacia $\mathcal{N}(0,I)$, tapando los agujeros. El resultado es un espacio latente del que sí se puede muestrear.

Encaja con la tercera definición de modelo generativo de las diapositivas (`docs/material_clase/slides/2026_Intro_Generative_Models.pdf`, diapositiva 4): un modelo que se usa para generar datos. Y con la primera, $p(X,Y)$, en cuanto se condiciona a la etiqueta (sección 7).

## 2. Formulación: ELBO, término de reconstrucción y KL

Partiendo de la log-verosimilitud marginal e introduciendo $q_\phi(z\mid x)$:

$$\log p_\theta(x)=\underbrace{\mathbb{E}_{q_\phi(z\mid x)}\!\left[\log p_\theta(x\mid z)\right]-D_{KL}\!\left(q_\phi(z\mid x)\,\Vert\,p(z)\right)}_{\mathcal{L}(\theta,\phi;x)\;=\;\text{ELBO}}\;+\;D_{KL}\!\left(q_\phi(z\mid x)\,\Vert\,p_\theta(z\mid x)\right).$$

El último término es no negativo y no computable, así que $\mathcal{L}\le\log p_\theta(x)$: el ELBO es una **cota inferior**. Maximizarlo sube la verosimilitud y a la vez aprieta $q_\phi$ contra el posterior real. En la práctica se minimiza $-\mathcal{L}$, que descompone en dos piezas con interpretaciones opuestas:

- **Reconstrucción** $-\mathbb{E}_{q}[\log p_\theta(x\mid z)]$: exige que el código conserve información suficiente para reconstruir $x$.
- **Regularización** $D_{KL}(q_\phi(z\mid x)\Vert p(z))$: exige que el posterior por dato no se aleje del prior. Es un coste en nats por la información que el código transporta.

La tensión entre ambos es el VAE. El primero quiere códigos informativos y separados; el segundo quiere que todos los códigos sean $\mathcal{N}(0,I)$, es decir, que no digan nada.

Con $q_\phi$ gaussiana diagonal y $p(z)=\mathcal{N}(0,I)$ el KL tiene forma cerrada:

$$D_{KL}=-\frac{1}{2}\sum_{j=1}^{J}\left(1+\log\sigma_j^2-\mu_j^2-\sigma_j^2\right).$$

Es exactamente lo implementado en `docs/material_clase/notebooks/Keras_AE_variational.ipynb`:

```python
# KL analitico frente a N(0, I); z_log_var parametriza log(sigma^2)
kl_loss = -0.5 * ops.mean(1 + z_log_var - ops.square(z_mean) - ops.exp(z_log_var), axis=-1)
kl_loss = ops.mean(kl_loss)
```

**Detalle crítico de escalado.** El notebook promedia el KL sobre las $J=2$ dimensiones latentes (`ops.mean(..., axis=-1)`) pero **suma** la reconstrucción sobre los 784 píxeles, disfrazando la suma como `mse(...) * image_size * image_size`:

```python
reconstruction_loss = ops.mean(
    mse(flat_x, flat_reconstruction) * self.image_size * self.image_size
)
```

Esa multiplicación por 784 no es cosmética: fija el peso relativo entre los dos términos. La formulación correcta suma sobre las dimensiones en **ambos** términos (el ELBO es una suma sobre coordenadas, no una media). Cualquier desviación introduce un $\beta$ implícito. Al portar el código a otro problema con otra dimensionalidad hay que rehacer este cálculo, no copiarlo; en la sección 8 se retoma con los números de nuestro panel.

## 3. Truco de reparametrización

El ELBO contiene una esperanza sobre $q_\phi$, cuyos parámetros son justamente los que hay que derivar. Estimar $\nabla_\phi\mathbb{E}_{q_\phi}[\cdot]$ muestreando $z\sim q_\phi(z\mid x)$ directamente rompe la cadena: el muestreo no es diferenciable respecto a $\mu_\phi,\sigma_\phi$.

La reparametrización mueve la aleatoriedad fuera del grafo de parámetros:

$$z=\mu_\phi(x)+\sigma_\phi(x)\odot\varepsilon,\qquad \varepsilon\sim\mathcal{N}(0,I_J).$$

Ahora $z$ es una función determinista y diferenciable de $(\mu_\phi,\sigma_\phi,\varepsilon)$, y $\varepsilon$ es ruido externo sin parámetros. El gradiente atraviesa el muestreo. Implementación del notebook:

```python
def sampling(args):
    z_mean, z_log_var = args
    batch = ops.shape(z_mean)[0]
    dim = ops.cast(ops.shape(z_mean)[1], 'int32')
    epsilon = tf.random.normal(shape=(batch, dim))
    # sigma = exp(0.5 * log sigma^2); el ruido entra de forma aditiva y diferenciable
    return z_mean + ops.exp(0.5 * z_log_var) * epsilon
```

Dos observaciones prácticas:

- La red predice $\log\sigma^2$ y no $\sigma$. Garantiza positividad sin activaciones raras y evita desbordamientos: $\sigma=\exp(0.5\log\sigma^2)$ es estable en todo el rango que puede emitir una capa lineal.
- Un solo $\varepsilon$ por dato y por paso es suficiente. El estimador es insesgado y su varianza es mucho menor que la de las alternativas basadas en el score.

> Ampliación (no cubierto en clase): la alternativa sin reparametrizar es el estimador REINFORCE / score function, $\nabla_\phi\mathbb{E}_q[f]=\mathbb{E}_q[f\,\nabla_\phi\log q_\phi]$. Es válido pero su varianza es de uno a dos órdenes de magnitud mayor, lo que en la práctica hace inviable el entrenamiento sin control de varianza. Es la razón por la que el VAE es entrenable y modelos con latentes discretos no lo son directamente.

En **generación** no se reparametriza: se muestrea $z\sim\mathcal{N}(0,I)$ y se decodifica. En **reconstrucción determinista** (por ejemplo para proyectar el conjunto de test y visualizar el latente) se usa $z=\mu_\phi(x)$, que es lo que hace `plot_results` en el notebook al leer la primera salida del encoder.

## 4. Arquitectura: encoder, espacio latente, decoder

El notebook define encoder y decoder por separado y los une en un `keras.Model` con `train_step` propio.

**Encoder** (`docs/material_clase/notebooks/Keras_AE_variational.ipynb`): dos bloques `Conv2D` con `strides=2` (16→32→64 filtros, kernel 3, ReLU), `Flatten`, `Dense(16)` y dos cabezas lineales en paralelo:

```python
z_mean    = Dense(latent_dim, name='z_mean')(x)
z_log_var = Dense(latent_dim, name='z_log_var')(x)
z = Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])
encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')
```

Las dos cabezas son **lineales sin activación**: $\mu$ vive en $\mathbb{R}$ y $\log\sigma^2$ también. Poner ahí una ReLU o una sigmoid es un error frecuente que rompe el modelo (imposibilita $\mu<0$ o acota artificialmente la varianza).

El encoder devuelve tres tensores. El VAE toma el tercero para alimentar al decoder, y los dos primeros para el KL:

```python
salida_enc = encoder(inputs)[2]   # z muestreado
outputs = decoder(salida_enc)
```

**Espacio latente.** En el notebook `latent_dim = 2`, elegido para poder dibujar el scatter de `z_mean` coloreado por clase y la rejilla de dígitos decodificados sobre $[-4,4]^2$. Es una decisión didáctica, no un valor por defecto. La dimensión latente es el ancho de banda del cuello de botella: demasiado pequeña fuerza reconstrucciones pobres; demasiado grande deja dimensiones inactivas y facilita el colapso parcial (sección 6).

**Decoder**: `Dense` → `Reshape` → dos `Conv2DTranspose` con `strides=2` → capa de salida

```python
outputs = Conv2DTranspose(filters=1, kernel_size=kernel_size,
                          activation='sigmoid', padding='same',
                          name='decoder_output')(x)
```

La `sigmoid` final está ligada a que MNIST se normaliza a $[0,1]$ (`x_train.astype('float32') / 255`). **La activación de salida no es libre: es una consecuencia de la verosimilitud elegida.** Este punto es el que hay que replantear al pasar a datos financieros y se desarrolla en la sección 8.

## 5. Diagnóstico de convergencia

El enunciado del taller (`docs/enunciado/Taller_B5_T1.pdf`, apartado 5) exige, para cada entrenamiento, "las curvas de loss donde se vea que el modelo ha convergido". En un VAE **una sola curva no es suficiente y además engaña**. La loss total puede bajar suave y aplanarse mientras el modelo está roto, porque el descenso lo está aportando el término de reconstrucción mientras el KL se ha ido a cero y el latente ha dejado de usarse. Hay que graficar las tres curvas por separado, en train y validación.

El notebook ya proporciona la infraestructura: mantiene tres `keras.metrics.Mean` independientes y los devuelve por separado en `train_step` y `test_step`.

```python
self.total_loss_tracker          = keras.metrics.Mean(name="total_loss")
self.reconstruction_loss_tracker = keras.metrics.Mean(name="reconstruction_loss")
self.kl_loss_tracker             = keras.metrics.Mean(name="kl_loss")
```

### 5.1 Las tres curvas y qué debe verse en cada una

| Curva | Patrón sano | Señal de alarma |
|---|---|---|
| Reconstrucción | descenso monótono rápido y meseta clara; `val` pegada a `train` | meseta muy alta e inmediata (igual a la varianza marginal del dato) → el decoder ignora $z$; `val` subiendo → sobreajuste del decoder |
| KL | sube desde ~0 en las primeras épocas, alcanza un máximo y **se estabiliza en una meseta estrictamente positiva** | cae a $\approx 0$ y se queda → posterior collapse; crece sin freno → latente sin regularizar, muestras del prior inservibles |
| Total | descenso y meseta, con `val` sin divergir | meseta perfecta que **no informa**: hay que leerla siempre junto a las otras dos |

La forma esperada del KL es contraintuitiva la primera vez: **debe subir**. Al principio el encoder emite $\mu\approx0,\sigma\approx1$ (KL $\approx 0$) porque no ha aprendido nada; el KL crece a medida que el modelo empieza a codificar información en el latente y se estabiliza cuando ha encontrado el punto de equilibrio con la reconstrucción. Un KL que arranca en cero y **se queda** en cero durante 20-30 épocas es colapso, no convergencia.

Criterio operativo de convergencia para el entregable, verificando las tres condiciones:

1. Reconstrucción de validación sin mejora relativa $>1\%$ durante `patience` épocas (típicamente 20).
2. KL en meseta, con pendiente relativa por época $<1\%$ y valor **estrictamente positivo**.
3. Brecha `val` − `train` estable, no creciente.

### 5.2 Métricas complementarias que hay que reportar

Además de las tres curvas conviene registrar dos diagnósticos por época que detectan problemas que las curvas agregadas no muestran:

- **KL por dimensión latente**, $D_{KL,j}=\frac{1}{N}\sum_i -\frac12(1+\log\sigma_{ij}^2-\mu_{ij}^2-\sigma_{ij}^2)$. Un heatmap $J\times$ épocas revela si el modelo está usando 3 dimensiones de 32.
- **Unidades activas**: $\#\{j: D_{KL,j}>0{,}01\text{ nats}\}$. Una sola cifra por época, muy informativa. Si va bajando durante el entrenamiento, hay colapso progresivo.

> Ampliación (no cubierto en clase): la métrica de unidades activas procede de la literatura sobre posterior collapse (Burda et al., 2016). El umbral 0,01 nats es convencional; lo relevante es la tendencia, no el valor absoluto.

### 5.3 Código de las figuras de convergencia

```python
import json
import matplotlib.pyplot as plt

def plot_convergencia_vae(hist, ruta_png, titulo="cVAE"):
    """Tres paneles: total, reconstruccion y KL. Cada uno con train y validacion.

    hist: dict con listas por epoca. Claves esperadas:
          total, recon, kl, val_total, val_recon, val_kl, unidades_activas
    """
    fig, ax = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)

    ax[0].plot(hist["total"], label="train")
    ax[0].plot(hist["val_total"], label="val")
    ax[0].set_title("Loss total (-ELBO)")

    ax[1].plot(hist["recon"], label="train")
    ax[1].plot(hist["val_recon"], label="val")
    ax[1].set_title("Reconstruccion")

    ax[2].plot(hist["kl"], label="train")
    ax[2].plot(hist["val_kl"], label="val")
    # la meseta del KL debe quedar por encima de cero: se marca la referencia
    ax[2].axhline(0.0, color="grey", lw=0.8, ls="--")
    ax[2].set_title("KL(q(z|x,c) || N(0,I))")

    for a in ax:
        a.set_xlabel("epoca")
        a.grid(alpha=0.3)
        a.legend()
    fig.suptitle(titulo)
    fig.savefig(ruta_png, dpi=140)
    plt.close(fig)


# el historial se serializa para poder regenerar las figuras sin reentrenar
with open("results/historiales/cvae_history.json", "w", encoding="utf-8") as f:
    json.dump(hist, f)
```

Escala logarítmica en el eje $y$ del panel de reconstrucción si el descenso inicial es de varios órdenes de magnitud; en caso contrario la meseta se ve plana desde la época 5 y no se distingue si sigue mejorando.

### 5.4 Qué hacer cuando el diagnóstico sale mal

- **KL en cero desde el principio** → el término de reconstrucción está infra-pesado respecto al KL. Reducir $\beta$ (equivalente a reducir la varianza asumida del decoder, sección 8) o aplicar KL annealing.
- **KL que sube y colapsa hacia la época 10-20** → el decoder ha encontrado el óptimo local de ignorar $z$. Free bits es el remedio más directo porque impide estructuralmente que ninguna dimensión baje de un suelo.
- **KL creciendo sin meseta y reconstrucción excelente** → el modelo está degenerando en un autoencoder determinista de latente muy informativo. Reconstruye bien pero el posterior agregado no se parece al prior, y muestrear $z\sim\mathcal{N}(0,I)$ dará basura. Subir $\beta$.

> Ampliación (no cubierto en clase): las tres herramientas de control del término KL.
>
> **β-VAE** (Higgins et al., 2017): se pondera el KL con un escalar fijo, $\mathcal{L}=\text{recon}+\beta\,D_{KL}$. Con $\beta>1$ se fuerza más regularización y latentes más desentrelazados a costa de reconstrucción; con $\beta<1$ lo contrario. Es el mando principal.
>
> **KL annealing** (Bowman et al., 2016): $\beta$ no es fijo sino que crece de 0 a su valor objetivo durante las primeras $T$ épocas. Permite al decoder aprender a usar $z$ antes de que el KL empiece a penalizarlo, evitando el colapso temprano. La variante cíclica repite la rampa varias veces.
>
> **Free bits** (Kingma et al., 2016): se sustituye el KL por $\sum_j\max(\lambda, D_{KL,j})$ con $\lambda$ del orden de 0,05–0,5 nats. Cada dimensión latente tiene un presupuesto gratuito de información; mientras esté por debajo, el gradiente del KL sobre esa dimensión es nulo y no puede ser aplastada. Es el remedio más robusto contra el colapso y el que menos toca el resto del modelo.

## 6. Patologías (posterior collapse, blurriness, desbalance β) y cómo detectarlas

### 6.1 Posterior collapse

$q_\phi(z\mid x)\to p(z)$ para todo $x$: el encoder emite $\mu\approx0$, $\sigma\approx1$ independientemente de la entrada, y el decoder aprende la media marginal de los datos. El KL vale ~0 y la reconstrucción se estanca en la varianza total del dataset.

Causas: decoder demasiado expresivo respecto al problema, $\beta$ efectivo demasiado alto, dimensión latente muy superior a la necesaria, o normalización de los datos que deja la varianza objetivo muy pequeña.

Detección, por orden de fiabilidad: (1) unidades activas $=0$; (2) KL en meseta $<10^{-2}$; (3) muestras generadas desde el prior prácticamente idénticas entre sí; (4) $\operatorname{Var}_i[\mu_\phi(x_i)_j]\approx 0$ para toda $j$.

Remedios: free bits, KL annealing, reducir capacidad del decoder, bajar $\beta$.

### 6.2 Suavizado excesivo (blurriness)

Con verosimilitud gaussiana factorizada, el decoder óptimo emite $\mathbb{E}[x\mid z]$. Cuando un mismo $z$ es compatible con varios $x$ plausibles, la salida que minimiza el MSE es su **promedio**, no ninguno de ellos. En imágenes esto se ve como desenfoque; en series financieras se manifiesta como retornos atenuados, saltos aplanados y colas más ligeras que las reales.

No es un bug, es la consecuencia de la función de pérdida. Detección: comparar momentos de orden alto entre real y sintético (curtosis por canal, percentiles 1 y 99). Un cVAE bien entrenado y no diagnosticado suele infra-estimar la curtosis en un factor considerable.

Mitigaciones parciales: aumentar $J$, reducir $\beta$, decoder con varianza aprendida por dimensión. Ninguna elimina el efecto; conviene medirlo y reportarlo (sección 8).

### 6.3 Desbalance $\beta$ por escalado inconsistente

El error más común al portar el notebook. Si la reconstrucción se **suma** sobre $D$ dimensiones y el KL se **promedia** sobre $J$, el ratio efectivo entre ambos es $D\cdot J$ veces el nominal. Con $D = 1.201$ y $J=32$ el KL queda multiplicado por un factor absurdo respecto a la formulación correcta.

Regla: sumar sobre dimensiones en ambos términos y promediar solo sobre el batch. Cualquier ponderación adicional debe ser explícita, con nombre `beta` y registrada en la configuración del experimento.

### 6.4 Huecos en el prior

Aun sin colapso, el posterior agregado $q_\phi(z)=\frac1N\sum_i q_\phi(z\mid x_i)$ puede no cubrir $\mathcal{N}(0,I)$. Reconstruye bien, pero al muestrear del prior se cae en regiones no visitadas y la salida degenera.

Detección: reconstrucción de validación buena **pero** muestras del prior de mala calidad. Comparar en un scatter (o proyección PCA de $z$ si $J>2$) la nube de $\mu_\phi(x)$ contra $\mathcal{N}(0,I)$; es la lectura útil del `plot_results` del notebook cuando $J=2$.

### 6.5 Tabla resumen

| Síntoma observado | Causa probable | Acción |
|---|---|---|
| KL $\approx 0$ estable, recon estancada | posterior collapse | free bits, annealing, bajar $\beta$ |
| KL creciente sin meseta, recon muy baja | latente sin regularizar | subir $\beta$ |
| Muestras suaves, curtosis baja | verosimilitud gaussiana | subir $J$, bajar $\beta$, decoder heterocedástico |
| Recon train baja / val sube | sobreajuste del decoder | early stopping, dropout, reducir capacidad |
| Recon buena, muestras del prior malas | huecos en el prior | subir $\beta$, más datos, revisar $J$ |
| KL enorme desde época 1 | escalado inconsistente recon/KL | revisar sum vs mean (§6.3) |

## 7. VAE condicional (cVAE): cómo se inyecta la etiqueta

> Ampliación (no cubierto en clase): la formulación del CVAE es de Sohn, Lee y Yan (2015). En clase se cubrió el VAE no condicional; el condicionamiento se vio en el contexto de GANs (`docs/material_clase/slides/2026_Intro_Generative_Models.pdf`, diapositiva 47, y `docs/material_clase/notebooks/GAN_1_Really_Simple_GAN_MNIST_etiquetas_y_balanceo.ipynb`).

Se añade una variable de condición $c$ a ambas redes. El ELBO condicional es

$$\mathcal{L}(x,c)=\mathbb{E}_{q_\phi(z\mid x,c)}\!\left[\log p_\theta(x\mid z,c)\right]-D_{KL}\!\left(q_\phi(z\mid x,c)\,\Vert\,p(z\mid c)\right),$$

con la elección estándar $p(z\mid c)=\mathcal{N}(0,I)$, independiente de $c$. Mantener el prior fijo simplifica el muestreo (se muestrea $z$ una vez y se decodifica con la $c$ que se quiera) y funciona bien cuando el condicionamiento es una etiqueta de pocas clases.

**Mecanismo de inyección.** La forma más simple y suficiente aquí es concatenación de un one-hot $c\in\{0,1\}^{K}$:

- Encoder: entrada $[\,x \,;\, c\,]$, dimensión $D+K$.
- Decoder: entrada $[\,z \,;\, c\,]$, dimensión $J+K$.

Hay que inyectarla en **los dos**. Si solo se condiciona el decoder, el encoder tiene que codificar la clase dentro de $z$ y se gasta capacidad latente en información que ya está disponible gratis; si solo se condiciona el encoder, el decoder no puede usar la etiqueta al generar y el condicionamiento no sirve para nada.

> Ampliación (no cubierto en clase): alternativas a la concatenación cuando $K$ es grande o la condición es continua: capa de *embedding* seguida de concatenación, o modulación FiLM ($h\leftarrow\gamma(c)\odot h+\beta(c)$) aplicada a las capas ocultas. Para $K=3$ el one-hot concatenado es la opción correcta por simplicidad.

**Contraste con el enfoque de clase.** El notebook `GAN_1_Really_Simple_GAN_MNIST_etiquetas_y_balanceo.ipynb` no condiciona: construye un **bloque conjunto** pegando la etiqueta reescalada como una columna más del vector de datos,

```python
Datos[:, 0:X_train.shape[1]*X_train.shape[2]] = X_train.reshape(...)
Datos[:, -1] = Y_train          # la etiqueta es un canal mas del dato
```

y al generar la lee de vuelta redondeando (`np.round(syntetic_labels*4.5+4.5)`). Es el modelo generativo conjunto $p(X,Y)$ de la diapositiva 2. Funciona, pero **no da control**: no se puede pedir "500 muestras de la clase minoritaria", solo generar mucho y filtrar, y la etiqueta sale ruidosa porque es una salida continua redondeada. El condicionamiento explícito del cVAE sí lo da, y esa es la razón de usarlo aquí.

## 8. Aplicación a nuestro problema

### 8.1 El dato

Cada muestra es una ventana de 60 días × 20 canales del panel híbrido de 15 activos (S&P 500, 9 SPDR sectoriales, VIX, tesoro a 20 y a 10 años, crédito grado de inversión e índice dólar): once canales de retornos (índice, nueve sectores y dólar) y nueve derivados (nivel y variación del VIX, volatilidad realizada, drawdown, momento, spread de crédito, pendiente de curva, correlación acción-bono y dispersión sectorial), aplanados a $D_X = 1.200$. El bloque conjunto es $[\,X\,;\,y_{vol}\,]$ con dimensión total $1.201$, y $y_{reg}$ va aparte, como condición. La etiqueta $y_{reg}$ es el régimen de mercado a 21 días con 3 clases, y la clase "crisis" pesa ~16 % en train (10,5 % en test).

**Decisión de diseño**: el cVAE se condiciona a $y_{reg}$ (one-hot, $K=3$) y el decoder reconstruye $[\,X\,;\,y_{vol}\,]$, dimensión $D = 1.201$. No tiene sentido reconstruir $y_{reg}$ cuando ya se está condicionando a ella. Para mantener la interfaz común con los demás generadores del taller, al escribir el dataset sintético se reinserta como $y_{reg}$ el one-hot de condicionamiento (que es exacto, no ruidoso). Esto es una ventaja neta frente al enfoque de bloque conjunto del notebook de GAN.

**Preprocesado**: z-score por canal con media y desviación calculadas **solo sobre el tramo de entrenamiento**, para no filtrar información del futuro. Partición temporal, sin barajar, con embargo contado en sesiones de mercado entre train y validación porque las ventanas se solapan y el horizonte de la etiqueta es de 21 días: la huella de una ventana es $60+21=81$ sesiones, el mínimo que neutraliza el solape es 80 y el valor adoptado es **85 sesiones**. Sin esa purga, la validación está contaminada y las curvas de la sección 5 mienten. Si $y_{vol}$ es volatilidad realizada (positiva y asimétrica), transformar a $\log$ antes de estandarizar.

### 8.2 Verosimilitud de reconstrucción: gaussiana/MSE, no Bernoulli/BCE

Este es el punto donde copiar el notebook de clase sin pensar rompe el modelo.

El notebook trabaja con MNIST normalizado a $[0,1]$ y remata el decoder con `activation='sigmoid'`. Con ese soporte acotado, la elección natural (y la que usan la mayoría de tutoriales de VAE, incluido el enlace de las diapositivas a `keras.io/examples/generative/vae`) es la **verosimilitud Bernoulli** por píxel:

$$-\log p_\theta(x\mid z)=-\sum_{d=1}^{D}\left[x_d\log\hat{x}_d+(1-x_d)\log(1-\hat{x}_d)\right],$$

que es la binary cross-entropy. **Requiere $x_d\in[0,1]$.** Nuestros datos son z-scores: soporte $\mathbb{R}$, media 0, valores negativos en aproximadamente la mitad de las entradas y colas que llegan a $\pm 5$ o más en los episodios de crisis. Con $x_d<0$ el término $x_d\log\hat x_d$ deja de tener sentido probabilístico y el gradiente empuja en direcciones arbitrarias; con $x_d>1$ ocurre lo simétrico. El modelo no lanza una excepción — devuelve una pérdida numéricamente finita y absurda, que es peor.

La tentación de "arreglarlo" con un min-max a $[0,1]$ es exactamente el error a evitar en este problema: comprime todo el rango útil de los retornos en la zona central, hace que el mínimo y el máximo del histórico definan la escala (dos observaciones, ambas de crisis), y aplasta precisamente las colas que el taller quiere que el generador reproduzca. Además, una sigmoid final satura: no puede emitir el $-4{,}5\sigma$ de marzo de 2020.

La elección correcta es **verosimilitud gaussiana isótropa con varianza fija** $\sigma_x^2$:

$$-\log p_\theta(x\mid z)=\frac{1}{2\sigma_x^2}\sum_{d=1}^{D}\left(x_d-\hat{x}_d\right)^2+\frac{D}{2}\log\left(2\pi\sigma_x^2\right),$$

cuyo segundo término es constante respecto a $\theta,\phi$ y se descarta. Queda el **error cuadrático sumado sobre las $D$ dimensiones**, escalado por $1/(2\sigma_x^2)$. Consecuencias directas:

1. **Salida del decoder lineal**, sin `sigmoid` ni `tanh`. El soporte de la gaussiana es $\mathbb{R}$.
2. La varianza asumida del decoder **es** el hiperparámetro $\beta$. Minimizar $\frac{1}{2\sigma_x^2}\mathrm{SSE}+D_{KL}$ es equivalente a minimizar $\mathrm{SSE}+\beta\,D_{KL}$ con $\beta=2\sigma_x^2$. No son dos hiperparámetros, es uno. Conviene parametrizar directamente $\beta$ y documentar qué $\sigma_x$ implica.
3. Con datos estandarizados, $\sigma_x\approx 1$ (es decir $\beta\approx 2$) es el punto de partida razonable, pero solo si el residuo típico es del orden de la unidad. En la práctica el residuo es bastante menor y $\beta$ acaba en el rango 0,5–5. Se calibra observando las tres curvas de la sección 5, no a ciegas.

**Órdenes de magnitud.** Con $D = 1.201$ y $J=32$: la reconstrucción sumada arranca en $\sim10^3$ y baja a $\sim10^2$; el KL vive en $10^0$–$10^2$ nats. La reconstrucción domina la loss total por uno o dos órdenes de magnitud, de ahí que la curva total sea prácticamente la de reconstrucción y no sirva para diagnosticar el KL. Refuerza la exigencia de la sección 5.

> Ampliación (no cubierto en clase): dos refinamientos posibles de la verosimilitud.
>
> **Decoder heterocedástico**: emitir $\hat\mu_d$ y $\log\hat\sigma_d^2$ por dimensión y usar la NLL gaussiana completa. Permite al modelo declarar incertidumbre alta en los canales difíciles en vez de promediarlos, lo que mitiga parcialmente el suavizado. Riesgo: $\hat\sigma_d\to0$ hace la NLL divergir a $-\infty$; hay que acotar $\log\hat\sigma^2$ a, por ejemplo, $[-6, 2]$.
>
> **Verosimilitud de colas pesadas**: Laplace (equivale a L1) o Student-t. Penalizan menos los residuos grandes y por tanto conservan mejor los eventos extremos, que es lo que interesa en la clase "crisis". A cambio, el entrenamiento es algo menos estable.

### 8.3 Dimensión latente y capacidad

Con $D = 1.201$, $J$ entre 16 y 32 es el rango razonable. Las 60×20 columnas están fuertemente correlacionadas (los 9 sectores comparten el factor mercado, VIX y volatilidad realizada son casi redundantes) y la dimensión intrínseca efectiva del panel es mucho menor que 1.201. Elegir $J$ con evidencia: entrenar con $J=32$ y contar unidades activas (§5.2). Si solo 12 son activas, $J=16$ es suficiente y reduce el riesgo de colapso parcial.

### 8.4 Evaluación de las muestras

La loss no basta. Antes de alimentar el downstream hay que comprobar, por clase de régimen:

- Momentos por canal: media, desviación, asimetría y **curtosis**. La curtosis es donde el VAE va a fallar.
- Autocorrelación de $|r_t|$ hasta 20 retardos: el clustering de volatilidad. Si el sintético la aplana, las ventanas generadas no tienen la estructura temporal que el downstream necesita.
- Matriz de correlación entre los 9 sectores: comparar la real y la sintética por clase; el desacoplamiento sectorial en crisis es una de las señales que el modelo debe reproducir.
- Proyección PCA o UMAP con real y sintético superpuestos, coloreados por régimen: detecta huecos de cobertura y modos duplicados.
- **Test final**: la métrica del downstream entrenado con distintas mezclas real/sintético, que es el objetivo del taller (`docs/enunciado/Taller_B5_T1.pdf`, apartados 3 y 5).

### 8.5 cVAE frente a cGAN para este problema

| Criterio | cVAE | cGAN |
|---|---|---|
| Estabilidad de entrenamiento | alta, una sola pérdida a minimizar | baja, equilibrio de dos redes |
| Convergencia diagnosticable | sí, las tres curvas de §5 son interpretables | no de forma fiable; las pérdidas oscilan sin indicar calidad |
| Coste en CPU | minutos | decenas de minutos, más reintentos |
| Cobertura de modos | buena; el KL fuerza a cubrir el prior | riesgo de mode collapse |
| Realismo de colas | pobre: muestras suavizadas | mejor, muestras más nítidas |
| Control de la clase generada | exacto vía $c$ | exacto si se condiciona; en el enfoque conjunto del notebook, no |

Las curvas de `d_loss` y `g_loss` del notebook de GAN de clase ilustran el problema: oscilan indefinidamente y no permiten afirmar que el modelo "ha convergido", que es literalmente lo que pide el enunciado. El cVAE sí lo permite.

Lo que hay que decir con honestidad en la presentación: **el suavizado del cVAE tiene un coste específico en datos financieros**. Los retornos reales tienen curtosis muy por encima de 3 y volatilidad agrupada; un generador que produce la media condicional atenúa ambos hechos. Las muestras sintéticas de la clase "crisis" tenderán a ser versiones amortiguadas de las crisis reales: dirección correcta, magnitud insuficiente. Para un downstream que clasifica regímenes esto puede seguir siendo útil (aporta variedad en la frontera de decisión de la clase rara, que es el cuello de botella con ~16 % de ejemplos en train y 10,5 % en test), pero para cualquier uso que dependa de la cola —VaR, stress testing— sería inadecuado. La recomendación es usar el cVAE como generador base fiable, el cGAN como contraste, y **reportar la curtosis y la autocorrelación de $|r|$ de ambos junto a la métrica del downstream**, no solo esta última.

## 9. Implementación de referencia (CPU)

Entorno: Python 3.13.7, torch 2.11.0+cpu, sin CUDA. Se usa PyTorch por la comodidad de escribir la pérdida a mano; la variante Keras es una traducción directa del `train_step` de `docs/material_clase/notebooks/Keras_AE_variational.ipynb` sustituyendo la reconstrucción por la de la §8.2.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(42)
torch.set_num_threads(8)   # ajustar al numero de nucleos fisicos disponibles


class CVAE(nn.Module):
    """cVAE con MLP. Condicionamiento por concatenacion de one-hot en encoder y decoder."""

    def __init__(self, d_in=1201, n_clases=3, d_lat=32, h=512):
        super().__init__()
        self.d_lat = d_lat
        self.n_clases = n_clases

        # encoder: [x ; c] -> h -> (mu, logvar)
        self.enc = nn.Sequential(
            nn.Linear(d_in + n_clases, h), nn.SiLU(),
            nn.Linear(h, h // 2), nn.SiLU(),
        )
        self.fc_mu     = nn.Linear(h // 2, d_lat)   # lineal, sin activacion
        self.fc_logvar = nn.Linear(h // 2, d_lat)   # lineal, sin activacion

        # decoder: [z ; c] -> h -> x_hat ; salida LINEAL (verosimilitud gaussiana)
        self.dec = nn.Sequential(
            nn.Linear(d_lat + n_clases, h // 2), nn.SiLU(),
            nn.Linear(h // 2, h), nn.SiLU(),
            nn.Linear(h, d_in),
        )

    def encode(self, x, c):
        hh = self.enc(torch.cat([x, c], dim=1))
        return self.fc_mu(hh), self.fc_logvar(hh)

    def reparametrizar(self, mu, logvar):
        # truco de reparametrizacion: el ruido entra fuera del grafo de parametros
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)

    def forward(self, x, c):
        mu, logvar = self.encode(x, c)
        z = self.reparametrizar(mu, logvar)
        return self.dec(torch.cat([z, c], dim=1)), mu, logvar


def perdida_cvae(x, x_hat, mu, logvar, beta=2.0, free_bits=0.05):
    """Devuelve (total, reconstruccion, kl) promediados sobre el batch.

    Reconstruccion: SSE sumada sobre las D dimensiones (verosimilitud gaussiana
    con varianza fija; beta = 2*sigma_x^2). NUNCA BCE: los datos son z-scores.
    KL: analitico, sumado sobre las J dimensiones latentes, con suelo free-bits.
    """
    recon = F.mse_loss(x_hat, x, reduction="none").sum(dim=1).mean()

    kl_por_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())      # (B, J)
    kl_dim_media = kl_por_dim.mean(dim=0)                            # (J,)
    kl_efectivo = torch.clamp(kl_dim_media, min=free_bits).sum()     # free bits
    kl_reportado = kl_dim_media.sum()                                # el que se grafica

    return recon + beta * kl_efectivo, recon, kl_reportado


def beta_annealing(epoca, beta_max=2.0, epocas_rampa=30):
    """Rampa lineal de 0 a beta_max: evita el colapso temprano del KL."""
    return beta_max * min(1.0, (epoca + 1) / epocas_rampa)
```

Bucle de entrenamiento con el registro de las tres pérdidas y de las unidades activas:

```python
def entrenar(modelo, dl_train, dl_val, epocas=200, lr=1e-3, beta_max=2.0):
    opt = torch.optim.Adam(modelo.parameters(), lr=lr)
    hist = {k: [] for k in ["total", "recon", "kl",
                            "val_total", "val_recon", "val_kl", "unidades_activas"]}

    for ep in range(epocas):
        beta = beta_annealing(ep, beta_max)

        modelo.train()
        acum = torch.zeros(3)
        for x, c in dl_train:
            opt.zero_grad()
            x_hat, mu, logvar = modelo(x, c)
            total, recon, kl = perdida_cvae(x, x_hat, mu, logvar, beta=beta)
            total.backward()
            torch.nn.utils.clip_grad_norm_(modelo.parameters(), 5.0)
            opt.step()
            acum += torch.tensor([total.item(), recon.item(), kl.item()])
        acum /= len(dl_train)

        modelo.eval()
        acum_val = torch.zeros(3)
        kl_dims = torch.zeros(modelo.d_lat)
        with torch.no_grad():
            for x, c in dl_val:
                x_hat, mu, logvar = modelo(x, c)
                total, recon, kl = perdida_cvae(x, x_hat, mu, logvar, beta=beta)
                acum_val += torch.tensor([total.item(), recon.item(), kl.item()])
                kl_dims += (-0.5 * (1 + logvar - mu.pow(2) - logvar.exp())).mean(0)
        acum_val /= len(dl_val)
        kl_dims /= len(dl_val)

        for k, v in zip(["total", "recon", "kl"], acum.tolist()):
            hist[k].append(v)
        for k, v in zip(["val_total", "val_recon", "val_kl"], acum_val.tolist()):
            hist[k].append(v)
        # unidades activas: dimensiones latentes que transportan informacion real
        hist["unidades_activas"].append(int((kl_dims > 0.01).sum().item()))

    return hist
```

Muestreo condicional para sobre-generar la clase minoritaria:

```python
@torch.no_grad()
def generar(modelo, n, clase, n_clases=3):
    """Muestrea n ventanas sinteticas del regimen indicado. clase=2 -> crisis."""
    modelo.eval()
    z = torch.randn(n, modelo.d_lat)                       # prior N(0, I)
    c = F.one_hot(torch.full((n,), clase), n_clases).float()
    x_hat = modelo.dec(torch.cat([z, c], dim=1))
    # la etiqueta de regimen del dataset sintetico es la condicion impuesta: exacta
    return x_hat, c
```

**Tiempos esperados en CPU** (estimación para el orden de magnitud, a validar en la máquina del equipo). Con $D = 1.201$, $h=512$, $J=32$, el modelo tiene aproximadamente 1,5 M de parámetros. Con las 3.696 ventanas de entrenamiento y `batch_size=128` son 29 pasos por época; en una CPU de escritorio moderna con 8 hilos eso son del orden de 1–3 s por época, así que **200 épocas quedan en 5–10 minutos**. El barrido de $\beta$ (3-4 valores) y de $J$ (2 valores) cabe holgadamente en una sesión. Ninguna parte del pipeline requiere GPU.

Notas operativas:

- Fijar `torch.manual_seed` y guardar el `hist` en `results/historiales/` para poder regenerar las figuras sin reentrenar.
- `clip_grad_norm_` a 5.0 evita picos en las primeras épocas cuando la reconstrucción sumada es grande.
- Los pesos van a `models/generadores/`, los datasets sintéticos a `data/synthetic/`, las figuras a `results/figures/`.
- Guardar también las medias y desviaciones del z-score: sin ellas las muestras generadas no se pueden devolver a la escala original.

## 10. Referencias

**Material de clase**

- `docs/material_clase/notebooks/Keras_AE_variational.ipynb` — implementación de referencia del VAE convolucional sobre MNIST: función `sampling`, doble cabeza `z_mean`/`z_log_var`, clase `VAE` con `train_step` propio y los tres trackers de pérdida, y `plot_results` para el latente 2D.
- `docs/material_clase/slides/2026_Intro_Generative_Models.pdf` — diapositivas 2-4 (definiciones de modelo generativo), 43 (familia de modelos generativos profundos), 50-52 (VAEs, referencia a Kingma y Welling y enlaces de código).
- `docs/material_clase/notebooks/GAN_1_Really_Simple_GAN_MNIST_etiquetas_y_balanceo.ipynb` — enfoque de bloque conjunto $[X;y]$ y balanceo de clases con GAN; contraste con el condicionamiento explícito del cVAE.
- `docs/enunciado/Taller_B5_T1.pdf` — requisitos del entregable, en particular la exigencia de curvas de loss que muestren convergencia.

**Fuentes primarias citadas en las diapositivas**

- Kingma, D. P. y Welling, M. (2013). *Auto-Encoding Variational Bayes*. arXiv:1312.6114.
- Ejemplo oficial de Keras: `https://keras.io/examples/generative/vae/`.
- Demo interactiva del latente de MNIST: `https://transcranial.github.io/keras-js/#/mnist-vae`.

**Ampliaciones (no cubiertas en clase)**

- Sohn, K., Lee, H. y Yan, X. (2015). *Learning Structured Output Representation using Deep Conditional Generative Models*. NeurIPS. — formulación del CVAE.
- Higgins, I. et al. (2017). *β-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework*. ICLR.
- Bowman, S. R. et al. (2016). *Generating Sentences from a Continuous Space*. CoNLL. — KL annealing y descripción del posterior collapse.
- Kingma, D. P. et al. (2016). *Improved Variational Inference with Inverse Autoregressive Flow*. NeurIPS. — free bits.
- Burda, Y., Grosse, R. y Salakhutdinov, R. (2016). *Importance Weighted Autoencoders*. ICLR. — métrica de unidades activas.
