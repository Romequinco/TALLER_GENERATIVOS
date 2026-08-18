# GANs y GANs condicionales

## 1. Intuición: el juego adversarial

Una GAN sustituye la pregunta *"¿cuál es la densidad de mis datos?"* por *"¿sabría alguien distinguir mis muestras de las reales?"*. Se entrenan dos redes con objetivos opuestos (`docs/material_clase/slides/GANs_general.pdf`, diap. 10):

- **Generador** $G$: recibe ruido $z \sim p_z$ y produce una muestra $G(z)$ en el espacio de los datos. No ve nunca datos reales de forma directa.
- **Discriminador** $D$: recibe una muestra y devuelve $D(x) \in (0,1)$, la probabilidad de que provenga de $p_{\text{data}}$ (diap. 13).

$G$ intenta maximizar el error de $D$; $D$ intenta minimizarlo. El equilibrio buscado es aquel en el que $D$ ya no puede hacer nada mejor que responder $0.5$ a todo.

La transparencia 14 resume la aportación real del método: *"What is really new about GANs? A 'new' and 'adaptative' cost function"*. La diferencia con un autoencoder o una regresión no está en la arquitectura sino en la función de pérdida: en lugar de fijar a mano una métrica de reconstrucción (MSE, MAE, verosimilitud gaussiana), **la métrica es otra red que se reentrena continuamente**. Esto tiene dos consecuencias que condicionan todo lo demás:

1. Es un **modelo generativo implícito**: no hay densidad evaluable $p_g(x)$, sólo un muestreador $z \mapsto G(z)$. No se puede calcular la log-verosimilitud de un dato, ni usarla como criterio de parada.
2. La pérdida es **no estacionaria**. El valor numérico de la pérdida de $G$ en la iteración 1.000 y en la 10.000 se ha medido con reglas distintas ($D$ ha cambiado). Comparar esos dos números carece de sentido. Esta es la raíz del problema de diagnóstico de la sección 5.

Para nuestro caso (datos financieros tabulares/temporales) la ventaja frente a un ajuste paramétrico es que $G$ puede reproducir dependencias no lineales entre canales, colas gruesas y agrupamiento de volatilidad sin que nadie las especifique. La desventaja es que no hay ninguna garantía de que lo haga, y ningún número que lo certifique automáticamente.

## 2. Formulación matemática (minimax, divergencia JS, non-saturating loss)

El objetivo de la diapositiva 13 (`docs/material_clase/slides/GANs_general.pdf`) es:

$$\min_G \max_D V(D,G) = \mathbb{E}_{x \sim p_{\text{data}}(x)}\big[\log D(x)\big] + \mathbb{E}_{z \sim p_z(z)}\big[\log\big(1 - D(G(z))\big)\big]$$

donde $D(x;\theta_d)$ es la probabilidad de que $x$ sea real y $G(z;\theta_g)$ transforma $z$ para que se parezca a $x \sim p_{\text{data}}$.

> Ampliación (no cubierto en clase): el desarrollo siguiente procede del artículo citado en la diapositiva 13 (Goodfellow et al., 2014, §4), pero no aparece explícitamente en las transparencias.

**Discriminador óptimo.** Para $G$ fijo, el integrando $p_{\text{data}}(x)\log D(x) + p_g(x)\log(1-D(x))$ se maximiza puntualmente en

$$D^*_G(x) = \frac{p_{\text{data}}(x)}{p_{\text{data}}(x) + p_g(x)}$$

**Objetivo resultante.** Sustituyendo $D^*_G$ en $V$:

$$C(G) = -\log 4 + 2 \cdot \mathrm{JS}\big(p_{\text{data}} \,\|\, p_g\big)$$

con $\mathrm{JS}$ la divergencia de Jensen-Shannon. Como $\mathrm{JS} \ge 0$ y vale $0$ sólo si $p_g = p_{\text{data}}$, el óptimo global es $p_g = p_{\text{data}}$, y ahí $C(G) = -\log 4 \approx -1{,}386$ y $D^* \equiv 1/2$.

**El ancla numérica de 0,69.** Este resultado es el que da sentido operativo a las curvas de pérdida. En el equilibrio $D^* = 1/2$, y la entropía cruzada binaria que se optimiza en el código de clase vale:

$$\mathcal{L}_D = -\tfrac{1}{2}\Big[\log D(x) + \log\big(1 - D(G(z))\big)\Big]\Big|_{D=1/2} = \log 2 \approx 0{,}693$$

$$\mathcal{L}_G^{\text{NS}} = -\log D(G(z))\big|_{D=1/2} = \log 2 \approx 0{,}693$$

Es decir: **en una GAN sana ambas pérdidas deben rondar 0,69, no bajar a cero**. El cuaderno `docs/material_clase/notebooks/pix2pix.ipynb` lo dice de forma explícita en su sección de interpretación de logs: *"The value log(2) = 0.69 is a good reference point for these losses, as it indicates a perplexity of 2: that the discriminator is on average equally uncertain about the two options"*.

**Non-saturating loss.** El pseudocódigo de la diapositiva 18 actualiza $G$ descendiendo el gradiente de $\log(1 - D(G(z)))$. Ese término satura: si $D$ acierta con confianza ($D(G(z)) \to 0$), su gradiente respecto a $\theta_g$ tiende a cero justo cuando $G$ más necesita aprender. La corrección estándar es maximizar $\log D(G(z))$ en lugar de minimizar $\log(1-D(G(z)))$:

$$\mathcal{L}_G^{\text{NS}} = -\mathbb{E}_{z \sim p_z}\big[\log D(G(z))\big]$$

Mismo punto fijo, gradiente grande cuando $G$ va mal. **Todo el código de clase implementa ya esta versión**, aunque no lo nombre: en `GAN_1_Really_Simple_GAN_MNIST.ipynb` el paso del generador es

```python
y_mislabled = np.ones((batch, 1))            # se etiquetan los falsos como "reales"
g_loss = model_gan.train_on_batch(noise, y_mislabled)
```

que con `binary_crossentropy` es exactamente $-\log D(G(z))$.

**Punto de silla.** La diapositiva 15 (*"GANs: Saddle point"*) es la advertencia clave: el óptimo conjunto no es un mínimo de una superficie, es un **punto de silla** de $V(\theta_d, \theta_g)$. El descenso de gradiente alterno no garantiza convergencia a ese punto; puede orbitarlo indefinidamente. Esto no es un defecto de implementación: es la naturaleza del problema, y explica por qué las curvas oscilan.

## 3. Arquitectura: generador y discriminador

La referencia de clase (`docs/material_clase/notebooks/GAN_1_Really_Simple_GAN_MNIST.ipynb`) es un par de MLP deliberadamente asimétricos:

```python
# Generador: expansión progresiva desde el ruido latente
model_gen = Sequential()
model_gen.add(Dense(256, input_shape=(100,)))     # z de dimensión 100
model_gen.add(LeakyReLU(alpha=0.2))
model_gen.add(BatchNormalization(momentum=0.8))
model_gen.add(Dense(512));  model_gen.add(LeakyReLU(alpha=0.2))
model_gen.add(BatchNormalization(momentum=0.8))
model_gen.add(Dense(1024)); model_gen.add(LeakyReLU(alpha=0.2))
model_gen.add(BatchNormalization(momentum=0.8))
model_gen.add(Dense(np.prod(in_shape), activation='tanh'))   # salida en [-1, 1]

# Discriminador: contracción, mucho más pequeño que el generador
model_Disc = Sequential()
model_Disc.add(Flatten(input_shape=in_shape))
model_Disc.add(Dense(128)); model_Disc.add(LeakyReLU(alpha=0.2))
model_Disc.add(Dense(64));  model_Disc.add(LeakyReLU(alpha=0.2))
model_Disc.add(Dense(1, activation='sigmoid'))
```

Elementos que no son arbitrarios:

- **`tanh` en la salida de $G$.** Obliga a normalizar los datos reales al mismo rango. En los cuadernos de clase: `X_train = (X_train - 127.5) / 127.5`. Para datos financieros habrá que hacer el equivalente (sección 8), y la elección del escalado deja de ser un detalle: si un canal real se sale de $[-1,1]$, $G$ no puede alcanzarlo nunca.
- **`LeakyReLU(0.2)` en ambas redes.** Evita neuronas muertas en $D$; una ReLU normal puede anular el gradiente que llega a $G$ a través de $D$.
- **Asimetría de capacidad.** $G$ tiene ~1,5 M de parámetros y $D$ ~110 K. Es intencionado: un $D$ demasiado potente gana el juego y deja de dar gradiente útil (sección 6).
- **`BatchNormalization` en $G$, no en $D$.** El discriminador de clase no lleva BN. Con BN en $D$, las estadísticas del lote mezclan reales y falsos y filtran información entre muestras.
- **Adam con `beta_1 = 0.5` y `lr = 2e-4`.** Aparece idéntico en los cuatro cuadernos de GAN. Reducir el momento de primer orden amortigua las oscilaciones del juego adversarial.

La variante convolucional (`GAN_1_Really_Simple_GAN_MNIST_CONV.ipynb`) sustituye las densas por `Conv2D` + `UpSampling2D` en $G$ y `Conv2D` + `MaxPool2D` en $D$. Sólo tiene sentido si el dato tiene estructura espacial local con pesos compartidos; **para nuestro bloque aplanado de ~1.100 dimensiones no la tiene** en el eje de canales (los 18 canales no son traslacionalmente equivalentes), así que la referencia útil es la MLP.

El **modelo apilado** es el mecanismo por el que $G$ recibe gradiente:

```python
model_gan = Sequential()
model_gan.add(model_gen)
model_gan.add(model_Disc)   # D se usa como "función de pérdida" diferenciable
```

## 4. Bucle de entrenamiento y equilibrio D/G

El algoritmo 1 de Goodfellow (diap. 18) alterna: $k$ pasos de $D$ por cada paso de $G$, con $k=1$ como opción elegida por los autores (*"the least expensive option"*). El bucle de clase lo instancia así:

1. Tomar $m$ reales y generar $m$ falsos; entrenar $D$ sobre el lote combinado con etiquetas $[1,\dots,1,0,\dots,0]$.
2. Muestrear ruido nuevo; entrenar el modelo apilado con etiquetas todas a $1$, **con $D$ congelado**.

**El punto crítico es el congelado de $D$.** En Keras el atributo `trainable` se resuelve en el momento de `compile()`. El patrón correcto es el de `GAN_2_Simple_GAN_CIFAR10.ipynb`:

```python
self.D.compile(loss='binary_crossentropy', optimizer=..., metrics=['accuracy'])
self.D.trainable = False          # ANTES de compilar el modelo apilado
model = Sequential([self.G, self.D])
model.compile(...)                # el apilado sólo actualizará G
```

`Taller_GANs.ipynb` y `GAN_1_Really_Simple_GAN_MNIST_etiquetas_y_balanceo.ipynb` **no lo hacen**: compilan `model_gan` sin poner `model_Disc.trainable = False`. El efecto es que el paso del generador también actualiza $D$ empujándolo a clasificar los falsos como reales, es decir, sabotea al discriminador en el mismo paso en que se supone que sólo aprende $G$. `GAN_1_..._CONV.ipynb` intenta arreglarlo alternando `layer.trainable` dentro del bucle, pero al no recompilar, el cambio no se propaga a la función de entrenamiento ya trazada. **Es el primer defecto a corregir al portar el código.**

### 4.1 Crítica del `ratio` adaptativo de `Taller_GANs.ipynb`

El bucle del taller introduce un controlador que no está en el algoritmo original:

```python
ratio = 1
for cnt in range(epochs):
    batch_discr = int(np.round(ratio * batch))          # tamaño de lote de D
    ...
    batch_gen = int(np.round(batch / ratio))            # tamaño de lote de G
    ...
    ratio = (DD_loss[cnt] + 1) / (GG_loss[cnt] + 1)     # realimentación
```

**Qué hace.** Es un lazo de realimentación negativa sobre el desequilibrio del juego. Si $\mathcal{L}_D > \mathcal{L}_G$ (el discriminador va peor), $\text{ratio} > 1$ y $D$ recibe un lote mayor mientras $G$ lo recibe menor. Si $D$ va ganando ($\mathcal{L}_D < \mathcal{L}_G$), se le recorta el lote y se amplía el de $G$. La versión de `..._etiquetas_y_balanceo.ipynb` usa `ratio = DD_loss/GG_loss` sin el $+1$, que es la misma idea sin amortiguar.

**Por qué la idea es razonable.** El problema que ataca es real: el desequilibrio $D$/$G$ es *la* patología dominante (sección 6), y automatizarlo evita ajustar $k$ a mano.

**Por qué la implementación es mala idea.**

1. **Modula el tamaño de lote, no el número de pasos.** El grado de libertad que el algoritmo original expone es $k$ (frecuencia de actualización) a lote fijo. Cambiar el tamaño del lote altera simultáneamente el ruido del gradiente y, con Adam, el tamaño de paso efectivo. Se mezclan dos efectos que deberían separarse.
2. **Rompe BatchNorm.** En la traza de la diapositiva 23 de `docs/material_clase/slides/2026_Taller_Generativos.pdf` el lote de $D$ cae a 28 muestras. Con lotes tan pequeños y variables, las estadísticas de BN son ruido.
3. **Realimenta sobre una medida de un solo lote.** Con `batch = 10`, `d_loss` y `g_loss` son estimadores extremadamente ruidosos. El controlador reacciona a ruido, no a señal. No hay suavizado ni banda muerta.
4. **Muestreo contiguo, no aleatorio.** `random_index = np.random.randint(...)` seguido de `Datos[random_index : random_index + n]` toma un **bloque contiguo**. Sobre un panel financiero ordenado en el tiempo, cada lote de $D$ son ventanas solapadas y casi idénticas: no es una muestra i.i.d. de $p_{\text{data}}$. Para nuestro problema es un error grave, porque la mitad "real" de cada lote tiene una diversidad artificialmente baja y $D$ aprende a detectar el bloque, no la distribución.
5. **No funciona.** En la propia traza del taller (diaps. 18-24) el controlador no impide que $D$ gane: de $\mathcal{L}_D = 0{,}748$ / $\mathcal{L}_G = 0{,}731$ en el paso 0 se pasa a $\mathcal{L}_D = 0{,}407$ / $\mathcal{L}_G = 1{,}365$ con precisión de $D$ del 80% en el paso 9.100. Es exactamente el escenario que pretendía evitar.

**Alternativas, en orden de preferencia para este taller.**

- **$k = 1$ fijo con lote fijo** (la opción de Goodfellow). Simple, reproducible, y la línea base contra la que comparar.
- **$k$ adaptativo con banda muerta sobre la precisión suavizada de $D$**: mantener una media móvil exponencial $\overline{\text{acc}}_D$ y aplicar: si $\overline{\text{acc}}_D > 0{,}80$, saltar el paso de $D$; si $\overline{\text{acc}}_D < 0{,}55$, hacer dos pasos de $D$. Lote constante. Conserva la intención del `ratio` sin ninguno de sus cinco problemas.
- **Debilitar $D$ en lugar de reprogramarlo**: *one-sided label smoothing* (etiqueta real $=0{,}9$), `Dropout` en $D$, o reducir su capacidad.

> Ampliación (no cubierto en clase): TTUR (tasas de aprendizaje distintas para $D$ y $G$), *spectral normalization* y WGAN-GP con `n_critic = 5` son las soluciones estándar al mismo problema. Quedan fuera del alcance del material de clase y sólo merecen la pena si las tres opciones anteriores fallan.

## 5. Diagnóstico de convergencia

El enunciado (`docs/enunciado/Taller_B5_T1.pdf`) exige *"para cada entrenamiento, incluir las curvas de loss donde se vea que el modelo ha convergido"*. Aplicado a una GAN esta frase requiere traducción, porque **la pérdida de una GAN no baja monótonamente y no mide calidad**.

### 5.1 Por qué la pérdida no es un indicador de calidad

Tres razones independientes:

1. **La regla de medida cambia.** $\mathcal{L}_G$ mide el rendimiento de $G$ *contra el $D$ actual*. Si $D$ empeora, $\mathcal{L}_G$ baja sin que $G$ haya mejorado. Es una puntuación relativa dentro de una partida, no una distancia a $p_{\text{data}}$.
2. **El óptimo no es cero.** Por la sección 2, el equilibrio está en $\log 2 \approx 0{,}69$ para ambas pérdidas. Una $\mathcal{L}_D$ que tiende a $0$ es una **señal de fallo**, no de éxito: significa que $D$ separa perfectamente y $G$ ha dejado de recibir gradiente.
3. **El objetivo es un punto de silla** (diap. 15), no un mínimo. Oscilación permanente de amplitud acotada es el comportamiento *esperado*, no un síntoma.

### 5.2 Qué se mira realmente

| Indicador | Valor sano | Lectura patológica |
|---|---|---|
| $\mathcal{L}_D$ (lote combinado real+falso) | banda $0{,}55 - 0{,}80$, centrada en $\approx 0{,}69$ | $< 0{,}35$: $D$ ha ganado. $> 1{,}2$: $D$ colapsado |
| $\mathcal{L}_G$ (non-saturating) | banda $0{,}55 - 1{,}20$, sin deriva creciente | crecimiento sostenido: gradiente desapareciendo |
| Precisión de $D$ | $50\% - 70\%$ | $> 85\%$ mantenido: $D$ domina |
| $\overline{D(x_{\text{real}})}$ | $\to 0{,}5$ | $\to 1$ junto con $\overline{D(G(z))} \to 0$ |
| $\overline{D(G(z))}$ | $\to 0{,}5$ | $\to 0$ |
| Cociente $\mathcal{L}_D / \mathcal{L}_G$ | estable en torno a 1 | deriva monótona en cualquier dirección |
| Desviación típica entre muestras generadas | estable, comparable a la real | decreciente: *mode collapse* |
| Distancia muestral a los datos reales | **decreciente** | plana o creciente |

Las seis primeras filas dicen si **el juego sigue vivo**. Las dos últimas dicen si **el modelo es bueno**. Son preguntas distintas y hacen falta ambas: un juego equilibrado con muestras malas es perfectamente posible al principio del entrenamiento, y muestras razonables con $D$ ya ganado indican que se debe parar y quedarse con un *checkpoint* anterior.

El criterio de "convergencia" defendible en la presentación es, por tanto:

> El entrenamiento se considera convergido cuando (a) $\mathcal{L}_D$ y $\mathcal{L}_G$ llevan $N$ iteraciones oscilando dentro de una banda estable alrededor de $\log 2$ sin deriva, (b) la precisión de $D$ permanece por debajo del 75%, y (c) la métrica de calidad muestral evaluada en *checkpoints* ha dejado de mejorar (meseta), habiendo seleccionado el *checkpoint* de mejor métrica, no el último.

### 5.3 Qué graficar

Cuatro figuras por entrenamiento. Las tres primeras cubren la exigencia literal de "curvas de loss"; la cuarta es la que realmente justifica el modelo elegido.

**Figura 1 — Pérdidas adversariales.** $\mathcal{L}_D$ y $\mathcal{L}_G$ frente a iteración, con media móvil exponencial superpuesta sobre la traza cruda, y **una línea horizontal en $\log 2 = 0{,}693$**. Esa línea es lo que convierte una gráfica ilegible en una gráfica interpretable: sin ella, un revisor no puede juzgar nada.

**Figura 2 — Salidas medias de $D$.** $\overline{D(x_{\text{real}})}$ y $\overline{D(G(z))}$ en el mismo eje, con línea horizontal en $0{,}5$. Convergencia de ambas curvas hacia $0{,}5$ es la firma visual del equilibrio; separación creciente hacia $1$ y $0$ es la firma del fallo. Es más legible que las pérdidas.

**Figura 3 — Precisión de $D$** desglosada en reales y falsos, con línea en $50\%$. Detecta asimetrías que las medias ocultan.

**Figura 4 — Calidad muestral frente a iteración.** Evaluada cada $K$ iteraciones (p. ej. $K=500$) sobre un lote grande generado con **una semilla $z$ fija**, contra un conjunto de validación real. Métricas concretas, en orden de coste:

- **Distancia de Wasserstein-1 media sobre las marginales** de los canales (`scipy.stats.wasserstein_distance` canal a canal, promediada). Barata y sensible a colas.
- **Error de la matriz de correlación**: $\|\mathrm{Corr}(X_{\text{real}}) - \mathrm{Corr}(X_{\text{synth}})\|_F$. Detecta si $G$ acierta las marginales pero pierde la estructura de dependencia entre sectores, que es justo lo que nos interesa preservar.
- **Solapamiento en el espacio de componentes principales**: ajustar PCA sobre los reales y superponer el *scatter* de reales y sintéticos en las 2-3 primeras componentes. Es exactamente el diagnóstico usado en `docs/material_clase/slides/2026_Taller_Generativos.pdf` (diaps. 18-25): la nube roja de generados debe cubrir la azul de reales, no concentrarse en un subconjunto. Es cualitativo pero es el más informativo de todos y va directo a la presentación.
- **Utilidad *downstream*** en *checkpoints* seleccionados: entrenar el modelo de la sección 8 con datos sintéticos y evaluar sobre reales. Es la métrica que el enunciado evalúa realmente; es cara, así que se calcula sólo en 3-5 puntos.

> Ampliación (no cubierto en clase): esta última idea se conoce como TSTR (*Train on Synthetic, Test on Real*). La FID, estándar en imagen, no es aplicable aquí porque requiere una red preentrenada de características sobre el dominio.

**Figura 5 (opcional, alto valor en la presentación)** — Rejilla de trayectorias generadas con $z$ fija en varios *checkpoints*. El equivalente para series temporales del panel de dígitos de `plot_images()` en `GAN_2_Simple_GAN_CIFAR10.ipynb`. Muestra visualmente la evolución y delata el colapso de modo mejor que cualquier escalar.

### 5.4 Instrumentación mínima

```python
# Se registra en cada iteración; el coste es despreciable frente al paso de optimización
historial = {
    "d_loss": [], "g_loss": [],
    "d_out_real": [], "d_out_fake": [],   # medias de D(x) y D(G(z))
    "acc_real": [], "acc_fake": [],
    "grad_norm_g": [],                    # detecta gradiente desvaneciente
    "std_muestras": [],                   # detecta mode collapse
}

def registrar(hist, d_loss, g_loss, d_real, d_fake, gen, x_fake):
    hist["d_loss"].append(d_loss);   hist["g_loss"].append(g_loss)
    hist["d_out_real"].append(d_real.mean().item())
    hist["d_out_fake"].append(d_fake.mean().item())
    hist["acc_real"].append((d_real > 0.5).float().mean().item())
    hist["acc_fake"].append((d_fake < 0.5).float().mean().item())
    # norma global del gradiente del generador tras backward()
    hist["grad_norm_g"].append(
        sum(p.grad.norm().item() ** 2 for p in gen.parameters() if p.grad is not None) ** 0.5
    )
    # dispersión media entre muestras del lote generado
    hist["std_muestras"].append(x_fake.std(dim=0).mean().item())
```

Con esto, la sección de resultados puede afirmar "el modelo ha convergido" y **respaldarlo con números**, en lugar de enseñar dos curvas ruidosas sin referencia.

## 6. Patologías (mode collapse, vanishing gradients, oscilación) y cómo detectarlas

**Mode collapse.** $G$ descubre un subconjunto de muestras que engaña a $D$ y colapsa toda la masa de probabilidad ahí. La marginal generada puede seguir siendo plausible mientras la diversidad ha desaparecido.
*Detección:* `std_muestras` decreciente; nube de sintéticos concentrada en una región del espacio PCA mientras la real se extiende (comparar con diap. 25 de `docs/material_clase/slides/2026_Taller_Generativos.pdf`, donde los generados sí cubren el rango de los reales); distancias al vecino más próximo entre muestras generadas anormalmente pequeñas. **En una cGAN, además: contar cuántos regímenes distintos aparecen realmente entre las muestras condicionadas.**
*Mitigación con material de clase:* bajar la capacidad o la tasa de aprendizaje de $G$, aumentar el tamaño de lote, no dejar que $D$ se estanque.
> Ampliación (no cubierto en clase): *minibatch discrimination*, *minibatch standard deviation*, *unrolled GAN*.

**Vanishing gradients.** $D$ separa perfectamente, $D(G(z)) \to 0$, y con la pérdida saturante el gradiente hacia $G$ se anula. Es la patología documentada en la propia traza del profesor: paso 9.100, $\mathcal{L}_D = 0{,}407$, precisión 80%, $\mathcal{L}_G = 1{,}365$ (diap. 23).
*Detección:* `grad_norm_g` cayendo dos o tres órdenes de magnitud; `acc_real` y `acc_fake` ambas por encima de 0,85; $\mathcal{L}_D \ll 0{,}69$ sostenido.
*Mitigación:* pérdida non-saturating (sección 2, ya presente en el código de clase); *label smoothing* a $0{,}9$; `Dropout` en $D$; saltar pasos de $D$ según la regla de banda muerta de la sección 4.1.

**Oscilación / no convergencia.** Las pérdidas ciclan sin estabilizarse y las muestras cambian de carácter periódicamente. Consecuencia directa del punto de silla (diap. 15).
*Detección:* autocorrelación de las pérdidas suavizadas con periodicidad marcada; la métrica de calidad de la Figura 4 sube y baja en lugar de asentarse.
*Mitigación:* `beta_1 = 0.5` en Adam (ya en el código de clase), bajar la tasa de aprendizaje, aumentar el lote.
> Ampliación (no cubierto en clase): promedio exponencial (EMA) de los pesos de $G$ para muestrear. Es barato, no toca el entrenamiento y estabiliza mucho las muestras en presencia de oscilación; se incluye en la implementación de la sección 9.

**Sobreajuste del discriminador.** Con pocos datos $D$ memoriza el conjunto de entrenamiento y $G$ aprende a copiarlo. **Riesgo alto en nuestro caso**: si la clase "crisis" tiene ~10% de unos pocos miles de ventanas, el número de ejemplos de crisis realmente independientes es muy bajo.
*Detección:* mantener un conjunto de validación real que $D$ no vea nunca y monitorizar $\mathcal{L}_D$ sobre él; una brecha creciente entre $\mathcal{L}_D^{\text{train}}$ y $\mathcal{L}_D^{\text{val}}$ es sobreajuste.
*Mitigación:* reducir $D$, `Dropout`, y comprobar explícitamente que las muestras generadas **no son copias**: distancia al vecino más próximo en el conjunto de entrenamiento, comparada con la distribución de distancias entre reales.

**Tabla de decisión rápida**

| Síntoma observado | Causa probable | Acción inmediata |
|---|---|---|
| $\mathcal{L}_D \to 0$, $\mathcal{L}_G$ creciente | $D$ demasiado fuerte | *label smoothing*, `Dropout` en $D$, saltar pasos de $D$ |
| $\mathcal{L}_D \to 1{,}4$, $\mathcal{L}_G \to 0$ | $D$ colapsado (bug de congelado) | revisar `trainable` / optimizadores separados |
| `std_muestras` decreciente | *mode collapse* | bajar `lr` de $G$, subir tamaño de lote |
| Ambas pérdidas planas en $0{,}69$ desde la iteración 0 | $D$ no aprende nada | subir capacidad o `lr` de $D$; revisar normalización de datos |
| Pérdidas sanas, muestras malas | falta entrenamiento | seguir; vigilar la Figura 4 |

## 7. GAN condicional (cGAN): cómo se inyecta la etiqueta

La cGAN (`docs/material_clase/slides/GANs_general.pdf`, diaps. 22-26; Mirza & Osindero, arXiv:1411.1784) añade una variable observada $y$ a ambas redes:

$$\min_G \max_D V(D,G) = \mathbb{E}_{x \sim p_{\text{data}}}\big[\log D(x|y)\big] + \mathbb{E}_{z \sim p_z}\big[\log\big(1 - D(G(z|y))\big)\big]$$

El cambio es mínimo en la ecuación y decisivo en la práctica: $G$ deja de modelar $p(x)$ y pasa a modelar $p(x \mid y)$, con lo que **la etiqueta pasa a ser una entrada que controlamos en el momento de generar**. La diapositiva 25 lo ilustra: cada fila de dígitos MNIST está condicionada a una etiqueta distinta.

**Mecanismo de inyección (diap. 24).** El esquema canónico es concatenación en la entrada de ambas redes:

- $G$: entrada $[\,z \;\Vert\; \text{onehot}(y)\,]$ de dimensión $d_z + C$.
- $D$: entrada $[\,x \;\Vert\; \text{onehot}(y)\,]$ de dimensión $d_x + C$.

Es imprescindible condicionar **también** a $D$. Si sólo se condiciona $G$, nada obliga a que la muestra generada corresponda a la etiqueta pedida: $D$ no puede penalizar la incoherencia porque no ve la etiqueta.

La diapositiva 26 (*"Distintas opciones"*) recoge las variantes: **cGAN** (etiqueta a $D$ y $G$), **Semi-Supervised GAN** ($D$ emite $C+1$ clases), **InfoGAN** (código latente no supervisado) y **AC-GAN** ($D$ emite además un clasificador auxiliar de clase). Para nuestro caso, con etiqueta observada y discreta, la cGAN estándar es suficiente.

> Ampliación (no cubierto en clase): con más de un puñado de clases, la concatenación de *one-hot* pierde fuerza y se sustituye por un *embedding* aprendido más un *projection discriminator* (Miyato & Koyama, 2018) o *conditional BatchNorm*. Con $C=3$ no es necesario.

### 7.1 Qué hace realmente `GAN_1_..._etiquetas_y_balanceo.ipynb`

Conviene leer ese cuaderno con precisión, porque **no implementa una cGAN**:

```python
Y_train = (Y_train - 4.5) / 4.5                        # etiqueta reescalada a [-1, 1] para tanh
Datos = np.zeros((X_train.shape[0], 28*28 + 1))
Datos[:, 0:28*28] = X_train.reshape(X_train.shape[0], -1)
Datos[:, -1]      = Y_train                             # la etiqueta se anexa como una dimensión más
```

La etiqueta se concatena **al vector de datos**, y la GAN aprende la distribución conjunta $p(x, y)$ del bloque $[x \Vert y]$. Al generar, la etiqueta sale como una salida más y se redondea:

```python
syntetic_labels = np.round(aux_synth[:, -1] * 4.5 + 4.5)
```

Esto es la **OPT2** de `docs/material_clase/slides/2026_Taller_Generativos.pdf` (diaps. 8 y 13): *"GANs for generating input-output pairs"*. Es una técnica válida y muy simple, pero tiene una consecuencia que nos afecta directamente:

> **Con el bloque conjunto no se puede controlar la etiqueta.** La proporción de clases generadas reproduce la del conjunto de entrenamiento, incluido el desequilibrio. Si "crisis" es el 10% de los datos reales, será ~10% de los sintéticos. No se puede sobre-generar la clase rara, que es exactamente lo que necesitamos.

Además, la etiqueta generada es un valor continuo redondeado: nada garantiza que caiga limpiamente en una clase, ni que la etiqueta redondeada sea coherente con el $x$ generado.

La **OPT3/OPT4** de la diapositiva 9 es la alternativa condicional: la etiqueta (o las características de entrada) se fija y el generador produce el resto. Ese es el diseño que adoptamos.

### 7.2 Balanceo mediante condicionamiento

Con una cGAN el balanceo es trivial y explícito: la distribución de $y$ en el momento de generar es una **decisión de diseño**, desacoplada de la distribución empírica.

```python
# Generación balanceada: se decide cuántas muestras de cada régimen se quieren
n_por_clase = {0: 2000, 1: 2000, 2: 6000}   # 2 = crisis, deliberadamente sobre-representada
y_gen = torch.cat([torch.full((n,), c, dtype=torch.long)
                   for c, n in n_por_clase.items()])
z = torch.randn(len(y_gen), D_Z)
x_gen = generador(z, y_gen)                  # la etiqueta es entrada, no salida
```

La etiqueta del dato sintético es **exacta por construcción**, no estimada. Este es el motivo por el que la cGAN, y no la GAN del bloque conjunto, es el modelo adecuado para nuestro problema.

Advertencia: condicionar no crea información. Si sólo hay un puñado de episodios de crisis reales, la cGAN aprenderá $p(x \mid \text{crisis})$ a partir de esos pocos episodios y sus muestras serán variaciones de ellos. Sobre-generar la clase rara amplifica lo que hay; no descubre crisis nuevas. Hay que decirlo en la presentación y respaldarlo con la comprobación de vecino más próximo de la sección 6.

## 8. Aplicación a nuestro problema

**Problema.** Panel híbrido diario: S&P500, 9 SPDR sectoriales, VIX, MOVE, spreads de crédito, pendiente de curva, *drawdown* y volatilidad realizada, ~18 canales. Una muestra es una ventana de 60 días, $X \in \mathbb{R}^{60 \times 18}$. Objetivos: $y_{\text{reg}}$ = régimen de mercado a 21 días (3 clases, "crisis" ~10%) y $y_{\text{vol}}$ = volatilidad futura.

**Elección de opción según el taller.** La diapositiva 8 de `docs/material_clase/slides/2026_Taller_Generativos.pdf` distingue OPT1 (generar sólo entradas) y OPT2 (generar pares entrada-salida). Generamos **el bloque conjunto** $[X \Vert y_{\text{reg}} \Vert y_{\text{vol}}]$, es decir OPT2, pero **condicionado al régimen** (diap. 9), de modo que $y_{\text{reg}}$ se fija como entrada y el generador produce $[X \Vert y_{\text{vol}}]$, de dimensión $60 \times 18 + 1 = 1081$. El régimen se reanexa al bloque después de generar, con lo que la etiqueta es exacta y la dimensión final del bloque es ~1.100 como estaba previsto.

**Justificación del condicionamiento.** Es la razón de ser del diseño: la clase minoritaria (~10%) es la que limita el rendimiento del modelo *downstream* y la que menos ejemplos reales tiene. Condicionar permite pedir 6.000 ventanas de crisis sintéticas frente a 2.000 de cada régimen normal.

**Procedimiento de 4 pasos** (diaps. 14-17 de `docs/material_clase/slides/2026_Taller_Generativos.pdf`):

- **STEP 1 — Entrenar la cGAN** con $\{X, y\}$ **del conjunto de entrenamiento únicamente**. Partición estrictamente temporal: entrenamiento hasta $T_1$, validación $[T_1, T_2]$, test $> T_2$, con un hueco de al menos 60+21 días entre bloques para evitar solapamiento de ventanas. Si la GAN ve ventanas que solapan con el test, cualquier mejora medida es fuga de información.
- **STEP 2 — Generar datos sintéticos** $\{X_g, y_g\}$ con la mezcla de regímenes elegida.
- **STEP 3 — Entrenar el modelo *downstream*** en varias versiones: sólo reales (NN 1) y reales + sintéticos en distintas proporciones (NN 2), **con la misma arquitectura y los mismos hiperparámetros** en todas las versiones. Es requisito explícito del enunciado.
- **STEP 4 — Evaluar todas las versiones sobre el mismo test real** y comparar.

**Preprocesado.** La salida `tanh` obliga a llevar cada canal a $[-1,1]$ con estadísticos calculados **sólo sobre entrenamiento**. Los retornos financieros tienen colas gruesas, así que un min-max simple deja el 99% de los datos aplastado en torno a cero. Recomendación: escalado por cuantiles robustos (p. ej. percentiles 0,5 y 99,5) seguido de recorte a $[-1,1]$, guardando el escalador para invertirlo tras generar. El desescalado es parte del modelo generativo, no un detalle.

**Expectativas realistas.** El resultado de referencia del material (diap. 28) es: sólo datos reales 4,81 K de error, reales + GAN 3,46 K, reales + RBIG 3,62 K. La GAN ayuda, y algo más que RBIG. Pero las diapositivas 30-32 (*"GANs vs RBIG"*) muestran el matiz que hay que interiorizar: con 500 y 1.000 datos reales la mejora al añadir sintéticos es grande; con 3.000 y sobre todo con 7.000 reales la ventaja se reduce y en algunas combinaciones **añadir sintéticos empeora**. La conclusión operativa:

> El beneficio de los datos sintéticos se concentra en el régimen de pocos datos. En nuestro problema, el "régimen de pocos datos" es la clase crisis. Por eso las curvas de resultados deben desglosarse **por clase**, y no sólo reportar métricas agregadas: es perfectamente posible que la métrica global no se mueva mientras el *recall* de crisis mejora de forma clara, y ese es el resultado que hace interesante el trabajo.

**Línea base obligatoria.** El enunciado exige un cuarto modelo simple. `Taller_GANs.ipynb` lo implementa como datos reales + ruido gaussiano:

```python
sig = 0.01
X_ruido = X_train[:n] + rng.normal(0, sig, X_train[:n].shape)
```

Es una línea base exigente y hay que reportarla con honestidad: si la cGAN no la supera, el resultado del taller es que la cGAN no aporta, y eso también es un resultado defendible siempre que esté bien medido.

## 9. Implementación de referencia (CPU)

Entorno: `torch 2.11.0+cpu`, Python 3.13.7, **sin CUDA**. Esto descarta arquitecturas convolucionales profundas y fija el dimensionado en una MLP condicional de 3-4 capas, en la línea del código de clase.

**Dimensionado y coste.** Con $d_z = 128$, $C = 3$, $d_{\text{out}} = 1081$:

| Red | Capas | Parámetros |
|---|---|---|
| $G$ | $131 \to 512 \to 1024 \to 1081$ | ≈ 1,70 M |
| $D$ | $1084 \to 512 \to 256 \to 1$ | ≈ 0,69 M |

Total ≈ 2,4 M parámetros. Con lote 64 y `torch.set_num_threads` ajustado a los núcleos físicos, el coste por iteración es del orden de **15-40 ms**; 20.000 iteraciones son aproximadamente **5-13 minutos**. Un barrido de 3 semillas × 4 mezclas de sintéticos cabe holgadamente en una tarde. Estas cifras son un orden de magnitud: hay que medirlas y reportarlas.

```python
import torch
import torch.nn as nn

torch.set_num_threads(8)          # ajustar al número de núcleos físicos
torch.manual_seed(0)

D_Z, N_CLASES, D_OUT = 128, 3, 60 * 18 + 1   # ruido, regímenes, bloque [X aplanada | y_vol]


class Generador(nn.Module):
    """G(z | y): concatena el ruido con la etiqueta one-hot del régimen."""
    def __init__(self):
        super().__init__()
        self.red = nn.Sequential(
            nn.Linear(D_Z + N_CLASES, 512),
            nn.BatchNorm1d(512), nn.LeakyReLU(0.2),
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024), nn.LeakyReLU(0.2),
            nn.Linear(1024, D_OUT),
            nn.Tanh(),                      # salida en [-1, 1]: los datos deben ir escalados igual
        )

    def forward(self, z, y):
        y_oh = nn.functional.one_hot(y, N_CLASES).float()
        return self.red(torch.cat([z, y_oh], dim=1))


class Discriminador(nn.Module):
    """D(x | y): la etiqueta entra tambien aqui; sin ella no puede penalizar
    que la muestra generada no corresponda al regimen solicitado."""
    def __init__(self):
        super().__init__()
        self.red = nn.Sequential(
            nn.Linear(D_OUT + N_CLASES, 512), nn.LeakyReLU(0.2), nn.Dropout(0.3),
            nn.Linear(512, 256),             nn.LeakyReLU(0.2), nn.Dropout(0.3),
            nn.Linear(256, 1),               # logits: se usa BCEWithLogits (mas estable)
        )

    def forward(self, x, y):
        y_oh = nn.functional.one_hot(y, N_CLASES).float()
        return self.red(torch.cat([x, y_oh], dim=1))


import copy

gen, disc = Generador(), Discriminador()
gen_ema = copy.deepcopy(gen).eval()      # copia promediada de G, se usa solo para muestrear
for p in gen_ema.parameters():
    p.requires_grad_(False)

# Dos optimizadores separados: en PyTorch no existe el problema de congelado de Keras,
# cada paso actualiza solo los parametros de su propio optimizador.
opt_g = torch.optim.Adam(gen.parameters(),  lr=2e-4, betas=(0.5, 0.999))
opt_d = torch.optim.Adam(disc.parameters(), lr=2e-4, betas=(0.5, 0.999))
bce = nn.BCEWithLogitsLoss()

LOTE, SUAVIZADO = 64, 0.9      # one-sided label smoothing: la etiqueta "real" es 0.9, no 1.0
acc_d_ema = 0.5                # media movil de la precision de D, para la banda muerta
```

Bucle de entrenamiento con muestreo aleatorio real y regulación por banda muerta (sustituye al `ratio` de `Taller_GANs.ipynb`, sección 4.1):

```python
for it in range(N_ITER):
    # ---- muestreo ALEATORIO, no un bloque contiguo -------------------------
    idx = torch.randint(0, len(X_real), (LOTE,))
    x_real, y_real = X_real[idx], y_real_all[idx]

    z = torch.randn(LOTE, D_Z)
    y_fake = y_real_all[torch.randint(0, len(y_real_all), (LOTE,))]
    x_fake = gen(z, y_fake)

    # ---- paso del discriminador --------------------------------------------
    # Se evalua siempre (hace falta para monitorizar), pero solo se actualiza
    # si D no esta dominando: banda muerta que sustituye al `ratio` adaptativo.
    opt_d.zero_grad()
    out_real = disc(x_real, y_real)
    out_fake = disc(x_fake.detach(), y_fake)           # detach: no propaga a G
    d_loss = (bce(out_real, torch.full_like(out_real, SUAVIZADO))
              + bce(out_fake, torch.zeros_like(out_fake))) / 2
    if acc_d_ema < 0.80:
        d_loss.backward()
        opt_d.step()

    # ---- paso del generador (non-saturating: etiquetas a 1) ---------------
    opt_g.zero_grad()
    out_fake_g = disc(x_fake, y_fake)
    g_loss = bce(out_fake_g, torch.ones_like(out_fake_g))
    g_loss.backward()
    opt_g.step()

    # ---- EMA de los pesos de G: se muestrea de gen_ema, no de gen ----------
    with torch.no_grad():
        for p_ema, p in zip(gen_ema.parameters(), gen.parameters()):
            p_ema.mul_(0.999).add_(p, alpha=0.001)

    # ---- monitorizacion (seccion 5.4) -------------------------------------
    with torch.no_grad():
        acc = ((torch.sigmoid(out_real) > 0.5).float().mean()
               + (torch.sigmoid(out_fake) < 0.5).float().mean()) / 2
        acc_d_ema = 0.99 * acc_d_ema + 0.01 * acc.item()
    registrar(historial, d_loss.item(), g_loss.item(),
              torch.sigmoid(out_real), torch.sigmoid(out_fake), gen, x_fake)

    if it % 500 == 0:
        guardar_checkpoint(gen_ema, it)      # se elige por metrica, no por ser el ultimo
```

Notas de implementación que importan:

- **`BCEWithLogitsLoss` en lugar de `Sigmoid` + `BCELoss`.** Numéricamente estable cuando $D$ se vuelve confiado, que es justo cuando el entrenamiento se rompe.
- **`x_fake.detach()` en el paso de $D$.** Equivalente correcto al `trainable = False` de Keras, sin su ambigüedad.
- **Sin `BatchNorm` en $D$, sí en $G$.** Coherente con el código de clase y evita fugas de información entre muestras del lote en $D$.
- **`gen_ema` para muestrear.** Copia de $G$ actualizada por media móvil; elimina buena parte del ruido de las oscilaciones sin coste apreciable. Es una ampliación sobre el material de clase, pero prácticamente gratuita.
- **Semillas fijas y `z` de evaluación fija** para que las figuras de las secciones 5.3 sean comparables entre *checkpoints* y entre ejecuciones.

## 10. Referencias

**Material de clase**

- `docs/material_clase/slides/GANs_general.pdf` — Valero Laparra, *Intro GANs*. Diap. 10 (arquitectura), 13 (minimax), 14 (coste adaptativo), 15 (punto de silla), 16 (evolución GAN → cGAN → AE-like), 18 (Algoritmo 1), 22-26 (cGAN y variantes), 27-31 (pix2pix), 33-43 (CycleGAN).
- `docs/material_clase/slides/2026_Taller_Generativos.pdf` — Diap. 8-9 (OPT1-OPT4), 13-17 (procedimiento de 4 pasos), 18-24 (traza de entrenamiento con pérdidas y precisiones), 25-26 (diagnóstico por componentes principales), 27-28 (resultados: 4,81 K → 3,46 K), 30-32 (GANs vs RBIG por volumen de datos reales).
- `docs/material_clase/notebooks/GAN_1_Really_Simple_GAN_MNIST.ipynb` — GAN MLP mínima, arquitectura de referencia.
- `docs/material_clase/notebooks/GAN_1_Really_Simple_GAN_MNIST_CONV.ipynb` — variante convolucional.
- `docs/material_clase/notebooks/GAN_1_Really_Simple_GAN_MNIST_etiquetas_y_balanceo.ipynb` — bloque conjunto $[x \Vert y]$ y `ratio` adaptativo. Referencia directa para la sección 7.1.
- `docs/material_clase/notebooks/GAN_2_Simple_GAN_CIFAR10.ipynb` — patrón correcto de congelado de $D$ y utilidad `plot_images()`.
- `docs/material_clase/notebooks/Taller_GANs.ipynb` — GAN sobre bloque conjunto de datos del S&P500; bucle con `ratio` adaptativo y línea base de ruido.
- `docs/material_clase/notebooks/pix2pix.ipynb` — cGAN con pérdida $\mathcal{L}_{\text{GAN}} + \lambda \mathcal{L}_1$ ($\lambda = 100$), PatchGAN, y la guía de interpretación de logs con la referencia $\log 2 = 0{,}69$.
- `docs/material_clase/notebooks/cyclegan_texto_2.ipynb` — traducción sin pares con pérdida de consistencia cíclica ($\lambda = 10$).
- `docs/enunciado/Taller_B5_T1.pdf` — enunciado del taller B5-T1.

**Artículos citados en las transparencias**

- Goodfellow, I. et al. (2014). *Generative Adversarial Networks*. arXiv:1406.2661.
- Mirza, M. & Osindero, S. (2014). *Conditional Generative Adversarial Nets*. arXiv:1411.1784 (diap. 23).
- Isola, P. et al. (2017). *Image-to-Image Translation with Conditional Adversarial Networks* (pix2pix). arXiv:1611.07004 (diap. 27).
- Zhu, J.-Y. et al. (2017). *Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks* (CycleGAN). arXiv:1703.10593 (diap. 33).
- Odena, A. et al. (2017). *Conditional Image Synthesis with Auxiliary Classifier GANs* (AC-GAN) (diap. 26).
- Chen, X. et al. (2016). *InfoGAN* (diap. 26).
- Chu, C. et al. (2017). *CycleGAN, a Master of Steganography*. arXiv:1712.02950 (diap. 42).
- Hoffman, J. et al. (2018). *CyCADA: Cycle-Consistent Adversarial Domain Adaptation* (diap. 43).

> Ampliación (no cubierto en clase): Salimans, T. et al. (2016), *Improved Techniques for Training GANs* (label smoothing, minibatch discrimination); Arjovsky, M. et al. (2017), *Wasserstein GAN*; Gulrajani, I. et al. (2017), *WGAN-GP*; Miyato, T. & Koyama, M. (2018), *cGANs with Projection Discriminator*; Esteban, C. et al. (2017), *Real-valued (Medical) Time Series Generation with Recurrent Conditional GANs* (protocolo TSTR).
