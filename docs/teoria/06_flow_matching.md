# Flow matching

Documento teórico del taller B5-T1 (generación de datos financieros sintéticos).

**Nota sobre las fuentes.** El material de clase cubre flow matching en dos sitios: `../docs/material_clase/slides/Normalizing Flows_2026.pdf`, diapositivas 20–23 (presentación conceptual —"*Normalizing Flows pero en suave, y estimando el cambio (la velocidad). La red predice cómo hay que cambiar el dato en el instante $t$: $V = M(x_t,t)$, $x_{t+1} = x_t + v$*"—, ReFlow y una tabla comparativa flujo clásico / difusión / OT-CFM), y los notebooks `FlowMatching_simple.ipynb` y `FlowMatching_imagenes.ipynb`, con la implementación completa en PyTorch (2D y MNIST).

El PDF de difusión, `../docs/material_clase/slides/Diffusion_Models_DM_2026.pdf`, **no cubre flow matching ni la probability flow ODE**: sus 33 páginas van de autoencoders de denoising a DDPM, DDIM, LDM y ControlNet, y solo menciona "Flow Matching" una vez, en la línea temporal de la última página ("2025 – Flow Matching"). Toda la formulación matemática de las secciones 2–4 se apoya por tanto en los notebooks y en la literatura estándar (Lipman et al. 2023; Liu et al. 2023), y va marcada como ampliación cuando excede lo visto en clase.

## 1. Intuición: aprender un campo de velocidades entre dos distribuciones

El problema generativo es siempre el mismo: convertir una distribución fácil de muestrear $p_0$ (típicamente $\mathcal{N}(0, I)$) en la distribución de los datos $p_1$. Los distintos modelos difieren en *cómo* representan ese transporte.

Flow matching lo representa como un **campo de velocidades**. Imaginemos cada muestra de ruido como una partícula en $\mathbb{R}^D$ y el tiempo $t \in [0,1]$ como un tiempo físico. Si conociéramos, para cada posición $x$ y cada instante $t$, la velocidad $v(x,t)$ que hay que imprimir a una partícula que pasa por ahí, generar sería trivial: soltamos una partícula en $x_0 \sim \mathcal{N}(0,I)$ y la dejamos avanzar hasta $t=1$. Donde acabe es una muestra sintética.

El truco del entrenamiento es que la velocidad *objetivo* se puede escribir de forma explícita si emparejamos ruido y dato. Tomamos $x_0 \sim \mathcal{N}(0,I)$ y $x_1 \sim p_{\text{datos}}$ de forma independiente, trazamos el **segmento recto** que los une y muestreamos un punto intermedio:

$$x_t = (1-t)\,x_0 + t\,x_1, \qquad t \sim \mathcal{U}[0,1]$$

La velocidad de una partícula que recorre ese segmento a ritmo constante es, por derivación directa,

$$u_t = \frac{d x_t}{dt} = x_1 - x_0$$

que no depende de $t$. Y entrenar consiste en hacer regresión de mínimos cuadrados de la red sobre ese vector. El notebook `FlowMatching_simple.ipynb` es exactamente esto en 15 líneas: celdas 9–14 dibujan paso a paso los puntos $x_0$ (rojo), $x_1$ (azul), el segmento verde que los une, el punto intermedio $x_t$ (negro) y la flecha $u_t$ (azul) frente a la predicción de la red (cian).

Dos observaciones que conviene tener claras desde el principio:

1. **El entrenamiento es *simulation-free***. No hay que integrar nada: el objetivo de cada muestra se calcula con una resta. Esto es lo que separa flow matching de los CNF originales (sección 2).
2. **Las trayectorias de entrenamiento son rectas, pero el campo aprendido no genera trayectorias rectas.** Distintos pares $(x_0, x_1)$ producen segmentos que se cruzan; en un cruce la red no puede predecir dos velocidades a la vez y devuelve el promedio. El campo resultante es suave y sus curvas integrales son curvas. La celda 16 del notebook lo visualiza con 200 pasos y flechas. De aquí sale la motivación de ReFlow (sección 4).

## 2. Formulación: continuous normalizing flows y la ODE de transporte

Un **flujo continuo** (continuous normalizing flow, CNF) define la transformación mediante una ecuación diferencial ordinaria parametrizada por una red:

$$\frac{d}{dt}\psi_t(x) = v_\theta(\psi_t(x), t), \qquad \psi_0(x) = x$$

El mapa $\psi_t : \mathbb{R}^D \to \mathbb{R}^D$ es el *flow map*. Si $x_0 \sim p_0$, la distribución de $\psi_t(x_0)$ es el *push-forward* $p_t = [\psi_t]_\# p_0$, y ese camino de densidades cumple la **ecuación de continuidad**:

$$\frac{\partial p_t(x)}{\partial t} + \nabla \cdot \big( p_t(x)\, v_t(x) \big) = 0$$

Es la ley de conservación de masa de la mecánica de fluidos: la densidad no se crea ni se destruye, solo se transporta. Cuando $v_t$ y $p_t$ la satisfacen, decimos que $v_t$ **genera** $p_t$.

La conexión con los flujos normalizantes vistos en clase (`../docs/material_clase/slides/Normalizing Flows_2026.pdf`, diapositivas 5–18) es directa. Allí la restricción era que $f$ fuese invertible y diferenciable, y el coste estaba en el determinante del jacobiano del cambio de variable. En el caso continuo ese determinante se convierte en una integral de la divergencia:

$$\log p_1(\psi_1(x)) = \log p_0(x) - \int_0^1 \nabla \cdot v_\theta(\psi_t(x), t)\, dt$$

Esto es una ventaja arquitectónica enorme: $v_\theta$ puede ser *cualquier* red, sin capas de acoplamiento ni jacobianos triangulares. Pero entrenar por máxima verosimilitud exige, por cada paso de gradiente, integrar la ODE hacia delante, estimar la traza del jacobiano (estimador de Hutchinson) y retropropagar a través del solver o mediante el método adjunto. Es lento, numéricamente delicado y en la práctica limitó los CNF a dimensiones bajas.

> Ampliación (no cubierto en clase): la formulación CNF + método adjunto es de Chen et al. 2018 (*Neural ODEs*); el estimador de traza de Hutchinson aplicado a CNF es de Grathwohl et al. 2019 (*FFJORD*).

**El objetivo ingenuo de flow matching.** Si fijáramos de antemano un camino $p_t$ con $p_0 = \mathcal{N}(0,I)$ y $p_1 \approx p_{\text{datos}}$, y existiera un campo $u_t$ que lo genera, bastaría con hacer regresión:

$$\mathcal{L}_{\text{FM}}(\theta) = \mathbb{E}_{t \sim \mathcal{U}[0,1],\; x \sim p_t} \Big[ \big\| v_\theta(x,t) - u_t(x) \big\|^2 \Big]$$

Si esta pérdida llegara a cero, $v_\theta$ generaría exactamente $p_t$ y muestrear sería integrar la ODE. El problema es que **este objetivo no es computable**: no sabemos muestrear de $p_t$ ni conocemos $u_t(x)$ en forma cerrada, porque ambos dependen de la densidad de los datos, que es justo lo que desconocemos.

## 3. El objetivo de flow matching condicional (CFM) y por qué es tratable

La solución (Lipman et al. 2023) es construir el camino marginal como **mezcla de caminos condicionales** anclados cada uno en un dato concreto, $p_t(x) = \int p_t(x \mid x_1)\, p_{\text{datos}}(x_1)\, dx_1$, eligiéndolos de forma que $p_t(x\mid x_1)$ y su campo generador $u_t(x \mid x_1)$ sean triviales. Con el camino lineal de la sección 1 y ruido $x_0 \sim \mathcal{N}(0,I)$:

$$x_t \mid (x_0, x_1) = (1-t)x_0 + t x_1, \qquad u_t(x_t \mid x_0, x_1) = x_1 - x_0$$

y el objetivo **condicional** es

$$\boxed{\;\mathcal{L}_{\text{CFM}}(\theta) = \mathbb{E}_{t \sim \mathcal{U}[0,1],\; x_0 \sim \mathcal{N}(0,I),\; x_1 \sim p_{\text{datos}}} \Big[ \big\| v_\theta(x_t, t) - (x_1 - x_0) \big\|^2 \Big]\;}$$

Todo lo que aparece dentro de la esperanza se puede muestrear: $x_1$ es un dato del batch, $x_0$ es `torch.randn_like`, $t$ es `torch.rand`. Ni una integral, ni un solver, ni un jacobiano.

**El teorema clave** es que ambos objetivos tienen el mismo gradiente, $\nabla_\theta \mathcal{L}_{\text{FM}} = \nabla_\theta \mathcal{L}_{\text{CFM}}$. La razón es elemental y merece la pena entenderla porque explica la sección 8: desarrollando el cuadrado, la diferencia entre las dos pérdidas son términos independientes de $\theta$ más un término cruzado, y el cruzado coincide porque el campo marginal es la esperanza condicional del campo condicional:

$$u_t(x) \;=\; \mathbb{E}\big[\, x_1 - x_0 \;\big|\; x_t = x \,\big]$$

Es decir: **la red no aprende $x_1 - x_0$, aprende su esperanza condicional dada la posición**. Es el mismo fenómeno que en cualquier regresión con MSE sobre un objetivo ruidoso: el minimizador es la media condicional, y la pérdida en el óptimo es la varianza condicional residual, que es estrictamente positiva. Esta es la razón por la que la loss de CFM **no converge a cero**, y de ahí sale el criterio cuantitativo de convergencia de la sección 8.

El bucle de entrenamiento completo, tomado de `../docs/material_clase/notebooks/FlowMatching_simple.ipynb` (celda 4):

```python
for epoch in range(epochs):
    x_1 = sample_data(batch_size)              # a) datos
    x_0 = torch.randn_like(x_1)                #    y ruido, independientes
    t = torch.rand(batch_size, 1)              # b) tiempo uniforme, uno por muestra
    x_t = (1 - t) * x_0 + t * x_1              # c) punto del segmento x_0 -> x_1
    u_t = x_1 - x_0                            # d) velocidad objetivo (constante)
    v_pred = model(torch.cat([x_t, t], dim=1)) # e) prediccion de la red
    loss = torch.nn.functional.mse_loss(v_pred, u_t)   # f) regresion estandar

    optimizer.zero_grad(); loss.backward(); optimizer.step()
```

No hay discriminador (a diferencia de las GAN), no hay término KL (a diferencia del VAE), no hay determinante de jacobiano (a diferencia de los flujos clásicos). Es una regresión de mínimos cuadrados, y esa simplicidad es la ventaja práctica principal.

## 4. Caminos de probabilidad: interpolación lineal / óptima

La familia general de caminos condicionales gaussianos es **afín** en $(x_0, x_1)$:

$$x_t = \alpha_t\, x_1 + \sigma_t\, x_0, \qquad u_t = \dot\alpha_t\, x_1 + \dot\sigma_t\, x_0$$

con las condiciones de contorno $\alpha_0 = 0,\ \sigma_0 = 1$ (ruido puro en $t=0$) y $\alpha_1 = 1,\ \sigma_1 \approx 0$ (dato en $t=1$). Distintas elecciones de $(\alpha_t, \sigma_t)$ dan modelos distintos:

| Camino | $\alpha_t$ | $\sigma_t$ | $u_t$ | Trayectoria condicional |
|---|---|---|---|---|
| Lineal / OT (rectified flow) | $t$ | $1-t$ | $x_1 - x_0$ | recta, velocidad constante |
| Difusión VP (equivalente a DDPM) | $\bar\alpha_t^{1/2}$ | $(1-\bar\alpha_t)^{1/2}$ | $\dot\alpha_t x_1 + \dot\sigma_t x_0$ | curva, velocidad variable |
| Coseno (DDIM del notebook 5) | $\cos(\phi_t)$ | $\sin(\phi_t)$ | idem | arco de circunferencia |

**El camino lineal es el que usan los notebooks de clase y el que recomendamos aquí.** Sus propiedades:

- $u_t = x_1 - x_0$ es **constante en $t$** y está acotado: su norma es la distancia entre el punto de ruido y el dato. No explota en ningún extremo del intervalo.
- La varianza del objetivo es aproximadamente uniforme en $t$, así que la pérdida no necesita ninguna ponderación $\lambda(t)$ para equilibrar la contribución de los distintos instantes. En difusión con parametrización de score, el objetivo diverge cuando $\sigma_t \to 0$ y hay que compensarlo con ponderaciones o recortes.
- No hay ningún *schedule* de ruido que ajustar: no existen `beta_min`, `beta_max`, `min_signal_rate`, `max_signal_rate` (compárese con la celda de hiperparámetros de `../docs/material_clase/notebooks/5_Difussion_Models_Keras_DDIM_Jordi_mod.ipynb`, que fija `min_signal_rate = 0.02`, `max_signal_rate = 0.95`). Un hiperparámetro menos que justificar en la defensa.

**Sobre el nombre "transporte óptimo".** Conviene ser preciso, porque el notebook lo llama así en los comentarios. El camino *condicional* sí es la interpolación de desplazamiento óptima entre las dos medidas condicionales; el mapa *marginal* aprendido **no** es en general el transporte óptimo entre $\mathcal{N}(0,I)$ y $p_{\text{datos}}$. La razón es el acoplamiento: emparejamos $x_0$ y $x_1$ **independientemente**, los segmentos se cruzan y el campo promedio se curva.

> Ampliación (no cubierto en clase): OT-CFM (Tong et al. 2024) resuelve una asignación de transporte óptimo *dentro de cada minibatch* (húngaro o Sinkhorn) antes de construir los pares, lo que reduce los cruces y endereza las trayectorias marginales. Coste $O(B^3)$ o $O(B^2)$ por batch, asumible con $B=256$.

**ReFlow / rectified flow.** El notebook dedica las celdas 18–23 a esta idea (Liu et al. 2023): (1) entrenar un primer modelo con pares independientes; (2) generar con él, de modo que cada $x_0$ produzca un $x_1^{\text{gen}}$ determinista —un acoplamiento inducido por el modelo, en el que los segmentos ya no se cruzan—; (3) reentrenar sobre esos pares con exactamente la misma pérdida. El resultado es un campo con curvas integrales mucho más rectas, lo que permite bajar el número de pasos (en el límite, uno solo). El coste es que el segundo modelo aprende la distribución del *primero*, no la de los datos: cualquier sesgo se hereda y puede amplificarse. En este taller ReFlow es opcional; solo tiene sentido si el cuello de botella fuese el tiempo de muestreo, y no lo es (sección 10).

## 5. Muestreo: integración de la ODE y número de pasos

Muestrear es resolver el problema de valor inicial

$$x(0) = x_0 \sim \mathcal{N}(0, I), \qquad \frac{dx}{dt} = v_\theta(x, t), \qquad \text{devolver } x(1)$$

Con **Euler explícito** y $N$ pasos de tamaño $\Delta t = 1/N$ (celda 5 del notebook):

```python
x = torch.randn(n_samples, D)          # partimos de ruido
dt = 1.0 / steps
with torch.no_grad():
    for k in range(steps):
        t = torch.full((n_samples, 1), k * dt)   # tiempo actual
        v = model(torch.cat([x, t], dim=1))      # consultamos el campo
        x = x + v * dt                           # paso de Euler
```

Puntos que importan:

- **Coste = número de evaluaciones de red (NFE)**. Euler con $N$ pasos son $N$ NFE. Heun (RK2) da error global $O(\Delta t^2)$ en vez de $O(\Delta t)$ a cambio de $2N$ NFE; con trayectorias casi rectas, Heun con $N=10$ suele igualar a Euler con $N=50$ a mitad de coste.
- **Cuántos pasos**. El notebook usa 20 y funciona; la diapositiva 23 del material de flows sitúa flow matching en 10–25 pasos frente a 20–100 de difusión. No elegir $N$ a ojo: barrer $N \in \{5,10,20,50,100\}$ y representar una métrica de calidad (p. ej. Wasserstein-1 medio sobre los canales) frente a $N$. Donde la curva se aplana está el $N$ que hay que reportar. Figura barata y muy vendible en la presentación.
- **El muestreo es determinista**. Mismo $x_0$ y mismo modelo dan siempre la misma muestra: la diversidad del conjunto sintético viene **solo** del ruido inicial. A cambio, integrar la ODE hacia atrás ($t: 1 \to 0$) da el latente de un dato real y permite interpolar entre dos ventanas históricas. Es la misma propiedad determinista de DDIM (`Diffusion_Models_DM_2026.pdf`, p. 27) frente a DDPM.
- **Error de discretización vs. error del modelo**. Aumentar $N$ solo reduce el primero. Si pasar de 20 a 100 pasos no mejora las métricas, el techo lo pone el campo aprendido y hay que tocar el modelo, no el solver.

## 6. Relación con difusión: qué comparten y en qué se diferencian

Esta sección es la que sostiene la decisión de entrenar **los dos** modelos en el taller.

### 6.1 Lo que comparten

Ambos aprenden un **transporte de ruido a datos** con el mismo esqueleto: (1) se define una interpolación entre dato y ruido indexada por un tiempo $t$; (2) se muestrea $t$ al azar, se construye $x_t$ y se hace **regresión de mínimos cuadrados** sobre un objetivo que depende de $(x_0, x_1, t)$; (3) la red recibe $x_t$ y un *embedding* del tiempo —en el DDIM del material de clase, `sinusoidal_embedding` del notebook 5; en flow matching, exactamente el mismo tipo—; (4) se genera arrancando de ruido y aplicando la red iterativamente.

La conexión formal es más fuerte de lo que parece. Toda SDE de difusión tiene asociada una **probability flow ODE** determinista con las mismas marginales $p_t$ en todo instante (Song et al. 2021), y **DDIM es precisamente una discretización de esa ODE**. En inferencia, por tanto, DDIM y flow matching hacen estructuralmente lo mismo: integrar una ODE entre ruido y datos.

Y en entrenamiento, con un camino gaussiano afín $x_t = \alpha_t x_1 + \sigma_t x_0$, las cuatro parametrizaciones habituales —predecir el ruido $\varepsilon$ (DDPM), predecir el dato $\hat{x}_1$, predecir el score $\nabla \log p_t$, predecir la velocidad $v$— son **reparametrizaciones afines unas de otras**. Se puede pasar de una a otra en cerrado:

$$v_t = \dot\alpha_t x_1 + \dot\sigma_t x_0, \qquad x_0 = \frac{x_t - \alpha_t x_1}{\sigma_t}, \qquad \nabla \log p_t(x_t) = -\frac{x_0}{\sigma_t}$$

Dicho crudamente: **flow matching con el camino VP *es* difusión, reescrita**. Lo que aporta no es un mecanismo nuevo, sino (a) desacoplar la elección del camino de la del ruido, (b) el camino lineal en particular, y (c) un objetivo de regresión con varianza acotada y casi uniforme en $t$.

> Ampliación (no cubierto en clase): la probability flow ODE y la equivalencia SDE↔ODE son de Song et al. 2021 (*Score-Based Generative Modeling through SDEs*); el PDF de difusión del máster no la trata.

### 6.2 En qué se diferencian

| | Difusión (DDPM / DDIM) | Flow matching (camino lineal) |
|---|---|---|
| Objetivo de regresión | $\varepsilon$ (ruido) o $\hat{x}_1$ | velocidad $x_1 - x_0$ |
| Comportamiento en los extremos | el score diverge cuando $\sigma_t \to 0$; objetivo de varianza alta cerca de $t=0$ y $t=1$ | objetivo acotado y de varianza casi uniforme en $t$ |
| Hiperparámetros de camino | schedule de ruido (lineal, coseno, `min/max_signal_rate`), ponderación $\lambda(t)$ | ninguno |
| Trayectoria de muestreo | curva | casi recta |
| Pasos típicos | 20–100 | 10–25 |
| Muestreo | estocástico (DDPM) o determinista (DDIM) | determinista |
| Verosimilitud | ELBO disponible | requiere integrar la divergencia (caro) |
| Madurez del ecosistema | muy alta (10⁴ papers, schedulers, guidance) | alta pero reciente (2023–) |

### 6.3 Dónde gana cada uno, sin adornos

**Gana difusión cuando:**

- Se quiere **estocasticidad en el muestreo**. El sampler ancestral de DDPM reinyecta ruido en cada paso, lo que corrige errores acumulados y aumenta la diversidad. Con pocos datos reales y riesgo de memorización, esa diversidad extra tiene valor.
- Se necesita una **cota de verosimilitud** barata (el ELBO) para comparar modelos.
- Se quiere **reutilizar herramienta existente**: `../docs/material_clase/notebooks/5_Difussion_Models_Keras_DDIM_Jordi_mod.ipynb` da una implementación de DDIM ya probada, con EMA, embedding sinusoidal y sampler; adaptarla es menos trabajo que escribir desde cero.
- El presupuesto de pasos de muestreo es irrelevante (el caso de este taller: generar unos miles de muestras tabulares cuesta segundos con cualquiera de los dos).

**Gana flow matching cuando:**

- Hay **poco presupuesto de tuning**. Cero hiperparámetros de schedule significa cero horas gastadas en descubrir que `max_signal_rate = 0.95` estaba mal para datos tabulares. El schedule de difusión del notebook está calibrado para imágenes en $[0,1]$; trasladarlo tal cual a un panel financiero estandarizado es una fuente real de fallos silenciosos.
- Se quiere una **curva de loss limpia** para demostrar convergencia, que es literalmente el criterio del enunciado. El objetivo acotado hace que la loss de CFM baje de forma monótona y con poca varianza, mientras que la de difusión, promediada sobre $t$, mezcla regímenes de varianza muy distinta y sale más ruidosa. La diapositiva 23 del material de clase lo dice así: "*Rápido y Estable. El objetivo $x_1-x_0$ está acotado. La loss es plana y baja más rápido.*"
- Importa el **número de evaluaciones** (no aquí, sí en producción), o se quiere **código auditable**: el bucle entero cabe en pantalla y no hay nada que se pueda implementar mal en silencio.

**Lo que no hay que afirmar.** En datos tabulares de dimensión moderada, con el mismo backbone, los mismos datos y suficientes pasos, ambos suelen aterrizar en calidad comparable. Flow matching no es "mejor modelo generativo"; es el mismo transporte con una parametrización más cómoda. Vender lo contrario en la defensa es fácil de rebatir.

### 6.4 Cómo justificar los dos en 5 minutos

El argumento honesto y fuerte es que **no son redundantes, son un experimento controlado**. Fijando datos, arquitectura, presupuesto de entrenamiento y evaluador downstream, la única variable que cambia entre ambos es la **parametrización del transporte**: camino curvo con schedule y muestreo estocástico frente a camino recto sin schedule y muestreo por ODE determinista. Con eso se puede responder empíricamente a "¿importa el schedule de ruido en datos financieros?", que es una pregunta con contenido. Además, la comparación de curvas de loss entre ambos hace visible la diferencia de estabilidad, que es un resultado reportable por sí mismo. Y, dado que el enunciado exige tres modelos generativos distintos más una baseline, cubrir difusión y flow matching con el mismo esfuerzo de ingeniería (comparten preprocesado, embedding de tiempo, condicionamiento y evaluación) es la vía más eficiente.

## 7. Condicionamiento por clase

En nuestro caso el régimen de mercado $y_{\text{reg}} \in \{0,1,2\}$ (normal / estrés / crisis) es la variable de condicionamiento: queremos poder pedir "genérame 500 ventanas de crisis".

**Mecanismo.** El campo pasa a ser $v_\theta(x, t, c)$. La implementación estándar es un `nn.Embedding(n_clases, d_emb)` cuya salida se suma o concatena al estado oculto, igual que el embedding de tiempo. Todo lo demás es idéntico: se muestrea el par $(x_0, x_1)$, se toma la etiqueta $c$ que acompaña a $x_1$ y se hace la misma regresión.

**Desbalanceo.** La clase "crisis" es ~10% de las ventanas. Dos decisiones distintas que conviene no confundir:

- *Durante el entrenamiento*: si el batch respeta la proporción natural, la rama de crisis recibe 10× menos gradiente y su campo queda peor estimado. Un muestreo balanceado por clase (`WeightedRandomSampler`) reparte el gradiente de forma uniforme. Precaución: eso cambia la distribución marginal implícita de $p_1$ que aprende el modelo; como después generamos *condicionando* en la clase, la marginal implícita es irrelevante y el balanceo es un beneficio neto.
- *Durante la generación*: la mezcla de clases del conjunto sintético es una decisión libre y es el objetivo del taller. Se puede generar 33/33/33 aunque los datos reales sean 60/30/10, para dar al modelo downstream ejemplos de la clase minoritaria.

**Guidance sin clasificador.**

> Ampliación (no cubierto en clase): el material de clase menciona modelos condicionales en difusión (`Diffusion_Models_DM_2026.pdf`, páginas 29–30: guiado con clasificador, entrenamiento en pares $(x,y)$, condicionamiento por texto con CLIP), pero no la formulación de classifier-free guidance ni su traslado a flow matching.

La receta es la misma que en difusión. Se entrena con *dropout* de la etiqueta (con probabilidad $p \approx 0.1$ se sustituye $c$ por un token nulo), de modo que la misma red aprende el campo condicional y el incondicional. En muestreo se extrapola:

$$v^{\text{guided}}(x,t,c) = v_\theta(x,t,\varnothing) + w \cdot \big( v_\theta(x,t,c) - v_\theta(x,t,\varnothing) \big)$$

Con $w = 1$ se recupera el condicional puro; $w > 1$ exagera los rasgos de la clase. **Advertencia para este taller**: guidance con $w > 1$ produce muestras más "prototípicas", con varianza reducida y colas recortadas. En imágenes eso se percibe como calidad; para *aumentar un conjunto de entrenamiento* es contraproducente, porque el modelo downstream aprende una versión caricaturizada de la crisis y generaliza peor. Empezar con $w=1$ y probar $w \in \{1.5, 2\}$ solo como ablación, midiendo el efecto en la métrica downstream y no a ojo. Además, guidance duplica el NFE.

**Alternativa**: generar la etiqueta como parte del bloque conjunto (one-hot suave dentro de $z$) en lugar de condicionar. Es más simple, pero pierde el control explícito sobre la mezcla de clases, que es justo lo que necesitamos. Recomendación: condicionar.

## 8. Diagnóstico de convergencia

El enunciado exige "*curvas de loss donde se vea que el modelo ha convergido*". Flow matching tiene la loss más limpia de los cuatro modelos del taller y hay que explotarlo, pero con un matiz que casi nadie reporta y que da mucho valor a la defensa.

### 8.1 La loss de CFM no converge a cero, y su valor límite se puede calcular

Como se vio en la sección 3, el minimizador de la pérdida es $v^*(x,t) = \mathbb{E}[x_1 - x_0 \mid x_t = x]$, luego el valor de la pérdida en el óptimo es

$$\mathcal{L}(\theta^*) = \mathbb{E}_{t, x_t}\Big[ \operatorname{Var}\big( x_1 - x_0 \mid x_t \big) \Big] \; > \; 0$$

Este **piso irreducible** es aleatoriedad genuina del emparejamiento: dado un punto intermedio $x_t$, hay muchos pares $(x_0, x_1)$ compatibles y la red no puede distinguirlos. Una loss que se estanca no es un fallo de optimización; es el modelo tocando su suelo.

Esto permite construir **dos líneas de referencia** que convierten la curva de loss en algo interpretable en lugar de un número sin escala. Con los datos estandarizados por canal (varianza 1 por dimensión) y la loss definida como MSE promediada por dimensión:

**Referencia 1 — predictor trivial $v \equiv 0$:**

$$\mathcal{L}_0 = \tfrac{1}{D}\,\mathbb{E}\|x_1 - x_0\|^2 = \operatorname{Var}(x_1) + \operatorname{Var}(x_0) = 1 + 1 = 2.0$$

Si la loss se queda cerca de 2.0, el modelo no ha aprendido nada.

**Referencia 2 — piso gaussiano.** Si los datos fueran ruido gaussiano estándar independiente (ninguna estructura que aprender), el piso se calcula analíticamente. Con $x_1, x_0 \sim \mathcal{N}(0,1)$ independientes, $\operatorname{Cov}(u, x_t) = 2t-1$, $\operatorname{Var}(x_t) = (1-t)^2 + t^2$ y $\operatorname{Var}(u) = 2$, de donde

$$\operatorname{Var}(u \mid x_t) = 2 - \frac{(2t-1)^2}{(1-t)^2 + t^2} = \frac{1}{2t^2 - 2t + 1}, \qquad \int_0^1 \frac{dt}{2t^2-2t+1} = \frac{\pi}{2} \approx 1.5708$$

Es decir: **1.571 es el valor al que llegaría una red perfecta entrenada sobre datos sin ninguna estructura.** Comprobado por Monte Carlo con el regresor óptimo analítico: 1.569 sobre 4·10⁵ muestras.

Para datos con correlaciones (nuestro panel lo está, y mucho), el piso es **menor** y se calcula a partir de los autovalores $\lambda_i$ de la covarianza empírica de entrenamiento:

$$\mathcal{L}_{\text{gauss}} = \int_0^1 \frac{1}{D}\sum_{i=1}^{D} \left[ (\lambda_i + 1) - \frac{(t\lambda_i - (1-t))^2}{(1-t)^2 + t^2 \lambda_i} \right] dt$$

```python
import numpy as np
from scipy.integrate import quad

def piso_gaussiano(X_train):
    """Piso de la loss CFM alcanzable por un modelo puramente gaussiano.
    X_train: (N, D) ya estandarizado. Traza de la covarianza ~ D."""
    lam = np.linalg.eigvalsh(np.cov(X_train, rowvar=False))
    lam = np.clip(lam, 0.0, None)
    f = lambda t: np.mean((lam + 1) - (t*lam - (1-t))**2 / ((1-t)**2 + t**2*lam + 1e-12))
    return quad(f, 0, 1, limit=200)[0]
```

Con espectros tipo ley de potencias $\lambda_i \propto i^{-\alpha}$ y traza normalizada a $D$, este piso vale 1.117 ($\alpha=1$), 0.583 ($\alpha=1.5$) y 0.282 ($\alpha=2$): cuanto más correlacionados los datos, más predecible es la velocidad y más baja la loss. Un panel financiero de 18 canales sobre 60 días tiene un espectro muy desigual, así que hay que esperar un piso claramente por debajo de 1.

**Lectura de la curva, entonces:**

| Valor de la loss | Interpretación |
|---|---|
| $\approx 2.0$ | el modelo no aprende; revisar escalado, lr, o que $t$ entre de verdad en la red |
| $\approx 1.57$ | ha aprendido la forma isótropa pero no la estructura de los datos |
| entre $\mathcal{L}_{\text{gauss}}$ y 1.57 | aprende correlaciones; margen de mejora |
| $\approx \mathcal{L}_{\text{gauss}}$ | ha capturado toda la estructura de segundo orden |
| $< \mathcal{L}_{\text{gauss}}$ | captura estructura no gaussiana (colas, asimetría de régimen). Es lo que buscamos |

**Trazar la curva de loss junto a estas dos líneas horizontales es una figura de una sola línea de código extra y comunica en tres segundos que el modelo ha convergido *y* a qué.** Es, con diferencia, la mejor manera de cumplir el requisito del enunciado.

### 8.2 Cómo conseguir una curva que se lea

La loss por batch tiene varianza alta porque en cada paso se resortean $t$ y $x_0$. Dos medidas, ambas baratas:

1. **Media móvil exponencial** de la loss de entrenamiento (factor 0.98) para la curva de train.
2. **Loss de validación con ruido común (*common random numbers*)**: fijar de antemano una rejilla determinista de pares $(x_0, t)$ —por ejemplo $t$ estratificado en 20 niveles— y evaluar siempre sobre esa misma rejilla y el mismo split de validación. Al eliminar el ruido de muestreo, la curva de validación sale prácticamente monótona y suave, y la convergencia se ve sin ambigüedad. Esto es lo que hay que reportar.

**Descomposición por tramos de $t$**: agrupar $t$ en 10 bins y graficar la loss por bin. Diagnostica dónde falla el modelo: error alto cerca de $t \to 1$ significa que no reproduce el detalle fino del dato; error alto cerca de $t \to 0$ significa que no organiza bien la salida del ruido. Con el camino lineal la curva por bins debe ser bastante plana; si tiene una U pronunciada, hay un problema de capacidad o de escalado.

**Criterio operativo de parada**: la loss de validación con ruido común mejora menos de un 0.5% relativo en 50 épocas consecutivas.

### 8.3 Por qué una loss baja no garantiza buenas muestras

Este es el punto que hay que decir explícitamente en la memoria, porque es donde se cae la mayoría de los trabajos.

1. **La loss está dominada por el término irreducible.** Si el piso es 0.8 y el modelo está en 0.85, la parte *aprendible* del error es 0.05 sobre un total de 0.85: una mejora del 40% en la parte que importa mueve el número total un 2%. Diferencias absolutas minúsculas en la curva corresponden a diferencias grandes en calidad. Corolario: **no se pueden comparar dos modelos por su loss absoluta si no comparten el piso**, y nunca se puede comparar la loss de flow matching con la de difusión.
2. **El error se integra.** La calidad de la muestra depende del error *acumulado* a lo largo de toda la trayectoria, no del error puntual promedio. Un campo con error pequeño pero sistemáticamente sesgado en una dirección desvía la trayectoria de forma acumulativa. La MSE no ve esa diferencia.
3. **La media oculta a la minoría.** El promedio sobre todas las muestras y dimensiones diluye por completo un fallo en la clase crisis (10% de los datos): el modelo puede estar generando basura para ese régimen sin que la curva se inmute. **Hay que graficar la loss de validación desagregada por clase.**
4. **La regresión con MSE es sobre-suavizadora.** El óptimo es una media condicional; un modelo con capacidad insuficiente tiende a producir muestras con las marginales aproximadamente correctas pero con las colas comprimidas y la estructura de dependencia aplanada. En datos financieros eso significa exactamente perder lo que nos interesa: curtosis, clustering de volatilidad y correlaciones que se disparan en crisis.

**Por tanto, la evidencia de convergencia debe ir siempre acompañada de comprobaciones distribucionales**, que en nuestro caso son:

- Marginales por canal: histogramas superpuestos real/sintético y estadístico de Kolmogórov-Smirnov o Wasserstein-1 por canal.
- Estructura temporal: autocorrelación de los retornos (debe ser ~0) y del valor absoluto de los retornos (debe decaer lentamente: *volatility clustering*).
- Estructura transversal: matriz de correlaciones entre los 9 sectores y el índice; reportar la distancia de Frobenius entre la matriz real y la sintética.
- Coherencia condicional: verificar que las ventanas generadas con $c=\text{crisis}$ tienen efectivamente volatilidad realizada alta, drawdown profundo y VIX elevado. Es la prueba de que el condicionamiento funciona.
- **Memorización**: histograma de la distancia al vecino más próximo del conjunto de entrenamiento, comparando muestras sintéticas contra muestras reales *held-out*. Si las sintéticas están sistemáticamente más cerca del train que las reales retenidas, el modelo está copiando y el aumento de datos no aportará nada.
- Métrica final: rendimiento del modelo downstream (macro-F1 y recall de la clase crisis) frente al porcentaje de datos sintéticos. Es la única métrica que responde a la pregunta del taller.

## 9. Aplicación a nuestro problema

**Qué se genera.** El bloque conjunto por ventana es

$$z = \big[\, \operatorname{vec}(X) \;;\; y_{\text{vol}} \,\big] \in \mathbb{R}^{1081}, \qquad X \in \mathbb{R}^{60 \times 18}$$

con $\operatorname{vec}(X)$ la ventana de 60 días × ~18 canales del panel híbrido (S&P 500, 9 SPDR sectoriales, VIX, MOVE, spreads de crédito, pendiente de curva, drawdown, volatilidad realizada) aplanada a 1080, más el objetivo continuo $y_{\text{vol}}$. El régimen $y_{\text{reg}} \in \{0,1,2\}$ **no se genera**: se usa como condición $c$ del campo de velocidad (sección 7), lo que permite fijar la mezcla de clases del conjunto sintético.

Generar el bloque conjunto (y no $X$ sola) es deliberado: el modelo downstream necesita pares $(X, y)$ coherentes, y la dependencia entre la ventana y su etiqueta es justo lo que un generador marginal destruiría.

**Preprocesado — la parte que más fallos causa.**

- **Estandarizar por canal** con media y desviación calculadas **solo sobre el split de entrenamiento**. El objetivo $x_1 - x_0$ mezcla el dato con ruido $\mathcal{N}(0,I)$; si un canal tiene escala 10 y otro 0.01, la loss la domina el primero y el segundo se ignora por completo. Además, así las referencias 2.0 y $\pi/2$ de la sección 8 son directamente aplicables. Los canales de colas muy pesadas (VIX, spreads) se benefician de una transformación previa (log o rangos gaussianos).
- **Split temporal, nunca aleatorio.** Las ventanas de 60 días se solapan; un split aleatorio pone ventanas casi idénticas a ambos lados y el modelo parece generalizar cuando está memorizando. Corte cronológico con un hueco de al menos 60+21 días entre train y test.

> Ampliación (no cubierto en clase): la transformación a rangos gaussianos (*rank-gauss*) es la misma idea que la gaussianización iterativa / RBIG que aparece en `../docs/material_clase/slides/Normalizing Flows_2026.pdf`, diapositiva 19, aplicada canal a canal como preproceso.

**Por qué flow matching encaja aquí.** La dimensión nominal es 1081 pero la efectiva es mucho menor: 18 canales fuertemente correlacionados y muy autocorrelacionados en el tiempo. El campo de velocidades resultante es suave y un MLP lo aproxima sin dificultad. El objetivo acotado hace el entrenamiento insensible al escalado residual, y no hay ningún schedule calibrado para imágenes que trasladar mal a datos tabulares.

**Riesgos que hay que reconocer en la memoria.**

- **Pocos datos.** Con ~15–25 años de historia diaria salen 4.000–6.000 ventanas, pero muy solapadas: los episodios de mercado *independientes* son decenas, no miles. Para 1081 dimensiones el riesgo de memorización es real, y por eso el test de vecino más próximo de la sección 8 es obligatorio.
- **La clase crisis es el cuello de botella.** Un 10% suena a 400–600 ejemplos, pero corresponden a un puñado de episodios (2008, 2011, 2020, 2022). Ningún modelo generativo inventa un tipo de crisis que no ha visto; como mucho interpola dentro de las vistas. La conclusión honesta será probablemente que el sintético regulariza y equilibra clases, no que crea información nueva.
- **El MLP ignora la estructura temporal.** Aplanar 60×18 trata cada (día, canal) como una dimensión aislada. Funciona, pero desperdicia estructura. Alternativa si sobra tiempo: sustituir el MLP por un stack de `Conv1d` sobre el eje temporal dejando idéntico el resto del código, porque el objetivo CFM no cambia en absoluto.

**Post-proceso.** Desestandarizar con las estadísticas de train, recortar $y_{\text{vol}}$ al rango plausible (la volatilidad es positiva) y aplicar los recortes de los canales acotados por construcción (drawdown $\le 0$). Reportar qué fracción de muestras necesita recorte: si es alta, el modelo no ha aprendido bien los soportes.

**Protocolo de evaluación.** Entrenar el generador solo con train; generar $M$ muestras con la mezcla de clases elegida; entrenar el modelo downstream con proporciones real:sintético de 100:0, 75:25, 50:50, 25:75, 0:100 y en modo aumento (100% real + $k$% sintético extra); evaluar todo sobre el mismo test temporal retenido, con la misma arquitectura y las mismas semillas. Reportar macro-F1 y recall de crisis.

## 10. Implementación de referencia (CPU)

Entorno objetivo: torch 2.11.0+cpu, Python 3.13.7, sin CUDA. Los tiempos de abajo están **medidos** en la máquina del proyecto (4 núcleos, 2 hilos de torch por defecto).

```python
import math, time
import torch
import torch.nn as nn

torch.set_num_threads(4)   # por defecto usa 2; con 4 nucleos merece la pena

D = 1081        # 60 dias x 18 canales aplanado (1080) + y_vol (1)
N_CLASES = 3    # regimen: normal / estres / crisis (+1 token nulo para guidance)


class EmbeddingTiempo(nn.Module):
    """Features de Fourier del tiempo t + MLP. Mismo esquema que el
    sinusoidal_embedding del notebook de DDIM."""
    def __init__(self, dim=128):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim))

    def forward(self, t):                       # t: (B, 1) en [0, 1]
        mitad = self.dim // 2
        frec = torch.exp(torch.linspace(math.log(1.0), math.log(1000.0), mitad))
        ang = 2 * math.pi * t * frec[None, :]
        return self.mlp(torch.cat([torch.sin(ang), torch.cos(ang)], dim=1))


class CampoVelocidad(nn.Module):
    """v_theta(x, t, c). MLP simple: la estructura del problema es suave."""
    def __init__(self, d=D, h=512, d_t=128, d_c=64, n_cls=N_CLASES):
        super().__init__()
        self.emb_t = EmbeddingTiempo(d_t)
        self.emb_c = nn.Embedding(n_cls + 1, d_c)   # indice n_cls = token nulo
        self.net = nn.Sequential(
            nn.Linear(d + d_t + d_c, h), nn.SiLU(),
            nn.Linear(h, h), nn.SiLU(),
            nn.Linear(h, h), nn.SiLU(),
            nn.Linear(h, d),
        )

    def forward(self, x, t, c):
        return self.net(torch.cat([x, self.emb_t(t), self.emb_c(c)], dim=1))


# ---------------------------------------------------------------- entrenamiento
def entrenar(X, C, epochs=500, batch_size=256, lr=2e-4, p_drop_clase=0.1):
    """X: (N, D) estandarizado con estadisticas de train. C: (N,) etiquetas."""
    modelo = CampoVelocidad()
    opt = torch.optim.AdamW(modelo.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    ema = {k: v.detach().clone() for k, v in modelo.state_dict().items()}
    historial, n = [], X.shape[0]

    for _ in range(epochs):
        perm = torch.randperm(n)
        acum, nb = 0.0, 0
        for i in range(0, n - batch_size + 1, batch_size):
            x_1, c = X[perm[i:i+batch_size]], C[perm[i:i+batch_size]].clone()
            c[torch.rand(c.shape[0]) < p_drop_clase] = N_CLASES  # dropout de clase

            x_0 = torch.randn_like(x_1)              # ruido base
            t = torch.rand(x_1.shape[0], 1)          # tiempo uniforme
            x_t = (1 - t) * x_0 + t * x_1            # camino lineal (OT)
            u_t = x_1 - x_0                          # velocidad objetivo

            loss = ((modelo(x_t, t, c) - u_t) ** 2).mean()
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(modelo.parameters(), 1.0)
            opt.step()
            with torch.no_grad():                    # EMA de pesos (0.999)
                for k, v in modelo.state_dict().items():
                    ema[k].mul_(0.999).add_(v, alpha=0.001)
            acum += loss.item(); nb += 1
        sched.step()
        historial.append(acum / nb)
    return modelo, ema, historial


@torch.no_grad()
def loss_validacion(modelo, X_val, C_val, n_niveles=20, semilla=0):
    """Common random numbers: misma rejilla (x_0, t) en todas las epocas.
    Convierte una curva ruidosa en una curva monotona y legible."""
    g = torch.Generator().manual_seed(semilla)
    total = 0.0
    for k in range(n_niveles):
        t = torch.full((X_val.shape[0], 1), (k + 0.5) / n_niveles)
        x_0 = torch.randn(X_val.shape, generator=g)
        x_t = (1 - t) * x_0 + t * X_val
        total += ((modelo(x_t, t, C_val) - (X_val - x_0)) ** 2).mean().item()
    return total / n_niveles


@torch.no_grad()
def muestrear(modelo, n, clase, pasos=20, metodo="euler", w=1.0):
    x = torch.randn(n, D)
    c = torch.full((n,), clase, dtype=torch.long)
    c_nulo = torch.full((n,), N_CLASES, dtype=torch.long)
    dt = 1.0 / pasos

    def v(x, t_val):                                 # guidance opcional: 2x NFE
        t = torch.full((n, 1), t_val)
        if w == 1.0:
            return modelo(x, t, c)
        v_c, v_0 = modelo(x, t, c), modelo(x, t, c_nulo)
        return v_0 + w * (v_c - v_0)

    for k in range(pasos):
        t0 = k * dt
        if metodo == "euler":                        # 1 evaluacion por paso
            x = x + v(x, t0) * dt
        else:                                        # Heun (RK2): 2 por paso
            v1 = v(x, t0)
            v2 = v(x + v1 * dt, min(t0 + dt, 1.0))
            x = x + 0.5 * (v1 + v2) * dt
    return x
```

**Números concretos medidos** (torch 2.11.0+cpu, 4 núcleos, $D = 1081$, $B = 256$):

| Configuración | Parámetros | fp32 | ms/paso | s/época (N=3000) | 500 épocas |
|---|---|---|---|---|---|
| `h=512` (recomendada) | 1.765.433 (1,77 M) | 7,1 MB | 133 | 1,5 | **~12 min** |
| `h=1024` | 4.545.081 (4,55 M) | 18,2 MB | 226 | 2,5 | ~21 min |

Con N=6000 ventanas: 3,1 s/época con `h=512` → ~25 min para 500 épocas.

Coste de muestreo (generar el conjunto sintético completo, una sola vez):

| Configuración | 2000 muestras × 20 pasos | 2000 muestras × 50 pasos |
|---|---|---|
| `h=512` | 4,2 s | 10,9 s |
| `h=1024` | 8,3 s | 22,1 s |

**Recomendación de hiperparámetros**: `h=512` (1,77 M parámetros), `batch_size=256`, AdamW con `lr=2e-4` y decaimiento coseno, `weight_decay=1e-4`, recorte de gradiente a norma 1.0, EMA 0.999, 400–600 épocas, `p_drop_clase=0.1`, muestreo con Euler y 20 pasos (verificando con el barrido de $N$ de la sección 5 que 20 basta). Presupuesto total: **~15 minutos de entrenamiento y ~5 segundos de generación en CPU**. Esto deja margen sobrado para repetir con 3 semillas y reportar dispersión, que es más convincente que una sola curva.

**Detalle a no pasar por alto.** `../docs/material_clase/notebooks/FlowMatching_imagenes.ipynb` contiene un fallo instructivo: el bucle de entrenamiento toma `x_1 = train_dataset.data[ii,:,:]`, que es el tensor crudo en $[0, 255]$, saltándose el `transforms.Normalize` definido en la celda 2. El ruido $x_0$ sigue siendo $\mathcal{N}(0,1)$, así que el objetivo $x_1 - x_0$ vive en escala de centenas y está dominado por el dato. El modelo aún produce dígitos reconocibles, pero es el ejemplo perfecto de por qué la estandarización de la sección 9 no es opcional en datos con escalas heterogéneas: en un panel financiero el mismo error no produce un resultado visiblemente feo, sino un modelo que ignora silenciosamente los canales de escala pequeña.

## 11. Referencias

**Material de clase**

- `../docs/material_clase/notebooks/FlowMatching_simple.ipynb` — implementación completa en 2D: entrenamiento simulation-free (celda 4), muestreo por Euler con 20 pasos (celda 5), desglose visual del objetivo (celdas 9–14), visualización de trayectorias con 200 pasos (celda 16) y ReFlow (celdas 18–23). Fuente principal de este documento.
- `../docs/material_clase/notebooks/FlowMatching_imagenes.ipynb` — el mismo objetivo con una CNN de 3 capas sobre MNIST; el tiempo entra como canal extra replicado espacialmente.
- `../docs/material_clase/slides/Normalizing Flows_2026.pdf` — diapositivas 5–18: cambio de variable, determinante del jacobiano y taxonomía de flujos. Diapositiva 20: flow matching. Diapositiva 22: ReFlow. Diapositiva 23: tabla comparativa flujo clásico / difusión / OT-CFM.
- `../docs/material_clase/slides/Diffusion_Models_DM_2026.pdf` — DDPM (p. 23), parametrización del schedule (p. 26), DDIM (p. 27), modelos condicionales (pp. 29–30). **No cubre flow matching**: solo lo cita en la línea temporal de la p. 33.
- `../docs/material_clase/notebooks/5_Difussion_Models_Keras_DDIM_Jordi_mod.ipynb` — DDIM en Keras con schedule de coseno y EMA; referencia para la comparación de la sección 6.
- `../enunciado/Taller_B5_T1.pdf` — requisito de curvas de loss con convergencia demostrada.

**Literatura**

> Ampliación (no cubierto en clase): las referencias siguientes sostienen las secciones 2–4, 6 y 8, que van más allá del material del máster.

- Lipman, Chen, Ben-Hamu, Nickel, Le (2023). *Flow Matching for Generative Modeling*. ICLR. arXiv:2210.02747. — Objetivo CFM, teorema de igualdad de gradientes, caminos condicionales gaussianos. Citado en la diapositiva 20 del material de clase.
- Liu, Gong, Liu (2023). *Flow Straight and Fast: Rectified Flow*. ICLR. arXiv:2209.03003. — Camino lineal y procedimiento ReFlow.
- Albergo, Vanden-Eijnden (2023). *Building Normalizing Flows with Stochastic Interpolants*. ICLR. arXiv:2209.15571. — Formulación equivalente e independiente.
- Tong et al. (2024). *Improving and Generalizing Flow-Based Generative Models with Minibatch Optimal Transport*. TMLR. arXiv:2302.00482. — OT-CFM.
- Chen, Rubanova, Bettencourt, Duvenaud (2018). *Neural Ordinary Differential Equations*. NeurIPS. arXiv:1806.07366. — CNF y método adjunto.
- Song et al. (2021). *Score-Based Generative Modeling through SDEs*. ICLR. arXiv:2011.13456. — Probability flow ODE; base formal de la sección 6.1.
- Song, Meng, Ermon (2021). *Denoising Diffusion Implicit Models*. ICLR. arXiv:2010.02502. — DDIM como integrador de la ODE determinista. | Ho, Jain, Abbeel (2020). *DDPM*. NeurIPS. arXiv:2006.11239.
- Ho, Salimans (2022). *Classifier-Free Diffusion Guidance*. arXiv:2207.12598. — Sección 7.
- Esser et al. (2024). *Scaling Rectified Flow Transformers for High-Resolution Image Synthesis*. ICML. arXiv:2403.03206. — Flow matching a escala (Stable Diffusion 3); muestreo de $t$ con densidad logit-normal.
- Fjelde, Mathieu, Dutordoir (2024). *An Introduction to Flow Matching*. Cambridge MLG blog. — Enlazado en la diapositiva 20 del material de clase.
