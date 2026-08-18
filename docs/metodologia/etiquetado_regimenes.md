# Etiquetado de regímenes de mercado

Documento metodológico del taller B5-T1. Fija cómo se construye `y_reg`, la variable objetivo de
clasificación del modelo *downstream*, y por qué se construye así. Es la referencia normativa del
notebook `01_etiquetado_regimenes.ipynb` y del módulo `src/regimenes.py`.

El taller hereda el marco causal y la canonicalización económica del TFM de detección de regímenes
(`PRUEBAS_DETECCION_REGIMENES_DE_MERCADO_TFM/`), pero **cambia el problema**: aquel detecta el
régimen de hoy; este predice el que dominará los próximos 21 días. La sección 7 desarrolla esa
diferencia, que es el núcleo del documento.

Notación: $r_t$ retorno logarítmico diario del S&P 500; $\mathbf{o}_t \in \mathbb{R}^{d}$ vector de
*features de etiquetado* ($d=5$, §8); $s_t \in \{0,\dots,K-1\}$ estado latente con $0$ = calma y
$K-1$ = crisis; $\mathcal{F}_t = \sigma(\{\mathbf{o}_u\}_{u \le t})$; $h = 21$ horizonte de agregación
en días de mercado; $X_t \in \mathbb{R}^{60 \times 20}$ ventana de entrada
(`data/catalog.yaml:73-104`); $y_{\text{reg}}(t)$ etiqueta de la ventana que acaba en $t$.

---

## 1. Qué es un régimen y por qué no hay etiqueta verdadera

Un **régimen** es un estado no observable que gobierna la distribución conjunta de los retornos:
dentro de un régimen los parámetros (media, volatilidad, correlaciones, colas) son aproximadamente
estables, y entre regímenes cambian de forma discreta. Se modela como variable latente $s_t$ con
dinámica propia, de la que solo se observan realizaciones $\mathbf{o}_t$ cuya densidad depende
de $s_t$.

La consecuencia es incómoda: **$s_t$ no se observa nunca**, ni siquiera *a posteriori*. No hay
registro oficial de "el mercado estuvo en crisis del día A al día B", solo convenciones parciales y no
equivalentes: las recesiones NBER miden el ciclo real, son trimestrales y se publican con meses de
retraso —y el mercado no cae cuando cae el PIB—; las reglas *bull/bear* (caída ≥ 20 %) usan un umbral
arbitrario y exigen conocer el suelo, que solo se sabe después; las ventanas de crisis dibujadas a
mano dependen de quien las dibuja; y los índices de estrés (OFR FSI, NFCI) son series construidas, con
metodología propia y revisiones.

El TFM declara estas convenciones material de **evaluación**, nunca de entrenamiento:
`CRISIS_WINDOWS` y `FALSE_POSITIVE_WINDOWS` (`src/evaluation.py:32-44`) miden cobertura y falsas
alarmas, no ajustan nada. Su glosario eleva la separación a regla: una serie de `rol=validation`
"**nunca** puede entrar a la vez como feature y como etiqueta" (`docs/GLOSARIO.md:45-50`).

De ahí las dos consecuencias que estructuran el resto: **la etiqueta es una decisión de modelado, no
un dato** —cambiar método, número de estados o ventana de agregación cambia `y_reg` y con ella todo el
problema *downstream*, por lo que el protocolo se congela por escrito (§8) en
`data/catalog.yaml:142-155`—, y **la etiqueta se valida, no se cree**: se acepta si supera controles
externos (coincide con las crisis catalogadas, produce estados económicamente distinguibles, no
parpadea), no porque el EM converja.

---

## 2. Familias de métodos

La columna de evidencia recoge números reales del banco de 12 detectores del TFM
(`capa1_exploracion/memory/99_conclusions.md`), todos evaluados con el mismo protocolo *walk-forward*
causal.

| Familia | Idea | Persistencia | Evidencia en el TFM |
|---|---|---|---|
| **Reglas / umbrales** | `crisis := VIX > c` o combinación de umbrales | ninguna (se impone con histéresis) | `rule_vix_threshold`: cobertura GFC+COVID 0.92, duración media 75 d, coste bajo |
| **Fechado de ciclos** | picos y valles sobre el precio (Pagan & Sossounov, 2003) | por construcción | no implementado: exige conocer el suelo *ex post* |
| **Clustering** | k-means / GMM sobre features, cada día independiente | **ninguna** → parpadeo | `clustering_gmm_k3`: `switching_rate` 0.126, duración 7.9 d (`memory/detectors/03_clustering_gmm.md:73-79`) |
| **HMM** | estado latente markoviano + emisiones $p(\mathbf{o}_t \mid s_t)$ | explícita, en la diagonal de $A$ | `hmm_gaussian_2s`: `switching_rate` 0.100, duración 9.9 d (`memory/detectors/04_hmm_gaussian_2s.md:96-102`) |
| **Markov-switching** | los parámetros de una regresión/VAR conmutan (Hamilton, 1989) | explícita | `markov_switching_var_2s`: mejor cobertura sistémica del banco (0.98), pero ~33 min de ajuste |
| **Change-point** | detecta el instante del cambio de nivel/varianza (CUSUM) | máxima | `changepoint_online`: `switching_rate` 0.002, duración 436 d, especificidad 1.00 |
| **Jump models** | clustering + penalización $\lambda$ por salto de estado (Nystrup et al.) | ajustable vía $\lambda$ | `jump_model` ($\lambda=50$): `switching_rate` 0.005, duración 176.6 d, pero cobertura Inflación 2022 solo 0.17 (`memory/detectors/09_jump_model.md:36-46`) |

**La persistencia es un mando, no una virtud.** El jump model reduce el parpadeo 24× frente al GMM
(0.126 → 0.005) y multiplica por 22 la duración de los episodios, pero pierde el mercado bajista lento
de 2022 (cobertura 0.17 frente a 0.87 del GMM). No hay comida gratis entre estabilidad y sensibilidad.

**El HMM es el punto medio defendible**, y la elección de este taller (§8): aporta persistencia sin
imponerla a mano (la aprende en $A$), da posteriores en vez de etiquetas duras, tiene verosimilitud
explícita —y por tanto BIC para elegir $K$— y su coste es medio. Su debilidad conocida es el supuesto
gaussiano: con curtosis en exceso de 25–40, un HMM t-Student mejora el ajuste con holgura
($\Delta\text{BIC} \approx 10963$ sobre idénticas features, `memory/99_conclusions.md:217-227`). Se
acepta esa mala especificación a cambio de simplicidad y reproducibilidad, y se declara (§8).

---

## 3. HMM gaussiano: formulación, ajuste y decodificación

**Modelo.** Cadena de Markov homogénea de $K$ estados con matriz de transición $A = (a_{ij})$,
$a_{ij} = P(s_t = j \mid s_{t-1} = i)$, distribución inicial $\pi$ y emisiones gaussianas
condicionadas al estado, $\mathbf{o}_t \mid s_t = i \sim \mathcal{N}(\boldsymbol{\mu}_i, \Sigma_i)$.
La verosimilitud marginaliza sobre todos los caminos de estados,

$$\mathcal{L}(\theta) = \sum_{s_{1:T}} \pi_{s_1} \prod_{t=2}^{T} a_{s_{t-1} s_t} \prod_{t=1}^{T} \mathcal{N}(\mathbf{o}_t \mid \boldsymbol{\mu}_{s_t}, \Sigma_{s_t})$$

y se evalúa en $O(TK^2)$ con el algoritmo *forward*.

**Ajuste.** Baum-Welch (EM): el paso E calcula posteriores suavizados
$\gamma_t(i) = P(s_t = i \mid \mathbf{o}_{1:T})$ y transiciones esperadas $\xi_t(i,j)$ con
*forward-backward*; el paso M actualiza $\pi, A, \boldsymbol{\mu}_i, \Sigma_i$ en forma cerrada. La
verosimilitud es multimodal: el resultado depende de la inicialización, así que el protocolo estándar
es lanzar varias semillas y quedarse con la de mayor $\log\mathcal{L}$. La tarea previa del TFM usó 10
semillas (42–51), todas convergieron al mismo óptimo, $\log\mathcal{L} = 3822{,}73$
(`docs/context/RESUMEN_DETECCION_REGIMENES.md:114-115`).

**Decodificación.** Tres objetos que se confunden con facilidad:

| Objeto | Definición | Usa el futuro | Uso legítimo |
|---|---|---|---|
| Filtrado | $P(s_t = i \mid \mathbf{o}_{1:t})$ | no | detección en tiempo real, features |
| Suavizado | $P(s_t = i \mid \mathbf{o}_{1:T})$ | sí | **etiquetado retrospectivo** |
| Viterbi | $\arg\max_{s_{1:T}} P(s_{1:T} \mid \mathbf{o}_{1:T})$ | sí | etiquetado retrospectivo, camino coherente |

`hmmlearn` devuelve suavizado en `predict_proba` y Viterbi en `predict`
([hmmlearn 0.3.x](https://hmmlearn.readthedocs.io/en/latest/api.html)): ambos miran el futuro. El TFM
tuvo que sustituirlos por filtrado *forward* explícito para su evaluación causal
(`capa1_exploracion/detectors/hmm_gaussian_2s.py:209-230`); este taller **no** lo necesita (§7.3).

**Persistencia y duración esperada.** El tiempo de permanencia en el estado $i$ es geométrico, con
$\mathbb{E}[\text{duración}_i] = 1/(1 - a_{ii})$, y la distribución de largo plazo es el autovector
izquierdo de $A$ con autovalor 1. El HMM de 2 estados de la tarea previa da $a_{00} = 0{,}9794$ y
$a_{11} = 0{,}9395$ (duraciones esperadas 48,5 y 16,5 días) y estacionaria 74,6 % / 25,4 %
(`docs/context/RESUMEN_DETECCION_REGIMENES.md:107-111`). Diagonales por encima de 0,93 son el patrón
normal en regímenes diarios: entrar en crisis es raro (2,1 % diario) y salir, lento (6,1 %).

---

## 4. Elección del número de estados

Con $\text{AIC} = 2k - 2\log\mathcal{L}$ y $\text{BIC} = k \log n - 2\log\mathcal{L}$, y para un HMM
gaussiano de $K$ estados y $d$ features con covarianza `full`, el conteo de parámetros que usa el TFM
(`capa1_exploracion/detectors/hmm_gaussian_2s.py:239-247`) es

$$k = \underbrace{K^2 - 1}_{\text{transición}} + \underbrace{K d}_{\text{medias}} + \underbrace{K\,d(d+1)/2}_{\text{covarianzas}}$$

Con $d=5$ (§8): $k = 43$ para $K=2$ y $k = 68$ para $K=3$. El salto es asumible porque $d$ es pequeño;
ese es el motivo de etiquetar con un subconjunto reducido de features y no con los 20 canales de $X$
(con $d=20$ se pasaría de 462 a 693 parámetros, insostenible con ~4.000 observaciones de train).

**Evidencia.** El TFM seleccionó $K$ por BIC sobre features de la misma naturaleza: $70\,975$ para
$K=2$ frente a $63\,016$ para $K=3$, gana $K=3$ con holgura
(`capa1_exploracion/memory/detectors/03_clustering_gmm.md:43-45`). La literatura aplicada reporta el
mismo signo, y añade que el modelo binario mezcla la volatilidad moderada dentro del régimen de calma,
enmascarando dinámica ([Gaussian HMM regime analysis, 2025](https://www.researchgate.net/publication/398722347_Machine_Learning-Driven_Market_Regime_Analysis_in_Equity_Markets_A_Gaussian_Hidden_Markov_Model_Approach)).
En el extremo, Guidolin & Timmermann (2007) necesitan **cuatro** regímenes —*crash*, crecimiento
lento, *bull*, recuperación— para capturar la conjunta de acciones y bonos.

**Por qué 3 y no 2**, en orden de peso. (i) *El objetivo del taller es la clase minoritaria*: con
$K=2$ el estado de riesgo absorbe correcciones ordinarias y crisis sistémicas, y en la tarea previa
ocupaba el 25,3 % del tiempo (`docs/context/RESUMEN_DETECCION_REGIMENES.md:102`); una clase de 1 de
cada 4 no está desbalanceada y no justifica generar datos sintéticos. Con $K=3$ el estado intermedio
absorbe la corrección y el extremo queda reservado a la crisis genuina (§7.5). (ii) *La limitación
diagnosticada del detector previo era exactamente esa*: "Solo 2 estados. Colapsa corrección normal,
crisis sistémica y estanflación en un único estado Crisis"
(`docs/context/RESUMEN_DETECCION_REGIMENES.md:168-170`). (iii) *BIC apunta al mismo sitio*.

**Por qué no 4.** Con ~4.000 observaciones de train y 10 episodios de estrés en la ventana, un cuarto
estado se identifica mal y produce clases de pocas decenas de ventanas. El TFM avisa del límite: con
~4 crisis efectivas "la complejidad extra no se paga"
(`capa1_exploracion/memory/99_conclusions.md:248-263`).

**BIC no decide solo.** Un $K$ se acepta si además (a) los estados son económicamente distinguibles
—vol y retorno medio monótonos en el orden canónico—, (b) la duración media de los episodios es
plausible (decenas de días, no 3) y (c) ninguna clase cae por debajo del 3 % de las ventanas. El
notebook comprueba las tres (§8).

---

## 5. El problema de la identificabilidad de estados y su canonicalización económica

La verosimilitud de un HMM es **invariante a permutaciones de las etiquetas**: si $\sigma$ permuta
$\{0,\dots,K-1\}$, el modelo con parámetros permutados
$(\pi_\sigma, A_\sigma, \boldsymbol{\mu}_\sigma, \Sigma_\sigma)$ tiene idéntica verosimilitud. El EM
converge a una de las $K!$ soluciones equivalentes según la inicialización: el índice que devuelve
`hmmlearn` no significa nada por sí solo. No es cosmético aquí: el generador aprende
$p(X, y_{\text{reg}}, y_{\text{vol}})$, así que si el significado de $y_{\text{reg}} = 2$ cambia entre
reejecuciones el modelo *downstream* aprende ruido, y la sobremuestra sintética de la clase crisis
solo tiene sentido si "crisis" es un índice estable.

**Solución: la canonicalización económica del TFM** (`src/detector_base.py:224-294`). Ordenar por la
media del retorno es frágil —con dos estados el z-score de dos elementos es siempre $\{-1,+1\}$, y el
signo ruidoso de una diferencia de medias casi nula puede invertir crisis y calma—, así que el
criterio es: (1) calcular por estado interno la media $\bar{r}_i$ y la desviación $\hat{\sigma}_i$ de
los retornos del S&P 500 en sus días; (2) agrupar los estados en **bandas de volatilidad** de ancho
$\text{tol} = \texttt{VOL\_CLOSE\_FRAC} \times \overline{\hat{\sigma}}$, con
$\texttt{VOL\_CLOSE\_FRAC} = 0{,}15$ (`src/detector_base.py:57`); (3) ordenar ascendentemente por
banda de volatilidad —criterio **primario**— y, **solo dentro de una misma banda**, descendentemente
por retorno medio (menor retorno ⇒ más severo); (4) el estado canónico $0$ es el menos severo y el
$K-1$ el más severo (`src/detector_base.py:296-299`).

La asimetría es deliberada: **la volatilidad manda y el retorno solo desempata**. El TFM llegó a este
diseño tras comprobar que el criterio anterior invertía crisis y calma en detectores que separan solo
en varianza (`capa1_exploracion/memory/99_conclusions.md:265-281`). Con $K=3$ el orden resultante es
calma ≺ transición ≺ crisis, y el estado intermedio queda definido sin nombrarlo a mano.

```python
# src/regimenes.py — canonicalización económica.
# Criterio heredado de PRUEBAS_DETECCION_REGIMENES_DE_MERCADO_TFM/src/detector_base.py:224-294
import numpy as np

VOL_CLOSE_FRAC = 0.15  # dos estados tienen "vol próxima" si difieren menos de este 15%

def orden_economico(etiquetas_internas, retornos_mercado, n_estados):
    """orden[i] = etiqueta interna que pasa a ser el estado canónico i
    (0 = calma, n_estados-1 = crisis)."""
    crudas = np.asarray(etiquetas_internas)
    r = np.asarray(retornos_mercado, dtype=float)
    observados = np.unique(crudas)
    # Perfil económico por estado: retorno medio y volatilidad (nan-safe).
    medias = np.array([np.nanmean(r[crudas == e]) for e in observados])
    sigmas = np.array([np.nanstd(r[crudas == e]) for e in observados])
    # La volatilidad es el criterio PRIMARIO: se agrupa en bandas.
    tol = VOL_CLOSE_FRAC * float(np.nanmean(sigmas))
    banda = np.round(sigmas / tol) if (tol > 0 and np.isfinite(tol)) else np.zeros(len(sigmas))
    # Asc. por banda de vol; dentro de la misma banda, desc. por retorno medio.
    severidad = sorted(range(len(observados)), key=lambda j: (banda[j], -medias[j]))
    orden = list(observados[severidad])
    orden += [e for e in range(n_estados) if e not in orden]  # no observados, al final
    return np.asarray(orden, dtype=int)


def aplicar_orden(etiquetas_internas, orden, n_estados):
    """Traduce etiquetas internas -> canónicas usando la permutación `orden`."""
    inversa = np.empty(n_estados, dtype=int)
    for canonico, interno in enumerate(orden):
        inversa[interno] = canonico
    return inversa[np.asarray(etiquetas_internas)]
```

---

## 6. Causalidad: por qué las features y el etiquetado no pueden mirar al futuro

La regla es que $X_t$ sea medible respecto a $\mathcal{F}_t$. Se viola de tres formas, dos sutiles.
**(a) Transformaciones que agregan el futuro**: un z-score calculado con la media y la desviación de
*toda* la muestra inyecta información de $t+1,\dots,T$ en el valor de $t$. Es el error que el TFM
identificó en su tarea previa (`docs/context/RESUMEN_DETECCION_REGIMENES.md:176-178`) y corrigió con
z-scores *expanding*/*rolling* (`src/features.py:56-95`); el catálogo del taller impone la misma
política, todos los `zscore_causal` son expanding con `min_periodos: 252`
(`data/catalog.yaml:73-104`). **(b) Estadísticos de ajuste estimados con test**: un `StandardScaler`
ajustado con el panel completo filtra media y varianza del test hacia el train, de ahí
`escalado.ajustar_con: train`. **(c) Solapamiento de ventanas entre particiones**: con ventanas de 60
días y paso 1, dos ventanas consecutivas comparten 59 días y un *split* aleatorio metería ventanas
casi idénticas en train y test; el taller usa *split* temporal con embargo de $60+21-1 = 80$ días
(`data/catalog.yaml:128-133`).

**Verificación, no confianza.** El TFM no argumenta la causalidad: la contrasta. `assert_causal`
computa las features sobre la muestra completa y sobre la muestra truncada en una fecha de corte y
exige $\max|\Delta| = 0$ en el tramo común (`src/features.py:199-226`). El notebook del taller replica
ese test sobre los 20 canales.

**El hallazgo incómodo que conviene heredar.** El TFM midió qué compraba realmente el look-ahead de la
tarea previa: al reimplementar el mismo HMM de forma causal, la cobertura de las crisis grandes **no
cayó** (COVID 0.96 in-sample y 0.96 causal), pero el `switching_rate` subió de 0.047 a 0.100. Su
conclusión textual: "el suavizado anti-causal de los z-scores de muestra completa regalaba persistencia
falsa […] no capacidad real de clasificación" (`capa1_exploracion/memory/99_conclusions.md:191-202`).
El look-ahead compraba **cosmética temporal**. Importa aquí: si al etiquetar se usa suavizado (§7.3),
los regímenes se verán más limpios de lo que un observador en tiempo real los vería, y eso se declara,
no se vende como calidad del modelo.

---

## 7. Régimen contemporáneo vs régimen futuro (la variable objetivo de este taller)

### 7.1. Dos problemas distintos

| | Detección (el TFM) | Predicción (este taller) |
|---|---|---|
| Pregunta | ¿en qué régimen estoy **hoy**? | ¿qué régimen dominará los **próximos 21 días**? |
| Objeto | $P(s_t \mid \mathcal{F}_t)$ | $P\big(\Phi(s_{t+1:t+h}) \mid \mathcal{F}_t\big)$ |
| Naturaleza | inferencia sobre un latente contemporáneo | pronóstico |
| Entrenamiento | no supervisado | supervisado, con etiqueta construida |
| Métrica | cobertura de crisis, falsas alarmas, *lead/lag* | *accuracy* / F1 por clase frente a la etiqueta |

La confusión es habitual, y los autores del método de detección más citado la marcan explícitamente:
su enfoque es "interpretativo más que predictivo", y su propósito "no es predecir cambios de régimen,
sino identificar cuándo ha ocurrido uno"
([Shu, Yu & Mulvey, 2024, arXiv:2402.05272](https://arxiv.org/html/2402.05272v2)). El mismo grupo
publica por separado el trabajo que sí convierte el régimen en objetivo: un esquema en **dos etapas**
donde un modelo no supervisado (jump model) produce las etiquetas históricas y después un clasificador
supervisado —*gradient boosting*— las predice hacia adelante a partir de features de retorno y
macro-features cruzadas ([Shu, Yu & Mulvey, 2024, arXiv:2406.09578](https://arxiv.org/abs/2406.09578)).
La arquitectura de este taller es exactamente esa, con el generativo sintético insertado entre ambas
etapas.

### 7.2. Definición formal de `y_reg`

Sea $s_{t+1},\dots,s_{t+h}$ la secuencia de estados decodificados en la ventana futura. La etiqueta es
la imagen de esa secuencia bajo un operador de agregación $\Phi$:

$$y_{\text{reg}}(t) \;=\; \Phi\big(s_{t+1},\dots,s_{t+h}\big), \qquad h = 21$$

Tres operadores razonables, con propiedades distintas:

**(i) Estado modal / voto mayoritario** — el adoptado.

$$\Phi_{\text{modal}} = \arg\max_{k} \sum_{u=t+1}^{t+h} \mathbb{1}[s_u = k]$$

Responde a "qué régimen **domina** el mes". Robusto a un día suelto de parpadeo y de interpretación
directa. Sesgo conocido: un episodio de crisis corto pero severo (5 días de 21) queda absorbido por la
calma. Los empates se resuelven a favor del estado **más severo**, para no perder eventos cortos.

**(ii) Severidad máxima**, $\Phi_{\max} = \max_{u \in [t+1,\,t+h]} s_u$. Responde a "¿habrá estrés en
algún momento del mes?". Detecta episodios cortos, pero infla la clase crisis: basta un día. Con las
diagonales de transición típicas (§3) y $h=21$, una fracción grande de ventanas tocaría el estado
extremo al menos una vez y la etiqueta dejaría de discriminar.

**(iii) Umbral sobre la probabilidad de crisis**: $\Phi_\tau = K-1$ si
$h^{-1}\sum_{u=t+1}^{t+h} P(s_u = K-1 \mid \cdot) \ge \tau$, y $\Phi_{\text{modal}}$ en otro caso. Usa
la información blanda del HMM y da un mando explícito sobre el balance de clases vía $\tau$, a costa
de un hiperparámetro más y de una etiqueta que ya no es función determinista de la secuencia de
estados. Se implementa como alternativa para el análisis de sensibilidad, no por defecto.

Decisión: **$\Phi_{\text{modal}}$** con desempate hacia el estado más severo, coherente con
`data/catalog.yaml:155` (`agregacion_horizonte: modal`).

### 7.3. La asimetría etiqueta / feature

Es el punto donde más se falla, y en las dos direcciones.

**La etiqueta puede mirar al futuro. Por definición.** $y_{\text{reg}}(t)$ es función de
$s_{t+1:t+h}$, es decir de $\mathcal{F}_{t+h}$. No es un fallo del diseño: **es el diseño**. Un
problema de predicción consiste precisamente en aprender $\hat{f}: X_t \mapsto y_t$ con $y_t$ en el
futuro de $X_t$. En el momento de entrenar, ese futuro ya ocurrió y es observable; en el momento de
usar el modelo sobre una fecha real $T$, $y_{\text{reg}}(T)$ es desconocido —y por eso el modelo sirve
para algo. **Las features no pueden mirar al futuro. Nunca.** $X_t$ debe ser medible respecto a
$\mathcal{F}_t$ (§6): si un canal contuviera información de $t+1{:}t+h$, el modelo aprendería a leer
la respuesta y la evaluación sería ficticia. La frontera se enuncia así:

> Es legítimo que **el valor** de la etiqueta dependa del futuro de su ventana. Es ilegítimo que **la
> definición** de la etiqueta —los parámetros del modelo que la produce— dependa de datos de
> validación o test.

Tres reglas operativas se siguen de ahí:

1. **El HMM se ajusta solo con el tramo de entrenamiento** (`train_hasta: "2018-12-31"`,
   `data/catalog.yaml:130`); luego se **decodifica** toda la serie con parámetros congelados. Si se
   ajustara con la muestra completa, las medias y covarianzas de los estados incorporarían el COVID y
   2022, y las etiquetas de train reflejarían un conocimiento que en 2018 no existía.
2. **La canonicalización también se fija con train.** El orden calma ≺ transición ≺ crisis se calcula
   con los retornos del tramo de entrenamiento y se congela; si no, el significado de la clase 2
   dependería del periodo de test.
3. **La decodificación de la etiqueta sí puede ser suavizada.** Al construir `y_reg` se puede usar
   Viterbi o *forward-backward* sobre toda la serie, porque el estado decodificado no entra en $X$: es
   materia prima de la etiqueta. El TFM tuvo que prohibirlo (`predict_online` con filtrado *forward*,
   `capa1_exploracion/detectors/hmm_gaussian_2s.py:209-230`) porque allí el estado decodificado **era
   la salida del sistema**, evaluada como alerta en tiempo real. Aquí no lo es. Requisito: el **mismo
   decodificador** en train, validación y test, para que la etiqueta signifique lo mismo en las tres.

**Consecuencia que se declara en el informe.** Las etiquetas de este taller no son comparables con las
métricas causales del TFM: al usar suavizado, los episodios salen más limpios y persistentes de lo que
vería un operador en tiempo real (§6: `switching_rate` 0.047 suavizado frente a 0.100 causal). Es
aceptable para definir una variable objetivo estable, y engañoso si se presenta como capacidad de
detección.

### 7.4. La línea base que hay que batir

Los regímenes son persistentes: con $a_{ii} \approx 0{,}94$–$0{,}98$ (§3), el régimen de hoy es un
predictor fortísimo del régimen dominante dentro de 21 días. **Toda evaluación honesta del modelo
*downstream* debe reportar la línea base de persistencia** $\hat{y}^{\,\text{base}}_{\text{reg}}(t) =
s_t$ —"el régimen futuro será el actual"—. Un clasificador que no la bata no ha aprendido nada sobre
el futuro: ha aprendido a leer el presente. Se reporta junto a la *accuracy* y al F1 de la clase
crisis en `results/metricas/`.

### 7.5. Cuantificación del desbalance

La justificación del taller —generar datos sintéticos porque la clase crisis es minoritaria— exige un
número, no una impresión. Lo que se sabe con precisión y de dónde sale:

| Anclaje | Valor | Fuente |
|---|---|---|
| Tiempo en el estado de riesgo, **HMM de 2 estados**, panel 2007-2026, 7 features | **25,3 %** (estacionaria 25,4 %) | `docs/context/RESUMEN_DETECCION_REGIMENES.md:102` y `:111` |
| Duración media del episodio de riesgo, mismo modelo | 17 días (esperada 16,5) | `docs/context/RESUMEN_DETECCION_REGIMENES.md:101` y `:110` |
| Días hábiles dentro de un episodio pico→suelo de las 10 crisis catalogadas en 2003-2026 | **17,7 %** (1.092 de 6.162) | cálculo sobre el bloque `crisis_catalog` del TFM (`data/catalog.yaml:2207-2343`), ventana 2003-01-02 – 2026-08-15 |
| Nº de crisis en la ventana de la pista B (2003+) | **10** | `docs/GLOSARIO.md:17` y el propio `crisis_catalog` |

Las dos primeras cifras corresponden a un modelo de **2 estados**, cuyo estado de riesgo agrega
corrección y crisis. Con $K=3$ ese 25,3 % se reparte entre el estado intermedio y el extremo, de modo
que **la clase crisis del taller será estrictamente menor**. La cota superior dura la marca el 17,7 %
de días dentro de un episodio pico→suelo, que incluye tramos de caída lenta y ordenada que un HMM no
clasifica como crisis. La estimación de trabajo, ~10 %, es coherente con ambos anclajes, pero **no es
un dato: es una expectativa a verificar**. El notebook mide la fracción real, la escribe en
`results/metricas/distribucion_regimenes.csv` y la contrasta con estos anclajes. Criterio de
aceptación: la clase crisis debe quedar en $[0{,}03,\,0{,}20]$; por debajo del 3 % no hay ventanas
suficientes para entrenar ni evaluar, y por encima del 20 % el etiquetado no está separando crisis de
corrección y el argumento del taller se cae.

---

## 8. Protocolo de etiquetado adoptado

Especificación ejecutable, fuente de verdad de `src/regimenes.py` y de
`notebooks/01_etiquetado_regimenes.ipynb`. Sus parámetros viven en `data/catalog.yaml:142-155`.

| Decisión | Valor | Justificación |
|---|---|---|
| Método | `hmmlearn.hmm.GaussianHMM` 0.3.3 | §2: persistencia aprendida + posteriores + BIC, coste medio |
| Nº de estados $K$ | **3** (0 calma, 1 transición, 2 crisis) | §4 |
| Covarianza | `full` | capta el cambio de signo de la correlación acción/bono entre regímenes |
| Features de etiquetado ($d=5$) | `ret_sp500`, `vol_realizada_z`, `drawdown_sp500`, `spread_credito_z`, `vix_nivel_z` | subconjunto de los 20 canales de $X$; $d$ bajo para que $K=3$ sea identificable (§4) |
| Inicializaciones | semillas 42–46, se elige la de mayor $\log\mathcal{L}$ | EM multimodal (§3) |
| `n_iter` / `tol` | 1000 / $10^{-4}$ | convergencia holgada; el ajuste es de segundos |
| Ajuste | **solo `train`** (hasta 2018-12-31) | §7.3, regla 1 |
| Decodificación | Viterbi sobre la serie completa, parámetros congelados | §7.3, regla 3 |
| Canonicalización | vol primaria en bandas del 15 %, retorno como desempate | §5, `src/detector_base.py:224-294` |
| Agregación | modal sobre $h=21$ días, empate al estado más severo | §7.2 |
| Alineación | $y_{\text{reg}}(t)$ usa $s_{t+1..t+21}$; $X_t$ usa $t-59..t$ | sin solapamiento |
| Embargo entre particiones | 80 días | $60 + 21 - 1$ (`data/catalog.yaml:128-133`) |
| Semilla global | 42 | reproducibilidad |

```python
# src/regimenes.py — ajuste y decodificación.
# CONTRATO: el HMM se ajusta SOLO con el tramo de entrenamiento; la serie completa
# se decodifica después con los parámetros congelados (§7.3).
import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM

# Retorno del índice, vol realizada 21d, drawdown corriente, proxy de spread de
# crédito (LQD-IEF) y NIVEL de VIX (no su variación). Todas causales (§6).
FEATURES_ETIQUETADO = ["ret_sp500", "vol_realizada_z", "drawdown_sp500",
                       "spread_credito_z", "vix_nivel_z"]
N_ESTADOS = 3
SEMILLAS = (42, 43, 44, 45, 46)


def ajustar_hmm(panel_train, features=FEATURES_ETIQUETADO, n_estados=N_ESTADOS,
                semillas=SEMILLAS, n_iter=1000, tol=1e-4):
    """Ajusta el HMM SOLO con el tramo de entrenamiento. El EM es multimodal: se
    prueban varias semillas y se devuelve la de mayor log-verosimilitud."""
    O = panel_train[features].to_numpy(dtype=float)
    if np.isnan(O).any():
        raise ValueError("El panel de etiquetado contiene NaN: recórtalos antes de ajustar.")
    mejor, mejor_ll = None, -np.inf
    for semilla in semillas:
        modelo = GaussianHMM(n_components=n_estados, covariance_type="full",
                             n_iter=n_iter, tol=tol, random_state=semilla)
        try:
            modelo.fit(O)
            ll = modelo.score(O)
        except Exception:            # alguna semilla puede no converger
            continue
        if np.isfinite(ll) and ll > mejor_ll:
            mejor, mejor_ll = modelo, ll
    if mejor is None:
        raise RuntimeError("Ninguna inicialización del HMM convergió.")
    # Orden canónico fijado con los retornos del TRAMO DE ENTRENAMIENTO.
    orden = orden_economico(mejor.predict(O), panel_train["ret_sp500"].to_numpy(), n_estados)
    return mejor, orden, mejor_ll


def decodificar_serie(modelo, orden, panel, features=FEATURES_ETIQUETADO, n_estados=N_ESTADOS):
    """Decodifica TODA la serie con los parámetros congelados del train. Usa Viterbi
    (suavizado): legítimo porque alimenta la ETIQUETA, no las features (§7.3)."""
    O = panel[features].to_numpy(dtype=float)
    estados = aplicar_orden(modelo.predict(O), orden, n_estados)
    proba = modelo.predict_proba(O)[:, orden]      # columnas ya en orden canónico
    return (pd.Series(estados, index=panel.index, name="estado"),
            pd.DataFrame(proba, index=panel.index,
                         columns=[f"p_estado_{i}" for i in range(n_estados)]))
```

```python
# src/regimenes.py — agregación de la ventana futura y controles de aceptación.

def etiqueta_regimen_futuro(estados, horizonte=21, criterio="modal",
                            proba_crisis=None, umbral=0.30, n_estados=N_ESTADOS):
    """y_reg(t) = régimen dominante en (t, t+horizonte].

    MIRA AL FUTURO por construcción: es la variable a predecir, no una feature. Las
    últimas `horizonte` fechas quedan a NaN (sin futuro observado). Criterios:
    'modal' voto mayoritario (empate -> más severo); 'severidad' máximo del estado
    en la ventana; 'umbral' crisis si la P(crisis) media >= `umbral`, si no modal.
    """
    s = np.asarray(estados, dtype=int)
    y = np.full(len(s), np.nan)
    for t in range(len(s) - horizonte):
        futuro = s[t + 1: t + 1 + horizonte]
        if criterio == "severidad":
            y[t] = futuro.max(); continue
        # Con el array de cuentas invertido, argmax devuelve el estado MÁS severo en
        # caso de empate (no queremos perder episodios de crisis cortos).
        cuentas = np.bincount(futuro, minlength=n_estados)
        modal = n_estados - 1 - int(np.argmax(cuentas[::-1]))
        if criterio == "umbral":
            p = np.asarray(proba_crisis, dtype=float)[t + 1: t + 1 + horizonte].mean()
            y[t] = (n_estados - 1) if p >= umbral else modal
        else:
            y[t] = modal
    return pd.Series(y, index=getattr(estados, "index", None), name="y_reg")


def controles_etiquetado(modelo, orden, estados, y_reg, retornos, n_estados=N_ESTADOS):
    """Controles de aceptación. Ninguno es opcional: si falla uno, no se congela."""
    A = modelo.transmat_[np.ix_(orden, orden)]     # transición en orden canónico
    duracion = 1.0 / (1.0 - np.diag(A))
    vals, vecs = np.linalg.eig(A.T)                # estacionaria: autovector izq.
    pi = np.real(vecs[:, np.argmin(np.abs(vals - 1.0))]); pi = pi / pi.sum()
    r = pd.Series(np.asarray(retornos), index=estados.index)
    vol = np.array([r[estados == k].std() for k in range(n_estados)])
    ret = np.array([r[estados == k].mean() for k in range(n_estados)])
    frec = y_reg.dropna().value_counts(normalize=True).sort_index()
    p_crisis = float(frec.get(float(n_estados - 1), 0.0))

    # C1 la vol crece con el índice canónico; C2 el estado de crisis pierde dinero;
    # C3 persistencia plausible; C4 desbalance dentro del rango declarado en §7.5.
    assert np.all(np.diff(vol) > 0), f"Orden canónico incoherente: vol = {vol}"
    assert ret[-1] < 0, f"El estado de crisis no tiene retorno negativo: {ret}"
    assert np.all(duracion > 5), f"Regímenes demasiado cortos: {duracion}"
    assert 0.03 <= p_crisis <= 0.20, f"Clase crisis fuera de rango: {p_crisis:.3f}"
    return {"transmat": A, "duracion": duracion, "estacionaria": pi,
            "vol_estado": vol, "ret_estado": ret, "frec_y_reg": frec}
```

**Artefactos del notebook** (versionados): `data/processed/regimenes.parquet` (`estado`,
`p_estado_0..2`, `y_reg`); `results/metricas/distribucion_regimenes.csv` (frecuencia por clase, global
y por partición); `results/metricas/transicion_regimenes.csv` ($A$ canónica, duraciones,
estacionaria); `results/figures/regimenes_sp500.png` (índice coloreado por estado, con las 10 crisis
del `crisis_catalog` marcadas) y `regimenes_timeline.png` (estados y $P(\text{crisis})$).

**Validación externa obligatoria.** El etiquetado se contrasta contra el `crisis_catalog` del TFM
(`data/catalog.yaml:2207-2343`): se reporta el % de días en estado 2 dentro de cada episodio
pico→suelo. Referencia de la tarea previa con 2 estados: Lehman 98,6 %, COVID 92,3 %, Inflación
78,9 %, y los puntos ciegos Taper Tantrum 2013 (10,9 %) y Q4 2018 (20,6 %)
(`docs/context/RESUMEN_DETECCION_REGIMENES.md:119-126`). Con $K=3$ se espera que parte de 2013 y 2018
caiga en el estado intermedio: **eso es el resultado buscado**, no un fallo. 2013 es un punto ciego
universal en el TFM —seis detectores independientes lo ignoran— porque fue un shock de tipos sin
estrés sistémico de renta variable (`capa1_exploracion/memory/99_conclusions.md:229-246`); no se
fuerza su captura.

**Limitaciones declaradas.** (1) Emisiones gaussianas con curtosis 25–40: el modelo está mal
especificado y subestima las colas; un HMM t-Student mejoraría el BIC con holgura, a costa de la
reproducibilidad. (2) La etiqueta se decodifica con suavizado: sale más limpia que la que vería un
operador en tiempo real (§6, §7.3). (3) Con 10 episodios de estrés en la ventana, la clase crisis
descansa en pocos eventos efectivos; ninguna diferencia de pocos puntos de *accuracy* entre modelos
*downstream* es estadísticamente distinguible (`capa1_exploracion/memory/99_conclusions.md:394-405`).
(4) `y_reg` no está definida en los últimos 21 días de la muestra.

---

## 9. Referencias

**Literatura**

- Hamilton, J. D. (1989). *A New Approach to the Economic Analysis of Nonstationary Time Series and the Business Cycle*. **Econometrica** 57(2), 357-384. [enlace](https://www.econometricsociety.org/publications/econometrica/1989/03/01/new-approach-economic-analysis-nonstationary-time-series-and)
- Ang, A. & Bekaert, G. (2002). *International Asset Allocation with Regime Shifts*. **Review of Financial Studies** 15(4), 1137-1187. [enlace](https://academic.oup.com/rfs/article-abstract/15/4/1137/1568247)
- Pagan, A. R. & Sossounov, K. A. (2003). *A Simple Framework for Analysing Bull and Bear Markets*. **J. of Applied Econometrics** 18(1), 23-46. [enlace](https://onlinelibrary.wiley.com/doi/abs/10.1002/jae.664)
- Guidolin, M. & Timmermann, A. (2007). *Asset Allocation under Multivariate Regime Switching*. **J. of Economic Dynamics and Control** 31(11), 3503-3544. [enlace](https://www.sciencedirect.com/science/article/abs/pii/S0165188906002272)
- Kritzman, M. & Li, Y. (2010). *Skulls, Financial Turbulence, and Risk Management*. **Financial Analysts Journal** 66(5), 30-41. [enlace](https://www.tandfonline.com/doi/abs/10.2469/faj.v66.n5.3)
- Shu, Y., Yu, C. & Mulvey, J. M. (2024). *Downside Risk Reduction Using Regime-Switching Signals: A Statistical Jump Model Approach*. **J. of Asset Management**. [arXiv:2402.05272](https://arxiv.org/html/2402.05272v2) — distinción explícita entre identificar un régimen y predecirlo.
- Shu, Y., Yu, C. & Mulvey, J. M. (2024). *Dynamic Asset Allocation with Asset-Specific Regime Forecasts*. **Annals of Operations Research**. [arXiv:2406.09578](https://arxiv.org/abs/2406.09578) — etiquetado no supervisado seguido de clasificador supervisado que predice el régimen.
- *Machine Learning-Driven Market Regime Analysis in Equity Markets: A Gaussian HMM Approach* (2025). [enlace](https://www.researchgate.net/publication/398722347_Machine_Learning-Driven_Market_Regime_Analysis_in_Equity_Markets_A_Gaussian_Hidden_Markov_Model_Approach) — comparación BIC de 2 frente a 3 estados.
- hmmlearn developers. *hmmlearn: Hidden Markov Models in Python*, v0.3.x. [enlace](https://hmmlearn.readthedocs.io/en/latest/api.html)

**Fuentes internas** (TFM `PRUEBAS_DETECCION_REGIMENES_DE_MERCADO_TFM/`). Núcleo:
`src/detector_base.py:57` (`VOL_CLOSE_FRAC`), `:224-294` (`_economic_state_order`), `:296-299`
(`crisis_state`); `src/features.py:56-95` (`causal_zscore`), `:199-226` (`assert_causal`);
`src/evaluation.py:32-56` (ventanas de crisis y trampas), `:141-262` (`walk_forward`). Datos:
`data/catalog.yaml:1118-1123` (pista B, 2003+), `:2207-2343` (`crisis_catalog`: 22 crisis verificadas
del S&P 500, 10 en la ventana 2003-2026); `docs/GLOSARIO.md:14-19` (pistas A/B), `:45-50` (regla de
oro anti-fuga), `:52-57` (causalidad). Tarea previa:
`docs/context/RESUMEN_DETECCION_REGIMENES.md:95-115` (HMM de 2 estados: 74,7 %/25,3 %, transición,
duraciones), `:119-126` (validación por evento), `:162-183` (limitaciones). Banco de detectores:
`capa1_exploracion/detectors/hmm_gaussian_2s.py:151-247` (multi-semilla, filtrado *forward*, conteo de
parámetros); `memory/detectors/03_clustering_gmm.md:43-45` (BIC $K=2$ vs $K=3$),
`04_hmm_gaussian_2s.md:96-102` (parpadeo in-sample vs causal), `09_jump_model.md:36-46` (persistencia
vs cobertura); `memory/99_conclusions.md:191-215` (qué compraba el look-ahead), `:229-246` (punto
ciego de 2013), `:265-281` (severidad vol-primaria), `:394-405` (límite inferencial con ~4 crisis).

**Configuración del taller.** `data/catalog.yaml:28-31` (ventana 2003-2026), `:73-104` (20 canales),
`:114-118` (ventanas 60/21), `:128-133` (particiones y embargo), `:142-155` (bloque `regimenes`).
Entorno: Python 3.13.7, `hmmlearn` 0.3.3, `scikit-learn` 1.8.0, `numpy` 2.3.4, `pandas` 2.3.3,
CPU-only.
