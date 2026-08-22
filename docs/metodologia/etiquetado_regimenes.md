# Etiquetado de regímenes de mercado

Documento metodológico del taller B5-T1. Explica cómo se construye `y_reg`, la variable objetivo de
clasificación del modelo *downstream*, y **por qué** se construye así.

Qué es y qué no es. Es la **justificación metodológica** de las decisiones que implementan
`src/regimenes.py` y el cuaderno `01_etiquetado_regimenes.ipynb`. **No** es una especificación
ejecutable: los parámetros vigentes viven en `data/catalog.yaml` y la API real, en el módulo. Cuando
este documento y el código discrepen, manda el código; los fragmentos de código que aquí aparecen son
ilustrativos y se citan por nombre de función para que la discrepancia se pueda comprobar.

El taller hereda el marco causal y la canonicalización económica del TFM de detección de regímenes
(`PRUEBAS_DETECCION_REGIMENES_DE_MERCADO_TFM/`), pero **cambia el problema**: aquel detecta el
régimen de hoy; este predice el que dominará los próximos 21 días. La sección 7 desarrolla esa
diferencia, que es el núcleo del documento.

Notación: $r_t$ retorno logarítmico diario del S&P 500; $\mathbf{o}_t \in \mathbb{R}^{d}$ vector de
*features de etiquetado* ($d=3$, §8); $s_t \in \{0,\dots,K-1\}$ estado latente con $0$ = calma y
$K-1$ = crisis; $\mathcal{F}_t = \sigma(\{\mathbf{o}_u\}_{u \le t})$; $h = 21$ horizonte de agregación
en días de mercado; $X_t \in \mathbb{R}^{60 \times 20}$ ventana de entrada
(`data/catalog.yaml`, bloque `canales`); $y_{\text{reg}}(t)$ etiqueta de la ventana que acaba en $t$.

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
problema *downstream*, por lo que el protocolo se congela por escrito (§8) en el bloque `regimenes` de
`data/catalog.yaml`—, y **la etiqueta se valida, no se cree**: se acepta si supera controles
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
| **Clustering** | k-means / GMM sobre features, cada día independiente | **ninguna** · parpadeo | `clustering_gmm_k3`: `switching_rate` 0.126, duración 7.9 d (`memory/detectors/03_clustering_gmm.md:73-79`) |
| **HMM** | estado latente markoviano + emisiones $p(\mathbf{o}_t \mid s_t)$ | explícita, en la diagonal de $A$ | `hmm_gaussian_2s`: `switching_rate` 0.100, duración 9.9 d (`memory/detectors/04_hmm_gaussian_2s.md:96-102`) |
| **Markov-switching** | los parámetros de una regresión/VAR conmutan (Hamilton, 1989) | explícita | `markov_switching_var_2s`: mejor cobertura sistémica del banco (0.98), pero ~33 min de ajuste |
| **Change-point** | detecta el instante del cambio de nivel/varianza (CUSUM) | máxima | `changepoint_online`: `switching_rate` 0.002, duración 436 d, especificidad 1.00 |
| **Jump models** | clustering + penalización $\lambda$ por salto de estado (Nystrup et al.) | ajustable vía $\lambda$ | `jump_model` ($\lambda=50$): `switching_rate` 0.005, duración 176.6 d, pero cobertura Inflación 2022 solo 0.17 (`memory/detectors/09_jump_model.md:36-46`) |

**La persistencia es un mando, no una virtud.** El jump model reduce el parpadeo 24× frente al GMM
(0.126 a 0.005) y multiplica por 22 la duración de los episodios, pero pierde el mercado bajista lento
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
(`capa1_exploracion/detectors/hmm_gaussian_2s.py:209-230`). Este taller **no** lo necesita para la
etiqueta —que puede mirar al futuro por definición (§7.3)— pero **sí** para la línea base de
persistencia, que es lo que se podía saber en el instante $t$: de ahí
`EtiquetadorRegimenes.predict_causal`, que rehace la recursión *forward* sobre las mismas emisiones
que usa Viterbi (§7.4).

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

Con $d=3$ (§8): $k = 21$ para $K=2$ y $k = 35$ para $K=3$. El salto es asumible porque $d$ es pequeño;
ese es el motivo de etiquetar con un subconjunto reducido de features y no con los 20 canales de $X$
(con $d=20$ se pasaría de 463 a 698 parámetros, insostenible con ~4.000 observaciones de train). El
ajuste rechazado de cinco features (§8) costaba 43 y 68 parámetros: el coste no era su problema —lo
fue el supuesto gaussiano—, pero conviene tener la referencia para leer la comparación del cuaderno 01.

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
cada 4 no está desbalanceada y no justifica generar datos sintéticos. Ese 25,3 % es un anclaje
externo —otro panel, otras features—, pero el cuaderno 01 **ajusta ahora el contraste de dos estados
sobre este panel** y mide **37,4 %** de estado extremo frente al 15,8 % de la versión de tres, con
**1 de los 4 controles bloqueantes suspendido** (el del peso de la clase de crisis, banda
[3 %, 20 %]): la medición propia es más adversa a $K=2$ que el anclaje. Con $K=3$ el estado intermedio
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
criterio es: (1) calcular por estado interno la media de la feature de volatilidad
$\bar{v}_i$ y la media del retorno $\bar{r}_i$; (2) agrupar los estados en **bandas de volatilidad**
de ancho $\text{tol} = \texttt{FRACCION\_VOL\_CERCANA} \times \overline{|v|}$, con
$\texttt{FRACCION\_VOL\_CERCANA} = 0{,}15$; (3) ordenar ascendentemente por
banda de volatilidad —criterio **primario**— y, **solo dentro de una misma banda**, descendentemente
por retorno medio (menor retorno ⇒ más severo); (4) el estado canónico $0$ es el menos severo y el
$K-1$ el más severo.

Una diferencia con el TFM que conviene declarar: allí la severidad se medía con la **desviación
típica de los retornos** del S&P 500 dentro de cada estado; aquí se mide con la **media del canal de
volatilidad** que ya está entre las features de etiquetado (`vol_realizada_z`, y `vix_nivel_z` como
respaldo). El criterio de orden —volatilidad primaria en bandas, retorno como desempate— es el mismo;
lo que cambia es de dónde sale la magnitud de volatilidad, que aquí se lee directamente de una feature
observada en vez de recalcularse sobre los retornos.

La asimetría es deliberada: **la volatilidad manda y el retorno solo desempata**. El TFM llegó a este
diseño tras comprobar que el criterio anterior invertía crisis y calma en detectores que separan solo
en varianza (`capa1_exploracion/memory/99_conclusions.md:265-281`). Con $K=3$ el orden resultante es
calma ≺ transición ≺ crisis, y el estado intermedio queda definido sin nombrarlo a mano.

Así se implementa, en `EtiquetadorRegimenes._orden_economico` y en la traducción que hace `predict`
(`src/regimenes.py`; fragmento ilustrativo, la versión vigente es la del módulo):

```python
FRACCION_VOL_CERCANA = 0.15  # dos estados tienen "vol próxima" si difieren menos de este 15 %

# El orden se calcula UNA vez, dentro de fit(), y queda congelado en self.orden.
estados = modelo.predict(features.to_numpy())
col_vol = self._columna(["vol_realizada", "vix_nivel", "vol"])
col_ret = self._columna(["ret_sp500", "ret", "retorno"])

vol_por_estado = np.array([features[col_vol].to_numpy()[estados == k].mean()
                           for k in range(self.n_estados)])
ret_por_estado = np.array([features[col_ret].to_numpy()[estados == k].mean()
                           for k in range(self.n_estados)])

# La volatilidad es el criterio PRIMARIO: se redondea a bandas de ancho `umbral`
# para que estados de volatilidad equivalente caigan en la misma banda y se
# desempaten por retorno (mayor retorno = mejor régimen = va antes).
umbral = FRACCION_VOL_CERCANA * float(np.abs(vol_por_estado).mean())
banda = np.round(vol_por_estado / umbral) if umbral > 0 else vol_por_estado
self.orden = np.lexsort((-ret_por_estado, banda))

# `orden[i]` es el estado crudo que ocupa la posición canónica i; para etiquetar
# hace falta la permutación inversa, que es lo que aplican predict/predict_causal.
inverso = np.empty_like(self.orden)
inverso[self.orden] = np.arange(self.n_estados)
canonicos = inverso[crudos]
```

---

## 6. Causalidad: por qué las features y el etiquetado no pueden mirar al futuro

La regla es que $X_t$ sea medible respecto a $\mathcal{F}_t$. Se viola de tres formas, dos sutiles.
**(a) Transformaciones que agregan el futuro**: un z-score calculado con la media y la desviación de
*toda* la muestra inyecta información de $t+1,\dots,T$ en el valor de $t$. Es el error que el TFM
identificó en su tarea previa (`docs/context/RESUMEN_DETECCION_REGIMENES.md:176-178`) y corrigió con
z-scores *expanding*/*rolling* (`src/features.py:56-95`); el catálogo del taller impone la misma
política, todos los `zscore_causal` son expanding con `min_periodos: 252`
(`data/catalog.yaml`, bloque `canales`). **(b) Estadísticos de ajuste estimados con test**: un
`StandardScaler` ajustado con el panel completo filtra media y varianza del test hacia el train, de ahí
`escalado.ajustar_con: train`. **(c) Solapamiento de ventanas entre particiones**: con ventanas de 60
días y paso 1, dos ventanas consecutivas comparten 59 días y un *split* aleatorio metería ventanas
casi idénticas en train y test; el taller usa *split* temporal con embargo de **85 sesiones de
mercado**, por encima del mínimo de $60+21-1 = 80$ que neutraliza el solape
(`data/catalog.yaml`, bloque `particiones`, clave `embargo_sesiones`). El embargo se cuenta en
sesiones de mercado, nunca en días naturales.

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
estados. **No está implementado**: `regimenes.regimen_dominante` solo admite `modal` y `maximo`, y
cualquier otro valor lanza `ValueError`. Queda documentado como alternativa por si el análisis de
sensibilidad llega a necesitarlo.

Decisión: **$\Phi_{\text{modal}}$** con desempate hacia el estado más severo, coherente con
`data/catalog.yaml`, bloque `regimenes`, clave `agregacion_horizonte: modal`. Lo implementa
`regimenes.regimen_dominante`, cuyo argumento `metodo` admite `modal` (por defecto) y `maximo`, que
es el $\Phi_{\max}$ de (ii). El desempate hacia el estado más severo afecta a 9 de las 5.649 ventanas
etiquetadas: es poco, pero la regla tiene que decir lo que hace.

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
   `data/catalog.yaml`, bloque `particiones`); luego se **decodifica** toda la serie con parámetros
   congelados. Si se ajustara con la muestra completa, las medias y covarianzas de los estados
   incorporarían el COVID y
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
   La excepción, y es una sola: la **línea base de persistencia** no es etiqueta sino predicción, y
   por tanto se decodifica con el filtro causal (§7.4).

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
crisis en `results/metricas/`, y lo produce `evaluacion.lineas_base`.

Publicar aquí un solo número es engañoso, porque la cifra depende de dos elecciones que hay que
declarar por separado, y de una tercera que decide qué comparaciones son legítimas.

**Sobre qué muestra.** El cuaderno 01 mide la persistencia sobre la **muestra completa** y publica
**las dos decodificaciones por separado**: el filtro causal da **0,7775 de accuracy y 0,7850 de
recall de crisis**, y Viterbi **0,8210 y 0,8208**. El 0,821 que se citaba antes aquí como "la
persistencia" es la fila de Viterbi, no la oficial. Esa muestra está además dominada por el tramo de
entrenamiento, de modo que ninguna de las dos cifras es la barra: son un diagnóstico del etiquetado.
La que hace de barra es la de **test**, que es donde se evalúa el modelo *downstream*: allí la
persistencia **causal** da **0,772 de accuracy y 0,754 de F1 macro**, y la de Viterbi 0,805 y 0,790.
Las cuatro son correctas y ninguna es intercambiable con otra; citar la de muestra completa como si
fuera la de test infla la referencia, y citar la de Viterbi como si fuera la causal la infla otra
vez.

**Con qué decodificador.** `hmmlearn.predict` es Viterbi y nombra el régimen de hoy **mirando la
serie entera, futuro incluido** (§3). Para construir la etiqueta eso es legítimo —la etiqueta es el
futuro—, pero no para la línea base: la barra que hay que batir es lo que se podía saber en el
instante $t$. El decodificador honesto es el filtro *forward*, implementado en
`EtiquetadorRegimenes.predict_causal`, y sobre test da **0,772 de accuracy, 0,754 de F1 macro y 0,800
de recall de crisis**. Las dos decodificaciones difieren en **321 de 5.670 sesiones (5,66 %)**.

**La barra oficial es la causal.** `evaluacion.lineas_base` publica las dos filas,
`persistencia_causal` y `persistencia_viterbi`, y marca la segunda con `usa_futuro = True`: se enseña
al lado, explícitamente etiquetada como **no alcanzable por un modelo causal**. La diferencia no es
cosmética: 0,790 frente a 0,754 de F1 macro sobre test son 3,6 puntos que se le regalarían a
cualquier modelo que se comparase con la versión de Viterbi.

**La banda, y qué comparaciones permite.** Las 110 ventanas de crisis del test **no son 110
observaciones**: son **3 rachas** contiguas, que es lo que cuenta `regimenes.tramos_contiguos`.
`evaluacion.banda_bloques` remuestrea por bloques de 81 ventanas —la huella de una ventana, $60+21$—
y da para ese `recall_crisis` de 0,800 un intervalo de confianza del 95 % de **[0,560 – 1,000]**,
frente al **[0,716 – 0,864]** que produce el intervalo binomial de Wilson tratando las ventanas como
independientes: el honesto es **3,0 veces más ancho**. La consecuencia se escribe explícitamente
porque condiciona todo el análisis del barrido: **ninguna comparación de `recall_crisis` entre dos
recetas que difiera en menos de unos 20 puntos es distinguible del ruido**. Ordenar la tabla de
resultados por esa columna y quedarse con la primera fila es quedarse con el ruido más afortunado.

### 7.5. Cuantificación del desbalance

La justificación del taller —generar datos sintéticos porque la clase crisis es minoritaria— exige un
número, no una impresión. Lo que se sabe con precisión y de dónde sale:

| Anclaje | Valor | Fuente |
|---|---|---|
| Tiempo en el estado de riesgo, **HMM de 2 estados**, panel 2007-2026, 7 features | **25,3 %** (estacionaria 25,4 %) | `docs/context/RESUMEN_DETECCION_REGIMENES.md:102` y `:111` |
| Duración media del episodio de riesgo, mismo modelo | 17 días (esperada 16,5) | `docs/context/RESUMEN_DETECCION_REGIMENES.md:101` y `:110` |
| Días hábiles dentro de un episodio pico-suelo de las 10 crisis catalogadas en 2003-2026 | **17,7 %** (1.092 de 6.162) | cálculo sobre el bloque `crisis_catalog` **del TFM** (`PRUEBAS_DETECCION_REGIMENES_DE_MERCADO_TFM/data/catalog.yaml:2207-2343`), ventana 2003-01-02 – 2026-08-15 |
| Nº de crisis en la ventana de la pista B (2003+) | **10** | `docs/GLOSARIO.md:17` del TFM y el propio `crisis_catalog` |

Aviso sobre las fuentes de esta tabla: `crisis_catalog` es un bloque del catálogo **del TFM**, no del
de este taller. El `data/catalog.yaml` del taller tiene 207 líneas y no contiene ningún catálogo de
crisis; aquí la validación externa la hacen los tres episodios de `regimenes.EPISODIOS_REFERENCIA`
(GFC 2008, COVID 2020, Inflación 2022), que viven en el código y no en el catálogo precisamente
porque no son un parámetro (§8).

Las dos primeras cifras corresponden a un modelo de **2 estados**, cuyo estado de riesgo agrega
corrección y crisis. Con $K=3$ ese 25,3 % se reparte entre el estado intermedio y el extremo, de modo
que **la clase crisis del taller será estrictamente menor**. La cota superior dura la marca el 17,7 %
de días dentro de un episodio pico-suelo, que incluye tramos de caída lenta y ordenada que un HMM no
clasifica como crisis.

**Lo medido, y de dónde sale exactamente cada cifra.** La fracción real global es del **15,8 %** y la
imprime el `control_bloqueante` del cuaderno 01; no está en ningún CSV. El reparto por particiones,
contado en **ventanas**, lo escribe el cuaderno **02** en
`results/metricas/reparto_particiones.csv`:

| Partición | Ventanas | Peso de la clase de crisis | Fuente |
|---|---|---|---|
| Train | 3.696 | **15,9 %** (587 ventanas) | `reparto_particiones.csv` (cuaderno 02) |
| Validación | 672 | **22,0 %** | `reparto_particiones.csv` (cuaderno 02) |
| Test | 1.052 | **10,5 %** (110 ventanas) | `reparto_particiones.csv` (cuaderno 02) |
| Global | — | **15,8 %** | `control_bloqueante` impreso en el cuaderno 01 |

`results/metricas/distribucion_regimenes.csv`, que sí escribe el cuaderno 01 con
`regimenes.distribucion`, **no contiene esta tabla** y no debe citarse como su fuente: tiene solo las
tres particiones —sin fila global— y cuenta **sesiones etiquetadas antes de aplicar el embargo**, de
donde salen 15,63 % en train, 20,58 % en validación y 13,27 % en test. Los dos CSV difieren porque uno
cuenta sesiones de un tramo de fechas y el otro las ventanas que sobreviven al embargo de D7.

Tres lecturas. (i) El 15,8 % queda por debajo de la cota dura del 17,7 % y muy por debajo del 25,3 %
del modelo de dos estados, que es lo que el argumento anterior predecía. (ii) Está dentro de la banda
de aceptación, aunque no holgadamente: el criterio es que la clase crisis quede en
$[0{,}03,\,0{,}20]$ —por debajo del 3 % no hay ventanas suficientes para entrenar ni evaluar, y por
encima del 20 % el etiquetado no está separando crisis de corrección y el argumento del taller se
cae— y lo aplica `regimenes.control_bloqueante` (§8). (iii) **La prevalencia se dobla entre validación
y test**, 22,0 % frente a 10,5 %, y eso no es un detalle contable: cualquier métrica que dependa de
la prevalencia —el F1 macro, señaladamente— cambia de valor al pasar de una partición a otra sin que
el modelo haya cambiado. Es la razón por la que la arquitectura *downstream* se selecciona con
`balanced_accuracy`, que es invariante a la prevalencia, y no con F1 macro (`src/downstream.py`,
`METRICA_SELECCION`).

La expectativa de trabajo previa, del orden del 10 %, era coherente con los anclajes pero **no era un
dato**: el valor medido la supera en casi seis puntos, y es el medido el que gobierna. El desbalance
sigue siendo el que justifica el taller —una clase de cada seis en train, y de cada diez en test—,
pero conviene enunciarlo con la cifra correcta y no con la esperada.

---

## 8. Protocolo de etiquetado adoptado

Resumen del protocolo vigente, con el porqué de cada casilla. La fuente de verdad de los parámetros
es el bloque `regimenes` de `data/catalog.yaml`, y la de la mecánica, `src/regimenes.py`; esta tabla
los documenta, no los define.

| Decisión | Valor | Justificación |
|---|---|---|
| Método | `hmmlearn.hmm.GaussianHMM` 0.3.3 | §2: persistencia aprendida + posteriores + BIC, coste medio |
| Nº de estados $K$ | **3** (0 calma, 1 transición, 2 crisis) | §4 |
| Covarianza | `full` | capta el cambio de signo de la correlación acción/bono entre regímenes |
| Features de etiquetado ($d=3$) | `ret_sp500`, `vol_realizada_z`, `vix_nivel_z` | subconjunto de los 20 canales de $X$; $d$ bajo para que $K=3$ sea identificable (§4) |
| Features **rechazadas** ($d=5$, ajuste descartado) | `drawdown_sp500` y `spread_credito_z`, sumadas a las tres anteriores | incumplen el supuesto gaussiano del HMM: `spread_credito_z` **deriva** 1,17 sigmas entre las dos mitades de la muestra y `drawdown_sp500` **satura**, con el 47 % de las sesiones en el 5 % superior de su recorrido. Viven en `regimenes.features_descartadas` del catálogo |
| Inicializaciones | semillas 42–46, se elige la de mayor $\log\mathcal{L}$ | EM multimodal (§3) |
| `n_iter` / `tol` | 1000 / $10^{-4}$ | convergencia holgada; el ajuste es de segundos |
| Ajuste | **solo `train`** (hasta 2018-12-31) | §7.3, regla 1 |
| Decodificación de la **etiqueta** | Viterbi sobre la serie completa, parámetros congelados (`predict`) | §7.3, regla 3 |
| Decodificación de la **línea base** | filtro *forward* (`predict_causal`) | §7.4: la barra a batir es lo que se sabía en $t$ |
| Canonicalización | vol primaria en bandas del 15 %, retorno como desempate | §5, `EtiquetadorRegimenes._orden_economico` |
| Agregación | modal sobre $h=21$ días, empate al estado más severo | §7.2 |
| Alineación | $y_{\text{reg}}(t)$ usa $s_{t+1..t+21}$; $X_t$ usa $t-59..t$ | sin solapamiento |
| Embargo entre particiones | **85 sesiones de mercado** | mínimo $60 + 21 - 1 = 80$, más 5 de margen (`data/catalog.yaml`, bloque `particiones`) |
| Semilla global | 42 | reproducibilidad |

**Por qué las cinco features no son el protocolo vigente, y por qué siguen documentadas.** El ajuste
con $d=5$ **suspende el control bloqueante** del cuaderno 01, y conviene ser exacto sobre por dónde,
porque no es por donde se esperaría. El **peso de la clase de crisis sale al 19,0 %**, que está
**dentro** de la banda de aceptación $[0{,}03,\,0{,}20]$: **ese control pasa**, pegado al techo, que
ya es un aviso, pero pasa. Lo que suspende es la **cobertura del episodio de Inflación 2022: 44,0 %
frente al 50 % exigido** —con las tres features sube al 56,5 %—. Y el estado extremo se enciende
donde no debe: el **40 % de las sesiones de 2021** quedan marcadas como crisis, el año en que el
S&P 500 subió un 26,9 % con volatilidad del 13 %. No es el año entero, pero son cuatro de cada diez
sesiones de un mercado alcista y tranquilo. Nada de esto lo habría delatado el peso de la clase
mirado por su cuenta: el control que suspende es el de cobertura, no el de prevalencia, y esa es la
lección que hay que llevarse. La causa es la que mide `regimenes.diagnostico_features`: un HMM
gaussiano supone emisiones normales y **estacionarias dentro de cada estado**, y dos de las cinco lo
incumplen. `spread_credito_z` deriva porque su z-score expanding no vuelve nunca de 2008;
`drawdown_sp500` satura contra su tope, y una masa puntual en el tope no cabe en ninguna normal. Con
ellas dentro, el estado extremo deja de significar "mercado en tensión" y pasa a significar "estamos
después de 2020". Retirándolas, la crisis baja al 15,8 %, el control pasa y el ajuste deja de depender
de la inicialización. Las dos descartadas se conservan en `features_descartadas` del catálogo, y no
borradas, porque el cuaderno 01 **reconstruye con ellas el ajuste rechazado**: la comparación entre
los dos ajustes es el argumento (D6 en `docs/DECISIONES.md`), no un residuo histórico. Que una feature
no sirva para *etiquetar* no dice nada sobre si sirve como *entrada*: las dos siguen entre los 20
canales de $X$.

### 8.1. La API real de `src/regimenes.py`

El módulo no expone funciones sueltas: expone una **clase con estado**, `EtiquetadorRegimenes`, más
cinco funciones libres. Que el ajuste y la permutación canónica vivan dentro del mismo objeto es lo
que hace imposible decodificar con un orden que no sea el que se fijó al ajustar, que es el fallo que
la canonicalización existe para evitar (§5).

| Miembro | Qué hace | Contrato |
|---|---|---|
| `EtiquetadorRegimenes.desde_catalogo()` | construye el etiquetador leyendo el bloque `regimenes` del catálogo | ningún parámetro se escribe a mano en el cuaderno |
| `.fit(features)` | ajusta el HMM con las semillas del catálogo, se queda el de mayor $\log\mathcal{L}$ y **calcula y congela** `self.orden` | `features` debe traer **solo el tramo de train** (§7.3, regla 1) |
| `.predict(features)` | régimen canonizado día a día, Viterbi | serie `regimen`; mira la serie entera (§3) |
| `.predict_causal(features)` | régimen canonizado con el filtro *forward*, información hasta $t$ y ni un día más | serie `regimen_causal`; es la que sirve para líneas base y diagnósticos en tiempo real, **no** para construir la etiqueta |
| `.predict_proba(features)` | posterior **suavizada** por estado, columnas ya canonizadas | `p_regimen_0..2`; no vale como feature causal: cada fila incorpora información posterior |
| `.guardar(ruta)` / `.cargar(ruta)` | serializa el etiquetador ajustado | los cuadernos posteriores reutilizan el mismo objeto, no lo reajustan |
| `regimen_dominante(regimen_diario, horizonte, metodo)` | agrega la ventana futura a `regimen_futuro` | `modal` (por defecto) o `maximo`; §7.2 |
| `control_bloqueante(...)` | veredicto de aceptación, fila a fila | §8.2 |
| `diagnostico_features(...)` | puerta previa: deriva y saturación de cada candidata | §8.2 |
| `distribucion(etiquetas, n_estados)` | reparto por régimen, absoluto y en porcentaje | la tabla de §7.5 |
| `tramos_contiguos(etiquetas, clase)` | número de **rachas** de una clase, no de muestras | el tamaño muestral que hay que citar en cualquier intervalo (§7.4) |

Dos detalles del ajuste que el código documenta y conviene no perder. `fit` silencia el aviso
`Model is not converging` de `hmmlearn`, que aparece cuando la verosimilitud retrocede unas milésimas
en el último paso del EM: es ruido numérico y el ajuste ganador converge, pero imprime varias líneas
por semilla que dentro del cuaderno parecen un error grave. Y `predict_causal` reimplementa la
recursión *forward* en logaritmos —normalizando en cada paso para que `alfa` no acumule la
log-verosimilitud de toda la serie— sobre las mismas log-emisiones que usa Viterbi, de modo que la
comparación entre las dos decodificaciones es limpia.

```python
# Uso canónico, tal y como lo hace notebooks/01_etiquetado_regimenes.ipynb.
from src.regimenes import EtiquetadorRegimenes, regimen_dominante, control_bloqueante

etiquetador = EtiquetadorRegimenes.desde_catalogo().fit(features.loc[:train_hasta])

regimen = etiquetador.predict(features)                  # Viterbi, alimenta la ETIQUETA
regimen_causal = etiquetador.predict_causal(features)    # filtro forward, alimenta la LÍNEA BASE
proba = etiquetador.predict_proba(features)              # p_regimen_0..2, suavizada

y_reg = regimen_dominante(regimen, horizonte=21, metodo="modal")
assert control_bloqueante(regimen, y_reg, n_estados=3)["ok"].all()
```

### 8.2. Los controles que hay que pasar

No hay una batería `C1`-`C4` de asertos: hay **dos puertas**, y ninguna es opcional.

**Puerta previa: `diagnostico_features`.** Se ejecuta sobre las candidatas *antes* de ajustar nada,
porque las dos patologías que rompen las emisiones gaussianas de un HMM se manifiestan igual en el
resultado —un estado de crisis demasiado grande— y conviene medirlas antes y no diagnosticarlas
después. Devuelve una fila por feature con `deriva` (salto de la media entre la primera y la segunda
mitad de la muestra, en sigmas; tope 0,5), `saturacion` (porcentaje de sesiones en el 5 % superior del
recorrido; tope 10 %) y `apta`. Es la prueba que retira `spread_credito_z` (deriva 1,17) y
`drawdown_sp500` (saturación 47 %). Los dos umbrales son convenciones, no teoremas, y la prueba es
necesaria pero no suficiente: una feature puede pasarla y aun así no aportar información de régimen.

**Puerta de aceptación: `control_bloqueante`.** Es lo que hay que cruzar antes de entrenar ningún
generador. Un HMM converge siempre, así que la convergencia no es evidencia de nada; lo que se
comprueba es que los estados coinciden con crisis que de verdad ocurrieron y que la clase extrema
tiene un tamaño con el que se pueda trabajar. Cuatro filas, y el cuaderno corta con
`assert tabla["ok"].all()`:

| Control | Criterio | Por qué ese valor |
|---|---|---|
| Cobertura de **GFC 2008** (2008-09-01 a 2009-03-31) | 50 % o más de sesiones en el estado de crisis | 50 y no 90 porque con tres estados parte de un episodio cae legítimamente en transición |
| Cobertura de **COVID 2020** (2020-02-20 a 2020-04-30) | ídem | ídem |
| Cobertura de **Inflación 2022** (2022-01-01 a 2022-10-31) | ídem | 2022 fue un mercado bajista lento, no un desplome: exigir 90 % sería exigir que el modelo se equivoque |
| **Peso de la clase de crisis** sobre `regimen_futuro` | entre el 3 % y el 20 % | §7.5 |

Las fechas de los tres episodios son `regimenes.EPISODIOS_REFERENCIA`, y viven en el código y no en
el catálogo **precisamente porque no son un parámetro**: moverlas para que el control pase sería el
fraude que el control existe para impedir.

Devuelve una tabla y no un booleano, a propósito: cuando falla hay que saber *qué* falló y por
cuánto, porque de eso depende si se toca el número de estados o las features de etiquetado. Y que el
control pase no demuestra que el etiquetado sea bueno: solo descarta las dos formas de estar mal que
se pueden medir sin disponer de una etiqueta verdadera. En particular **no** detecta que el estado
intermedio quede vacío en alguna partición, que es un fallo real y hay que mirarlo en `distribucion`.

**Artefactos del cuaderno** (versionados): `data/processed/regimenes.parquet`, con las columnas
`regimen`, `regimen_futuro` y `p_regimen_0..2`; `results/metricas/distribucion_regimenes.csv`
(frecuencia por clase en train, validación y test, **sin fila global** y contando sesiones antes del
embargo, §7.5); `results/metricas/transicion_regimenes.csv` ($A$ canónica más la columna `duracion`
con la permanencia esperada, **sin** la distribución estacionaria); y dos figuras,
`results/figures/control_etiquetado.png` y `results/figures/linea_base_persistencia.png`.

Las dos figuras conservan su nombre heredado —el resto de las del cuaderno 01 lleva ya el prefijo
`01_`— precisamente para que estas referencias no se rompan, y las dos han cambiado de contenido:
`control_etiquetado.png` enfrenta ahora los dos etiquetados sobre el índice, con el peso de la clase
de crisis y los controles suspendidos de cada uno en su título, y una banda inferior que separa las
sesiones que sólo ve una de las dos versiones (**discrepan en el 18,5 % de la muestra**);
`linea_base_persistencia.png` es la misma figura que `01_persistencia_causal_vs_viterbi.png` —dos
matrices de confusión, filtro causal y Viterbi, con la segunda tramada por usar el futuro—, que el
cuaderno escribe dos veces bajo los dos nombres con dos llamadas seguidas a `viz.guardar`.

**Validación externa.** El etiquetado se contrasta contra los tres episodios de
`regimenes.EPISODIOS_REFERENCIA`, no contra un catálogo de crisis: el `crisis_catalog` que aparece en
la bibliografía interna es un bloque del catálogo **del TFM**, y el `data/catalog.yaml` de este taller
no lo contiene. Referencia de la tarea previa con 2 estados, para leer el orden de magnitud esperable:
Lehman 98,6 %, COVID 92,3 %, Inflación 78,9 %, y los puntos ciegos Taper Tantrum 2013 (10,9 %) y
Q4 2018 (20,6 %) (`docs/context/RESUMEN_DETECCION_REGIMENES.md:119-126`). Con $K=3$ se espera que
parte de 2013 y 2018 caiga en el estado intermedio: **eso es el resultado buscado**, no un fallo. 2013
es un punto ciego universal en el TFM —seis detectores independientes lo ignoran— porque fue un shock
de tipos sin estrés sistémico de renta variable
(`capa1_exploracion/memory/99_conclusions.md:229-246`); no se fuerza su captura.

**Limitaciones declaradas.** (1) Emisiones gaussianas con curtosis 25–40: el modelo está mal
especificado y subestima las colas; un HMM t-Student mejoraría el BIC con holgura, a costa de la
reproducibilidad. (2) La etiqueta se decodifica con suavizado: sale más limpia que la que vería un
operador en tiempo real (§6, §7.3). (3) La clase crisis descansa en pocos eventos efectivos: medido
con `regimenes.tramos_contiguos`, las ventanas de crisis forman **8 rachas en train, 2 en validación y
3 en test**, no varios cientos de observaciones independientes. De ahí que ninguna diferencia de pocos
puntos de *accuracy* entre modelos *downstream* sea estadísticamente distinguible
(`capa1_exploracion/memory/99_conclusions.md:394-405`), y que el umbral concreto para el
`recall_crisis` sea el de §7.4: unos 20 puntos.
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
`src/evaluation.py:32-56` (ventanas de crisis y trampas), `:141-262` (`walk_forward`). Datos —todas
estas rutas son del repositorio del TFM, no de este taller—:
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

**Configuración del taller.** Se cita por **nombre de bloque** de `data/catalog.yaml`, y no por número
de línea, para que la referencia no caduque cada vez que el fichero se reordena: bloque `periodo`
(ventana 2003-2026), `canales` (los 20 canales de $X$), `ventanas` (60/21/paso 1), `particiones`
(cortes y `embargo_sesiones: 85`), `regimenes` (método, $K$, semillas, `features_etiquetado`,
`features_descartadas` y `agregacion_horizonte`) y `escalado`. Código: `src/regimenes.py`
(`EtiquetadorRegimenes`, `regimen_dominante`, `control_bloqueante`, `diagnostico_features`,
`distribucion`, `tramos_contiguos`, `EPISODIOS_REFERENCIA`) y `src/evaluacion.py`
(`lineas_base`, `linea_base_persistencia`, `banda_bloques`).
Entorno: Python 3.13.7, `hmmlearn` 0.3.3, `scikit-learn` 1.8.0, `numpy` 2.3.4, `pandas` 2.3.3,
CPU-only.
