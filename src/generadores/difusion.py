"""Modelo de difusión condicional (DDIM) sobre el bloque tabular del taller.

Qué hace
--------
Aprende la distribución del bloque ``(n, d)`` (ventana aplanada + volatilidad
futura, ya escalada a media 0 / desviación 1) condicionada al régimen de
mercado, y muestrea bloques nuevos por difusión inversa determinista.

El proceso directo tiene forma cerrada, de modo que nunca hace falta simular la
cadena completa para entrenar::

    x_t = sqrt(alfa_barra_t) · x_0 + sqrt(1 - alfa_barra_t) · eps,   eps ~ N(0, I)

La red aprende a predecir el ``eps`` que se inyectó (parametrización
*epsilon-prediction*) minimizando un MSE. Esa pérdida es de las más limpias que
se pueden enseñar en el informe: empieza exactamente en 1.0 (la última capa se
inicializa a cero, así que la red predice el vector nulo y el MSE vale
``E[eps²] = 1``) y baja de forma monótona hasta estabilizarse. Conviene decirlo
con todas las letras en la memoria: **una pérdida baja no garantiza buenas
muestras**. El MSE de ``eps`` mide lo bien que la red resuelve un problema de
regresión promediado sobre todos los niveles de ruido; la calidad del generador
se juzga con las métricas de los notebooks 12-14 (distancias de distribución,
utilidad downstream, riesgo de memorización), no con esta curva.

Por qué un MLP y no una U-Net
-----------------------------
El notebook de clase (`5_Difussion_Models_Keras_DDIM_Jordi_mod.ipynb`) trabaja
con imágenes de 64x64x3 y usa una U-Net residual: las convoluciones y las
conexiones de salto explotan la localidad y la invarianza a traslación de una
imagen. Aquí la muestra es un vector de 1201 componentes que mezcla 60 pasos
temporales de 20 canales heterogéneos con la volatilidad futura pegada al
final; no hay una rejilla 2D sobre la que convolucionar, y una convolución 1D a
lo largo del eje temporal obligaría a desaplanar el bloque y rompería el
contrato de `ventanas.empaquetar()`. Un MLP residual con condicionamiento FiLM
captura las dependencias globales que aquí importan (correlaciones entre
canales y entre horizontes) y, con ~1,4 M de parámetros, cabe en CPU. Una U-Net
equivalente costaría un orden de magnitud más de tiempo sin ganar nada.

Por qué DDIM con 50 pasos y no DDPM con 1000
--------------------------------------------
El muestreo es la parte cara: cada paso es una pasada completa de la red sobre
todo el lote. DDPM exige recorrer los mismos ``T`` pasos con los que se entrenó;
DDIM define un proceso no markoviano que comparte los marginales del directo y
permite saltarse pasos, de modo que se puede muestrear sobre una subsecuencia de
50 de los 1000. En CPU eso es la diferencia entre minutos y horas: generar 2000
bloques cuesta ~1,5 min con 50 pasos y pasaría de media hora con los 1000 de
DDPM, y habría que pagarlo en cada figura y en cada barrido del notebook 14. Con
`guiado = 1.0` (por defecto) el recorrido es además determinista dado el ruido
inicial, lo que permite reproducir exactamente cualquier muestra.

Coste en CPU: leer esto ANTES de lanzar el entrenamiento
-------------------------------------------------------
Este es el generador más caro de los siete y conviene saber a qué se enfrenta
uno. Cifras medidas sobre un portátil de 4 CPU lógicas con torch 2.11+cpu (2
hilos) y sin CUDA, con d = 1201 y 4000 muestras de train:

- Entrenamiento con los valores por defecto: **4-7 s por época**, es decir
  **20-35 minutos** para las 300 épocas. El rango depende de lo cargada que esté
  la máquina; en un equipo con más núcleos baja de forma apreciable.
- Muestreo de 2000 bloques con 50 pasos DDIM: **~30 s**. Se duplica si se activa
  el guiado, que obliga a una segunda pasada incondicional en cada paso.
- Es viable en CPU: no hace falta recortar el bloque ni reducir el dataset.

Esos números salen de tres decisiones que conviene no deshacer sin medir,
porque cada una vale un factor grande:

1. **Lotes grandes.** Con la misma red, entrenar con lotes de 128 rinde 76
   muestras/s y con lotes de 1024 rinde 1095: catorce veces más. El cuello de
   botella no son los FLOP (las matrices son pequeñas) sino el coste fijo por
   operación del backend en modo ansioso, que se amortiza sobre el lote. De ahí
   que el defecto sea 512 y no el 32-128 habitual en GPU. Bajarlo "para que
   aprenda mejor" convierte 30 minutos en más de tres horas sin ganancia alguna.
2. **Bucle de entrenamiento manual** en lugar de `fit(epochs=1)` por época: con
   solo ~8 lotes por época la maquinaria de `fit` se lleva un 40 % del tiempo.
3. **Red pequeña**: 1,4 M de parámetros. Subir el ancho de 256 a 384 y el
   contexto de 192 a 256 la deja en 2,5 M y multiplica por ~1,7 el coste, sin
   evidencia de que compense con solo 4000 muestras.

Corolario práctico: entrenar una vez, `guardar()`, y que el resto del grupo haga
`cargar()`. Reentrenar por costumbre en cada ejecución del notebook no tiene
sentido con estos tiempos.

Si aun así resulta inviable (máquina más lenta, o hay que barrer
hiperparámetros), el parámetro ``n_componentes`` entrena la difusión sobre una
proyección PCA blanqueada del bloque (p. ej. 128 componentes) y reproyecta al
espacio original al generar; el coste baja aproximadamente un orden de magnitud.
Viene **desactivado por defecto** porque la PCA es una compresión lineal:
recorta la cola no gaussiana y las interacciones de orden superior que son justo
lo interesante del régimen de crisis. Úsese como plan B consciente y dígase en
la memoria, no como configuración normal.

Referencias
-----------
Ho et al. (2020), *Denoising Diffusion Probabilistic Models*.
Song et al. (2021), *Denoising Diffusion Implicit Models*.
Nichol & Dhariwal (2021), *Improved DDPM* (planificador coseno).
Ho & Salimans (2022), *Classifier-Free Diffusion Guidance*.
"""

from __future__ import annotations

import math
import pickle
import time
from pathlib import Path

import keras
import numpy as np
from keras import layers, ops

from .base import GeneradorSintetico

# ─────────────────────────────────────────────────────────────────────────────
# Planificadores de ruido
#
# Ambos devuelven el vector ``alfa_barra`` de longitud T, donde el índice k
# corresponde al paso de difusión k: alfa_barra[0] ~ 1 (casi sin ruido) y
# alfa_barra[T-1] ~ 0 (ruido puro).
# ─────────────────────────────────────────────────────────────────────────────


def planificador_coseno(pasos: int, desplazamiento: float = 0.008) -> np.ndarray:
    """Planificador coseno de Nichol & Dhariwal (2021).

    ``alfa_barra(t) ∝ cos²((t/T + s)/(1 + s) · π/2)``, normalizado para que
    ``alfa_barra(0) = 1``. El desplazamiento ``s`` evita que el primer paso ya
    destruya señal, que es lo que hace inestable el entrenamiento al principio.

    Es el que se usa por defecto: en dimensión baja la relación señal-ruido cae
    mucho más deprisa que en una imagen, porque hay menos componentes entre las
    que repartir la información. El coseno mantiene señal aprovechable durante
    más pasos y evita que la mitad final de la cadena sea ruido indistinguible.
    """
    t = np.arange(pasos + 1, dtype=np.float64) / pasos
    f = np.cos((t + desplazamiento) / (1.0 + desplazamiento) * math.pi / 2.0) ** 2
    alfa_barra = f / f[0]

    # Se pasa por betas para poder acotarlas: sin el recorte, los últimos pasos
    # tienen beta -> 1 y la varianza numérica se dispara.
    betas = np.clip(1.0 - alfa_barra[1:] / alfa_barra[:-1], 1e-8, 0.999)
    return np.cumprod(1.0 - betas)


def planificador_lineal(
    pasos: int, beta_minima: float = 1e-4, beta_maxima: float = 0.02
) -> np.ndarray:
    """Planificador lineal original de Ho et al. (2020).

    Se deja como opción para poder comparar en el informe. La diferencia
    práctica: el lineal se calibró sobre imágenes de 32x32 y destruye la señal
    demasiado pronto: con T = 1000 el último ~20 % de los pasos es ruido
    prácticamente puro, capacidad de red desperdiciada en un tramo donde no hay
    nada que aprender. En un bloque de 1201 componentes el efecto se nota más
    todavía, y se traduce en muestras con la estructura temporal desdibujada.
    """
    betas = np.linspace(beta_minima, beta_maxima, pasos, dtype=np.float64)
    return np.cumprod(1.0 - betas)


PLANIFICADORES = {"coseno": planificador_coseno, "lineal": planificador_lineal}


# ─────────────────────────────────────────────────────────────────────────────
# Incrustación sinusoidal del paso temporal
# ─────────────────────────────────────────────────────────────────────────────


@keras.saving.register_keras_serializable(package="taller_b5t1")
class IncrustacionSinusoidal(layers.Layer):
    """Codifica el instante de difusión en una base de senos y cosenos.

    Es la misma idea que las codificaciones posicionales de los transformers y
    la que usa el notebook de clase. Un escalar normalizado entrando directo en
    un Dense daría una respuesta casi lineal en ``t``, y la red necesita ser muy
    sensible al nivel de ruido: el mismo vector de entrada significa cosas
    distintas según si queda mucho o poco ruido por quitar.

    Parameters
    ----------
    dimension
        Tamaño del vector de salida. Debe ser par: mitad senos, mitad cosenos.
    frecuencia_maxima
        Frecuencia angular más alta de la base. Controla la resolución temporal
        más fina que la red puede distinguir.
    """

    def __init__(
        self, dimension: int = 128, frecuencia_maxima: float = 1000.0, **kwargs
    ) -> None:
        super().__init__(**kwargs)
        if dimension % 2 != 0:
            raise ValueError(f"La dimensión debe ser par, recibido {dimension}.")
        self.dimension = dimension
        self.frecuencia_maxima = frecuencia_maxima

    def call(self, tiempos):
        frecuencias = ops.exp(
            ops.linspace(0.0, math.log(self.frecuencia_maxima), self.dimension // 2)
        )
        velocidades = ops.cast(2.0 * math.pi * frecuencias, "float32")
        angulos = velocidades * ops.cast(tiempos, "float32")
        return ops.concatenate([ops.sin(angulos), ops.cos(angulos)], axis=-1)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.dimension)

    def get_config(self) -> dict:
        return {
            **super().get_config(),
            "dimension": self.dimension,
            "frecuencia_maxima": self.frecuencia_maxima,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Generador
# ─────────────────────────────────────────────────────────────────────────────


class DifusionCondicional(GeneradorSintetico):
    """Difusión condicional con muestreo DDIM y *classifier-free guidance*.

    Parameters
    ----------
    n_regimenes
        Número de regímenes de mercado. La red usa ``n_regimenes + 1`` clases:
        la extra es la etiqueta nula que hace posible el guiado sin clasificador.
    semilla
        Semilla de la fuente de aleatoriedad del generador.
    pasos_difusion
        ``T``, número de pasos del proceso directo con el que se entrena.
    planificador
        ``"coseno"`` (por defecto) o ``"lineal"``. Ver los planificadores.
    n_pasos_muestreo
        Pasos de la subsecuencia DDIM al generar. 50 sobre 1000 es el compromiso
        habitual; bajar de ~20 empieza a sesgar visiblemente la distribución.
    guiado
        Escala del *classifier-free guidance*. ``1.0`` = sin guiado, la
        predicción condicional se usa tal cual. Ver `_generate` para el
        compromiso fidelidad/diversidad.
    prob_dropout_condicion
        Fracción de ejemplos que se entrenan con la etiqueta nula. Sin esto la
        red nunca aprende el modelo incondicional y el guiado no es aplicable.
    ancho, n_bloques, dim_contexto, dim_embedding_tiempo
        Tamaño de la red. Los valores por defecto dan ~1,4 M de parámetros con
        ``d = 1201``, que es lo que cabe en CPU en un tiempo razonable.
    epocas, tam_lote, tasa_aprendizaje, decaimiento_pesos
        Hiperparámetros de optimización. Se pueden sobreescribir en `fit`.
        El lote por defecto (512) es grande a propósito: ver la nota sobre el
        coste en CPU en el docstring del módulo. La tasa de aprendizaje va
        acompasada a ese lote, así que bajar uno sin bajar la otra deja el
        entrenamiento inestable.
    recorte_x0
        Acota la estimación de ``x_0`` en cada paso de muestreo, en unidades de
        desviación típica. Los primeros pasos dividen por ``sqrt(alfa_barra)``,
        que es diminuto, así que cualquier error de la red se amplifica y puede
        producir infinitos. El valor por defecto (10) es deliberadamente
        generoso: recortar más apretado amputaría las colas, que es justo lo que
        aporta el régimen de crisis. ``None`` desactiva el recorte.
    n_componentes
        Si no es ``None``, entrena sobre una proyección PCA blanqueada del
        bloque con ese número de componentes. Ver el docstring del módulo.
    verboso
        ``0`` silencio, ``1`` una línea de progreso cada 10 % de las épocas.
    """

    nombre = "difusion"
    etiqueta = "Difusión (DDIM) condicional"
    tiene_curva_perdida = True

    def __init__(
        self,
        n_regimenes: int,
        semilla: int = 42,
        *,
        pasos_difusion: int = 1000,
        planificador: str = "coseno",
        n_pasos_muestreo: int = 50,
        guiado: float = 1.0,
        prob_dropout_condicion: float = 0.1,
        ancho: int = 256,
        n_bloques: int = 3,
        dim_contexto: int = 192,
        dim_embedding_tiempo: int = 128,
        epocas: int = 300,
        tam_lote: int = 512,
        tasa_aprendizaje: float = 2e-3,
        decaimiento_pesos: float = 1e-4,
        recorte_x0: float | None = 10.0,
        n_componentes: int | None = None,
        verboso: int = 1,
    ) -> None:
        super().__init__(n_regimenes=n_regimenes, semilla=semilla)

        if planificador not in PLANIFICADORES:
            raise ValueError(
                f"Planificador {planificador!r} desconocido. "
                f"Opciones: {sorted(PLANIFICADORES)}."
            )
        if not 0.0 <= prob_dropout_condicion < 1.0:
            raise ValueError("prob_dropout_condicion debe estar en [0, 1).")
        if n_pasos_muestreo > pasos_difusion:
            raise ValueError(
                "No tiene sentido muestrear con más pasos de los entrenados "
                f"({n_pasos_muestreo} > {pasos_difusion})."
            )

        self.pasos_difusion = pasos_difusion
        self.planificador = planificador
        self.n_pasos_muestreo = n_pasos_muestreo
        self.guiado = guiado
        self.prob_dropout_condicion = prob_dropout_condicion

        self.ancho = ancho
        self.n_bloques = n_bloques
        self.dim_contexto = dim_contexto
        self.dim_embedding_tiempo = dim_embedding_tiempo

        self.epocas = epocas
        self.tam_lote = tam_lote
        self.tasa_aprendizaje = tasa_aprendizaje
        self.decaimiento_pesos = decaimiento_pesos
        self.recorte_x0 = recorte_x0
        self.n_componentes = n_componentes
        self.verboso = verboso

        #: Vector alfa_barra del proceso directo, precalculado una sola vez.
        self.alfa_barra: np.ndarray = np.clip(
            PLANIFICADORES[planificador](pasos_difusion), 1e-5, 1.0
        ).astype(np.float64)

        self.modelo: keras.Model | None = None
        self.pca = None
        #: Dimensión en la que trabaja la red: ``d`` o ``n_componentes``.
        self.dim_modelo: int | None = None

    # ── Índice de la clase nula del guiado sin clasificador ─────────────────

    @property
    def _clase_nula(self) -> int:
        return self.n_regimenes

    # ── Arquitectura ────────────────────────────────────────────────────────

    def _bloque_residual(self, h, contexto):
        """Bloque residual con condicionamiento FiLM.

        El contexto (tiempo + régimen) no se concatena a la entrada sino que
        modula la activación normalizada mediante una escala y un desplazamiento
        aprendidos. Concatenar solo al principio diluye la condición conforme se
        avanza en profundidad; FiLM la reinyecta en cada bloque, que es lo que
        hace que el régimen se note de verdad en las muestras.

        La escala arranca en 1 y el desplazamiento en 0 (núcleos a cero, sesgo a
        uno), de forma que el bloque empieza siendo la identidad modulada y no
        perturba el entrenamiento inicial.
        """
        z = layers.LayerNormalization()(h)
        escala = layers.Dense(
            self.ancho, kernel_initializer="zeros", bias_initializer="ones"
        )(contexto)
        desplazamiento = layers.Dense(self.ancho, kernel_initializer="zeros")(contexto)
        z = layers.Multiply()([z, escala])
        z = layers.Add()([z, desplazamiento])

        # Swish en lugar de ReLU: el notebook de clase insiste en que las
        # activaciones no monótonas van claramente mejor en difusión.
        z = layers.Dense(self.ancho, activation="swish")(z)
        z = layers.Dense(self.ancho)(z)
        return layers.Add()([h, z])

    def _construir_red(self, dimension: int) -> keras.Model:
        """Red de predicción de ruido: ``(x_t, t, condición) -> eps``."""
        entrada_x = keras.Input(shape=(dimension,), name="x_ruidoso")
        entrada_t = keras.Input(shape=(1,), name="tiempo")
        entrada_c = keras.Input(shape=(self.n_regimenes + 1,), name="condicion")

        contexto = IncrustacionSinusoidal(self.dim_embedding_tiempo)(entrada_t)
        contexto = layers.Dense(self.dim_contexto, activation="swish")(contexto)
        contexto = layers.Add()(
            [contexto, layers.Dense(self.dim_contexto)(entrada_c)]
        )
        contexto = layers.Dense(self.dim_contexto, activation="swish")(contexto)

        h = layers.Dense(self.ancho)(entrada_x)
        for _ in range(self.n_bloques):
            h = self._bloque_residual(h, contexto)
        h = layers.LayerNormalization()(h)

        # Núcleo a cero: la red predice el vector nulo en la primera iteración,
        # con lo que el MSE arranca exactamente en 1.0. Truco del notebook de
        # clase; además de estabilizar el inicio, da una curva de pérdida con
        # una referencia interpretable (1.0 = no ha aprendido nada).
        salida = layers.Dense(dimension, kernel_initializer="zeros", name="ruido")(h)

        return keras.Model(
            [entrada_x, entrada_t, entrada_c], salida, name="denoiser_mlp"
        )

    # ── Proyección opcional a espacio reducido ──────────────────────────────

    def _proyectar(self, bloque: np.ndarray, ajustar: bool = False) -> np.ndarray:
        """Lleva el bloque al espacio en el que trabaja la red."""
        if self.n_componentes is None:
            return np.ascontiguousarray(bloque, dtype=np.float32)

        if ajustar:
            from sklearn.decomposition import PCA

            # `whiten=True` deja las componentes con varianza 1, que es lo que
            # la difusión asume: el proceso directo mezcla con ruido N(0, I) y
            # si las componentes tuvieran escalas dispares la relación
            # señal-ruido efectiva sería distinta en cada una.
            self.pca = PCA(
                n_components=self.n_componentes, whiten=True, random_state=self.semilla
            )
            self.pca.fit(bloque)

        return self.pca.transform(bloque).astype(np.float32)

    def _reconstruir(self, latente: np.ndarray) -> np.ndarray:
        """Operación inversa de `_proyectar`."""
        if self.pca is None:
            return latente.astype(np.float32)
        return self.pca.inverse_transform(latente).astype(np.float32)

    # ── Condición ───────────────────────────────────────────────────────────

    def _codificar_condicion(
        self, y_reg: np.ndarray, con_dropout: bool
    ) -> np.ndarray:
        """One-hot del régimen con ``n_regimenes + 1`` posiciones.

        Con `con_dropout`, una fracción de los ejemplos se reetiqueta a la clase
        nula. Ese es el mecanismo del *classifier-free guidance*: una sola red
        aprende a la vez el modelo condicional y el incondicional, y en el
        muestreo se pueden combinar sus predicciones.
        """
        etiquetas = np.asarray(y_reg, dtype=np.int64).copy()

        if con_dropout and self.prob_dropout_condicion > 0.0:
            sin_etiqueta = self.rng.random(len(etiquetas)) < self.prob_dropout_condicion
            etiquetas[sin_etiqueta] = self._clase_nula

        codigo = np.zeros((len(etiquetas), self.n_regimenes + 1), dtype=np.float32)
        codigo[np.arange(len(etiquetas)), etiquetas] = 1.0
        return codigo

    def _condicion_fija(self, n: int, clase: int) -> np.ndarray:
        codigo = np.zeros((n, self.n_regimenes + 1), dtype=np.float32)
        codigo[:, clase] = 1.0
        return codigo

    # ── Entrenamiento ───────────────────────────────────────────────────────

    def _fit(
        self,
        bloque: np.ndarray,
        y_reg: np.ndarray,
        *,
        epocas: int | None = None,
        tam_lote: int | None = None,
        verboso: int | None = None,
        **_,
    ) -> None:
        epocas = self.epocas if epocas is None else epocas
        tam_lote = self.tam_lote if tam_lote is None else tam_lote
        verboso = self.verboso if verboso is None else verboso

        # Fija las semillas del backend para que la inicialización de pesos sea
        # reproducible. Tiene efecto global (numpy, random, torch): es
        # deliberado, es lo que hace comparables dos ejecuciones del notebook.
        keras.utils.set_random_seed(self.semilla)

        x0 = self._proyectar(bloque, ajustar=True)
        self.dim_modelo = x0.shape[1]
        n = len(x0)

        self.modelo = self._construir_red(self.dim_modelo)

        # Decaimiento coseno de la tasa de aprendizaje. No es cosmético: sin él
        # la pérdida se queda oscilando por el ruido de muestrear ``t`` y
        # ``eps`` en cada época, y la curva del informe no deja ver si el modelo
        # ha convergido o no. Con el decaimiento la meseta final es inequívoca.
        pasos_por_epoca = max(1, math.ceil(n / tam_lote))
        planificacion = keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=self.tasa_aprendizaje,
            decay_steps=epocas * pasos_por_epoca,
            alpha=0.05,
        )
        self.modelo.compile(
            optimizer=keras.optimizers.AdamW(
                learning_rate=planificacion, weight_decay=self.decaimiento_pesos
            ),
            loss="mse",
        )

        if verboso:
            print(
                f"{self.etiqueta}: {self.modelo.count_params():,} parámetros, "
                f"{n} muestras, d={self.dim_modelo}, "
                f"{epocas} épocas x {pasos_por_epoca} pasos."
            )

        cada = max(1, epocas // 10)
        inicio = time.perf_counter()

        for epoca in range(1, epocas + 1):
            # Se remuestrean el instante y el ruido en cada época. Cada bloque
            # real se ve así con un nivel de ruido distinto en cada pasada, que
            # es la forma barata de cubrir los T niveles con solo ~4000 muestras.
            pasos = self.rng.integers(0, self.pasos_difusion, size=n)
            ruido = self.rng.standard_normal((n, self.dim_modelo)).astype(np.float32)

            alfa = self.alfa_barra[pasos].astype(np.float32)[:, None]
            x_ruidoso = np.sqrt(alfa) * x0 + np.sqrt(1.0 - alfa) * ruido
            tiempos = ((pasos + 1) / self.pasos_difusion).astype(np.float32)[:, None]
            condicion = self._codificar_condicion(y_reg, con_dropout=True)

            # Bucle manual en lugar de `fit(epochs=1)`: con solo ~8 lotes por
            # época, la maquinaria de `fit` (iterador, callbacks, métricas) pesa
            # tanto como el cálculo y se lleva un 40 % del tiempo de pared.
            orden = self.rng.permutation(n)
            acumulado = 0.0
            for i in range(0, n, tam_lote):
                corte = orden[i : i + tam_lote]
                perdida_lote = self.modelo.train_on_batch(
                    [x_ruidoso[corte], tiempos[corte], condicion[corte]],
                    ruido[corte],
                )
                # Devuelve un escalar al no haber métricas compiladas, pero se
                # normaliza por si alguien añade alguna más adelante.
                if isinstance(perdida_lote, (list, tuple)):
                    perdida_lote = perdida_lote[0]
                acumulado += float(perdida_lote) * len(corte)

            perdida = acumulado / n
            self.registrar(epoca=epoca, perdida=perdida)

            if verboso and (epoca % cada == 0 or epoca == epocas):
                transcurrido = time.perf_counter() - inicio
                print(
                    f"  época {epoca:>4}/{epocas}  mse_eps={perdida:.4f}  "
                    f"({transcurrido:.0f} s, {transcurrido / epoca:.2f} s/época)"
                )

    # ── Muestreo ────────────────────────────────────────────────────────────

    def _subsecuencia_ddim(self) -> np.ndarray:
        """Pasos de difusión visitados al muestrear, en orden descendente.

        Rejilla uniforme sobre ``[0, T-1]``. DDIM permite este salto porque su
        proceso inverso solo depende de los marginales ``q(x_t | x_0)``, que son
        los mismos para cualquier subsecuencia; DDPM, en cambio, está atado a la
        cadena de Markov con la que se entrenó.
        """
        rejilla = np.linspace(0, self.pasos_difusion - 1, self.n_pasos_muestreo)
        return np.unique(np.round(rejilla).astype(int))[::-1]

    def _predecir_ruido(
        self,
        x: np.ndarray,
        tiempos: np.ndarray,
        condicion: np.ndarray,
        tam_lote: int,
    ) -> np.ndarray:
        """Pasada directa por trozos, para no reservar lotes enormes en CPU."""
        salidas = []
        for i in range(0, len(x), tam_lote):
            corte = slice(i, i + tam_lote)
            salidas.append(
                self.modelo.predict_on_batch(
                    [x[corte], tiempos[corte], condicion[corte]]
                )
            )
        return np.concatenate(salidas).astype(np.float32)

    def _generate(self, n: int, regimen: int) -> np.ndarray:
        """Difusión inversa determinista (DDIM) partiendo de ruido gaussiano.

        En cada paso la red predice ``eps``, de ahí se despeja la estimación de
        la muestra limpia y se recombina con el nivel de ruido del paso
        siguiente::

            x_0_est = (x_t - sqrt(1 - a_t) · eps) / sqrt(a_t)
            x_prev  = sqrt(a_prev) · x_0_est + sqrt(1 - a_prev) · eps

        No se inyecta ruido nuevo (``sigma = 0``): el recorrido es una ODE, así
        que el resultado queda determinado por el ruido inicial. Eso permite
        reproducir exactamente una muestra concreta y, si hiciera falta,
        interpolar entre dos.

        Sobre el guiado: ``eps = eps_nulo + guiado · (eps_cond - eps_nulo)``.
        Con ``guiado = 1.0`` se recupera la predicción condicional pura y se
        ahorra la mitad del cómputo, porque no hace falta la pasada
        incondicional. Subirlo empuja las muestras hacia el centro de masa del
        régimen: se parecen más al prototipo de "crisis" pero se parecen más
        entre sí. Para aumentar datos eso es contraproducente — el valor de una
        muestra sintética está en la variabilidad que aporta, no en repetir la
        media de la clase. Por eso el defecto es 1.0; valores por encima de ~2
        solo tienen sentido si el diagnóstico muestra que las muestras no
        respetan la condición.
        """
        indices = self._subsecuencia_ddim()
        # Lote de inferencia grande por el mismo motivo que el de entrenamiento:
        # el coste en CPU está dominado por la sobrecarga fija por operación, no
        # por los FLOP. 1024 x 1201 flotantes son 5 MB, no hay problema de
        # memoria y la pasada directa va casi el doble de rápida que con 512.
        tam_lote = max(self.tam_lote, 1024)

        x = self.rng.standard_normal((n, self.dim_modelo)).astype(np.float32)
        condicion = self._condicion_fija(n, regimen)
        con_guiado = abs(self.guiado - 1.0) > 1e-8
        nula = self._condicion_fija(n, self._clase_nula) if con_guiado else None

        for i, paso in enumerate(indices):
            alfa = float(self.alfa_barra[paso])
            # En el último paso el destino es t=0: señal pura, ruido nulo.
            alfa_previa = (
                float(self.alfa_barra[indices[i + 1]]) if i + 1 < len(indices) else 1.0
            )
            tiempos = np.full(
                (n, 1), (paso + 1) / self.pasos_difusion, dtype=np.float32
            )

            eps = self._predecir_ruido(x, tiempos, condicion, tam_lote)
            if con_guiado:
                eps_nulo = self._predecir_ruido(x, tiempos, nula, tam_lote)
                eps = eps_nulo + self.guiado * (eps - eps_nulo)

            x_limpio = (x - math.sqrt(1.0 - alfa) * eps) / math.sqrt(alfa)
            if self.recorte_x0 is not None:
                np.clip(x_limpio, -self.recorte_x0, self.recorte_x0, out=x_limpio)

            x = (
                math.sqrt(alfa_previa) * x_limpio
                + math.sqrt(1.0 - alfa_previa) * eps
            ).astype(np.float32)

        return self._reconstruir(x)

    # ── Persistencia ────────────────────────────────────────────────────────

    def _guardar(self, destino: Path) -> None:
        """Pesos en formato nativo de Keras, resto del estado en pickle.

        El ``.keras`` guarda arquitectura y pesos y se recarga sin necesidad de
        reconstruir la red a mano. Lo que no cabe ahí (planificador de ruido,
        hiperparámetros de muestreo y la PCA opcional de scikit-learn) va aparte.
        """
        self.modelo.save(str(destino / "modelo.keras"))

        estado = {
            "pasos_difusion": self.pasos_difusion,
            "planificador": self.planificador,
            "alfa_barra": self.alfa_barra,
            "n_pasos_muestreo": self.n_pasos_muestreo,
            "guiado": self.guiado,
            "prob_dropout_condicion": self.prob_dropout_condicion,
            "recorte_x0": self.recorte_x0,
            "n_componentes": self.n_componentes,
            "dim_modelo": self.dim_modelo,
            "tam_lote": self.tam_lote,
            "pca": self.pca,
        }
        with open(destino / "estado.pkl", "wb") as f:
            pickle.dump(estado, f)

    def _cargar(self, origen: Path) -> None:
        with open(origen / "estado.pkl", "rb") as f:
            estado = pickle.load(f)

        self.pasos_difusion = estado["pasos_difusion"]
        self.planificador = estado["planificador"]
        self.alfa_barra = estado["alfa_barra"]
        self.n_pasos_muestreo = estado["n_pasos_muestreo"]
        self.guiado = estado["guiado"]
        self.prob_dropout_condicion = estado["prob_dropout_condicion"]
        self.recorte_x0 = estado["recorte_x0"]
        self.n_componentes = estado["n_componentes"]
        self.dim_modelo = estado["dim_modelo"]
        self.tam_lote = estado["tam_lote"]
        self.pca = estado["pca"]

        self.modelo = keras.saving.load_model(str(origen / "modelo.keras"))
