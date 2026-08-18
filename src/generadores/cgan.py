"""GAN condicional (cGAN) sobre el bloque de ventanas del taller.

Es el generador que justifica el diseño condicional del proyecto: al recibir el
régimen como **entrada** y no como salida, la etiqueta de cada muestra sintética
es exacta por construcción y la mezcla de clases pasa a ser una decisión de
diseño. Con la GAN del bloque conjunto ``[X ‖ y]`` de
`material_clase/notebooks/GAN_1_..._etiquetas_y_balanceo.ipynb` la proporción de
clases generadas reproduce la del entrenamiento, incluido su desequilibrio, y
entonces no hay forma de sobre-representar la crisis, que es exactamente lo que
el taller necesita.

Arquitectura
------------
Dos MLP condicionales (Mirza & Osindero, 2014) con concatenación de la etiqueta
*one-hot* en la entrada de **ambas** redes. Condicionar sólo al generador no
sirve: si el discriminador no ve la etiqueta, no puede penalizar que la muestra
no corresponda al régimen pedido, y el condicionamiento se vuelve decorativo.

Cómo se lee la convergencia de este modelo
------------------------------------------
El enunciado pide "curvas de loss donde se vea que el modelo ha convergido".
En una GAN esa frase necesita traducción, porque **la pérdida no mide calidad y
su óptimo no es cero**. Con discriminador óptimo el juego minimax equivale a
minimizar la divergencia de Jensen-Shannon, cuyo óptimo es ``p_g = p_datos``; en
ese punto ``D ≡ 0,5`` y ambas entropías cruzadas valen ``log 2 ≈ 0,693``. Es
decir:

* ``perdida_disc → 0`` **no** es éxito: significa que el discriminador ha ganado
  y el generador ha dejado de recibir gradiente útil.
* Lo que hay que enseñar es el **equilibrio**: ``acc_disc_real`` y
  ``acc_disc_falso`` oscilando en torno a ``0,5`` (banda sana 0,50-0,75) y las
  dos pérdidas orbitando ``0,693`` sin deriva monótona.
* Además, la pérdida es **no estacionaria**: ``perdida_gen`` en la época 10 y en
  la 300 se han medido contra discriminadores distintos, así que comparar esos
  dos números no significa nada por sí solo.

Por eso el historial registra, junto a las dos pérdidas, las precisiones y las
salidas medias del discriminador (las que dicen si el juego sigue vivo) y la
dispersión del lote generado (la que delata el *mode collapse*).

Notas de implementación
-----------------------
Keras 3 con backend PyTorch. El bucle de entrenamiento es explícito en lugar de
``train_on_batch`` sobre un modelo apilado: el patrón apilado obliga a congelar
el discriminador con ``trainable = False`` antes de compilar, y los cuadernos de
clase se olvidan de hacerlo, con lo que el paso del generador sabotea al
discriminador en la misma iteración. Con dos optimizadores separados y un
``backward()`` por red el problema desaparece porque cada optimizador sólo toca
sus propias variables, y de paso el coste por iteración es visible y medible.
"""

from __future__ import annotations

from pathlib import Path

import keras
import numpy as np
import pandas as pd

# El backend está fijado a PyTorch en `src/__init__.py`. Keras 3 no expone una
# API de gradientes agnóstica del backend, así que el bucle usa `backward()` de
# torch directamente: es el patrón documentado por Keras para este backend.
import torch

from .base import GeneradorSintetico

#: Hiperparámetros que `fit(**kwargs)` puede sobreescribir sobre los del `__init__`.
_AJUSTABLES = frozenset(
    {
        "dim_latente",
        "epocas",
        "tam_lote",
        "tasa_aprendizaje",
        "capas_gen",
        "capas_disc",
        "dropout",
        "momento_bn",
        "suavizado",
        "umbral_disc",
        "equilibrar_clases",
        "hilos_torch",
        "verboso",
    }
)


# ─────────────────────────────────────────────────────────────────────────────
# Construcción de las redes
# ─────────────────────────────────────────────────────────────────────────────


def _construir_generador(
    dimension: int,
    n_regimenes: int,
    dim_latente: int,
    capas: tuple[int, ...],
    momento_bn: float,
) -> keras.Model:
    """``G(z | y)``: ruido gaussiano + régimen *one-hot* → bloque completo.

    La salida es **lineal**, no ``tanh``. Los cuadernos de clase usan ``tanh``
    porque sus datos van escalados a ``[-1, 1]``; aquí el bloque está
    estandarizado a media 0 y desviación 1, y una ``tanh`` recortaría todo lo
    que caiga fuera de ``[-1, 1]`` — es decir, precisamente las colas de los
    retornos y los picos de volatilidad que definen el régimen de crisis.
    """
    z = keras.Input(shape=(dim_latente,), name="z")
    condicion = keras.Input(shape=(n_regimenes,), name="condicion")
    h = keras.layers.Concatenate(name="entrada_condicionada")([z, condicion])

    for i, unidades in enumerate(capas):
        h = keras.layers.Dense(unidades, name=f"densa_{i}")(h)
        # LeakyReLU en lugar de ReLU: una ReLU muerta en G corta el gradiente que
        # llega desde el discriminador y esa parte de la red deja de aprender.
        h = keras.layers.LeakyReLU(negative_slope=0.2, name=f"lrelu_{i}")(h)
        # BatchNorm sólo en el generador. En el discriminador mezclaría en un
        # mismo lote estadísticas de muestras reales y falsas, filtrando
        # información entre ellas y haciendo trampas al juego.
        h = keras.layers.BatchNormalization(momentum=momento_bn, name=f"bn_{i}")(h)

    salida = keras.layers.Dense(dimension, name="bloque")(h)
    return keras.Model([z, condicion], salida, name="generador")


def _construir_discriminador(
    dimension: int,
    n_regimenes: int,
    capas: tuple[int, ...],
    dropout: float,
) -> keras.Model:
    """``D(x | y)``: bloque + régimen *one-hot* → probabilidad de ser real.

    Se dimensiona deliberadamente por debajo del generador. Un discriminador
    sobrado gana la partida enseguida, sus salidas saturan y el gradiente que
    llega a ``G`` se desvanece; el `Dropout` va en la misma dirección.
    """
    bloque = keras.Input(shape=(dimension,), name="bloque")
    condicion = keras.Input(shape=(n_regimenes,), name="condicion")
    h = keras.layers.Concatenate(name="entrada_condicionada")([bloque, condicion])

    for i, unidades in enumerate(capas):
        h = keras.layers.Dense(unidades, name=f"densa_{i}")(h)
        h = keras.layers.LeakyReLU(negative_slope=0.2, name=f"lrelu_{i}")(h)
        h = keras.layers.Dropout(dropout, name=f"dropout_{i}")(h)

    salida = keras.layers.Dense(1, activation="sigmoid", name="veredicto")(h)
    return keras.Model([bloque, condicion], salida, name="discriminador")


# ─────────────────────────────────────────────────────────────────────────────
# Utilidades del bucle adversarial
# ─────────────────────────────────────────────────────────────────────────────


def _una_caliente(etiquetas: np.ndarray, n_regimenes: int) -> np.ndarray:
    """Codificación *one-hot* del régimen, la forma en que entra a las dos redes."""
    matriz = np.zeros((len(etiquetas), n_regimenes), dtype=np.float32)
    matriz[np.arange(len(etiquetas)), etiquetas] = 1.0
    return matriz


def _entropia_cruzada(prediccion, objetivo: float):
    """Entropía cruzada binaria contra una etiqueta constante.

    `keras.ops.binary_crossentropy` recorta internamente la predicción, de modo
    que un discriminador muy confiado no produce ``log(0)``.
    """
    diana = keras.ops.full_like(prediccion, objetivo)
    return keras.ops.mean(keras.ops.binary_crossentropy(diana, prediccion))


def _paso_descenso(modelo: keras.Model, optimizador: keras.optimizers.Optimizer, perdida) -> None:
    """Retropropaga `perdida` y actualiza **sólo** las variables de `modelo`.

    Sustituye al truco del modelo apilado con ``trainable = False``: aquí el
    aislamiento entre redes es explícito y no depende del momento en que se
    compiló nada.
    """
    perdida.backward()
    gradientes = [variable.value.grad for variable in modelo.trainable_weights]
    with torch.no_grad():
        optimizador.apply(gradientes, modelo.trainable_weights)


def _valor(tensor) -> float:
    """Extrae un escalar de Python de un tensor del backend."""
    return float(keras.ops.convert_to_numpy(tensor))


# ─────────────────────────────────────────────────────────────────────────────
# Generador sintético del taller
#
# "Generador" está sobrecargado en este módulo: aquí abajo designa la clase que
# cumple el contrato de `base.py`; arriba, la red G del juego adversarial.
# ─────────────────────────────────────────────────────────────────────────────


class CGAN(GeneradorSintetico):
    """GAN condicional MLP sobre el bloque ``[X aplanada ‖ y_vol]``.

    Parameters
    ----------
    n_regimenes
        Número de clases de la condición (3 en el catálogo: calma, transición,
        crisis).
    semilla
        Semilla del muestreo de ruido y de la inicialización de pesos.
    dim_latente
        Dimensión de ``z``. 128 es holgado para ``d ≈ 1200``: subirlo no aporta
        y hace el espacio latente más difícil de recorrer.
    capas_gen, capas_disc
        Unidades de las capas ocultas. Reproducen la estructura del cuaderno de
        clase ``GAN_1_Really_Simple_GAN_MNIST.ipynb`` (``256 → 512`` en G,
        ``128 → 64`` en D), con la misma asimetría deliberada de capacidad: el
        discriminador tiene menos de la mitad de parámetros que el generador.
        Es además donde se decide el coste en CPU, porque el discriminador se
        evalúa tres veces por iteración y el generador una.
    dropout
        Regularización del discriminador.
    epocas, tam_lote, tasa_aprendizaje
        Presupuesto de entrenamiento. Todos ellos, y el resto de
        hiperparámetros, son sobreescribibles desde ``fit(**kwargs)``.

        El lote es grande a propósito. Medido en esta máquina (4 núcleos, sin
        CUDA) el coste por iteración apenas se mueve entre ``tam_lote = 128`` y
        ``tam_lote = 512`` — 137 ms frente a 195 ms — porque con lotes pequeños las
        multiplicaciones no llenan el BLAS y domina el coste fijo por operación.
        Como el número de pasos por época sí baja con el lote, pasar de 128 a
        512 divide el tiempo por época por tres (4,3 s → 1,4 s). Con ~4.000
        ventanas de train, el ajuste por defecto son 7 pasos por época, ~2.100
        pasos de gradiente y 1,7 s por época medidos de extremo a extremo: unos
        **8 minutos** de entrenamiento completo. De regalo, los lotes grandes
        amortiguan las dos patologías con las que pelea una GAN: el colapso de
        modo y la oscilación en torno al punto de silla.
    suavizado
        *One-sided label smoothing*: la etiqueta de "real" es 0,9 en vez de 1,0.
        Impide que el discriminador empuje sus logits a infinito y mantenga
        gradiente vivo hacia el generador.
    umbral_disc
        Banda muerta: si la precisión media móvil del discriminador supera este
        valor, su actualización se salta esa iteración. Es la versión sensata
        del ``ratio`` adaptativo de `Taller_GANs.ipynb`, que modulaba el tamaño
        de lote (rompiendo BatchNorm y el paso efectivo de Adam) en vez de la
        frecuencia de actualización, que es el grado de libertad ``k`` del
        algoritmo original.
    equilibrar_clases
        Si ``True``, cada lote se muestrea con cuota fija por régimen. Con la
        proporción real (~10% de crisis) el generador dedicaría una décima parte
        de su gradiente a ``p(x | crisis)``, que es justo la condicional que el
        taller quiere sobre-representar. Las etiquetas de las muestras
        falsas se toman del mismo vector que las reales, de modo que la marginal
        de ``y`` es idéntica a ambos lados y el discriminador no puede usar la
        frecuencia de la clase como atajo para detectar falsificaciones.
    hilos_torch
        Si se indica, fija ``torch.set_num_threads``. Sin CUDA el entrenamiento
        es puro BLAS de CPU y torch arranca aquí con 2 hilos sobre 4 núcleos.
    verboso
        Traza una línea de diagnóstico cada 10% de las épocas.

    Notes
    -----
    El diagnóstico de convergencia de este modelo se lee en el equilibrio, no en
    una pérdida decreciente; ver el docstring del módulo.
    """

    nombre = "cgan"
    etiqueta = "GAN condicional"
    tiene_curva_perdida = True

    def __init__(
        self,
        n_regimenes: int,
        semilla: int = 42,
        *,
        # ── redes ──
        dim_latente: int = 128,
        capas_gen: tuple[int, ...] = (256, 512),
        capas_disc: tuple[int, ...] = (128, 64),
        dropout: float = 0.3,
        momento_bn: float = 0.8,
        # ── optimización ──
        epocas: int = 300,
        tam_lote: int = 512,
        tasa_aprendizaje: float = 2e-4,
        suavizado: float = 0.9,
        umbral_disc: float = 0.80,
        equilibrar_clases: bool = True,
        # ── entorno ──
        hilos_torch: int | None = None,
        verboso: bool = True,
    ) -> None:
        super().__init__(n_regimenes=n_regimenes, semilla=semilla)

        self.dim_latente = dim_latente
        self.capas_gen = tuple(capas_gen)
        self.capas_disc = tuple(capas_disc)
        self.dropout = dropout
        self.momento_bn = momento_bn

        self.epocas = epocas
        self.tam_lote = tam_lote
        self.tasa_aprendizaje = tasa_aprendizaje
        self.suavizado = suavizado
        self.umbral_disc = umbral_disc
        self.equilibrar_clases = equilibrar_clases

        self.hilos_torch = hilos_torch
        self.verboso = verboso

        #: Las dos redes del juego. Se construyen en `_fit`, cuando ya se conoce
        #: la dimensión del bloque.
        self._gen: keras.Model | None = None
        self._disc: keras.Model | None = None

    # ── Ajuste ──────────────────────────────────────────────────────────────

    def _fit(self, bloque: np.ndarray, y_reg: np.ndarray, **kwargs) -> None:
        self._aplicar_ajustes(kwargs)
        self.historial = pd.DataFrame()

        if self.hilos_torch is not None:
            torch.set_num_threads(self.hilos_torch)

        # Reproducibilidad de la inicialización de pesos y de las máscaras de
        # Dropout. Toca las semillas globales de python/numpy/torch, que es la
        # única forma de fijar la inicialización de Keras sin sembrar cada capa.
        keras.utils.set_random_seed(self.semilla)

        self._gen = _construir_generador(
            self.dimension, self.n_regimenes, self.dim_latente, self.capas_gen, self.momento_bn
        )
        self._disc = _construir_discriminador(
            self.dimension, self.n_regimenes, self.capas_disc, self.dropout
        )

        # beta_1 = 0.5 en vez del 0.9 por defecto: reducir el momento de primer
        # orden amortigua las oscilaciones del juego, que no converge a un
        # mínimo sino a un punto de silla. tasa_aprendizaje = 2e-4 es el valor
        # que usan los cuatro cuadernos de GAN del material de clase.
        opt_gen = keras.optimizers.Adam(learning_rate=self.tasa_aprendizaje, beta_1=0.5)
        opt_disc = keras.optimizers.Adam(learning_rate=self.tasa_aprendizaje, beta_1=0.5)

        # Los datos pasan a tensores del backend una sola vez. Además de ahorrar
        # una conversión por lote, evita mezclar `ndarray` y tensor en una misma
        # llamada al modelo: Keras 3 lo rechaza en cuanto una de las entradas
        # llega ya como tensor (el bloque falso lo hace siempre).
        datos = keras.ops.convert_to_tensor(bloque)
        condiciones = keras.ops.convert_to_tensor(_una_caliente(y_reg, self.n_regimenes))
        grupos = self._indices_por_regimen(y_reg)
        pasos = max(1, len(bloque) // self.tam_lote)

        if self.verboso:
            print(
                f"cGAN · d={self.dimension} · z={self.dim_latente} · "
                f"G {self._gen.count_params():,} params · "
                f"D {self._disc.count_params():,} params · "
                f"{self.epocas} épocas × {pasos} pasos"
            )

        acc_media_movil = 0.5  # arranca en el equilibrio para no sesgar la banda muerta

        for epoca in range(1, self.epocas + 1):
            acumulado = {
                "perdida_disc": 0.0,
                "perdida_gen": 0.0,
                "acc_disc_real": 0.0,
                "acc_disc_falso": 0.0,
                "salida_disc_real": 0.0,
                "salida_disc_falso": 0.0,
                "dispersion_sintetica": 0.0,
            }
            pasos_disc = 0

            for _ in range(pasos):
                indices = self._muestrear_indices(grupos, len(bloque))
                reales = keras.ops.take(datos, indices, axis=0)
                cond = keras.ops.take(condiciones, indices, axis=0)

                self._gen.zero_grad()
                self._disc.zero_grad()

                ruido = keras.ops.convert_to_tensor(
                    self.rng.standard_normal((len(indices), self.dim_latente)).astype(np.float32)
                )
                falsos = self._gen([ruido, cond], training=True)

                # ── Paso del discriminador ──────────────────────────────────
                # `stop_gradient` corta el paso hacia el generador: en esta fase
                # sólo se entrena el crítico.
                d_real = self._disc([reales, cond], training=True)
                d_falso = self._disc([keras.ops.stop_gradient(falsos), cond], training=True)
                perdida_disc = 0.5 * (
                    _entropia_cruzada(d_real, self.suavizado) + _entropia_cruzada(d_falso, 0.0)
                )

                with torch.no_grad():
                    acc_real = _valor(keras.ops.mean(keras.ops.cast(d_real > 0.5, "float32")))
                    acc_falso = _valor(keras.ops.mean(keras.ops.cast(d_falso < 0.5, "float32")))
                    acc_media_movil = 0.95 * acc_media_movil + 0.05 * (acc_real + acc_falso) / 2

                # La pérdida se mide siempre (hace falta para el historial) pero
                # sólo se actualiza si el discriminador no está dominando.
                if acc_media_movil < self.umbral_disc:
                    _paso_descenso(self._disc, opt_disc, perdida_disc)
                    pasos_disc += 1

                # ── Paso del generador ──────────────────────────────────────
                # Pérdida NO saturante: se maximiza log D(G(z)) etiquetando los
                # falsos como reales, en lugar de minimizar log(1 - D(G(z))),
                # cuyo gradiente se anula justo cuando el generador va peor.
                # Se reutiliza el lote `falsos` sin detach: su grafo sigue vivo
                # porque el paso anterior lo consumió por la rama cortada.
                d_enganado = self._disc([falsos, cond], training=True)
                perdida_gen = _entropia_cruzada(d_enganado, 1.0)
                _paso_descenso(self._gen, opt_gen, perdida_gen)

                with torch.no_grad():
                    acumulado["perdida_disc"] += _valor(perdida_disc)
                    acumulado["perdida_gen"] += _valor(perdida_gen)
                    acumulado["acc_disc_real"] += acc_real
                    acumulado["acc_disc_falso"] += acc_falso
                    acumulado["salida_disc_real"] += _valor(keras.ops.mean(d_real))
                    acumulado["salida_disc_falso"] += _valor(keras.ops.mean(d_falso))
                    # Dispersión entre muestras del lote: si cae de forma
                    # sostenida, hay colapso de modo aunque las pérdidas parezcan
                    # sanas.
                    acumulado["dispersion_sintetica"] += _valor(
                        keras.ops.mean(keras.ops.std(falsos, axis=0))
                    )

            self.registrar(
                epoca=epoca,
                **{clave: total / pasos for clave, total in acumulado.items()},
                fraccion_pasos_disc=pasos_disc / pasos,
            )

            if self.verboso and (epoca % max(1, self.epocas // 10) == 0 or epoca == 1):
                self._informar(epoca)

    def _informar(self, epoca: int) -> None:
        """Traza compacta con los cuatro números que dicen si el juego sigue vivo."""
        fila = self.historial.iloc[-1]
        print(
            f"  época {epoca:4d}/{self.epocas} · "
            f"L_D {fila['perdida_disc']:.3f} · L_G {fila['perdida_gen']:.3f} "
            f"(referencia log 2 = 0.693) · "
            f"acc real {fila['acc_disc_real']:.2f} · acc falso {fila['acc_disc_falso']:.2f}"
        )

    # ── Muestreo ────────────────────────────────────────────────────────────

    def _generate(self, n: int, regimen: int) -> np.ndarray:
        condicion = np.zeros((n, self.n_regimenes), dtype=np.float32)
        condicion[:, regimen] = 1.0

        # En inferencia BatchNorm usa sus medias acumuladas, así que el tamaño de
        # trozo no altera el resultado: sólo acota la memoria.
        trozos = []
        with torch.no_grad():
            for inicio in range(0, n, 1024):
                fin = min(inicio + 1024, n)
                ruido = self.rng.standard_normal((fin - inicio, self.dim_latente)).astype(np.float32)
                salida = self._gen([ruido, condicion[inicio:fin]], training=False)
                trozos.append(keras.ops.convert_to_numpy(salida))

        return np.concatenate(trozos).astype(np.float32)

    # ── Persistencia ────────────────────────────────────────────────────────

    def _guardar(self, destino: Path) -> None:
        """Formato nativo de Keras en vez del pickle de la base.

        El pickle de un `keras.Model` no es portable entre versiones ni entre
        backends; el ``.keras`` guarda arquitectura y pesos de forma estable. Se
        persiste también el discriminador: sin él no se puede reanudar el
        entrenamiento ni auditar el equilibrio del juego a posteriori.
        """
        self._gen.save(destino / "generador.keras")
        self._disc.save(destino / "discriminador.keras")

    def _cargar(self, origen: Path) -> None:
        self._gen = keras.saving.load_model(origen / "generador.keras")
        self._disc = keras.saving.load_model(origen / "discriminador.keras")
        # La dimensión latente se recupera de la propia red: es la única forma
        # de garantizar que el ruido que se muestrea encaja con los pesos.
        self.dim_latente = int(self._gen.inputs[0].shape[-1])

    # ── Utilidades internas ─────────────────────────────────────────────────

    def _aplicar_ajustes(self, kwargs: dict) -> None:
        """Sobreescribe hiperparámetros desde `fit(**kwargs)`."""
        for clave, valor in kwargs.items():
            if clave not in _AJUSTABLES:
                raise TypeError(
                    f"{self.nombre}: hiperparámetro desconocido {clave!r}. "
                    f"Disponibles: {sorted(_AJUSTABLES)}"
                )
            setattr(self, clave, tuple(valor) if clave.startswith("capas_") else valor)

    def _indices_por_regimen(self, y_reg: np.ndarray) -> dict[int, np.ndarray]:
        """Índices de train agrupados por régimen, para el muestreo equilibrado."""
        grupos = {}
        for regimen in range(self.n_regimenes):
            posiciones = np.flatnonzero(y_reg == regimen)
            if len(posiciones) == 0:
                # Condicionar a una clase ausente produce muestras sin sentido:
                # más vale avisar que devolver ruido plausible.
                print(
                    f"AVISO · {self.nombre}: el régimen {regimen} no aparece en train. "
                    "Las muestras condicionadas a él no serán fiables."
                )
                continue
            grupos[regimen] = posiciones
        return grupos

    def _muestrear_indices(self, grupos: dict[int, np.ndarray], n_total: int) -> np.ndarray:
        """Índices de un lote, con o sin cuota por régimen.

        El muestreo es aleatorio, nunca un tramo contiguo. Sobre un panel
        ordenado en el tiempo, un tramo contiguo son ventanas que se solapan en
        59 de sus 60 días: el discriminador aprendería a reconocer ese tramo, no
        la distribución.
        """
        if not self.equilibrar_clases:
            return self.rng.integers(0, n_total, self.tam_lote)

        cuota, resto = divmod(self.tam_lote, len(grupos))
        partes = [
            self.rng.choice(posiciones, size=cuota + (1 if i < resto else 0), replace=True)
            for i, posiciones in enumerate(grupos.values())
        ]
        return self.rng.permutation(np.concatenate(partes))
