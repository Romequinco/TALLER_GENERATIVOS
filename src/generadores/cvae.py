"""Autoencoder variacional condicional (cVAE) sobre el bloque ventana+volatilidad.

Implementa la sección 8 de `docs/teoria/03_vae.md`. El modelo aprende
``p(x | y_reg)`` con ``x`` el bloque aplanado de `ventanas.empaquetar()` y
``y_reg`` el régimen de mercado, inyectado como one-hot de 3 clases en el
encoder **y** en el decoder.

Dos decisiones lo separan del notebook de clase
(`docs/material_clase/notebooks/Keras_AE_variational.ipynb`), y ambas son
consecuencia del dato, no de gustos:

**Verosimilitud gaussiana, no Bernoulli.** El notebook trabaja con MNIST en
``[0, 1]``, remata el decoder con `sigmoid` y usa BCE. Nuestro bloque son
z-scores: soporte en todo ``R``, mitad de las entradas negativas y colas de
±5σ en las crisis. Con BCE la pérdida sale finita y absurda —no lanza
excepción, que es lo peligroso— y una `sigmoid` final no puede emitir el
-4,5σ de marzo de 2020. Aquí la verosimilitud es gaussiana isótropa de
varianza fija: **salida lineal** y error cuadrático **sumado** sobre las ``d``
dimensiones.

**Escalado consistente de los dos términos.** El ELBO es una suma sobre
coordenadas. El notebook suma la reconstrucción sobre 784 píxeles pero
promedia el KL sobre las 2 dimensiones latentes, lo que introduce un β
implícito de factor ``d·J``. Con ``d≈1201`` y ``J=32`` copiarlo sería un
desastre silencioso. Aquí se suma sobre dimensiones en ambos términos, se
promedia solo sobre el batch, y la única ponderación es `beta`, explícita.
`beta` es la varianza asumida del decoder: minimizar ``SSE/(2σ²) + KL``
equivale a minimizar ``SSE + β·KL`` con ``β = 2σ²``. No son dos
hiperparámetros, es uno.

Diagnóstico de convergencia
---------------------------
El enunciado exige curvas que demuestren convergencia, y en un VAE **una sola
curva engaña**. Con ``d≈1201`` la reconstrucción domina la pérdida total por
uno o dos órdenes de magnitud, así que la curva total es prácticamente la de
reconstrucción y puede bajar suave y aplanarse mientras el modelo está roto:
el KL se ha ido a cero, el latente ha dejado de usarse y el decoder emite la
media marginal. Por eso `historial` registra las tres pérdidas por separado
—total, reconstrucción y KL— más las **unidades activas**, el número de
dimensiones latentes con ``D_KL > 0.01`` nats. Esa cifra es la que detecta el
colapso cuando las curvas agregadas no lo muestran: si baja época a época hay
colapso progresivo aunque la pérdida total siga descendiendo. El KL sano
**sube** desde ~0, alcanza un máximo y se estabiliza en una meseta
estrictamente positiva.

Contra el colapso se aplican las dos profilaxis estándar, ambas configurables:
rampa lineal de `beta` durante las primeras épocas (KL annealing, Bowman et
al. 2016) y `free bits` (Kingma et al. 2016), que garantiza a cada dimensión
latente un presupuesto de información por debajo del cual el KL no ejerce
gradiente y no puede aplastarla.
"""

from __future__ import annotations

import json
from pathlib import Path

import keras
import numpy as np

from .base import GeneradorSintetico

# ─────────────────────────────────────────────────────────────────────────────
# Capa de reparametrización
# ─────────────────────────────────────────────────────────────────────────────


@keras.saving.register_keras_serializable(package="taller_b5t1")
class MuestreoGaussiano(keras.layers.Layer):
    """Truco de reparametrización: ``z = μ + exp(0.5·log σ²) ⊙ ε``.

    Se implementa como subclase de `keras.Layer` y no con `keras.layers.Lambda`
    a propósito. `Lambda` serializa la función Python con `marshal`, lo que
    obliga a `safe_mode=False` al recargar y rompe entre versiones de Python;
    una capa registrada con `@register_keras_serializable` round-trip'ea limpia
    en formato `.keras`, que es lo que necesita `CVAE._cargar`.

    El ruido se muestrea con `keras.random.normal`, no con `tf.random.normal`
    como hace el notebook de clase: bajo el backend PyTorch esa llamada
    devolvería un tensor de TensorFlow y rompería el grafo de autograd.
    """

    def __init__(self, semilla: int = 42, **kwargs) -> None:
        super().__init__(**kwargs)
        self.semilla = semilla
        # El SeedGenerator mantiene el estado del RNG como variable de la capa,
        # de modo que el muestreo es reproducible y se guarda con el modelo.
        self.generador = keras.random.SeedGenerator(semilla)

    def call(self, entradas):
        z_media, z_log_var = entradas
        ruido = keras.random.normal(shape=keras.ops.shape(z_media), seed=self.generador)
        # σ = exp(0.5·log σ²): positividad garantizada sin activaciones raras y
        # sin desbordar en el rango que puede emitir una capa lineal.
        return z_media + keras.ops.exp(0.5 * z_log_var) * ruido

    def compute_output_shape(self, input_shape):
        return input_shape[0]

    def get_config(self) -> dict:
        return {**super().get_config(), "semilla": self.semilla}


# ─────────────────────────────────────────────────────────────────────────────
# Modelo entrenable
#
# Se sobreescribe `compute_loss` en vez de `train_step`: el `train_step` del
# backend torch ya resuelve zero_grad / backward / apply, y reescribirlo obliga
# a importar torch y a duplicar esa mecánica. `call` devuelve los tres tensores
# que necesita el ELBO para no tener que guardarlos como estado del objeto.
# ─────────────────────────────────────────────────────────────────────────────


class _RedCVAE(keras.Model):
    """Envoltorio encoder+decoder que calcula el ELBO negativo."""

    def __init__(
        self,
        codificador: keras.Model,
        decodificador: keras.Model,
        free_bits: float,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.codificador = codificador
        self.decodificador = decodificador
        self.free_bits = float(free_bits)
        # `beta` es variable, no constante: la rampa de annealing la actualiza
        # entre épocas sin recompilar el modelo.
        self.beta = keras.Variable(0.0, trainable=False, dtype="float32", name="beta")
        # Keras rastrea automáticamente las métricas asignadas como atributos;
        # sus nombres son las claves del historial que devuelve `fit`.
        self.rastreador_recon = keras.metrics.Mean(name="reconstruccion")
        self.rastreador_kl = keras.metrics.Mean(name="kl")

    def call(self, entradas, training=False):
        bloque, condicion = entradas
        z_media, z_log_var, z = self.codificador([bloque, condicion], training=training)
        reconstruccion = self.decodificador([z, condicion], training=training)
        return reconstruccion, z_media, z_log_var

    def compute_loss(self, x=None, y=None, y_pred=None, sample_weight=None, training=True):
        """ELBO negativo promediado sobre el batch.

        Reconstrucción: SSE **sumada** sobre las ``d`` dimensiones
        (verosimilitud gaussiana de varianza fija). KL: analítico frente a
        ``N(0, I)``, **sumado** sobre las ``J`` dimensiones latentes, con suelo
        de free bits. El KL que se reporta es el crudo, sin suelo: el suelo
        afecta al gradiente, no al diagnóstico.
        """
        bloque = x[0]
        reconstruccion, z_media, z_log_var = y_pred

        recon = keras.ops.mean(
            keras.ops.sum(keras.ops.square(bloque - reconstruccion), axis=-1)
        )

        kl_por_dim = -0.5 * (
            1.0 + z_log_var - keras.ops.square(z_media) - keras.ops.exp(z_log_var)
        )
        kl_medio_dim = keras.ops.mean(kl_por_dim, axis=0)          # (J,)
        kl_reportado = keras.ops.sum(kl_medio_dim)
        # Free bits: por debajo de λ nats el gradiente del KL sobre esa
        # dimensión es nulo, así que ninguna dimensión puede ser aplastada.
        kl_efectivo = keras.ops.sum(keras.ops.maximum(kl_medio_dim, self.free_bits))

        self.rastreador_recon.update_state(recon)
        self.rastreador_kl.update_state(kl_reportado)
        return recon + keras.ops.convert_to_tensor(self.beta) * kl_efectivo


# ─────────────────────────────────────────────────────────────────────────────
# Registro por época
#
# El annealing y el volcado del historial van en un callback, no en un bucle de
# `fit(epochs=1)` repetido: cada llamada a `fit` reconstruye el iterador de
# datos y la función de entrenamiento, y sobre el bloque completo ese coste fijo
# (~4 s medidos) multiplicado por 200 épocas supera al del entrenamiento.
# ─────────────────────────────────────────────────────────────────────────────


class _RegistroPorEpoca(keras.callbacks.Callback):
    """Aplica la rampa de `beta` y vuelca las métricas de época al generador."""

    def __init__(
        self, generador: "CVAE", diag_bloque: np.ndarray, diag_condicion: np.ndarray
    ) -> None:
        super().__init__()
        self.generador = generador
        self.diag_bloque = diag_bloque
        self.diag_condicion = diag_condicion
        self.kl_dims: list[np.ndarray] = []
        self.beta_epoca = 0.0

    def on_epoch_begin(self, epoch, logs=None) -> None:
        self.beta_epoca = self.generador._beta_en(epoch)
        self.model.beta.assign(self.beta_epoca)

    def on_epoch_end(self, epoch, logs=None) -> None:
        logs = logs or {}
        gen = self.generador

        kl_dim = gen._kl_por_dimension(self.diag_bloque, self.diag_condicion)
        self.kl_dims.append(kl_dim)

        fila = {
            "epoca": epoch,
            "beta": self.beta_epoca,
            "perdida_total": float(logs["loss"]),
            "perdida_reconstruccion": float(logs["reconstruccion"]),
            "perdida_kl": float(logs["kl"]),
        }
        if "val_loss" in logs:
            fila["val_perdida_total"] = float(logs["val_loss"])
            fila["val_perdida_reconstruccion"] = float(logs["val_reconstruccion"])
            fila["val_perdida_kl"] = float(logs["val_kl"])
        # Indicador de colapso: dimensiones latentes que transportan información
        # real. Si baja época a época hay colapso progresivo, por mucho que la
        # pérdida total siga descendiendo.
        fila["unidades_activas"] = int((kl_dim > gen.umbral_unidad_activa).sum())

        gen.registrar(**fila)

        if gen.verboso and (epoch + 1) % gen.verboso == 0:
            print(
                f"[cvae] época {epoch + 1:>4}/{gen.epocas}  "
                f"total={fila['perdida_total']:>10.2f}  "
                f"recon={fila['perdida_reconstruccion']:>10.2f}  "
                f"kl={fila['perdida_kl']:>7.3f}  "
                f"activas={fila['unidades_activas']:>3}/{gen.dim_latente}",
                flush=True,
            )


# ─────────────────────────────────────────────────────────────────────────────
# Generador
# ─────────────────────────────────────────────────────────────────────────────


class CVAE(GeneradorSintetico):
    """VAE condicional con MLP, condicionado al régimen por one-hot.

    El decoder reconstruye únicamente el bloque ``[X ; y_vol]``; el régimen no
    forma parte de lo reconstruido porque es la condición. La consecuencia
    práctica es que la etiqueta del dato sintético es **exacta**: es la ``c``
    impuesta al muestrear, no una salida continua redondeada como en el enfoque
    de bloque conjunto del notebook de GAN de clase. Eso es lo que permite
    pedir "500 muestras de crisis" en vez de generar mucho y filtrar.

    Parameters
    ----------
    n_regimenes
        Número de clases de la condición. Fija el ancho del one-hot.
    semilla
        Semilla de la generación de muestras y de la capa de reparametrización.
    dim_latente
        ``J``, ancho de banda del cuello de botella. 32 es el punto de partida
        del documento teórico; contar unidades activas al final del
        entrenamiento dice si sobran.
    unidades_ocultas
        ``h``, anchura de la primera capa oculta del encoder (el decoder es su
        espejo).
    beta
        Peso del KL. Equivale a ``2σ_x²`` con σ_x la desviación asumida del
        decoder. Con datos estandarizados el rango útil es 0,5–5.
    epocas_rampa_beta
        Épocas de la rampa lineal de `beta` desde 0 (KL annealing). 0 la
        desactiva.
    free_bits
        Suelo λ en nats por dimensión latente. 0 lo desactiva.
    epocas, tasa_aprendizaje, clip_norma
        Parámetros del bucle de optimización.
    tam_lote
        128 es el valor del documento teórico y el que mejor aprovecha las
        ~4.000 ventanas de train (31 actualizaciones por época). Es además el
        mando de coste en CPU: con este bloque de ~1200 dimensiones, subirlo a
        512 reduce el tiempo por época en un factor 3-4 porque las capas densas
        pasan a GEMM más eficientes y hay menos pasos por época. Si el
        entrenamiento no cabe en el tiempo disponible, este es el primer
        parámetro que tocar, no `epocas`.
    umbral_unidad_activa
        Nats por encima de los cuales una dimensión latente cuenta como activa.
    n_diagnostico
        Muestras usadas cada época para el KL por dimensión. Acotarlo mantiene
        el coste del diagnóstico despreciable frente al del entrenamiento.

    Notes
    -----
    Todos los hiperparámetros admiten sobreescritura desde
    ``fit(bloque, y_reg, beta=1.0, epocas=50, ...)``. `fit` acepta además tres
    argumentos que no son hiperparámetros del modelo:

    ``bloque_val`` e ``y_reg_val``
        Tramo de validación. Añade al historial las columnas ``val_*``, que son
        las que revelan el sobreajuste del decoder. Debe venir del split
        temporal con embargo, nunca de un corte aleatorio.
    ``hilos``
        Hilos de PyTorch. Por defecto torch ya usa los núcleos **físicos**;
        subirlo al número de hilos lógicos sobresuscribe los GEMM y mide peor,
        así que solo tiene sentido tocarlo para bajarlo y dejar la máquina
        libre.
    """

    nombre = "cvae"
    etiqueta = "VAE condicional"
    tiene_curva_perdida = True

    #: Hiperparámetros que `fit(**kwargs)` puede sobreescribir.
    _HIPERPARAMETROS = (
        "dim_latente",
        "unidades_ocultas",
        "beta",
        "epocas_rampa_beta",
        "free_bits",
        "epocas",
        "tam_lote",
        "tasa_aprendizaje",
        "clip_norma",
        "umbral_unidad_activa",
        "n_diagnostico",
        "verboso",
    )

    def __init__(
        self,
        n_regimenes: int,
        semilla: int = 42,
        *,
        dim_latente: int = 32,
        unidades_ocultas: int = 512,
        beta: float = 2.0,
        epocas_rampa_beta: int = 30,
        free_bits: float = 0.05,
        epocas: int = 200,
        tam_lote: int = 128,
        tasa_aprendizaje: float = 1e-3,
        clip_norma: float = 5.0,
        umbral_unidad_activa: float = 0.01,
        n_diagnostico: int = 2048,
        verboso: int = 25,
    ) -> None:
        super().__init__(n_regimenes=n_regimenes, semilla=semilla)

        self.dim_latente = dim_latente
        self.unidades_ocultas = unidades_ocultas
        self.beta = beta
        self.epocas_rampa_beta = epocas_rampa_beta
        self.free_bits = free_bits
        self.epocas = epocas
        self.tam_lote = tam_lote
        self.tasa_aprendizaje = tasa_aprendizaje
        self.clip_norma = clip_norma
        self.umbral_unidad_activa = umbral_unidad_activa
        self.n_diagnostico = n_diagnostico
        self.verboso = verboso

        self.codificador: keras.Model | None = None
        self.decodificador: keras.Model | None = None
        #: Matriz ``(epocas, J)`` con el KL de cada dimensión latente. Es el
        #: heatmap que muestra si el modelo usa 3 dimensiones de 32.
        self.kl_por_dimension: np.ndarray | None = None

    # ── Arquitectura ────────────────────────────────────────────────────────

    def _construir_codificador(self) -> keras.Model:
        """``[x ; c] -> (μ, log σ², z)``.

        Las dos cabezas son lineales sin activación: μ vive en ``R`` y log σ²
        también. Una ReLU o una sigmoid ahí rompen el modelo.
        """
        entrada_x = keras.Input(shape=(self.dimension,), name="bloque")
        entrada_c = keras.Input(shape=(self.n_regimenes,), name="condicion")

        h = keras.layers.Concatenate(name="x_condicionada")([entrada_x, entrada_c])
        h = keras.layers.Dense(self.unidades_ocultas, activation="silu", name="enc_1")(h)
        h = keras.layers.Dense(self.unidades_ocultas // 2, activation="silu", name="enc_2")(h)

        z_media = keras.layers.Dense(self.dim_latente, name="z_media")(h)
        z_log_var = keras.layers.Dense(self.dim_latente, name="z_log_var")(h)
        z = MuestreoGaussiano(semilla=self.semilla, name="z")([z_media, z_log_var])

        return keras.Model([entrada_x, entrada_c], [z_media, z_log_var, z], name="codificador")

    def _construir_decodificador(self) -> keras.Model:
        """``[z ; c] -> x̂``, con salida **lineal**.

        La activación de salida no es libre: es consecuencia de la
        verosimilitud. Gaussiana ⇒ soporte en ``R`` ⇒ sin `sigmoid` ni `tanh`.
        """
        entrada_z = keras.Input(shape=(self.dim_latente,), name="z")
        entrada_c = keras.Input(shape=(self.n_regimenes,), name="condicion")

        h = keras.layers.Concatenate(name="z_condicionada")([entrada_z, entrada_c])
        h = keras.layers.Dense(self.unidades_ocultas // 2, activation="silu", name="dec_1")(h)
        h = keras.layers.Dense(self.unidades_ocultas, activation="silu", name="dec_2")(h)
        salida = keras.layers.Dense(self.dimension, activation=None, name="reconstruccion")(h)

        return keras.Model([entrada_z, entrada_c], salida, name="decodificador")

    # ── Ajuste ──────────────────────────────────────────────────────────────

    def _fit(
        self,
        bloque: np.ndarray,
        y_reg: np.ndarray,
        *,
        bloque_val: np.ndarray | None = None,
        y_reg_val: np.ndarray | None = None,
        hilos: int | None = None,
        **hiperparametros,
    ) -> None:
        self._aplicar_hiperparametros(hiperparametros)

        if hilos is not None:
            # El cuello de botella en CPU son los GEMM de las capas densas.
            import torch

            torch.set_num_threads(hilos)

        # Reproducibilidad de la inicialización de pesos y del barajado de fit.
        keras.utils.set_random_seed(self.semilla)

        condicion = self._una_caliente(y_reg)
        self.codificador = self._construir_codificador()
        self.decodificador = self._construir_decodificador()

        red = _RedCVAE(self.codificador, self.decodificador, free_bits=self.free_bits)
        red.compile(
            optimizer=keras.optimizers.Adam(
                learning_rate=self.tasa_aprendizaje,
                # Evita los picos de las primeras épocas, cuando la
                # reconstrucción sumada sobre ~1200 dimensiones es enorme.
                global_clipnorm=self.clip_norma,
            ),
            # torch.compile no compensa en lotes de 128 sobre CPU: el coste de
            # compilación supera lo que ahorra en 200 épocas cortas.
            jit_compile=False,
        )

        validacion = None
        if bloque_val is not None and y_reg_val is not None:
            # Tupla ``(x, y)`` con y a None: el ELBO no necesita objetivo, la
            # reconstrucción se compara contra la propia entrada.
            validacion = (
                [
                    np.asarray(bloque_val, dtype=np.float32),
                    self._una_caliente(np.asarray(y_reg_val, dtype=np.int64)),
                ],
                None,
            )

        registro = _RegistroPorEpoca(self, *self._muestra_diagnostico(bloque, condicion))
        red.fit(
            [bloque, condicion],
            batch_size=self.tam_lote,
            epochs=self.epocas,
            shuffle=True,
            validation_data=validacion,
            callbacks=[registro],
            verbose=0,
        )
        self.kl_por_dimension = np.stack(registro.kl_dims) if registro.kl_dims else None

    def _generate(self, n: int, regimen: int) -> np.ndarray:
        """Muestrea del prior y decodifica con la condición pedida.

        En generación no se reparametriza: ``z ~ N(0, I)`` directamente. Se usa
        el RNG de la clase base para que dos llamadas con la misma semilla den
        el mismo dataset sintético.
        """
        z = self.rng.standard_normal((n, self.dim_latente)).astype(np.float32)
        condicion = self._una_caliente(np.full(n, regimen, dtype=np.int64))
        return self.decodificador.predict(
            [z, condicion], batch_size=min(n, 512), verbose=0
        )

    # ── Persistencia ────────────────────────────────────────────────────────
    #
    # Se sobreescribe el pickle de la base: los objetos de Keras no son
    # picklables de forma fiable entre versiones. El formato `.keras` guarda
    # arquitectura y pesos en un único archivo y `MuestreoGaussiano` está
    # registrada como serializable, así que la recarga no necesita
    # `custom_objects`.

    def _guardar(self, destino: Path) -> None:
        self.codificador.save(destino / "codificador.keras")
        self.decodificador.save(destino / "decodificador.keras")

        with open(destino / "hiperparametros.json", "w", encoding="utf-8") as f:
            json.dump(self._hiperparametros_actuales(), f, indent=2, ensure_ascii=False)

        if self.kl_por_dimension is not None:
            # El heatmap J×épocas del KL por dimensión no cabe en historial.csv
            # sin ensuciar las figuras automáticas del notebook 13.
            np.savetxt(
                destino / "kl_por_dimension.csv",
                self.kl_por_dimension,
                delimiter=",",
                fmt="%.6f",
            )

    def _cargar(self, origen: Path) -> None:
        with open(origen / "hiperparametros.json", encoding="utf-8") as f:
            for clave, valor in json.load(f).items():
                setattr(self, clave, valor)

        self.codificador = keras.saving.load_model(origen / "codificador.keras")
        self.decodificador = keras.saving.load_model(origen / "decodificador.keras")

        kl = origen / "kl_por_dimension.csv"
        if kl.exists():
            self.kl_por_dimension = np.loadtxt(kl, delimiter=",", ndmin=2)

    # ── Utilidades internas ─────────────────────────────────────────────────

    def _aplicar_hiperparametros(self, hiperparametros: dict) -> None:
        """Sobreescribe los valores por defecto con lo que llegue de `fit`."""
        desconocidos = set(hiperparametros) - set(self._HIPERPARAMETROS)
        if desconocidos:
            raise TypeError(
                f"Hiperparámetros desconocidos para {self.nombre}: "
                f"{sorted(desconocidos)}. Válidos: {sorted(self._HIPERPARAMETROS)}."
            )
        for clave, valor in hiperparametros.items():
            setattr(self, clave, valor)

    def _hiperparametros_actuales(self) -> dict:
        return {clave: getattr(self, clave) for clave in self._HIPERPARAMETROS}

    def _una_caliente(self, y_reg: np.ndarray) -> np.ndarray:
        """One-hot ``(n, K)`` de la condición, en float32."""
        y_reg = np.asarray(y_reg, dtype=np.int64)
        if y_reg.size and (y_reg.min() < 0 or y_reg.max() >= self.n_regimenes):
            raise ValueError(
                f"Etiquetas fuera de [0, {self.n_regimenes}): "
                f"[{y_reg.min()}, {y_reg.max()}]."
            )
        codigo = np.zeros((len(y_reg), self.n_regimenes), dtype=np.float32)
        codigo[np.arange(len(y_reg)), y_reg] = 1.0
        return codigo

    def _beta_en(self, epoca: int) -> float:
        """Rampa lineal de 0 a `beta` durante `epocas_rampa_beta` épocas.

        Da al decoder margen para aprender a usar ``z`` antes de que el KL
        empiece a penalizarlo, que es cuando se produce el colapso temprano.
        """
        if self.epocas_rampa_beta <= 0:
            return float(self.beta)
        return float(self.beta) * min(1.0, (epoca + 1) / self.epocas_rampa_beta)

    def _muestra_diagnostico(
        self, bloque: np.ndarray, condicion: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Submuestra fija sobre la que se mide el KL por dimensión cada época."""
        if len(bloque) <= self.n_diagnostico:
            return bloque, condicion
        # RNG propio en vez de `self.rng`: si el diagnóstico consumiera el flujo
        # del generador, `cargar()` + `generate()` no reproduciría lo que da
        # `fit()` + `generate()` en la misma sesión.
        sorteo = np.random.default_rng(self.semilla)
        idx = sorteo.choice(len(bloque), size=self.n_diagnostico, replace=False)
        return bloque[idx], condicion[idx]

    def _kl_por_dimension(self, bloque: np.ndarray, condicion: np.ndarray) -> np.ndarray:
        """``D_KL,j`` promediado sobre las muestras de diagnóstico.

        Se evalúa sobre μ y log σ² directamente, sin muestrear: el KL analítico
        no depende de ε y así el diagnóstico no añade varianza propia.
        """
        z_media, z_log_var, _ = self.codificador.predict(
            [bloque, condicion], batch_size=len(bloque), verbose=0
        )
        kl = -0.5 * (1.0 + z_log_var - np.square(z_media) - np.exp(z_log_var))
        return kl.mean(axis=0)
