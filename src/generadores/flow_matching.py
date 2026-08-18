"""Flow matching condicional con caminos rectos (conditional flow matching).

Idea en una línea
-----------------
En lugar de aprender la densidad de los datos, se aprende el **campo de
velocidades** que transporta ruido gaussiano hasta los datos. Muestrear es
soltar una partícula en :math:`x_0 \\sim \\mathcal{N}(0, I)` e integrar la ODE
:math:`dx/dt = v_\\theta(x, t, c)` de :math:`t=0` a :math:`t=1`.

Entrenamiento (*simulation-free*)
---------------------------------
Por cada muestra del lote se sortea :math:`t \\sim \\mathcal{U}(0,1)` y un ruido
:math:`x_0 \\sim \\mathcal{N}(0,I)`, se interpola en línea recta

.. math::  x_t = (1-t)\\,x_0 + t\\,x_1

y se hace regresión de mínimos cuadrados sobre la velocidad del segmento

.. math::  u = x_1 - x_0, \\qquad
           \\mathcal{L} = \\| v_\\theta(x_t, t, c) - u \\|^2

No hay que integrar nada para entrenar, ni discriminador, ni término KL, ni
determinante jacobiano, ni *schedule* de ruido. Esa ausencia de schedule es la
diferencia práctica frente a difusión: los `min_signal_rate` / `max_signal_rate`
del notebook de DDIM están calibrados para imágenes en ``[0,1]`` y trasladarlos
a un panel financiero estandarizado es una fuente real de fallos silenciosos.
Aquí no hay nada equivalente que calibrar mal.

Sobre la curva de pérdida
-------------------------
Este es el punto que hace a este generador el candidato natural para el
requisito del enunciado ("curvas de loss donde se vea que el modelo ha
convergido"). A diferencia de la GAN —cuyas pérdidas son el marcador de un
juego de suma no nula y pueden subir, bajar u oscilar sin que eso diga nada
sobre la calidad de las muestras— aquí la pérdida es un **MSE ordinario sobre
un objetivo acotado**: baja de forma limpia y casi monótona, y estancarse
significa haber tocado el suelo del problema. La curva de validación se calcula
además con *common random numbers* (rejilla fija de pares ``(x_0, t)``, la misma
en todas las épocas), lo que elimina el ruido de muestreo y deja una curva
prácticamente monótona.

Dos advertencias que hay que escribir en la memoria, no ocultar:

1. **La pérdida no converge a cero.** El minimizador del MSE es la esperanza
   condicional :math:`\\mathbb{E}[x_1 - x_0 \\mid x_t]`, así que el valor límite
   es la varianza condicional residual, estrictamente positiva. Un modelo que
   se estanca no está roto: está en su piso. Referencias útiles para leer la
   escala, con los datos estandarizados: ``2.0`` es el predictor trivial
   :math:`v \\equiv 0` (el modelo no ha aprendido nada) y ``1.571`` (:math:`\\pi/2`)
   es el piso de datos sin ninguna estructura. Por debajo de eso se está
   aprendiendo correlación real.
2. **Una pérdida baja NO garantiza buenas muestras.** El error se integra a lo
   largo de toda la trayectoria, el promedio sobre muestras diluye un fallo en
   la clase minoritaria (crisis, ~10 %) y la regresión con MSE sobre-suaviza:
   marginales correctas pero colas comprimidas, que es justo lo que interesa en
   datos financieros. La convergencia hay que acompañarla siempre de
   comprobaciones distribucionales (ver `docs/metodologia/`).

Detalle de la teoría completa en `docs/teoria/06_flow_matching.md`.

Requisitos de entorno
---------------------
Módulo escrito para **Keras 3 con backend PyTorch en CPU**, que es lo que fija
`src/__init__.py`. El bucle de entrenamiento y el integrador son de PyTorch
puro sobre los pesos del modelo de Keras (ver `_fit`), así que el backend no es
opcional aquí.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import keras
import numpy as np
import torch

from .base import GeneradorSintetico

# ─────────────────────────────────────────────────────────────────────────────
# Embedding sinusoidal del tiempo
# ─────────────────────────────────────────────────────────────────────────────


@keras.saving.register_keras_serializable(package="taller_generativos")
class EmbeddingSinusoidal(keras.layers.Layer):
    """Features de Fourier del instante ``t``.

    Por qué no meter ``t`` como un escalar más: la primera capa densa vería el
    tiempo como **una sola dirección** del espacio de features, y una red con
    activaciones suaves aproxima mal una dependencia en ``t`` que cambie de
    forma rápida entre instantes próximos. Proyectar ``t`` sobre un banco de
    senos y cosenos de frecuencias geométricamente espaciadas le da a la red
    resolución sobre el tiempo a varias escalas de golpe: las frecuencias bajas
    distinguen "principio / mitad / final" y las altas separan instantes casi
    idénticos. Es el mismo `sinusoidal_embedding` del notebook de DDIM del
    material de clase, y es un ejemplo de por qué difusión y flow matching
    comparten esqueleto.

    Parameters
    ----------
    dimension
        Tamaño del embedding. Debe ser par: la mitad son senos y la mitad
        cosenos.
    frecuencia_min, frecuencia_max
        Extremos del banco de frecuencias, espaciadas geométricamente. El rango
        por defecto cubre desde una oscilación por intervalo hasta mil.
    """

    def __init__(
        self,
        dimension: int = 128,
        frecuencia_min: float = 1.0,
        frecuencia_max: float = 1000.0,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        if dimension % 2 != 0:
            raise ValueError(f"La dimensión del embedding debe ser par, recibida {dimension}.")
        self.dimension = dimension
        self.frecuencia_min = frecuencia_min
        self.frecuencia_max = frecuencia_max

    def call(self, t):
        frecuencias = keras.ops.exp(
            keras.ops.linspace(
                math.log(self.frecuencia_min),
                math.log(self.frecuencia_max),
                self.dimension // 2,
            )
        )
        frecuencias = keras.ops.cast(frecuencias, self.compute_dtype)
        # t llega con forma (lote, 1); el broadcasting produce (lote, dimension/2).
        angulos = 2.0 * math.pi * keras.ops.cast(t, self.compute_dtype) * frecuencias[None, :]
        return keras.ops.concatenate(
            [keras.ops.sin(angulos), keras.ops.cos(angulos)], axis=-1
        )

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.dimension)

    def get_config(self) -> dict:
        config = super().get_config()
        config.update(
            {
                "dimension": self.dimension,
                "frecuencia_min": self.frecuencia_min,
                "frecuencia_max": self.frecuencia_max,
            }
        )
        return config


# ─────────────────────────────────────────────────────────────────────────────
# Generador
# ─────────────────────────────────────────────────────────────────────────────

#: Opciones que `fit(**kwargs)` puede sobreescribir sobre lo fijado en
#: `__init__`. La lista blanca evita que un typo en el notebook cree en
#: silencio un atributo nuevo que nadie lee.
_OPCIONES = (
    "ancho",
    "n_capas",
    "dim_tiempo",
    "dim_regimen",
    "epocas",
    "tam_lote",
    "tasa_aprendizaje",
    "decaimiento_pesos",
    "recorte_gradiente",
    "ema",
    "balanceo_clases",
    "frac_validacion",
    "hueco_validacion",
    "niveles_validacion",
    "max_muestras_validacion",
    "n_pasos",
    "metodo_integracion",
    "lote_muestreo",
    "hilos_torch",
    "verboso",
)


class FlowMatching(GeneradorSintetico):
    """Campo de velocidades condicionado por régimen, entrenado con CFM.

    El bloque de entrada es la ventana aplanada más ``y_vol`` (``d ≈ 1201``), ya
    estandarizada por canal con estadísticos de train. El régimen
    ``y_reg ∈ {0,1,2}`` **no** se genera: entra como condición del campo, que es
    lo que permite pedir "500 ventanas de crisis" y fijar a voluntad la mezcla
    de clases del conjunto sintético.

    Parameters
    ----------
    n_regimenes
        Número de clases de la condición.
    semilla
        Semilla de numpy y de los generadores de PyTorch usados en el
        entrenamiento y el muestreo.
    ancho, n_capas
        Geometría del MLP que aproxima ``v_θ``. La dimensión nominal del bloque
        es alta, pero la efectiva no lo es (canales muy correlacionados y muy
        autocorrelacionados), así que el campo es suave y un MLP basta. El
        defecto ``ancho=256`` (≈0,8 M parámetros) es el compromiso entre el
        presupuesto de CPU y el riesgo de memorización: con ~4.000 ventanas muy
        solapadas en 1.201 dimensiones, más capacidad no es gratis. Duplicar a
        ``ancho=512`` multiplica por ~2,3 el coste por época.
    dim_tiempo, dim_regimen
        Dimensiones del embedding sinusoidal de ``t`` y de la proyección del
        one-hot del régimen.
    epocas, tam_lote, tasa_aprendizaje, decaimiento_pesos, recorte_gradiente
        Presupuesto de optimización. AdamW con decaimiento coseno de la tasa.
    ema
        Factor de la media móvil exponencial de los pesos. ``None`` la desactiva.
    balanceo_clases
        Si ``True``, cada época se remuestrea con probabilidad inversa a la
        frecuencia de la clase. Sin esto, la rama de crisis (~10 % de las
        ventanas) recibe diez veces menos gradiente y su campo queda peor
        estimado. Cambia la marginal implícita que aprende el modelo, pero como
        después se genera *condicionando* en la clase, esa marginal es
        irrelevante y el balanceo es beneficio neto.
    frac_validacion, hueco_validacion
        Validación interna: se reserva la **cola cronológica** del bloque, no
        una muestra aleatoria, y se descartan `hueco_validacion` ventanas en el
        corte porque las ventanas consecutivas se solapan (60 días de pasado
        más 21 de horizonte). Es solo un indicador de convergencia; la
        evaluación seria es la del split de validación real del proyecto.
    niveles_validacion, max_muestras_validacion
        Rejilla de *common random numbers* de la pérdida de validación.
    n_pasos, metodo_integracion, lote_muestreo
        Parámetros del muestreo por ODE. Ver `_generate`.
    hilos_torch
        Si se indica, fija ``torch.set_num_threads``. Por defecto torch usa 2
        hilos; en esta máquina de 4 núcleos pasar a 4 recorta el tiempo de época
        de forma apreciable y no tiene contrapartida.

    Coste medido en CPU
    -------------------
    Con ``d = 1201``, ~4.000 ventanas (≈12 pasos de gradiente por época) y 4
    hilos de torch, sobre la máquina del proyecto (4 núcleos, sin CUDA):

    ==================  ==========  ============  ==================
    Configuración       Parámetros  s/época       250 épocas
    ==================  ==========  ============  ==================
    ``ancho=256`` (def) 0,81 M      2,6           **≈11 min**
    ``ancho=512``       1,87 M      5,6           ≈23 min
    ==================  ==========  ============  ==================

    Generar 2.000 ventanas cuesta ~5 s con 20 pasos de Euler y ~20 s con 50. El
    muestreo no es el cuello de botella de este taller, y por eso el número de
    pasos por defecto es holgado en vez de apurado.
    """

    nombre = "flow_matching"
    etiqueta = "Flow matching condicional"
    tiene_curva_perdida = True

    def __init__(
        self,
        n_regimenes: int,
        semilla: int = 42,
        *,
        # ── red ──
        ancho: int = 256,
        n_capas: int = 3,
        dim_tiempo: int = 128,
        dim_regimen: int = 64,
        # ── optimización ──
        epocas: int = 250,
        tam_lote: int = 256,
        tasa_aprendizaje: float = 5e-4,
        decaimiento_pesos: float = 1e-4,
        recorte_gradiente: float = 1.0,
        ema: float | None = 0.999,
        balanceo_clases: bool = True,
        # ── validación interna ──
        frac_validacion: float = 0.15,
        hueco_validacion: int = 80,
        niveles_validacion: int = 5,
        max_muestras_validacion: int = 256,
        # ── muestreo ──
        n_pasos: int = 50,
        metodo_integracion: str = "euler",
        lote_muestreo: int = 512,
        # ── entorno ──
        hilos_torch: int | None = None,
        verboso: bool = True,
    ) -> None:
        super().__init__(n_regimenes=n_regimenes, semilla=semilla)

        self.ancho = ancho
        self.n_capas = n_capas
        self.dim_tiempo = dim_tiempo
        self.dim_regimen = dim_regimen

        self.epocas = epocas
        self.tam_lote = tam_lote
        # El documento de teoría recomienda 2e-4 para un presupuesto de 400-600
        # épocas. Aquí el presupuesto es ~250 épocas (≈4.000 pasos de gradiente)
        # y una tasa algo mayor con decaimiento coseno llega al mismo sitio en
        # menos tiempo de CPU. Si se amplía el presupuesto, bajarla a 2e-4.
        self.tasa_aprendizaje = tasa_aprendizaje
        self.decaimiento_pesos = decaimiento_pesos
        self.recorte_gradiente = recorte_gradiente
        self.ema = ema
        self.balanceo_clases = balanceo_clases

        self.frac_validacion = frac_validacion
        self.hueco_validacion = hueco_validacion
        self.niveles_validacion = niveles_validacion
        self.max_muestras_validacion = max_muestras_validacion

        self.n_pasos = n_pasos
        self.metodo_integracion = metodo_integracion
        self.lote_muestreo = lote_muestreo

        self.hilos_torch = hilos_torch
        self.verboso = verboso

        #: `keras.Model` que implementa v_θ(x_t, t, c). Se construye en `_fit`,
        #: cuando ya se conoce la dimensión del bloque.
        self.red: keras.Model | None = None

    # ── Construcción de la red ──────────────────────────────────────────────

    def _construir_red(self) -> keras.Model:
        """MLP que devuelve la velocidad. Salida **lineal**: el objetivo
        ``x_1 - x_0`` no está acotado ni tiene signo fijo, cualquier activación
        final lo mutilaría."""
        entrada_x = keras.Input(shape=(self.dimension,), name="x_t")
        entrada_t = keras.Input(shape=(1,), name="t")
        entrada_c = keras.Input(shape=(self.n_regimenes,), name="regimen")

        emb_t = EmbeddingSinusoidal(self.dim_tiempo, name="fourier_tiempo")(entrada_t)
        emb_t = keras.layers.Dense(self.dim_tiempo, activation="silu", name="proy_tiempo")(emb_t)
        # Densa sobre el one-hot: es exactamente una tabla de embeddings, pero
        # deja la puerta abierta a condicionar con probabilidades blandas.
        emb_c = keras.layers.Dense(self.dim_regimen, activation="silu", name="proy_regimen")(
            entrada_c
        )

        h = keras.layers.Concatenate(name="concatenacion")([entrada_x, emb_t, emb_c])
        for i in range(self.n_capas):
            h = keras.layers.Dense(self.ancho, activation="silu", name=f"oculta_{i}")(h)
        salida = keras.layers.Dense(self.dimension, name="velocidad")(h)

        return keras.Model([entrada_x, entrada_t, entrada_c], salida, name="campo_velocidad")

    # ── Entrenamiento ───────────────────────────────────────────────────────

    def _fit(self, bloque: np.ndarray, y_reg: np.ndarray, **kwargs) -> None:
        """Regresión de la velocidad con un bucle de PyTorch explícito.

        **Por qué un bucle manual y no `model.fit`.** El objetivo de CFM se
        resortea en cada paso: ``t``, ``x_0`` y por tanto ``x_t`` y ``u`` son
        distintos cada vez que se ve la misma ventana. Encajar eso en
        `model.fit` exige o bien un `PyDataset` que genere el objetivo al vuelo,
        o bien un `train_step` propio; ambas cosas añaden capas entre el código
        y la matemática sin ganar nada, porque en backend torch `fit` también
        corre en modo eager. Con el backend fijado a PyTorch, un `keras.Model`
        **es** un `torch.nn.Module`: `red.parameters()` devuelve los tensores
        reales y `AdamW` de torch los optimiza directamente. El bucle resultante
        es el del notebook del profesor línea por línea, que es exactamente lo
        que se quiere poder enseñar en la defensa.
        """
        self._aplicar_opciones(kwargs)
        self._exigir_backend_torch()

        if self.hilos_torch is not None:
            torch.set_num_threads(self.hilos_torch)

        # Semillas propias del generador: no dependemos de que el notebook haya
        # llamado antes a `config.fijar_semillas`.
        keras.utils.set_random_seed(self.semilla)
        gen = torch.Generator().manual_seed(self.semilla)

        self.red = self._construir_red()

        X = torch.from_numpy(np.ascontiguousarray(bloque, dtype=np.float32))
        C = torch.from_numpy(self._una_caliente(y_reg))

        idx_train, idx_val = self._partir_interno(len(X))
        X_tr, C_tr = X[idx_train], C[idx_train]
        X_val = X[idx_val] if idx_val is not None else None
        C_val = C[idx_val] if idx_val is not None else None

        pesos_muestreo = self._pesos_balanceo(y_reg[idx_train.numpy()])
        tam_lote = min(self.tam_lote, len(X_tr))

        optimizador = torch.optim.AdamW(
            self.red.parameters(),
            lr=self.tasa_aprendizaje,
            weight_decay=self.decaimiento_pesos,
        )
        # Decaimiento coseno: los últimos pasos con tasa pequeña son los que
        # afinan el campo cerca de t=1, donde vive el detalle del dato.
        programador = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizador, T_max=max(self.epocas, 1)
        )

        pesos_ema = None
        if self.ema is not None:
            pesos_ema = [v.value.detach().clone() for v in self.red.trainable_weights]

        for epoca in range(1, self.epocas + 1):
            orden = self._orden_epoca(len(X_tr), pesos_muestreo, gen)
            acumulado, n_lotes = 0.0, 0

            for inicio in range(0, len(orden) - tam_lote + 1, tam_lote):
                seleccion = orden[inicio : inicio + tam_lote]
                x_1 = X_tr[seleccion]
                c = C_tr[seleccion]

                # ── el objetivo CFM, en cuatro líneas ──
                x_0 = torch.randn(x_1.shape, generator=gen)          # ruido base
                t = torch.rand((x_1.shape[0], 1), generator=gen)     # instante uniforme
                x_t = (1.0 - t) * x_0 + t * x_1                      # camino recto
                u = x_1 - x_0                                        # velocidad objetivo

                perdida = torch.mean((self.red([x_t, t, c], training=True) - u) ** 2)

                optimizador.zero_grad(set_to_none=True)
                perdida.backward()
                if self.recorte_gradiente:
                    torch.nn.utils.clip_grad_norm_(
                        self.red.parameters(), self.recorte_gradiente
                    )
                optimizador.step()

                if pesos_ema is not None:
                    with torch.no_grad():
                        for sombra, variable in zip(pesos_ema, self.red.trainable_weights):
                            sombra.mul_(self.ema).add_(
                                variable.value.detach(), alpha=1.0 - self.ema
                            )

                acumulado += float(perdida.detach())
                n_lotes += 1

            programador.step()

            fila = {"epoca": epoca, "perdida": acumulado / max(n_lotes, 1)}
            if X_val is not None:
                fila["perdida_val"] = self._perdida_validacion(X_val, C_val)
            self.registrar(**fila)

            if self.verboso and (epoca % 25 == 0 or epoca == 1):
                self._informar(fila)

        if pesos_ema is not None:
            # Los pesos promediados son los que se conservan: reducen la varianza
            # residual del último tramo del entrenamiento y dan muestras algo
            # mejores sin coste de inferencia.
            with torch.no_grad():
                for sombra, variable in zip(pesos_ema, self.red.trainable_weights):
                    variable.assign(sombra)

    def _orden_epoca(
        self, n: int, pesos: torch.Tensor | None, gen: torch.Generator
    ) -> torch.Tensor:
        """Orden de visita de las muestras en una época."""
        if pesos is None:
            return torch.randperm(n, generator=gen)
        # Remuestreo con reemplazo según el peso de cada clase: cada época ve
        # los tres regímenes en proporciones parecidas.
        return torch.multinomial(pesos, n, replacement=True, generator=gen)

    def _pesos_balanceo(self, y_reg: np.ndarray) -> torch.Tensor | None:
        """Peso de muestreo de cada observación, inverso a la frecuencia de su clase."""
        if not self.balanceo_clases:
            return None
        conteos = np.bincount(y_reg, minlength=self.n_regimenes).astype(np.float64)
        conteos[conteos == 0] = 1.0  # una clase ausente no debe dividir por cero
        pesos = 1.0 / conteos[y_reg]
        return torch.from_numpy(pesos / pesos.sum())

    def _partir_interno(self, n: int) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Separa una cola cronológica como validación interna.

        Devuelve ``(indices_train, indices_val)``, con ``indices_val = None`` si
        el bloque es demasiado pequeño para que la partición tenga sentido (caso
        de las pruebas de humo y de los barridos con muy pocos datos reales).
        """
        n_val = int(round(n * self.frac_validacion))
        n_train = n - n_val - self.hueco_validacion

        if n_val < 32 or n_train < 64:
            return torch.arange(n), None

        indices = torch.arange(n)
        return indices[:n_train], indices[n - n_val :]

    @torch.no_grad()
    def _perdida_validacion(self, X_val: torch.Tensor, C_val: torch.Tensor) -> float:
        """Pérdida de validación con *common random numbers*.

        La pérdida por lote de CFM tiene varianza alta porque en cada paso se
        resortean ``t`` y ``x_0``: la curva sale dentada y no se ve si ha
        convergido. Aquí se fija de antemano una rejilla determinista —``t``
        estratificado en `niveles_validacion` niveles y el mismo ruido en todas
        las épocas, regenerado desde una semilla propia— de modo que la única
        cosa que cambia entre épocas son los pesos. El resultado es una curva
        prácticamente monótona, que es la que hay que llevar al informe.
        """
        gen = torch.Generator().manual_seed(self.semilla + 9973)
        m = min(len(X_val), self.max_muestras_validacion)
        # Submuestreo equiespaciado, no los primeros m: la cola de validación es
        # cronológica y quedarse con su principio sesgaría la medida hacia un
        # único tramo de mercado.
        seleccion = torch.linspace(0, len(X_val) - 1, m).long()
        X_val, C_val = X_val[seleccion], C_val[seleccion]

        total = 0.0
        for k in range(self.niveles_validacion):
            t = torch.full((m, 1), (k + 0.5) / self.niveles_validacion)
            x_0 = torch.randn(X_val.shape, generator=gen)
            x_t = (1.0 - t) * x_0 + t * X_val
            v = self.red([x_t, t, C_val], training=False)
            total += float(torch.mean((v - (X_val - x_0)) ** 2))
        return total / self.niveles_validacion

    # ── Muestreo ────────────────────────────────────────────────────────────

    @torch.no_grad()
    def _generate(self, n: int, regimen: int) -> np.ndarray:
        """Integra ``dx/dt = v_θ(x, t, c)`` desde ruido gaussiano hasta ``t=1``.

        Compromiso del número de pasos
        ------------------------------
        El coste es exactamente el número de evaluaciones de red (NFE): Euler
        con ``N`` pasos son ``N`` evaluaciones y comete un error global
        ``O(1/N)``. Menos pasos es más rápido pero deja error de discretización;
        más pasos solo reduce **ese** error, nunca el del campo aprendido. El
        valor por defecto de 50 es deliberadamente conservador: generar unos
        miles de ventanas cuesta segundos, así que no hay razón para apurar. La
        forma correcta de justificar la elección en la memoria es un barrido
        ``N ∈ {5, 10, 20, 50, 100}`` midiendo Wasserstein-1 medio por canal: la
        curva se aplana en algún punto y ese es el ``N`` que hay que reportar.
        Si al pasar de 20 a 100 pasos las métricas no mejoran, el techo lo pone
        el modelo y hay que tocar la red, no el integrador.

        Euler frente a Heun
        -------------------
        Heun (RK2, predictor-corrector) cuesta 2 evaluaciones por paso pero da
        error ``O(1/N²)``. Compensa cuando el presupuesto de NFE es el cuello de
        botella: con trayectorias casi rectas, Heun con 10 pasos (20 NFE) suele
        igualar a Euler con 50. En este taller el muestreo no es el cuello de
        botella, así que Euler con 50 pasos es la opción por defecto y Heun
        queda como comprobación: si Euler-50 y Heun-25 dan las mismas métricas,
        el error de discretización ya es despreciable.

        Nota: la integración es **determinista**. Toda la diversidad del
        conjunto sintético viene del ruido inicial ``x_0``.
        """
        self._exigir_backend_torch()
        if self.red is None:
            raise RuntimeError("No hay red entrenada; llama a fit() o a cargar().")
        if self.metodo_integracion not in ("euler", "heun"):
            raise ValueError(
                f"Método de integración desconocido: {self.metodo_integracion!r}. "
                "Opciones: 'euler', 'heun'."
            )

        gen = torch.Generator().manual_seed(int(self.rng.integers(0, 2**31 - 1)))
        condicion = np.zeros((1, self.n_regimenes), dtype=np.float32)
        condicion[0, regimen] = 1.0

        trozos = []
        for inicio in range(0, n, self.lote_muestreo):
            m = min(self.lote_muestreo, n - inicio)
            trozos.append(self._integrar(m, condicion, gen))
        return torch.cat(trozos).numpy()

    def _integrar(
        self, m: int, condicion: np.ndarray, gen: torch.Generator
    ) -> torch.Tensor:
        """Resuelve la ODE para un sub-lote de ``m`` partículas."""
        x = torch.randn((m, self.dimension), generator=gen)
        c = torch.from_numpy(np.repeat(condicion, m, axis=0))
        dt = 1.0 / self.n_pasos

        def velocidad(estado: torch.Tensor, instante: float) -> torch.Tensor:
            t = torch.full((m, 1), float(instante))
            return self.red([estado, t, c], training=False)

        for paso in range(self.n_pasos):
            t0 = paso * dt
            if self.metodo_integracion == "euler":
                x = x + velocidad(x, t0) * dt
            else:
                # Heun: se predice con Euler y se corrige con la media de las
                # velocidades en los dos extremos del paso.
                v1 = velocidad(x, t0)
                v2 = velocidad(x + v1 * dt, min(t0 + dt, 1.0))
                x = x + 0.5 * (v1 + v2) * dt
        return x

    # ── Persistencia ────────────────────────────────────────────────────────

    def _guardar(self, destino: Path) -> None:
        """Formato nativo de Keras (`.keras`) más los hiperparámetros de muestreo.

        El `.keras` lleva arquitectura y pesos, así que al cargar no hace falta
        recordar `ancho` ni `n_capas`. Lo que no viaja dentro son las opciones
        que solo afectan a la generación (`n_pasos`, método de integración), y
        esas se dejan en un JSON aparte para que quien cargue el modelo
        reproduzca exactamente las muestras del informe.
        """
        self.red.save(destino / "modelo.keras")
        with open(destino / "hiperparametros.json", "w", encoding="utf-8") as f:
            json.dump(self._hiperparametros(), f, indent=2, ensure_ascii=False)

    def _cargar(self, origen: Path) -> None:
        ruta = origen / "modelo.keras"
        if not ruta.exists():
            raise FileNotFoundError(f"Falta el modelo de Keras en {ruta}.")
        self.red = keras.saving.load_model(ruta)

        config = origen / "hiperparametros.json"
        if config.exists():
            with open(config, encoding="utf-8") as f:
                for clave, valor in json.load(f).items():
                    if clave in _OPCIONES:
                        setattr(self, clave, valor)

    def _hiperparametros(self) -> dict:
        return {clave: getattr(self, clave) for clave in _OPCIONES}

    # ── Utilidades internas ─────────────────────────────────────────────────

    def _aplicar_opciones(self, opciones: dict) -> None:
        """Sobreescribe la configuración con lo que llegue desde `fit(**kwargs)`."""
        desconocidas = set(opciones) - set(_OPCIONES)
        if desconocidas:
            raise TypeError(
                f"Opciones desconocidas para {self.nombre}: {sorted(desconocidas)}. "
                f"Válidas: {sorted(_OPCIONES)}."
            )
        for clave, valor in opciones.items():
            setattr(self, clave, valor)

    def _una_caliente(self, y_reg: np.ndarray) -> np.ndarray:
        """One-hot del régimen, en float32 para entrar directo en la red."""
        if y_reg.min() < 0 or y_reg.max() >= self.n_regimenes:
            raise ValueError(
                f"Regímenes fuera de [0, {self.n_regimenes}): "
                f"observados [{y_reg.min()}, {y_reg.max()}]."
            )
        caliente = np.zeros((len(y_reg), self.n_regimenes), dtype=np.float32)
        caliente[np.arange(len(y_reg)), y_reg] = 1.0
        return caliente

    @staticmethod
    def _exigir_backend_torch() -> None:
        if keras.backend.backend() != "torch":
            raise RuntimeError(
                "flow_matching necesita el backend de PyTorch. Importa `src` antes "
                "que cualquier otro módulo que use Keras: es quien fija "
                "KERAS_BACKEND='torch'."
            )

    def _informar(self, fila: dict) -> None:
        partes = [f"época {fila['epoca']:4d}", f"pérdida {fila['perdida']:.4f}"]
        if "perdida_val" in fila:
            partes.append(f"val {fila['perdida_val']:.4f}")
        print(" | ".join(partes))
