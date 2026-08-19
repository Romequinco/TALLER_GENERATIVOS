"""Modelo downstream: la arquitectura que mide el valor de los datos sintéticos.

El enunciado del taller es explícito: se **busca** una arquitectura válida con
datos reales y luego se entrena esa misma arquitectura, sin tocarla, sobre cada
dataset real/sintético. La comparación solo es honesta si lo único que cambia
entre versiones son los datos.

Este módulo materializa las tres piezas que hacen cierta esa frase:

``CANDIDATAS``   seis arquitecturas, cada una aislando un eje distinto, que es
                 lo que convierte "buscar" en algo defendible y no en elegir a
                 ojo la única que se probó.
``buscar``       las entrena con varias semillas y las puntúa **sobre
                 validación**, nunca sobre test.
``congelar``     escribe en disco la ganadora y su presupuesto, para que el
                 notebook 12 no herede el valor por defecto del código fuente.

El presupuesto de optimización (épocas, tamaño de lote, paciencia y **criterio
de parada**) también se congela, y por la misma razón que la arquitectura: si
cambia entre versiones, no se sabría si la diferencia viene de los datos, de
haber entrenado más o de haber parado en otro sitio (D20).

El criterio de parada de la tarea de régimen no es `val_loss`: es la misma
métrica que elige la arquitectura, la balanced accuracy de validación. La razón
está medida y documentada en `ParadaPorMetrica`; en dos líneas, train y
validación tienen repartos de clase muy distintos, la entropía cruzada de
validación premia por eso al modelo que aún no ha aprendido el prior de train, y
sobre los 22 entrenamientos del cuaderno 03 parar por ella cuesta **1,7 puntos de
balanced accuracy de media**.

Se resuelven dos tareas sobre la misma ventana de entrada:

``regimen``  clasificación del régimen dominante a 21 días (3 clases)
``volatilidad``  regresión de la volatilidad realizada a 21 días

Comparten troncal —la que fije la `Arquitectura` que se les pase; la congelada
por el cuaderno 03 es `lineal`, que **no tiene convoluciones** y aplana la ventana
directamente— y solo difieren en la capa de salida y la pérdida. Que la troncal
sea idéntica permite atribuir cualquier diferencia entre tareas a la naturaleza
del problema, no a la arquitectura.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from . import DIR_HISTORIALES, DIR_MODELOS_DOWN

if TYPE_CHECKING:  # solo para anotar; evita importar ventanas en tiempo de carga
    from .ventanas import Conjunto

TAREAS = ("regimen", "volatilidad")

# Cómo se colapsa el eje temporal antes de la cabeza densa. `aplanado` conserva
# la posición de cada instante dentro de la ventana; `promedio_global` la
# descarta y solo retiene "cuánto" de cada patrón hubo, no "cuándo".
AGREGACIONES = ("aplanado", "promedio_global")


# ---------------------------------------------------------------------------
# Arquitectura y presupuesto
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Arquitectura:
    """Hiperparámetros de la troncal. Se eligen en el notebook 03 y no se tocan.

    Los valores por defecto están dimensionados para CPU: con ~4.000 ventanas
    de (60, 20) una época tarda del orden de un segundo, que es lo que hace
    viable el barrido de varios cientos de entrenamientos.

    Attributes
    ----------
    filtros
        Anchura de cada bloque convolucional. La tupla vacía deja el modelo sin
        convoluciones, que es la candidata de control.
    unidades_densa
        Anchura de la capa densa previa a la salida. Cero la suprime.
    agregacion
        Uno de `AGREGACIONES`.
    """

    filtros: tuple[int, ...] = (64, 128, 128)
    tamano_kernel: int = 3
    unidades_densa: int = 100
    dropout: float = 0.3
    tasa_aprendizaje: float = 1e-3
    agregacion: str = "aplanado"

    def __post_init__(self) -> None:
        # Un valor mal escrito aquí produciría un modelo distinto del que dice
        # la tabla del informe, y nada más aguas abajo lo detectaría.
        if self.agregacion not in AGREGACIONES:
            raise ValueError(
                f"Agregación desconocida: {self.agregacion!r}. Opciones: {AGREGACIONES}"
            )

    def __str__(self) -> str:
        troncal = "-".join(map(str, self.filtros)) if self.filtros else "sin conv"
        densa = f"densa {self.unidades_densa}" if self.unidades_densa else "sin densa"
        return (
            f"CNN1D(filtros {troncal} · kernel {self.tamano_kernel} · "
            f"{self.agregacion} · {densa} · dropout {self.dropout})"
        )


ARQUITECTURA = Arquitectura()

# Métrica con la que se elige la arquitectura, y su desempate. Ver el docstring
# de `buscar` para el argumento de por qué no es el f1-macro. Vive aquí arriba, y
# no junto al catálogo de candidatas, porque `Presupuesto` la usa como criterio
# de parada por defecto: la métrica que decide y la que para tienen que ser la
# misma, y la única forma de garantizarlo es que sean literalmente la misma
# constante.
METRICA_SELECCION = "balanced_accuracy"
METRICA_DESEMPATE = "recall_crisis"
MARGEN_EMPATE = 0.005

# Criterios de parada admisibles y su sentido: True si mayor es mejor.
#
# Las claves **sin** prefijo son métricas de `evaluacion.metricas_regimen` y las
# calcula `ParadaPorMetrica` al final de cada época, porque Keras no las conoce.
# Las que empiezan por `val_` ya están en el log de la época y las gestiona
# `keras.callbacks.EarlyStopping` sin coste añadido. El prefijo es, por tanto, lo
# que decide qué callback se monta: `accuracy` se recalcula sobre validación al
# final de cada época, `val_loss` se lee del log que Keras ya ha rellenado.
MONITORES: dict[str, bool] = {
    "balanced_accuracy": True,
    "recall_crisis": True,
    "f1_macro": True,
    "accuracy": True,
    "val_loss": False,
    "val_mae": False,
}


def _mayor_mejor(monitor: str) -> bool:
    """Sentido de un criterio de parada, con el error explícito si no existe.

    Un monitor mal escrito es un fallo silencioso caro: `EarlyStopping` avisa por
    `warnings` —que en un notebook con 570 entrenamientos nadie lee— y sigue
    entrenando hasta agotar las épocas sin restaurar nada.
    """
    if monitor not in MONITORES:
        raise ValueError(
            f"Criterio de parada desconocido: {monitor!r}. Opciones: "
            f"{sorted(MONITORES)}. Las que no empiezan por 'val_' son métricas de "
            "`evaluacion.metricas_regimen` y solo valen en la tarea de régimen."
        )
    return MONITORES[monitor]


@dataclass(frozen=True)
class Presupuesto:
    """Presupuesto de optimización, congelado igual que la arquitectura.

    D20 exige congelar también esto. Mientras fueron argumentos por defecto de
    `entrenar`, cada celda de notebook los sobrescribía y el barrido acabó
    corriendo con lote 64 pese a que el docstring del módulo defendía 256: una
    contradicción que nadie veía porque el valor estaba en el sitio equivocado.

    Attributes
    ----------
    monitor
        Criterio de parada de la tarea de **régimen**. Por defecto
        `METRICA_SELECCION`, es decir, la misma métrica que elige la
        arquitectura. Forma parte de lo que se congela: si cambiara entre el
        notebook 03 y el 12, una diferencia de resultados podría venir del
        criterio de parada y no de los datos, que es justo lo que D20 prohíbe.
    monitor_volatilidad
        Criterio de parada de la tarea de **regresión**, donde no hay balanced
        accuracy que monitorizar. Ver Notes.

    Notes
    -----
    `tam_lote` es deliberadamente grande. En CPU el coste por iteración lo
    domina la sobrecarga fija por operación, no el número de operaciones
    aritméticas, así que los lotes pequeños no llegan a llenar las rutinas de
    álgebra lineal. Medido en esta máquina sobre las 3.696 ventanas de train con
    validación: lote 64 cuesta 3,30 s/época, lote 256 cuesta 2,43 y lote 512
    cuesta 2,34. Se adopta 256 y no 512 porque ya recoge el 91 % de la mejora
    disponible —0,87 de los 0,96 segundos que separan al 64 del 512— y el 9 %
    restante costaría dividir por ocho, en lugar de por cuatro, el número de
    actualizaciones de gradiente por época respecto al lote 64 de partida. Ver
    D22 en `docs/DECISIONES.md`.

    En volatilidad se para por `val_mae` y no por `val_loss`, y es el mismo
    principio aplicado a la otra tarea: `CRITERIOS["volatilidad"]` dice que quien
    ordena es el MAE, así que el MAE es quien tiene que parar. La pérdida
    compilada es el MSE, que no es una transformación monótona del MAE —pondera
    el cuadrado del error, así que un único pico de volatilidad mal predicho pesa
    más que muchos errores pequeños—, y su mínimo no tiene por qué caer en la
    misma época. La elección **no** obedece al problema que motiva
    `ParadaPorMetrica`: aquí no hay reponderación ni clases, y la pérdida de
    validación es una medida honesta del objetivo. Es que, pudiendo alinear
    gratis el criterio que para con el que decide, dejarlos distintos solo
    añadiría una fuente de discrepancia que habría que explicar después.
    `val_mae` está en el log de la época porque `construir` compila la regresión
    con ``metrics=["mae"]`, así que monitorizarlo cuesta cero.
    """

    epocas: int = 60
    tam_lote: int = 256
    paciencia: int = 12
    monitor: str = METRICA_SELECCION
    monitor_volatilidad: str = "val_mae"

    def __post_init__(self) -> None:
        # Se valida en la construcción y no en `entrenar` porque este objeto se
        # serializa a disco: un monitor inválido congelado en `arquitectura.json`
        # solo daría la cara en el notebook 12, a mitad del barrido.
        _mayor_mejor(self.monitor)
        _mayor_mejor(self.monitor_volatilidad)
        if not self.monitor_volatilidad.startswith("val_"):
            raise ValueError(
                f"monitor_volatilidad={self.monitor_volatilidad!r} es una métrica de "
                "clasificación. La tarea de volatilidad es una regresión: su criterio "
                "de parada tiene que ser una clave del log de Keras ('val_loss' o "
                "'val_mae')."
            )

    def __str__(self) -> str:
        return (
            f"Presupuesto({self.epocas} épocas · lote {self.tam_lote} · "
            f"paciencia {self.paciencia} · para por {self.monitor}, "
            f"{self.monitor_volatilidad} en volatilidad)"
        )


PRESUPUESTO = Presupuesto()


# ---------------------------------------------------------------------------
# Catálogo de candidatas
# ---------------------------------------------------------------------------
#
# Seis arquitecturas, ni una más. Cada una mueve **un** eje respecto a
# `cnn_base`, de modo que la diferencia de puntuación se pueda atribuir a ese
# eje y no a una combinación de tres cambios simultáneos. Barrer el producto
# cartesiano daría un ganador mejor y un argumento peor: con 3.696 ventanas de
# train y solo 672 de validación, que es donde se puntúa, la diferencia entre las
# mejores configuraciones de una rejilla grande cabe dentro del ruido entre
# semillas, y el "ganador" sería el que tuvo suerte.

CANDIDATAS: dict[str, Arquitectura] = {
    # Control. Es una regresión logística multinomial sobre la ventana aplanada:
    # si empata con las convolucionales, la convolución no aporta nada y todo el
    # coste de la troncal es gasto sin contrapartida.
    "lineal": Arquitectura(filtros=(), unidades_densa=0),
    # ¿Sobra capacidad? Dos bloques y una densa estrecha.
    "cnn_pequena": Arquitectura(filtros=(32, 64), tamano_kernel=3, unidades_densa=64),
    # La de referencia, la que el módulo traía como única opción.
    "cnn_base": Arquitectura(filtros=(64, 128, 128), tamano_kernel=3, unidades_densa=100),
    # ¿Falta capacidad? El doble de canales en cada bloque.
    "cnn_ancha": Arquitectura(filtros=(128, 256, 256), tamano_kernel=3, unidades_densa=128),
    # Campo receptivo: kernel 7 en lugar de 3. Un salto de volatilidad se
    # reconoce mejor mirando semana y media que tres sesiones.
    "cnn_kernel7": Arquitectura(filtros=(64, 128, 128), tamano_kernel=7, unidades_densa=100),
    # Agregación temporal invariante: promedio global en vez de aplanado. Deja
    # de importar en qué día de la ventana ocurrió el patrón.
    "cnn_pool_global": Arquitectura(
        filtros=(64, 128, 128), tamano_kernel=3, unidades_densa=100,
        agregacion="promedio_global",
    ),
}

# La tarea de volatilidad no elige arquitectura (lo hace la de régimen, que es
# donde está la hipótesis del trabajo), pero `buscar` se puede correr sobre ella
# como control de consistencia y necesita un criterio de orden.
CRITERIOS = {
    "regimen": (METRICA_SELECCION, METRICA_DESEMPATE, True),
    "volatilidad": ("mae", "qlike", False),
}


# ---------------------------------------------------------------------------
# Construcción y entrenamiento
# ---------------------------------------------------------------------------


def construir(
    tarea: str,
    forma_entrada: tuple[int, int],
    n_clases: int = 3,
    arquitectura: Arquitectura = ARQUITECTURA,
):
    """Construye el modelo downstream para la tarea indicada.

    Parameters
    ----------
    tarea
        ``regimen`` (clasificación) o ``volatilidad`` (regresión).
    forma_entrada
        ``(pasado, canales)``, típicamente ``(60, 20)``.
    n_clases
        Número de regímenes. Se ignora en la tarea de regresión.
    arquitectura
        Configuración de la troncal. A partir del notebook 03 lo que se pasa
        aquí es lo que devuelve `cargar_congelada()`, no el valor por defecto.

    Returns
    -------
    Modelo de Keras ya compilado.
    """
    import keras
    from keras import layers

    if tarea not in TAREAS:
        raise ValueError(f"Tarea desconocida: {tarea!r}. Opciones: {TAREAS}")

    entrada = keras.Input(shape=forma_entrada, name="ventana")
    x = entrada

    # Troncal convolucional: cada bloque reduce la longitud temporal a la mitad
    # y amplía el número de canales, extrayendo patrones cada vez menos locales.
    # Con `filtros=()` el bucle no se ejecuta y queda el modelo de control.
    for i, n_filtros in enumerate(arquitectura.filtros):
        x = layers.Conv1D(
            filters=n_filtros,
            kernel_size=arquitectura.tamano_kernel,
            activation="relu",
            padding="same",
            name=f"conv_{i}",
        )(x)
        x = layers.MaxPooling1D(pool_size=2, name=f"pool_{i}")(x)

    if arquitectura.agregacion == "promedio_global":
        x = layers.GlobalAveragePooling1D(name="promedio_global")(x)
    else:
        x = layers.Flatten(name="aplanado")(x)

    if arquitectura.unidades_densa:
        x = layers.Dense(arquitectura.unidades_densa, activation="relu", name="densa")(x)

    # El dropout es la única regularización explícita, y se aplica también en la
    # candidata lineal para que la comparación no le regale una ventaja. Importa
    # porque los datasets con mucho sintético son grandes pero poco diversos, y
    # sin él el modelo memoriza las muestras generadas.
    x = layers.Dropout(arquitectura.dropout, name="dropout")(x)

    if tarea == "regimen":
        salida = layers.Dense(n_clases, activation="softmax", name="regimen")(x)
        perdida = "sparse_categorical_crossentropy"
        metricas = ["accuracy"]
    else:
        salida = layers.Dense(1, name="volatilidad")(x)
        perdida = "mse"
        metricas = ["mae"]

    # El nombre no es decorativo: `entrenar` lo usa para saber qué tarea está
    # entrenando cuando la celda que llama no se lo dice.
    modelo = keras.Model(entrada, salida, name=f"downstream_{tarea}")
    modelo.compile(
        optimizer=keras.optimizers.Adam(learning_rate=arquitectura.tasa_aprendizaje),
        loss=perdida,
        metrics=metricas,
    )
    return modelo


def pesos_por_clase(y_reg: np.ndarray, n_clases: int = 3) -> dict[int, float]:
    """Pesos inversamente proporcionales a la frecuencia de cada régimen.

    Es la alternativa clásica a generar datos sintéticos para tratar el
    desbalance, y por eso es una comparación obligada en el análisis (D15): si
    reponderar la pérdida iguala al mejor generador, los sintéticos no aportan
    nada que no se consiguiera gratis.

    Raises
    ------
    ValueError
        Si alguna clase de ``[0, n_clases)`` no tiene ninguna muestra.

    Notes
    -----
    Que una clase esté vacía se trata como error y no como peso especial, y la
    elección es deliberada. Las dos alternativas fallan de formas peores:

    - **Devolver 0,0** —lo que hacía la versión anterior— se lee como "esta
      clase no importa" cuando en realidad significa "esta clase no está". Keras
      lo acepta sin decir nada y entrena un modelo que nunca predecirá el
      régimen de crisis, que es exactamente el fallo que el trabajo mide.
    - **Devolver `np.nan`** propaga a la pérdida: el entrenamiento arranca, la
      curva sale entera a NaN y el diagnóstico aparece sesenta épocas más tarde
      y a varias capas de distancia de su causa.

    Una clase vacía en el conjunto de entrenamiento es un defecto de la receta
    de mezcla, no un caso de uso, y el sitio barato de detectarlo es aquí.
    """
    conteo = np.bincount(np.asarray(y_reg, dtype=int), minlength=n_clases)
    vacias = np.flatnonzero(conteo[:n_clases] == 0)
    if len(vacias):
        raise ValueError(
            f"No se pueden calcular pesos por clase: las clases {vacias.tolist()} "
            f"no tienen ninguna muestra en un conjunto de {int(conteo.sum())}. "
            "Revisa la receta de mezcla o el número de reales de la submuestra."
        )

    total = conteo.sum()
    return {k: float(total / (n_clases * c)) for k, c in enumerate(conteo[:n_clases])}


def _tarea_del_modelo(modelo) -> str:
    """Deduce la tarea del nombre que `construir` puso al modelo.

    No es una heurística: `construir` es el único constructor del proyecto y
    nombra sus modelos ``downstream_<tarea>``, así que la correspondencia es
    exacta. Existe solo para que las celdas escritas antes de que `entrenar`
    aceptara `tarea` sigan funcionando; lo correcto es pasarla explícitamente.
    """
    nombre = getattr(modelo, "name", "") or ""
    for tarea in TAREAS:
        if nombre == f"downstream_{tarea}":
            return tarea
    raise ValueError(
        f"No se puede deducir la tarea del modelo {nombre!r}. Pásala explícitamente: "
        f"entrenar(..., tarea=...). Opciones: {TAREAS}"
    )


_CLASE_PARADA = None


def _clase_parada():
    """Construye una sola vez la clase `ParadaPorMetrica` y la devuelve.

    Existe por la misma razón que las importaciones diferidas de `keras` del
    resto del módulo: la clase hereda de `keras.callbacks.Callback`, y definirla
    en el cuerpo del módulo obligaría a cargar Keras —varios segundos— en
    cualquier notebook que importe `downstream` solo para leer `CANDIDATAS`. El
    `__getattr__` del final del módulo hace que `downstream.ParadaPorMetrica`
    funcione igual que si la clase estuviera escrita aquí.
    """
    global _CLASE_PARADA
    if _CLASE_PARADA is not None:
        return _CLASE_PARADA

    import keras

    from . import evaluacion

    class ParadaPorMetrica(keras.callbacks.Callback):
        """Parada temprana que monitoriza una métrica de `metricas_regimen`.

        `keras.callbacks.EarlyStopping` solo sabe mirar lo que hay en el log de
        la época, y ahí solo están la pérdida y la métrica compilada. En este
        proyecto eso significa parar por `val_loss`, que es entropía cruzada, y
        medido sobre estos datos es el criterio equivocado.

        Cómo se lee: al final de cada época se predice sobre validación, se
        calcula la métrica y se guarda en el log con el prefijo ``val_``, de modo
        que acaba como una columna más del CSV de `guardar_historial` y su curva
        se puede dibujar junto a la de la pérdida. `mejor_epoca` y `mejor_valor`
        —numerada desde 1, como la columna `epoca` del historial— dicen qué
        modelo es el que queda cargado al terminar, que no es el de la última
        época sino el de esa.

        Qué la invalida: que `X_val`/`y_val` no sean los mismos que se pasan a
        `fit` como `validation_data`. Serían dos validaciones distintas, la curva
        de la pérdida y la de la métrica dejarían de ser comparables, y peor: el
        criterio de parada estaría mirando un conjunto que nadie más ve. Y, como
        siempre, que entre ahí el test: el criterio de parada lee este conjunto
        una vez por época, así que pasarle el test es la fuga más directa que
        existe en todo el proyecto.

        Notes
        -----
        **La medición que justifica la clase.** Sale de los 22 entrenamientos
        que el cuaderno 03 deja en `results/historiales/*.csv` y que su última
        celda recorre, así que se puede rehacer sin reentrenar nada: parar por
        `val_loss` en vez de por la métrica que decide cuesta **1,7 puntos de
        balanced accuracy de media** y hasta **6,6 puntos** en el peor caso
        (`cnn_kernel7`, semilla 2). En la arquitectura congelada el mínimo de
        `val_loss` cae en la época 38 y el máximo de balanced accuracy en la 55:
        **2,6 puntos**.

        Una versión anterior de esta nota citaba un experimento auxiliar con
        `cnn_ancha` y pérdida reponderada —mínimo de `val_loss` en la época 1,
        coste de 8,5 puntos—. Ese experimento no está en el repositorio y no es
        reproducible desde los historiales, así que se retira en favor de las
        cifras de arriba (D20).

        La causa es que los repartos de clase de train y validación son
        radicalmente distintos —train 54,6 / 29,5 / 15,9 %, validación 18,3 /
        59,7 / 22,0 %—, así que minimizar la entropía cruzada sobre validación
        premia al modelo que todavía no ha aprendido el prior de train.
        Reponderar la pérdida separa aún más ambos priores y agrava el efecto. La
        balanced accuracy es inmune por construcción: es la media de los recalls
        por clase, y un recall no depende de cuántas muestras haya de esa clase.

        **Diferencia deliberada con `EarlyStopping`.** Keras solo restaura los
        mejores pesos cuando la paciencia llega a dispararse; si el presupuesto
        de épocas se agota antes, deja los de la última época sin decir nada.
        Aquí la restauración va en `on_train_end`, que se ejecuta siempre, así
        que el modelo que queda cargado es el de `mejor_epoca` en los dos casos.

        **Coste.** Un `predict` sobre las 672 ventanas de validación por época.
        Medido en esta máquina con `cnn_ancha` y lote 256: 0,065 s de callback
        frente a 2,58 s de época, un 2,5 % del entrenamiento. El `batch_size` de
        512 no es decorativo: con el 32 por defecto de Keras el mismo `predict`
        tarda algo más del doble, que es la diferencia entre despreciable y
        discutible cuando por delante hay varios cientos de entrenamientos.

        Parameters
        ----------
        X_val, y_val
            El **mismo** conjunto de validación que recibe `fit`.
        monitor
            Clave de `evaluacion.metricas_regimen`, sin el prefijo ``val_``.
        paciencia
            Épocas sin mejora antes de parar.
        n_clases
            Número de regímenes, para que la métrica no se calcule sobre un
            conjunto de clases distinto del que produjo las predicciones.
        mayor_mejor
            Sentido del monitor. Lo fija `MONITORES`, no quien llama.
        tam_lote
            Lote de inferencia. Ver Notes.

        Attributes
        ----------
        mejor_epoca, mejor_valor
            Época restaurada (numerada desde 1) y su valor del monitor.
        epoca_parada
            Época en la que se agotó la paciencia, o 0 si no llegó a agotarse.
        valores
            Serie completa del monitor, la misma que va al log.
        segundos_evaluando
            Coste acumulado del callback, para poder afirmar que es despreciable
            en vez de suponerlo.
        """

        def __init__(
            self,
            X_val: np.ndarray,
            y_val: np.ndarray,
            monitor: str = METRICA_SELECCION,
            paciencia: int = 12,
            n_clases: int = 3,
            mayor_mejor: bool = True,
            tam_lote: int = 512,
        ) -> None:
            super().__init__()
            self.X_val = X_val
            self.y_val = np.asarray(y_val)
            self.monitor = monitor
            self.clave_log = f"val_{monitor}"
            self.paciencia = int(paciencia)
            self.n_clases = int(n_clases)
            self.mayor_mejor = bool(mayor_mejor)
            self.tam_lote = int(tam_lote)
            self.mejor_valor = -np.inf if mayor_mejor else np.inf
            self.mejor_epoca = 0
            self.epoca_parada = 0
            self.valores: list[float] = []
            self.segundos_evaluando = 0.0
            self._mejores_pesos = None
            self._espera = 0

        def on_train_begin(self, logs=None) -> None:
            # Reiniciar el estado no es paranoia: `buscar` construye un callback
            # por entrenamiento, pero una celda de notebook que reutilice el
            # mismo objeto en dos `fit` restauraría al final los pesos del
            # entrenamiento anterior, que ni siquiera son del mismo modelo.
            self.mejor_valor = -np.inf if self.mayor_mejor else np.inf
            self.mejor_epoca = 0
            self.epoca_parada = 0
            self.valores = []
            self.segundos_evaluando = 0.0
            self._mejores_pesos = None
            self._espera = 0

        def _mejora(self, valor: float) -> bool:
            # Comparación estricta: ante un empate exacto gana la época más
            # temprana, que es la más barata y la menos sobreajustada.
            if not np.isfinite(valor):
                return False
            return valor > self.mejor_valor if self.mayor_mejor else valor < self.mejor_valor

        def on_epoch_end(self, epoca: int, logs=None) -> None:
            inicio = time.perf_counter()
            probabilidades = self.model.predict(
                self.X_val, verbose=0, batch_size=self.tam_lote
            )
            metricas = evaluacion.metricas_regimen(
                self.y_val, probabilidades.argmax(axis=1), n_clases=self.n_clases
            )
            self.segundos_evaluando += time.perf_counter() - inicio

            valor = float(metricas[self.monitor])
            self.valores.append(valor)
            # El prefijo `val_` no es cosmético: es lo que hace que la métrica
            # entre en `historial.history`, y de ahí al CSV y a la figura de
            # convergencia. Sin él la curva del criterio de parada no existiría.
            if logs is not None:
                logs[self.clave_log] = valor

            if self._mejora(valor):
                self.mejor_valor = valor
                self.mejor_epoca = epoca + 1
                self._mejores_pesos = self.model.get_weights()
                self._espera = 0
                return

            self._espera += 1
            if self._espera >= self.paciencia:
                self.epoca_parada = epoca + 1
                self.model.stop_training = True

        def on_train_end(self, logs=None) -> None:
            if self._mejores_pesos is not None:
                self.model.set_weights(self._mejores_pesos)

    _CLASE_PARADA = ParadaPorMetrica
    return _CLASE_PARADA


def __getattr__(nombre: str):
    """Expone `ParadaPorMetrica` sin cargar Keras al importar el módulo (PEP 562)."""
    if nombre == "ParadaPorMetrica":
        return _clase_parada()
    raise AttributeError(f"module {__name__!r} has no attribute {nombre!r}")


def _crear_parada(
    tarea: str,
    monitor: str,
    paciencia: int,
    X_val: np.ndarray,
    y_val: np.ndarray,
    n_clases: int,
):
    """Devuelve el callback de parada que corresponde al criterio pedido.

    Las claves con prefijo ``val_`` ya están en el log de la época y las atiende
    `EarlyStopping` sin coste; el resto son métricas de `metricas_regimen` y las
    calcula `ParadaPorMetrica`. Mantener las dos vías tiene un motivo concreto:
    `val_loss` es el criterio anterior, y poder pedirlo explícitamente es lo que
    permite reproducir la medición de la tabla de `ParadaPorMetrica` en lugar de
    tener que creérsela.
    """
    import keras

    mayor_mejor = _mayor_mejor(monitor)

    if monitor.startswith("val_"):
        return keras.callbacks.EarlyStopping(
            monitor=monitor,
            patience=paciencia,
            mode="max" if mayor_mejor else "min",
            restore_best_weights=True,
            verbose=0,
        )

    if tarea != "regimen":
        raise ValueError(
            f"El criterio de parada {monitor!r} es una métrica de clasificación y la "
            f"tarea es {tarea!r}. Usa 'val_loss' o 'val_mae'."
        )

    return _clase_parada()(
        X_val, y_val, monitor=monitor, paciencia=paciencia,
        n_clases=n_clases, mayor_mejor=mayor_mejor,
    )


def entrenar(
    modelo,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    tarea: str | None = None,
    presupuesto: Presupuesto = PRESUPUESTO,
    epocas: int | None = None,
    tam_lote: int | None = None,
    paciencia: int | None = None,
    monitor: str | None = None,
    verboso: int = 0,
    usar_pesos: bool = False,
    n_clases: int | None = None,
):
    """Entrena el modelo con parada temprana y devuelve el historial.

    La parada temprana restaura los pesos de la mejor época de validación. Sin
    ella, los datasets con muchos sintéticos sobreajustan y la comparación
    mediría cuánto sobreajusta cada generador, no cuánto aporta.

    El criterio de "mejor época" lo fija el presupuesto y **no** es `val_loss` en
    la tarea de régimen: es la balanced accuracy de validación, la misma métrica
    que elige la arquitectura. Ver `ParadaPorMetrica` para la medición que lo
    justifica —parar por `val_loss` cuesta 1,7 puntos de balanced accuracy de
    media sobre los 22 entrenamientos del cuaderno 03, y hasta 6,6 en el peor
    caso— y `Presupuesto` para el criterio de la tarea de volatilidad.

    El conjunto de validación es **siempre real**, en todas las versiones. Solo
    el conjunto de entrenamiento cambia entre experimentos.

    Parameters
    ----------
    tarea
        ``regimen`` o ``volatilidad``. Si se omite se deduce del nombre del
        modelo, que `construir` fija; pasarla es lo correcto.
    presupuesto
        Épocas, tamaño de lote, paciencia y criterio de parada. Es el camino por
        defecto y forma parte de lo que se congela junto con la arquitectura
        (D20).
    epocas, tam_lote, paciencia, monitor
        Sobrescriben el presupuesto de uno en uno. Existen para pruebas cortas
        y para las celdas antiguas; usarlos en el barrido rompe la premisa de
        que entre versiones solo cambian los datos.
    usar_pesos
        Reponderación inversa a la frecuencia de clase (D15). Solo tiene sentido
        en la tarea de régimen.

    Returns
    -------
    keras.callbacks.History
        Con `loss` y `val_loss` como siempre, más la curva del criterio de
        parada cuando lo calcula `ParadaPorMetrica` (columna
        ``val_balanced_accuracy``). Se le engancha además el atributo `parada`
        con el callback usado, que es de donde salen `mejor_epoca` y
        `mejor_valor`; en la tarea de volatilidad ese atributo es un
        `EarlyStopping` de Keras y sus campos se llaman `best_epoch` (contada
        desde 0) y `best`.

    Raises
    ------
    ValueError
        Si `tarea` no está en `TAREAS`, o si se piden pesos por clase en la
        tarea de volatilidad. La versión anterior decidía si aplicarlos mirando
        si la salida tenía más de una unidad, lo que convertía
        ``usar_pesos=True`` en la regresión en un no-op **silencioso**: quien
        escribiera la variante reponderada creería haberla aplicado y estaría
        comparando dos entrenamientos idénticos.
    """
    tarea = _tarea_del_modelo(modelo) if tarea is None else tarea
    if tarea not in TAREAS:
        raise ValueError(f"Tarea desconocida: {tarea!r}. Opciones: {TAREAS}")
    if usar_pesos and tarea != "regimen":
        raise ValueError(
            "usar_pesos=True no es aplicable a la tarea 'volatilidad': es una "
            "regresión y no tiene clases que reponderar. Si buscabas la variante "
            "de D15, entrénala sobre la tarea 'regimen'."
        )

    epocas = presupuesto.epocas if epocas is None else epocas
    tam_lote = presupuesto.tam_lote if tam_lote is None else tam_lote
    paciencia = presupuesto.paciencia if paciencia is None else paciencia
    if monitor is None:
        monitor = (
            presupuesto.monitor if tarea == "regimen" else presupuesto.monitor_volatilidad
        )

    # Aquí `output_shape[-1]` es la anchura declarada de la capa de salida, no
    # una conjetura sobre qué problema se resuelve: la tarea ya está decidida
    # arriba. La usan tanto los pesos por clase como el callback de parada, y
    # tiene que ser la misma en los dos: si el callback calculara la métrica
    # sobre otro número de clases, el criterio de parada y el de evaluación
    # estarían midiendo cosas distintas.
    n_clases = modelo.output_shape[-1] if n_clases is None else n_clases

    parada = _crear_parada(tarea, monitor, paciencia, X_val, y_val, n_clases)

    kwargs = {}
    if usar_pesos:
        kwargs["class_weight"] = pesos_por_clase(y_train, n_clases)

    historial = modelo.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=epocas,
        batch_size=tam_lote,
        callbacks=[parada],
        verbose=verboso,
        **kwargs,
    )
    # El callback viaja con el historial porque es donde están la época
    # restaurada y su valor, y `fit` no los devuelve por ningún otro camino.
    # Engancharlo aquí evita que cada celda tenga que construir el callback a
    # mano solo para poder informar qué época se quedó.
    historial.parada = parada
    return historial


def guardar_historial(historial, nombre: str) -> None:
    """Vuelca la curva de entrenamiento a `results/historiales/<nombre>.csv`.

    Persistir las curvas es lo que permite regenerar cualquier figura del
    informe sin reentrenar. Con varios cientos de entrenamientos en el barrido,
    reentrenar para rehacer un gráfico no es una opción.
    """
    DIR_HISTORIALES.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(historial.history)
    df.insert(0, "epoca", np.arange(1, len(df) + 1))
    df.to_csv(DIR_HISTORIALES / f"{nombre}.csv", index=False)


# ---------------------------------------------------------------------------
# Búsqueda de arquitectura
# ---------------------------------------------------------------------------


def _objetivo(conjunto: "Conjunto", tarea: str) -> np.ndarray:
    return conjunto.y_reg if tarea == "regimen" else conjunto.y_vol


def buscar(
    candidatas: dict[str, Arquitectura],
    train: "Conjunto",
    val: "Conjunto",
    tarea: str,
    n_clases: int = 3,
    semillas: tuple[int, ...] = (0, 1, 2),
    presupuesto: Presupuesto = PRESUPUESTO,
    guardar_historiales: bool = True,
) -> pd.DataFrame:
    """Entrena cada candidata con cada semilla y la puntúa sobre validación.

    Es el "se buscará una arquitectura válida" del enunciado. Cada fila del
    resultado es un entrenamiento completo y trazable: su historial queda en
    `results/historiales/busqueda__<clave>__s<semilla>__<tarea>.csv`, de modo que
    las curvas de **todos** los entrenamientos se pueden mostrar sin reentrenar
    nada. En la tarea de régimen ese CSV trae además la columna
    ``val_balanced_accuracy``, que es la curva del criterio que decide dónde para
    cada entrenamiento y no la de la pérdida.

    Cómo se lee: una fila por par (candidata, semilla). `parametros` y
    `segundos` son la restricción de coste (D19, D22) y están para poder
    defender que la ganadora no se eligió por ser la más cara ni se descartó por
    serlo; `epocas` son las efectivas tras la parada temprana, y si igualan al
    presupuesto la candidata no había convergido. Las métricas son **de
    validación**: no son las cifras del informe, son las que deciden la
    arquitectura. La dispersión entre semillas de una misma candidata es la vara
    de medir: dos candidatas separadas por menos de lo que se mueve una sola al
    cambiar de semilla no están separadas por nada.

    Qué la invalida: que se le pase como `val` algo que no sea real y previo al
    test, o que se reutilice la tabla tras haber tocado `train`. El conjunto de
    test **no puede entrar aquí**: no hay ningún parámetro por el que colarlo, y
    esa ausencia es deliberada, porque elegir la arquitectura mirando el test
    convierte el test en un segundo conjunto de validación y todo el resto del
    trabajo pierde su referencia independiente.

    Notes
    -----
    La métrica de selección en la tarea de régimen es `balanced_accuracy`, con
    `recall_crisis` como desempate cuando dos candidatas quedan dentro de
    `MARGEN_EMPATE`. No es el f1-macro, y la razón es medible: validación tiene
    un 22,0 % de ventanas de crisis y test un 10,5 %. Bajo ese desplazamiento de
    prevalencia el f1-macro se mueve entre -0,013 y +0,044 según lo liberal que
    sea el modelo con la clase de crisis, e **invierte el orden del 8,1 % de los
    pares** de candidatas; la balanced accuracy es exactamente invariante
    (diferencia 0,0 en toda la rejilla probada), porque es la media de los
    recalls por clase y un recall no depende de cuántas muestras haya de esa
    clase. Elegir con f1-macro sobre una validación con doble prevalencia de
    crisis que el test es elegir con una regla que sabemos que se romperá al
    cambiar de tramo. El f1-macro se sigue reportando en la tabla, solo que no
    decide.

    Antes de **cada** construcción se llama a `config.fijar_semillas(semilla)`.
    Sin eso, el primer modelo de cada kernel recibe una inicialización distinta
    de la que recibiría en cualquier otra posición del bucle, y la primera
    candidata del catálogo se evaluaría con una semilla efectiva que ninguna
    otra comparte (ver el docstring de `config.fijar_semillas`).

    Parameters
    ----------
    candidatas
        Diccionario ``clave · Arquitectura``, típicamente `CANDIDATAS`.
    train, val
        Particiones reales. `val` es la que puntúa.
    tarea
        ``regimen`` o ``volatilidad``.
    semillas
        Repeticiones por candidata. Tres es el mínimo para que la desviación
        entre semillas signifique algo.

    Returns
    -------
    pandas.DataFrame
        Columnas ``candidata``, ``semilla``, ``arquitectura``, ``parametros``,
        ``epocas``, ``segundos``, ``segundos_por_epoca``, todas las métricas que
        devuelva `evaluacion.evaluar` para la tarea, y ``conjunto``, que vale
        siempre ``validacion``.
    """
    if tarea not in TAREAS:
        raise ValueError(f"Tarea desconocida: {tarea!r}. Opciones: {TAREAS}")

    # Importación diferida por el mismo motivo que la de keras: mantener barato
    # el `import src.downstream` de los notebooks que no entrenan nada.
    from . import config, evaluacion

    y_train = _objetivo(train, tarea)
    y_val = _objetivo(val, tarea)
    forma_entrada = train.X.shape[1:]

    filas = []
    for clave, arq in candidatas.items():
        for sem in semillas:
            config.fijar_semillas(sem)
            modelo = construir(tarea, forma_entrada, n_clases=n_clases, arquitectura=arq)

            inicio = time.perf_counter()
            historial = entrenar(
                modelo, train.X, y_train, val.X, y_val,
                tarea=tarea, presupuesto=presupuesto,
            )
            segundos = time.perf_counter() - inicio

            if guardar_historiales:
                guardar_historial(historial, f"busqueda__{clave}__s{sem}__{tarea}")

            epocas = len(historial.history["loss"])
            filas.append(
                {
                    "candidata": clave,
                    "semilla": sem,
                    "arquitectura": str(arq),
                    "parametros": int(modelo.count_params()),
                    "epocas": epocas,
                    "segundos": round(segundos, 1),
                    "segundos_por_epoca": round(segundos / max(epocas, 1), 2),
                    # La llamada es posicional a propósito: `evaluacion` es de
                    # otro módulo y así no se depende de sus nombres de
                    # parámetro. La etiqueta se escribe **después** del volcado
                    # para que sea esta función quien responda de ella: aquí
                    # solo se puntúa sobre validación, y la columna tiene que
                    # decirlo aunque el valor por defecto de `evaluar` cambie.
                    **evaluacion.evaluar(modelo, val.X, y_val, tarea),
                    "conjunto": "validacion",
                }
            )

    return pd.DataFrame(filas)


def _criterio(tabla: pd.DataFrame) -> tuple[str, str, bool]:
    """Métrica de orden, desempate y si mayor es mejor, según lo que traiga la tabla."""
    for tarea, (metrica, desempate, mayor_mejor) in CRITERIOS.items():
        if metrica in tabla.columns:
            return metrica, desempate, mayor_mejor
    raise ValueError(
        "La tabla no contiene ninguna métrica de selección conocida "
        f"({[m for m, _, _ in CRITERIOS.values()]}). "
        "Comprueba que procede de `buscar()` y no de un CSV recortado."
    )


def resumen_busqueda(tabla: pd.DataFrame) -> pd.DataFrame:
    """Agrega la tabla de `buscar` por candidata: media y desviación entre semillas.

    Cómo se lee: una fila por candidata, ordenada de mejor a peor por la métrica
    de selección. De cada columna numérica se dan `_media` y `_desv`, y la
    segunda es la que evita la conclusión falsa: si la distancia entre la
    primera y la segunda candidata es menor que su desviación entre semillas, lo
    que separa a ambas es la inicialización, no la arquitectura, y hay que
    elegir por el criterio secundario —coste— y decirlo. `n_semillas` avisa de
    si alguna candidata se entrenó menos veces que las demás, en cuyo caso su
    desviación no es comparable.

    Qué la invalida: mezclar en la misma tabla dos tareas, dos particiones o dos
    presupuestos. Las medias se calcularían sobre cosas que no son la misma
    magnitud y el orden resultante no significaría nada.
    """
    metrica, _, mayor_mejor = _criterio(tabla)

    numericas = [
        c for c in tabla.select_dtypes("number").columns if c != "semilla"
    ]
    agregado = tabla.groupby("candidata", sort=False)[numericas].agg(["mean", "std"])
    agregado.columns = [
        f"{col}_{'media' if est == 'mean' else 'desv'}" for col, est in agregado.columns
    ]
    agregado.insert(0, "n_semillas", tabla.groupby("candidata", sort=False).size())
    return agregado.sort_values(f"{metrica}_media", ascending=not mayor_mejor)


def elegir(
    tabla: pd.DataFrame, candidatas: dict[str, Arquitectura] = CANDIDATAS
) -> tuple[str, Arquitectura]:
    """Aplica la regla de selección sobre la tabla de `buscar`.

    La regla completa, en orden: se ordena por la media entre semillas de la
    métrica de selección; se toman como empatadas todas las candidatas que
    queden dentro de `MARGEN_EMPATE` de la mejor; y entre ellas gana la de mayor
    `recall_crisis` medio, porque es la magnitud que decide si el trabajo tiene
    sentido (D17). Si el desempate tampoco separa, gana la de menos parámetros:
    a igualdad de resultado, la barata (D19).

    Qué la invalida: que la tabla no venga de `buscar` sobre validación, o que
    `candidatas` no sea el mismo catálogo con el que se generó. Lo segundo se
    detecta y lanza, porque devolver una `Arquitectura` que no es la que se
    midió sería el peor fallo posible de esta función.

    Returns
    -------
    tuple
        ``(clave, Arquitectura)`` de la ganadora.
    """
    metrica, desempate, mayor_mejor = _criterio(tabla)
    resumen = resumen_busqueda(tabla)

    columna = f"{metrica}_media"
    tope = resumen[columna].iloc[0]
    empatadas = resumen[(resumen[columna] - tope).abs() <= MARGEN_EMPATE]

    if len(empatadas) > 1:
        col_desempate = f"{desempate}_media"
        if col_desempate in empatadas.columns:
            mejor_desempate = (
                empatadas[col_desempate].max() if mayor_mejor
                else empatadas[col_desempate].min()
            )
            empatadas = empatadas[
                (empatadas[col_desempate] - mejor_desempate).abs() <= MARGEN_EMPATE
            ]
        # Último criterio: a igualdad de resultado, la más barata.
        if len(empatadas) > 1 and "parametros_media" in empatadas.columns:
            empatadas = empatadas.loc[[empatadas["parametros_media"].idxmin()]]

    clave = str(empatadas.index[0])
    if clave not in candidatas:
        raise ValueError(
            f"La candidata ganadora {clave!r} no está en el catálogo recibido "
            f"({sorted(candidatas)}). La tabla y el catálogo no se corresponden."
        )
    return clave, candidatas[clave]


# ---------------------------------------------------------------------------
# Congelación en disco
# ---------------------------------------------------------------------------

RUTA_ARQUITECTURA = DIR_MODELOS_DOWN / "arquitectura.json"


def huella(arq: Arquitectura, presupuesto: Presupuesto | None = None) -> str:
    """Identificador reproducible de una configuración congelada.

    Se resume el **contenido** de los campos, no el objeto ni el fichero, con el
    mismo criterio que `datos.huella`: SHA-256 y los doce primeros caracteres
    hexadecimales. Sirve para lo único para lo que sirve una huella: decir en
    una línea si dos ejecuciones usaron la misma configuración, sin comparar
    campo a campo ni fiarse de que nadie editó el módulo entre medias.

    Parameters
    ----------
    presupuesto
        Si se pasa, entra en la huella. `congelar` y `cargar_congelada` lo pasan
        siempre, y no es un detalle: D20 congela la arquitectura **y** el
        presupuesto, así que una huella que solo cubriera la arquitectura
        dejaría pasar sin avisar dos ejecuciones con criterios de parada
        distintos, que es exactamente la diferencia que la congelación existe
        para impedir. Omitirlo da la huella de la arquitectura sola, que es lo
        que se quiere al comparar candidatas entre sí.
    """
    import hashlib

    contenido: dict = {"arquitectura": asdict(arq)}
    if presupuesto is not None:
        contenido["presupuesto"] = asdict(presupuesto)

    serializado = json.dumps(contenido, sort_keys=True, ensure_ascii=True)
    return hashlib.sha256(serializado.encode("utf-8")).hexdigest()[:12]


def congelar(
    arq: Arquitectura,
    presupuesto: Presupuesto = PRESUPUESTO,
    ruta: Path | None = None,
) -> Path:
    """Escribe la arquitectura elegida y su presupuesto en `models/downstream/`.

    D20 dice que la arquitectura se busca en el notebook 03 y no se vuelve a
    tocar, pero mientras el notebook 12 llamara a `construir()` sin pasar
    `arquitectura=`, lo que heredaba era el valor por defecto **del código
    fuente**: cualquier edición del módulo entre un notebook y otro cambiaba el
    experimento sin que nada lo detectara. Con la configuración en disco, el 12
    lee lo que el 03 decidió y la huella delata cualquier discrepancia.

    Returns
    -------
    pathlib.Path
        La ruta escrita.
    """
    ruta = RUTA_ARQUITECTURA if ruta is None else Path(ruta)
    ruta.parent.mkdir(parents=True, exist_ok=True)

    contenido = {
        "arquitectura": asdict(arq),
        "presupuesto": asdict(presupuesto),
        "huella": huella(arq, presupuesto),
    }
    ruta.write_text(json.dumps(contenido, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    return ruta


def cargar_congelada(ruta: Path | None = None) -> tuple[Arquitectura, Presupuesto]:
    """Lee la arquitectura y el presupuesto congelados por el notebook 03.

    Raises
    ------
    FileNotFoundError
        Si no se ha congelado nada todavía.
    ValueError
        Si el contenido no cuadra con su huella. Ocurre si el fichero se editó a
        mano o si los campos de `Arquitectura` o `Presupuesto` cambiaron desde
        que se congeló, que es justo el caso que esta función existe para
        cazar: entrenar el barrido con una arquitectura distinta de la que el
        informe dice que se eligió.

    Notes
    -----
    El caso peligroso es el silencioso, y por eso la huella cubre también el
    presupuesto. Un `arquitectura.json` escrito antes de que existiera el campo
    `monitor` reconstruye un `Presupuesto` perfectamente válido —los dataclass
    rellenan los campos que faltan con su valor por defecto—, de modo que el
    notebook 12 heredaría un criterio de parada que el 03 nunca decidió y nada
    lo delataría. Con el presupuesto dentro de la huella, ese fichero no valida y
    obliga a volver a congelar.
    """
    ruta = RUTA_ARQUITECTURA if ruta is None else Path(ruta)
    if not ruta.exists():
        raise FileNotFoundError(
            f"No existe {ruta}. Ejecuta antes notebooks/03_downstream_baseline.ipynb, "
            "que es donde se busca la arquitectura y se congela."
        )

    contenido = json.loads(ruta.read_text(encoding="utf-8"))
    try:
        campos = dict(contenido["arquitectura"])
        # JSON no distingue tupla de lista y `Arquitectura` es hashable: sin esta
        # conversión el dataclass dejaría de serlo y la huella no coincidiría.
        campos["filtros"] = tuple(campos["filtros"])
        arq = Arquitectura(**campos)
        presupuesto = Presupuesto(**contenido["presupuesto"])
    except (KeyError, TypeError) as error:
        raise ValueError(
            f"{ruta} no tiene la estructura esperada ({error}). Los campos de "
            "Arquitectura o Presupuesto han cambiado desde que se congeló: vuelve a "
            "ejecutar notebooks/03_downstream_baseline.ipynb."
        ) from error

    if huella(arq, presupuesto) != contenido.get("huella"):
        raise ValueError(
            f"La configuración de {ruta} no coincide con su huella "
            f"({huella(arq, presupuesto)} frente a {contenido.get('huella')}). El fichero "
            "se editó a mano o los campos de Arquitectura o Presupuesto cambiaron: vuelve "
            "a ejecutar notebooks/03_downstream_baseline.ipynb en lugar de corregirlo a "
            "mano."
        )

    return arq, presupuesto
