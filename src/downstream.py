"""Modelo downstream: la arquitectura que mide el valor de los datos sintéticos.

El enunciado del taller es explícito: se busca **una** arquitectura válida con
datos reales y luego se entrena esa misma arquitectura, sin tocarla, sobre cada
dataset real/sintético. La comparación solo es honesta si lo único que cambia
entre versiones son los datos.

Por eso este módulo expone una única función constructora y los notebooks no
definen capas por su cuenta. La arquitectura se fija en el notebook 03 y a
partir de ahí queda congelada.

Se resuelven dos tareas sobre la misma ventana de entrada:

``regimen``  clasificación del régimen dominante a 21 días (3 clases)
``volatilidad``  regresión de la volatilidad realizada a 21 días

Comparten troncal (una CNN 1D sobre el eje temporal, en la línea de los
notebooks del máster) y solo difieren en la capa de salida y la pérdida. Que la
troncal sea idéntica permite atribuir cualquier diferencia entre tareas a la
naturaleza del problema, no a la arquitectura.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import DIR_HISTORIALES

TAREAS = ("regimen", "volatilidad")


@dataclass(frozen=True)
class Arquitectura:
    """Hiperparámetros de la troncal. Se fijan en el notebook 03 y no se tocan.

    Los valores por defecto están dimensionados para CPU: con ~4.000 ventanas
    de (60, 20) una época tarda del orden de un segundo, que es lo que hace
    viable el barrido de varios cientos de entrenamientos.
    """

    filtros: tuple[int, ...] = (64, 128, 128)
    tamano_kernel: int = 3
    unidades_densa: int = 100
    dropout: float = 0.3
    tasa_aprendizaje: float = 1e-3

    def __str__(self) -> str:
        return (
            f"CNN1D(filtros={self.filtros}, kernel={self.tamano_kernel}, "
            f"densa={self.unidades_densa}, dropout={self.dropout})"
        )


ARQUITECTURA = Arquitectura()


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
    for i, n_filtros in enumerate(arquitectura.filtros):
        x = layers.Conv1D(
            filters=n_filtros,
            kernel_size=arquitectura.tamano_kernel,
            activation="relu",
            padding="same",
            name=f"conv_{i}",
        )(x)
        x = layers.MaxPooling1D(pool_size=2, name=f"pool_{i}")(x)

    x = layers.Flatten(name="aplanado")(x)
    x = layers.Dense(arquitectura.unidades_densa, activation="relu", name="densa")(x)
    # El dropout es la única regularización explícita. Importa porque los
    # datasets con mucho sintético son grandes pero poco diversos, y sin él el
    # modelo memoriza las muestras generadas.
    x = layers.Dropout(arquitectura.dropout, name="dropout")(x)

    if tarea == "regimen":
        salida = layers.Dense(n_clases, activation="softmax", name="regimen")(x)
        perdida = "sparse_categorical_crossentropy"
        metricas = ["accuracy"]
    else:
        salida = layers.Dense(1, name="volatilidad")(x)
        perdida = "mse"
        metricas = ["mae"]

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
    desbalance, y por eso es una comparación obligada en el análisis: si
    reponderar la pérdida iguala al mejor generador, los sintéticos no aportan
    nada que no se consiguiera gratis.
    """
    conteo = np.bincount(np.asarray(y_reg, dtype=int), minlength=n_clases)
    total = conteo.sum()
    return {
        k: float(total / (n_clases * c)) if c > 0 else 0.0
        for k, c in enumerate(conteo)
    }


def entrenar(
    modelo,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    epocas: int = 60,
    tam_lote: int = 256,
    paciencia: int = 12,
    verboso: int = 0,
    usar_pesos: bool = False,
):
    """Entrena el modelo con parada temprana y devuelve el historial.

    La parada temprana restaura los pesos de la mejor época de validación. Sin
    ella, los datasets con muchos sintéticos sobreajustan y la comparación
    mediría cuánto sobreajusta cada generador, no cuánto aporta.

    El conjunto de validación es **siempre real**, en todas las versiones. Solo
    el conjunto de entrenamiento cambia entre experimentos.

    Notes
    -----
    `tam_lote` es deliberadamente grande. En CPU el coste por iteración lo
    domina la sobrecarga fija por operación, no el número de operaciones
    aritméticas, así que lotes pequeños desaprovechan la máquina. Con 570
    entrenamientos por delante, la diferencia decide si el barrido cabe en una
    tarde o en un fin de semana. No lo bajes sin medir (ver D22 en
    `docs/DECISIONES.md`).

    El valor es el mismo en todas las versiones del experimento: forma parte del
    presupuesto de optimización congelado, igual que la arquitectura.
    """
    import keras

    parada = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=paciencia,
        restore_best_weights=True,
        verbose=0,
    )

    kwargs = {}
    if usar_pesos and modelo.output_shape[-1] > 1:
        kwargs["class_weight"] = pesos_por_clase(y_train, modelo.output_shape[-1])

    return modelo.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=epocas,
        batch_size=tam_lote,
        callbacks=[parada],
        verbose=verboso,
        **kwargs,
    )


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
