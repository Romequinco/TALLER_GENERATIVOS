"""Construcción de los datasets mixtos real/sintético.

Aquí se define la rejilla experimental del taller. El enunciado pide "datasets
que tengan distinto porcentaje de datos sintéticos y reales", y este módulo la
materializa sobre dos ejes, no uno:

**Eje 1 — proporción de sintéticos.** Con el conjunto real fijo, se añaden
0, 0.5×, 1×, 2× y 5× muestras sintéticas.

**Eje 2 — escasez de datos reales.** Se repite todo lo anterior partiendo de
250, 500, 1000, 2000 y todas las ventanas reales.

El segundo eje es imprescindible. El material del taller muestra que el
beneficio de los sintéticos se concentra en el régimen de pocos datos y se
desvanece —o se vuelve negativo— cuando ya hay muchos reales. Barrer solo el
porcentaje de sintéticos con todos los reales disponibles produciría una línea
plana y la conclusión errónea de que los generadores no sirven.

Además se distinguen dos políticas de reparto por clase, porque la hipótesis
central del trabajo es que el valor de los sintéticos está en la clase rara:

``proporcional``  el sintético replica el desbalance real. Mide el efecto de
                  "más datos" sin más.
``equilibrado``   el sintético se concentra en las clases minoritarias hasta
                  igualarlas. Mide el efecto de "más datos donde hacen falta".
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import numpy as np

from .ventanas import Conjunto

# ── Rejilla experimental ─────────────────────────────────────────────────────
#
# Estas listas gobiernan el tamaño del barrido del notebook 12. Reducirlas es
# la palanca para acortar el cómputo si el calendario aprieta; si se recortan,
# hay que decirlo explícitamente en el informe.

NIVELES_REALES: tuple[int | None, ...] = (250, 500, 1000, 2000, None)  # None = todos
RATIOS_SINTETICOS: tuple[float, ...] = (0.0, 0.5, 1.0, 2.0, 5.0)
POLITICAS: tuple[str, ...] = ("proporcional", "equilibrado")


@dataclass(frozen=True)
class Receta:
    """Especificación de un dataset del barrido.

    Su `identificador` da nombre al fichero de historial y a la fila de la tabla
    de resultados, de modo que cualquier punto de cualquier figura se puede
    rastrear hasta el experimento que lo produjo.
    """

    generador: str
    n_reales: int | None
    ratio: float
    politica: str = "proporcional"

    @property
    def identificador(self) -> str:
        reales = "todos" if self.n_reales is None else str(self.n_reales)
        # El ratio se escribe sin punto decimal para no ensuciar los nombres de
        # fichero en Windows: 0.5 -> 0p5.
        ratio = str(self.ratio).replace(".", "p")
        return f"{self.generador}__r{reales}__s{ratio}__{self.politica}"

    @property
    def etiqueta(self) -> str:
        reales = "todos" if self.n_reales is None else f"{self.n_reales} reales"
        return f"{reales} + {self.ratio:g}× sintéticos"

    @property
    def es_referencia(self) -> bool:
        """`True` si no lleva sintéticos: es la línea base contra la que se compara."""
        return self.ratio == 0.0


def rejilla(
    generadores: list[str],
    niveles_reales: tuple[int | None, ...] = NIVELES_REALES,
    ratios: tuple[float, ...] = RATIOS_SINTETICOS,
    politicas: tuple[str, ...] = POLITICAS,
) -> list[Receta]:
    """Genera todas las recetas del barrido, sin duplicar las de referencia.

    Las combinaciones con ``ratio=0`` no dependen del generador ni de la
    política —son solo datos reales—, así que se emiten una sola vez con el
    generador nominal ``solo_real``. Sin esta deduplicación se entrenaría el
    mismo modelo 14 veces por nivel de reales.
    """
    recetas = [
        Receta(generador="solo_real", n_reales=n, ratio=0.0, politica="proporcional")
        for n in niveles_reales
    ]

    recetas += [
        Receta(generador=g, n_reales=n, ratio=r, politica=p)
        for g, n, r, p in product(generadores, niveles_reales, ratios, politicas)
        if r > 0.0
    ]
    return recetas


# ── Reparto de muestras sintéticas por régimen ───────────────────────────────


def repartir(
    y_reg_real: np.ndarray,
    n_sinteticos: int,
    n_regimenes: int,
    politica: str = "proporcional",
) -> dict[int, int]:
    """Decide cuántas muestras sintéticas generar de cada régimen.

    Parameters
    ----------
    y_reg_real
        Regímenes del conjunto real que se va a ampliar.
    n_sinteticos
        Total de muestras sintéticas a repartir.
    politica
        ``proporcional`` mantiene el desbalance observado.
        ``equilibrado`` reparte para acercar el conjunto ampliado a clases
        iguales, dando prioridad a las más escasas.

    Returns
    -------
    Diccionario ``{regimen: n_muestras}`` listo para `generate_dataset`.
    """
    conteo = np.bincount(np.asarray(y_reg_real, dtype=int), minlength=n_regimenes)

    if politica == "proporcional":
        pesos = conteo / conteo.sum()

    elif politica == "equilibrado":
        # Se apunta a que cada clase alcance el tamaño de la mayoritaria; el
        # déficit de cada una marca cuántos sintéticos merece.
        objetivo = conteo.max()
        deficit = np.maximum(objetivo - conteo, 0).astype(float)
        # Si el conjunto ya está equilibrado no hay déficit: se cae en reparto
        # uniforme para no dividir por cero.
        pesos = deficit / deficit.sum() if deficit.sum() > 0 else np.full(n_regimenes, 1 / n_regimenes)

    else:
        raise ValueError(f"Política desconocida: {politica!r}. Opciones: {POLITICAS}")

    # Reparto por mayor resto: asigna la parte entera y reparte los sobrantes
    # entre las clases con mayor fracción pendiente, de modo que la suma sea
    # exactamente `n_sinteticos`.
    exacto = pesos * n_sinteticos
    asignado = np.floor(exacto).astype(int)
    sobrantes = n_sinteticos - asignado.sum()
    if sobrantes > 0:
        orden = np.argsort(-(exacto - asignado))
        asignado[orden[:sobrantes]] += 1

    return {k: int(asignado[k]) for k in range(n_regimenes)}


# ── Montaje del dataset ──────────────────────────────────────────────────────


def montar(
    receta: Receta,
    real: Conjunto,
    sintetico: tuple[np.ndarray, np.ndarray] | None,
    ventanas,
    n_regimenes: int,
    semilla: int = 42,
) -> Conjunto:
    """Construye el conjunto de entrenamiento que describe la receta.

    Parameters
    ----------
    real
        Conjunto real de entrenamiento completo, ya escalado.
    sintetico
        Par ``(bloques, y_reg)`` con el banco de muestras del generador, tal y
        como lo dejó `GeneradorSintetico.exportar_muestras`. Se ignora si la
        receta es de referencia.

    Returns
    -------
    Conjunto mezclado y barajado, listo para `downstream.entrenar`.

    Notes
    -----
    Las muestras sintéticas se toman del banco ya generado en vez de invocar al
    generador. Así el barrido no depende de que los siete modelos estén
    cargados en memoria, y dos ejecuciones del notebook 12 usan exactamente las
    mismas muestras.
    """
    rng = np.random.default_rng(semilla)

    base = real if receta.n_reales is None else real.submuestra(receta.n_reales, semilla)

    if receta.es_referencia:
        return base

    if sintetico is None:
        raise ValueError(
            f"La receta {receta.identificador} pide sintéticos pero no se han "
            f"proporcionado muestras de '{receta.generador}'."
        )

    bloques_sint, y_sint = sintetico
    n_pedidos = int(round(receta.ratio * len(base)))
    reparto = repartir(base.y_reg, n_pedidos, n_regimenes, receta.politica)

    seleccion_bloques, seleccion_reg = [], []
    for regimen, cuota in reparto.items():
        if cuota <= 0:
            continue
        disponibles = np.flatnonzero(y_sint == regimen)
        if len(disponibles) == 0:
            print(
                f"Aviso [{receta.identificador}]: no hay muestras sintéticas "
                f"del régimen {regimen}; se omiten {cuota}."
            )
            continue
        # Con reemplazo: en la clase de crisis se piden más muestras de las que
        # el banco contiene, que es precisamente el escenario de interés.
        idx = rng.choice(disponibles, size=cuota, replace=len(disponibles) < cuota)
        seleccion_bloques.append(bloques_sint[idx])
        seleccion_reg.append(np.full(cuota, regimen, dtype=np.int64))

    if not seleccion_bloques:
        return base

    from .ventanas import desempaquetar

    ampliacion = desempaquetar(
        np.concatenate(seleccion_bloques), np.concatenate(seleccion_reg), ventanas
    )
    return concatenar(base, ampliacion, rng)


def concatenar(a: Conjunto, b: Conjunto, rng: np.random.Generator) -> Conjunto:
    """Une dos conjuntos y baraja, para que el orden no informe del origen."""
    import pandas as pd

    X = np.concatenate([a.X, b.X])
    y_reg = np.concatenate([a.y_reg, b.y_reg])
    y_vol = np.concatenate([a.y_vol, b.y_vol])
    fechas = pd.DatetimeIndex(list(a.fechas) + list(b.fechas))

    orden = rng.permutation(len(X))
    return Conjunto(X=X[orden], y_reg=y_reg[orden], y_vol=y_vol[orden], fechas=fechas[orden])


def resumen_rejilla(recetas: list[Receta]) -> "object":
    """Tabla con el contenido del barrido. Se imprime antes de lanzarlo.

    Sirve para dimensionar el trabajo: número de entrenamientos por tarea y
    coste total estimado.
    """
    import pandas as pd

    return pd.DataFrame(
        [
            {
                "identificador": r.identificador,
                "generador": r.generador,
                "n_reales": "todos" if r.n_reales is None else r.n_reales,
                "ratio": r.ratio,
                "politica": r.politica,
            }
            for r in recetas
        ]
    )
