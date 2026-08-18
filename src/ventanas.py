"""Construcción de ventanas deslizantes, particiones temporales y escalado.

Este módulo concentra las decisiones que protegen el experimento de la fuga de
información. Dos son críticas:

**Particiones temporales, no aleatorias.** Las ventanas de 60 días se solapan
en 59 observaciones con su vecina. Un `train_test_split` aleatorio coloca
ventanas casi idénticas a ambos lados del corte, y el modelo obtiene métricas
excelentes por memorizar, no por generalizar. Los notebooks guiados del máster
usan split aleatorio; aquí no se hace, y esa es una decisión deliberada
documentada en `docs/DECISIONES.md`.

**Embargo entre particiones.** No basta con cortar por fecha: la última ventana
de train termina consumiendo datos posteriores al corte (su Y mira `horizonte`
días hacia adelante). Se descartan `embargo` días a cada lado del corte para que
ninguna observación aparezca en dos particiones.

El bloque que consumen los generadores se define aquí: `X` aplanada más
`y_vol`, condicionado por `y_reg`. Mantener ese formato en un único sitio evita
que cada generador invente su propio empaquetado.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .config import Ventanas, n_canales


@dataclass
class Conjunto:
    """Una partición del dataset con todo lo necesario para entrenar.

    Attributes
    ----------
    X
        Ventanas de entrada, forma ``(n, pasado, canales)``.
    y_reg
        Régimen del periodo futuro, entero en ``[0, n_regimenes)``.
    y_vol
        Volatilidad realizada del periodo futuro, escalar.
    fechas
        Fecha del último día de cada ventana X. Sirve para trazar cualquier
        muestra hasta el calendario real.
    """

    X: np.ndarray
    y_reg: np.ndarray
    y_vol: np.ndarray
    fechas: pd.DatetimeIndex

    def __len__(self) -> int:
        return len(self.X)

    def __repr__(self) -> str:
        # Sin caracteres fuera de ASCII: la consola de Windows usa cp1252 por
        # defecto y una flecha tipográfica aquí revienta cualquier print().
        if len(self) == 0:
            return "Conjunto(vacio)"
        return (
            f"Conjunto(n={len(self)}, X={self.X.shape}, "
            f"{self.fechas[0].date()} a {self.fechas[-1].date()})"
        )

    def submuestra(self, n: int, semilla: int = 42) -> "Conjunto":
        """Extrae `n` muestras al azar, conservando la proporción de regímenes.

        Se usa en el barrido de escasez de datos reales: el eje del análisis que
        muestra cómo el valor de los sintéticos depende de cuántos datos reales
        haya. El muestreo es estratificado para que reducir el tamaño no haga
        desaparecer la clase de crisis.
        """
        if n >= len(self):
            return self

        rng = np.random.default_rng(semilla)
        indices = []
        for clase in np.unique(self.y_reg):
            candidatos = np.flatnonzero(self.y_reg == clase)
            cuota = max(1, round(n * len(candidatos) / len(self.y_reg)))
            cuota = min(cuota, len(candidatos))
            indices.append(rng.choice(candidatos, size=cuota, replace=False))

        idx = np.sort(np.concatenate(indices))[:n]
        return Conjunto(self.X[idx], self.y_reg[idx], self.y_vol[idx], self.fechas[idx])


@dataclass
class Particion:
    """Las tres particiones del dataset más el escalador ajustado en train."""

    train: Conjunto
    val: Conjunto
    test: Conjunto
    media: np.ndarray = field(repr=False)
    desviacion: np.ndarray = field(repr=False)


# ─────────────────────────────────────────────────────────────────────────────
# Construcción de ventanas
# ─────────────────────────────────────────────────────────────────────────────


def construir_ventanas(
    canales: pd.DataFrame,
    regimen_futuro: pd.Series,
    vol_futura: pd.Series,
    ventanas: Ventanas,
) -> Conjunto:
    """Apila ventanas deslizantes de X con sus objetivos alineados.

    La ventana que termina en la posición `i` cubre las filas ``[i-pasado+1, i]``
    de `canales`, y sus objetivos describen el periodo ``(i, i+horizonte]``. El
    alineado lo garantizan `features.objetivos()` y `regimenes.regimen_dominante()`,
    que ya devuelven las series desplazadas; aquí solo se comprueba que no haya
    NaN en las posiciones seleccionadas.
    """
    indice = canales.index
    objetivo_reg = regimen_futuro.reindex(indice)
    objetivo_vol = vol_futura.reindex(indice)

    valores = canales.to_numpy(dtype=np.float32)
    pasado = ventanas.pasado

    posiciones = [
        i
        for i in range(pasado - 1, len(indice), ventanas.paso)
        if np.isfinite(objetivo_reg.iloc[i]) and np.isfinite(objetivo_vol.iloc[i])
    ]

    if not posiciones:
        raise ValueError(
            "No se ha podido construir ninguna ventana. Revisa que los "
            "objetivos no estén enteramente a NaN."
        )

    X = np.stack([valores[i - pasado + 1 : i + 1] for i in posiciones]).astype(np.float32)
    y_reg = objetivo_reg.iloc[posiciones].to_numpy().astype(np.int64)
    y_vol = objetivo_vol.iloc[posiciones].to_numpy().astype(np.float32)

    return Conjunto(X=X, y_reg=y_reg, y_vol=y_vol, fechas=indice[posiciones])


def tamano_muestral_efectivo(
    indice: pd.DatetimeIndex, ventanas: Ventanas
) -> pd.Series:
    """Cuántas observaciones **independientes** hay detrás de las ventanas.

    Cómo se lee: ``ventanas_nominales`` es el número de filas del dataset y
    ``bloques_disjuntos`` el número de tramos que no comparten ni una sola
    sesión. El cociente es el factor por el que el dataset se está contando a sí
    mismo: cualquier intervalo de confianza calculado sobre las ventanas
    nominales es aproximadamente ese factor de veces más estrecho de lo que
    debería.

    Qué la invalida: hay que pasarle ``canales.index``, **no**
    ``precios.index``. Las 272 sesiones de arranque que consume el z-score
    causal no producen ninguna ventana, y contarlas infla el resultado.

    Parameters
    ----------
    indice
        Índice de fechas del panel de canales.
    ventanas
        Geometría de las ventanas, de `config.ventanas()`.

    Returns
    -------
    pandas.Series
        Con ``sesiones``, ``ventanas_nominales``, ``bloques_disjuntos`` y
        ``factor_inflacion``. El recuento nominal es geométrico: cuenta las
        posiciones posibles, sin descontar aquellas cuyo objetivo es NaN.
    """
    bloque = ventanas.pasado + ventanas.horizonte
    nominales = max(len(indice) - bloque + 1, 0)
    disjuntos = len(indice) // bloque
    return pd.Series(
        {
            "sesiones": len(indice),
            "ventanas_nominales": nominales,
            "bloques_disjuntos": disjuntos,
            "factor_inflacion": (
                round(nominales / disjuntos, 1) if disjuntos else float("nan")
            ),
        }
    )


# ─────────────────────────────────────────────────────────────────────────────
# Particiones temporales con embargo
# ─────────────────────────────────────────────────────────────────────────────


def partir(
    conjunto: Conjunto,
    train_hasta: str,
    val_hasta: str,
    embargo_sesiones: int,
    ventanas: Ventanas,
) -> tuple[Conjunto, Conjunto, Conjunto]:
    """Divide en train / validación / test por fecha, con embargo en sesiones.

    El embargo descarta las `embargo_sesiones` ventanas que siguen a cada corte.
    Se cuenta en **posiciones del índice de mercado**, no en días naturales: la
    huella de una ventana son `pasado + horizonte` sesiones, y los festivos no
    forman parte de ella. Contarlo en días naturales —lo que hacía la versión
    anterior— da un hueco que depende de dónde caiga el corte en el calendario:
    85 días naturales son 59 sesiones, 22 menos de las necesarias.

    Parameters
    ----------
    conjunto
        Dataset completo de `construir_ventanas`, ordenado por fecha.
    train_hasta, val_hasta
        Últimas fechas de train y de validación. El resto es test.
    embargo_sesiones
        Ventanas a descartar tras cada corte. Debe ser al menos
        ``ventanas.solape_maximo``; el catálogo lo fija en 85.
    ventanas
        Geometría de las ventanas. Se pide explícitamente para poder validar el
        embargo contra `solape_maximo`: es la comprobación que habría impedido
        el fallo original.

    Returns
    -------
    tuple of Conjunto
        Train, validación y test, en orden cronológico.

    Raises
    ------
    ValueError
        Si el embargo es menor que ``ventanas.solape_maximo``, o si un corte
        deja una partición vacía.
    """
    if embargo_sesiones < ventanas.solape_maximo:
        raise ValueError(
            f"El embargo es de {embargo_sesiones} sesiones y el minimo que "
            f"neutraliza el solape es {ventanas.solape_maximo}: con menos, "
            "alguna sesion queda en dos particiones a la vez."
        )

    fechas = conjunto.fechas
    posicion = np.arange(len(fechas))
    mascaras, desde = [], 0
    for corte in (train_hasta, val_hasta):
        elegibles = posicion[(posicion >= desde) & (fechas <= pd.Timestamp(corte))]
        if not len(elegibles):
            raise ValueError(f"El corte {corte} deja una particion vacia.")
        hasta = int(elegibles.max())
        mascaras.append((posicion >= desde) & (posicion <= hasta))
        desde = hasta + embargo_sesiones + 1
    mascaras.append(posicion >= desde)

    return tuple(_filtrar(conjunto, m) for m in mascaras)


def partir_por_dias_naturales(
    conjunto: Conjunto, train_hasta: str, val_hasta: str, embargo_dias: int
) -> tuple[Conjunto, Conjunto, Conjunto]:
    """La versión defectuosa de `partir`: cuenta el embargo en días naturales.

    **No la llames desde código nuevo.** Existe por la misma razón por la que el
    catálogo conserva `features_descartadas`: el notebook 02 reconstruye con ella
    el split que fallaba y mide la fuga que producía. La comparación *es* el
    argumento; sin ella habría que creerse la cifra.

    Con la geometría de este proyecto deja 22 sesiones en train y en validación a
    la vez, y otras 22 entre validación y test.
    """
    fechas = conjunto.fechas
    margen = pd.Timedelta(days=embargo_dias)
    corte_train, corte_val = pd.Timestamp(train_hasta), pd.Timestamp(val_hasta)

    return tuple(
        _filtrar(conjunto, m)
        for m in (
            fechas <= corte_train,
            (fechas > corte_train + margen) & (fechas <= corte_val),
            fechas > corte_val + margen,
        )
    )


def auditar_solape(
    conjuntos: dict[str, Conjunto],
    ventanas: Ventanas,
    indice: pd.DatetimeIndex,
) -> pd.DataFrame:
    """Sesiones de mercado que comparten cada par de particiones.

    Cómo se lee: una fila por pareja. ``sesiones_compartidas`` es lo único que
    importa y tiene que ser **cero**: es el número de sesiones del panel que
    entran en la huella ``[t - pasado + 1, t + horizonte]`` de alguna ventana de
    las dos particiones a la vez. ``hueco_sesiones`` es la distancia en
    posiciones entre la última ventana de una y la primera de la otra, y
    ``minimo_exigido`` lo que tendría que valer.

    Se cuenta por **conjuntos de posiciones** y no restando fechas a propósito:
    restar fechas es justamente lo que produjo el defecto que esta función
    existe para detectar.

    Qué la invalida: `indice` tiene que ser el mismo `canales.index` con el que
    se construyeron las ventanas. Con otro índice las posiciones no significan
    nada y la tabla dará ceros tranquilizadores y falsos. Y no detecta fuga por
    parámetros: un escalador ajustado con test pasa esta tabla sin despeinarse.

    Parameters
    ----------
    conjuntos
        ``{nombre: Conjunto}`` en orden cronológico.
    ventanas
        Geometría de las ventanas.
    indice
        Índice de fechas del panel de canales.

    Returns
    -------
    pandas.DataFrame
        Indexado por ``"a | b"``, con ``ventanas_a``, ``ventanas_b``,
        ``hueco_sesiones``, ``minimo_exigido``, ``sesiones_compartidas``,
        ``desde``, ``hasta`` y ``ok``.
    """
    posicion = pd.Series(np.arange(len(indice)), index=indice)

    def _huella(conjunto: Conjunto) -> set[int]:
        centros = posicion[conjunto.fechas].to_numpy()
        return set(
            np.concatenate(
                [
                    np.arange(i - ventanas.pasado + 1, i + ventanas.horizonte + 1)
                    for i in centros
                ]
            )
        )

    huellas = {n: _huella(c) for n, c in conjuntos.items()}
    nombres = list(conjuntos)
    filas = []
    for k, a in enumerate(nombres):
        for b in nombres[k + 1 :]:
            compartidas = sorted(huellas[a] & huellas[b])
            filas.append(
                {
                    "frontera": f"{a} | {b}",
                    "ventanas_a": len(conjuntos[a].fechas),
                    "ventanas_b": len(conjuntos[b].fechas),
                    "hueco_sesiones": int(
                        posicion[conjuntos[b].fechas[0]]
                        - posicion[conjuntos[a].fechas[-1]]
                    ),
                    "minimo_exigido": ventanas.solape_maximo + 1,
                    "sesiones_compartidas": len(compartidas),
                    "desde": indice[compartidas[0]] if compartidas else pd.NaT,
                    "hasta": indice[compartidas[-1]] if compartidas else pd.NaT,
                    "ok": not compartidas,
                }
            )
    return pd.DataFrame(filas).set_index("frontera")


def _huecos_por_dias(indice: pd.DatetimeIndex, dias_naturales: int) -> np.ndarray:
    """Hueco en sesiones que produce un embargo en días, según dónde caiga el corte."""
    fin = indice + pd.Timedelta(days=dias_naturales)
    completo = fin <= indice[-1]
    return (
        np.searchsorted(indice.to_numpy(), fin.to_numpy(), side="right")
        - np.arange(len(indice))
    )[completo]


def comparar_embargos(
    conjunto: Conjunto,
    particiones,
    ventanas: Ventanas,
    indice: pd.DatetimeIndex,
    dias_naturales: tuple[int, ...] = (85, 119, 125),
) -> pd.DataFrame:
    """Coste y garantía de cada forma de fijar el embargo, medidos.

    Cómo se lee: ``solape`` tiene que ser cero y ``descartadas`` es lo que cuesta
    conseguirlo. La columna que decide es ``cortes_cortos_pct``: el porcentaje de
    las fechas de corte posibles del panel para las que ese número de días
    naturales **no llega** al hueco mínimo. Un embargo en días con
    ``cortes_cortos_pct`` distinto de cero funciona por dónde caen los festivos
    alrededor de este corte concreto, no por diseño; la fila en sesiones vale
    cero por construcción, no por suerte.

    Qué la invalida: ``cortes_cortos_pct`` se mide sobre el calendario de **este**
    panel. Otro periodo daría otros porcentajes, y esa dependencia es justamente
    el argumento contra medir el embargo en días.

    Returns
    -------
    pandas.DataFrame
        Una fila por alternativa, con ``train``, ``val``, ``test``,
        ``descartadas``, ``solape`` y ``cortes_cortos_pct``.
    """
    minimo = ventanas.solape_maximo + 1
    filas = []

    for dias in dias_naturales:
        partes = partir_por_dias_naturales(
            conjunto, particiones.train_hasta, particiones.val_hasta, dias
        )
        nombres = dict(zip(("train", "val", "test"), partes))
        solape = int(
            auditar_solape(nombres, ventanas, indice)["sesiones_compartidas"].sum()
        )
        cortos = _huecos_por_dias(indice, dias) < minimo
        filas.append(
            {
                "embargo": f"{dias} días naturales",
                "train": len(partes[0].fechas),
                "val": len(partes[1].fechas),
                "test": len(partes[2].fechas),
                "descartadas": len(conjunto.fechas) - sum(len(p.fechas) for p in partes),
                "solape": solape,
                "cortes_cortos_pct": round(100 * float(cortos.mean()), 1),
            }
        )

    partes = partir(
        conjunto,
        particiones.train_hasta,
        particiones.val_hasta,
        particiones.embargo_sesiones,
        ventanas,
    )
    nombres = dict(zip(("train", "val", "test"), partes))
    filas.append(
        {
            "embargo": f"{particiones.embargo_sesiones} sesiones",
            "train": len(partes[0].fechas),
            "val": len(partes[1].fechas),
            "test": len(partes[2].fechas),
            "descartadas": len(conjunto.fechas) - sum(len(p.fechas) for p in partes),
            "solape": int(
                auditar_solape(nombres, ventanas, indice)["sesiones_compartidas"].sum()
            ),
            "cortes_cortos_pct": 0.0,
        }
    )
    return pd.DataFrame(filas).set_index("embargo")


def perfil_por_canal(
    conjuntos: dict[str, Conjunto], nombres: list[str], estadistico: str = "media"
) -> pd.DataFrame:
    """Media o desviación de cada canal en cada partición, sobre todas las ventanas.

    Es la comprobación de que el escalado se ajustó **solo con train**: la columna
    de train tiene que dar media 0 y desviación 1 en los veinte canales, y las
    otras dos no. Si validación o test dieran también 0 y 1, el escalador habría
    visto datos que no debía.

    De paso mide el desplazamiento de distribución entre particiones, que es lo
    que acota cuánto se puede extrapolar de los resultados del notebook 12.

    Parameters
    ----------
    conjuntos
        ``{nombre: Conjunto}`` ya escalados.
    nombres
        Nombres de canal, de `config.nombres_canales()`. Fijan el orden, que es
        el orden de columnas de `X`.
    estadistico
        ``"media"`` o ``"desviacion"``.

    Returns
    -------
    pandas.DataFrame
        Canales en el índice, una columna por partición.

    Raises
    ------
    ValueError
        Si `estadistico` no es uno de los dos admitidos.
    """
    if estadistico not in {"media", "desviacion"}:
        raise ValueError(f"Estadistico desconocido: {estadistico!r}")
    funcion = np.mean if estadistico == "media" else np.std
    return pd.DataFrame(
        {
            n: funcion(c.X.reshape(-1, c.X.shape[-1]), axis=0)
            for n, c in conjuntos.items()
        },
        index=list(nombres),
    )


def _filtrar(conjunto: Conjunto, mascara: np.ndarray) -> Conjunto:
    return Conjunto(
        X=conjunto.X[mascara],
        y_reg=conjunto.y_reg[mascara],
        y_vol=conjunto.y_vol[mascara],
        fechas=conjunto.fechas[mascara],
    )


# ─────────────────────────────────────────────────────────────────────────────
# Escalado
# ─────────────────────────────────────────────────────────────────────────────


def ajustar_escalado(train: Conjunto) -> tuple[np.ndarray, np.ndarray]:
    """Calcula media y desviación por canal usando SOLO el tramo de train.

    Devuelve vectores de forma ``(canales,)``. El escalado es por canal y no por
    posición temporal: cada canal tiene su propia escala física (un retorno
    diario y un z-score del VIX no son comparables), pero dentro de un canal
    todas las posiciones de la ventana comparten estadísticos.
    """
    media = train.X.reshape(-1, train.X.shape[-1]).mean(axis=0)
    desviacion = train.X.reshape(-1, train.X.shape[-1]).std(axis=0)
    # Un canal constante daría división por cero; se deja su escala a 1.
    desviacion[desviacion == 0] = 1.0
    return media.astype(np.float32), desviacion.astype(np.float32)


def escalar(conjunto: Conjunto, media: np.ndarray, desviacion: np.ndarray) -> Conjunto:
    """Aplica el escalado ajustado en train. No modifica el conjunto original."""
    return Conjunto(
        X=((conjunto.X - media) / desviacion).astype(np.float32),
        y_reg=conjunto.y_reg,
        y_vol=conjunto.y_vol,
        fechas=conjunto.fechas,
    )


def desescalar(X: np.ndarray, media: np.ndarray, desviacion: np.ndarray) -> np.ndarray:
    """Invierte el escalado. Necesario para interpretar muestras sintéticas
    en unidades de mercado (retornos, no desviaciones típicas)."""
    return X * desviacion + media


# ─────────────────────────────────────────────────────────────────────────────
# Empaquetado del bloque que consumen los generadores
#
# Los generadores no modelan X e Y por separado: modelan el bloque conjunto, de
# modo que cada muestra sintética viene con su objetivo coherente. Es la opción
# "OPT2" del planteamiento del taller (generar pares entrada-salida).
# El régimen NO va dentro del bloque: entra como condición del generador.
# ─────────────────────────────────────────────────────────────────────────────


def empaquetar(conjunto: Conjunto) -> np.ndarray:
    """Aplana ``(n, pasado, canales)`` + ``y_vol`` en una matriz ``(n, d)``.

    El último elemento de cada fila es la volatilidad futura. Devolver una única
    matriz permite que cualquier generador tabular trabaje sin saber nada de la
    estructura temporal.
    """
    n = len(conjunto)
    plano = conjunto.X.reshape(n, -1)
    return np.concatenate([plano, conjunto.y_vol.reshape(n, 1)], axis=1).astype(np.float32)


def desempaquetar(
    bloque: np.ndarray, y_reg: np.ndarray, ventanas: Ventanas, fechas=None
) -> Conjunto:
    """Operación inversa de `empaquetar`, para reconstruir muestras sintéticas.

    `y_reg` se pasa aparte porque es la condición con la que se generó el
    bloque, no algo que el generador tenga que reproducir.
    """
    n = len(bloque)
    canales = n_canales()
    X = bloque[:, :-1].reshape(n, ventanas.pasado, canales).astype(np.float32)
    y_vol = bloque[:, -1].astype(np.float32)

    if fechas is None:
        # Las muestras sintéticas no tienen fecha real; se marcan como NaT para
        # que nunca se confundan con observaciones de mercado.
        fechas = pd.DatetimeIndex([pd.NaT] * n)

    return Conjunto(X=X, y_reg=np.asarray(y_reg).astype(np.int64), y_vol=y_vol, fechas=fechas)


def dimension_bloque(ventanas: Ventanas) -> int:
    """Dimensión del vector que ve un generador: ``pasado * canales + 1``."""
    return ventanas.pasado * n_canales() + 1


# ─────────────────────────────────────────────────────────────────────────────
# Persistencia
#
# Es la pieza que hace utilizable el repositorio a tres manos: el notebook 02
# escribe una sola vez y los once siguientes leen. Nadie necesita re-ejecutar la
# descarga, el etiquetado ni la construcción de ventanas para empezar a trabajar
# en su generador.
# ─────────────────────────────────────────────────────────────────────────────

NOMBRE_PROCESADO = "ventanas.npz"


def guardar_procesado(particion: Particion, destino=None) -> "Path":
    """Escribe las tres particiones y el escalador en `data/processed/`.

    Se usa un único `.npz` comprimido en lugar de un fichero por conjunto para
    que sea imposible mezclar particiones de dos ejecuciones distintas.
    """
    from pathlib import Path

    from . import DIR_PROCESADO

    destino = Path(destino) if destino else DIR_PROCESADO / NOMBRE_PROCESADO
    destino.parent.mkdir(parents=True, exist_ok=True)

    campos = {}
    for nombre in ("train", "val", "test"):
        conjunto: Conjunto = getattr(particion, nombre)
        campos[f"{nombre}_X"] = conjunto.X
        campos[f"{nombre}_y_reg"] = conjunto.y_reg
        campos[f"{nombre}_y_vol"] = conjunto.y_vol
        # Las fechas se guardan como enteros nanosegundo: npz no admite
        # DatetimeIndex y perder la trazabilidad temporal no es una opción.
        campos[f"{nombre}_fechas"] = conjunto.fechas.asi8

    campos["media"] = particion.media
    campos["desviacion"] = particion.desviacion

    np.savez_compressed(destino, **campos)
    return destino


def cargar_procesado(origen=None) -> Particion:
    """Lee lo que dejó `guardar_procesado`. Es el punto de entrada de los
    notebooks 03 a 14.

    Raises
    ------
    FileNotFoundError
        Con indicación de qué ejecutar, porque es el error que se encontrará
        cualquiera que clone el repositorio y abra un notebook por el medio.
    """
    from pathlib import Path

    from . import DIR_PROCESADO

    origen = Path(origen) if origen else DIR_PROCESADO / NOMBRE_PROCESADO
    if not origen.exists():
        raise FileNotFoundError(
            f"No existe {origen}. Ejecuta notebooks/02_ventanas_y_splits.ipynb, "
            "o descarga data/processed/ del repositorio."
        )

    datos = np.load(origen)
    conjuntos = {
        nombre: Conjunto(
            X=datos[f"{nombre}_X"],
            y_reg=datos[f"{nombre}_y_reg"],
            y_vol=datos[f"{nombre}_y_vol"],
            fechas=pd.DatetimeIndex(datos[f"{nombre}_fechas"]),
        )
        for nombre in ("train", "val", "test")
    }
    return Particion(
        **conjuntos, media=datos["media"], desviacion=datos["desviacion"]
    )
