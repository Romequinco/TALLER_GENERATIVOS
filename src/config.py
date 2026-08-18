"""Lectura del catálogo de datos y fijación de semillas.

`data/catalog.yaml` es la fuente de verdad del proyecto. Ningún módulo debe
llevar tickers, fechas ni longitudes de ventana escritos a mano: todo se lee
desde aquí.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import numpy as np
import pandas as pd
import yaml

from . import CATALOGO


@lru_cache(maxsize=1)
def cargar_catalogo() -> dict[str, Any]:
    """Devuelve el catálogo completo como diccionario.

    El resultado se cachea: el fichero se lee una sola vez por sesión. Si se
    edita `catalog.yaml` en caliente hay que reiniciar el kernel del notebook.
    """
    with open(CATALOGO, encoding="utf-8") as f:
        return yaml.safe_load(f)


# ─────────────────────────────────────────────────────────────────────────────
# Vistas tipadas sobre el catálogo
#
# Son azúcar sintáctico: evitan encadenar cat["ventanas"]["pasado"] por todo el
# código y dan autocompletado en el editor.
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Ventanas:
    """Geometría de las ventanas deslizantes."""

    pasado: int
    horizonte: int
    paso: int

    @property
    def solape_maximo(self) -> int:
        """**Sesiones de mercado** que puede compartir una ventana con otra.

        Es el tamaño mínimo del embargo entre particiones: una ventana que
        empieza en `t` usa datos hasta `t + pasado + horizonte - 1`. Son
        sesiones, no días naturales: los festivos no forman parte de la huella.
        """
        return self.pasado + self.horizonte - 1


@dataclass(frozen=True)
class Particiones:
    """Cortes temporales de train / validación / test.

    El embargo va en **sesiones de mercado**, y el nombre del campo lo dice
    porque la versión anterior se llamaba `embargo_dias` y se aplicaba con
    `pd.Timedelta(days=...)`: producía un hueco de 59 sesiones donde hacían
    falta 81, y dejaba 22 sesiones en dos particiones a la vez.
    """

    train_hasta: str
    val_hasta: str
    embargo_sesiones: int


def ventanas() -> Ventanas:
    v = cargar_catalogo()["ventanas"]
    return Ventanas(pasado=v["pasado"], horizonte=v["horizonte"], paso=v["paso"])


def particiones() -> Particiones:
    p = cargar_catalogo()["particiones"]
    return Particiones(
        train_hasta=str(p["train_hasta"]),
        val_hasta=str(p["val_hasta"]),
        embargo_sesiones=p["embargo_sesiones"],
    )


def huecos() -> dict:
    """Política de huecos del catálogo: calendario de anclaje y relleno máximo."""
    return cargar_catalogo()["huecos"]


def tickers(rol: str | None = None) -> list[str]:
    """Tickers del universo, opcionalmente filtrados por rol.

    Parameters
    ----------
    rol
        Uno de ``indice``, ``sector``, ``riesgo``, ``renta_fija``, ``divisa``.
        Si es ``None`` devuelve el universo completo.
    """
    universo = cargar_catalogo()["universo"]
    if rol is not None:
        universo = [a for a in universo if a["rol"] == rol]
    return [a["ticker"] for a in universo]


def mapa_nombres() -> dict[str, str]:
    """Diccionario ``ticker -> nombre legible`` (ej. ``^GSPC -> sp500``).

    Los nombres legibles son los que se usan como columnas en todo el proyecto;
    los tickers solo aparecen en la descarga.
    """
    return {a["ticker"]: a["nombre"] for a in cargar_catalogo()["universo"]}


def nombres_canales() -> list[str]:
    """Nombres de los canales de X, en el orden en que forman las columnas.

    Este orden es un contrato: cambiarlo invalida los datos ya procesados y los
    generadores ya entrenados.
    """
    return [c["nombre"] for c in cargar_catalogo()["canales"]]


def parametros_canales() -> dict[str, dict]:
    """Hiperparámetros declarados de cada canal, indexados por nombre.

    Devuelve la entrada completa del catálogo de cada canal, de modo que
    `features.construir_canales` pueda leer de aquí las longitudes de ventana en
    lugar de llevarlas escritas a mano. Es lo que hace cierta la afirmación de
    que ninguna longitud de ventana vive fuera de `data/catalog.yaml`.

    Returns
    -------
    dict
        ``{nombre_canal: {tipo, fuente, ventana?, min_periodos?, ...}}``. Las
        claves opcionales solo están presentes en los canales que las declaran.
    """
    return {c["nombre"]: dict(c) for c in cargar_catalogo()["canales"]}


def familias_canales() -> pd.Series:
    """Familia de cada canal: ``retorno`` si es un log-retorno, ``derivada`` si no.

    Es una partición binaria del panel, y binaria a propósito: las figuras
    colorean la familia, nunca el valor del estadístico, que es lo que evita el
    anti-patrón de una rampa de color sobre veinte categorías. Once canales son
    retornos y nueve son derivadas.

    Returns
    -------
    pandas.Series
        Indexada por nombre de canal en el orden del catálogo, con valores
        ``"retorno"`` o ``"derivada"``.
    """
    tipos = {n: c["tipo"] for n, c in parametros_canales().items()}
    return pd.Series(
        {
            nombre: ("retorno" if tipos[nombre] == "log_return" else "derivada")
            for nombre in nombres_canales()
        }
    )


def n_canales() -> int:
    return len(nombres_canales())


def n_regimenes() -> int:
    return cargar_catalogo()["regimenes"]["n_estados"]


def semilla() -> int:
    return cargar_catalogo()["semilla_global"]


def fijar_semillas(valor: int | None = None) -> int:
    """Fija las semillas de random, numpy y torch. Devuelve la semilla usada.

    Se llama al principio de cada notebook. No garantiza determinismo bit a bit
    en operaciones de PyTorch multihilo, pero sí que dos ejecuciones seguidas
    den resultados equivalentes a efectos del análisis.
    """
    valor = semilla() if valor is None else valor

    random.seed(valor)
    np.random.seed(valor)

    try:
        import torch

        torch.manual_seed(valor)
    except ImportError:  # el módulo se puede usar sin PyTorch instalado
        pass

    return valor
