"""Descarga y alineado del panel de mercado.

Los precios se bajan una sola vez de yfinance y quedan cacheados en
`data/raw/precios.parquet`. A partir de ahí todo el proyecto trabaja contra ese
fichero, de modo que los resultados no cambian porque yfinance revise su
histórico entre dos ejecuciones.

`data/raw/` está fuera del control de versiones: quien clone el repo ejecuta
`descargar_precios()` una vez y sigue desde ahí.
"""

from __future__ import annotations

import pandas as pd

from . import DIR_CRUDO
from .config import cargar_catalogo, mapa_nombres, tickers

RUTA_PRECIOS = DIR_CRUDO / "precios.parquet"


def descargar_precios(forzar: bool = False) -> pd.DataFrame:
    """Descarga los cierres ajustados del universo y los deja cacheados.

    Parameters
    ----------
    forzar
        Si es ``True`` vuelve a bajar los datos aunque exista la caché. Útil
        solo para extender la ventana temporal; en uso normal se deja en
        ``False`` para que todo el grupo trabaje sobre los mismos precios.

    Returns
    -------
    DataFrame indexado por fecha, con una columna por activo nombrada con el
    nombre legible del catálogo (``sp500``, ``vix``, ``sector_energia``...).
    """
    if RUTA_PRECIOS.exists() and not forzar:
        return pd.read_parquet(RUTA_PRECIOS)

    import yfinance as yf  # importación diferida: solo hace falta al descargar

    catalogo = cargar_catalogo()
    periodo = catalogo["periodo"]

    crudo = yf.download(
        tickers(),
        start=periodo["inicio"],
        end=periodo["fin"],
        auto_adjust=True,
        progress=True,
    )["Close"]

    # yfinance devuelve las columnas ordenadas alfabéticamente por ticker;
    # las renombramos al vocabulario del catálogo y fijamos el orden.
    nombres = mapa_nombres()
    crudo = crudo.rename(columns=nombres)
    crudo = crudo[[nombres[t] for t in tickers()]]

    precios = alinear(crudo)

    DIR_CRUDO.mkdir(parents=True, exist_ok=True)
    precios.to_parquet(RUTA_PRECIOS)
    return precios


def alinear(precios: pd.DataFrame) -> pd.DataFrame:
    """Recorta el panel al tramo en que TODAS las series tienen dato.

    Política heredada del TFM de regímenes: no se imputa nunca. Un hueco en una
    serie es una fecha que no existe para el panel, no un NaN que rellenar con
    el valor anterior; rellenar inventaría movimientos de precio que no
    ocurrieron y contaminaría la volatilidad realizada.

    En la práctica esto recorta el inicio hasta el activo más tardío del
    universo y elimina los días festivos parciales (una bolsa abierta y otra
    cerrada).
    """
    return precios.dropna(how="any").sort_index()


def cargar_precios() -> pd.DataFrame:
    """Lee el panel cacheado. Falla con un mensaje claro si no se ha descargado."""
    if not RUTA_PRECIOS.exists():
        raise FileNotFoundError(
            f"No existe {RUTA_PRECIOS}. Ejecuta primero "
            "notebooks/00_datos_y_features.ipynb, o llama a "
            "src.datos.descargar_precios()."
        )
    return pd.read_parquet(RUTA_PRECIOS)


def resumen(precios: pd.DataFrame) -> pd.DataFrame:
    """Tabla de control del panel: rango, observaciones y huecos por activo.

    Se imprime en el notebook 00 para dejar constancia de con qué datos se
    trabajó, ya que yfinance no es una fuente inmutable.
    """
    return pd.DataFrame(
        {
            "inicio": precios.apply(lambda s: s.first_valid_index()),
            "fin": precios.apply(lambda s: s.last_valid_index()),
            "observaciones": precios.notna().sum(),
            "huecos": precios.isna().sum(),
        }
    )
