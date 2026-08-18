"""Descarga y alineado del panel de mercado.

Los precios se bajan una sola vez de yfinance y quedan cacheados en
`data/raw/precios.parquet`. A partir de ahí todo el proyecto trabaja contra ese
fichero, de modo que los resultados no cambian porque yfinance revise su
histórico entre dos ejecuciones.

`data/raw/` está fuera del control de versiones: quien clone el repo ejecuta
`descargar_precios()` una vez y sigue desde ahí.
"""

from __future__ import annotations

import numpy as np
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
        precios = pd.read_parquet(RUTA_PRECIOS)
        validar_cache(precios)
        return precios

    import yfinance as yf  # importación diferida: solo hace falta al descargar

    catalogo = cargar_catalogo()
    periodo = catalogo["periodo"]

    crudo = yf.download(
        tickers(),
        start=periodo["inicio"],
        end=periodo["fin"],
        auto_adjust=True,
        # `progress=True` escribe una barra con retornos de carro que queda
        # incrustada en el .ipynb y lo vuelve ilegible en GitHub, que es parte
        # de lo que se evalúa.
        progress=False,
    )["Close"]

    # yfinance devuelve las columnas ordenadas alfabéticamente por ticker;
    # las renombramos al vocabulario del catálogo y fijamos el orden.
    nombres = mapa_nombres()
    crudo = crudo.rename(columns=nombres)
    crudo = crudo[[nombres[t] for t in tickers()]]

    # Un ticker que falla vuelve como columna entera de NaN, sin excepción. Sin
    # esta comprobación `alinear()` devolvería un panel vacío y el error
    # aparecería tres celdas más tarde, ininteligible.
    vacias = [c for c in crudo.columns if crudo[c].isna().all()]
    if vacias:
        raise ValueError(
            f"yfinance no ha devuelto ningún dato para: {vacias}. "
            "Revisa los tickers del catálogo y vuelve a intentarlo."
        )

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

    Al salir se comprueban los invariantes de integridad. Este es el embudo
    único por el que un panel se vuelve canónico, así que es el sitio donde la
    comprobación cuesta unos milisegundos y evita que un defecto se manifieste
    tres celdas más tarde en forma de un `dropna` que se lleva por delante un
    tercio del dataset.
    """
    from .calidad import invariantes

    panel = precios.dropna(how="any").sort_index()
    invariantes(panel)
    return panel


def validar_cache(precios: pd.DataFrame) -> None:
    """Comprueba que el panel cacheado corresponde al catálogo vigente.

    Sin esta comprobación, añadir un ticker a `data/catalog.yaml` no tiene
    ningún efecto visible: el notebook sigue leyendo la caché antigua y
    construye los canales sobre un universo que ya no es el declarado. El fallo
    es silencioso y sobrevive a un `Run All`.

    Parameters
    ----------
    precios
        Panel leído de `data/raw/precios.parquet`.

    Raises
    ------
    ValueError
        Si las columnas del panel no coinciden, en contenido u orden, con el
        universo del catálogo. El mensaje indica cómo regenerar la caché.
    """
    esperadas = [mapa_nombres()[t] for t in tickers()]
    if list(precios.columns) != esperadas:
        sobran = sorted(set(precios.columns) - set(esperadas))
        faltan = sorted(set(esperadas) - set(precios.columns))
        raise ValueError(
            "La caché de precios no corresponde al catálogo vigente "
            f"(faltan {faltan}, sobran {sobran}, o el orden ha cambiado). "
            "Ejecuta datos.descargar_precios(forzar=True) para regenerarla."
        )


def cargar_precios() -> pd.DataFrame:
    """Lee el panel cacheado. Falla con un mensaje claro si no se ha descargado."""
    if not RUTA_PRECIOS.exists():
        raise FileNotFoundError(
            f"No existe {RUTA_PRECIOS}. Ejecuta primero "
            "notebooks/00_datos_y_features.ipynb, o llama a "
            "src.datos.descargar_precios()."
        )
    precios = pd.read_parquet(RUTA_PRECIOS)
    # Es la puerta que usan los cuadernos posteriores al 00. Sin esta llamada,
    # el único control de correspondencia con el catálogo se podía saltar
    # entrando por aquí.
    validar_cache(precios)
    return precios


def huella(precios: pd.DataFrame) -> str:
    """Identificador reproducible del contenido de un panel.

    Se resume el **contenido**, no el fichero: los bytes de un `.parquet`
    cambian con la versión de pyarrow aunque los números sean idénticos, así que
    la huella del fichero no sirve para comparar entre dos máquinas. Se resumen
    los valores en float64 y orden C, los nombres de columna y el índice en
    nanosegundos, que es exactamente lo que determina el resultado del pipeline.

    Sirve para lo único para lo que sirve una huella: si un compañero obtiene
    métricas distintas, esta cadena dice en una línea si el desacuerdo viene de
    los datos o del código. yfinance revisa su histórico y reajusta dividendos
    hacia atrás, de modo que dos descargas separadas en el tiempo no coinciden.

    Parameters
    ----------
    precios
        Cualquier DataFrame numérico indexado por fecha.

    Returns
    -------
    str
        Los doce primeros caracteres hexadecimales del SHA-256.
    """
    import hashlib

    digest = hashlib.sha256()
    digest.update(np.ascontiguousarray(precios.to_numpy(dtype="float64")).tobytes())
    digest.update("|".join(map(str, precios.columns)).encode("utf-8"))
    digest.update(precios.index.asi8.tobytes())
    return digest.hexdigest()[:12]


def huecos(indice: pd.DatetimeIndex, umbral_dias: int = 4) -> pd.DataFrame:
    """Saltos del calendario mayores que `umbral_dias` días naturales.

    `alinear()` recorta con `dropna(how="any")`, que puede abrir un agujero en
    medio del índice sin avisar. Es un problema real y no cosmético: las ventanas
    se construyen por posición, de modo que un agujero interior empalmaría dos
    tramos no contiguos dentro de la misma ventana de 60 días.

    Cómo se lee: un hueco legítimo **no** es un error. El cierre de la NYSE por
    el huracán Sandy (29 y 30 de octubre de 2012) cae dentro del periodo y
    aparece aquí. El criterio de alarma no es que la tabla esté vacía, sino que
    ningún salto pase de la decena de días naturales.

    Parameters
    ----------
    indice
        Índice de fechas del panel alineado.
    umbral_dias
        Solo se listan los saltos **estrictamente mayores** que este valor. El
        valor por defecto, 4, deja fuera los puentes ordinarios (fin de semana
        más un festivo) y deja dentro los cierres de mercado.

    Returns
    -------
    pandas.DataFrame
        Columnas ``desde``, ``hasta`` y ``dias``, ordenada por ``dias``
        descendente. Puede estar vacía.
    """
    salto = pd.Series(indice).diff().dt.days
    marcados = salto[salto > umbral_dias].index.to_numpy()
    tabla = pd.DataFrame(
        {
            "desde": indice[marcados - 1],
            "hasta": indice[marcados],
            "dias": salto.loc[marcados].to_numpy().astype(int),
        }
    )
    return tabla.sort_values("dias", ascending=False).reset_index(drop=True)


def cobertura(precios: pd.DataFrame) -> pd.DataFrame:
    """Origen declarado de cada serie frente al arranque efectivo del panel.

    Responde a la pregunta que la tabla de control anterior no podía responder:
    qué activo impide empezar antes. Se lee del catálogo y no de una descarga
    adicional, porque el panel cacheado ya viene recortado y no conserva la
    información del arranque de cada serie por separado.

    Cómo se lee: las primeras filas son los activos más tardíos del universo, y
    son los que fijan el suelo de disponibilidad. Si ese suelo es **anterior** a
    `periodo.inicio`, el arranque del panel es una decisión del catálogo y no un
    efecto del alineado; si fuera posterior, el catálogo estaría pidiendo un
    periodo que los datos no soportan y habría que corregirlo.

    Parameters
    ----------
    precios
        Panel alineado, del que solo se usa la primera fecha.

    Returns
    -------
    pandas.DataFrame
        Una fila por activo, ordenada por ``desde_fuente`` descendente, con
        columnas ``ticker``, ``rol``, ``desde_fuente``, ``inicio_panel`` y
        ``limita_arranque``.
    """
    tabla = pd.DataFrame(cargar_catalogo()["universo"]).set_index("nombre")
    tabla = tabla.rename(columns={"desde": "desde_fuente"})
    tabla["desde_fuente"] = pd.to_datetime(tabla["desde_fuente"])
    tabla["inicio_panel"] = precios.index[0]
    tabla["limita_arranque"] = tabla["desde_fuente"] == tabla["desde_fuente"].max()
    return tabla.sort_values("desde_fuente", ascending=False)[
        ["ticker", "rol", "desde_fuente", "inicio_panel", "limita_arranque"]
    ]


def resumen(precios: pd.DataFrame) -> pd.DataFrame:
    """Tabla de control del panel: rango, observaciones y huecos por activo.

    .. warning::
       **No sirve como control de calidad y el notebook 00 ya no la llama.**
       `descargar_precios` aplica `alinear()` antes de cachear, de modo que
       cuando esta función recibe el panel ya no queda ni un NaN: devuelve
       siempre la misma fecha de inicio, la misma de fin y cero huecos para
       todos los activos. Un control que solo puede decir "todo correcto" no es
       un control. Para lo que esta tabla pretendía, usa `cobertura()` y
       `huecos()`.
    """
    return pd.DataFrame(
        {
            "inicio": precios.apply(lambda s: s.first_valid_index()),
            "fin": precios.apply(lambda s: s.last_valid_index()),
            "observaciones": precios.notna().sum(),
            "huecos": precios.isna().sum(),
        }
    )
