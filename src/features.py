"""Construcción de los canales de entrada.

Todas las transformaciones son **causales**: el valor en el instante `t` solo
depende de observaciones de `t` o anteriores. Esto descarta el z-score sobre la
muestra completa, que es el error más habitual en series financieras: da
resultados aparentemente mejores porque el modelo conoce la media y la
desviación de todo el histórico, incluido el futuro.

Las primitivas siguen el criterio del TFM de detección de regímenes, donde se
comprobó que el z-score no causal compraba suavidad en las señales pero no
acierto fuera de muestra.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import cargar_catalogo, nombres_canales

# Días de mercado al año, para anualizar volatilidades.
DIAS_ANIO = 252


# ─────────────────────────────────────────────────────────────────────────────
# Primitivas causales
# ─────────────────────────────────────────────────────────────────────────────


def log_returns(precios: pd.Series | pd.DataFrame) -> pd.Series | pd.DataFrame:
    """Retornos logarítmicos. Pierde la primera observación."""
    return np.log(precios).diff()


def zscore_causal(serie: pd.Series, min_periodos: int = 252) -> pd.Series:
    """Z-score con media y desviación acumuladas hasta `t` (expanding).

    Las primeras `min_periodos` observaciones quedan a NaN: con menos historia
    la estimación de la desviación es demasiado inestable y produce picos
    espurios que el modelo interpretaría como estrés de mercado.
    """
    media = serie.expanding(min_periods=min_periodos).mean()
    desv = serie.expanding(min_periods=min_periodos).std()
    return (serie - media) / desv


def volatilidad_realizada(
    retornos: pd.Series, ventana: int = 21, anualizar: bool = True
) -> pd.Series:
    """Desviación típica móvil de los retornos.

    Es una media móvil hacia atrás, luego es causal por construcción.
    """
    vol = retornos.rolling(ventana, min_periods=ventana).std()
    return vol * np.sqrt(DIAS_ANIO) if anualizar else vol


def drawdown(precios: pd.Series) -> pd.Series:
    """Caída relativa desde el máximo histórico alcanzado hasta `t`.

    El máximo es expanding, no rolling: no se puede conocer un máximo futuro.
    Devuelve valores en $[-1, 0]$.
    """
    maximo = precios.expanding(min_periods=1).max()
    return precios / maximo - 1.0


def momentum(
    precios: pd.Series, ventana_larga: int = 252, ventana_corta: int = 21
) -> pd.Series:
    """Momento como diferencia de retornos acumulados a dos plazos.

    Positivo cuando la tendencia reciente supera a la de largo plazo.
    """
    largo = np.log(precios / precios.shift(ventana_larga))
    corto = np.log(precios / precios.shift(ventana_corta))
    return corto - largo / (ventana_larga / ventana_corta)


def correlacion_movil(a: pd.Series, b: pd.Series, ventana: int = 60) -> pd.Series:
    """Correlación de Pearson móvil entre dos series de retornos.

    La correlación acción-bono es un indicador clásico de régimen: se vuelve
    positiva en episodios de estrés inflacionario (2022) y negativa en las
    huidas hacia la calidad (2008, 2020).
    """
    return a.rolling(ventana, min_periods=ventana).corr(b)


def dispersion_transversal(retornos: pd.DataFrame) -> pd.Series:
    """Desviación típica entre activos, día a día.

    Mide cuánto se separan los sectores entre sí. Se dispara cuando el mercado
    deja de moverse en bloque y empieza a discriminar.
    """
    return retornos.std(axis=1)


def assert_causal(transformacion, serie: pd.Series, nombre: str = "serie") -> None:
    """Comprueba que una transformación no mira al futuro.

    Aplica `transformacion` sobre la serie completa y sobre un prefijo, y exige
    que los valores del prefijo coincidan. Si una transformación usara
    estadísticos de la muestra entera —el error habitual, un z-score global—,
    añadir observaciones al final cambiaría retroactivamente los valores del
    principio y el contraste fallaría.

    Parameters
    ----------
    transformacion
        Función de una serie en una serie, por ejemplo ``zscore_causal`` o
        ``lambda s: volatilidad_realizada(log_returns(s))``.
    serie
        Serie o DataFrame de entrada **sin recortar**. Debe aplicarse a las
        primitivas, no al panel que devuelve `construir_canales`: ese ya viene
        con `dropna` aplicado y ha perdido el arranque, que es justo donde se
        detecta el problema. Se admite un DataFrame para poder contrastar las
        transformaciones multi-serie (`dispersion_transversal`,
        `correlacion_movil`), que reciben varias columnas a la vez.

    Raises
    ------
    AssertionError
        Si el prefijo cambia al conocerse el futuro.
    """
    if bool(np.asarray(serie.isna()).all()):
        raise ValueError(f"{nombre}: la entrada está completamente vacía.")

    corte = len(serie) // 2
    completa = transformacion(serie).iloc[:corte]
    prefijo = transformacion(serie.iloc[:corte])

    # `to_numpy` unifica el tratamiento de Series y DataFrame: sin él, las
    # reducciones booleanas sobre un DataFrame devuelven una Series y las
    # comparaciones se vuelven ambiguas.
    a, b = np.asarray(completa, dtype=float), np.asarray(prefijo, dtype=float)
    comparables = np.isfinite(a) & np.isfinite(b)
    if not comparables.any():
        raise ValueError(
            f"{nombre}: no hay valores comparables en la primera mitad; "
            "usa una serie más larga para el contraste."
        )

    discrepancia = float(np.abs(a[comparables] - b[comparables]).max())
    if discrepancia > 1e-10:
        raise AssertionError(
            f"{nombre}: los valores del pasado cambian al conocer el futuro "
            f"(discrepancia máxima {discrepancia:.3e}). La transformación no es "
            "causal."
        )


# ─────────────────────────────────────────────────────────────────────────────
# Ensamblado del panel de canales
# ─────────────────────────────────────────────────────────────────────────────


def construir_canales(precios: pd.DataFrame) -> pd.DataFrame:
    """Construye los 20 canales declarados en `data/catalog.yaml`.

    Parameters
    ----------
    precios
        Panel de cierres ajustados devuelto por `datos.cargar_precios()`.

    Returns
    -------
    DataFrame indexado por fecha con una columna por canal, en el orden fijado
    por el catálogo, y sin las filas iniciales en que algún canal aún no tiene
    historia suficiente.
    """
    catalogo = cargar_catalogo()
    ret = log_returns(precios)

    columnas_sector = [
        a["nombre"] for a in catalogo["universo"] if a["rol"] == "sector"
    ]

    canales: dict[str, pd.Series] = {}

    # -- Retornos del panel: índice y nueve sectores ------------------------
    canales["ret_sp500"] = ret["sp500"]
    for nombre in columnas_sector:
        canales[f"ret_{nombre}"] = ret[nombre]

    # -- Volatilidad y estrés ----------------------------------------------
    canales["vix_nivel_z"] = zscore_causal(precios["vix"])
    canales["vix_cambio"] = precios["vix"].diff()
    canales["vol_realizada_z"] = zscore_causal(
        volatilidad_realizada(ret["sp500"], ventana=21)
    )
    canales["drawdown_sp500"] = drawdown(precios["sp500"])

    # -- Momento ------------------------------------------------------------
    canales["momento_sp500"] = momentum(precios["sp500"])

    # -- Crédito y curva ----------------------------------------------------
    # Proxy de spread de crédito: comportamiento relativo del crédito grado de
    # inversión frente al tesoro de duración similar. Se ensancha cuando el
    # crédito lo hace peor que el bono soberano, que es la firma del estrés.
    spread_credito = np.log(precios["credito_ig"] / precios["tesoro_10y"])
    canales["spread_credito_z"] = zscore_causal(spread_credito)

    # Proxy de pendiente de curva: tesoro largo frente a tesoro intermedio.
    pendiente = np.log(precios["tesoro_20y"] / precios["tesoro_10y"])
    canales["pendiente_curva_z"] = zscore_causal(pendiente)

    # -- Estructura transversal --------------------------------------------
    canales["corr_accion_bono"] = correlacion_movil(
        ret["sp500"], ret["tesoro_10y"], ventana=60
    )
    canales["dispersion_sectorial"] = dispersion_transversal(ret[columnas_sector])

    # -- Macro --------------------------------------------------------------
    canales["ret_dolar"] = ret["dolar_indice"]

    panel = pd.DataFrame(canales)

    # El orden de columnas es un contrato con los datos ya procesados.
    esperados = nombres_canales()
    faltan = set(esperados) - set(panel.columns)
    if faltan:
        raise KeyError(f"Canales del catálogo no construidos: {sorted(faltan)}")
    panel = panel[esperados]

    # Se recorta el arranque hasta que todos los canales tienen valor. El corte
    # lo gobierna el z-score causal, que necesita 252 sesiones de historia.
    return panel.dropna(how="any")


def objetivos(precios: pd.DataFrame, horizonte: int) -> pd.DataFrame:
    """Variables objetivo calculadas sobre los `horizonte` días SIGUIENTES.

    Estas series miran deliberadamente al futuro: son la etiqueta, no una
    entrada del modelo. El alineado con las ventanas de X lo hace
    `ventanas.construir_ventanas()`, que garantiza que la X de una muestra
    termina justo antes de donde empieza su Y.

    Returns
    -------
    DataFrame con:
      ``vol_futura``  volatilidad realizada anualizada del periodo futuro
      ``ret_futuro``  retorno acumulado del periodo futuro
      ``dd_futuro``   caída máxima dentro del periodo futuro
    """
    ret_sp500 = log_returns(precios["sp500"])

    # `shift(-1)` alinea la ventana para que empiece el día siguiente a `t`.
    futuro = ret_sp500.shift(-1)

    vol_futura = (
        futuro.rolling(horizonte, min_periods=horizonte).std().shift(-(horizonte - 1))
        * np.sqrt(DIAS_ANIO)
    )
    ret_futuro = (
        futuro.rolling(horizonte, min_periods=horizonte).sum().shift(-(horizonte - 1))
    )

    # Caída máxima dentro del periodo futuro, sobre el precio acumulado.
    def _caida_maxima(ventana: np.ndarray) -> float:
        acumulado = np.cumsum(ventana)
        maximo = np.maximum.accumulate(np.concatenate([[0.0], acumulado]))[1:]
        return float(np.min(acumulado - maximo))

    dd_futuro = (
        futuro.rolling(horizonte, min_periods=horizonte)
        .apply(_caida_maxima, raw=True)
        .shift(-(horizonte - 1))
    )

    return pd.DataFrame(
        {"vol_futura": vol_futura, "ret_futuro": ret_futuro, "dd_futuro": dd_futuro}
    )
