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

from .config import cargar_catalogo, nombres_canales, parametros_canales

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

    # Los hiperparámetros se leen del catálogo, no se escriben aquí: es lo que
    # hace cierta la afirmación de que basta editar `catalog.yaml` para
    # regenerar el dataset completo.
    par = parametros_canales()

    columnas_sector = [
        a["nombre"] for a in catalogo["universo"] if a["rol"] == "sector"
    ]

    canales: dict[str, pd.Series] = {}

    # -- Retornos del panel: índice y nueve sectores ------------------------
    canales["ret_sp500"] = ret["sp500"]
    for nombre in columnas_sector:
        canales[f"ret_{nombre}"] = ret[nombre]

    # -- Volatilidad y estrés ----------------------------------------------
    canales["vix_nivel_z"] = zscore_causal(
        precios["vix"], par["vix_nivel_z"]["min_periodos"]
    )
    canales["vix_cambio"] = precios["vix"].diff()
    canales["vol_realizada_z"] = zscore_causal(
        volatilidad_realizada(ret["sp500"], ventana=par["vol_realizada_z"]["ventana"]),
        par["vol_realizada_z"]["min_periodos"],
    )
    canales["drawdown_sp500"] = drawdown(precios["sp500"])

    # -- Momento ------------------------------------------------------------
    canales["momento_sp500"] = momentum(
        precios["sp500"],
        par["momento_sp500"]["ventana_larga"],
        par["momento_sp500"]["ventana_corta"],
    )

    # -- Crédito y curva ----------------------------------------------------
    # Proxy de spread de crédito: comportamiento relativo del crédito grado de
    # inversión frente al tesoro de duración similar. Se ensancha cuando el
    # crédito lo hace peor que el bono soberano, que es la firma del estrés.
    spread_credito = np.log(precios["credito_ig"] / precios["tesoro_10y"])
    canales["spread_credito_z"] = zscore_causal(
        spread_credito, par["spread_credito_z"]["min_periodos"]
    )

    # Proxy de pendiente de curva: tesoro largo frente a tesoro intermedio.
    pendiente = np.log(precios["tesoro_20y"] / precios["tesoro_10y"])
    canales["pendiente_curva_z"] = zscore_causal(
        pendiente, par["pendiente_curva_z"]["min_periodos"]
    )

    # -- Estructura transversal --------------------------------------------
    canales["corr_accion_bono"] = correlacion_movil(
        ret["sp500"], ret["tesoro_10y"], ventana=par["corr_accion_bono"]["ventana"]
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


# ─────────────────────────────────────────────────────────────────────────────
# Contraste de causalidad sobre los 20 canales
#
# La versión anterior vivía en el notebook y cubría 7 canales: los que se pueden
# expresar como función de una sola serie. Los otros 13 exigían envolver cada
# uno en una closure que recortaba el panel auxiliar a mano, y por eso se
# quedaron fuera. Aquí cada transformación recibe el sub-panel de sus propias
# fuentes, de modo que truncar filas trunca todas las fuentes en el mismo
# instante y el contraste vale igual para una serie que para nueve.
# ─────────────────────────────────────────────────────────────────────────────


def _transformaciones() -> dict[str, tuple[list[str], object]]:
    """Los 20 canales expresados como ``(fuentes, transformación del sub-panel)``.

    Es una segunda implementación del panel, independiente de
    `construir_canales`. Existe para poder contrastar la causalidad canal a
    canal, y `contraste_causalidad` comprueba que ambas coinciden: dos
    implementaciones que tienen que dar el mismo número son una prueba más
    fuerte que una sola, siempre que alguien verifique la coincidencia.
    """
    catalogo = cargar_catalogo()
    par = parametros_canales()
    sectores = [a["nombre"] for a in catalogo["universo"] if a["rol"] == "sector"]

    reg: dict[str, tuple[list[str], object]] = {
        "ret_sp500": (["sp500"], lambda s: log_returns(s["sp500"]))
    }
    for nombre in sectores:
        # `n=nombre` congela el valor en la definición: sin ese enlace todas las
        # closures compartirían la última iteración del bucle.
        reg[f"ret_{nombre}"] = ([nombre], lambda s, n=nombre: log_returns(s[n]))

    reg["vix_nivel_z"] = (
        ["vix"],
        lambda s: zscore_causal(s["vix"], par["vix_nivel_z"]["min_periodos"]),
    )
    reg["vix_cambio"] = (["vix"], lambda s: s["vix"].diff())
    reg["vol_realizada_z"] = (
        ["sp500"],
        lambda s: zscore_causal(
            volatilidad_realizada(
                log_returns(s["sp500"]), par["vol_realizada_z"]["ventana"]
            ),
            par["vol_realizada_z"]["min_periodos"],
        ),
    )
    reg["drawdown_sp500"] = (["sp500"], lambda s: drawdown(s["sp500"]))
    reg["momento_sp500"] = (
        ["sp500"],
        lambda s: momentum(
            s["sp500"],
            par["momento_sp500"]["ventana_larga"],
            par["momento_sp500"]["ventana_corta"],
        ),
    )
    reg["spread_credito_z"] = (
        ["credito_ig", "tesoro_10y"],
        lambda s: zscore_causal(
            np.log(s["credito_ig"] / s["tesoro_10y"]),
            par["spread_credito_z"]["min_periodos"],
        ),
    )
    reg["pendiente_curva_z"] = (
        ["tesoro_20y", "tesoro_10y"],
        lambda s: zscore_causal(
            np.log(s["tesoro_20y"] / s["tesoro_10y"]),
            par["pendiente_curva_z"]["min_periodos"],
        ),
    )
    reg["corr_accion_bono"] = (
        ["sp500", "tesoro_10y"],
        lambda s: correlacion_movil(
            log_returns(s["sp500"]),
            log_returns(s["tesoro_10y"]),
            par["corr_accion_bono"]["ventana"],
        ),
    )
    reg["dispersion_sectorial"] = (
        sectores,
        lambda s: dispersion_transversal(log_returns(s)),
    )
    reg["ret_dolar"] = (["dolar_indice"], lambda s: log_returns(s["dolar_indice"]))
    return reg


def discrepancia_causal(transformacion, entrada: pd.DataFrame) -> float:
    """Cuánto cambia el pasado de una transformación al conocerse el futuro.

    Aplica `transformacion` a la entrada completa y a su primera mitad, y
    devuelve la máxima diferencia absoluta en el tramo común. Es la misma lógica
    que `assert_causal`, pero devolviendo el número en lugar de lanzar: una tabla
    con veinte discrepancias es una prueba, y veinte asertos que no saltan solo
    son una ausencia de error.

    Una transformación causal devuelve exactamente ``0.0``. Un z-score sobre la
    muestra completa —el error clásico— devuelve un número del orden de la
    unidad, porque añadir observaciones al final reescribe retroactivamente la
    media y la desviación con las que se normalizó el principio.

    Parameters
    ----------
    transformacion
        Función que va del sub-panel de fuentes a una serie.
    entrada
        Sub-panel **sin recortar**: las primitivas, no el panel de canales, que
        ya viene con el arranque eliminado, que es justo donde se detecta el
        problema.

    Returns
    -------
    tuple of (float, int)
        Discrepancia máxima —``nan`` si no hay ningún valor comparable— y número
        de posiciones que el prefijo deja sin valor y la pasada completa sí
        rellena. Hacen falta las dos: la primera caza las fugas que **reescriben**
        el pasado y la segunda las que lo **desplazan**.
    """
    corte = len(entrada) // 2
    completa = np.asarray(transformacion(entrada).iloc[:corte], dtype=float)
    prefijo = np.asarray(transformacion(entrada.iloc[:corte]), dtype=float)

    # Una transformación que mira hacia delante —`shift(-1)`, una media
    # centrada— no altera ningún valor interior del prefijo: solo deja sin
    # rellenar las últimas posiciones, porque ahí le falta el futuro que la
    # pasada completa sí tenía. Comparando únicamente los valores finitos, esa
    # familia de fugas es invisible; contar las posiciones huérfanas la delata.
    huerfanas = int((np.isfinite(completa) & ~np.isfinite(prefijo)).sum())

    comparables = np.isfinite(completa) & np.isfinite(prefijo)
    if not comparables.any():
        return float("nan"), huerfanas
    return (
        float(np.abs(completa[comparables] - prefijo[comparables]).max()),
        huerfanas,
    )


def contraste_causalidad(
    precios: pd.DataFrame, tolerancia: float = 1e-10
) -> pd.DataFrame:
    """Contrasta la causalidad de los **veinte** canales, no de siete.

    Cómo se lee: la columna ``causal`` debe ser verdadera en las veinte filas, y
    para eso deben cumplirse dos cosas a la vez: ``discrepancia`` cero y
    ``huerfanas`` cero. Si una fila falla, ese canal está usando estadísticos del
    futuro y el resultado del taller entero queda invalidado, porque el modelo
    tendría acceso a información que en producción no existe.

    Hacen falta las dos columnas porque hay dos formas distintas de mirar al
    futuro. Un z-score sobre la muestra completa **reescribe** el pasado y lo
    delata ``discrepancia``. Un ``shift(-1)`` o una media centrada **desplazan**:
    no cambian ningún valor interior y ``discrepancia`` seguiría dando cero. Lo
    que los delata es que el prefijo se queda sin poder rellenar sus últimas
    posiciones, que es lo que cuenta ``huerfanas``.

    La función reconstruye los veinte canales por su cuenta y, además de
    contrastarlos, comprueba que su reconstrucción coincide exactamente con la
    que devuelve `construir_canales`. Esa comprobación es la que impide que las
    dos implementaciones se separen con el tiempo: si alguien cambia una y no la
    otra, esto falla de inmediato y de forma ruidosa.

    Parameters
    ----------
    precios
        Panel de cierres ajustados, sin recortar.
    tolerancia
        Umbral por debajo del cual una discrepancia se considera ruido de coma
        flotante. En la práctica todas las transformaciones dan cero exacto.

    Returns
    -------
    pandas.DataFrame
        Veinte filas en el orden del catálogo, con columnas ``fuentes``,
        ``discrepancia``, ``huerfanas`` y ``causal``.

    Raises
    ------
    AssertionError
        Si algún canal no es causal, o si la reconstrucción no coincide con
        `construir_canales`.
    """
    registro = _transformaciones()
    filas, reconstruido = [], {}
    for nombre, (fuentes, transformacion) in registro.items():
        sub = precios[fuentes]
        discrepancia, huerfanas = discrepancia_causal(transformacion, sub)
        filas.append((nombre, "+".join(fuentes), discrepancia, huerfanas))
        reconstruido[nombre] = transformacion(sub)

    tabla = pd.DataFrame(
        filas, columns=["canal", "fuentes", "discrepancia", "huerfanas"]
    ).set_index("canal")
    tabla["causal"] = (tabla["discrepancia"] <= tolerancia) & (tabla["huerfanas"] == 0)

    esperados = nombres_canales()
    if list(tabla.index) != esperados:
        raise AssertionError(
            f"El contraste no cubre exactamente los canales del catálogo: {esperados}"
        )
    if not tabla["causal"].all():
        raise AssertionError(
            f"Canales no causales: {sorted(tabla.index[~tabla['causal']])}"
        )

    panel = construir_canales(precios)
    diferencia = (
        pd.DataFrame(reconstruido)[esperados].reindex(panel.index) - panel
    ).abs()
    if float(diferencia.to_numpy().max()) > tolerancia:
        raise AssertionError(
            "La reconstrucción del contraste no coincide con construir_canales: "
            "las dos definiciones de los canales se han separado."
        )
    return tabla


# ─────────────────────────────────────────────────────────────────────────────
# Hechos estilizados de los datos reales
#
# Es la caracterización contra la que el notebook 14 juzgará a los generadores,
# y lo único del proyecto que predice el resultado de los notebooks 05 y 09
# antes de ejecutarlos. De los once hechos de Cont (2001), el panel permite
# contrastar ocho: quedan fuera la intermitencia y la asimetría entre escalas
# temporales, que exigen alta frecuencia, y la relación volumen-volatilidad,
# que exige volumen.
# ─────────────────────────────────────────────────────────────────────────────


def _autocorrelacion(x: np.ndarray, lag: int) -> float:
    """Autocorrelación muestral al retardo indicado, centrando la serie."""
    x = x - x.mean()
    return float(np.sum(x[lag:] * x[:-lag]) / np.sum(x * x))


def hechos_estilizados(
    retornos: pd.Series | pd.DataFrame,
    max_lag: int = 100,
    horizonte: int | None = None,
    bloque: int | None = None,
) -> pd.Series | pd.DataFrame:
    """Los hechos estilizados de Cont (2001) que este panel puede contrastar.

    Cómo se lee: la tabla no describe los datos, los **acusa**. Cada fila es una
    predicción que un generador tendrá que reproducir en los notebooks 05 a 14,
    y la comparación relevante no es entre canales sino entre la columna real y
    las columnas de procesos conocidos que devuelve `calibracion_hechos`.

    Qué la invalida: la curtosis es frágil —cae de 16,5 a 14,7 al eliminar una
    sola observación, la del 16 de marzo de 2020—, así que se reporta pero **no
    se usa como criterio de aceptación**. Y ``beta_acf_abs`` se devuelve por
    compatibilidad con la metodología del proyecto, pero su banda bootstrap
    medida es [0,35 – 1,20]: el ajuste log-log degenera cuando la ACF cruza
    cero, y por eso el notebook 00 la excluye de la tabla que reporta.

    Parameters
    ----------
    retornos
        Serie de retornos logarítmicos, o DataFrame con una columna por serie.
    max_lag
        Retardo máximo del ajuste log-log de ``beta_acf_abs``.
    horizonte
        Sesiones sobre las que se agregan los retornos y se estima la volatilidad
        que estandariza los residuos. ``None`` lo lee del catálogo, que es lo
        correcto: es el mismo horizonte que define las variables objetivo.
    bloque
        Longitud de la ventana cuando la serie es en realidad una concatenación
        de ventanas independientes, que es el caso de cualquier banco de muestras
        sintéticas. Con este argumento la volatilidad que estandariza los
        residuos **no cruza la frontera entre ventanas**. Sin él, en la primera
        sesión de cada ventana se dividiría por la volatilidad de una ventana
        ajena, lo que inventa colas condicionales que el generador no produce.
        ``None`` trata la entrada como una serie continua, que es lo correcto
        para los datos reales.

    Returns
    -------
    pandas.Series or pandas.DataFrame
        Indexado por nombre de estadístico. Si la entrada es un DataFrame,
        devuelve un DataFrame con una columna por serie de entrada.
    """
    if horizonte is None:
        from .config import ventanas as _ventanas

        horizonte = _ventanas().horizonte

    if isinstance(retornos, pd.DataFrame):
        return pd.DataFrame(
            {
                c: hechos_estilizados(retornos[c], max_lag, horizonte, bloque)
                for c in retornos.columns
            }
        )

    r = pd.Series(np.asarray(retornos, dtype=float)).dropna().reset_index(drop=True)
    centrado = (r - r.mean()).to_numpy()
    absolutos = np.abs(r.to_numpy())

    # Hecho 4: bajo agregación las colas se adelgazan hacia la normal.
    n_bloques = len(r) // horizonte * horizonte
    mensual = pd.Series(
        r.iloc[:n_bloques].to_numpy().reshape(-1, horizonte).sum(axis=1)
    )
    # Hecho 7: al dividir por la volatilidad reciente las colas NO desaparecen.
    # Es lo que separa "colas por agrupamiento" de "colas condicionales", y es la
    # única fila de la tabla que ningún proceso de referencia consigue reproducir.
    if bloque:
        # Una columna por ventana: `rolling` recorre cada ventana por separado y
        # nunca estandariza una sesión con la volatilidad de la ventana vecina.
        marco = pd.DataFrame(
            r.iloc[: len(r) // bloque * bloque].to_numpy().reshape(-1, bloque).T
        )
        residuos = pd.Series(
            (marco / marco.rolling(horizonte).std().shift(1)).to_numpy().ravel()
        )
    else:
        residuos = r / r.rolling(horizonte).std().shift(1)
    residuos = residuos.replace([np.inf, -np.inf], np.nan).dropna()

    valores = {
        "curtosis": float(r.kurt() + 3.0),
        f"curtosis_h{horizonte}": float(mensual.kurt() + 3.0),
        "curtosis_residuos": float(residuos.kurt() + 3.0),
        "asimetria": float(r.skew()),
        "ac1_retorno": _autocorrelacion(centrado, 1),
        "ac1_absoluto": _autocorrelacion(absolutos, 1),
        "ac10_absoluto": _autocorrelacion(absolutos, 10),
        "ac40_absoluto": _autocorrelacion(absolutos, 40),
        "ac100_absoluto": _autocorrelacion(absolutos, 100),
        "apalancamiento": float(
            pd.Series(centrado).corr(pd.Series(centrado**2).shift(-1))
        ),
    }

    lags = np.arange(1, max_lag + 1)
    curva = np.array([_autocorrelacion(absolutos, int(l)) for l in lags])
    positivos = curva > 0
    # `polyfit` sobre una máscara vacía lanza TypeError; devolver nan es la
    # respuesta correcta a "la ACF nunca es positiva" —el caso de un generador
    # sin memoria, que el notebook 14 tiene que poder medir— y no una excepción.
    valores["beta_acf_abs"] = (
        float(-np.polyfit(np.log(lags[positivos]), np.log(curva[positivos]), 1)[0])
        if positivos.sum() > 2
        else float("nan")
    )
    return pd.Series(valores)


def banda_bootstrap(
    retornos: pd.Series,
    n_replicas: int = 300,
    longitud: int = 250,
    semilla: int = 42,
) -> pd.DataFrame:
    """Banda de confianza al 95 % de los hechos estilizados, por bloques móviles.

    Cómo se lee: la banda es el error muestral de **nuestra** estimación, no un
    contraste de hipótesis. Un estadístico cuya banda no contiene el valor
    predicho por un proceso de referencia es un estadístico que discrimina; uno
    cuya banda lo contiene, no.

    El remuestreo es por **bloques móviles** y no observación a observación:
    remuestrear puntos sueltos destruiría el agrupamiento de volatilidad, que es
    justo lo que se está midiendo, y devolvería bandas artificialmente
    estrechas. La longitud de bloque de 250 sesiones —un año de mercado— es
    holgadamente mayor que la memoria que se pretende preservar.

    Parameters
    ----------
    retornos
        Serie de retornos logarítmicos.
    n_replicas, longitud
        Réplicas del bootstrap y longitud del bloque, en sesiones.
    semilla
        Semilla de un generador **local**: se usa `default_rng`, nunca el estado
        global de numpy, para que reejecutar solo esta celda dé el mismo
        resultado que ejecutarla dentro de un `Run All`.

    Returns
    -------
    pandas.DataFrame
        Indexado por estadístico, con columnas ``banda_inf`` y ``banda_sup``
        (percentiles 2,5 y 97,5).
    """
    rng = np.random.default_rng(semilla)
    valores = retornos.dropna().to_numpy()
    n = len(valores)
    n_bloques = int(np.ceil(n / longitud))

    replicas = []
    for _ in range(n_replicas):
        inicios = rng.integers(0, n - longitud, size=n_bloques)
        serie = np.concatenate([valores[i : i + longitud] for i in inicios])[:n]
        replicas.append(hechos_estilizados(pd.Series(serie)))

    cuantiles = pd.DataFrame(replicas).quantile([0.025, 0.975]).T
    return cuantiles.set_axis(["banda_inf", "banda_sup"], axis=1)


def calibracion_hechos(
    n: int = 20_000,
    semilla: int = 42,
    proporciones: tuple[float, ...] = (0.6, 0.3, 0.1),
    ratios_vol: tuple[float, ...] = (1.0, 1.8, 3.9),
) -> pd.DataFrame:
    """Los mismos hechos medidos sobre cuatro procesos de respuesta conocida.

    Cómo se lee: es la columna de contraste que convierte la tabla en un
    argumento. Cada proceso reproduce una propiedad del mercado y falla en otra,
    y **ninguno las reproduce todas**; ese hueco es lo que justifica un generador
    aprendido.

    - ``gaussiano``: una única normal multivariante. Curtosis exactamente 3 y,
      por el teorema de Isserlis, agrupamiento de volatilidad exactamente cero.
    - ``t4``: recupera las colas y sigue dando memoria cero. Las colas gruesas
      por sí solas no generan agrupamiento.
    - ``garch11``: recupera la memoria, pero sus residuos estandarizados vuelven
      a ser gaussianos. El agrupamiento por sí solo no genera colas condicionales.
    - ``mezcla``: **es el proceso que de verdad produce nuestro generador
      gaussiano**, no el de la primera columna. La decisión D10 ajusta un modelo
      independiente por régimen, de modo que el régimen es constante dentro de
      cada ventana y lo que sale es una mezcla de escala. Una mezcla sí tiene
      colas gruesas y, bajo el protocolo de autocorrelación agrupada, sí tiene
      agrupamiento. Comparar contra la primera columna sería comparar contra un
      objeto que este repositorio no implementa.

    Qué la invalida: la asimetría de una t(4) no es estimable —su tercer momento
    apenas existe— y sale distinta de cero por puro ruido muestral. Es la razón
    por la que la asimetría tampoco se usa como criterio de aceptación.

    Parameters
    ----------
    n
        Longitud de cada serie simulada.
    semilla
        Semilla del generador local `default_rng`.
    proporciones, ratios_vol
        Reparto de los regímenes y su desviación típica relativa a la del régimen
        de calma, para la columna ``mezcla``. Los valores por defecto son los
        medidos sobre este panel partiendo `vol_realizada_z` por los cuantiles
        0,6 y 0,9, que es el reparto que D6 espera del HMM.

    Returns
    -------
    pandas.DataFrame
        Indexado por estadístico, con columnas ``gaussiano``, ``t4``,
        ``garch11`` y ``mezcla``.
    """
    from .config import ventanas as _ventanas

    rng = np.random.default_rng(semilla)

    # Parámetros típicos de un índice de renta variable diario: la persistencia
    # alfa + beta = 0,99 es la que reproduce el decaimiento lento observado.
    omega, alfa, beta = 1e-6, 0.09, 0.90
    var = np.empty(n)
    innovacion = np.empty(n)
    var[0] = omega / (1 - alfa - beta)
    ruido = rng.standard_normal(n)
    for i in range(n):
        if i:
            var[i] = omega + alfa * innovacion[i - 1] ** 2 + beta * var[i - 1]
        innovacion[i] = np.sqrt(var[i]) * ruido[i]

    # La volatilidad se mantiene constante a lo largo de toda la ventana porque
    # el modelo por régimen genera la ventana entera de una vez, condicionada a
    # un único régimen. Sortear el régimen observación a observación daría
    # agrupamiento cero y describiría un generador que nadie ha escrito.
    pasado = _ventanas().pasado
    n_ventanas = int(np.ceil(n / pasado))
    regimen = rng.choice(len(proporciones), size=n_ventanas, p=proporciones)
    escala = np.repeat(np.asarray(ratios_vol)[regimen], pasado)[:n]

    return pd.DataFrame(
        {
            "gaussiano": hechos_estilizados(pd.Series(rng.standard_normal(n))),
            "t4": hechos_estilizados(pd.Series(rng.standard_t(4, n))),
            "garch11": hechos_estilizados(pd.Series(innovacion)),
            # `bloque=pasado` porque esta serie no es continua: es una tira de
            # ventanas independientes, igual que el banco de muestras que
            # devolverá el generador. Sin ese argumento, la volatilidad móvil
            # cruzaría de una ventana a la siguiente y le atribuiría a la mezcla
            # unas colas condicionales que no tiene.
            "mezcla": hechos_estilizados(
                pd.Series(rng.standard_normal(n) * escala), bloque=pasado
            ),
        }
    )


# ─────────────────────────────────────────────────────────────────────────────
# Perfil del panel: escala, señal y muestra efectiva
# ─────────────────────────────────────────────────────────────────────────────


def auc_univariante(
    canales: pd.DataFrame,
    objetivo: pd.Series,
    hasta: str | None = None,
    cuantil: float = 0.9,
) -> pd.DataFrame:
    """Poder discriminante de cada canal por separado, sin ajustar ningún modelo.

    Cómo se lee: es el área bajo la curva ROC de cada canal usado como único
    predictor de "el próximo mes estará en el decil superior de volatilidad".
    0,5 es el azar. Sustituye una cifra prestada —la metodología cita un AUC de
    0,80 para un canal `MOVE` que no existe en este catálogo— por una medida
    sobre este panel.

    Qué la invalida: se reporta ``max(AUC, 1-AUC)``, porque el signo de un canal
    es arbitrario y un canal que predice perfectamente al revés discrimina igual
    de bien. Esa simetrización es **levemente optimista**: un canal de puro
    ruido no da 0,50 exacto sino algo por encima. Por eso el umbral de lectura
    es 0,70 y no 0,52.

    Se calcula sobre el tramo indicado por `hasta`, que en el notebook es el de
    entrenamiento: es el único donde a un modelo le está permitido mirar.

    Parameters
    ----------
    canales
        Panel de canales.
    objetivo
        Serie continua a binarizar, típicamente ``objetivos["vol_futura"]``.
    hasta
        Fecha de corte inclusive. ``None`` usa todo el histórico.
    cuantil
        Cuantil del objetivo que define la clase positiva.

    Returns
    -------
    pandas.DataFrame
        Indexado por canal y **ordenado por AUC descendente**, con columnas
        ``auc``, ``n`` y ``positivos``. El orden del índice es el que consumen
        las dos figuras del perfil de canales, que solo se leen juntas si
        comparten ordenación.
    """
    y = objetivo.reindex(canales.index)
    if hasta is not None:
        dentro = canales.index <= pd.Timestamp(hasta)
        canales, y = canales[dentro], y[dentro]

    validos = y.notna().to_numpy()
    canales, y = canales[validos], y[validos]

    positivos = (y >= y.quantile(cuantil)).to_numpy()
    n_pos, n_neg = int(positivos.sum()), int((~positivos).sum())

    # AUC por el estadístico de rangos de Mann-Whitney: exacto, vectorizado
    # sobre los veinte canales a la vez y sin dependencias externas.
    rangos = canales.rank()
    bruto = (rangos[positivos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)

    return pd.DataFrame(
        {"auc": np.maximum(bruto, 1 - bruto), "n": len(canales), "positivos": n_pos}
    ).sort_values("auc", ascending=False)


def correlacion_media(
    canales: pd.DataFrame,
    objetivo: pd.Series,
    hasta: str | None = None,
    cuantil: float = 0.9,
) -> pd.Series:
    """Correlación media entre retornos sectoriales, en calma y en estrés.

    Cómo se lee: son los dos números que justifican condicionar los generadores
    por régimen. Si la correlación transversal fuese estable, un generador
    incondicional bastaría; como se dispara en estrés, un generador que no
    condiciona produce en crisis la estructura de dependencia de un mercado
    tranquilo.

    El estrés se define **igual que el proxy del AUC** —decil superior de
    `vol_futura` dentro del mismo tramo— y no de otra forma: una sola definición
    de estrés en todo el notebook, para que los dos números sean comparables
    entre sí. Por eso `hasta` existe y debe recibir el corte de entrenamiento:
    medir sobre el histórico completo daría un cuantil calculado con datos de
    test y dos cifras que ya no se pueden comparar con las del AUC.

    Parameters
    ----------
    canales
        Panel de canales; se usan solo las columnas ``ret_sector_*``.
    objetivo
        Serie continua que define el estrés, típicamente ``vol_futura``.
    hasta
        Fecha de corte inclusive. ``None`` usa todo el histórico.
    cuantil
        Cuantil a partir del cual una sesión se considera de estrés.

    Returns
    -------
    pandas.Series
        Con las entradas ``calma``, ``estres``, ``n_calma`` y ``n_estres``.
    """
    catalogo = cargar_catalogo()
    columnas = [
        f"ret_{a['nombre']}" for a in catalogo["universo"] if a["rol"] == "sector"
    ]

    y = objetivo.reindex(canales.index)
    if hasta is not None:
        dentro = canales.index <= pd.Timestamp(hasta)
        canales, y = canales[dentro], y[dentro]
    retornos = canales[columnas]

    validos = y.notna()
    estres = validos & (y >= y[validos].quantile(cuantil))
    calma = validos & ~estres

    def _media(sub: pd.DataFrame) -> float:
        matriz = sub.corr().to_numpy()
        return float(matriz[np.triu_indices_from(matriz, 1)].mean())

    return pd.Series(
        {
            "calma": _media(retornos[calma]),
            "estres": _media(retornos[estres]),
            "n_calma": int(calma.sum()),
            "n_estres": int(estres.sum()),
        }
    )


def episodios_estres(
    senal: pd.Series, umbral: float = 2.0, hueco: int | None = None
) -> pd.DataFrame:
    """Rachas independientes de estrés de mercado.

    Cómo se lee: cada fila es **un** evento de mercado, no una ventana. Es la
    unidad muestral que de verdad limita lo que se puede aprender sobre crisis:
    el panel tiene miles de ventanas pero solo una decena de crisis, y ninguna
    técnica de remuestreo crea información sobre eventos que nunca ocurrieron.

    Qué la invalida: el recuento depende de **dos** grados de libertad y no tiene
    sentido citarlo sin declarar los dos. Por la regla de fusión: con 63 sesiones
    salen 11 episodios, con 42 salen 13 y con 21 salen 17; se adopta 63 —tres
    veces el horizonte de predicción— por ser la más conservadora, la que hace
    parecer más pequeña nuestra propia muestra. Y por el umbral, que pesa más:
    con 1,5 salen 13 episodios, con 2,0 salen 11 y con 2,5 solo 5.

    Y una advertencia sobre cómo leer el resultado: seis de los once episodios
    duran una única sesión. El recuento que de verdad limita lo que un generador
    puede aprender no es este sino el del tramo de entrenamiento contando solo
    los episodios sustanciales, que son dos.

    Parameters
    ----------
    senal
        Serie del canal que define el estrés, típicamente ``vix_nivel_z``.
    umbral
        Valor por encima del cual una sesión se considera de estrés.
    hueco
        Sesiones de separación por debajo de las cuales dos rachas se consideran
        el mismo episodio. ``None`` lo fija en tres veces el horizonte del
        catálogo, para no dejar otra longitud de ventana escrita a mano.

    Returns
    -------
    pandas.DataFrame
        Una fila por episodio con ``inicio``, ``fin``, ``sesiones`` (duración
        total, respiros interiores incluidos) y ``sesiones_estres`` (las que
        superan el umbral).
    """
    if hueco is None:
        from .config import ventanas as _ventanas

        hueco = 3 * _ventanas().horizonte

    posiciones = np.flatnonzero((senal > umbral).to_numpy())
    if posiciones.size == 0:
        return pd.DataFrame(columns=["inicio", "fin", "sesiones", "sesiones_estres"])

    grupos = np.split(posiciones, np.flatnonzero(np.diff(posiciones) > hueco) + 1)
    return pd.DataFrame(
        [
            {
                "inicio": senal.index[g[0]],
                "fin": senal.index[g[-1]],
                "sesiones": int(g[-1] - g[0] + 1),
                "sesiones_estres": int(len(g)),
            }
            for g in grupos
        ]
    )
