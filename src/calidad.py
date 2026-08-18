"""Auditoría de integridad del panel de precios.

`datos.alinear()` fija la política de huecos —recortar, nunca imputar— pero no
comprueba nada más, y los defectos que quedan fuera de su alcance no producen
ningún error visible: se traducen en filas que desaparecen en el `dropna` final
de `features.construir_canales` o en canales que salen mal sin avisar. Tres
ejemplos medidos sobre este panel:

* un solo cierre puesto a cero evapora **61 ventanas** y deja `drawdown_sp500` en
  −1.0, una caída del 100 % que nunca ocurrió, en un canal con AUC 0,88;
* una fecha duplicada hace que `construir_canales` devuelva 5.669 filas en vez de
  5.668, con esa sesión contada dos veces dentro de todas las ventanas que la
  contienen;
* una columna constante deja el panel de canales en **(0, 20)**: el dataset
  entero desaparece.

Ninguno de los tres lanza una excepción hoy. De ahí este módulo.

Vive aparte de `datos.py` por dos razones. La primera es de capas: los controles
necesitan log-retornos, que son de `features`, y meterlos en `datos` invertiría
el orden de dependencias. La segunda es que la misma auditoría sirve para un
panel **sintético**, que falla exactamente por estos modos —un canal colapsado a
constante, saltos imposibles al deshacer la escala, valores no positivos al
exponenciar—, de modo que el cuaderno 14 la reutiliza tal cual.

Dos naturalezas distintas, no dos grados de gravedad
----------------------------------------------------
* **Invariantes** (`tipos`, `indice`, `finitud`, `positividad`, `degeneradas`):
  cosas que no pueden ocurrir en datos correctos. Lanzan.
* **Avisos** (`congelados`, `saltos`, `calendario`, `densidad`, `cobertura`):
  cosas inverosímiles pero posibles. Nunca lanzan, porque la observación más
  extrema de este panel es un VIX de **+115,6 %** el 5 de febrero de 2018, que es
  un dato correcto. Un control que aborta sobre datos buenos enseña al equipo a
  subir el umbral, y un umbral subido para callar al control es peor que no tener
  control.

Sobre la limpieza
-----------------
En este proyecto **no se limpia nada**: se audita y se documenta. No es purismo,
es que las técnicas habituales destruyen justo lo que el trabajo mide. Winsorizar
al 1 % baja la curtosis de `ret_sp500` de 16,5 a 6,4 —por debajo del 8,0 que
produce el propio generador gaussiano del proyecto—, de modo que el mercado
pasaría a tener colas más finas que el modelo contra el que se compara. Al 5 %,
la curtosis de los residuos cae a 5,3, por debajo de la banda [5,7 – 11,2], y la
hipótesis H1 del cuaderno 00 se volvería falsa sobre los datos reales: cualquier
generador quedaría automáticamente no refutado. El detalle está en
`docs/DECISIONES.md`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import cargar_catalogo

#: Filas cuya violación es imposible en datos correctos y silenciosa aguas
#: abajo. Son las que lanzan.
INVARIANTES = ("tipos", "indice", "finitud", "positividad", "degeneradas")


def invariantes(precios: pd.DataFrame) -> None:
    """Comprueba las cinco propiedades que un panel de precios no puede violar.

    Se llama desde `datos.alinear()`, que es el embudo único por el que un panel
    se vuelve canónico. Cuesta unos milisegundos y evita que un defecto se
    manifieste cinco celdas más tarde en forma de un `dropna` que se lleva por
    delante un tercio del dataset.

    Parameters
    ----------
    precios
        Panel de cierres indexado por fecha.

    Raises
    ------
    ValueError
        Nombrando la propiedad violada y dónde. El mensaje es el que va a leer
        quien encuentre el problema, así que dice qué mirar.
    """
    tabla = auditar_panel(precios, estricto=False)
    fallos = tabla.loc[list(INVARIANTES)]
    fallos = fallos[fallos["veredicto"] == "fallo"]
    if not fallos.empty:
        detalle = "; ".join(f"{i}: {f.donde}" for i, f in fallos.iterrows())
        raise ValueError(f"El panel viola invariantes de integridad. {detalle}")


def auditar_panel(
    precios: pd.DataFrame,
    *,
    umbral_z: float = 50.0,
    congelado_max: int = 3,
    hueco_dias: int = 10,
    estricto: bool = True,
) -> pd.DataFrame:
    """Tabla de control del panel: una fila por comprobación, un veredicto por fila.

    Sustituye a `datos.resumen()`, que no podía fallar porque recibía el panel ya
    alineado y sin NaN. Aquí las diez filas miden cosas que el alineado no
    arregla: el tipo del índice, el signo de los precios, las rachas de cierres
    congelados y los saltos imposibles.

    Cómo se lee: la columna ``veredicto`` toma tres valores, y ``aviso`` y
    ``fallo`` **no son dos grados de gravedad sino dos naturalezas distintas**.
    ``fallo`` marca un invariante roto, algo que no puede ocurrir en datos
    correctos. ``aviso`` marca lo inverosímil pero posible, y pide mirar la
    columna ``donde``, no subir el umbral: el mayor salto de este panel es un VIX
    de +115,6 % el 5 de febrero de 2018, que es un dato bueno.

    Qué la invalida: ninguna fila prueba que los **valores** sean correctos. Un
    panel con los precios de otro periodo, o con los dividendos reajustados por
    yfinance entre dos descargas, pasa las diez comprobaciones. De eso se ocupa
    `datos.huella()`, y las dos cosas son complementarias, no alternativas.

    Parameters
    ----------
    precios
        Panel de cierres ajustados indexado por fecha. Sirve igual para un panel
        sintético: la función no lee del catálogo más que el periodo declarado.
    umbral_z
        Máximo z robusto —mediana y MAD por columna— admisible en un
        log-retorno. Se usa un z robusto y no un umbral porcentual porque el
        panel mezcla un índice de volatilidad con ETF de renta fija: un umbral
        del 10 % marcaría 836 observaciones correctas. El valor por defecto queda
        entre el peor evento real medido (27,0 en `credito_ig` el 29-09-2008) y
        la firma de un desdoblamiento 2:1 sin ajustar (87,7).
    congelado_max
        Sesiones consecutivas con cierre idéntico que se toleran. El valor por
        defecto es la racha máxima observada en el panel real. Importa porque una
        racha inventa retornos nulos que contraen `vol_realizada_z`.
    hueco_dias
        Salto máximo del índice, en días naturales, que se considera un cierre de
        mercado legítimo y no un tramo perdido.
    estricto
        Si es ``True`` lanza cuando alguna fila de `INVARIANTES` da ``fallo``,
        para que un `Run All` no siga sobre un panel roto. Los avisos no lanzan
        nunca.

    Returns
    -------
    pandas.DataFrame
        Diez filas indexadas por ``control``, con ``veredicto``
        (``ok`` / ``aviso`` / ``fallo``), ``medida``, ``limite`` y ``donde``.

    Raises
    ------
    ValueError
        Si `estricto` y algún invariante falla.
    """
    filas = []
    numerico = precios.select_dtypes(include=[np.number])

    # ── Invariantes ─────────────────────────────────────────────────────────
    no_float = [c for c in precios.columns if not pd.api.types.is_float_dtype(precios[c])]
    indice_ok = (
        isinstance(precios.index, pd.DatetimeIndex)
        and precios.index.tz is None
        and bool((precios.index.normalize() == precios.index).all())
    )
    filas.append((
        "tipos", len(no_float) + (0 if indice_ok else 1), 0,
        "todo float64 + DatetimeIndex naive" if not no_float and indice_ok
        else f"columnas no float: {no_float or 'ninguna'}; índice válido: {indice_ok}",
    ))

    idx = precios.index
    duplicadas = int(idx.duplicated().sum())
    desorden = int((idx.to_series().diff() <= pd.Timedelta(0)).sum())
    fin_semana = int((idx.dayofweek >= 5).sum())
    filas.append((
        "indice", duplicadas + desorden + fin_semana, 0,
        f"dup={duplicadas} desorden={desorden} fin_semana={fin_semana}",
    ))

    valores = numerico.to_numpy(dtype="float64", copy=False)
    no_finitos = int((~np.isfinite(valores)).sum())
    filas.append(("finitud", no_finitos, 0, "sin NaN ni inf" if not no_finitos
                  else f"{no_finitos} valores no finitos"))

    no_positivos = int((numerico <= 0).to_numpy().sum())
    minimo = float(numerico.to_numpy().min()) if numerico.size else np.nan
    filas.append(("positividad", no_positivos, 0,
                  f"mínimo global {minimo:.2f}" if not no_positivos
                  else f"{no_positivos} cierres <= 0"))

    constantes = [c for c in numerico.columns if numerico[c].nunique(dropna=False) <= 1]
    correlacion = numerico.pct_change().corr().abs().to_numpy()
    np.fill_diagonal(correlacion, 0.0)
    maxima = float(np.nanmax(correlacion)) if correlacion.size else 0.0
    gemelas = int((correlacion > 0.9999).sum() // 2)
    filas.append((
        "degeneradas", len(constantes) + gemelas, 0,
        f"corr máxima fuera de la diagonal {maxima:.3f}" if not constantes and not gemelas
        else f"constantes: {constantes}; pares idénticos: {gemelas}",
    ))

    # ── Avisos ──────────────────────────────────────────────────────────────
    igual = precios.eq(precios.shift())
    grupo = (~igual).cumsum()
    rachas = {c: int(igual[c].groupby(grupo[c]).sum().max()) + 1 for c in precios.columns}
    peor_racha = max(rachas, key=rachas.get)
    filas.append(("congelados", rachas[peor_racha], congelado_max,
                  f"{peor_racha}, racha de {rachas[peor_racha]} sesiones"))

    # Z robusto por columna: la mediana y la MAD se autoescalan al VIX, cuyos
    # movimientos legítimos multiplican por diez a los de un ETF de renta fija.
    retornos = np.log(precios.where(precios > 0)).diff()
    escala = (1.4826 * (retornos - retornos.median()).abs().median()).replace(0.0, np.nan)
    z = ((retornos - retornos.median()).abs() / escala)
    peor_z = float(np.nanmax(z.to_numpy())) if z.size else 0.0
    if np.isfinite(peor_z) and z.notna().to_numpy().any():
        pos = np.unravel_index(np.nanargmax(z.to_numpy()), z.shape)
        donde_z = f"{z.columns[pos[1]]} {z.index[pos[0]].date()}"
    else:
        donde_z = "sin retornos calculables"
    filas.append(("saltos", round(peor_z, 1), umbral_z, donde_z))

    salto = pd.Series(idx).diff().dt.days
    mayor = int(salto.max()) if salto.notna().any() else 0
    corte = int(salto.idxmax()) if salto.notna().any() else 0
    filas.append(("calendario", mayor, hueco_dias,
                  f"{idx[corte - 1].date()} a {idx[corte].date()}" if mayor else "sin saltos"))

    # Los años parciales del extremo se excluyen: no informan de pérdidas.
    por_anio = precios.groupby(idx.year).size()
    completos = por_anio.iloc[1:-1]
    minimo_anual = int(completos.min()) if len(completos) else 0
    filas.append(("densidad", minimo_anual, 240,
                  f"mínimo en {int(completos.idxmin())} ({minimo_anual} sesiones)"
                  if len(completos) else "sin años completos"))

    periodo = cargar_catalogo()["periodo"]
    desfase = abs((idx[0] - pd.Timestamp(periodo["inicio"])).days)
    filas.append(("cobertura", desfase, 7,
                  f"{idx[0].date()} a {idx[-1].date()} ({len(idx)} sesiones)"))

    # Relleno: cuántas sesiones consecutivas ha arrastrado `alinear`. Un relleno
    # de más de una sesión mete un retorno nulo y concentra en el siguiente el
    # movimiento de varios días, lo que contamina la volatilidad realizada.
    plano = precios.eq(precios.shift()).all(axis=1)
    grupo = (~plano).cumsum()
    relleno = int(plano.groupby(grupo).sum().max()) if len(plano) else 0
    filas.append(("relleno", relleno, 1, f"{relleno} sesiones planas seguidas como máximo"))

    tabla = pd.DataFrame(filas, columns=["control", "medida", "limite", "donde"]).set_index("control")
    excede = tabla["medida"] > tabla["limite"]
    # `densidad` es la única fila donde la medida debe SUPERAR al límite.
    excede["densidad"] = tabla.loc["densidad", "medida"] < tabla.loc["densidad", "limite"]
    tabla["veredicto"] = np.where(
        ~excede, "ok", np.where(tabla.index.isin(INVARIANTES), "fallo", "aviso")
    )
    tabla = tabla[["veredicto", "medida", "limite", "donde"]]

    if estricto and (tabla["veredicto"] == "fallo").any():
        rotos = list(tabla.index[tabla["veredicto"] == "fallo"])
        raise ValueError(
            f"El panel no supera los invariantes de integridad: {rotos}.\n{tabla.to_string()}"
        )
    return tabla


#: Controles del dataset de ventanas cuya violación es imposible en datos
#: correctos. Igual que en el panel, lanzan.
INVARIANTES_VENTANAS = (
    "finitud_X",
    "finitud_objetivos",
    "rango_regimen",
    "positividad_vol",
    "alineamiento",
)


def auditar_ventanas(
    conjuntos: dict,
    n_regimenes: int,
    *,
    minimo_por_clase: int = 30,
    estricto: bool = True,
) -> pd.DataFrame:
    """Tabla de control del dataset de ventanas: `X`, `y_reg` e `y_vol` a la vez.

    Hermana de `auditar_panel`, con la misma doctrina: ``fallo`` y ``aviso`` no
    son dos grados de gravedad sino dos naturalezas. Las cinco primeras filas son
    invariantes y lanzan; la última es un aviso.

    Cómo se lee: si todo está en ``ok``, el dataset que se va a escribir es
    consumible tal cual por los once cuadernos siguientes. Si algo da ``fallo``,
    **no se imputa nada**: `features.construir_canales` ya recortó las filas
    incompletas y `construir_ventanas` ya descarta las posiciones sin objetivo
    finito, así que un NaN aquí solo puede venir de un error aguas arriba —un
    canal nuevo mal declarado, un `shift` mal alineado, un parquet de otra
    ejecución— y rellenarlo lo convertiría en un error silencioso. La reparación
    es volver al cuaderno 00 o al 01, no tapar la fila.

    Qué la invalida: no comprueba que los **valores** sean correctos. Un dataset
    construido con el índice desalineado entre canales y objetivos pasa las seis
    filas. De eso se ocupan `features.contraste_causalidad` en el 00 y
    `ventanas.auditar_solape` aquí.

    Parameters
    ----------
    conjuntos
        ``{nombre: Conjunto}``, típicamente las tres particiones.
    n_regimenes
        Número de clases admisibles para ``y_reg``.
    minimo_por_clase
        Ventanas por debajo de las cuales una clase se considera demasiado
        escasa. 30 es una convención: por debajo, sus métricas por clase son
        ruido. No lanza.
    estricto
        Si es ``True`` lanza cuando algún invariante falla.

    Returns
    -------
    pandas.DataFrame
        Seis filas indexadas por ``control``, con ``veredicto``, ``medida``,
        ``limite`` y ``donde``.

    Raises
    ------
    ValueError
        Si `estricto` y algún invariante falla.
    """
    filas = []

    no_finitos = {n: int((~np.isfinite(c.X)).sum()) for n, c in conjuntos.items()}
    total = sum(no_finitos.values())
    filas.append(("finitud_X", total, 0,
                  "sin NaN ni inf en X" if not total else f"por partición: {no_finitos}"))

    no_finitos_y = {n: int((~np.isfinite(c.y_vol)).sum()) for n, c in conjuntos.items()}
    total_y = sum(no_finitos_y.values())
    filas.append(("finitud_objetivos", total_y, 0,
                  "sin NaN ni inf en y_vol" if not total_y else f"por partición: {no_finitos_y}"))

    fuera = {n: int(((c.y_reg < 0) | (c.y_reg >= n_regimenes)).sum()) for n, c in conjuntos.items()}
    filas.append(("rango_regimen", sum(fuera.values()), 0,
                  f"y_reg dentro de [0, {n_regimenes})" if not sum(fuera.values()) else f"{fuera}"))

    negativas = {n: int((c.y_vol <= 0).sum()) for n, c in conjuntos.items()}
    minimo_vol = min(float(c.y_vol.min()) for c in conjuntos.values())
    filas.append(("positividad_vol", sum(negativas.values()), 0,
                  f"mínimo de y_vol {minimo_vol:.4f}"))

    desalineados = sum(
        len({len(c.X), len(c.y_reg), len(c.y_vol), len(c.fechas)}) != 1
        or not c.fechas.is_monotonic_increasing
        for c in conjuntos.values()
    )
    filas.append(("alineamiento", desalineados, 0,
                  "longitudes iguales e índice creciente" if not desalineados
                  else "hay particiones descuadradas"))

    conteo = {
        f"{n}/{k}": int((c.y_reg == k).sum())
        for n, c in conjuntos.items()
        for k in range(n_regimenes)
    }
    escasa = min(conteo, key=conteo.get)
    filas.append(("clases_pobladas", conteo[escasa], minimo_por_clase,
                  f"la más escasa es {escasa} con {conteo[escasa]} ventanas"))

    tabla = pd.DataFrame(filas, columns=["control", "medida", "limite", "donde"]).set_index("control")
    excede = tabla["medida"] > tabla["limite"]
    # `clases_pobladas` es la única fila donde la medida debe SUPERAR al límite.
    excede["clases_pobladas"] = (
        tabla.loc["clases_pobladas", "medida"] < tabla.loc["clases_pobladas", "limite"]
    )
    tabla["veredicto"] = np.where(
        ~excede, "ok", np.where(tabla.index.isin(INVARIANTES_VENTANAS), "fallo", "aviso")
    )
    tabla = tabla[["veredicto", "medida", "limite", "donde"]]

    if estricto and (tabla["veredicto"] == "fallo").any():
        rotos = list(tabla.index[tabla["veredicto"] == "fallo"])
        raise ValueError(
            f"El dataset de ventanas no supera los invariantes: {rotos}.\n{tabla.to_string()}"
        )
    return tabla
