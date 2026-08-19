"""Métricas del modelo downstream y de la calidad de los datos sintéticos.

Dos bloques independientes:

**Downstream.** Mide si el dataset mixto produce un modelo mejor. El conjunto
de test es siempre real. En la tarea de régimen las clases están muy
desbalanceadas (la de crisis ronda el 10 %), así que el accuracy es engañoso:
un modelo que nunca prediga crisis acierta el 90 % y es inútil. Las métricas de
referencia son el F1 macro y el recall de la clase de crisis.

**Calidad sintética.** Mide si las muestras generadas se parecen a las reales,
con independencia de que ayuden o no al downstream. Un generador puede mejorar
el downstream por pura regularización mientras produce muestras irreconocibles,
y conviene saberlo.

**Líneas base.** Una métrica sola no dice nada: hay que saber qué da no hacer
nada. `lineas_base` produce las referencias triviales de la tarea de régimen
—incluida la persistencia de D21— y `linea_base_volatilidad` la de regresión.
`banda_bloques` da el intervalo de confianza que corresponde al tamaño muestral
real, que son episodios de crisis y no ventanas.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import DIR_METRICAS

#: Tareas downstream admitidas. Se replica aquí, y no se importa de
#: `downstream`, para que evaluar una tabla de métricas ya calculada no obligue
#: a cargar Keras.
TAREAS = ("regimen", "volatilidad")

#: Piso del QLIKE, como fracción de la mediana de `y_real`. Ver
#: `metricas_volatilidad` para la medición que fija el valor.
FRACCION_PISO_QLIKE = 0.10

# ─────────────────────────────────────────────────────────────────────────────
# Métricas del modelo downstream
# ─────────────────────────────────────────────────────────────────────────────


def _conteos(y_real: np.ndarray, y_pred: np.ndarray, n_clases: int) -> np.ndarray:
    """Matriz de conteos ``real x predicho``, sin pasar por sklearn.

    Todas las métricas de clasificación salen de esta matriz. Tenerlas en un
    único sitio es lo que impide que dos de ellas se calculen sobre conjuntos de
    clases distintos, que es exactamente el defecto que tenía la versión
    anterior (`f1_score` recibía `labels` y `balanced_accuracy_score` no, y con
    una clase ausente de `y_real` devolvían 0,489 y 0,750).
    """
    y_real = np.asarray(y_real).ravel().astype(np.int64)
    y_pred = np.asarray(y_pred).ravel().astype(np.int64)
    if len(y_real) != len(y_pred):
        raise ValueError(
            f"y_real tiene {len(y_real)} elementos y y_pred {len(y_pred)}: "
            "no son la misma evaluacion."
        )
    extremos = (
        (int(y_real.min()), int(y_real.max()), int(y_pred.min()), int(y_pred.max()))
        if len(y_real)
        else ()
    )
    fuera = [v for v in extremos if not 0 <= v < n_clases]
    if fuera:
        raise ValueError(
            f"Hay etiquetas fuera de [0, {n_clases}): {sorted(set(fuera))}. "
            "Revisa `n_clases` o si estas pasando probabilidades sin argmax."
        )
    return np.bincount(
        y_real * n_clases + y_pred, minlength=n_clases * n_clases
    ).reshape(n_clases, n_clases)


def _desde_conteos(conteos: np.ndarray, n_clases: int) -> dict[str, float]:
    """Traduce la matriz de conteos a las métricas publicadas.

    Política de indefinidos, deliberada y uniforme: **lo que no se puede medir
    es NaN, no cero**. Una métrica de la clase de crisis vale NaN cuando su
    denominador es cero —el conjunto no contiene crisis, o el modelo no predijo
    ninguna—, y no 0,0. Con `zero_division=0` un conjunto sin crisis producía un
    recall de 0,0 indistinguible de un modelo que falla todas, y el control de
    integridad del notebook 12, que busca NaN, no podía dispararse nunca.

    Las dos métricas macro promedian sobre las clases **presentes en `y_real`**,
    que es la convención de `balanced_accuracy_score`: una clase que no aparece
    en el conjunto de evaluación no tiene recall, y contarla como 0 mezcla "el
    modelo falla" con "no había nada que medir". Sus falsos positivos siguen
    penalizando, vía el recall de la clase verdadera.
    """
    crisis = n_clases - 1
    total = int(conteos.sum())
    reales = conteos.sum(axis=1)
    predichos = conteos.sum(axis=0)
    aciertos = np.diag(conteos)
    presentes = reales > 0

    vacio = np.full(n_clases, np.nan)
    recall = np.divide(aciertos, reales, out=vacio.copy(), where=reales > 0)
    precision = np.divide(aciertos, predichos, out=vacio.copy(), where=predichos > 0)
    # F1 en su forma 2*aciertos / (soporte real + soporte predicho): evita el
    # 0/0 de precision y recall por separado cuando el modelo nunca predice la
    # clase, caso en que el F1 vale 0 y no es un indefinido.
    suma = reales + predichos
    f1 = np.divide(2 * aciertos, suma, out=vacio.copy(), where=suma > 0)

    soporte_crisis = int(reales[crisis])
    return {
        "f1_macro": float(f1[presentes].mean()) if presentes.any() else np.nan,
        "balanced_accuracy": float(recall[presentes].mean()) if presentes.any() else np.nan,
        "accuracy": float(aciertos.sum() / total) if total else np.nan,
        "recall_crisis": float(recall[crisis]),
        "precision_crisis": float(precision[crisis]),
        # Sin ventanas de crisis no hay F1 de crisis, aunque el modelo prediga
        # crisis y la fórmula devuelva 0: el 0 sería un juicio, y no lo hay.
        "f1_crisis": float(f1[crisis]) if soporte_crisis else np.nan,
        "soporte_crisis": soporte_crisis,
    }


def metricas_regimen(
    y_real: np.ndarray, y_pred: np.ndarray, n_clases: int = 3
) -> dict[str, float]:
    """Métricas de clasificación, con foco en la clase minoritaria.

    Se reporta `recall_crisis` aparte porque es la magnitud que decide si el
    trabajo tiene sentido: si añadir sintéticos de crisis no sube el recall de
    crisis, la hipótesis del taller no se sostiene, por bien que se comporte el
    resto de métricas.

    Parameters
    ----------
    y_real
        Etiqueta verdadera del conjunto evaluado.
    y_pred
        Predicción del modelo, ya en clases (no probabilidades).
    n_clases
        Número de regímenes; el de crisis es siempre ``n_clases - 1``.

    Returns
    -------
    dict
        ``f1_macro``, ``balanced_accuracy``, ``accuracy``, ``recall_crisis``,
        ``precision_crisis``, ``f1_crisis`` y ``soporte_crisis``.

    Notes
    -----
    Cómo se lee: `soporte_crisis` no es una métrica sino el número de ventanas
    de crisis del conjunto evaluado, y hay que leerlo **antes** que las tres
    métricas de crisis: con 110 ventanas en 3 episodios, una diferencia de
    recall de un par de puntos no distingue dos modelos (ver `banda_bloques`).
    Un NaN en las métricas de crisis significa denominador cero, no fracaso.

    Qué la invalida: pasar probabilidades en vez de clases, o un `n_clases`
    distinto del que produjo `y_pred`. Ambos casos lanzan ahora `ValueError` en
    vez de devolver un número sin sentido.

    Las cifras coinciden con `sklearn` cuando todas las clases están presentes
    en `y_real`, que es el caso de test; se calculan a mano para poder fijar la
    política de indefinidos descrita en `_desde_conteos`.
    """
    return _desde_conteos(_conteos(y_real, y_pred, n_clases), n_clases)


def metricas_volatilidad(
    y_real: np.ndarray, y_pred: np.ndarray, referencia: float | None = None
) -> dict[str, float]:
    """Métricas de regresión sobre la volatilidad realizada futura.

    Se incluye QLIKE además de MAE y RMSE: es la pérdida estándar en previsión
    de volatilidad porque penaliza de forma asimétrica, castigando más
    infraestimar el riesgo que sobreestimarlo.

    Parameters
    ----------
    referencia
        Predictor constante contra el que se calcula el R2. ``None`` (por
        defecto, y comportamiento histórico) usa la **media del propio conjunto
        evaluado**: es un oráculo —nadie conoce esa media antes de ver el
        test— y sube el R2 de todas las recetas por igual. Pasar la media de
        train da el R2 honesto, el único comparable con lo que un modelo podía
        saber. Los dos números son legítimos, pero no son el mismo número y no
        se pueden mezclar en una tabla.

    Returns
    -------
    dict
        ``mae``, ``rmse``, ``sesgo``, ``qlike``, ``r2`` y el diagnóstico
        ``diag_predicciones_recortadas``.

    Notes
    -----
    Cómo se lee el QLIKE: es una pérdida, cuanto menor mejor, pero no es
    robusto. Con `y_real` a 0,151 y 1.052 muestras: predicción perfecta 0,000;
    sesgo del -10 % en **todas** las predicciones 0,0058; **una sola**
    predicción negativa, con el piso de 1e-6 de la versión anterior, **143,5**.
    Es decir, un único valor recortado pesaba 25.000 veces más que un sesgo del
    10 % en todo el conjunto, y el ranking de generadores por QLIKE era el
    ranking de "cuántos negativos escupió la red". Peor: el contador antiguo
    miraba ``y_pred <= 0`` y no veía las predicciones en ``(0, 1e-6)``, que se
    recortan igual; con una predicción de 1e-7 el QLIKE valía 143,5 y el
    diagnóstico decía que no se había recortado nada.

    Ahora el piso es relativo a la escala del objetivo,
    ``FRACCION_PISO_QLIKE * mediana(y_real)``, y el contador cuenta exactamente
    lo que toca el piso. Medido sobre el test real (mediana 0,1325, piso
    0,0133): una predicción recortada pasa de aportar 267,1 a aportar 0,0163,
    del orden del sesgo del 10 % (0,0058) en vez de tres órdenes por encima.

    Qué lo invalida: si ``diag_predicciones_recortadas`` es grande, el QLIKE ya
    no mide previsión sino cuántas salidas degeneradas hubo. El piso hace la
    métrica comparable entre recetas, no borra el problema: hay que mirar el
    contador antes de leer la columna.
    """
    y_real = np.asarray(y_real, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()
    if len(y_real) != len(y_pred):
        raise ValueError(
            f"y_real tiene {len(y_real)} elementos y y_pred {len(y_pred)}: "
            "no son la misma evaluacion."
        )

    error = y_pred - y_real
    metricas = {
        "mae": float(np.abs(error).mean()),
        "rmse": float(np.sqrt((error**2).mean())),
        "sesgo": float(error.mean()),
    }

    # QLIKE exige positividad en ambos argumentos y diverge cuando el
    # denominador tiende a cero. El piso se ata a la escala del objetivo para
    # que el recorte cueste lo que cuesta un error grande, y no infinito.
    escala = float(np.median(np.abs(y_real)))
    piso = FRACCION_PISO_QLIKE * escala if escala > 0 else 1e-6
    real_pos = np.maximum(y_real, piso)
    pred_pos = np.maximum(y_pred, piso)
    metricas["qlike"] = float(np.mean(real_pos / pred_pos - np.log(real_pos / pred_pos) - 1.0))
    # Prefijo `diag_`: es un recuento de diagnóstico, no una métrica, y acaba
    # como columna de `barrido.csv` junto a las demás. Sin el prefijo se lee
    # como si fuera comparable con un MAE.
    metricas["diag_predicciones_recortadas"] = int((y_pred < piso).sum())

    # R2 contra un predictor constante. Ver `referencia` en Parameters.
    centro = float(np.mean(y_real)) if referencia is None else float(referencia)
    varianza = ((y_real - centro) ** 2).mean()
    metricas["r2"] = float(1.0 - (error**2).mean() / varianza) if varianza > 0 else np.nan
    return metricas


def evaluar(
    modelo,
    X_test: np.ndarray,
    y_test: np.ndarray,
    tarea: str,
    conjunto: str = "test",
    tam_lote: int = 512,
    referencia: float | None = None,
) -> dict[str, float | str]:
    """Evalúa un modelo entrenado sobre un conjunto real.

    Parameters
    ----------
    tarea
        Una de `TAREAS`. Se valida: antes, cualquier cadena que no fuera
        exactamente ``"regimen"`` caía en la rama de regresión y devolvía MAE y
        QLIKE calculados sobre etiquetas enteras de régimen, sin avisar.
    conjunto
        Nombre del conjunto evaluado, que viaja **dentro** del diccionario. Sin
        él es imposible auditar a posteriori si una cifra salió de validación o
        de test, y esa ambigüedad es justo el camino por el que se fuga el test
        en un barrido largo.
    tam_lote
        Lote de inferencia. Medido sobre las 1.052 ventanas de test: 0,27 s con
        512 frente a 0,57 s con el valor por defecto de Keras. Con 570
        entrenamientos por delante son minutos regalados.
    referencia
        Se pasa tal cual a `metricas_volatilidad`; se ignora en clasificación.

    Returns
    -------
    dict
        Las métricas de la tarea más ``conjunto``. **Ojo**: el diccionario deja
        de ser puramente numérico, así que ``pd.Series(m).round(4)`` falla con
        `TypeError`. Para mostrarlo, `como_serie(m)`.

    Raises
    ------
    ValueError
        Si `tarea` no está en `TAREAS`.
    """
    if tarea not in TAREAS:
        raise ValueError(f"Tarea desconocida: {tarea!r}. Opciones: {TAREAS}")

    predicho = modelo.predict(X_test, verbose=0, batch_size=tam_lote)
    if tarea == "regimen":
        metricas = metricas_regimen(
            y_test, predicho.argmax(axis=1), n_clases=predicho.shape[1]
        )
    else:
        metricas = metricas_volatilidad(y_test, predicho, referencia=referencia)

    return {**metricas, "conjunto": conjunto}


def como_serie(metricas: dict, decimales: int = 4) -> pd.Series:
    """Serie legible de un diccionario de métricas con campos no numéricos.

    `evaluar` devuelve también `conjunto`, que es texto, y eso convierte la
    serie en `object`: `pandas` entonces no redondea y `Series.round` lanza
    `TypeError`. Esta función redondea lo numérico y deja el texto intacto.
    """
    return pd.Series(
        {
            k: (round(v, decimales) if isinstance(v, float) else v)
            for k, v in metricas.items()
        }
    )


def matriz_confusion(
    y_real: np.ndarray,
    y_pred: np.ndarray,
    n_clases: int = 3,
    estricto: bool = True,
) -> pd.DataFrame:
    """Matriz de confusión con nombres legibles de régimen.

    Parameters
    ----------
    y_real
        Etiqueta **verdadera**. Va en las filas.
    y_pred
        **Predicción** que se está juzgando. Va en las columnas. En la línea
        base de persistencia la predicción es el régimen de HOY, y la verdad es
        el régimen futuro: es el orden contrario al que sugiere la intuición
        cronológica, y por eso existe la comprobación de abajo.
    n_clases
        Número de regímenes.
    estricto
        Si es True (por defecto), lanza `ValueError` cuando los nombres de las
        series delatan que los ejes van cambiados. Ponerlo a False lo desactiva.

    Returns
    -------
    pandas.DataFrame
        Filas indexadas por la clase real, columnas por la predicha.

    Notes
    -----
    Cómo se lee: cada fila es una clase verdadera, así que **el recall se lee
    por filas** (diagonal dividida por el total de la fila) y la precisión por
    columnas. Transponer la matriz intercambia exactamente esas dos lecturas sin
    cambiar ni el accuracy ni el F1, que son simétricos: por eso un cambio de
    ejes no se detecta mirando el total.

    Qué la invalida: pasar los argumentos al revés. Ocurrió: el notebook 01
    llamaba ``matriz_confusion(regimen, regimen_futuro)``, con la predicción en
    el eje de la verdad, y publicó como "recall de crisis 81,4 %" lo que era la
    **precisión**; el recall verdadero sobre muestra completa es 82,1 %. El
    accuracy (82,1 %) y el F1 de crisis (0,817) coincidían en las dos
    orientaciones, así que nada chirriaba.

    Contra eso, dos defensas. La estructural: `linea_base_persistencia` recibe
    ``y_reg`` y ``regimen_hoy``, nombres que no se pueden confundir, y es la vía
    recomendada para calcular esa línea base. La de aquí: si `y_pred` llega con
    nombre de verdad (``regimen_futuro``, ``objetivo``, ``y_real``...) o `y_real`
    con nombre de predicción (``regimen_hoy``, ``persistencia``...), se lanza
    `ValueError`. Solo actúa sobre `pandas.Series` con nombre, así que las
    llamadas con arrays de numpy —las de los notebooks 03 y 13— no cambian.
    """
    if estricto:
        _exigir_ejes(y_real, y_pred)

    nombres = nombres_regimenes(n_clases)
    matriz = _conteos(y_real, y_pred, n_clases)
    return pd.DataFrame(matriz, index=nombres, columns=nombres).rename_axis(
        index="real", columns="predicho"
    )


#: Fragmentos que, en el nombre de una serie, delatan el papel que juega. La
#: lista es corta a propósito: solo cubre los nombres que este repositorio usa,
#: porque un detector genérico daría falsos positivos y acabaría desactivado.
_NOMBRES_DE_VERDAD = ("futuro", "verdad", "objetivo", "etiqueta", "y_real", "y_reg")
_NOMBRES_DE_PREDICCION = ("pred", "hoy", "persistencia", "causal", "estimad")


def _exigir_ejes(y_real, y_pred) -> None:
    """Lanza si los nombres de las series indican que los ejes van cambiados."""

    def _nombre(x) -> str:
        return str(getattr(x, "name", "") or "").lower()

    real, pred = _nombre(y_real), _nombre(y_pred)
    sospechas = [t for t in _NOMBRES_DE_VERDAD if t in pred]
    sospechas += [t for t in _NOMBRES_DE_PREDICCION if t in real]
    if sospechas:
        raise ValueError(
            f"Ejes probablemente cambiados: y_real={real!r}, y_pred={pred!r}. "
            "El primer argumento es la verdad (filas) y el segundo la prediccion "
            "(columnas); para la persistencia, y_real=regimen_futuro y "
            "y_pred=regimen de hoy. Usa linea_base_persistencia(), o pasa "
            "estricto=False si el nombre despista."
        )


def nombres_regimenes(n_clases: int = 3) -> list[str]:
    """Etiquetas legibles de los regímenes, para figuras y tablas."""
    if n_clases == 2:
        return ["calma", "crisis"]
    if n_clases == 3:
        return ["calma", "transición", "crisis"]
    return [f"régimen {k}" for k in range(n_clases - 1)] + ["crisis"]


# ─────────────────────────────────────────────────────────────────────────────
# Líneas base (D21)
#
# Ninguna cifra del notebook 12 significa nada por sí sola: hay que saber contra
# qué se compara. Estas funciones producen esas referencias, y ninguna de ellas
# entrena nada.
# ─────────────────────────────────────────────────────────────────────────────


def linea_base_persistencia(
    y_reg: np.ndarray, regimen_hoy: np.ndarray, n_clases: int = 3
) -> dict[str, float]:
    """Métricas del predictor trivial "mañana será como hoy" (D21).

    Los regímenes son muy persistentes —la diagonal de la matriz de transición
    ronda 0,94-0,98—, así que esta línea base es fortísima: un clasificador que
    no la bata no está prediciendo el futuro, está leyendo el presente.

    Parameters
    ----------
    y_reg
        Etiqueta verdadera: el régimen dominante del periodo futuro.
    regimen_hoy
        La **predicción**: el régimen del día en que termina cada ventana.
        Decodificado con `EtiquetadorRegimenes.predict_causal` si se quiere la
        barra honesta, o con `predict` (Viterbi) si se quiere la referencia con
        información futura.

    Returns
    -------
    dict
        Las mismas claves que `metricas_regimen`.

    Notes
    -----
    Los nombres de los parámetros son la defensa contra el fallo que ya ocurrió
    una vez: con `y_real` y `y_pred` es fácil poner el régimen de hoy en el eje
    de la verdad, y entonces lo que se publica como recall es la precisión. Ver
    `matriz_confusion`.
    """
    return metricas_regimen(y_reg, regimen_hoy, n_clases=n_clases)


def lineas_base(
    y_reg: np.ndarray,
    regimen_hoy_causal: np.ndarray,
    regimen_hoy_viterbi: np.ndarray,
    y_reg_train: np.ndarray,
    n_clases: int = 3,
    semillas: int = 200,
) -> pd.DataFrame:
    """Tabla con todas las referencias triviales de la tarea de régimen.

    Parameters
    ----------
    y_reg
        Etiqueta verdadera del conjunto evaluado.
    regimen_hoy_causal, regimen_hoy_viterbi
        Régimen del día final de cada ventana, decodificado con el filtro
        forward y con Viterbi respectivamente.
    y_reg_train
        Etiquetas de train. De ahí salen la clase mayoritaria y las frecuencias
        del muestreo al azar; usar las del conjunto evaluado sería mirar el
        test para construir la referencia con la que se juzga el test.
    semillas
        Réplicas del muestreo al azar que se promedian.

    Returns
    -------
    pandas.DataFrame
        Una fila por referencia, las columnas de `metricas_regimen` y la
        booleana `usa_futuro`.

    Notes
    -----
    Cómo se lee: la fila que manda es **`persistencia_causal`**, la única
    implementable en el instante `t`. `persistencia_viterbi` se publica al lado,
    marcada con ``usa_futuro=True``, porque decodifica el estado de hoy con la
    serie entera —incluido lo que pasó después— y por eso sale mejor: sobre test,
    f1_macro 0,790 frente a 0,754. Presentarla como la barra a batir regala 3,6
    puntos a cualquier modelo.

    `siempre_crisis` y `azar_estratificado` no son alternativas serias: están
    para que la columna `recall_crisis` no se pueda presentar como un logro sin
    contexto. "Siempre crisis" tiene recall de crisis 1,000, el máximo posible, y
    es inútil: su precisión es la frecuencia de la clase y su F1 macro se
    desploma. Cualquier lectura de `recall_crisis` que no mire a la vez
    `precision_crisis` y `f1_macro` está reproduciendo esa trampa.

    Qué la invalida: que `regimen_hoy_*` no esté alineado con `y_reg` ventana a
    ventana. Los dos vectores deben venir del mismo conjunto y en el mismo
    orden; la forma segura es indexar la serie diaria de régimen con
    `conjunto.fechas`.
    """
    y_reg = np.asarray(y_reg).ravel().astype(np.int64)
    y_reg_train = np.asarray(y_reg_train).ravel().astype(np.int64)
    crisis = n_clases - 1

    frecuencias = np.bincount(y_reg_train, minlength=n_clases) / len(y_reg_train)
    mayoritaria = int(np.argmax(frecuencias))

    # El azar se promedia sobre varias réplicas: una sola tiraría un número que
    # cambia de una ejecución a otra y no sirve como referencia publicada.
    rng = np.random.default_rng(42)
    replicas = [
        metricas_regimen(y_reg, rng.choice(n_clases, size=len(y_reg), p=frecuencias), n_clases)
        for _ in range(semillas)
    ]
    azar = {
        k: float(np.mean([r[k] for r in replicas])) for k in replicas[0]
    }

    filas = {
        "mayoritaria_train": metricas_regimen(
            y_reg, np.full(len(y_reg), mayoritaria), n_clases
        ),
        "azar_estratificado": azar,
        "siempre_crisis": metricas_regimen(y_reg, np.full(len(y_reg), crisis), n_clases),
        "persistencia_causal": linea_base_persistencia(y_reg, regimen_hoy_causal, n_clases),
        "persistencia_viterbi": linea_base_persistencia(y_reg, regimen_hoy_viterbi, n_clases),
    }

    tabla = pd.DataFrame(filas).T
    tabla["soporte_crisis"] = tabla["soporte_crisis"].astype(int)
    tabla["usa_futuro"] = tabla.index == "persistencia_viterbi"
    return tabla.rename_axis("linea_base")


def banda_bloques(
    y_reg: np.ndarray,
    y_pred: np.ndarray,
    metrica: str = "recall_crisis",
    longitud: int = 81,
    replicas: int = 20000,
    semilla: int = 42,
    n_clases: int = 3,
) -> tuple[float, float]:
    """Intervalo de confianza al 95 % por bootstrap de bloques circular.

    Parameters
    ----------
    y_reg, y_pred
        Verdad y predicción, **en orden cronológico**. El bootstrap de bloques
        no significa nada sobre un vector barajado.
    metrica
        Cualquier clave de `metricas_regimen`.
    longitud
        Longitud del bloque, en ventanas. Por defecto 81, que es
        ``pasado + horizonte``: la huella de una ventana en sesiones y, por
        tanto, la distancia a la que dos ventanas dejan de compartir ni una sola
        sesión. Es el mismo 81 del que el catálogo deriva el embargo entre
        particiones (mínimo 80, adoptado 85). Remuestrear ventana a ventana
        supondría que son independientes, y no lo son.
    replicas
        Réplicas del bootstrap.

    Returns
    -------
    tuple of float
        Percentiles 2,5 y 97,5 de la métrica sobre las réplicas.

    Notes
    -----
    Cómo se lee: es el ancho de la banda lo que informa, no el centro. Las 110
    ventanas de crisis del test **no son 110 observaciones**: son 3 episodios
    contiguos (`regimenes.tramos_contiguos` lo cuenta). Medido sobre la
    persistencia en test, `recall_crisis` = 0,800 con IC 95 % **[0,560, 1,000]**,
    frente al **[0,716, 0,864]** del intervalo binomial de Wilson que trata las
    ventanas como independientes: el honesto es **3,0 veces más ancho**.

    De ahí la consecuencia práctica, que hay que escribir en el informe y no
    solo aquí: **ninguna comparación de `recall_crisis` entre dos recetas del
    notebook 12 que difiera en menos de unos 20 puntos es distinguible del
    ruido.** Ordenar la tabla por esa columna y quedarse con la primera fila es
    quedarse con el ruido más afortunado.

    Qué lo invalida: aplicarlo a métricas de la clase de crisis en conjuntos que
    apenas la contienen. Las réplicas que no capturan ninguna ventana de crisis
    dan métrica indefinida y se descartan; si se descartan muchas, el intervalo
    describe las réplicas afortunadas y no el conjunto. En el test son 220 de
    20.000 (1,1 %), que no mueve el resultado; en validación, con la crisis mucho
    más escasa, habría que comprobarlo antes de publicar la banda.
    """
    claves = _desde_conteos(np.zeros((n_clases, n_clases), dtype=np.int64), n_clases)
    if metrica not in claves:
        raise ValueError(
            f"Metrica desconocida: {metrica!r}. Opciones: {tuple(claves)}"
        )

    y_reg = np.asarray(y_reg).ravel().astype(np.int64)
    y_pred = np.asarray(y_pred).ravel().astype(np.int64)
    n = len(y_reg)
    if longitud >= n:
        raise ValueError(
            f"El bloque ({longitud}) no cabe en el conjunto ({n} ventanas)."
        )

    rng = np.random.default_rng(semilla)
    n_bloques = int(np.ceil(n / longitud))
    # Circular: los bloques se toman en modulo n, de modo que toda ventana tiene
    # la misma probabilidad de entrar. Sin la vuelta al principio, las del final
    # de la serie estarian infrarrepresentadas.
    inicios = rng.integers(0, n, size=(replicas, n_bloques))
    desplazamiento = np.arange(longitud)

    valores = np.empty(replicas)
    for i in range(replicas):
        indice = (inicios[i][:, None] + desplazamiento).ravel()[:n] % n
        valores[i] = _desde_conteos(
            _conteos(y_reg[indice], y_pred[indice], n_clases), n_clases
        )[metrica]

    valores = valores[np.isfinite(valores)]
    if not len(valores):
        return (np.nan, np.nan)
    return float(np.percentile(valores, 2.5)), float(np.percentile(valores, 97.5))


def linea_base_volatilidad(
    y_vol_train: np.ndarray, y_vol_eval: np.ndarray
) -> dict[str, float]:
    """Referencia trivial de la tarea de regresión: predecir la media de train.

    Hace falta porque no hay ninguna equivalente a la persistencia: el canal
    `vol_realizada_z` está z-scoreado con estadísticos causales, así que la
    volatilidad de hoy **no se puede leer de `X`** en las unidades de `y_vol`.
    Sin esta fila no hay forma de saber si un MAE de 0,04 es bueno o es lo que
    da no hacer nada.

    Returns
    -------
    dict
        Claves de `metricas_volatilidad`, con el R2 calculado contra la media de
        train. Vale 0,000 por construcción: es el cero honesto de la tarea, y el
        R2 de cualquier receta hay que leerlo contra él y no contra el R2 con
        media de test, que es un oráculo.
    """
    media = float(np.asarray(y_vol_train, dtype=np.float64).mean())
    y_vol_eval = np.asarray(y_vol_eval, dtype=np.float64).ravel()
    return metricas_volatilidad(
        y_vol_eval, np.full(len(y_vol_eval), media), referencia=media
    )


# ─────────────────────────────────────────────────────────────────────────────
# Calidad de los datos sintéticos
# ─────────────────────────────────────────────────────────────────────────────


def proyector(reales: np.ndarray, n_componentes: int = 50, semilla: int = 42):
    """Ajusta una PCA sobre los datos reales para comparar en un espacio útil.

    Las métricas basadas en distancias no discriminan en el espacio original: el
    bloque tiene ~1.200 dimensiones y apenas unos miles de muestras, así que
    todas las distancias entre pares convergen al mismo valor y el vecino más
    cercano deja de significar nada. Proyectar a ~50 componentes devuelve al
    régimen en que las distancias son informativas.

    La PCA se ajusta **solo con los datos reales de entrenamiento**: los ejes
    deben representar la estructura del mercado, no la del generador que se está
    juzgando. Si se reajustara por generador, cada uno se evaluaría en su propio
    espacio y los resultados no serían comparables entre sí.
    """
    from sklearn.decomposition import PCA

    n_componentes = min(n_componentes, *reales.shape)
    return PCA(n_components=n_componentes, random_state=semilla).fit(reales)


def puntuacion_discriminativa(
    reales: np.ndarray, sinteticos: np.ndarray, semilla: int = 42, pca=None
) -> dict[str, float]:
    """Entrena un clasificador para distinguir reales de sintéticos.

    Es la métrica de fidelidad más informativa. Un AUC de 0.5 significa que el
    clasificador no logra separarlos: las distribuciones son indistinguibles.
    Un AUC cercano a 1 significa que el generador deja una firma evidente.

    Se reporta ``|AUC - 0.5|`` porque es lo que hay que minimizar; el AUC crudo
    se incluye para poder interpretar el signo.

    Referencia de lectura: por debajo de 0.60 el generador es bueno, entre 0.60
    y 0.75 aceptable, por encima de 0.85 hay que rechazarlo.
    """
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import train_test_split

    # El clasificador se entrena en el espacio proyectado, no en las ~1.200
    # dimensiones originales. Dos razones: en dimensión tan alta con estas
    # muestras el clasificador separa siempre (memoriza) y el AUC deja de
    # informar; y el coste de ajuste se vuelve prohibitivo cuando hay que
    # repetirlo por cada generador.
    pca = pca if pca is not None else proyector(reales, semilla=semilla)
    reales_p, sinteticos_p = pca.transform(reales), pca.transform(sinteticos)

    n = min(len(reales_p), len(sinteticos_p))
    rng = np.random.default_rng(semilla)
    X = np.concatenate(
        [
            reales_p[rng.choice(len(reales_p), n, replace=False)],
            sinteticos_p[rng.choice(len(sinteticos_p), n, replace=False)],
        ]
    )
    y = np.concatenate([np.ones(n), np.zeros(n)])

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.3, random_state=semilla, stratify=y
    )

    # Boosting por histogramas: aquí sí conviene un modelo flexible, porque el
    # objetivo es detectar cualquier diferencia entre distribuciones, incluidas
    # las no lineales que una logística no vería.
    clasificador = HistGradientBoostingClassifier(
        max_iter=150, early_stopping=True, random_state=semilla
    )
    clasificador.fit(X_tr, y_tr)
    auc = float(roc_auc_score(y_te, clasificador.predict_proba(X_te)[:, 1]))

    return {"auc_discriminativo": auc, "distancia_a_indistinguible": abs(auc - 0.5)}


def distancia_vecino_mas_cercano(
    reales: np.ndarray,
    sinteticos: np.ndarray,
    muestra: int = 500,
    semilla: int = 42,
    pca=None,
) -> dict[str, float]:
    """Test de memorización: cuánto se acercan los sintéticos a los reales.

    Si el generador memoriza, sus muestras caen prácticamente encima de puntos
    de entrenamiento y la distancia mínima colapsa a cero. Se compara contra la
    distancia entre reales para tener una referencia de qué es "cerca" en este
    espacio: un cociente muy por debajo de 1 delata copia.

    Es una **condición de validez**, no una métrica más: si el generador copia,
    cualquier mejora que produzca aguas abajo es un artefacto y el resto de
    resultados no significan nada.

    Parameters
    ----------
    pca
        Proyector devuelto por `proyector()`. Si se omite se ajusta uno sobre
        `reales`. Comparar en el espacio original de ~1.200 dimensiones no
        funciona: con tan pocas muestras por dimensión todas las distancias se
        parecen y el test pierde poder.
    """
    from sklearn.neighbors import NearestNeighbors

    pca = pca if pca is not None else proyector(reales, semilla=semilla)
    reales_p, sinteticos_p = pca.transform(reales), pca.transform(sinteticos)

    rng = np.random.default_rng(semilla)
    ref = reales_p[rng.choice(len(reales_p), min(muestra, len(reales_p)), replace=False)]
    gen = sinteticos_p[rng.choice(len(sinteticos_p), min(muestra, len(sinteticos_p)), replace=False)]

    d_sint, _ = NearestNeighbors(n_neighbors=1).fit(reales_p).kneighbors(gen)
    # Para los reales se pide el segundo vecino: el primero es el punto mismo.
    d_real, _ = NearestNeighbors(n_neighbors=2).fit(reales_p).kneighbors(ref)
    d_real = d_real[:, 1]

    mediana_real = float(np.median(d_real))
    mediana_sint = float(np.median(d_sint))
    return {
        "dvmc_sintetico": mediana_sint,
        "dvmc_real": mediana_real,
        # Cociente cercano a 1: el sintético está tan lejos de los reales como
        # los reales entre sí, que es lo esperable de un generador honesto.
        "cociente_dvmc": mediana_sint / mediana_real if mediana_real > 0 else np.nan,
        # Fracción de sintéticos más cercanos a un real que la mediana real:
        # por encima de ~0.5 empieza a haber sospecha de copia.
        "frac_mas_cerca": float((d_sint.ravel() < mediana_real).mean()),
        "duplicados_exactos": int((d_sint.ravel() < 1e-8).sum()),
    }


def comparar_momentos(reales: np.ndarray, sinteticos: np.ndarray) -> pd.DataFrame:
    """Compara los cuatro primeros momentos, dimensión a dimensión.

    La curtosis es la fila que importa: los retornos financieros tienen colas
    gruesas y un generador gaussiano no puede reproducirlas por construcción.
    Esta tabla lo demuestra de forma cuantitativa.
    """
    from scipy import stats

    def _resumen(X: np.ndarray) -> dict[str, float]:
        return {
            "media": float(X.mean()),
            "desviacion": float(X.std()),
            "asimetria": float(np.mean(stats.skew(X, axis=0))),
            "curtosis": float(np.mean(stats.kurtosis(X, axis=0))),
        }

    return pd.DataFrame({"real": _resumen(reales), "sintetico": _resumen(sinteticos)}).assign(
        diferencia=lambda d: d["sintetico"] - d["real"]
    )


def _a_canales(bloque: np.ndarray, pasado: int, n_canales: int) -> np.ndarray:
    """Deshace el aplanado y apila los instantes: ``(n, d)`` -> ``(n*pasado, canales)``.

    Se descarta la última columna del bloque, que es la volatilidad futura y no
    forma parte de la ventana.
    """
    n = len(bloque)
    return bloque[:, : pasado * n_canales].reshape(n * pasado, n_canales)


def error_correlaciones(
    reales: np.ndarray, sinteticos: np.ndarray, pasado: int, n_canales: int
) -> float:
    """Error relativo entre las matrices de correlación **entre canales**.

    Se compara la matriz de ``canales × canales`` (20 × 20), no la del bloque
    aplanado completo (1.200 × 1.200). La segunda tendría del orden de 720.000
    entradas independientes estimadas con unos miles de muestras: casi todo
    ruido, y su norma mediría el error de estimación más que el del generador.

    La matriz entre canales, en cambio, captura lo que de verdad define un
    panel de mercado: que los sectores se muevan juntos, que la correlación
    acción-bono tenga signo, que el VIX suba cuando el índice cae.
    """
    corr_real = np.corrcoef(_a_canales(reales, pasado, n_canales), rowvar=False)
    corr_sint = np.corrcoef(_a_canales(sinteticos, pasado, n_canales), rowvar=False)
    np.nan_to_num(corr_real, copy=False)
    np.nan_to_num(corr_sint, copy=False)
    # Normalizado por la norma de la real, para que sea un error relativo
    # comparable entre generadores.
    return float(np.linalg.norm(corr_real - corr_sint, ord="fro") / np.linalg.norm(corr_real, ord="fro"))


def error_autocorrelacion(
    reales: np.ndarray, sinteticos: np.ndarray, pasado: int, n_canales: int, lags: int = 20
) -> dict[str, float]:
    """Compara la función de autocorrelación de los retornos y de su valor absoluto.

    Es la métrica que separa de verdad a los generadores. Los retornos casi no
    tienen autocorrelación, y reproducir eso es fácil; el valor absoluto sí la
    tiene y decae lentamente —el agrupamiento de volatilidad—, y ahí es donde un
    generador que solo ha aprendido medias y covarianzas se delata.

    Se usa el primer canal (retorno del índice), que es el de referencia.
    """

    def _acf(bloque: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        # Cada fila es una ventana temporal completa del canal 0.
        serie = bloque[:, : pasado * n_canales].reshape(len(bloque), pasado, n_canales)[:, :, 0]
        centrada = serie - serie.mean(axis=1, keepdims=True)
        absoluta = np.abs(serie) - np.abs(serie).mean(axis=1, keepdims=True)

        def _por_lag(x: np.ndarray) -> np.ndarray:
            var = (x**2).mean(axis=1)
            var[var == 0] = 1.0
            return np.array(
                [(x[:, : pasado - k] * x[:, k:]).mean(axis=1).mean() / var.mean()
                 for k in range(1, lags + 1)]
            )

        return _por_lag(centrada), _por_lag(absoluta)

    acf_r, acf_abs_r = _acf(reales)
    acf_s, acf_abs_s = _acf(sinteticos)
    return {
        "error_acf_retornos": float(np.abs(acf_r - acf_s).mean()),
        "error_acf_absolutos": float(np.abs(acf_abs_r - acf_abs_s).mean()),
    }


def bateria_calidad(
    reales: np.ndarray,
    sinteticos: np.ndarray,
    pasado: int,
    n_canales: int,
    semilla: int = 42,
) -> dict[str, float]:
    """Aplica todas las métricas de calidad y devuelve una fila de resultados.

    El proyector PCA se ajusta una sola vez sobre los reales y se reutiliza, de
    modo que todos los generadores se juzgan en el mismo espacio.
    """
    pca = proyector(reales, semilla=semilla)

    resultado: dict[str, float] = {}
    resultado.update(puntuacion_discriminativa(reales, sinteticos, semilla, pca=pca))
    resultado.update(distancia_vecino_mas_cercano(reales, sinteticos, semilla=semilla, pca=pca))
    resultado["error_correlaciones"] = error_correlaciones(reales, sinteticos, pasado, n_canales)
    resultado.update(error_autocorrelacion(reales, sinteticos, pasado, n_canales))

    momentos = comparar_momentos(reales, sinteticos)
    resultado["dif_curtosis"] = float(momentos.loc["curtosis", "diferencia"])
    resultado["dif_asimetria"] = float(momentos.loc["asimetria", "diferencia"])
    return resultado


# ─────────────────────────────────────────────────────────────────────────────
# Tabla maestra de resultados
# ─────────────────────────────────────────────────────────────────────────────


def acumular(filas: list[dict], nombre: str) -> pd.DataFrame:
    """Guarda las filas del barrido en `results/metricas/<nombre>.csv`.

    Persistir cada resultado en cuanto se obtiene permite reanudar un barrido
    interrumpido y regenerar cualquier figura sin reentrenar. Con varios cientos
    de entrenamientos en CPU, esto no es una comodidad sino un requisito.
    """
    DIR_METRICAS.mkdir(parents=True, exist_ok=True)
    tabla = pd.DataFrame(filas)
    tabla.to_csv(DIR_METRICAS / f"{nombre}.csv", index=False)
    return tabla


def cargar_metricas(nombre: str) -> pd.DataFrame:
    """Lee una tabla de resultados ya calculada."""
    ruta = DIR_METRICAS / f"{nombre}.csv"
    if not ruta.exists():
        raise FileNotFoundError(
            f"No existe {ruta}. Ejecuta antes notebooks/12_barrido_entrenamiento.ipynb."
        )
    return pd.read_csv(ruta)
