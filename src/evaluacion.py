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
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import DIR_METRICAS

# ─────────────────────────────────────────────────────────────────────────────
# Métricas del modelo downstream
# ─────────────────────────────────────────────────────────────────────────────


def metricas_regimen(
    y_real: np.ndarray, y_pred: np.ndarray, n_clases: int = 3
) -> dict[str, float]:
    """Métricas de clasificación, con foco en la clase minoritaria.

    Se reporta `recall_crisis` aparte porque es la magnitud que decide si el
    trabajo tiene sentido: si añadir sintéticos de crisis no sube el recall de
    crisis, la hipótesis del taller no se sostiene, por bien que se comporte el
    resto de métricas.
    """
    from sklearn.metrics import (
        balanced_accuracy_score,
        f1_score,
        precision_score,
        recall_score,
    )

    crisis = n_clases - 1
    etiquetas = list(range(n_clases))

    return {
        "f1_macro": float(f1_score(y_real, y_pred, average="macro", labels=etiquetas, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_real, y_pred)),
        "accuracy": float((y_real == y_pred).mean()),
        "recall_crisis": float(
            recall_score(y_real, y_pred, labels=[crisis], average="macro", zero_division=0)
        ),
        "precision_crisis": float(
            precision_score(y_real, y_pred, labels=[crisis], average="macro", zero_division=0)
        ),
        "f1_crisis": float(
            f1_score(y_real, y_pred, labels=[crisis], average="macro", zero_division=0)
        ),
    }


def metricas_volatilidad(y_real: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Métricas de regresión sobre la volatilidad realizada futura.

    Se incluye QLIKE además de MAE y RMSE: es la pérdida estándar en previsión
    de volatilidad porque penaliza de forma asimétrica, castigando más
    infraestimar el riesgo que sobreestimarlo.
    """
    y_real = np.asarray(y_real, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()

    error = y_pred - y_real
    metricas = {
        "mae": float(np.abs(error).mean()),
        "rmse": float(np.sqrt((error**2).mean())),
        "sesgo": float(error.mean()),
    }

    # QLIKE exige positividad en ambos argumentos; la red puede predecir
    # negativos, así que se recortan y se registra cuántos hubo.
    piso = 1e-6
    n_negativos = int((y_pred <= 0).sum())
    real_pos = np.maximum(y_real, piso)
    pred_pos = np.maximum(y_pred, piso)
    metricas["qlike"] = float(np.mean(real_pos / pred_pos - np.log(real_pos / pred_pos) - 1.0))
    metricas["predicciones_negativas"] = n_negativos

    # R2 frente al predictor trivial (la media del test).
    varianza = ((y_real - y_real.mean()) ** 2).mean()
    metricas["r2"] = float(1.0 - (error**2).mean() / varianza) if varianza > 0 else np.nan
    return metricas


def evaluar(modelo, X_test: np.ndarray, y_test: np.ndarray, tarea: str) -> dict[str, float]:
    """Evalúa un modelo entrenado sobre el conjunto de test real."""
    predicho = modelo.predict(X_test, verbose=0)
    if tarea == "regimen":
        return metricas_regimen(y_test, predicho.argmax(axis=1), n_clases=predicho.shape[1])
    return metricas_volatilidad(y_test, predicho)


def matriz_confusion(y_real: np.ndarray, y_pred: np.ndarray, n_clases: int = 3) -> pd.DataFrame:
    """Matriz de confusión con nombres legibles de régimen."""
    from sklearn.metrics import confusion_matrix

    nombres = nombres_regimenes(n_clases)
    matriz = confusion_matrix(y_real, y_pred, labels=list(range(n_clases)))
    return pd.DataFrame(matriz, index=nombres, columns=nombres).rename_axis(
        index="real", columns="predicho"
    )


def nombres_regimenes(n_clases: int = 3) -> list[str]:
    """Etiquetas legibles de los regímenes, para figuras y tablas."""
    if n_clases == 2:
        return ["calma", "crisis"]
    if n_clases == 3:
        return ["calma", "transición", "crisis"]
    return [f"régimen {k}" for k in range(n_clases - 1)] + ["crisis"]


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
