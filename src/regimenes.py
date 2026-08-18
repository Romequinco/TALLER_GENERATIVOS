"""Etiquetado de regímenes de mercado.

No existe una etiqueta verdadera de "régimen": es una variable latente que hay
que estimar. Aquí se usa un HMM gaussiano sobre un conjunto reducido de
indicadores de estrés, y los estados resultantes se ordenan por volatilidad
creciente para que la numeración sea estable entre ejecuciones.

Dos precauciones que condicionan todo el diseño:

1. **El HMM se ajusta solo con el tramo de entrenamiento.** Ajustarlo sobre
   todo el histórico filtraría al modelo downstream información del periodo de
   test a través de las etiquetas.

2. **El régimen que se predice es el del periodo futuro**, no el actual. La
   etiqueta de una muestra es el régimen dominante en los `horizonte` días
   siguientes al final de su ventana X. Esa etiqueta sí mira al futuro, que es
   legítimo: es el objetivo, no una entrada.

El criterio de canonicalización se hereda del TFM de detección de regímenes,
donde se comprobó que ordenar por retorno medio invierte los estados de forma
errática cuando dos regímenes tienen volatilidad parecida.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import cargar_catalogo

# Dos estados se consideran de volatilidad equivalente si sus medias distan
# menos de esta fracción de la volatilidad media global. Dentro de una misma
# banda, el desempate lo hace el retorno medio (menor retorno = peor régimen).
FRACCION_VOL_CERCANA = 0.15

#: Episodios de crisis contra los que se valida el etiquetado. No salen del
#: modelo y no se ajustan a él: son el patrón externo. Las fechas son de dominio
#: público —de la quiebra de Lehman al suelo de marzo de 2009, el desplome del
#: COVID y el mercado bajista de tipos de 2022— y viven aquí, y no en el
#: catálogo, precisamente porque no son un parámetro: moverlas para que el
#: control pase sería el fraude que el control existe para impedir.
EPISODIOS_REFERENCIA: dict[str, tuple[str, str]] = {
    "GFC 2008": ("2008-09-01", "2009-03-31"),
    "COVID 2020": ("2020-02-20", "2020-04-30"),
    "Inflación 2022": ("2022-01-01", "2022-10-31"),
}


@dataclass
class EtiquetadorRegimenes:
    """Ajusta un HMM gaussiano y produce etiquetas de régimen canonizadas.

    Attributes
    ----------
    n_estados
        Número de regímenes. Con 3 se obtiene calma / transición / crisis, que
        separa mejor la cola que los 2 estados habituales.
    orden
        Permutación que lleva los estados crudos del HMM al orden económico
        ``0 = calma ... n-1 = crisis``. Se calcula al ajustar.
    """

    n_estados: int
    covarianza: str = "full"
    n_iteraciones: int = 1000
    tolerancia: float = 1e-4
    semillas: tuple[int, ...] = (42,)

    modelo: object | None = None
    orden: np.ndarray | None = None
    columnas: list[str] | None = None

    @classmethod
    def desde_catalogo(cls) -> "EtiquetadorRegimenes":
        """Construye el etiquetador con los parámetros de `data/catalog.yaml`."""
        cfg = cargar_catalogo()["regimenes"]
        return cls(
            n_estados=cfg["n_estados"],
            covarianza=cfg["covarianza"],
            n_iteraciones=cfg["n_iteraciones"],
            tolerancia=cfg["tolerancia"],
            semillas=tuple(cfg["semillas"]),
        )

    # ── Ajuste ──────────────────────────────────────────────────────────────

    def fit(self, features: pd.DataFrame) -> "EtiquetadorRegimenes":
        """Ajusta el HMM. `features` debe contener SOLO el tramo de train.

        Se prueban varias semillas porque el algoritmo de Baum-Welch converge a
        óptimos locales; se conserva el ajuste de mayor log-verosimilitud.
        """
        import logging

        from hmmlearn.hmm import GaussianHMM

        self.columnas = list(features.columns)
        X = features.to_numpy()

        # hmmlearn avisa de "Model is not converging" cuando la verosimilitud
        # retrocede unas milésimas en el último paso del EM. Es ruido numérico
        # —el ajuste ganador converge— pero imprime varias líneas por semilla que
        # parecen un error grave dentro del notebook. Se silencia aquí y no en el
        # cuaderno, para que el ruido no dependa de quién llame.
        registro = logging.getLogger("hmmlearn")
        nivel = registro.level
        registro.setLevel(logging.ERROR)

        mejor, mejor_ll = None, -np.inf
        try:
            for s in self.semillas:
                modelo = GaussianHMM(
                    n_components=self.n_estados,
                    covariance_type=self.covarianza,
                    n_iter=self.n_iteraciones,
                    tol=self.tolerancia,
                    random_state=s,
                )
                modelo.fit(X)
                ll = modelo.score(X)
                if ll > mejor_ll:
                    mejor, mejor_ll = modelo, ll
        finally:
            registro.setLevel(nivel)

        self.modelo = mejor
        self.orden = self._orden_economico(mejor, features)
        return self

    def _orden_economico(self, modelo, features: pd.DataFrame) -> np.ndarray:
        """Devuelve la permutación que ordena los estados de calma a crisis.

        Criterio primario: volatilidad media del estado. Criterio de desempate,
        solo entre estados de volatilidad parecida: retorno medio. Sin el
        desempate acotado, el ruido en la media de retornos puede intercambiar
        calma y crisis entre ejecuciones.
        """
        estados = modelo.predict(features.to_numpy())

        col_vol = self._columna(["vol_realizada", "vix_nivel", "vol"])
        col_ret = self._columna(["ret_sp500", "ret", "retorno"])

        vol_por_estado = np.array(
            [features[col_vol].to_numpy()[estados == k].mean() for k in range(self.n_estados)]
        )
        ret_por_estado = np.array(
            [features[col_ret].to_numpy()[estados == k].mean() for k in range(self.n_estados)]
        )

        umbral = FRACCION_VOL_CERCANA * float(np.abs(vol_por_estado).mean())

        # Clave de orden: la volatilidad se redondea a bandas de ancho `umbral`
        # para que estados de volatilidad equivalente caigan en la misma banda y
        # se desempaten por retorno (mayor retorno = mejor régimen = va antes).
        banda = np.round(vol_por_estado / umbral) if umbral > 0 else vol_por_estado
        return np.lexsort((-ret_por_estado, banda))

    def _columna(self, candidatos: list[str]) -> str:
        """Busca la primera columna cuyo nombre empiece por alguno de los candidatos."""
        for pref in candidatos:
            for col in self.columnas or []:
                if col.startswith(pref):
                    return col
        raise KeyError(
            f"Ninguna columna de {self.columnas} coincide con {candidatos}. "
            "Revisa `regimenes.features_etiquetado` en data/catalog.yaml."
        )

    # ── Inferencia ──────────────────────────────────────────────────────────

    def predict(self, features: pd.DataFrame) -> pd.Series:
        """Régimen canonizado día a día (0 = calma ... n-1 = crisis)."""
        self._exigir_ajustado()
        crudos = self.modelo.predict(features[self.columnas].to_numpy())
        # `orden[i]` es el estado crudo que ocupa la posición i; se invierte
        # para pasar de estado crudo a posición canónica.
        inverso = np.empty_like(self.orden)
        inverso[self.orden] = np.arange(self.n_estados)
        return pd.Series(inverso[crudos], index=features.index, name="regimen")

    def predict_proba(self, features: pd.DataFrame) -> pd.DataFrame:
        """Probabilidad filtrada de cada régimen, con las columnas canonizadas."""
        self._exigir_ajustado()
        proba = self.modelo.predict_proba(features[self.columnas].to_numpy())
        return pd.DataFrame(
            proba[:, self.orden],
            index=features.index,
            columns=[f"p_regimen_{k}" for k in range(self.n_estados)],
        )

    @property
    def estado_crisis(self) -> int:
        """Índice del régimen más severo."""
        return self.n_estados - 1

    def _exigir_ajustado(self) -> None:
        if self.modelo is None:
            raise RuntimeError("El etiquetador no está ajustado: llama antes a fit().")

    # ── Persistencia ────────────────────────────────────────────────────────

    def guardar(self, ruta) -> None:
        """Serializa el etiquetador para que los notebooks posteriores lo reusen."""
        import pickle

        with open(ruta, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def cargar(ruta) -> "EtiquetadorRegimenes":
        import pickle

        with open(ruta, "rb") as f:
            return pickle.load(f)


# ─────────────────────────────────────────────────────────────────────────────
# Agregación al horizonte de predicción
# ─────────────────────────────────────────────────────────────────────────────


def regimen_dominante(
    regimen_diario: pd.Series, horizonte: int, metodo: str = "modal"
) -> pd.Series:
    """Resume en una etiqueta el régimen de los `horizonte` días siguientes.

    Parameters
    ----------
    regimen_diario
        Serie de regímenes día a día devuelta por `EtiquetadorRegimenes.predict`.
    horizonte
        Número de días futuros a resumir.
    metodo
        ``modal``  régimen más frecuente en la ventana futura (por defecto).
        ``maximo`` régimen más severo alcanzado; etiqueta como crisis cualquier
                   ventana que la toque, lo que sube mucho la clase minoritaria.

    Returns
    -------
    Serie alineada con `regimen_diario`: el valor en `t` describe el periodo
    ``(t, t + horizonte]``. Las últimas `horizonte` posiciones quedan a NaN.
    """
    if metodo not in {"modal", "maximo"}:
        raise ValueError(f"Método de agregación desconocido: {metodo!r}")

    futuro = regimen_diario.shift(-1)
    ventana = futuro.rolling(horizonte, min_periods=horizonte)

    if metodo == "maximo":
        agregado = ventana.max()
    else:
        # `mode` no está disponible en rolling; se resuelve con bincount. El
        # empate se rompe a favor del estado MÁS severo, como exige la
        # metodología: `argmax` a secas devolvería el índice más bajo y
        # etiquetaría como calma un mes mitad calmado mitad en crisis. Afecta a
        # 9 de 5.647 ventanas, pero la regla tiene que decir lo que hace.
        agregado = ventana.apply(
            lambda v: float(
                len(c := np.bincount(v.astype(int))) - 1 - int(np.argmax(c[::-1]))
            ),
            raw=True,
        )

    return agregado.shift(-(horizonte - 1)).rename("regimen_futuro")


def control_bloqueante(
    regimen_diario: pd.Series,
    regimen_futuro: pd.Series,
    n_estados: int,
    episodios: dict[str, tuple[str, str]] | None = None,
    banda: tuple[float, float] = (3.0, 20.0),
    cobertura_minima: float = 50.0,
) -> pd.DataFrame:
    """Veredicto de aceptación del etiquetado, fila a fila.

    Es la puerta que hay que cruzar antes de entrenar cualquier generador. Un
    HMM converge siempre, así que la convergencia no es evidencia de nada: lo
    que se comprueba es que los estados coinciden con crisis que de verdad
    ocurrieron y que la clase extrema tiene un tamaño con el que se pueda
    trabajar.

    Devuelve una tabla y no un booleano, a propósito: cuando falla hay que saber
    **qué** falló y por cuánto, porque de eso depende si se toca el número de
    estados o las features de etiquetado.

    Parameters
    ----------
    regimen_diario
        Régimen canonizado día a día, con índice de fechas.
    regimen_futuro
        Etiqueta agregada al horizonte. Los NaN del final se descartan.
    n_estados
        Número de regímenes; el de crisis es ``n_estados - 1``.
    episodios
        ``{nombre: (desde, hasta)}``. ``None`` usa `EPISODIOS_REFERENCIA`.
    banda
        Porcentaje mínimo y máximo admisible para la clase de crisis. Por debajo
        del 3 % no hay con qué entrenar ni evaluar; por encima del 20 % el estado
        extremo ha dejado de separar crisis de corrección y el taller se queda
        sin motivo.
    cobertura_minima
        Porcentaje mínimo de días en crisis dentro de cada episodio. 50 y no 90
        porque con tres estados parte de un episodio cae legítimamente en
        transición: 2022 fue un mercado bajista lento, no un desplome.

    Returns
    -------
    pandas.DataFrame
        Una fila por control, con ``medido``, ``criterio`` y ``ok``. El notebook
        debe cortar con ``assert tabla["ok"].all()``.

    Notes
    -----
    Que el control pase no demuestra que el etiquetado sea bueno: solo descarta
    las dos formas de estar mal que se pueden medir sin disponer de una etiqueta
    verdadera. En particular no detecta que el estado intermedio quede vacío en
    alguna partición, que es un fallo real y hay que mirarlo en `distribucion`.
    """
    episodios = EPISODIOS_REFERENCIA if episodios is None else episodios
    crisis = n_estados - 1
    filas = []

    for nombre, (desde, hasta) in episodios.items():
        tramo = regimen_diario.loc[desde:hasta]
        medido = 100.0 * float((tramo == crisis).mean()) if len(tramo) else float("nan")
        filas.append(
            (
                f"cubre {nombre}",
                round(medido, 1),
                f"{cobertura_minima:.0f} % o mas",
                bool(medido >= cobertura_minima),
            )
        )

    etiquetas = pd.Series(regimen_futuro).dropna()
    peso = 100.0 * float((etiquetas == crisis).mean()) if len(etiquetas) else float("nan")
    filas.append(
        (
            "peso de la clase de crisis",
            round(peso, 1),
            f"entre {banda[0]:.0f} % y {banda[1]:.0f} %",
            bool(banda[0] <= peso <= banda[1]),
        )
    )

    return pd.DataFrame(
        filas, columns=["control", "medido", "criterio", "ok"]
    ).set_index("control")


def diagnostico_features(
    features: pd.DataFrame,
    deriva_maxima: float = 0.5,
    saturacion_maxima: float = 10.0,
) -> pd.DataFrame:
    """Mide las dos patologías que rompen las emisiones gaussianas de un HMM.

    Un `GaussianHMM` supone que dentro de cada estado las observaciones son
    normales y de parámetros constantes. Cuando una feature **deriva**, el modelo
    gasta estados en separar épocas en vez de regímenes; cuando **satura** contra
    un tope, la masa puntual del tope no cabe en ninguna normal y el ajuste se
    vuelve multimodal. Las dos se manifiestan igual en el resultado —un estado de
    crisis demasiado grande— así que conviene medirlas antes y no diagnosticarlas
    después.

    Parameters
    ----------
    features
        Candidatas a `regimenes.features_etiquetado`, una por columna.
    deriva_maxima
        Umbral de la deriva, en desviaciones típicas de la propia serie.
    saturacion_maxima
        Umbral de la saturación, en porcentaje de sesiones.

    Returns
    -------
    pandas.DataFrame
        Una fila por feature con ``deriva`` (salto de la media entre la primera y
        la segunda mitad de la muestra, en sigmas), ``saturacion`` (porcentaje de
        sesiones en el 5 % superior del recorrido) y ``apta``.

    Notes
    -----
    Los dos umbrales son convenciones, no teoremas: separan con holgura los
    canales de este panel y no se han calibrado contra ningún otro. Y la prueba
    es necesaria, no suficiente: una feature puede pasar las dos y aun así no
    aportar información de régimen, cosa que solo dirá `control_bloqueante`.
    """
    filas = {}
    for nombre, serie in features.items():
        serie = serie.dropna()
        mitad = len(serie) // 2
        deriva = float(
            abs(serie.iloc[mitad:].mean() - serie.iloc[:mitad].mean()) / serie.std()
        )
        recorrido = serie.max() - serie.min()
        saturacion = 100.0 * float((serie >= serie.max() - 0.05 * recorrido).mean())
        filas[nombre] = {
            "deriva": round(deriva, 2),
            "saturacion": round(saturacion, 1),
            "apta": bool(deriva <= deriva_maxima and saturacion <= saturacion_maxima),
        }
    return pd.DataFrame(filas).T


def distribucion(etiquetas: pd.Series | np.ndarray, n_estados: int) -> pd.DataFrame:
    """Reparto de muestras por régimen, en absoluto y en porcentaje.

    Es la tabla que justifica todo el taller: cuantifica cuán minoritaria es la
    clase de crisis y, por tanto, cuánto margen hay para que los datos
    sintéticos aporten algo.
    """
    valores = pd.Series(np.asarray(etiquetas).ravel()).dropna().astype(int)
    conteo = valores.value_counts().reindex(range(n_estados), fill_value=0).sort_index()
    return pd.DataFrame(
        {
            "muestras": conteo,
            "porcentaje": (100 * conteo / conteo.sum()).round(2),
        }
    ).rename_axis("regimen")
