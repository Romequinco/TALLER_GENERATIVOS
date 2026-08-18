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
        from hmmlearn.hmm import GaussianHMM

        self.columnas = list(features.columns)
        X = features.to_numpy()

        mejor, mejor_ll = None, -np.inf
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
        # `mode` no está disponible en rolling; se resuelve con bincount.
        agregado = ventana.apply(
            lambda v: float(np.bincount(v.astype(int)).argmax()), raw=True
        )

    return agregado.shift(-(horizonte - 1)).rename("regimen_futuro")


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
