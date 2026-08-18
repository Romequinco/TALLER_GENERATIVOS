"""Generador gaussiano multivariante, uno por régimen.

Es el generador con el que el profesor abre el taller
(`docs/material_clase/notebooks/Taller_Gaussian_solution.ipynb`): estima la
media y la covarianza del bloque conjunto y muestrea de una normal
multivariante. Aquí se añade el condicionamiento por régimen, ajustando un
modelo independiente por clase.

El problema real: covarianza en alta dimensión
----------------------------------------------
El bloque tiene del orden de 1.200 dimensiones y el tramo de entrenamiento
apenas unos miles de ventanas. Peor aún, esas ventanas se solapan 59 de cada 60
días, así que el número de observaciones **efectivamente independientes** es
mucho menor que el nominal. En la clase de crisis pueden quedar solo unos
cientos de ventanas, que corresponden a un puñado de episodios históricos.

Con $n < d$ la covarianza muestral es singular por construcción: tiene a lo
sumo rango $n-1$. Muestrear de ella produce muestras confinadas al subespacio
que generan los datos de entrenamiento, o directamente un error numérico. Por
eso el estimador por defecto no es la covarianza muestral sino el **shrinkage
de Ledoit-Wolf**, que la contrae hacia una diagonal escalada con un peso
óptimo estimado de los propios datos y garantiza un resultado bien
condicionado.

Este generador no puede reproducir colas gruesas ni asimetría: por
construcción, todo lo que genera es gaussiano. Esa limitación es informativa,
porque cuantifica cuánto de la mejora del downstream se explica solo por
reproducir medias y correlaciones.
"""

from __future__ import annotations

import numpy as np

from .base import GeneradorSintetico


class GaussianoCondicional(GeneradorSintetico):
    """Normal multivariante ajustada por separado a cada régimen.

    Parameters
    ----------
    n_regimenes
        Número de regímenes del problema.
    estimador
        ``ledoit_wolf`` (por defecto) aplica shrinkage automático hacia una
        diagonal; es el único seguro cuando ``n < d``.
        ``muestral`` usa la covarianza empírica con una regularización diagonal
        mínima; se incluye para poder reproducir el resultado del notebook de
        clase y enseñar en el informe por qué falla.
    regularizacion
        Valor que se suma a la diagonal en el estimador muestral, como fracción
        de la varianza media. Sin él, la factorización de Cholesky falla.
    """

    nombre = "gaussiano"
    etiqueta = "Gaussiano multivariante"
    tiene_curva_perdida = False

    def __init__(
        self,
        n_regimenes: int,
        semilla: int = 42,
        estimador: str = "ledoit_wolf",
        regularizacion: float = 1e-4,
    ) -> None:
        super().__init__(n_regimenes=n_regimenes, semilla=semilla)
        if estimador not in {"ledoit_wolf", "muestral"}:
            raise ValueError(f"Estimador desconocido: {estimador!r}")
        self.estimador = estimador
        self.regularizacion = regularizacion

        self.medias: dict[int, np.ndarray] = {}
        #: Factor de Cholesky de la covarianza. Se guarda factorizado porque
        #: muestrear es entonces una multiplicación matricial, mucho más rápido
        #: que llamar a `multivariate_normal` (que refactoriza en cada llamada).
        self.factores: dict[int, np.ndarray] = {}

    # ── Ajuste ──────────────────────────────────────────────────────────────

    def _fit(self, bloque: np.ndarray, y_reg: np.ndarray, **kwargs) -> None:
        self.estimador = kwargs.get("estimador", self.estimador)
        d = bloque.shape[1]

        for k in range(self.n_regimenes):
            muestras = bloque[y_reg == k].astype(np.float64)
            n = len(muestras)

            if n < 2:
                raise RuntimeError(
                    f"El régimen {k} tiene {n} muestras: no se puede estimar "
                    "una covarianza. Revisa el etiquetado o agrupa regímenes."
                )
            if n <= d:
                print(
                    f"Régimen {k}: {n} muestras para {d} dimensiones. La "
                    "covarianza muestral es singular; el shrinkage es "
                    "obligatorio aquí."
                )

            media = muestras.mean(axis=0)
            covarianza = self._estimar_covarianza(muestras)
            factor, saltos = self._cholesky_robusto(covarianza)

            self.medias[k] = media.astype(np.float32)
            self.factores[k] = factor.astype(np.float32)

            # Diagnóstico en lugar de curva de pérdida. El número de
            # condición dice cuán degenerada quedó la covarianza: valores
            # enormes indican que el muestreo estará confinado a un subespacio.
            autovalores = np.linalg.eigvalsh(covarianza)
            condicion = float(autovalores.max() / max(autovalores.min(), 1e-12))
            self.registrar(
                epoca=k,
                regimen=k,
                n_muestras=n,
                dimension=d,
                numero_condicion=condicion,
                autovalor_minimo=float(autovalores.min()),
                saltos_cholesky=saltos,
            )

    def _estimar_covarianza(self, muestras: np.ndarray) -> np.ndarray:
        """Covarianza regularizada, con el estimador configurado."""
        if self.estimador == "ledoit_wolf":
            from sklearn.covariance import LedoitWolf

            # `assume_centered=False` deja que el estimador reste la media.
            return LedoitWolf(assume_centered=False).fit(muestras).covariance_

        covarianza = np.cov(muestras, rowvar=False)
        # Regularización de Tikhonov: empuja los autovalores nulos por encima
        # de cero para que Cholesky no falle.
        traza_media = np.trace(covarianza) / len(covarianza)
        return covarianza + self.regularizacion * traza_media * np.eye(len(covarianza))

    @staticmethod
    def _cholesky_robusto(covarianza: np.ndarray) -> tuple[np.ndarray, int]:
        """Factoriza la covarianza, aumentando la diagonal si hace falta.

        Aunque el shrinkage garantiza definición positiva en teoría, el error
        de redondeo en dimensión alta puede dejar autovalores minúsculamente
        negativos. Se reintenta con una diagonal creciente y se devuelve cuántos
        intentos hicieron falta, como señal de lo mal condicionado que estaba.
        """
        salto = 0
        escala = np.trace(covarianza) / len(covarianza)
        while salto < 10:
            try:
                ajuste = (10.0**salto) * 1e-10 * escala * np.eye(len(covarianza))
                return np.linalg.cholesky(covarianza + ajuste), salto
            except np.linalg.LinAlgError:
                salto += 1
        raise np.linalg.LinAlgError(
            "La covarianza no es factorizable ni tras regularizarla. "
            "Reduce la dimensión del bloque o usa el estimador ledoit_wolf."
        )

    # ── Muestreo ────────────────────────────────────────────────────────────

    def _generate(self, n: int, regimen: int) -> np.ndarray:
        if regimen not in self.factores:
            raise RuntimeError(f"El régimen {regimen} no se ajustó en fit().")

        # x = mu + L z, con z normal estándar y L el factor de Cholesky.
        z = self.rng.standard_normal(size=(n, len(self.medias[regimen])))
        return self.medias[regimen] + z @ self.factores[regimen].T

    # ── Persistencia ────────────────────────────────────────────────────────

    def _estado(self) -> dict:
        return {
            "estimador": self.estimador,
            "regularizacion": self.regularizacion,
            "medias": self.medias,
            "factores": self.factores,
        }

    def _restaurar(self, estado: dict) -> None:
        self.estimador = estado["estimador"]
        self.regularizacion = estado["regularizacion"]
        self.medias = estado["medias"]
        self.factores = estado["factores"]
