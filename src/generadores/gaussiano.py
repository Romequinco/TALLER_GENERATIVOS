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

Las coordenadas de ``X`` no pueden reproducir colas gruesas ni asimetría: en el
espacio interno siguen siendo gaussianas. Esa limitación es informativa, porque
cuantifica cuánto de la mejora del downstream se explica solo por reproducir
medias y correlaciones.

La última componente, ``y_vol``, es estrictamente positiva y llega en unidades
naturales, a diferencia de las 1.200 componentes escaladas de ``X``. El ajuste
normaliza internamente solo ``log(y_vol)`` dentro de cada régimen. Al generar se
invierte esa transformación: el bloque público sigue conteniendo la volatilidad
en sus unidades originales y nunca es negativa. Las demás componentes conservan
exactamente el preprocesado original.
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
    transformar_vol
        Si es ``True`` (por defecto), modela la última columna como
        ``log(y_vol)`` y aplica la exponencial al generar. ``False`` desactiva
        únicamente esa transformación para comparaciones de diagnóstico.
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
        transformar_vol: bool = True,
    ) -> None:
        super().__init__(n_regimenes=n_regimenes, semilla=semilla)
        if estimador not in {"ledoit_wolf", "muestral"}:
            raise ValueError(f"Estimador desconocido: {estimador!r}")
        self.estimador = estimador
        self.regularizacion = regularizacion
        self.transformar_vol = transformar_vol

        self.medias: dict[int, np.ndarray] = {}
        self.medias_log_vol: dict[int, float] = {}
        self.desviaciones_log_vol: dict[int, float] = {}
        #: Factor de Cholesky de la covarianza. Se guarda factorizado porque
        #: muestrear es entonces una multiplicación matricial, mucho más rápido
        #: que llamar a `multivariate_normal` (que refactoriza en cada llamada).
        self.factores: dict[int, np.ndarray] = {}

    # ── Ajuste ──────────────────────────────────────────────────────────────

    def _fit(self, bloque: np.ndarray, y_reg: np.ndarray, **kwargs) -> None:
        self.estimador = kwargs.get("estimador", self.estimador)
        d = bloque.shape[1]

        for k in range(self.n_regimenes):
            muestras = bloque[y_reg == k].astype(np.float64).copy()
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

            # Se transforma únicamente y_vol, la última columna del contrato de
            # ventanas.empaquetar(). X conserva exactamente sus coordenadas de
            # entrada. Normalizar log(y_vol) evita tanto el soporte negativo como
            # la diferencia de escala que inflaba su varianza con Ledoit-Wolf.
            if self.transformar_vol:
                if np.any(muestras[:, -1] <= 0):
                    raise ValueError(
                        "y_vol debe ser estrictamente positiva para aplicar "
                        "log(y_vol). Revisa features.objetivos()."
                    )
                log_vol = np.log(muestras[:, -1])
                media_log_vol = float(log_vol.mean())
                desviacion_log_vol = float(log_vol.std())
                if desviacion_log_vol < np.finfo(np.float64).eps:
                    desviacion_log_vol = 1.0
                muestras[:, -1] = (log_vol - media_log_vol) / desviacion_log_vol
                self.medias_log_vol[k] = media_log_vol
                self.desviaciones_log_vol[k] = desviacion_log_vol

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

        # x = mu + L z en las coordenadas originales de X. Solo la última
        # columna vive temporalmente como log-volatilidad normalizada.
        z = self.rng.standard_normal(size=(n, len(self.medias[regimen])))
        muestras = self.medias[regimen] + z @ self.factores[regimen].T
        if self.transformar_vol:
            log_vol = (
                muestras[:, -1] * self.desviaciones_log_vol[regimen]
                + self.medias_log_vol[regimen]
            )
            muestras[:, -1] = np.exp(log_vol)
        return muestras

    # ── Persistencia ────────────────────────────────────────────────────────

    def _estado(self) -> dict:
        return {
            "estimador": self.estimador,
            "regularizacion": self.regularizacion,
            "transformar_vol": self.transformar_vol,
            "medias": self.medias,
            "medias_log_vol": self.medias_log_vol,
            "desviaciones_log_vol": self.desviaciones_log_vol,
            "factores": self.factores,
        }

    def _restaurar(self, estado: dict) -> None:
        self.estimador = estado["estimador"]
        self.regularizacion = estado["regularizacion"]
        # Los modelos antiguos no tenían transformación logarítmica: sus
        # factores ya están expresados enteramente en las unidades públicas.
        self.transformar_vol = estado.get("transformar_vol", False)
        self.medias = estado["medias"]
        self.medias_log_vol = estado.get("medias_log_vol", {})
        self.desviaciones_log_vol = estado.get("desviaciones_log_vol", {})
        self.factores = estado["factores"]
