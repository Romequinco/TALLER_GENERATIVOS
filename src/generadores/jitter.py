"""Generador simple: datos reales más ruido gaussiano.

El enunciado del taller exige "un cuarto modelo simple, por ejemplo que coja
datos originales y les añada ruido". Este es ese modelo, y conviene tomárselo
en serio: en la literatura de aumento de datos los baselines triviales ganan a
menudo, y si un generador neuronal no bate al jitter, ese resultado negativo es
tan publicable como el positivo.

Lo que hace y lo que no
-----------------------
El jitter **no aprende una distribución**: perturba muestras existentes. Genera
puntos en una bola alrededor de cada dato real, así que solo interpola
localmente y nunca produce una configuración de mercado que no se pareciera ya
a una observada. Es exactamente por eso que sirve de suelo: mide cuánto del
beneficio de los datos sintéticos viene de una regularización barata por ruido
y cuánto de haber modelado de verdad la distribución conjunta.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .base import GeneradorSintetico


class Jitter(GeneradorSintetico):
    """Muestrea un dato real del régimen pedido y le suma ruido gaussiano.

    Parameters
    ----------
    n_regimenes
        Número de regímenes del problema.
    sigma
        Amplitud del ruido, **en fracción de la desviación típica de cada
        dimensión**. Que sea relativa y no absoluta es deliberado: el bloque
        mezcla retornos diarios y z-scores, cuyas escalas difieren en órdenes de
        magnitud, y un sigma absoluto ahogaría unos canales y no tocaría otros.

    Notes
    -----
    Elección de `sigma`. Es el único hiperparámetro y tiene un compromiso claro:

    * `sigma` muy pequeño (< 0.05) produce copias casi exactas de los datos
      reales. El dataset crece pero no aporta información nueva, y el modelo
      downstream simplemente ve las mismas muestras repetidas.
    * `sigma` grande (> 0.5) destruye la estructura temporal y las
      correlaciones entre canales: las muestras dejan de parecer mercado.

    El valor por defecto (0.1, una décima de desviación típica) sitúa el ruido
    por debajo de la variabilidad natural de los datos sin ser despreciable.
    El notebook 04 barre varios valores y documenta el elegido.
    """

    nombre = "jitter"
    etiqueta = "Datos reales + ruido"
    tiene_curva_perdida = False

    def __init__(self, n_regimenes: int, semilla: int = 42, sigma: float = 0.1) -> None:
        super().__init__(n_regimenes=n_regimenes, semilla=semilla)
        self.sigma = sigma
        self.bloques_por_regimen: dict[int, np.ndarray] = {}
        self.escala_ruido: np.ndarray | None = None

    # ── Ajuste ──────────────────────────────────────────────────────────────

    def _fit(self, bloque: np.ndarray, y_reg: np.ndarray, **kwargs) -> None:
        """Memoriza los datos de train agrupados por régimen.

        "Ajustar" aquí es guardar las muestras y medir la escala de cada
        dimensión. No hay optimización de por medio.
        """
        self.sigma = float(kwargs.get("sigma", self.sigma))

        # La escala del ruido se calcula sobre TODO el train, no por régimen:
        # usar la desviación de la clase de crisis, que tiene pocas muestras,
        # daría una estimación inestable justo donde más importa.
        self.escala_ruido = bloque.std(axis=0)
        self.escala_ruido[self.escala_ruido == 0] = 1.0

        for k in range(self.n_regimenes):
            muestras = bloque[y_reg == k]
            if len(muestras) == 0:
                print(f"Aviso: el régimen {k} no tiene muestras en train.")
            self.bloques_por_regimen[k] = muestras

        # Diagnóstico en lugar de curva de pérdida: cómo de lejos queda una
        # muestra perturbada de su original, en unidades de desviación típica.
        # Es la magnitud que hay que reportar para justificar el sigma elegido.
        for k, muestras in self.bloques_por_regimen.items():
            self.registrar(
                epoca=k,
                regimen=k,
                n_muestras=len(muestras),
                sigma=self.sigma,
                desviacion_relativa=self.sigma * float(np.sqrt(bloque.shape[1])),
            )

    # ── Muestreo ────────────────────────────────────────────────────────────

    def _generate(self, n: int, regimen: int) -> np.ndarray:
        base = self.bloques_por_regimen.get(regimen)
        if base is None or len(base) == 0:
            raise RuntimeError(
                f"No hay muestras reales del régimen {regimen} sobre las que "
                "generar. El jitter no puede inventar una clase que no vio."
            )

        # Muestreo con reemplazo: se piden más sintéticos que reales
        # disponibles, sobre todo en la clase de crisis.
        indices = self.rng.integers(0, len(base), size=n)
        ruido = self.rng.normal(0.0, 1.0, size=(n, base.shape[1]))
        return base[indices] + self.sigma * self.escala_ruido * ruido

    # ── Persistencia ────────────────────────────────────────────────────────

    def _estado(self) -> dict:
        return {
            "sigma": self.sigma,
            "escala_ruido": self.escala_ruido,
            "bloques_por_regimen": self.bloques_por_regimen,
        }

    def _restaurar(self, estado: dict) -> None:
        self.sigma = estado["sigma"]
        self.escala_ruido = estado["escala_ruido"]
        self.bloques_por_regimen = estado["bloques_por_regimen"]
