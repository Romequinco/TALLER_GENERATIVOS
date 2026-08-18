"""RBIG condicional: gaussianización iterativa basada en rotaciones.

RBIG (*Rotation-Based Iterative Gaussianization*, Laparra et al. 2011) es el
flujo normalizante más barato del taller: no se entrena por gradiente, no
necesita GPU y cada capa se resuelve en forma cerrada. La capa repite dos
operaciones triviales por separado pero potentes al encadenarlas:

1. **Gaussianización marginal.** Cada dimensión se pasa por su CDF empírica y
   luego por la función probit. El resultado tiene marginales normales estándar,
   pero conserva intacta la dependencia entre dimensiones.
2. **Rotación.** Una PCA sobre el resultado redistribuye esa dependencia entre
   todas las dimensiones, de modo que la siguiente gaussianización marginal
   pueda atacar una parte distinta de la estructura.

Iterando, la multi-información (la dependencia estadística total) se destruye
capa a capa hasta que los datos son gaussianos e independientes. Como ambas
operaciones son invertibles, generar es muestrear ``z ~ N(0, I)`` y desandar el
camino.

Decisión de implementación
--------------------------
El algoritmo se implementa **a mano** en vez de usar la librería `rbig` de
IPL-UV que aparece en el notebook del profesor. Esa librería se instala desde
GitHub, no se mantiene desde hace años y su instalación no es fiable en Python
3.13. El algoritmo es corto —numpy, scipy y la PCA de scikit-learn bastan—, así
que implementarlo deja el repositorio autocontenido, reproducible y sin
dependencias frágiles. El único coste es escribir explícitamente la inversa,
que es justo lo que hace falta para generar.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.decomposition import PCA

from .base import GeneradorSintetico

#: Nº máximo de puntos con los que se tabula la CDF empírica de una dimensión.
#: Por encima de este valor la interpolación no gana precisión y sí memoria: el
#: modelo guarda una tabla ``(puntos, dimensiones)`` por capa.
PUNTOS_CDF_MAXIMOS = 512


# ─────────────────────────────────────────────────────────────────────────────
# Gaussianización marginal invertible
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class _GaussianizacionMarginal:
    """CDF empírica tabulada de cada dimensión, con su inversa.

    La transformación directa es ``z = probit(F(x))`` y la inversa
    ``x = F⁻¹(Φ(z))``. Ambas se resuelven por interpolación lineal sobre la
    misma tabla de cuantiles, lo que garantiza que una deshace exactamente a la
    otra dentro del rango observado.

    Attributes
    ----------
    probabilidades
        Rejilla de probabilidades ``(m,)``, común a todas las dimensiones. Nunca
        llega a 0 ni a 1: `probit` devolvería infinitos.
    soporte
        Estadísticos de orden ``(m, k)``, uno por dimensión y probabilidad.
    """

    probabilidades: np.ndarray
    soporte: np.ndarray

    @classmethod
    def ajustar(cls, X: np.ndarray) -> "_GaussianizacionMarginal":
        """Tabula la CDF empírica de cada columna de `X`.

        La tabla se construye sobre los estadísticos de orden y **siempre
        incluye el mínimo y el máximo** de cada dimensión. Esto es lo que hace
        que el flujo sea exactamente invertible sobre los datos de train: si el
        soporte empezara en un cuantil interior, los puntos más extremos caerían
        fuera de la tabla, la transformación directa los saturaría y el error se
        acumularía capa a capa.

        Con muchas muestras la tabla se submuestrea a `PUNTOS_CDF_MAXIMOS`
        puntos. Eso no rompe la invertibilidad —directa e inversa interpolan
        sobre la misma tabla, así que se deshacen exactamente— solo aproxima la
        CDF empírica por tramos.
        """
        n_muestras, _ = X.shape
        n_puntos = int(min(max(n_muestras, 2), PUNTOS_CDF_MAXIMOS))

        posiciones = np.unique(
            np.round(np.linspace(0, n_muestras - 1, n_puntos)).astype(int)
        )
        # Posiciones de trazado de Hazen: el i-ésimo dato de n acumula
        # (i + 0.5) / n de probabilidad. Deja los extremos a distancia 0.5/n de
        # 0 y de 1, que es lo más lejos que la muestra permite afirmar.
        probabilidades = (posiciones + 0.5) / n_muestras
        soporte = np.sort(X, axis=0)[posiciones]

        return cls(probabilidades, _forzar_creciente(soporte))

    def aplicar(self, X: np.ndarray) -> np.ndarray:
        """Lleva cada columna a normal estándar."""
        probabilidades = np.empty_like(X)
        for j in range(X.shape[1]):
            probabilidades[:, j] = np.interp(
                X[:, j], self.soporte[:, j], self.probabilidades
            )
        return norm.ppf(probabilidades)

    def invertir(self, Z: np.ndarray) -> np.ndarray:
        """Devuelve cada columna al espacio original.

        `np.interp` satura en los extremos de la tabla, y esa saturación es
        deliberada: una muestra ``z`` en la cola lejana se mapea al máximo (o
        mínimo) valor observado en train en vez de extrapolar sobre una CDF que
        ahí no tiene ningún dato. Sin ella, un ``z = 6`` produciría valores
        arbitrariamente grandes.

        El acotado es **por coordenada del espacio reducido**, no sobre la
        muestra final: la proyección de vuelta al espacio original es una
        combinación lineal de esas coordenadas, y puede rebasar el rango
        observado en una dimensión concreta. Acota la cola, no la elimina.
        """
        probabilidades = norm.cdf(Z)
        X = np.empty_like(probabilidades)
        for j in range(Z.shape[1]):
            X[:, j] = np.interp(
                probabilidades[:, j], self.probabilidades, self.soporte[:, j]
            )
        return X


def _forzar_creciente(soporte: np.ndarray) -> np.ndarray:
    """Hace estrictamente creciente cada columna de la tabla de cuantiles.

    Con muestras pequeñas o valores repetidos aparecen cuantiles empatados, y
    un tramo plano rompe la invertibilidad: `np.interp` ya no sabe qué valor
    devolver. Se acumula el máximo y se añade una rampa despreciable frente a
    la escala del dato, suficiente para desempatar sin desplazar la
    distribución.
    """
    soporte = np.maximum.accumulate(np.asarray(soporte, dtype=np.float64), axis=0)
    rango = np.ptp(soporte, axis=0)
    paso = np.maximum(rango, 1.0) * 1e-9
    rampa = np.arange(soporte.shape[0], dtype=np.float64)[:, None] * paso[None, :]
    return soporte + rampa


# ─────────────────────────────────────────────────────────────────────────────
# Diagnóstico: reducción de multi-información por capa
# ─────────────────────────────────────────────────────────────────────────────


#: Entropía diferencial de una normal estándar, ``½·log(2πe)``.
ENTROPIA_NORMAL = 0.5 * float(np.log(2.0 * np.pi * np.e))


def _entropias_marginales(X: np.ndarray) -> np.ndarray:
    """Entropía diferencial de cada columna por espaciados muestrales (Vasicek).

    Se descarta el estimador por histograma, que es el habitual en las
    implementaciones de referencia: su sesgo depende de la *forma* de la
    distribución, y aquí se comparan distribuciones de formas distintas, así que
    el sesgo no se cancela y contamina el diagnóstico. El estimador de
    espaciados tiene un sesgo que depende solo de ``n`` y del solape ``m``, que
    son idénticos en las dos distribuciones comparadas.
    """
    n_muestras, _ = X.shape
    m = max(1, int(round(np.sqrt(n_muestras))))

    ordenado = np.sort(X, axis=0)
    posicion = np.arange(n_muestras)
    superior = ordenado[np.minimum(posicion + m, n_muestras - 1)]
    inferior = ordenado[np.maximum(posicion - m, 0)]

    espaciados = np.maximum(superior - inferior, np.finfo(np.float64).tiny)
    return np.log(espaciados * n_muestras / (2 * m)).mean(axis=0)


def _negentropias(X: np.ndarray) -> np.ndarray:
    """Distancia de cada marginal a la gaussiana de su misma varianza, en nats."""
    varianzas = np.maximum(X.var(axis=0), np.finfo(np.float64).tiny)
    entropia_gaussiana = ENTROPIA_NORMAL + 0.5 * np.log(varianzas)
    # La negentropía es no negativa por definición: los valores negativos que
    # devuelve el estimador son ruido, y recortarlos evita contabilizar
    # información destruida que no existe.
    return np.maximum(entropia_gaussiana - _entropias_marginales(X), 0.0)


def _reduccion_informacion(rotado: np.ndarray) -> float:
    """Multi-información destruida por una capa, en nats.

    La multi-información es ``I(x) = Σ_d H(x_d) − H(x)``. Dentro de una capa la
    gaussianización marginal no la altera (actúa dimensión a dimensión) y la
    rotación conserva la entropía conjunta (es ortogonal, ``|det| = 1``), así
    que los términos ``H(x)`` se cancelan y queda

        ΔI = k·h_N − Σ_d H(x_d)

    con ``h_N`` la entropía de una normal estándar, que es exactamente lo que
    tienen las marginales antes de rotar. Descomponiendo cada ``H(x_d)`` en su
    parte de escala y su parte de forma, la expresión se separa en dos términos
    no negativos y calculables por separado:

        ΔI = −½·Σ_d log σ_d²  +  Σ_d J(x_d)

    El primero es la **dispersión de varianzas** que introduce la rotación (los
    autovalores de la PCA, normalizados a media 1); se calcula exacto, sin
    estimador. El segundo es la **no-gaussianidad marginal** residual, y es lo
    único que necesita estimarse. Que la parte grande sea analítica es lo que
    hace utilizable el diagnóstico con pocas muestras.

    Al ser suma de dos términos no negativos, la curva acumulada es monótona
    creciente, como la de la implementación de referencia del profesor.
    """
    varianzas = rotado.var(axis=0)
    varianzas = varianzas / max(float(varianzas.mean()), np.finfo(np.float64).tiny)

    dispersion = -0.5 * float(np.log(np.maximum(varianzas, 1e-300)).sum())
    return dispersion + float(_negentropias(rotado).sum())


# ─────────────────────────────────────────────────────────────────────────────
# Estructuras del modelo ajustado
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class _Capa:
    """Una capa de RBIG: gaussianizar marginales y rotar.

    `rotacion` es ``None`` en la capa de cierre, que solo gaussianiza. Sin ella
    la cima del flujo sería la salida de una PCA, cuyas componentes tienen
    varianzas distintas de 1; muestrear ``N(0, I)`` ahí introduciría un sesgo de
    escala en las muestras generadas.
    """

    marginal: _GaussianizacionMarginal
    rotacion: PCA | None


@dataclass
class _ModeloRegimen:
    """RBIG completo de un régimen, más su reducción de dimensión."""

    reductor: PCA
    capas: list[_Capa]
    sigma_residual: np.ndarray | None
    n_componentes: int
    n_muestras: int
    varianza_explicada: float
    capas_usadas: int
    reduccion_total: float


# ─────────────────────────────────────────────────────────────────────────────
# Generador
# ─────────────────────────────────────────────────────────────────────────────


class RBIGCondicional(GeneradorSintetico):
    """Flujo normalizante por gaussianización iterativa, uno por régimen.

    Condicionamiento
    ----------------
    RBIG no admite condición: no hay ningún sitio donde inyectar la etiqueta.
    La estrategia es ajustar un **modelo independiente por régimen** y elegir el
    que toque al generar. Es la opción honesta —cada régimen conserva su propia
    estructura de dependencia— pero traslada todo el peso al tamaño muestral: el
    régimen de crisis puede tener unos pocos cientos de ventanas, y ahí la CDF
    empírica se estima con muy poca información. El ajuste avisa por consola
    cuando esto ocurre.

    Reducción de dimensión
    ----------------------
    El bloque tiene ``d ≈ 1201`` dimensiones y el régimen minoritario puede
    aportar solo unos cientos de muestras. Una PCA de rotación en ese espacio es
    degenerada: con ``n < d`` la matriz de covarianza no tiene rango completo y
    las direcciones sobrantes son ruido puro. Por eso se aplica primero una PCA
    de reducción a `n_componentes` (por defecto ``min(n//2, 128)``), se ajusta
    RBIG en el espacio reducido y al generar se proyecta de vuelta.

    El precio es explícito: las muestras sintéticas viven en un subespacio afín
    de dimensión ``k``, y la varianza descartada por la proyección se pierde. El
    ajuste reporta la varianza explicada por régimen y, si `ruido_residual` está
    activo, repone la varianza descartada como ruido gaussiano independiente por
    dimensión, para que la covarianza sintética no sea exactamente singular.

    Diagnóstico de convergencia
    ---------------------------
    No hay curva de pérdida: nada se optimiza por gradiente. El equivalente en
    RBIG es la **reducción de multi-información por capa**, que mide cuánta
    dependencia estadística ha destruido cada capa. Se registra en `historial`
    como ``reduccion_info`` (por capa) y ``reduccion_acumulada``, y se grafica
    junto a las curvas de pérdida de los demás generadores.

    La lectura es la misma que la de una loss, con el signo invertido: la curva
    acumulada crece rápido en las primeras capas y se aplana cuando ya no queda
    dependencia que extraer. Ese aplanamiento es el criterio de parada; seguir
    añadiendo capas a partir de ahí solo memoriza los cuantiles de la muestra de
    train.

    Conviene leer la curva junto a `resumen_regimenes`: con pocas muestras por
    dimensión, la reducción se estanca enseguida en el suelo de ruido del
    estimador, y eso no significa que RBIG haya modelado bien el régimen, sino
    que ya no hay información que extraer de tan pocos datos.

    Parameters
    ----------
    n_regimenes
        Número de regímenes; se ajusta un RBIG por cada uno presente en train.
    semilla
        Semilla del muestreo y de las PCA.
    n_componentes
        Dimensión del espacio reducido. ``None`` usa ``min(n//2, 128)``, que
        garantiza al menos dos muestras por dimensión.
    max_capas
        Tope de capas por régimen.
    min_capas
        Capas que se ajustan siempre, antes de evaluar la parada. Evita cortar
        en una meseta inicial engañosa.
    umbral_info
        Umbral de convergencia limpia, en **nats por dimensión**. Se para cuando
        la reducción de una capa cae por debajo de ``umbral_info * k``.
    paciencia
        Capas consecutivas sin mejorar la mejor reducción antes de parar. Es el
        criterio que actúa cuando la reducción se estanca en el suelo de ruido
        del estimador sin llegar nunca a `umbral_info`.
    ruido_residual
        Si ``True``, repone la varianza que descarta la PCA de reducción.
    verboso
        Imprime el resumen de cada régimen durante el ajuste.
    """

    nombre = "rbig"
    etiqueta = "RBIG (normalizing flow)"
    tiene_curva_perdida = False

    def __init__(
        self,
        n_regimenes: int,
        semilla: int = 42,
        n_componentes: int | None = None,
        max_capas: int = 40,
        min_capas: int = 3,
        umbral_info: float = 0.01,
        paciencia: int = 3,
        ruido_residual: bool = True,
        verboso: bool = True,
    ) -> None:
        super().__init__(n_regimenes=n_regimenes, semilla=semilla)

        self.n_componentes = n_componentes
        self.max_capas = int(max_capas)
        self.min_capas = int(min_capas)
        self.umbral_info = float(umbral_info)
        self.paciencia = int(paciencia)
        self.ruido_residual = bool(ruido_residual)
        self.verboso = bool(verboso)

        self.modelos: dict[int, _ModeloRegimen] = {}

    # ── Ajuste ──────────────────────────────────────────────────────────────

    def _fit(self, bloque: np.ndarray, y_reg: np.ndarray, **kwargs) -> None:
        """Ajusta un RBIG independiente por cada régimen presente en train."""
        del kwargs  # el generador no admite opciones por llamada

        self.modelos = {}
        bloque = np.asarray(bloque, dtype=np.float64)

        for regimen in np.unique(y_reg):
            X = bloque[y_reg == regimen]
            if len(X) < 4:
                print(
                    f"  [{self.nombre}] régimen {regimen}: solo {len(X)} muestras, "
                    "se omite. generate() fallará para este régimen."
                )
                continue
            self.modelos[int(regimen)] = self._ajustar_regimen(X, int(regimen))

        if not self.modelos:
            raise ValueError(
                "Ningún régimen tiene muestras suficientes para ajustar RBIG."
            )

    def _ajustar_regimen(self, X: np.ndarray, regimen: int) -> _ModeloRegimen:
        """Ajusta el RBIG de un único régimen y registra su convergencia."""
        n_muestras, n_dim = X.shape
        k = self._dimension_reducida(n_muestras, n_dim)
        self._avisar_muestras_escasas(regimen, n_muestras, n_dim, k)

        # PCA de reducción: fija el espacio de trabajo. Es la única proyección
        # que pierde información; todo lo que viene después es invertible.
        reductor = PCA(n_components=k, random_state=self.semilla).fit(X)
        varianza_explicada = float(reductor.explained_variance_ratio_.sum())

        actual = reductor.transform(X)

        sigma_residual = None
        if self.ruido_residual:
            residuo = X - reductor.inverse_transform(actual)
            sigma_residual = residuo.std(axis=0)

        capas: list[_Capa] = []
        acumulada = 0.0
        umbral = self.umbral_info * k
        mejor = np.inf
        estancadas = 0

        for capa in range(1, self.max_capas + 1):
            marginal = _GaussianizacionMarginal.ajustar(actual)
            gaussianizado = marginal.aplicar(actual)

            # La rotación conserva la dimensión: aquí no se reduce nada, solo se
            # redistribuye la dependencia para que la siguiente gaussianización
            # marginal tenga algo nuevo que eliminar.
            rotacion = PCA(n_components=k, random_state=self.semilla).fit(gaussianizado)
            rotado = rotacion.transform(gaussianizado)

            reduccion = _reduccion_informacion(rotado)
            acumulada += reduccion

            capas.append(_Capa(marginal, rotacion))
            actual = rotado

            self.registrar(
                epoca=capa,
                regimen=regimen,
                reduccion_info=reduccion,
                reduccion_acumulada=acumulada,
            )

            # Parada como la de una pérdida de validación: la reducción por capa
            # es la derivada de la curva acumulada, y se para cuando deja de
            # bajar. La alternativa de usar solo un umbral absoluto no vale para
            # todos los regímenes: con pocas muestras el estimador tiene un suelo
            # de ruido propio, y la señal de convergencia es que la reducción se
            # estanque en ese suelo, no que llegue a cero.
            if reduccion < mejor:
                mejor, estancadas = reduccion, 0
            else:
                estancadas += 1

            convergido = reduccion < umbral or estancadas >= self.paciencia
            if capa >= self.min_capas and convergido:
                break

        # Capa de cierre: deja la cima del flujo con marginales N(0,1) exactas,
        # que es lo que asume el muestreo.
        capas.append(_Capa(_GaussianizacionMarginal.ajustar(actual), None))

        modelo = _ModeloRegimen(
            reductor=reductor,
            capas=capas,
            sigma_residual=sigma_residual,
            n_componentes=k,
            n_muestras=n_muestras,
            varianza_explicada=varianza_explicada,
            capas_usadas=len(capas) - 1,
            reduccion_total=acumulada,
        )

        if self.verboso:
            print(
                f"  [{self.nombre}] régimen {regimen}: {n_muestras} muestras, "
                f"k={k} ({varianza_explicada:.1%} de varianza), "
                f"{modelo.capas_usadas} capas, "
                f"reducción acumulada {acumulada:.2f} nats"
            )
        return modelo

    def _dimension_reducida(self, n_muestras: int, n_dim: int) -> int:
        """Dimensión del espacio de trabajo, acotada por lo que la muestra sostiene.

        El tope ``n//2`` no es arbitrario: la rotación interna es una PCA en el
        espacio reducido, y estimar una covarianza ``k × k`` con menos de dos
        muestras por dimensión da direcciones dominadas por el ruido muestral.
        """
        if self.n_componentes is None:
            k = min(n_muestras // 2, 128)
        else:
            k = self.n_componentes
        return int(np.clip(k, 1, min(n_dim, n_muestras - 1)))

    def _avisar_muestras_escasas(
        self, regimen: int, n_muestras: int, n_dim: int, k: int
    ) -> None:
        """Avisa cuando el régimen no sostiene la dimensión que se le pide."""
        if n_muestras < n_dim:
            print(
                f"  [{self.nombre}] aviso: el régimen {regimen} tiene {n_muestras} "
                f"muestras para {n_dim} dimensiones del bloque. La reducción a "
                f"k={k} es obligatoria, no opcional."
            )
        if n_muestras < 4 * k:
            print(
                f"  [{self.nombre}] aviso: el régimen {regimen} deja "
                f"{n_muestras / k:.1f} muestras por dimensión efectiva. Las CDF "
                "empíricas serán groseras y el modelo poco fiable."
            )

    # ── Generación ──────────────────────────────────────────────────────────

    def _generate(self, n: int, regimen: int) -> np.ndarray:
        """Muestrea ``z ~ N(0, I)`` y recorre el flujo en sentido inverso."""
        modelo = self.modelos.get(int(regimen))
        if modelo is None:
            raise RuntimeError(
                f"{self.nombre}: no hay modelo para el régimen {regimen}. No tenía "
                "muestras suficientes en train."
            )

        Z = self.rng.standard_normal((n, modelo.n_componentes))

        # Orden inverso al ajuste: primero deshacer la rotación (ortogonal, su
        # inversa es la transpuesta, que es lo que aplica PCA.inverse_transform)
        # y después la gaussianización marginal.
        for capa in reversed(modelo.capas):
            if capa.rotacion is not None:
                Z = capa.rotacion.inverse_transform(Z)
            Z = capa.marginal.invertir(Z)

        X = modelo.reductor.inverse_transform(Z)

        if modelo.sigma_residual is not None:
            # Repone la varianza que descartó la PCA de reducción. Sin esto la
            # covarianza sintética es singular y cualquier clasificador
            # distingue lo sintético de lo real por ese detalle, no por la
            # calidad de la distribución.
            X = X + self.rng.standard_normal(X.shape) * modelo.sigma_residual

        return X

    # ── Informe ─────────────────────────────────────────────────────────────

    def resumen_regimenes(self) -> pd.DataFrame:
        """Tabla por régimen: muestras, dimensión reducida, varianza y capas.

        Es el complemento del historial: el historial cuenta cómo convergió cada
        RBIG, y esta tabla con cuánta información se trabajó.
        """
        self._exigir_ajustado()
        return pd.DataFrame(
            [
                {
                    "regimen": regimen,
                    "n_muestras": m.n_muestras,
                    "n_componentes": m.n_componentes,
                    "varianza_explicada": m.varianza_explicada,
                    "capas": m.capas_usadas,
                    "reduccion_total": m.reduccion_total,
                }
                for regimen, m in sorted(self.modelos.items())
            ]
        )

    # ── Persistencia ────────────────────────────────────────────────────────

    def _estado(self) -> dict:
        """Todo el modelo es numpy y objetos de sklearn: el pickle de la base vale."""
        return {
            "modelos": self.modelos,
            "n_componentes": self.n_componentes,
            "max_capas": self.max_capas,
            "min_capas": self.min_capas,
            "umbral_info": self.umbral_info,
            "paciencia": self.paciencia,
            "ruido_residual": self.ruido_residual,
        }

    def _restaurar(self, estado: dict) -> None:
        self.modelos = estado["modelos"]
        self.n_componentes = estado["n_componentes"]
        self.max_capas = estado["max_capas"]
        self.min_capas = estado["min_capas"]
        self.umbral_info = estado["umbral_info"]
        self.paciencia = estado["paciencia"]
        self.ruido_residual = estado["ruido_residual"]
