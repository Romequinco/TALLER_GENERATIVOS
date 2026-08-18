"""Interfaz común de los generadores de datos sintéticos.

Este fichero es el contrato que permite que los tres integrantes trabajen en
paralelo. Cada generador vive en su propio módulo y lo desarrolla una persona,
pero todos exponen la misma API, de modo que los notebooks de mezcla, barrido y
análisis (11 a 14) tratan a los siete por igual y las figuras salen homogéneas.

Contrato de un generador
------------------------
1. `fit(bloque, y_reg)` se entrena **solo** con el tramo de train.
2. `generate(n, regimen)` devuelve `n` muestras del régimen pedido.
3. `historial` deja una tabla de convergencia que el notebook 13 grafica sin
   saber nada del modelo concreto.
4. `guardar` / `cargar` persisten el modelo en `models/generadores/` para que
   nadie tenga que reentrenar lo de otro.

El punto 4 es lo que hace el repositorio utilizable: quien clone el proyecto
puede saltar directamente al análisis sin repetir siete entrenamientos.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import pandas as pd

from .. import DIR_MODELOS_GEN, DIR_SINTETICO


class GeneradorSintetico(ABC):
    """Clase base de todos los generadores del taller.

    Trabaja sobre el **bloque aplanado** que produce `ventanas.empaquetar()`:
    una matriz ``(n, d)`` con la ventana X aplanada y la volatilidad futura en
    la última columna. El régimen no forma parte del bloque: es la condición.

    Las subclases implementan `_fit` y `_generate`; el resto (validación,
    registro del historial, persistencia) lo resuelve la base.
    """

    #: Nombre corto en snake_case. Da nombre a los ficheros de checkpoint,
    #: historial y muestras, y es la etiqueta de este generador en las figuras.
    nombre: str = "base"

    #: Etiqueta legible para leyendas y tablas del informe.
    etiqueta: str = "Generador base"

    #: ``True`` si el modelo se entrena por descenso de gradiente y produce una
    #: curva de pérdida por época. Los generadores analíticos (jitter,
    #: gaussiano, RBIG) lo dejan en ``False`` y reportan otra cosa como
    #: diagnóstico de convergencia.
    tiene_curva_perdida: bool = True

    def __init__(self, n_regimenes: int, semilla: int = 42) -> None:
        self.n_regimenes = n_regimenes
        self.semilla = semilla
        self.rng = np.random.default_rng(semilla)

        self.dimension: int | None = None
        self.ajustado: bool = False
        #: Tabla de convergencia: una fila por época (o por capa/iteración en
        #: los modelos sin gradiente). Columna obligatoria: ``epoca``.
        self.historial: pd.DataFrame = pd.DataFrame()

    # ── API pública ─────────────────────────────────────────────────────────

    def fit(self, bloque: np.ndarray, y_reg: np.ndarray, **kwargs) -> "GeneradorSintetico":
        """Ajusta el generador. `bloque` e `y_reg` deben ser SOLO de train.

        Parameters
        ----------
        bloque
            Matriz ``(n, d)`` de `ventanas.empaquetar()`, ya escalada.
        y_reg
            Vector ``(n,)`` de regímenes enteros, la variable de condición.
        """
        bloque = np.asarray(bloque, dtype=np.float32)
        y_reg = np.asarray(y_reg, dtype=np.int64)

        if bloque.ndim != 2:
            raise ValueError(f"Se esperaba un bloque 2D, recibido {bloque.shape}.")
        if len(bloque) != len(y_reg):
            raise ValueError(
                f"Desajuste de longitudes: {len(bloque)} bloques y {len(y_reg)} etiquetas."
            )
        if not np.isfinite(bloque).all():
            raise ValueError("El bloque contiene NaN o infinitos.")

        self.dimension = bloque.shape[1]
        self._fit(bloque, y_reg, **kwargs)
        self.ajustado = True
        return self

    def generate(self, n: int, regimen: int) -> np.ndarray:
        """Genera `n` muestras sintéticas del régimen indicado.

        Returns
        -------
        Matriz ``(n, d)`` en el mismo espacio escalado que el bloque de entrada.
        """
        self._exigir_ajustado()
        if not 0 <= regimen < self.n_regimenes:
            raise ValueError(
                f"Régimen {regimen} fuera de rango [0, {self.n_regimenes})."
            )
        if n <= 0:
            return np.empty((0, self.dimension), dtype=np.float32)

        muestras = np.asarray(self._generate(n, regimen), dtype=np.float32)

        if muestras.shape != (n, self.dimension):
            raise RuntimeError(
                f"{self.nombre}: se pidieron {(n, self.dimension)} muestras y se "
                f"devolvieron {muestras.shape}."
            )
        if not np.isfinite(muestras).all():
            n_malas = int((~np.isfinite(muestras)).any(axis=1).sum())
            raise RuntimeError(
                f"{self.nombre}: {n_malas} de {n} muestras contienen NaN o "
                "infinitos. Suele indicar divergencia del entrenamiento."
            )
        return muestras

    def generate_dataset(self, reparto: dict[int, int]) -> tuple[np.ndarray, np.ndarray]:
        """Genera un dataset completo con el número de muestras pedido por régimen.

        Parameters
        ----------
        reparto
            Diccionario ``{regimen: n_muestras}``. Es el mecanismo con el que se
            sobre-representa la clase de crisis, que es la razón de ser del
            taller.

        Returns
        -------
        ``(bloques, y_reg)`` con las muestras de todos los regímenes barajadas.
        """
        bloques, etiquetas = [], []
        for regimen, n in sorted(reparto.items()):
            if n <= 0:
                continue
            bloques.append(self.generate(n, regimen))
            etiquetas.append(np.full(n, regimen, dtype=np.int64))

        if not bloques:
            raise ValueError("El reparto no pide ninguna muestra.")

        X = np.concatenate(bloques)
        y = np.concatenate(etiquetas)

        orden = self.rng.permutation(len(X))
        return X[orden], y[orden]

    # ── Métodos que implementa cada generador ───────────────────────────────

    @abstractmethod
    def _fit(self, bloque: np.ndarray, y_reg: np.ndarray, **kwargs) -> None:
        """Lógica de ajuste. Debe rellenar `self.historial`."""

    @abstractmethod
    def _generate(self, n: int, regimen: int) -> np.ndarray:
        """Lógica de muestreo para un único régimen."""

    # ── Diagnóstico de convergencia ─────────────────────────────────────────

    def registrar(self, **columnas) -> None:
        """Añade una fila al historial de convergencia.

        Se llama dentro del bucle de entrenamiento::

            self.registrar(epoca=e, perdida_gen=lg, perdida_disc=ld)

        El notebook 13 grafica todas las columnas numéricas frente a ``epoca``,
        así que basta con registrar lo relevante para que aparezca en el informe.
        """
        fila = pd.DataFrame([columnas])
        self.historial = (
            fila if self.historial.empty else pd.concat([self.historial, fila], ignore_index=True)
        )

    def resumen_convergencia(self) -> dict[str, float]:
        """Últimos valores del historial, para la tabla comparativa del informe."""
        if self.historial.empty:
            return {}
        numericas = self.historial.select_dtypes(include=np.number)
        return {f"final_{c}": float(numericas[c].iloc[-1]) for c in numericas.columns}

    # ── Persistencia ────────────────────────────────────────────────────────

    @property
    def ruta_modelo(self) -> Path:
        return DIR_MODELOS_GEN / self.nombre

    def guardar(self) -> Path:
        """Persiste modelo, historial y metadatos en `models/generadores/<nombre>/`.

        Deja tres ficheros:
          ``historial.csv``  curva de convergencia, para regenerar las figuras
          ``meta.json``      dimensión, semilla y resumen de convergencia
          ``modelo.*``       pesos o parámetros, según lo que escriba `_guardar`
        """
        self._exigir_ajustado()
        destino = self.ruta_modelo
        destino.mkdir(parents=True, exist_ok=True)

        self.historial.to_csv(destino / "historial.csv", index=False)

        meta = {
            "nombre": self.nombre,
            "etiqueta": self.etiqueta,
            "clase": type(self).__name__,
            "dimension": self.dimension,
            "n_regimenes": self.n_regimenes,
            "semilla": self.semilla,
            "tiene_curva_perdida": self.tiene_curva_perdida,
            **self.resumen_convergencia(),
        }
        with open(destino / "meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

        self._guardar(destino)
        return destino

    def cargar(self) -> "GeneradorSintetico":
        """Restaura un generador ya entrenado desde disco.

        Es lo que permite que cada integrante entrene solo su generador: los
        demás se cargan de `models/generadores/` sin reentrenar.
        """
        origen = self.ruta_modelo
        if not origen.exists():
            raise FileNotFoundError(
                f"No hay ningún {self.nombre} entrenado en {origen}. "
                f"Ejecuta antes su notebook, o pide a quien lo entrenó que "
                "suba la carpeta al repositorio."
            )

        with open(origen / "meta.json", encoding="utf-8") as f:
            meta = json.load(f)
        self.dimension = meta["dimension"]
        self.n_regimenes = meta["n_regimenes"]

        historial = origen / "historial.csv"
        if historial.exists():
            self.historial = pd.read_csv(historial)

        self._cargar(origen)
        self.ajustado = True
        return self

    def _guardar(self, destino: Path) -> None:
        """Persiste los parámetros del modelo. Por defecto, pickle del objeto."""
        import pickle

        with open(destino / "modelo.pkl", "wb") as f:
            pickle.dump(self._estado(), f)

    def _cargar(self, origen: Path) -> None:
        """Restaura los parámetros del modelo."""
        import pickle

        with open(origen / "modelo.pkl", "rb") as f:
            self._restaurar(pickle.load(f))

    def _estado(self) -> dict:
        """Diccionario serializable con los parámetros aprendidos.

        Las subclases con pesos de red sobreescriben `_guardar`/`_cargar` en su
        lugar y usan el formato nativo de Keras.
        """
        return {}

    def _restaurar(self, estado: dict) -> None:
        pass

    # ── Muestras en disco ───────────────────────────────────────────────────

    def exportar_muestras(self, bloques: np.ndarray, y_reg: np.ndarray) -> Path:
        """Guarda un dataset sintético en `data/synthetic/<nombre>.npz`.

        Estos ficheros sí se versionan: son pequeños y son lo que permite al
        resto del grupo montar las mezclas sin ejecutar ningún generador.
        """
        DIR_SINTETICO.mkdir(parents=True, exist_ok=True)
        ruta = DIR_SINTETICO / f"{self.nombre}.npz"
        np.savez_compressed(
            ruta,
            bloques=bloques.astype(np.float32),
            y_reg=np.asarray(y_reg, dtype=np.int64),
        )
        return ruta

    @staticmethod
    def importar_muestras(nombre: str) -> tuple[np.ndarray, np.ndarray]:
        """Lee las muestras exportadas por un generador."""
        ruta = DIR_SINTETICO / f"{nombre}.npz"
        if not ruta.exists():
            raise FileNotFoundError(
                f"No hay muestras de '{nombre}' en {ruta}. Ejecuta su notebook "
                "o descarga la carpeta data/synthetic/ del repositorio."
            )
        datos = np.load(ruta)
        return datos["bloques"], datos["y_reg"]

    # ── Utilidades internas ─────────────────────────────────────────────────

    def _exigir_ajustado(self) -> None:
        if not self.ajustado:
            raise RuntimeError(
                f"{self.nombre} no está ajustado. Llama a fit() o a cargar()."
            )

    def __repr__(self) -> str:
        estado = "ajustado" if self.ajustado else "sin ajustar"
        return f"{type(self).__name__}(nombre={self.nombre!r}, d={self.dimension}, {estado})"


# ─────────────────────────────────────────────────────────────────────────────
# Registro de generadores
#
# Los notebooks 11-14 recorren este diccionario en vez de importar cada clase a
# mano. Añadir un generador nuevo al taller es registrarlo aquí.
# ─────────────────────────────────────────────────────────────────────────────

REGISTRO: dict[str, str] = {
    "jitter": "src.generadores.jitter.Jitter",
    "gaussiano": "src.generadores.gaussiano.GaussianoCondicional",
    "cgan": "src.generadores.cgan.CGAN",
    "cvae": "src.generadores.cvae.CVAE",
    "rbig": "src.generadores.rbig.RBIGCondicional",
    "flow_matching": "src.generadores.flow_matching.FlowMatching",
    "difusion": "src.generadores.difusion.DifusionCondicional",
}


def instanciar(nombre: str, n_regimenes: int, **kwargs) -> GeneradorSintetico:
    """Crea un generador por su nombre corto, sin importarlo explícitamente."""
    from importlib import import_module

    if nombre not in REGISTRO:
        raise KeyError(f"Generador desconocido: {nombre!r}. Opciones: {sorted(REGISTRO)}")

    modulo, clase = REGISTRO[nombre].rsplit(".", 1)
    return getattr(import_module(modulo), clase)(n_regimenes=n_regimenes, **kwargs)
