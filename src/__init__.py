"""Taller B5-T1 · Generación de datos financieros sintéticos.

Importar este paquete tiene un efecto lateral deliberado: fija el backend de
Keras a PyTorch ANTES de que ningún módulo importe `keras`. Keras 3 lee la
variable de entorno en el momento de la importación, así que hacerlo más tarde
no surte efecto y el backend quedaría en TensorFlow.

Por eso todos los notebooks empiezan por::

    import src            # fija el backend
    from src import datos, features
"""

from __future__ import annotations

import os
from pathlib import Path

# Debe ejecutarse antes de cualquier `import keras` del proyecto.
os.environ.setdefault("KERAS_BACKEND", "torch")

# Silencia los avisos informativos de oneDNN de TensorFlow, que se importa de
# forma indirecta y ensucia la salida de los notebooks.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

# ── Rutas del proyecto ───────────────────────────────────────────────────────
# Todas las rutas se derivan de la raíz del repositorio para que los notebooks
# funcionen igual desde notebooks/ que desde la raíz.
RAIZ = Path(__file__).resolve().parent.parent

DIR_DATOS = RAIZ / "data"
DIR_CRUDO = DIR_DATOS / "raw"
DIR_PROCESADO = DIR_DATOS / "processed"
DIR_SINTETICO = DIR_DATOS / "synthetic"
CATALOGO = DIR_DATOS / "catalog.yaml"

DIR_MODELOS = RAIZ / "models"
DIR_MODELOS_GEN = DIR_MODELOS / "generadores"
DIR_MODELOS_DOWN = DIR_MODELOS / "downstream"

DIR_RESULTADOS = RAIZ / "results"
DIR_HISTORIALES = DIR_RESULTADOS / "historiales"
DIR_METRICAS = DIR_RESULTADOS / "metricas"
DIR_FIGURAS = DIR_RESULTADOS / "figures"

__all__ = [
    "RAIZ",
    "DIR_DATOS",
    "DIR_CRUDO",
    "DIR_PROCESADO",
    "DIR_SINTETICO",
    "CATALOGO",
    "DIR_MODELOS",
    "DIR_MODELOS_GEN",
    "DIR_MODELOS_DOWN",
    "DIR_RESULTADOS",
    "DIR_HISTORIALES",
    "DIR_METRICAS",
    "DIR_FIGURAS",
]
