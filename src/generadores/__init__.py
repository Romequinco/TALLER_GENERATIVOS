"""Generadores de datos sintéticos del taller.

Todos comparten la interfaz definida en `base.GeneradorSintetico`, de modo que
los notebooks de mezcla y análisis los tratan por igual.

Los módulos concretos no se importan aquí: cada uno arrastra sus propias
dependencias (Keras, scipy) y cargarlas todas al importar el paquete haría
lento cualquier notebook. Para obtener un generador por su nombre corto se usa
`base.instanciar`::

    from src.generadores.base import instanciar
    gen = instanciar("cgan", n_regimenes=3)
"""

from __future__ import annotations

from .base import REGISTRO, GeneradorSintetico, instanciar

__all__ = ["GeneradorSintetico", "REGISTRO", "instanciar"]
