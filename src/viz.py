"""Estilo y figuras del proyecto.

Todas las gráficas del informe salen de aquí. Centralizarlas cumple el
requisito del enunciado —"el código debe generar todas las gráficas y tablas
reportadas"— y garantiza que las siete curvas de generadores distintos usen los
mismos colores en todas las figuras, que es lo que hace comparable un panel con
otro.

Criterios de diseño
-------------------
* **Un color por generador, fijo.** El color identifica la entidad, no su
  posición en el ranking: filtrar generadores no debe repintar los que quedan.
* **Una sola escala por eje.** Nunca dos ejes verticales con magnitudes
  distintas; si hay dos métricas, son dos paneles.
* **Leyenda siempre con dos o más series**, y etiquetas directas en los extremos
  de las líneas cuando caben. La identidad nunca depende solo del color.
* **Paleta verificada para daltonismo.** El orden de colores está fijado y
  validado; no se amplía inventando tonos nuevos.
* Rejilla y ejes discretos: la tinta se reserva para los datos.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import DIR_FIGURAS

# ─────────────────────────────────────────────────────────────────────────────
# Paleta
#
# Orden fijo y validado para separación bajo deuteranopia, protanopia y
# tritanopia sobre fondo claro. Se asigna por orden de declaración, nunca
# ciclando: un octavo generador no genera un tono nuevo, se agrupa aparte.
# ─────────────────────────────────────────────────────────────────────────────

PALETA = (
    "#2a78d6",  # azul
    "#eb6834",  # naranja
    "#1baf7a",  # aguamarina
    "#eda100",  # amarillo
    "#e87ba4",  # magenta
    "#008300",  # verde
    "#4a3aa7",  # violeta
    "#e34948",  # rojo
)

#: Color de cada generador, fijo en todo el informe. El experimento sin
#: sintéticos va en gris: es la referencia, no una serie más que comparar.
COLOR_GENERADOR: dict[str, str] = {
    "solo_real": "#6b6b6b",
    "jitter": PALETA[0],
    "gaussiano": PALETA[1],
    "cgan": PALETA[2],
    "cvae": PALETA[3],
    "rbig": PALETA[4],
    "flow_matching": PALETA[5],
    "difusion": PALETA[6],
    "autoregresivo": PALETA[7],
}

TINTA_PRIMARIA = "#0b0b0b"
TINTA_SECUNDARIA = "#52514e"
REJILLA = "#dcdcd8"


def color(generador: str) -> str:
    """Color asignado a un generador. Gris neutro si no está registrado."""
    return COLOR_GENERADOR.get(generador, "#9a9a95")


def rampa_secuencial(n: int) -> list[str]:
    """Rampa de un solo tono, de claro a oscuro, para magnitudes ordenadas.

    Se usa cuando la serie codifica una cantidad creciente (la proporción de
    sintéticos), no una identidad. Un tono único con luminancia monótona se lee
    como orden; una paleta categórica en ese papel se leería como categorías sin
    relación.
    """
    mapa = plt.get_cmap("Blues")
    # Se arranca en 0.35 para que el primer paso no sea casi blanco sobre el
    # fondo del gráfico.
    return [mapa(v) for v in np.linspace(0.35, 0.95, max(n, 2))]


def aplicar_estilo() -> None:
    """Fija el estilo de matplotlib. Se llama una vez por notebook."""
    plt.rcParams.update(
        {
            "figure.figsize": (9, 5),
            "figure.dpi": 110,
            "savefig.dpi": 200,
            "savefig.bbox": "tight",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "axes.labelsize": 10,
            "axes.labelcolor": TINTA_SECUNDARIA,
            "axes.edgecolor": REJILLA,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": REJILLA,
            "grid.linewidth": 0.7,
            "xtick.color": TINTA_SECUNDARIA,
            "ytick.color": TINTA_SECUNDARIA,
            "legend.frameon": False,
            "lines.linewidth": 2.0,
            "lines.markersize": 5,
        }
    )


def guardar(fig, nombre: str) -> None:
    """Guarda la figura en `results/figures/<nombre>.png`.

    El informe se compone con estos ficheros, así que el nombre debe ser estable
    entre ejecuciones.
    """
    DIR_FIGURAS.mkdir(parents=True, exist_ok=True)
    fig.savefig(DIR_FIGURAS / f"{nombre}.png")


# ─────────────────────────────────────────────────────────────────────────────
# Convergencia de los generadores
# ─────────────────────────────────────────────────────────────────────────────


def curva_convergencia(
    historial: pd.DataFrame, titulo: str, referencia: float | None = None, eje=None
):
    """Grafica el historial de un generador, una línea por métrica registrada.

    Parameters
    ----------
    referencia
        Valor de equilibrio a marcar con una horizontal discontinua. En una GAN
        es $\\log 2 \\approx 0.69$: ahí es donde deben oscilar ambas pérdidas si
        el juego está equilibrado. Sin esa línea las curvas de una GAN son
        prácticamente ilegibles, porque no bajan monótonamente y no hay forma de
        juzgar a ojo si convergieron.
    """
    eje = eje or plt.subplots()[1]
    columnas = [c for c in historial.select_dtypes(include=np.number).columns if c != "epoca"]

    for i, columna in enumerate(columnas):
        eje.plot(
            historial["epoca"],
            historial[columna],
            color=PALETA[i % len(PALETA)],
            label=columna.replace("_", " "),
        )

    if referencia is not None:
        eje.axhline(
            referencia,
            color=TINTA_SECUNDARIA,
            linestyle="--",
            linewidth=1.2,
            label=f"equilibrio ({referencia:.2f})",
        )

    eje.set_title(titulo)
    eje.set_xlabel("época")
    eje.set_ylabel("valor")
    eje.legend(ncols=2)
    return eje


# ─────────────────────────────────────────────────────────────────────────────
# Resultados del barrido
# ─────────────────────────────────────────────────────────────────────────────


def metrica_vs_reales(
    metricas: pd.DataFrame,
    metrica: str,
    ratio: float,
    politica: str = "equilibrado",
    eje=None,
):
    """Figura principal: métrica frente al número de datos reales.

    Una línea por generador, más la referencia sin sintéticos en gris. Es el
    gráfico que responde a la pregunta del taller: los sintéticos aportan
    cuando hay pocos datos reales y dejan de aportar cuando ya hay muchos, de
    modo que las líneas de colores deben separarse de la gris por la izquierda
    y confluir con ella por la derecha.

    El eje horizontal es logarítmico porque los niveles de reales crecen
    geométricamente y en escala lineal se amontonarían a la izquierda.
    """
    eje = eje or plt.subplots()[1]

    seleccion = metricas[
        (metricas["ratio"] == ratio) & (metricas["politica"] == politica)
    ]
    referencia = metricas[metricas["generador"] == "solo_real"]

    for datos, nombre in [(referencia, "solo_real")] + [
        (g, n) for n, g in seleccion.groupby("generador")
    ]:
        serie = datos.sort_values("n_reales_num")
        if serie.empty:
            continue
        eje.plot(
            serie["n_reales_num"],
            serie[metrica],
            marker="o",
            color=color(nombre),
            label=nombre.replace("_", " "),
            linestyle="--" if nombre == "solo_real" else "-",
            zorder=3 if nombre == "solo_real" else 2,
        )
        # Etiqueta directa en el extremo derecho: la leyenda sola obliga a ir y
        # venir entre la curva y el recuadro para identificar cada línea.
        eje.annotate(
            nombre.replace("_", " "),
            xy=(serie["n_reales_num"].iloc[-1], serie[metrica].iloc[-1]),
            xytext=(6, 0),
            textcoords="offset points",
            color=TINTA_SECUNDARIA,
            fontsize=8,
            va="center",
        )

    eje.set_xscale("log")
    eje.set_title(f"{metrica.replace('_', ' ')} frente al volumen de datos reales")
    eje.set_xlabel("ventanas reales de entrenamiento (escala log)")
    eje.set_ylabel(metrica.replace("_", " "))
    eje.legend(ncols=2, fontsize=8)
    return eje


def metrica_vs_ratio(
    metricas: pd.DataFrame,
    metrica: str,
    n_reales,
    politica: str = "equilibrado",
    eje=None,
):
    """Métrica frente a la proporción de sintéticos, con los reales fijos.

    Barras agrupadas por generador. Se deja un hueco de fondo entre barras
    adyacentes para que los bloques de color no se fundan visualmente.
    """
    eje = eje or plt.subplots()[1]

    seleccion = metricas[
        (metricas["n_reales"] == n_reales) & (metricas["politica"] == politica)
    ]
    generadores = sorted(seleccion["generador"].unique())
    ratios = sorted(seleccion["ratio"].unique())

    ancho = 0.8 / max(len(generadores), 1)
    posiciones = np.arange(len(ratios))

    for i, generador in enumerate(generadores):
        datos = seleccion[seleccion["generador"] == generador].set_index("ratio")
        valores = [datos[metrica].get(r, np.nan) for r in ratios]
        eje.bar(
            posiciones + i * ancho,
            valores,
            width=ancho * 0.9,  # el 10 % restante es el hueco entre barras
            color=color(generador),
            label=generador.replace("_", " "),
        )

    # Línea de referencia: el resultado sin ningún sintético.
    base = metricas[
        (metricas["generador"] == "solo_real") & (metricas["n_reales"] == n_reales)
    ]
    if not base.empty:
        eje.axhline(
            base[metrica].iloc[0],
            color=TINTA_SECUNDARIA,
            linestyle="--",
            linewidth=1.2,
            label="sin sintéticos",
        )

    eje.set_xticks(posiciones + 0.4 - ancho / 2)
    eje.set_xticklabels([f"{r:g}×" for r in ratios])
    eje.set_title(f"{metrica.replace('_', ' ')} con {n_reales} ventanas reales")
    eje.set_xlabel("muestras sintéticas por muestra real")
    eje.set_ylabel(metrica.replace("_", " "))
    eje.legend(ncols=3, fontsize=8)
    return eje


# ─────────────────────────────────────────────────────────────────────────────
# Diagnóstico de datos y de muestras sintéticas
# ─────────────────────────────────────────────────────────────────────────────


def confusion(matriz: pd.DataFrame, titulo: str, eje=None):
    """Matriz de confusión normalizada por fila, con los valores impresos.

    Se normaliza por fila (por clase real) porque con clases desbalanceadas la
    matriz en absoluto solo muestra que la clase mayoritaria es grande.
    """
    eje = eje or plt.subplots()[1]
    normalizada = matriz.div(matriz.sum(axis=1), axis=0).fillna(0)

    imagen = eje.imshow(normalizada, cmap="Blues", vmin=0, vmax=1)
    eje.set_xticks(range(len(matriz.columns)), matriz.columns)
    eje.set_yticks(range(len(matriz.index)), matriz.index)

    for i in range(len(matriz.index)):
        for j in range(len(matriz.columns)):
            valor = normalizada.iloc[i, j]
            eje.text(
                j,
                i,
                f"{valor:.0%}\n({matriz.iloc[i, j]})",
                ha="center",
                va="center",
                fontsize=8,
                # Texto claro sobre celda oscura para mantener el contraste.
                color="white" if valor > 0.5 else TINTA_PRIMARIA,
            )

    eje.set_title(titulo)
    eje.set_xlabel("predicho")
    eje.set_ylabel("real")
    eje.grid(False)
    plt.colorbar(imagen, ax=eje, fraction=0.046, label="proporción de la clase real")
    return eje


def real_vs_sintetico(
    reales: np.ndarray, sinteticos: np.ndarray, titulo: str, eje=None
):
    """Proyección PCA de reales y sintéticos superpuestos.

    Dos series solamente, así que la separación de color está holgada. Es la
    comprobación visual más rápida: si la nube sintética no cubre la real, el
    generador ha colapsado; si la desborda, está inventando configuraciones que
    el mercado nunca produjo.

    La PCA se ajusta **solo con los reales**, para que los ejes representen la
    estructura del mercado y no la del generador.
    """
    from sklearn.decomposition import PCA

    eje = eje or plt.subplots()[1]

    pca = PCA(n_components=2).fit(reales)
    proyeccion_real = pca.transform(reales)
    proyeccion_sint = pca.transform(sinteticos)

    eje.scatter(
        proyeccion_real[:, 0], proyeccion_real[:, 1],
        s=8, alpha=0.5, color=PALETA[0], label="real", edgecolors="none",
    )
    eje.scatter(
        proyeccion_sint[:, 0], proyeccion_sint[:, 1],
        s=8, alpha=0.5, color=PALETA[1], label="sintético", edgecolors="none",
    )

    varianza = pca.explained_variance_ratio_
    eje.set_title(titulo)
    eje.set_xlabel(f"componente 1 ({varianza[0]:.0%} de la varianza)")
    eje.set_ylabel(f"componente 2 ({varianza[1]:.0%} de la varianza)")
    eje.legend()
    return eje


def serie_regimenes(
    precios: pd.Series, regimen: pd.Series, titulo: str = "Regímenes detectados", eje=None
):
    """Precio del índice con el fondo sombreado según el régimen.

    Es la figura que valida el etiquetado de un vistazo: las bandas de crisis
    deben caer sobre 2008, 2020 y 2022. Si no lo hacen, el HMM no ha convergido
    a algo interpretable y no tiene sentido seguir.
    """
    eje = eje or plt.subplots()[1]

    eje.plot(precios.index, precios.to_numpy(), color=TINTA_PRIMARIA, linewidth=1.2)

    n_estados = int(regimen.max()) + 1
    tonos = rampa_secuencial(n_estados)
    cambios = regimen.ne(regimen.shift()).cumsum()

    for _, tramo in regimen.groupby(cambios):
        estado = int(tramo.iloc[0])
        if estado == 0:  # la calma se deja sin sombrear, para no saturar
            continue
        eje.axvspan(tramo.index[0], tramo.index[-1], color=tonos[estado], alpha=0.25, lw=0)

    from .evaluacion import nombres_regimenes

    from matplotlib.patches import Patch

    etiquetas = nombres_regimenes(n_estados)
    eje.legend(
        handles=[
            Patch(facecolor=tonos[k], alpha=0.25, label=etiquetas[k])
            for k in range(1, n_estados)
        ],
        loc="upper left",
    )
    eje.set_title(titulo)
    eje.set_ylabel("nivel del índice")
    eje.set_yscale("log")
    return eje
