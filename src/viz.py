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

#: Gris de contexto para series que son referencia y no protagonistas. Es el
#: mismo tono que `COLOR_GENERADOR["solo_real"]` y por la misma razón: no se
#: introduce ningún color nuevo en la paleta.
GRIS_CONTEXTO = "#6b6b6b"


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


def rampa_divergente():
    """Mapa de color para magnitudes con signo, con el cero en el punto medio.

    Devuelve un `Colormap` y no una lista como `rampa_secuencial`, porque su
    consumidor es `imshow`, que interpola de forma continua.

    Los polos son azul y rojo —frío y cálido se leen como opuestos— y el punto
    medio es el gris de la rejilla, que se lee como "nada". Un tono en el centro
    haría que el cero pareciera un valor. Quien la use **debe** fijar límites
    simétricos: si no, el gris deja de caer en cero y el color miente sobre el
    signo.
    """
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list(
        "divergente", [PALETA[0], REJILLA, PALETA[7]]
    )


def _catalogo():
    """Catálogo del proyecto. Import diferido para no acoplar `viz` a `config`."""
    from .config import cargar_catalogo

    return cargar_catalogo()


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
            # Trama fina: es el canal que sustituye al color cuando la figura se
            # imprime en gris, y a grosor por defecto tapa la barra que marca.
            "hatch.linewidth": 0.6,
        }
    )


def guardar(fig, nombre: str) -> None:
    """Guarda la figura en `results/figures/<nombre>.png`.

    El informe se compone con estos ficheros, así que el nombre debe ser estable
    entre ejecuciones.

    Recompone el reparto de la figura antes de guardar. `savefig.bbox = "tight"`
    solo recorta el margen exterior: no evita que el título de un panel se monte
    sobre el del vecino, que es lo que ocurre en cuanto una figura tiene dos
    paneles y títulos que afirman una conclusión en vez de nombrar un eje.
    """
    DIR_FIGURAS.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(DIR_FIGURAS / f"{nombre}.png")


# ─────────────────────────────────────────────────────────────────────────────
# Panel de mercado y canales
#
# Las figuras del notebook 00. Todas aceptan `titulo=None` y, en ese caso,
# componen un título que afirma la conclusión con la cifra ya calculada. Es una
# extensión deliberada del contrato del módulo: en estas figuras el título es un
# resultado, no una etiqueta, y componerlo fuera obligaría al notebook a
# recalcular lo que la función ya tiene delante.
# ─────────────────────────────────────────────────────────────────────────────


def panel_mercado(precios: pd.DataFrame, titulo: str | None = None, eje=None):
    """Todo el universo descargado, en base 100 y escala logarítmica.

    Cómo se lee: cada línea es un activo y el color es su **rol** en el panel, no
    su identidad. Son quince series y la paleta tiene ocho colores, así que
    colorear activo a activo obligaría a ciclar tonos; agrupar por rol da cinco
    grupos, que es además la lectura útil: los sectores se mueven en bloque, la
    renta fija va por su cuenta y el VIX se dispara donde todo lo demás cae.

    La escala es logarítmica porque los niveles van de 20 a 6.000: en escala
    lineal el índice aplastaría al resto contra el eje. Y la base 100 hace
    comparables activos que empiezan en precios muy distintos.

    Qué la invalida: si alguna serie empezara más tarde que las demás, el panel
    no estaría alineado y el recorte habría fallado. Todas arrancan en el mismo
    punto por construcción.
    """
    from matplotlib.lines import Line2D

    eje = eje or plt.subplots()[1]
    roles = {a["nombre"]: a["rol"] for a in _catalogo()["universo"]}
    orden = ["indice", "sector", "riesgo", "renta_fija", "divisa"]
    tono = {rol: PALETA[i] for i, rol in enumerate(orden)}

    base = precios / precios.iloc[0] * 100
    for columna in base.columns:
        rol = roles[columna]
        # El índice va grueso y por encima: es la serie que gobierna el
        # etiquetado de regímenes, las demás son contexto.
        principal = rol == "indice"
        eje.plot(
            base.index,
            base[columna],
            color=tono[rol],
            linewidth=1.8 if principal else 0.9,
            alpha=1.0 if principal else 0.65,
            zorder=3 if principal else 2,
        )

    eje.set_yscale("log")
    eje.set_title(titulo or f"Los {len(base.columns)} activos del panel, base 100", fontsize=10, fontweight="normal")
    eje.set_ylabel("nivel, base 100 (escala log)")
    eje.legend(
        handles=[
            Line2D([], [], color=tono[r], linewidth=2, label=r.replace("_", " "))
            for r in orden
        ],
        loc="upper left",
        ncols=5,
        fontsize=8,
    )
    return eje


def mapa_canales(canales: pd.DataFrame, titulo: str | None = None, eje=None):
    """Los 20 canales de X a lo largo del tiempo, en un solo mapa.

    Cómo se lee: una fila por canal, el tiempo en horizontal y el color es el
    valor tipificado y recortado a ±3 desviaciones. Es la forma de ver de un
    vistazo la matriz que van a recibir los generadores. Lo que hay que buscar
    son las **columnas verticales**: momentos en que casi todos los canales se
    encienden a la vez. Ahí es donde está la señal de régimen, y es también la
    razón por la que un generador tiene que modelar los canales conjuntamente y
    no uno a uno.

    La tipificación es **solo para pintar**: iguala escalas que difieren en dos
    órdenes de magnitud para que un canal no monopolice el color. No se guarda,
    no alimenta a ningún modelo y no sustituye al escalado del cuaderno 02, que
    sí se ajusta únicamente con el tramo de entrenamiento.

    Qué la invalida: una franja horizontal de color plano significa un canal
    constante o casi; una columna en blanco, un hueco en el índice.
    """
    eje = eje or plt.subplots()[1]
    z = ((canales - canales.mean()) / canales.std()).clip(-3, 3)

    imagen = eje.imshow(
        z.T,
        aspect="auto",
        cmap=rampa_divergente(),
        vmin=-3,
        vmax=3,
        interpolation="nearest",
        extent=[0, len(z), len(z.columns) - 0.5, -0.5],
    )
    # El eje temporal se rotula con los años, no con el número de fila.
    anios = pd.Series(z.index.year)
    marcas = anios.drop_duplicates(keep="first")
    paso = max(len(marcas) // 8, 1)
    eje.set_xticks(marcas.index[::paso], marcas.to_numpy()[::paso])

    eje.set_yticks(range(len(z.columns)), list(z.columns), fontsize=7)
    eje.set_title(
        titulo or f"X · {len(z)} sesiones × {len(z.columns)} canales",
        fontsize=10,
        fontweight="normal",
    )
    eje.grid(False)
    plt.colorbar(
        imagen, ax=eje, fraction=0.025, pad=0.01,
        label="desviaciones típicas (recortado a ±3)",
    )
    return eje


def carril_particiones(
    conjuntos: dict,
    indice: pd.DatetimeIndex,
    ventanas,
    solape: pd.DataFrame,
    titulo: str | None = None,
    eje=None,
):
    """Las tres particiones sobre el eje temporal, con el embargo entre ellas.

    Cómo se lee: cada barra es la extensión de una partición, no una ventana. Lo
    que hay que mirar son los **huecos entre barras**: son el embargo, y su
    anchura tiene que bastar para que ninguna sesión entre en la huella de dos
    particiones a la vez. Las marcas rojas señalan las sesiones que sí lo hacen;
    si aparece alguna, el experimento está contaminado y no vale seguir.

    La huella de una ventana no es su fecha: son las `pasado + horizonte`
    sesiones que abarca. Por eso el hueco se mide en sesiones y no en días
    naturales, y por eso las barras se dibujan extendidas hasta el final de la
    huella de su última ventana.

    Qué la invalida: si las barras se tocan, el hueco es cero y no hay embargo.
    Si hay marcas rojas, el embargo existe pero es corto. Y si `indice` no es el
    mismo con el que se construyeron las ventanas, las posiciones no significan
    nada y la figura dará un aprobado falso.
    """
    eje = eje or plt.subplots()[1]
    posicion = pd.Series(np.arange(len(indice)), index=indice)
    huella = ventanas.pasado + ventanas.horizonte

    for fila, (nombre, conjunto) in enumerate(conjuntos.items()):
        inicio = conjunto.fechas[0] - pd.Timedelta(days=0)
        fin_pos = min(int(posicion[conjunto.fechas[-1]]) + ventanas.horizonte, len(indice) - 1)
        eje.barh(
            0, indice[fin_pos] - inicio, left=inicio, height=0.55,
            color=PALETA[fila], alpha=0.85,
        )
        eje.text(
            inicio + (indice[fin_pos] - inicio) / 2, 0,
            f"{nombre}\n{len(conjunto.fechas)} ventanas",
            ha="center", va="center", fontsize=9, color="white", fontweight="bold",
        )

    contaminadas = solape[solape["sesiones_compartidas"] > 0]
    for _, fila in contaminadas.iterrows():
        eje.axvspan(fila["desde"], fila["hasta"], color=PALETA[7], alpha=0.9, lw=0)
        eje.annotate(
            f"{int(fila['sesiones_compartidas'])} sesiones en dos particiones",
            xy=(fila["desde"], 0.42), xytext=(0, 6), textcoords="offset points",
            ha="center", fontsize=8, color=PALETA[7], fontweight="bold",
        )

    total = int(solape["sesiones_compartidas"].sum())
    eje.set_title(
        titulo
        or (f"Sin solape: la huella de {huella} sesiones no cruza ninguna frontera"
            if not total else f"{total} sesiones caen en dos particiones a la vez"),
        fontsize=10, fontweight="normal",
    )
    eje.set_yticks([])
    eje.set_ylim(-0.5, 0.75)
    eje.set_xlabel(
        f"los huecos entre barras son el embargo; cada ventana abarca {huella} sesiones"
    )
    eje.grid(False, axis="y")
    return eje


def supervivencia_colas(retornos: pd.Series, titulo: str | None = None, eje=None):
    """Probabilidad de superar cada nivel de desviaciones típicas, real y normal.

    Cómo se lee: el eje vertical es logarítmico, así que la distancia vertical
    entre las dos curvas **es** el factor por el que el mercado excede a la
    normal, y ese factor crece con el nivel. Por eso la figura es una función de
    supervivencia y no un histograma con una normal superpuesta: el histograma
    pinta la masa central, que es donde ambas coinciden, y esconde la cola, que
    es lo único que se quiere enseñar.

    Qué la invalida: si las dos curvas se solapasen por encima de tres
    desviaciones típicas, el argumento central del trabajo —que un generador
    gaussiano no puede reproducir este panel— sería falso y habría que
    abandonarlo. Si la curva real quedase por debajo de la normal en el extremo,
    alguien ha recortado valores atípicos.
    """
    from scipy.stats import norm

    eje = eje or plt.subplots()[1]
    z = ((retornos - retornos.mean()) / retornos.std()).dropna()
    magnitud = np.sort(np.abs(z.to_numpy()))
    empirica = 1.0 - np.arange(len(magnitud)) / len(magnitud)
    rejilla = np.linspace(0, magnitud.max() * 1.02, 400)

    eje.semilogy(magnitud, empirica, color=PALETA[0], linewidth=2.0, label="real")
    # Discontinua y en gris: la referencia teórica se distingue de los datos por
    # el trazo, no solo por el color, para que sobreviva impresa en gris.
    eje.semilogy(
        rejilla,
        2 * norm.sf(rejilla),
        color=TINTA_SECUNDARIA,
        linestyle="--",
        linewidth=1.4,
        label="normal",
    )

    # El factor va solo en el título: repetirlo aquí llenaría de texto un panel
    # que se tiene que leer de un vistazo.
    factor = float((np.abs(z) > 4).mean()) / (2 * norm.sf(4))
    peor = z.idxmin()
    exponente = int(np.floor(-np.log10(norm.cdf(z.min()) * 252)))
    # Se marca el punto en lugar de apuntarlo con un conector largo: sobre ejes
    # log-lineales una línea guía diagonal se lee como un ajuste de ley de
    # potencias superpuesto, que es justo lo que aquí no hay.
    eje.plot(
        abs(z.min()), 1 / len(z),
        marker="o", markersize=5, markerfacecolor="white",
        markeredgecolor=PALETA[0], markeredgewidth=1.4, zorder=4,
    )
    eje.annotate(
        f"{peor.date()} · {z.min():.1f}σ\nbajo normalidad, uno cada $10^{{{exponente}}}$ años",
        xy=(abs(z.min()), 1 / len(z)),
        xytext=(-8, 20),
        textcoords="offset points",
        ha="right",
        fontsize=8,
        color=TINTA_PRIMARIA,
    )

    # Subtítulo de panel, no título: el título de la figura es el `suptitle` que
    # compone el notebook, y dos negritas del mismo peso competirían.
    eje.set_title(
        titulo or f"Colas: ×{factor:.0f} más días de |z| > 4 que bajo una normal",
        fontsize=10,
        fontweight="normal",
    )
    eje.set_xlabel("x, en desviaciones típicas")
    # El tamaño muestral va en el eje: la curva termina en P = 1/n, que es un
    # solo día, y sin `n` a la vista el extremo se lee tan sólido como el centro.
    eje.set_ylabel(f"P(|z| > x), escala log · n = {len(z)} sesiones")
    eje.set_ylim(0.5 / len(z), 1.5)
    eje.legend(loc="upper right", fontsize=8)
    return eje


def autocorrelacion_absolutos(
    retornos: pd.Series, max_lag: int = 100, titulo: str | None = None, eje=None
):
    """Autocorrelación de |r| y de r hasta el retardo indicado, con banda i.i.d.

    Cómo se lee: lo que se juzga es la **posición respecto a la banda**, no la
    forma del decaimiento; por eso los ejes son lineales y no log-log, que
    convertiría la comparación en un ejercicio de lectura. La serie de |r| debe
    quedar muy por encima de la banda durante decenas de retardos —eso es el
    agrupamiento de volatilidad— mientras que la de r debe meterse dentro salvo
    en el primer retardo.

    Qué la invalida: un gaussiano da, por el teorema de Isserlis,
    autocorrelación de |r| exactamente cero. Si esta curva estuviera dentro de
    la banda, la predicción de que los generadores gaussianos fracasarán no
    tendría base. Un primer retardo de r exactamente nulo también es sospechoso:
    el real es negativo y del orden de una décima, y el criterio para un
    generador es reproducir la magnitud, no exigir cero.
    """
    from .features import _autocorrelacion

    eje = eje or plt.subplots()[1]
    r = retornos.dropna()
    centrado = (r - r.mean()).to_numpy()
    absolutos = np.abs(r.to_numpy())
    lags = np.arange(1, max_lag + 1)

    banda = 1.96 / np.sqrt(len(r))
    # Relleno propio y borde de rejilla: pintarla del color exacto de la rejilla
    # la haría leerse como una línea de rejilla gruesa, no como un intervalo.
    eje.axhspan(
        -banda,
        banda,
        facecolor="#eceae4",
        edgecolor=REJILLA,
        linewidth=0.8,
        zorder=0,
        label=f"banda i.i.d. (±{banda:.3f})",
    )
    eje.axhline(0, color=TINTA_SECUNDARIA, linewidth=0.8, zorder=1)

    curva_abs = [_autocorrelacion(absolutos, int(l)) for l in lags]
    curva_ret = [_autocorrelacion(centrado, int(l)) for l in lags]
    eje.plot(lags, curva_abs, color=PALETA[0], linewidth=2.0, label="|r| (magnitud)")
    # Trazo continuo, no discontinuo: en el panel gemelo de esta misma figura la
    # línea discontinua significa "referencia teórica", y usarla aquí para una
    # serie real haría que `r` se leyera como una curva calculada. La separación
    # ya la dan la posición respecto a la banda y la etiqueta directa.
    eje.plot(
        lags,
        curva_ret,
        color=GRIS_CONTEXTO,
        linewidth=1.2,
        alpha=0.85,
        label="r (signo)",
    )

    # Etiqueta directa en el extremo: la identidad no depende solo del color.
    for curva, texto in ((curva_abs, "|r|"), (curva_ret, "r")):
        eje.annotate(
            texto,
            xy=(max_lag, curva[-1]),
            xytext=(6, 0),
            textcoords="offset points",
            color=TINTA_SECUNDARIA,
            fontsize=9,
            va="center",
        )

    eje.set_title(
        titulo
        or f"Memoria: la ACF de |r| sigue en {curva_abs[-1]:.2f} en el retardo {max_lag}",
        fontsize=10,
        fontweight="normal",
    )
    eje.set_xlabel("retardo, en sesiones")
    eje.set_ylabel("autocorrelación")
    eje.legend(loc="upper right", fontsize=8)
    return eje


def puntos_escala(
    dispersion: pd.Series, familias: pd.Series, titulo: str | None = None, eje=None
):
    """Dispersión de cada canal en un diagrama de puntos, con eje logarítmico.

    Cómo se lee: el eje es logarítmico porque las escalas difieren en dos
    órdenes de magnitud, y esa es exactamente la conclusión. Lo que hay que
    mirar no es un punto sino la anchura total de la nube: un modelo entrenado
    sin escalar por canal optimizaría el canal de escala mayor e ignoraría el
    resto. Justifica el escalado por canal del notebook 02.

    El color codifica la **familia** del canal, nunca su valor: dos colores para
    veinte filas. Por eso `dispersion_sectorial` aparece en color de derivada
    con la escala más pequeña del panel; es una excepción visible, y es
    información, no un defecto de la figura. El relleno del marcador duplica la
    codificación para que la figura sobreviva impresa en gris.

    Qué la invalida: el orden de las filas lo fija el AUC, no la dispersión. Si
    se reordenara por dispersión, los dos paneles dejarían de poder leerse a la
    vez y la figura perdería su razón de ser. Un canal constante tendría
    dispersión cero y desaparecería del eje logarítmico sin avisar.
    """
    eje = eje or plt.subplots()[1]
    # Las posiciones se numeran al revés en lugar de invertir el eje: con
    # `sharey` la inversión se aplicaría una vez por panel y se cancelaría,
    # dejando el orden del revés justo cuando los dos paneles se comparten.
    posiciones = np.arange(len(dispersion))[::-1]
    minimo = float(dispersion.min())

    for y, (nombre, valor) in zip(posiciones, dispersion.items()):
        derivada = familias[nombre] == "derivada"
        tono = PALETA[0] if derivada else GRIS_CONTEXTO
        eje.hlines(y, minimo * 0.7, valor, color=tono, linewidth=1.0, alpha=0.6)
        eje.plot(
            valor,
            y,
            marker="o",
            markersize=6,
            color=tono,
            markerfacecolor=tono if derivada else "white",
            markeredgewidth=1.4,
        )

    eje.set_xscale("log")
    eje.set_yticks(posiciones, list(dispersion.index), fontsize=8)
    eje.set_title(
        titulo
        or f"Escala: ×{dispersion.max() / minimo:.0f} entre el canal mayor y el menor",
        fontsize=10,
        fontweight="normal",
    )
    eje.set_xlabel("desviación típica del canal, en sus propias unidades (escala log)")
    return eje


def barras_auc(
    auc: pd.Series, familias: pd.Series, titulo: str | None = None, eje=None
):
    """AUC univariante de cada canal, con las barras creciendo desde el azar.

    Cómo se lee: las barras arrancan en 0,5 y no en cero, porque 0,5 es el azar
    y una barra que arrancara en cero haría parecer que un canal inútil aporta
    la mitad que el mejor. La longitud visible es, literalmente, lo que el canal
    aporta sobre no saber nada.

    Qué la invalida: en escala de grises los dos tonos de la paleta no se
    separan, así que la familia la lleva además una trama en las barras de
    retorno. Sin esa redundancia la leyenda mostraría dos parches idénticos y la
    figura se contradiría a sí misma. Que un retorno diario superase 0,70 sería
    señal de fuga: ninguno de un solo día debería anticipar la volatilidad del
    mes siguiente.
    """
    from matplotlib.patches import Patch

    eje = eje or plt.subplots()[1]
    posiciones = np.arange(len(auc))[::-1]
    es_derivada = [familias[n] == "derivada" for n in auc.index]
    tonos = [PALETA[0] if d else GRIS_CONTEXTO for d in es_derivada]

    barras = eje.barh(
        posiciones, auc.to_numpy() - 0.5, left=0.5, color=tonos, height=0.7
    )
    for barra, derivada in zip(barras, es_derivada):
        barra.set_hatch("" if derivada else "////")
    eje.bar_label(
        barras,
        labels=[f"{v:.2f}" for v in auc],
        fontsize=8,
        padding=3,
        color=TINTA_SECUNDARIA,
    )
    eje.axvline(0.5, color=TINTA_SECUNDARIA, linewidth=1.0)
    # El umbral que cita el título, dibujado: si no, hay que interpolar a ojo
    # entre los ticks para verificar la afirmación.
    eje.axvline(0.70, color=TINTA_SECUNDARIA, linewidth=0.8, linestyle="--", zorder=0)

    eje.set_yticks(posiciones, list(auc.index), fontsize=8)
    eje.grid(False, axis="y")
    eje.set_xlim(0.5, 1.0)
    superan = int((auc > 0.70).sum())
    # El techo de los retornos se calcula: escrito a mano quedaría desmentido por
    # la etiqueta de la propia barra en cuanto cambiaran los datos.
    techo = float(auc[[not d for d in es_derivada]].max())
    eje.set_title(
        titulo
        or f"Señal: {superan} canales superan 0,70; el mejor retorno se queda en {techo:.2f}",
        fontsize=10,
        fontweight="normal",
    )
    eje.set_xlabel("AUC univariante sobre el tramo de entrenamiento (0,5 = azar)")
    # La leyenda de familia va aquí y no en el panel de escala: en aquel las
    # líneas guía cruzan todo el ancho y no queda hueco limpio; aquí las barras
    # cortas de la parte baja dejan la esquina inferior derecha vacía.
    eje.legend(
        handles=[
            Patch(facecolor=PALETA[0], label="derivada"),
            Patch(facecolor=GRIS_CONTEXTO, hatch="////", label="retorno"),
        ],
        loc="lower right",
        fontsize=8,
    )
    return eje


def carril_bloques(
    estres: pd.Series,
    episodios: pd.DataFrame,
    ventanas,
    particiones,
    titulo: str | None = None,
    eje=None,
):
    """Carril temporal partido en bloques disjuntos, teñidos por estrés.

    Cómo se lee: cada celda es un bloque que no comparte ni una sesión con sus
    vecinas, es decir una observación verdaderamente independiente. Contarlas es
    contar el tamaño muestral real del experimento; el número de ventanas es
    unas ochenta veces mayor y no significa nada equivalente. Las celdas oscuras
    son las únicas que contienen crisis, y son un puñado.

    Qué la invalida: si al mirar la figura se concluyera que hay crisis
    repartidas por todo el periodo, el argumento de que los eventos de estrés
    son escasos quedaría desmentido. Y si el carril tuviera huecos en el eje
    temporal, el índice no sería contiguo y las ventanas del notebook 02
    estarían empalmando días no consecutivos en silencio.
    """
    eje = eje or plt.subplots()[1]
    tamano = ventanas.pasado + ventanas.horizonte
    n_bloques = len(estres) // tamano
    # Cuatro niveles discretos en vez de una escala continua: así la figura se
    # lee sin barra de color, que en un carril de menos de tres pulgadas no cabe.
    # El nivel sin estrés va en blanco, no en el gris de la rejilla: contra ese
    # gris el primer tono de la rampa tiene un contraste de 1,2 y las celdas
    # ligeramente tensas —que son la mayoría de las teñidas— desaparecían al
    # imprimir.
    tonos = ["#ffffff"] + rampa_secuencial(3)

    for k in range(n_bloques):
        tramo = estres.iloc[k * tamano : (k + 1) * tamano]
        fraccion = float(tramo.mean())
        nivel = 0 if fraccion == 0 else 1 if fraccion <= 0.10 else 2 if fraccion <= 0.30 else 3
        # `facecolor` y no `color`: este último sobrescribe el borde, y el borde
        # es lo que separa una celda de la siguiente y permite contarlas —que es
        # el gesto que hace el argumento—. Por eso el nivel sin estrés va en
        # blanco pero perfilado: sin el perfil las celdas vacías se funden con el
        # fondo y el carril deja de leerse como una cuenta.
        # `ymax` reserva la franja superior para los rótulos de episodio, que
        # dentro del carril caerían sobre las celdas más oscuras.
        eje.axvspan(
            tramo.index[0],
            tramo.index[-1],
            ymin=0,
            ymax=0.72,
            facecolor=tonos[nivel],
            edgecolor=REJILLA,
            linewidth=0.5,
        )

    # Los once episodios se marcan con una raya sobre el carril y solo se rotulan
    # los sustanciales: si no, el título afirma once y el lector cuenta tres.
    eje.plot(
        episodios["inicio"],
        [0.78] * len(episodios),
        marker="|",
        markersize=7,
        color=TINTA_SECUNDARIA,
        linestyle="none",
    )
    # Solo se rotulan los episodios sustanciales; los once quedan marcados por la
    # raya, para que el recuento del título se pueda verificar contándolas.
    for _, fila in episodios[
        episodios["sesiones_estres"] > ventanas.horizonte
    ].iterrows():
        eje.annotate(
            str(fila["inicio"].year),
            xy=(fila["inicio"], 0.84),
            ha="center",
            fontsize=8,
            color=TINTA_PRIMARIA,
        )

    # Se marcan las fronteras entre particiones, no el embargo: el embargo se
    # mide en sesiones y su suficiencia no se decide en este notebook.
    for fecha, etiqueta in (
        (particiones.train_hasta, "fin de train"),
        (particiones.val_hasta, "fin de validación"),
    ):
        eje.axvline(
            pd.Timestamp(fecha),
            ymax=0.72,
            color=TINTA_PRIMARIA,
            linestyle=":",
            linewidth=1.2,
        )
        eje.annotate(
            etiqueta,
            xy=(pd.Timestamp(fecha), 0.06),
            xytext=(4, 0),
            textcoords="offset points",
            fontsize=8,
            color=TINTA_SECUNDARIA,
            bbox=dict(
                boxstyle="round,pad=0.15",
                facecolor="white",
                edgecolor="none",
                alpha=0.85,
            ),
        )

    from matplotlib.patches import Patch

    nominales = len(estres) - tamano + 1
    eje.set_title(
        titulo
        or f"{nominales:,}".replace(",", ".")
        + f" ventanas son {n_bloques} bloques independientes "
        f"y {len(episodios)} episodios de estrés",
        fontsize=13,
    )
    eje.set_xlabel(
        f"cada celda es un bloque disjunto de {tamano} sesiones; "
        f"las {len(estres) - n_bloques * tamano} finales no completan bloque"
    )
    # Cuatro niveles discretos piden cuatro parches, no una barra de color: en un
    # carril de menos de tres pulgadas la barra no cabe, la leyenda sí.
    eje.legend(
        handles=[
            Patch(facecolor=t, edgecolor=REJILLA, label=l)
            for t, l in zip(tonos, ["sin estrés", "hasta 10 %", "hasta 30 %", "más del 30 %"])
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.32),
        ncols=4,
        fontsize=8,
        title="fracción de sesiones del bloque en estrés",
        title_fontsize=8,
    )
    eje.set_yticks([])
    eje.set_ylim(0, 1)
    eje.set_xlim(estres.index[0], estres.index[n_bloques * tamano - 1])
    eje.grid(False)
    return eje


# ─────────────────────────────────────────────────────────────────────────────
# Modelo downstream: contra qué se compite, qué arquitectura y si convergió
#
# Las figuras del notebook 03. Van juntas porque se leen en cadena:
# `lineas_base` fija el listón antes de entrenar nada, `busqueda_arquitectura`
# justifica la arquitectura que después se congela, `convergencia_downstream`
# enseña que ese entrenamiento paró donde debía y `comparativa_metricas`
# contrasta el resultado con la alternativa gratuita al desbalance. `confusion`,
# más abajo, cierra la cadena mirando la clase que decide el trabajo.
#
# `curva_convergencia` —la sección siguiente— no sirve para este modelo: pinta
# todas las columnas numéricas del historial sobre un mismo eje, así que en
# clasificación mezclaría `loss` (~1,0) con `accuracy` (~0,8) y en regresión
# `mse` con `mae`. Está bien para los generadores, que registran varias pérdidas
# comparables, y por eso se conserva intacta en vez de arreglarla.
# ─────────────────────────────────────────────────────────────────────────────


def lineas_base(
    tabla: pd.DataFrame,
    metrica: str,
    referencia: float | None = None,
    etiqueta_referencia: str = "modelo entrenado",
    titulo: str | None = None,
    eje=None,
):
    """Las referencias triviales de la tarea, ordenadas por la métrica.

    Cómo se lee: cada barra es un predictor que no aprende nada, y la longitud
    es lo que saca igualmente. La mejor va **arriba**, con el mismo criterio que
    `barras_auc`: lo primero que se lee es la referencia más difícil, que es la
    que decide si el modelo entrenado sirve para algo. Un clasificador que no
    llegue a la barra de `persistencia_causal` —en color, y es la única que lo
    lleva— no está prediciendo el futuro, está leyendo el presente.

    Las barras con trama usan información futura: decodifican el régimen de hoy
    con la serie entera, incluido lo que pasó después, así que salen mejor y no
    son alcanzables por ningún modelo causal. Van con trama y no solo con color
    porque presentarlas como el listón regalaría puntos a cualquier modelo, y ese
    error no puede depender de que la figura se imprima en color.

    Qué la invalida: la figura da por hecho que **más es mejor**. Con una métrica
    de las que se minimizan el orden queda del revés y la barra a batir deja de
    ser la más larga, sin que nada avise. También la invalida que falte la fila
    `persistencia_causal`: entonces el título afirma el máximo de la tabla, que
    puede ser el de la referencia con información futura.

    Parameters
    ----------
    tabla
        Una fila por línea base, las métricas en columnas. Es lo que devuelve
        `evaluacion.lineas_base`, incluida la columna booleana ``usa_futuro``.
        Si esa columna no está, ninguna barra se marca como no alcanzable.
    metrica
        Columna a dibujar. Debe ser de las que se maximizan.
    referencia
        Valor del modelo ya entrenado, que se dibuja como vertical con su
        etiqueta. ``None`` no dibuja ninguna, que es lo correcto la primera vez
        que se muestra la figura: aún no hay modelo con el que compararse.
    etiqueta_referencia
        Rótulo de esa vertical.
    titulo
        Título del panel. Afirma la conclusión con su cifra, no nombra el eje.
    eje
        Eje de matplotlib. Si es ``None`` se crea uno.

    Returns
    -------
    matplotlib.axes.Axes
    """
    from matplotlib.patches import Patch

    eje = eje or plt.subplots()[1]
    orden = tabla.sort_values(metrica, ascending=False)
    # Posiciones al revés en lugar de invertir el eje, por el mismo motivo que
    # en `puntos_escala`: así la figura sigue siendo componible con `sharey`.
    posiciones = np.arange(len(orden))[::-1]

    if "usa_futuro" in orden.columns:
        usa_futuro = orden["usa_futuro"].astype(bool)
    else:
        usa_futuro = pd.Series(False, index=orden.index)

    es_objetivo = [n == "persistencia_causal" for n in orden.index]
    tonos = [PALETA[0] if o else GRIS_CONTEXTO for o in es_objetivo]

    barras = eje.barh(posiciones, orden[metrica].to_numpy(), color=tonos, height=0.7)
    for barra, nombre, objetivo in zip(barras, orden.index, es_objetivo):
        # La trama es el canal que sobrevive al gris; el color solo la duplica.
        barra.set_hatch("////" if usa_futuro[nombre] else "")
        if objetivo:
            # Impreso en gris, el azul de la paleta y el gris de contexto caen
            # casi en la misma luminancia. El perfil oscuro es lo que mantiene
            # separada la barra a batir, y va también en su parche de leyenda:
            # dos parches idénticos contradirían a la propia figura.
            barra.set_edgecolor(TINTA_PRIMARIA)
            barra.set_linewidth(1.5)

    etiquetas = eje.bar_label(
        barras,
        labels=[f"{v:.3f}" for v in orden[metrica]],
        fontsize=8,
        padding=3,
        color=TINTA_SECUNDARIA,
    )
    for etiqueta, objetivo in zip(etiquetas, es_objetivo):
        if objetivo:
            etiqueta.set_color(TINTA_PRIMARIA)
            etiqueta.set_fontweight("bold")

    if referencia is not None:
        # `zorder=2` deja la vertical por encima de las barras y por debajo de
        # las etiquetas de valor, que si no quedarían tachadas por la línea justo
        # en la barra que más importa.
        eje.axvline(referencia, color=TINTA_PRIMARIA, linestyle="--", linewidth=1.3, zorder=2)
        for etiqueta in etiquetas:
            # Y un respaldo blanco bajo cada cifra, que si no la vertical se lee
            # como un tachado justo sobre el valor que hay que comparar con ella.
            etiqueta.set_bbox(
                dict(boxstyle="round,pad=0.12", facecolor="white", edgecolor="none", alpha=0.85)
            )
        # El rótulo va en la franja que se reserva encima de la barra más larga.
        # Abajo chocaría con la leyenda, y sobre las barras taparía un valor.
        eje.annotate(
            f"{etiqueta_referencia} · {referencia:.3f}",
            xy=(referencia, 0.985),
            xycoords=eje.get_xaxis_transform(),
            xytext=(-6, 0),
            textcoords="offset points",
            ha="right",
            va="top",
            fontsize=8,
            color=TINTA_PRIMARIA,
        )

    eje.set_yticks(posiciones, [n.replace("_", " ") for n in orden.index], fontsize=8)
    for rotulo, objetivo in zip(eje.get_yticklabels(), es_objetivo):
        if objetivo:
            rotulo.set_fontweight("bold")
            rotulo.set_color(TINTA_PRIMARIA)

    legible = metrica.replace("_", " ")
    # El listón se calcula: escrito a mano quedaría desmentido por la etiqueta de
    # la propia barra en cuanto cambiara el conjunto de evaluación.
    if "persistencia_causal" in orden.index:
        liston = float(orden.loc["persistencia_causal", metrica])
        quien = "la persistencia causal"
    else:
        # Sin esa fila el título no puede nombrarla: afirmaría una barra que no
        # está dibujada, que es la forma más rápida de que una figura mienta.
        liston = float(orden[metrica].max())
        quien = str(orden[metrica].idxmax()).replace("_", " ")
    if "persistencia_viterbi" in orden.index:
        con_futuro = float(orden.loc["persistencia_viterbi", metrica])
        cola = f" · Viterbi llega a {con_futuro:.3f} mirando al futuro"
    else:
        cola = ""

    eje.set_title(
        titulo or f"El listón es {liston:.3f} de {legible}, {quien}{cola}",
        fontsize=10,
        fontweight="normal",
    )
    eje.set_xlabel(f"{legible} sobre el conjunto evaluado (0 a 1) · más alto es mejor")
    tope = max(float(orden[metrica].max()), float(referencia or 0.0))
    eje.set_xlim(0, tope * 1.18)
    # Franja libre sobre la barra más larga: es donde cabe el rótulo de la
    # vertical de referencia sin montarse ni sobre una barra ni sobre la leyenda.
    eje.set_ylim(-0.6, len(orden) - 0.5 + 0.55)
    eje.grid(False, axis="y")
    eje.legend(
        handles=[
            Patch(
                facecolor=PALETA[0], edgecolor=TINTA_PRIMARIA, linewidth=1.5,
                label="la barra a batir",
            ),
            Patch(facecolor=GRIS_CONTEXTO, label="referencia trivial"),
            Patch(facecolor=GRIS_CONTEXTO, hatch="////", label="usa el futuro · no alcanzable"),
        ],
        loc="lower right",
        fontsize=8,
        # Fondo blanco, que no marco: la vertical de referencia puede caer justo
        # sobre la leyenda, y el mismo recurso que usan las anotaciones de
        # `carril_bloques` la despeja sin añadir un recuadro a la figura.
        frameon=True,
        facecolor="white",
        edgecolor="none",
        framealpha=0.9,
    )
    return eje


def _columna_resumen(tabla: pd.DataFrame, nombre: str) -> str | None:
    """Localiza una columna que la agregación pudo renombrar a ``<nombre>_media``.

    `downstream.resumen_busqueda` agrega por candidata y sufija cada columna con
    ``_media`` y ``_desv``. Aceptar los dos nombres evita que el notebook tenga
    que saber cuál de las dos formas trae la tabla que está pasando.
    """
    for candidata in (nombre, f"{nombre}_media"):
        if candidata in tabla.columns:
            return candidata
    return None


def busqueda_arquitectura(
    resumen: pd.DataFrame,
    metrica: str,
    metrica_error: str | None = None,
    coste: str = "parametros",
    ganadora: str | None = None,
    titulo: str | None = None,
    eje=None,
):
    """Las candidatas de arquitectura frente a su coste, en dos paneles.

    Cómo se lee: los dos paneles comparten la lista de candidatas, así que cada
    fila se lee de izquierda a derecha como "cuánto da" y "cuánto cuesta". El
    segundo panel no es un adorno: la elección tiene una restricción de coste
    —el barrido del notebook 12 son varios cientos de entrenamientos en CPU— y
    sin él no hay forma de defender que la ganadora no se eligió por ser la más
    cara, ni de justificar haber descartado a una que puntuaba igual. El coste va
    en escala logarítmica porque entre la candidata lineal y la ancha hay dos
    órdenes de magnitud y en escala lineal todas menos una se pegarían al eje.

    El orden de las filas es el que trae la tabla, no se recalcula:
    `downstream.resumen_busqueda` ya la entrega de mejor a peor y es la única que
    sabe si la métrica se maximiza o se minimiza. La ganadora va en color, con su
    rótulo en negrita, para que se distinga también impresa en gris.

    Qué la invalida: **las barras de error**. Dos candidatas separadas por menos
    que la dispersión entre semillas no están separadas por nada, y sin las
    barras la figura invita justo a la conclusión contraria; por eso, cuando la
    tabla no trae la columna de dispersión, el título lo dice en vez de callarlo.
    También la invalida mezclar en una misma tabla dos tareas o dos presupuestos:
    las medias serían de magnitudes distintas y el orden no significaría nada.

    Parameters
    ----------
    resumen
        Una fila por candidata. Salida de `downstream.resumen_busqueda`.
    metrica
        Métrica de selección. Se acepta tanto ``balanced_accuracy`` como
        ``balanced_accuracy_media``.
    metrica_error
        Columna con la desviación entre semillas. ``None`` la deduce cambiando
        el sufijo ``_media`` por ``_desv``, y si no existe dibuja sin barras.
    coste
        ``parametros`` o ``segundos``. Es el eje del panel derecho.
    ganadora
        Candidata a destacar. ``None`` toma la primera fila, que es la mejor si
        la tabla viene de `resumen_busqueda`.
    titulo
        Subtítulo del panel izquierdo. El del derecho siempre se calcula.
    eje
        Par de ejes ya creados, con la vertical compartida. Si es ``None`` se
        crean.

    Returns
    -------
    tuple
        Los dos ejes, ``(métrica, coste)``.
    """
    from matplotlib.patches import Patch

    col_metrica = _columna_resumen(resumen, metrica)
    col_coste = _columna_resumen(resumen, coste)
    if col_metrica is None or col_coste is None:
        falta = metrica if col_metrica is None else coste
        raise ValueError(
            f"La tabla no tiene columna {falta!r} ni {falta + '_media'!r}. "
            f"Trae {list(resumen.columns)}. Comprueba que procede de "
            "`downstream.resumen_busqueda` y no de un CSV recortado."
        )

    if metrica_error is None:
        deducida = (
            col_metrica[: -len("_media")] + "_desv"
            if col_metrica.endswith("_media")
            else f"{col_metrica}_desv"
        )
        metrica_error = deducida if deducida in resumen.columns else None
    dispersion = resumen[metrica_error] if metrica_error else None

    if len(resumen) < 2:
        raise ValueError(
            "Con una sola candidata no hay búsqueda que enseñar: la figura compara "
            "la ganadora con su rival más cercana y no hay ninguna."
        )

    ganadora = str(resumen.index[0]) if ganadora is None else ganadora
    if ganadora not in resumen.index:
        raise ValueError(
            f"La candidata {ganadora!r} no está en la tabla ({list(resumen.index)}). "
            "Destacar una fila que no existe dejaría la figura sin ganadora y sin aviso."
        )

    if eje is None:
        # Ancha y baja: seis candidatas de nombre largo piden filas, no columnas.
        eje = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)[1]
    eje_metrica, eje_coste = eje

    valores = resumen[col_metrica]
    costes = resumen[col_coste]
    posiciones = np.arange(len(resumen))[::-1]
    es_ganadora = [n == ganadora for n in resumen.index]
    tonos = [PALETA[0] if g else GRIS_CONTEXTO for g in es_ganadora]

    barras = eje_metrica.barh(
        posiciones,
        valores.to_numpy(),
        height=0.7,
        color=tonos,
        xerr=None if dispersion is None else dispersion.to_numpy(),
        # La barra de error va en tinta primaria y no en el color de la barra:
        # es una magnitud distinta y tiene que verse sobre las dos.
        ecolor=TINTA_PRIMARIA,
        capsize=3,
    )
    for barra, gana in zip(barras, es_ganadora):
        if gana:
            # Perfil oscuro además del color, por lo mismo que en `lineas_base`:
            # en gris, el azul de la paleta y el gris de contexto se confunden.
            barra.set_edgecolor(TINTA_PRIMARIA)
            barra.set_linewidth(1.5)
    eje_metrica.bar_label(
        barras,
        labels=[f"{v:.3f}" for v in valores],
        fontsize=8,
        padding=3,
        color=TINTA_SECUNDARIA,
    )

    eje_metrica.set_yticks(
        posiciones, [n.replace("_", " ") for n in resumen.index], fontsize=8
    )
    for rotulo, gana in zip(eje_metrica.get_yticklabels(), es_ganadora):
        if gana:
            rotulo.set_fontweight("bold")
            rotulo.set_color(TINTA_PRIMARIA)

    extremo = float((valores + (0 if dispersion is None else dispersion)).max())
    # Holgura a la derecha: ahí van las etiquetas de valor y, debajo, la leyenda.
    eje_metrica.set_xlim(0, extremo * 1.28)
    eje_metrica.grid(False, axis="y")

    # Panel derecho: mismo idioma que `puntos_escala` —línea guía y punto— y no
    # barras, para que no se lea como una segunda puntuación.
    minimo = float(costes.min())
    for y, (nombre, valor), gana in zip(posiciones, costes.items(), es_ganadora):
        tono = PALETA[0] if gana else GRIS_CONTEXTO
        eje_coste.hlines(y, minimo * 0.7, valor, color=tono, linewidth=1.0, alpha=0.6)
        eje_coste.plot(
            valor, y, marker="o", markersize=6, color=tono,
            markerfacecolor=tono if gana else "white", markeredgewidth=1.4,
        )
        eje_coste.annotate(
            _cifra_coste(float(valor)),
            xy=(valor, y), xytext=(8, 0), textcoords="offset points",
            va="center", fontsize=8, color=TINTA_SECUNDARIA,
        )
    eje_coste.set_xscale("log")
    eje_coste.set_xlim(minimo * 0.55, float(costes.max()) * 4.0)
    eje_coste.grid(False, axis="y")

    legible = metrica.replace("_media", "").replace("_", " ")
    nombre_ganadora = ganadora.replace("_", " ")
    resto = valores.drop(ganadora)
    rival = (resto - float(valores[ganadora])).abs().idxmin()
    hueco = abs(float(valores[ganadora]) - float(resto[rival]))
    ruido = (
        None if dispersion is None
        else float(max(dispersion[ganadora], dispersion[rival]))
    )
    if ruido is None:
        conclusion = (
            f"{nombre_ganadora} dista {hueco:.3f} de {rival.replace('_', ' ')} · "
            "sin dispersión no se puede juzgar"
        )
    elif hueco > ruido:
        # El signo importa: `elegir` puede destacar una candidata que no encabeza
        # la métrica, porque el desempate por `recall_crisis` o por coste la
        # adelanta. Escribir "gana por" sobre un hueco en valor absoluto haría
        # que el título afirmara justo lo contrario de lo que enseña la figura.
        delante = float(valores[ganadora]) >= float(resto[rival])
        conclusion = (
            f"{nombre_ganadora} {'gana por' if delante else 'cede'} {hueco:.3f} "
            f"{'a' if delante else 'frente a'} {rival.replace('_', ' ')} · "
            f"más que el ruido entre semillas ({ruido:.3f})"
        )
    else:
        conclusion = (
            f"{nombre_ganadora} y {rival.replace('_', ' ')} distan {hueco:.3f} · "
            f"menos que el ruido entre semillas ({ruido:.3f})"
        )

    if "n_semillas" in resumen.columns and resumen["n_semillas"].nunique() == 1:
        sufijo = f" entre {int(resumen['n_semillas'].iloc[0])} semillas"
    else:
        sufijo = " entre semillas"

    # Cada subtítulo se ancla al borde exterior de su panel en vez de centrarse.
    # Centradas, dos conclusiones con cifras se encuentran en el hueco entre
    # paneles —es el choque contra el que avisa `guardar`— y ninguna de las dos
    # se puede acortar sin perder la cifra que la sostiene.
    eje_metrica.set_title(titulo or conclusion, fontsize=10, fontweight="normal", loc="left")
    eje_metrica.set_xlabel(
        f"{legible} · media{sufijo}"
        + (" y ±1 desviación" if dispersion is not None else ", sin dispersión en la tabla")
    )
    eje_metrica.legend(
        handles=[
            Patch(
                facecolor=PALETA[0], edgecolor=TINTA_PRIMARIA, linewidth=1.5,
                label="ganadora",
            ),
            Patch(facecolor=GRIS_CONTEXTO, label="descartadas"),
        ],
        loc="lower right",
        fontsize=8,
        # Fondo blanco por el mismo motivo que en `lineas_base`: la esquina que
        # queda libre depende de lo larga que sea la barra peor, y una leyenda
        # sin fondo se lee encima de ella en cuanto la búsqueda va apretada.
        frameon=True,
        facecolor="white",
        edgecolor="none",
        framealpha=0.9,
    )

    unidades = {
        "parametros": "parámetros del modelo",
        "segundos": "segundos por entrenamiento",
    }
    coste_ganadora = float(costes[ganadora])
    if coste_ganadora == minimo:
        conclusion_coste = (
            f"Y es la más barata: ×{float(costes.max()) / minimo:.0f} bajo la más cara"
        )
    elif coste_ganadora == float(costes.max()):
        conclusion_coste = f"Pero es la más cara: ×{coste_ganadora / minimo:.0f} sobre la más barata"
    else:
        conclusion_coste = (
            f"Coste: ×{coste_ganadora / minimo:.0f} sobre la más barata, "
            f"×{float(costes.max()) / coste_ganadora:.0f} bajo la más cara"
        )
    eje_coste.set_title(conclusion_coste, fontsize=10, fontweight="normal", loc="right")
    eje_coste.set_xlabel(f"{unidades.get(coste, coste.replace('_', ' '))} (escala log)")
    return eje_metrica, eje_coste


def _cifra_coste(valor: float) -> str:
    """Coste en formato compacto. Un recuento de parámetros con seis dígitos
    ocupa más que el propio punto que rotula, y en un panel estrecho el número
    largo empuja el eje sin añadir precisión útil."""
    if valor >= 1e6:
        return f"{valor / 1e6:.1f} M"
    if valor >= 1e4:
        return f"{valor / 1e3:.0f} k"
    if valor >= 1e3:
        return f"{valor / 1e3:.1f} k"
    return f"{valor:.0f}"


def convergencia_downstream(historial: pd.DataFrame, titulo: str | None = None, eje=None):
    """Pérdida de entrenamiento y de validación de un modelo downstream.

    Cómo se lee: dos curvas de la **misma** magnitud, así que la distancia
    vertical entre ellas es literalmente el sobreajuste. La vertical marca la
    época de mejor validación, que es la que la parada temprana restaura: el
    modelo que se evalúa después es el de esa época, no el de la última, y el
    tramo sombreado a su derecha no llegó a ninguna parte. Sin esa marca no se
    puede juzgar lo único que la figura tiene que decidir —si el presupuesto de
    épocas bastó—, porque una curva que aún baja al final y otra que se aplanó
    hace veinte épocas se parecen mucho de lejos.

    La métrica auxiliar del historial (`accuracy` o `mae`) no se pinta. No es un
    olvido: no comparte escala con la pérdida, y meterla en el mismo eje es lo
    que hace inservible aquí a `curva_convergencia`.

    Qué la invalida: que la mejor época sea la última o casi. Entonces la parada
    temprana no llegó a dispararse, el modelo se quedó a medias y las cifras que
    salgan de él miden el presupuesto, no la arquitectura. En la tarea de
    volatilidad conviene además pasar el eje a escala logarítmica desde fuera
    (``eje.set_yscale("log")``): el MSE cae dos órdenes de magnitud en las
    primeras épocas y en escala lineal aplasta contra el eje todo lo demás.

    Parameters
    ----------
    historial
        Columnas ``epoca``, ``loss`` y ``val_loss``, y opcionalmente una métrica
        auxiliar con su pareja de validación. Es lo que escribe
        `downstream.guardar_historial`.
    titulo
        Subtítulo del panel. Afirma la conclusión con su cifra, no nombra el eje.
    eje
        Eje de matplotlib. Si es ``None`` se crea uno.

    Returns
    -------
    matplotlib.axes.Axes
    """
    faltan = [c for c in ("epoca", "loss", "val_loss") if c not in historial.columns]
    if faltan:
        raise ValueError(
            f"Al historial le faltan las columnas {faltan}; trae {list(historial.columns)}. "
            "Sin `val_loss` no hay época que marcar y la figura no puede juzgar la "
            "convergencia, que es lo único para lo que existe."
        )

    eje = eje or plt.subplots()[1]
    epocas = historial["epoca"].to_numpy()
    entrenamiento = historial["loss"].to_numpy(dtype=float)
    validacion = historial["val_loss"].to_numpy(dtype=float)
    n = len(historial)

    # La época que se marca tiene que ser la que la parada temprana REALMENTE
    # restauró, no la del mínimo de `val_loss`. Desde que el presupuesto para por
    # `balanced_accuracy` (ver D20), las dos no coinciden: en la variante
    # reponderada el mínimo de `val_loss` cae en la época 1 y el máximo de
    # balanced accuracy en la 5. Marcar la de `val_loss` haría que la figura
    # contradijera al modelo que hay debajo, sombreando como descartado un tramo
    # que sí está en los pesos evaluados.
    # `val_mae` va en la lista y con el signo contrario: es el monitor de la tarea
    # de volatilidad, y omitirlo dejaba la corrección aplicada a la mitad de las
    # tareas, marcando en la figura de regresión una época que nadie restauró.
    MONITORES_FIGURA = {
        "val_balanced_accuracy": True, "val_recall_crisis": True,
        "val_f1_macro": True, "val_mae": False,
    }
    monitor = next((c for c in MONITORES_FIGURA if c in historial.columns), None)
    if monitor is not None:
        serie_monitor = historial[monitor].to_numpy(dtype=float)
        mejor = int(np.argmax(serie_monitor) if MONITORES_FIGURA[monitor]
                    else np.argmin(serie_monitor))
        nombre_monitor, valor_monitor = monitor, serie_monitor[mejor]
    else:
        # Posición y no etiqueta: `idxmin` devolvería el índice del DataFrame, que
        # no tiene por qué coincidir con la numeración de épocas.
        mejor = int(np.argmin(validacion))
        nombre_monitor, valor_monitor = "val_loss", validacion[mejor]
    mejor_epoca = int(epocas[mejor])

    # El tramo posterior a la mejor época se descarta al restaurar pesos. Va
    # sombreado y con entrada propia en la leyenda: es la parte de la curva que
    # no está en el modelo que luego se evalúa.
    descartado = None
    if mejor < n - 1:
        descartado = eje.axvspan(
            epocas[mejor], epocas[-1], facecolor="#eceae4", edgecolor=REJILLA,
            linewidth=0.8, zorder=0, label="descartado al restaurar pesos",
        )

    # Color y trazo a la vez: continua para train, discontinua para validación,
    # de modo que la figura siga separando las dos curvas impresa en gris.
    linea_train = eje.plot(
        epocas, entrenamiento, color=PALETA[0], linewidth=2.0, label="entrenamiento"
    )[0]
    linea_val = eje.plot(
        epocas, validacion, color=PALETA[1], linewidth=1.8, linestyle="--", label="validación"
    )[0]

    eje.axvline(mejor_epoca, color=TINTA_SECUNDARIA, linestyle=":", linewidth=1.2, zorder=1)
    eje.plot(
        mejor_epoca, validacion[mejor], marker="o", markersize=6,
        markerfacecolor="white", markeredgecolor=PALETA[1], markeredgewidth=1.6, zorder=4,
    )
    # El rótulo cambia de lado cerca del borde derecho; si no, se saldría del eje
    # justo en el caso que más importa, que es el del presupuesto corto.
    tarde = mejor > 0.7 * n
    eje.annotate(
        f"mejor {nombre_monitor}\népoca {mejor_epoca} · {valor_monitor:.3g}",
        xy=(mejor_epoca, validacion[mejor]),
        xytext=(-10 if tarde else 10, 16),
        textcoords="offset points",
        ha="right" if tarde else "left",
        fontsize=8,
        color=TINTA_PRIMARIA,
    )

    hueco = float(validacion[-1] - entrenamiento[-1])
    relativo = hueco / abs(float(entrenamiento[-1])) if entrenamiento[-1] else float("nan")
    # "Casi la última" se define como el último 10 % del recorrido: con la
    # paciencia del proyecto, un entrenamiento que convergió para a bastantes
    # más épocas de la mejor, y uno que agotó el presupuesto para justo después.
    margen = max(1, int(round(0.1 * n)))
    corto = (n - 1 - mejor) <= margen

    if corto:
        conclusion = (
            f"Presupuesto corto: la mejor validación es la época {mejor_epoca} de {n} · "
            f"hueco final {hueco:+.3g}"
        )
    else:
        conclusion = (
            f"Convergió: mejor validación en la época {mejor_epoca} de {n} · "
            f"hueco final {hueco:+.3g} ({relativo * 100:+.0f} % sobre train)"
        )

    auxiliares = [
        c for c in historial.columns
        if c not in ("epoca", "loss", "val_loss") and f"val_{c}" in historial.columns
    ]
    nota = (
        f" · {auxiliares[0].replace('_', ' ')} no se pinta: no comparte escala con la pérdida"
        if auxiliares else ""
    )

    eje.set_title(titulo or conclusion, fontsize=10, fontweight="normal")
    eje.set_xlabel(f"época{nota}")
    eje.set_ylabel("pérdida · misma escala en train y validación")
    eje.set_xlim(float(epocas[0]), float(epocas[-1]))
    # Handles explícitos: por orden de dibujo la franja sombreada encabezaría la
    # leyenda, y lo primero que hay que identificar son las dos curvas.
    eje.legend(
        handles=[linea_train, linea_val] + ([descartado] if descartado is not None else []),
        loc="upper right",
        fontsize=8,
    )
    return eje


def comparativa_metricas(
    tabla: pd.DataFrame, metricas, titulo: str | None = None, eje=None
):
    """Unas pocas variantes del mismo modelo sobre las mismas métricas.

    Cómo se lee: se agrupa por métrica y dentro de cada grupo hay una barra por
    variante, siempre en el mismo orden, de modo que la identidad la lleva la
    posición dentro del grupo y no solo el color. La comparación que importa es
    **vertical dentro del grupo**, nunca entre grupos: dos métricas distintas no
    son la misma magnitud aunque compartan el rango. Las líneas base van en el
    gris de contexto y con trama, con el mismo criterio que
    ``COLOR_GENERADOR["solo_real"]``: son la referencia contra la que se mide, no
    una variante más que comparar.

    El valor va escrito encima de cada barra porque son pocas barras y la
    decisión que cuelga de esta figura —si reponderar la pérdida iguala a lo que
    aportan los sintéticos (D15)— se juega en la tercera cifra, que a ojo no se
    lee de una altura de barra.

    Qué la invalida: meter en `metricas` columnas con rangos distintos. El eje es
    uno solo, así que un MAE junto a un f1 macro haría que la barra corta
    pareciera peor cuando es mejor. Todas las métricas de la lista tienen que
    vivir en la misma escala y en el mismo sentido.

    Parameters
    ----------
    tabla
        Una fila por variante, las métricas en columnas. El nombre de la fila es
        lo que decide el color: las que empiezan por ``persistencia`` y
        ``solo_real`` se dibujan como referencia.
    metricas
        Columnas a mostrar, en el orden en que se quieren leer. La primera es la
        que decide el título.
    titulo
        Título del panel. Afirma la conclusión con su cifra.
    eje
        Eje de matplotlib. Si es ``None`` se crea uno.

    Returns
    -------
    matplotlib.axes.Axes
    """
    eje = eje or plt.subplots()[1]
    metricas = list(metricas)
    variantes = list(tabla.index)
    posiciones = np.arange(len(metricas))
    ancho = 0.8 / max(len(variantes), 1)

    def es_referencia(nombre: str) -> bool:
        return str(nombre).startswith("persistencia") or str(nombre) == "solo_real"

    # El color se reparte solo entre las variantes que compiten, con un contador
    # aparte: si se usara la posición en la tabla, mover la línea base de sitio
    # repintaría a las demás, que es justo lo que la paleta fija impide.
    siguiente = 0
    for i, nombre in enumerate(variantes):
        referencia = es_referencia(nombre)
        if referencia:
            tono = GRIS_CONTEXTO
        else:
            tono = PALETA[siguiente % len(PALETA)]
            siguiente += 1

        valores = [float(tabla.loc[nombre, m]) for m in metricas]
        barras = eje.bar(
            posiciones + i * ancho,
            valores,
            width=ancho * 0.9,  # el 10 % restante es el hueco entre barras
            color=tono,
            hatch="////" if referencia else "",
            label=str(nombre).replace("_", " ") + (" (referencia)" if referencia else ""),
        )
        eje.bar_label(
            barras,
            labels=[f"{v:.3f}" for v in valores],
            fontsize=7,
            padding=2,
            color=TINTA_SECUNDARIA,
        )

    eje.set_xticks(
        posiciones + 0.4 - ancho / 2, [m.replace("_", " ") for m in metricas]
    )

    principal = metricas[0]
    legible = principal.replace("_", " ")
    modelos = [n for n in variantes if not es_referencia(n)]
    referencias = [n for n in variantes if es_referencia(n)]
    # El título se calcula sobre la primera métrica de la lista: es la que el
    # notebook pone delante y la que decide, y escribirlo a mano lo dejaría
    # desmentido por las etiquetas de las propias barras.
    if modelos:
        lider = max(modelos, key=lambda n: float(tabla.loc[n, principal]))
        valor = float(tabla.loc[lider, principal])
        if referencias:
            base = max(referencias, key=lambda n: float(tabla.loc[n, principal]))
            valor_base = float(tabla.loc[base, principal])
            verbo = "bate" if valor > valor_base else "no bate"
            conclusion = (
                f"{lider.replace('_', ' ')} {verbo} a {base.replace('_', ' ')} en "
                f"{legible}: {valor:.3f} frente a {valor_base:.3f}"
            )
        else:
            otros = [n for n in modelos if n != lider]
            segundo = (
                max(otros, key=lambda n: float(tabla.loc[n, principal])) if otros else lider
            )
            conclusion = (
                f"{lider.replace('_', ' ')} lidera {legible} con {valor:.3f} · "
                f"{valor - float(tabla.loc[segundo, principal]):+.3f} sobre "
                f"{segundo.replace('_', ' ')}"
            )
    else:
        conclusion = f"{legible} de las líneas base"

    eje.set_title(titulo or conclusion, fontsize=10, fontweight="normal")
    eje.set_xlabel("métrica · el mismo conjunto de evaluación en todas las variantes")
    eje.set_ylabel("valor de la métrica (todas en 0 a 1)")
    eje.set_ylim(0, float(tabla[metricas].to_numpy(dtype=float).max()) * 1.28)
    eje.grid(False, axis="x")
    # Hasta tres por fila: con más, la leyenda cruza el panel entero y se cuela
    # sobre las barras del grupo más alto.
    eje.legend(ncols=min(len(variantes), 3), fontsize=8, loc="upper right")
    return eje


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
    precios: pd.Series,
    regimen: pd.Series,
    titulo: str = "Regímenes detectados",
    episodios: dict[str, tuple[str, str]] | None = None,
    corte=None,
    eje=None,
):
    """Precio del índice con el fondo sombreado según el régimen estimado.

    Cómo se lee: la línea negra es el índice en escala logarítmica y el fondo es
    el estado del HMM. La calma se deja **sin sombrear** para que la tinta marque
    solo lo excepcional, y la opacidad crece con la severidad, porque sobre una
    rampa de un solo tono dos estados con la misma opacidad no se separan
    impresos. Las barras del borde superior son los episodios de referencia: no
    salen del modelo, son el patrón externo contra el que se juzga, y la lectura
    consiste en comprobar que debajo de cada barra hay banda oscura.

    La línea de puntos marca el fin del tramo de entrenamiento. Todo lo que queda
    a su derecha es inferencia con los parámetros congelados, así que acertar ahí
    vale más que acertar a la izquierda.

    Qué la invalida: bandas oscuras repartidas por todo el histórico significan
    que el estado extremo no separa crisis de corrección, y el peso de la clase
    en `regimenes.control_bloqueante` lo confirmará. Una barra de referencia sin
    banda debajo invalida el etiquetado entero.

    Parameters
    ----------
    precios
        Serie del índice, indexada por fecha. Debe cubrir el índice de `regimen`.
    regimen
        Régimen canonizado día a día, ``0 = calma ... n-1 = crisis``.
    titulo
        Título del panel. Afirma la conclusión con su cifra, no nombra el eje.
    episodios
        ``{nombre: (desde, hasta)}`` de las crisis de referencia. ``None`` no
        dibuja ninguna, que es lo correcto cuando la figura no es un control.
    corte
        Fecha de fin de train. ``None`` no dibuja la línea.
    eje
        Eje de matplotlib. Si es ``None`` se crea uno.

    Returns
    -------
    matplotlib.axes.Axes
    """
    from matplotlib.patches import Patch

    from .evaluacion import nombres_regimenes

    eje = eje or plt.subplots()[1]
    eje.plot(precios.index, precios.to_numpy(), color=TINTA_PRIMARIA, linewidth=1.1, zorder=3)

    n_estados = int(regimen.max()) + 1
    tonos = rampa_secuencial(n_estados)
    alfas = np.linspace(0.18, 0.42, n_estados)

    for _, tramo in regimen.groupby(regimen.ne(regimen.shift()).cumsum()):
        estado = int(tramo.iloc[0])
        if estado == 0:  # la calma se deja limpia, para no saturar el fondo
            continue
        eje.axvspan(
            tramo.index[0], tramo.index[-1],
            color=tonos[estado], alpha=alfas[estado], lw=0, zorder=0,
        )

    # `ymin`/`ymax` van en fracción del eje: evitan componer una transformada
    # mixta, que sobre un eje vertical logarítmico falla al dibujar.
    trans = eje.get_xaxis_transform()
    for nombre, (desde, hasta) in (episodios or {}).items():
        a, b = pd.Timestamp(desde), pd.Timestamp(hasta)
        eje.axvspan(a, b, ymin=0.955, ymax=0.99, color=TINTA_PRIMARIA, lw=0, zorder=5)
        eje.text(
            a + (b - a) / 2, 0.935, nombre, transform=trans, ha="center",
            va="top", fontsize=8, color=TINTA_PRIMARIA,
        )

    if corte is not None:
        eje.axvline(pd.Timestamp(corte), color=TINTA_SECUNDARIA, linestyle=":", linewidth=1.2, zorder=4)
        eje.text(
            pd.Timestamp(corte), 0.02, "fin de train ", transform=trans,
            ha="right", va="bottom", fontsize=8, color=TINTA_SECUNDARIA,
        )

    etiquetas = nombres_regimenes(n_estados)
    eje.legend(
        handles=[
            Patch(facecolor=tonos[k], alpha=alfas[k], label=etiquetas[k])
            for k in range(1, n_estados)
        ],
        loc="lower right",
        fontsize=8,
        ncols=n_estados - 1,
    )
    eje.set_title(titulo, fontsize=10, fontweight="normal")
    eje.set_ylabel("nivel del índice (escala log)")
    eje.set_yscale("log")
    return eje
