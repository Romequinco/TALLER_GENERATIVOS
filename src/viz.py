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
