# ADR-001 · Elección del problema y del diseño experimental

- **Estado:** aceptado
- **Fecha:** 2026-08-18
- **Ámbito:** define el objeto del taller completo; condiciona datos, generadores y análisis
- **Decisiones operativas derivadas:** D1, D2, D3, D5, D13 en [`../DECISIONES.md`](../DECISIONES.md)

## Contexto

El enunciado del taller B5-T1 deja los datos y el problema a elección del grupo,
con una única condición: debe ser un problema financiero *que pudiera
beneficiarse de tener datos sintéticos*. Sobre él hay que entrenar tres modelos
generativos más uno simple, generar datasets con distinta proporción real/
sintético, entrenar una arquitectura fija sobre cada uno y analizar el efecto.

El material guiado del máster desarrolla un ejemplo completo: predecir el
retorno medio de los 30 días siguientes de un panel de 23 valores del S&P 500 a
partir de una ventana de 60 días.

## Problema

Reutilizar el ejemplo del material tiene tres inconvenientes serios:

1. **La señal es casi ruido.** Predecir retornos medios futuros da errores cuyas
   diferencias entre modelos son de tercer decimal. En una presentación de cinco
   minutos, un gráfico de MSE que va de 0.00031 a 0.00028 no comunica nada.
2. **La justificación de "por qué necesito sintéticos" es débil.** El argumento
   se reduce a "más datos suelen venir bien", que es genérico y difícil de
   defender frente a la pregunta obvia: *el histórico ya tiene 20.000 ventanas,
   ¿qué te falta?*
3. **No hay diferenciación.** Es la solución que el profesor ya ha mostrado.

Hacía falta un problema donde la escasez de datos fuera **estructural y
demostrable**, no una hipótesis de trabajo.

## Opciones consideradas

### A · Replicar el ejemplo del material (retorno medio futuro)

Riesgo mínimo de ejecución, cero riesgo de salirse del enunciado. A cambio,
arrastra los tres inconvenientes anteriores y no aporta nada propio.

### B · Regresión de volatilidad realizada futura

Problema real, señal mucho más fuerte que la de los retornos (la volatilidad es
persistente y predecible) y métricas interpretables. Pero **no hay escasez**: la
volatilidad se define en todos los puntos del histórico y no hay clases raras.
La justificación de los sintéticos vuelve a ser genérica.

### C · Predicción del régimen de mercado a 21 días

Clasificación en tres clases del régimen que dominará el próximo mes. La clase
de crisis es minoritaria por la naturaleza del fenómeno, no por un defecto del
muestreo: en 2003-2026 hay del orden de diez episodios de estrés independientes.

Ventajas: la justificación de los sintéticos es autoevidente y cuantificable;
permite generadores condicionales, que es un nivel por encima de generar la
distribución global; las métricas (recall de crisis, matriz de confusión) se
leen de un vistazo en un slide; y conecta con el TFM en curso de uno de los
integrantes, lo que da acceso a un criterio de etiquetado ya validado en vez de
uno improvisado.

Riesgos: el etiquetado no es observable y hay que estimarlo, lo que añade una
pieza que puede fallar; y el problema es genuinamente difícil, así que las
métricas absolutas serán modestas.

### D · Régimen como problema principal y volatilidad como control

La opción C más la B, resueltas sobre la misma ventana de entrada, con la misma
troncal y compartiendo el mismo banco de datos sintéticos.

## Decisión

Se adopta la **opción D**.

El régimen es el problema principal porque es donde la hipótesis del taller —los
datos sintéticos compensan la escasez— se puede poner a prueba de verdad. La
volatilidad se conserva como **control experimental**: es una tarea sobre los
mismos datos y sin desbalance de clases.

La lógica del control es la que da valor al diseño:

| Resultado | Interpretación |
|---|---|
| Mejora en régimen, no en volatilidad | El mecanismo es el rebalanceo de la clase rara |
| Mejora en ambas | El mecanismo es enriquecimiento general de la distribución |
| No mejora en ninguna | Los generadores no capturan estructura útil a esta escala |

Los tres desenlaces son publicables y defendibles. Un diseño de una sola tarea
solo distingue entre "funcionó" y "no funcionó", sin poder explicar por qué.

El coste marginal de la segunda tarea es bajo: comparte descarga, features,
ventanas, splits, generadores y bancos de muestras. Solo duplica el barrido de
entrenamiento downstream, que está cacheado en CSV y es reanudable.

## Consecuencias

**Aceptadas.**

- Aparece una pieza que el ejemplo guiado no tiene: el etiquetado de regímenes.
  Se mitiga reutilizando el criterio de canonicalización del TFM y sometiendo el
  etiquetado a cuatro controles de aceptación antes de continuar.
- El barrido dobla de tamaño (dos tareas). Se compensa haciendo el notebook 12
  reanudable y persistiendo cada resultado en cuanto se obtiene.
- Las métricas absolutas serán modestas. Se reportan siempre contra la línea
  base de persistencia (D21), que es exigente.

**Que hay que vigilar.**

- El régimen se estima con un HMM ajustado solo en train. Si ese ajuste no
  converge a algo económicamente interpretable —las bandas de crisis deben caer
  sobre 2008, 2020 y 2022—, el resto del trabajo carece de sentido. La
  verificación está en el notebook 01 y es bloqueante.
- La proporción de la clase de crisis es una **expectativa a medir**, no un dato
  conocido. Se estimó en torno al 10 % a partir de dos anclajes: el HMM de dos
  estados del trabajo previo da 25,3 % para la clase de riesgo, y el 17,7 % de
  las sesiones de 2003-2026 caen dentro de un episodio pico-suelo catalogado.
  Con tres estados el extremo debe quedar por debajo de ambos. Si sale muy fuera
  del rango [3 %, 20 %], hay que revisar el número de estados.

## Notas

Se descartó explícitamente usar el panel de 23 tickers del ejemplo guiado
(ver D3): con 1.380 dimensiones los generadores densos se degradan, y es
precisamente donde el ejemplo del material muestra sus límites.
