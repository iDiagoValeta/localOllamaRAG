# Lenguaje visual de MonkeyGrab

Este documento describe el sistema visual que ejecuta la interfaz web. No es un
plan de migración: describe lo que la interfaz es después del cambio y por qué.

## Objetivos

MonkeyGrab dejará de ser una aplicación aislada: su interfaz va a vivir dentro
de una aplicación mayor. Una interfaz que ya habla el mismo idioma visual que su
anfitrión se integra moviendo componentes; una que no, se integra
reescribiéndolos. Ese es el motivo del cambio, y también el criterio para
resolver cualquier duda que aparezca al aplicarlo: entre dos opciones, gana la
que hace el componente más portable.

El sistema anterior era coherente pero opuesto en casi todo: tipografía sans,
un naranja quemado como acento, todas las esquinas a cero y las respuestas del
modelo dentro de una caja. Ninguna de esas decisiones era mala; simplemente no
es el idioma al que vamos.

## El mecanismo, antes que los valores

El cambio de fondo no es la paleta, es cómo se consume.

Hasta ahora los colores semánticos vivían en `:root` como variables planas
(`--bg`, `--surface`, `--text`) y se consumían con valores arbitrarios:
`bg-[var(--surface)]`, `text-[var(--text-muted)]`.

Pasan a declararse dentro de `@theme` con el prefijo que Tailwind reconoce como
color (`--color-surface`, `--color-ink`, `--color-edge`), lo que genera las
utilidades correspondientes. El código pasa a decir `bg-surface`,
`text-ink-muted`, `border-edge`.

Tres consecuencias, en orden de importancia:

1. Un componente escrito con clases semánticas se mueve a cualquier proyecto que
   declare los mismos tokens y sigue pintando bien. Uno escrito con
   `bg-[var(--surface)]` arrastra la definición de la variable.
2. Las clases se pueden buscar. `grep bg-surface` encuentra todo lo que pinta
   sobre fondo; `bg-[var(--surface)]` es una cadena arbitraria que nadie va a
   escribir igual dos veces.
3. Los estados de Tailwind (`hover:`, `dark:`, `md:`) se componen sin sintaxis
   de escape.

## Tokens

Nombres semánticos, no descriptivos: `ink` es el color del texto, no "gris
claro". Un token descriptivo miente en cuanto cambia el tema.

| Token | Oscuro (por defecto) | Claro |
|---|---|---|
| `surface` | `#1a1917` | `#c7c5c0` |
| `surface-raised` | `#21201d` | `#d6d3cc` |
| `field` | `#141311` | `#bcbab3` |
| `ink` | `#e8e4dc` | `#1a191c` |
| `ink-soft` | `#d2cec6` | `#333236` |
| `ink-muted` | `#a09c93` | `#5c5a54` |
| `ink-faint` | `#6f6c64` | `#807d75` |
| `edge` | `rgb(255 255 255 / 0.10)` | `rgb(0 0 0 / 0.12)` |
| `divider` | `rgb(255 255 255 / 0.18)` | `rgb(0 0 0 / 0.22)` |
| `brand` | `#f6b8d0` | `#b23a76` |
| `composer` | `#262421` | `#dbd8d1` |
| `composer-border` | `rgb(255 255 255 / 0.14)` | `rgb(0 0 0 / 0.18)` |

Cuatro grados de tinta, no dos. La jerarquía de la interfaz se construye con
`ink`, `ink-soft`, `ink-muted` e `ink-faint` sobre un fondo casi plano, en lugar
de con cajas y bordes. `edge` es el borde de un control; `divider` es una
costura estructural (barra lateral, paneles) y por eso es más presente.

El tema claro no es blanco: es el gris cálido de un chasis. Un fondo blanco puro
con tipografía monoespaciada produce demasiado contraste para leer documentos
largos.

## Estados, que no son decoración

`brand` es el único color decorativo del sistema y se usa con cuentagotas: el
acento del logo y el botón de enviar.

Los colores de estado se quedan porque informan de algo real: el error de
conexión, el servidor de Ollama parado, una indexación fallida. Lo que
desaparece es el color que solo adornaba. El criterio: si el color desaparece y
el usuario pierde información, se queda; si solo pierde alegría, se va.

## Tema

El tema se selecciona con un atributo de datos en el elemento raíz,
`[data-theme="dark"]` o `[data-theme="light"]`, no con una clase. El estado del
tema deja de vivir en el mismo espacio de nombres que las utilidades, que es
donde se mezclaba con ellas.

Oscuro es el modo por defecto.

## Tipografía

JetBrains Mono como única familia: cuerpo, títulos y código. Se sirve con
`@fontsource-variable/jetbrains-mono`, empaquetada con la aplicación, porque la
app de escritorio tiene que arrancar sin red. Geist y Geist Mono salen del
proyecto.

Una sola familia monoespaciada en toda la interfaz pone el envoltorio en el
mismo registro que el contenido que muestra, que son documentos técnicos, y
elimina de paso la jerarquía decorativa que aportaba el contraste entre dos
familias. La jerarquía la hacen el tamaño y los cuatro grados de tinta.

Las fórmulas matemáticas son la excepción: KaTeX trae su propia familia y su
propio CSS, y debe seguir pintando con ella. Una fórmula en monoespaciada no es
una fórmula.

## Forma

El sistema anterior colapsaba a cero todos los radios de Tailwind desde
`@theme`, de modo que la interfaz entera era cuadrada aunque el marcado
estuviera lleno de clases `rounded-*`. Ese colapso se retira: el marcado ya
está escrito, así que la interfaz curva sin reescribir una sola clase.

El composer es la pieza con forma propia: caja de radio amplio (`1.6rem`), fondo
`composer`, borde `composer-border` y una sombra larga y difusa que lo despega
del fondo sin recuadrarlo.

## El chat

La decisión más visible: **la respuesta del modelo pierde la caja**. Se pinta
como texto sobre el fondo, en `ink-soft`, sin borde ni relleno de contenedor.
El mensaje del usuario conserva una caja discreta alineada a la derecha, con
fondo `field` y radio amplio: es lo único que necesita para distinguirse.

En una herramienta de lectura, encajonar la respuesta la convierte en un objeto
de interfaz que compite con el documento. Sin caja, la respuesta es texto para
leer, que es lo que es.

Las acciones de cada mensaje (copiar, reintentar, ver fuentes) son iconos
pequeños en `ink-faint` que se activan al pasar por encima. Presentes, no
prominentes.

## Textura

Ambos temas llevan una trama de líneas horizontales muy tenue sobre toda la
ventana, aplicada con `body::after` y un `repeating-linear-gradient`, sin
capturar eventos de puntero. Es lo único ornamental del sistema y su función es
que la superficie plana no se lea como papel en blanco.

## Lo que no cambia

- **El logo del mono y su naranja.** Es la única marca propia y sobrevive
  intacta; no se repinta con los tokens.
- **La librería de iconos.** Se conservan lucide y phosphor. Lo que se unifica
  es su uso: tamaño pequeño, `ink-faint` en reposo.
- **Layout, endpoints y comportamiento.** Este documento describe cómo se ve la
  aplicación, no qué hace.

## Riesgos conocidos

1. **Retirar el colapso de radios afecta a toda la interfaz de una vez.** Hay
   elementos cuyo `rounded-xl` nunca se vio y que ahora se verá. Exige repaso
   visual pantalla por pantalla, no solo que compile.
2. **Hay color fuera del sistema de tokens.** El marcado actual contiene clases
   de la paleta por defecto de Tailwind (`text-zinc-300`, `text-orange-500`,
   `text-red-400`, `bg-amber-500/10`) y una clase propia, `glass-panel`. Cada
   una hay que migrarla a mano; la que se olvide aparecerá como un color fuera
   de paleta en una pantalla que quizá solo se ve en un caso de error.
3. **Las pantallas de error y carga se ven poco.** Son justo donde vive la mayor
   parte del color suelto del punto anterior. Hay que forzarlas para revisarlas.

## Verificación

El sistema está bien aplicado cuando:

- `pnpm run lint` y `pnpm run build` quedan verdes.
- Ninguna clase de la paleta por defecto de Tailwind sobrevive en el marcado,
  salvo en el logo. Es comprobable:
  `grep -rnE '(text|bg|border)-(zinc|orange|red|amber|green|slate|gray)-[0-9]' src/`
- Ninguna clase de color arbitraria del sistema anterior sobrevive:
  `grep -rn 'var(--surface)\|var(--text)\|var(--border)\|var(--accent)' src/`
- Las dos pantallas de estado (indexando, error de conexión) y el panel de
  ajustes se han revisado en los dos temas, no solo el chat.
- Una fórmula matemática sigue pintando con la tipografía de KaTeX.
