# Proyecto Final I.A

> **Autores:** Alessandro Soccol, Samuel Serna, Paula Arroyave, Alejandro Correa
> **Curso:** Inteligencia Artificial — Universidad EAFIT
> **Fecha:** Noviembre de 2026

Este proyecto desarrolla un agente de inteligencia artificial capaz de completar
de forma autónoma el primer nivel de *Super Mario Bros*. El agente no recibe
ninguna instrucción explícita sobre cuándo saltar o cómo evitar a un enemigo:
partiendo de cero, aprende por sí mismo —a través de sucesivas partidas— a
correr, saltar, esquivar obstáculos y alcanzar la meta. El trabajo se apoya en un
clon del juego implementado en Pygame y en aprendizaje por refuerzo, concretamente
el algoritmo PPO.

---

## 1. Introducción

Los juegos de plataformas constituyen un escenario especialmente exigente para un
agente de inteligencia artificial, pues integran simultáneamente varias
capacidades: la comprensión del espacio, la anticipación de los obstáculos y la
sincronización precisa de cada acción. Un error de apenas unos instantes basta
para que el personaje caiga a un vacío o colisione con un enemigo. Por ello
representan un banco de pruebas idóneo para evaluar la toma de decisiones
secuencial de un agente.

El objetivo central del proyecto fue precisamente ese: lograr que el agente
aprendiera a superar el nivel 1-1 **sin indicaciones previas ni movimientos
codificados manualmente**, basándose únicamente en la observación del estado del
juego y en las recompensas obtenidas.

El valor del trabajo no reside solo en que el personaje alcance la meta, sino en
el proceso completo que lo hace posible: la integración de un juego preexistente
con las herramientas estándar de aprendizaje por refuerzo (Gymnasium) **sin
modificar su código fuente**, el diseño de una función de recompensa que evite
inducir comportamientos degenerados, y el desarrollo de una metodología de
diagnóstico para identificar las causas del estancamiento del agente. Cabe
destacar que uno de los hallazgos más relevantes no fue de naturaleza algorítmica,
sino del propio juego: se identificó un obstáculo insuperable con la física
original. En conjunto, el proyecto ofrece una base sólida y reproducible para
experimentar con el diseño de recompensas y con la representación del estado.
## 2. Objetivo

**General:** Entrenar un agente de RL capaz de completar el nivel 1-1 de Super
Mario Bros de forma autónoma.

**Específicos:**

1. Construir un entorno Gymnasium que envuelva el juego Pygame sin alterar su
   código, operable headless y de forma determinista.
2. Diseñar una función de recompensa que induzca avance sostenido hacia la meta
   y penalice el tiempo, evitando que la política colapse a estrategias pasivas.
3. Alcanzar una tasa de victorias (llegar a la bandera) ≥ 30 % en evaluación
   sobre episodios independientes.

## 3. Descripción de los datos

Este proyecto **no usa un dataset estático**: los datos son la experiencia que el
agente genera al interactuar con el entorno (transiciones estado→acción→
recompensa). La "fuente de datos" es el propio juego, muestreado en cada paso.

**Observación (estado) que recibe el agente** — vector de 132 valores:

| Grupo | Dimensión | Descripción |
|---|---|---|
| Escalares del jugador | 6 | pos. x, pos. y, vel. x, vel. y, en-suelo (0/1), nivel de poder |
| Grid de bloques | 7×9 = 63 | sólidos alrededor de Mario (1 = bloque, 0 = vacío) |
| Grid de mobs | 7×9 = 63 | enemigos (+1), power-ups hongo/flor (−1), nada (0) |

**Acciones** — 7 discretas: nada, derecha, derecha+salto, derecha+correr,
derecha+correr+salto, salto, izquierda.

**Preprocesamiento:** los escalares se normalizan a rangos comparables; las
grillas se construyen centradas en Mario desde la estructura interna del mapa y
la lista de mobs del juego. El tiempo del juego se simula con un reloj
determinista (frame-skip de 4 ticks por acción) para entrenar a máxima velocidad
sin ventana ni audio.

**Limitaciones conocidas:** el nivel 1-1 original del clon contiene un tubo de 4
tiles (128 px) que supera la altura de salto base de Mario (≈101 px con el valor
original `JUMP_POWER=4.3`); el salto base solo corona ese tubo rebotando sobre un
enemigo. Para hacer el nivel superable de forma fiable por el agente se ajustó la
constante física `JUMP_POWER` (ver Metodología). El entorno depende de los
recursos del juego (`game/MarioPygame/Next/`), que conservan su licencia original.

## 4. Metodología

**Etapas del pipeline:**

1. **Entorno (`mario_env.py`):** envuelve el juego en una interfaz Gymnasium.
   Arranca headless, controla a Mario por los flags internos del juego (sin
   eventos de teclado), parchea el reloj para ser determinista y construye
   observación y recompensa en cada paso.
2. **Modelado:** política PPO (`MlpPolicy`, red 64×64 para actor y crítico) de
   Stable-Baselines3, con observación vectorial y acción discreta.
3. **Entrenamiento (`train.py`):** 8 entornos en paralelo, normalización de
   recompensa, callbacks que registran métricas y guardan el mejor modelo, y
   promoción automática a un "campeón global" si la corrida supera al histórico.
4. **Evaluación (`evaluate.py`):** episodios independientes con estadísticas
   (tasa de victorias con IC de Wilson, avance medio, distribución de acciones),
   comparables contra una línea base aleatoria.
5. **Visualización (`watch.py`, `plot_eval.py`):** ver al agente jugar en ventana
   y graficar los resultados de la evaluación.

**Justificación de las decisiones principales:**

- **Elección de PPO.** Se optó por PPO por ser uno de los algoritmos más fiables
  cuando el estado se representa como un vector numérico y el conjunto de acciones
  es reducido y discreto. Ofrece, además, un entrenamiento más estable que
  alternativas como DQN en este tipo de problemas de control.
- **Representación vectorial del estado.** En lugar de procesar los píxeles de la
  pantalla, el agente recibe un resumen numérico del estado (posición, velocidad y
  el entorno inmediato), lo que acelera considerablemente el aprendizaje.
  Posteriormente se constató que esta representación omitía información esencial
  —el agente no percibía a los enemigos ni a los power-ups—, por lo que se incorporó
  una segunda capa que los hace visibles.
- **Diseño iterativo de la recompensa.** El principio es simple: el avance debe ser
  lo más recompensado, el tiempo transcurrido debe penalizar y alcanzar la meta
  debe constituir el mayor incentivo. No obstante, las primeras versiones indujeron
  un comportamiento degenerado —el agente optaba por permanecer inmóvil—, lo que
  obligó a rediseñar la función para eliminar ese atajo.
- **Ajuste de la fuerza de salto (4.3 → 4.8).** Fue la decisión más delicada, al
  requerir una modificación del juego. Tras un análisis empírico se determinó que el
  nivel contenía un tubo de 128 px que el personaje no podía superar, ya que su
  salto alcanzaba apenas ~101 px. No se trataba de una deficiencia del agente, sino
  de una imposibilidad física. En consecuencia, se elevó la fuerza de salto hasta un
  valor que permite superar el obstáculo (129 px). Fue la única constante del juego
  modificada, y únicamente tras descartar que el problema fuera de aprendizaje.

**Función de recompensa:**

| Término | Valor | Propósito |
|---|---|---|
| Avance | `Δx × 1.0` | señal densa hacia la derecha |
| Récord de avance | `(x − máx) × 1.0` | premia terreno nuevo (rompe el quedarse quieto) |
| Tiempo | `−0.1` / paso | incentiva terminar rápido |
| Estancamiento | `−0.05` / paso + corte a 120 | penaliza no progresar |
| Puntuación | `Δscore × 0.01` | matar enemigos, monedas, hongo |
| Bloque power-up | `+15` por bloque | favorece golpear cajas de hongo/flor |
| Victoria | `+500 + tiempo_restante × 5` | meta + bono por rapidez |
| Muerte | `−50` | castigo moderado (no paraliza) |

**Hiperparámetros clave:** `learning_rate=2.5e-4`, `n_steps=512`,
`batch_size=256`, `gamma=0.99`, `gae_lambda=0.95`, `clip_range=0.2`,
`ent_coef=0.03` (exploración elevada para evitar colapso de la política),
`frame_skip=4`, 8 entornos paralelos.

**Diagrama de flujo metodológico:**

![Diagrama](docs/images/diagrama.png)

## 5. Diagrama de interacción (UML)

El uso del sistema es directo: el usuario ejecuta un único comando y se abre la
ventana del juego con el agente entrenado operando de forma autónoma. No se
requiere servidor ni base de datos; el modelo se carga desde un archivo en disco
y, a partir de ese momento, es él quien controla al personaje. El siguiente
diagrama detalla la interacción entre el usuario, el modelo y el juego a lo largo
del tiempo.

```mermaid
sequenceDiagram
    actor U as Usuario
    participant W as watch.py / evaluate.py
    participant M as Modelo PPO (campeón)
    participant E as MarioEnv
    participant G as Juego (Core)

    U->>W: Ejecuta el comando
    W->>M: Carga models/champion/champion.zip
    W->>E: reset()
    E->>G: inicia nivel 1-1 (headless o ventana)
    E-->>W: observación inicial (132-d)
    loop cada paso hasta fin del episodio
        W->>M: predict(observación)
        M-->>W: acción (0..6)
        W->>E: step(acción)
        E->>G: aplica flags + avanza ticks
        G-->>E: nuevo estado (pos, mobs, score, fin)
        E-->>W: observación, recompensa, terminado
    end
    E-->>W: resultado (victoria / muerte, avance, score)
    W-->>U: muestra gameplay y/o estadísticas
```

## 6. Resultados

La tabla compara las tres grandes etapas del proyecto, para que se vea de un
vistazo cómo fue cambiando el agente a medida que arreglábamos las cosas. La
última columna es el agente final (`JUMP_POWER=4.8`, recompensa completa con
incentivo de puntuación y el agente viendo a los enemigos), entrenado los 3
millones de pasos completos.

| Métrica | Versión inicial (colapso) | Corrida con tubo insuperable | Agente final |
|---|---|---|---|
| Avance máximo medio | 129 px (salida) | ~1 430 px (atascado en tubo) | **~6 040 px** (meta ≈ 6 336) |
| Tasa de victorias | 0 % | 0 % | **92 %** |

**Evaluación final (50 episodios independientes, modo estocástico):**

| Agente | Victorias | Tasa (IC 95 %) | Recompensa media | Avance medio |
|---|---|---|---|---|
| **PPO (nuestro)** | 46/50 | **92 % [81.2 %, 96.9 %]** | 14 122 | 6 040 px |
| Aleatorio (línea base) | 0/50 | 0 % [0 %, 7.1 %] | 1 709 | 1 016 px |

> La tasa del 0 % del agente aleatorio es el resultado esperado y constituye la
> **línea base** del experimento: un agente que actúa al azar no logra completar
> el nivel en ninguna ocasión. El contraste con el 92 % del agente entrenado es
> precisamente lo que evidencia que el aprendizaje fue efectivo.

**Curva de aprendizaje.** La siguiente figura representa la evolución, a lo largo
del entrenamiento (eje X en escala logarítmica), del porcentaje de avance hacia la
meta (azul) y del porcentaje de éxito —nivel completado— (verde). Se observa el
patrón característico del aprendizaje por refuerzo: el agente adquiere primero la
capacidad de *avanzar* y solo en una fase posterior la de *completar* el nivel.

![Curva de aprendizaje del agente final](rl/figuras/run_v9_1/curva_aprendizaje.png)

Las figuras de la evaluación (avance y recompensa por episodio, distribución de
resultados y comparación contra la línea base) se generan en
`rl/figuras/<evaluación>/`, y la curva de aprendizaje en
`rl/figuras/<corrida>/curva_aprendizaje.png` con `plot_training.py`.

**Análisis de los resultados.**

- **Aspectos exitosos.** El rediseño de la función de recompensa eliminó el
  comportamiento pasivo —permanecer inmóvil— que presentaban las primeras
  versiones. La incorporación de la percepción de enemigos, junto con la corrección
  de la física del salto, permitió finalmente superar el obstáculo que bloqueaba el
  nivel. El agente final lo completa en el **92 % de los intentos**, frente al 0 %
  de un agente aleatorio, lo que evidencia de manera concluyente que adquirió una
  política de juego efectiva.
- **Limitaciones encontradas.** Durante una fase del desarrollo, diversos ajustes
  de la recompensa (premiar la puntuación o la rotura de bloques) no produjeron
  mejora alguna, pues el agente permanecía detenido en el mismo punto. La conclusión
  fue clara: antes de continuar modificando la recompensa es imprescindible
  determinar la causa del estancamiento. En este caso, dicha causa no era de
  naturaleza algorítmica, sino propia del juego (un tubo imposible de saltar).
- **Observación relevante sobre la evaluación.** Se constató que, al seleccionar
  siempre la acción más probable (modo determinista), el agente se estancaba y su
  tasa de éxito caía al 0 %. En cambio, al introducir cierta variabilidad en las
  decisiones (modo estocástico, coherente con el entrenamiento), la tasa ascendía al
  92 %. Este fenómeno es esperable: la superación de ciertos obstáculos exige variar
  la sincronización de las acciones, mientras que una política completamente rígida
  reproduce el mismo error de forma reiterada.
  Por eso tanto la evaluación como la visualización usan el modo estocástico por
  defecto.
- **Reproducir estos números:**
  ```
  ./.venv/Scripts/python.exe rl/evaluate.py --episodes 50 --baseline-random
  ./.venv/Scripts/python.exe rl/plot_eval.py
  ```

## 7. Reproducibilidad

**Requisitos e instalación:**

```
# Crear y activar un entorno virtual, luego:
pip install -r requirements.txt
```

Dependencias principales: `pygame`, `pytmx`, `gymnasium`, `numpy`, `torch`,
`stable-baselines3`, `tensorboard`, `matplotlib`.

**Comandos mínimos** (desde la raíz del proyecto, con el Python del entorno):

```
# Entrenar un agente (crea models/<run-name>/ y promueve al campeón al terminar)
python rl/train.py --timesteps 3000000 --n-envs 8 --run-name final

# Ver al campeón jugar en una ventana
python rl/watch.py

# Evaluar con estadísticas, comparando contra un agente aleatorio
python rl/evaluate.py --episodes 50 --baseline-random

# Graficar los resultados de la evaluación (-> rl/figuras/<evaluación>/)
python rl/plot_eval.py

# Graficar la curva de aprendizaje de una corrida (eje X logarítmico)
python rl/plot_training.py --run final_1

# Ver curvas de entrenamiento
python -m tensorboard.main --logdir runs
```

> En Windows con el venv del repo, reemplazar `python` por
> `./.venv/Scripts/python.exe`.

**Estructura del repositorio:**

```
Super-AI-bruh/
├── game/MarioPygame/Next/   # Juego Pygame (no modificar, salvo JUMP_POWER justificado)
├── rl/                      # Código de Reinforcement Learning
│   ├── mario_env.py         # Entorno Gymnasium que envuelve el juego
│   ├── train.py             # Entrenamiento PPO + promoción de campeón
│   ├── evaluate.py          # Evaluación con estadísticas (CSV + JSON)
│   ├── watch.py             # Ver al agente jugar en ventana
│   ├── plot_eval.py         # Gráficas de la evaluación
│   ├── plot_training.py     # Curva de aprendizaje (avance y éxito vs pasos, log)
│   ├── callbacks.py         # Métricas, checkpoints, mejor modelo
│   ├── champion.py          # Lógica del campeón global
│   └── figuras/             # Salida de las gráficas (una subcarpeta por corrida)
├── models/                  # Modelos entrenados (champion/ + run_x/)
├── runs/                    # Logs de TensorBoard
└── requirements.txt
```

## 8. Instructivo de ejecución (paso a paso)

Guía para ejecutar el proyecto de principio a fin. Todos los comandos se lanzan
**desde la raíz del proyecto**. En Windows con el entorno del repo, usa
`./.venv/Scripts/python.exe` en lugar de `python`.

**Paso 0 — Preparar el entorno (una sola vez).**
Instala las dependencias en un entorno virtual:
```
pip install -r requirements.txt
```

**Paso 1 — Entrenar al agente.**
Crea `models/<run-name>/` y, al terminar, lo compara con el campeón global y lo
promueve si es mejor. Ajusta `--run-name` a tu gusto (debe ser único por corrida):
```
python rl/train.py --timesteps 3000000 --n-envs 8 --run-name final
```
> Tarda ~25-30 min. Si quieres solo probar que todo funciona, usa un valor
> pequeño como `--timesteps 50000` (el agente no jugará bien, pero verifica el
> flujo).

**Paso 2 — (Opcional) Seguir el entrenamiento en vivo.**
Mientras entrena, en otra terminal, abre TensorBoard y entra a
`http://localhost:6006`:
```
python -m tensorboard.main --logdir runs
```

**Paso 3 — Ver al campeón jugar.**
Abre la ventana del juego con el mejor agente jugando solo (modo estocástico por
defecto, que es como juega bien):
```
python rl/watch.py
```

**Paso 4 — Evaluar el rendimiento.**
Hace jugar al campeón 50 partidas independientes y lo compara contra un agente
aleatorio. Escribe `episodes.csv` y `summary.json`:
```
python rl/evaluate.py --episodes 50 --baseline-random
```

**Paso 5 — Graficar los resultados de la evaluación.**
Lee la evaluación más reciente del paso 4 y genera las figuras en
`rl/figuras/<evaluación>/`:
```
python rl/plot_eval.py
```

**Paso 6 — Graficar la curva de aprendizaje.**
Dibuja cómo subieron el % de avance y el % de éxito durante el entrenamiento
(eje X logarítmico). Sustituye `final_1` por el nombre de tu corrida en `runs/`:
```
python rl/plot_training.py --run final_1
```

**Resumen del orden:** `instalar → entrenar → (ver) → evaluar → graficar`. Los
pasos 1 y 4 generan los artefactos (modelo y datos); los demás solo los
consumen, así que requieren que sus pasos previos se hayan ejecutado.
