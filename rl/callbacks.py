"""
callbacks.py

Callbacks personalizados para el entrenamiento del agente de Mario con
Stable-Baselines3 (SB3).

Estos callbacks estan pensados para encajar con el entorno MarioEnv
(ver mario_env.py), que en el diccionario `info` de cada step expone:

    x_pos, max_x, score, coins, powerLVL, lives, sim_clock, status

donde `status` es uno de: "running", "win", "death".

Contenido:
    - MarioMetricsCallback: loguea metricas propias de Mario a TensorBoard
      al finalizar cada episodio (progreso maximo, monedas, score, tasa de
      victorias, etc.).
    - TrainingProgressCallback: imprime un resumen legible en consola cada
      cierto numero de pasos.
    - SaveOnBestTrainingRewardCallback: guarda el mejor modelo segun la
      recompensa media reciente.
    - make_callbacks(...): helper para construir una CallbackList lista para
      pasar a model.learn(...).

Autor: Alejandro C
"""

import os
from collections import deque

import numpy as np

from stable_baselines3.common.callbacks import (
    BaseCallback,
    CallbackList,
    CheckpointCallback,
)


class MarioMetricsCallback(BaseCallback):
    """
    Registra en TensorBoard metricas especificas del juego de Mario.

    Funciona tanto con entornos simples como vectorizados (VecEnv). SB3
    entrega, en cada paso, listas con un elemento por entorno paralelo en
    `self.locals["infos"]` y `self.locals["dones"]`.

    Para cada episodio que termina acumula:
        - rollout/mario_max_x       : avance horizontal maximo (pixeles)
        - rollout/mario_coins       : monedas al terminar
        - rollout/mario_score       : puntaje al terminar
        - rollout/mario_ep_len      : duracion del episodio (steps)
        - rollout/mario_win_rate    : fraccion de episodios ganados
                                      (ventana movil)
        - rollout/mario_wins_total  : total acumulado de victorias
        - rollout/mario_deaths_total: total acumulado de muertes
    """

    def __init__(self, window: int = 100, verbose: int = 0):
        super().__init__(verbose)
        self.window = window

        # Resultados recientes (1.0 = win, 0.0 = no win) para tasa movil.
        self._recent_wins = deque(maxlen=window)
        self._recent_max_x = deque(maxlen=window)
        self._recent_coins = deque(maxlen=window)
        self._recent_score = deque(maxlen=window)
        self._recent_ep_len = deque(maxlen=window)

        # Contador de steps por cada entorno paralelo, para medir ep_len.
        self._ep_steps = None

        # Acumulados globales.
        self.wins_total = 0
        self.deaths_total = 0
        self.episodes_total = 0

    def _init_counters(self, n_envs: int):
        if self._ep_steps is None:
            self._ep_steps = np.zeros(n_envs, dtype=np.int64)

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", None)

        # Normalizamos a lista aunque sea un unico entorno.
        if dones is None:
            dones = [False] * len(infos)

        self._init_counters(len(infos))
        self._ep_steps += 1

        for i, info in enumerate(infos):
            if not dones[i]:
                continue

            self.episodes_total += 1
            status = info.get("status", "running")
            is_win = 1.0 if status == "win" else 0.0

            if status == "win":
                self.wins_total += 1
            elif status == "death":
                self.deaths_total += 1

            self._recent_wins.append(is_win)
            self._recent_max_x.append(float(info.get("max_x", 0.0)))
            self._recent_coins.append(float(info.get("coins", 0.0)))
            self._recent_score.append(float(info.get("score", 0.0)))
            self._recent_ep_len.append(int(self._ep_steps[i]))

            # Reiniciamos el contador de pasos de ese entorno.
            self._ep_steps[i] = 0

            self._record_to_tensorboard()

        return True

    def current_metrics(self) -> dict:
        """
        Resumen de las metricas recientes, usado por train.py para decidir si
        la corrida merece convertirse en el campeon global.

        win_rate y mean_max_x se calculan sobre la ventana movil; los totales
        son acumulados de toda la corrida.
        """
        win_rate = float(np.mean(self._recent_wins)) if self._recent_wins else 0.0
        mean_max_x = float(np.mean(self._recent_max_x)) if self._recent_max_x else 0.0
        return {
            "win_rate": win_rate,
            "mean_max_x": mean_max_x,
            "wins_total": int(self.wins_total),
            "deaths_total": int(self.deaths_total),
            "episodes_total": int(self.episodes_total),
            "window": self.window,
        }

    def _record_to_tensorboard(self):
        if len(self._recent_max_x) == 0:
            return

        self.logger.record("rollout/mario_max_x", float(np.mean(self._recent_max_x)))
        self.logger.record("rollout/mario_coins", float(np.mean(self._recent_coins)))
        self.logger.record("rollout/mario_score", float(np.mean(self._recent_score)))
        self.logger.record("rollout/mario_ep_len", float(np.mean(self._recent_ep_len)))
        self.logger.record("rollout/mario_win_rate", float(np.mean(self._recent_wins)))
        self.logger.record("rollout/mario_wins_total", int(self.wins_total))
        self.logger.record("rollout/mario_deaths_total", int(self.deaths_total))
        self.logger.record("rollout/mario_episodes_total", int(self.episodes_total))


class TrainingProgressCallback(BaseCallback):
    """
    Imprime un resumen legible del entrenamiento cada `log_freq` pasos.

    Util para seguir el progreso sin tener que abrir TensorBoard. Muestra
    los timesteps, la tasa de victorias reciente y el avance maximo medio.
    """

    def __init__(self, log_freq: int = 10_000, window: int = 100, verbose: int = 1):
        super().__init__(verbose)
        self.log_freq = log_freq
        self.window = window

        self._recent_wins = deque(maxlen=window)
        self._recent_max_x = deque(maxlen=window)

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", None)
        if dones is None:
            dones = [False] * len(infos)

        for i, info in enumerate(infos):
            if not dones[i]:
                continue
            status = info.get("status", "running")
            self._recent_wins.append(1.0 if status == "win" else 0.0)
            self._recent_max_x.append(float(info.get("max_x", 0.0)))

        if self.num_timesteps % self.log_freq == 0 and self.verbose > 0:
            win_rate = np.mean(self._recent_wins) if self._recent_wins else 0.0
            mean_max_x = np.mean(self._recent_max_x) if self._recent_max_x else 0.0
            print(
                f"[progreso] timesteps={self.num_timesteps:,} | "
                f"win_rate={win_rate:.2%} | "
                f"avance_max_medio={mean_max_x:.1f}px | "
                f"episodios={len(self._recent_max_x)} (ventana)"
            )

        return True


class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Guarda el modelo cuando la recompensa media reciente mejora su marca.

    SB3 mantiene las recompensas de los ultimos episodios en
    `self.model.ep_info_buffer` (cada entrada tiene la clave "r"). Cada
    `check_freq` pasos calculamos la media y, si supera el mejor valor
    registrado, guardamos el modelo en `save_path`.
    """

    def __init__(
        self,
        check_freq: int = 10_000,
        save_path: str = "models",
        name: str = "best_model",
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.save_path = save_path
        self.name = name
        self.best_mean_reward = -np.inf

    def _init_callback(self) -> None:
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.num_timesteps % self.check_freq != 0:
            return True

        buffer = getattr(self.model, "ep_info_buffer", None)
        if not buffer or len(buffer) == 0:
            return True

        rewards = [ep_info["r"] for ep_info in buffer if "r" in ep_info]
        if not rewards:
            return True

        mean_reward = float(np.mean(rewards))

        if mean_reward > self.best_mean_reward:
            self.best_mean_reward = mean_reward
            path = os.path.join(self.save_path, self.name)
            self.model.save(path)
            if self.verbose > 0:
                print(
                    f"[checkpoint] nuevo mejor reward medio="
                    f"{mean_reward:.2f} -> guardado en {path}.zip"
                )

        return True


def make_callbacks(
    save_path: str = "models",
    checkpoint_freq: int = 50_000,
    best_check_freq: int = 10_000,
    progress_freq: int = 10_000,
    window: int = 100,
    verbose: int = 1,
) -> CallbackList:
    """
    Construye una CallbackList lista para pasar a model.learn(callback=...).

    Agrupa:
        - MarioMetricsCallback          (metricas de Mario en TensorBoard)
        - TrainingProgressCallback      (resumen en consola)
        - SaveOnBestTrainingRewardCallback (mejor modelo por reward medio)
        - CheckpointCallback            (guardado periodico de respaldo)

    Parametros
    ----------
    save_path : carpeta donde se guardan modelos y checkpoints.
    checkpoint_freq : cada cuantos pasos se guarda un checkpoint periodico.
    best_check_freq : cada cuantos pasos se evalua el mejor modelo.
    progress_freq : cada cuantos pasos se imprime el resumen en consola.
    window : tamano de la ventana movil para las metricas promediadas.
    verbose : nivel de verbosidad de los callbacks.

    Devuelve
    --------
    (callbacks, metrics_cb) : la CallbackList lista para model.learn(...) y la
    referencia al MarioMetricsCallback, para que train.py pueda leer las
    metricas finales (win_rate, avance) y decidir la promocion a campeon.
    """
    os.makedirs(save_path, exist_ok=True)

    metrics_cb = MarioMetricsCallback(window=window, verbose=verbose)

    callbacks = CallbackList([
        metrics_cb,
        TrainingProgressCallback(
            log_freq=progress_freq, window=window, verbose=verbose
        ),
        SaveOnBestTrainingRewardCallback(
            check_freq=best_check_freq,
            save_path=save_path,
            name="best_model",
            verbose=verbose,
        ),
        CheckpointCallback(
            save_freq=checkpoint_freq,
            save_path=os.path.join(save_path, "checkpoints"),
            name_prefix="mario_ppo",
            verbose=verbose,
        ),
    ])
    return callbacks, metrics_cb
