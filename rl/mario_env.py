import os
import sys
import contextlib

import numpy as np
import gymnasium as gym
from gymnasium import spaces

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_THIS_DIR)
GAME_DIR = os.path.join(_PROJECT_ROOT, "game", "MarioPygame", "Next")



# ACCIONES: (keyR, keyL, keyU, keyShift)
ACTIONS = [
    (False, False, False, False),  # 0 no hacer nada
    (True,  False, False, False),  # 1 derecha
    (True,  False, True,  False),  # 2 derecha + salto
    (True,  False, False, True),   # 3 derecha + correr
    (True,  False, True,  True),   # 4 derecha + correr + salto
    (False, False, True,  False),  # 5 salto en el sitio
    (False, True,  False, False),  # 6 izquierda
]


class MarioEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(self, frame_skip=4, max_steps=3000, render_mode=None):
        super().__init__()

        self.frame_skip = frame_skip      # ticks de juego por cada accion del agente
        self.max_steps = max_steps        # tope de acciones por episodio (evita quedarse colgado)
        self.render_mode = render_mode

        # Reloj simulado en base a cuantos milisegundos equivalen a cada tick
        self._sim_ms_per_tick = 10
        self._sim_clock = 0

        self.action_space = spaces.Discrete(len(ACTIONS)) #Listado de acciones

        obs_dim = self._obs_dim()
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        self.core = None
        self._steps = 0
        self._last_x = 0.0
        self._max_x = 0.0

        # Entorno de ejecucion del juego
        self._bootstrap_game_module()

    def _bootstrap_game_module(self):
        # Headless
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        os.environ.setdefault("SDL_AUDIODRIVER", "dummy")


        if GAME_DIR not in sys.path:
            sys.path.insert(0, GAME_DIR)
        os.chdir(GAME_DIR)

        import pygame as pg
        self._pg = pg

        # Tiempo con el reloj somulado
        pg.time.get_ticks = lambda: self._sim_clock

    def _new_core(self):
        from Core import Core

        #Asumimos que ya estamos en el juego, saltando la pantalla de carga.
        core = Core()
        core.get_mm().currentGameState = "Game"
        core.get_map().in_event = False
        return core

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        self._sim_clock = 0
        self._steps = 0

        # Creamos un nuevo estado siempre antes de ejecutarse
        self.core = self._new_core()

        player = self.core.get_map().get_player()
        self._last_x = float(player.rect.x)
        self._max_x = float(player.rect.x)

        obs = self._get_obs()
        info = self._get_info()
        return obs, info

    def step(self, action):
        keyR, keyL, keyU, keyShift = ACTIONS[int(action)]

        terminated = False
        truncated = False

        # Sostener una accion.
        for _ in range(self.frame_skip):
            self._apply_keys(keyR, keyL, keyU, keyShift)
            self._tick_game()

            status = self._episode_status()
            if status != "running":
                terminated = True
                break

        self._steps += 1
        if self._steps >= self.max_steps:
            truncated = True

        reward, status = self._compute_reward(terminated)
        obs = self._get_obs()
        info = self._get_info()
        info["status"] = status

        return obs, reward, terminated, truncated, info

    def _apply_keys(self, keyR, keyL, keyU, keyShift):
        #Definimos los controles
        self.core.keyR = keyR
        self.core.keyL = keyL
        self.core.keyU = keyU
        self.core.keyShift = keyShift
        self.core.keyD = False

    def _tick_game(self):
        self._sim_clock += self._sim_ms_per_tick
        self.core.update()

    def _episode_status(self):
        #Estado actual del juego
        game_map = self.core.get_map()
        event = game_map.get_event()

        # El juego entra en 'in_event' cuando gana (flag) o muere.
        if game_map.in_event:
            # event.type: 1 = win (flag), 0 = kill/game over
            if event.type == 1:
                return "win"
            return "death"
        return "running"

    def _compute_reward(self, terminated):
        #Sistema de recompensas
        player = self.core.get_map().get_player()
        x = float(player.rect.x)

        status = self._episode_status()

        # Recompensa principal: progreso hacia la derecha.
        dx = x - self._last_x
        self._last_x = x
        self._max_x = max(self._max_x, x)
        reward = dx * 0.1

        # Castigo por existir.
        reward -= 0.01

        if status == "win":
            reward += 500.0 #Recompensa por ganar
        elif status == "death":
            reward -= 50.0 #Castigo por perder

        return reward, status

    def _obs_dim(self):
        # Tamaño vector de observacion. 6 estados del jugador + Campo de vision del agente
        self._grid_rows = 7
        self._grid_cols = 9
        return 6 + self._grid_rows * self._grid_cols

    def _get_obs(self):
        game_map = self.core.get_map()
        player = game_map.get_player()

        level_w = game_map.mapSize[0] * 32.0 #Ancho del nivel

        # Escalares del jugador, normalizados a rangos manejables, dividos entre el valor max.
        scalars = np.array([
            player.rect.x / level_w,           # progreso horizontal 0..1
            player.rect.y / 448.0,             # altura 0..1
            player.x_vel / 3.0,                # velocidad x normalizada
            player.y_vel / 5.5,                # velocidad y normalizada
            1.0 if player.on_ground else 0.0,  # jugador esta en el suelo
            player.powerLVL / 2.0,             # 0 small, 0.5 big, 1.0 fire
        ], dtype=np.float32)

        # Grid local de bloques solidos alrededor del agente.
        grid = self._local_grid(game_map, player)
        #Vector unificado con los escalares del jugador y el campo de vision en su momento
        return np.concatenate([scalars, grid.flatten()]).astype(np.float32) 

    def _local_grid(self, game_map, player):

        #Matriz inicial de 0s
        grid = np.zeros((self._grid_rows, self._grid_cols), dtype=np.float32)

        #cords de mario
        px = player.rect.x // 32
        py = player.rect.y // 32
        half_r = self._grid_rows // 2
        half_c = self._grid_cols // 2

        max_x = game_map.mapSize[0]
        max_y = game_map.mapSize[1]

        #1 = bloque, 0 = vacio
        for r in range(self._grid_rows):
            for c in range(self._grid_cols):
                gx = px + (c - half_c)
                gy = py + (r - half_r)
                if 0 <= gx < max_x and 0 <= gy < max_y:
                    block = game_map.map[gx][gy]
                    if block != 0 and getattr(block, "type", None) != "BGObject":
                        grid[r, c] = 1.0
        return grid

    def _get_info(self):
        player = self.core.get_map().get_player()
        return {
            "x_pos": float(player.rect.x),
            "max_x": self._max_x,
            "score": player.score,
            "coins": player.coins,
            "powerLVL": player.powerLVL,
            "lives": player.numOfLives,
            "sim_clock": self._sim_clock,
        }

    def render(self):
        return None

    def close(self):
        if self.core is not None:
            with contextlib.suppress(Exception):
                self._pg.quit()
            self.core = None
