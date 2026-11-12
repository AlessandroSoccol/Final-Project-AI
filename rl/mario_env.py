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

    def __init__(self, frame_skip=4, max_steps=3000, render_mode=None,
                 max_stuck_steps=120):
        super().__init__()

        self.frame_skip = frame_skip      # ticks de juego por cada accion del agente
        self.max_steps = max_steps        # tope de acciones por episodio
        self.render_mode = render_mode
        # Pasos seguidos sin batir el record de avance antes de cortar el episodio por estancamiento.
        self._max_stuck_steps = max_stuck_steps

        # Reloj simulado en base a cuantos milisegundos equivalen a cada tick
        self._sim_ms_per_tick = 10
        self._sim_clock = 0

        self.action_space = spaces.Discrete(len(ACTIONS)) #Listado de acciones

        obs_dim = self._obs_dim()

        self.observation_space = spaces.Box(
            low=-5.0, high=5.0, shape=(obs_dim,), dtype=np.float32
        )

        self.core = None
        self._steps = 0
        self._last_x = 0.0
        self._max_x = 0.0

        # Entorno de ejecucion del juego
        self._bootstrap_game_module()

    def _bootstrap_game_module(self):
        # En modo "human" queremos ventana real y sonido; en cualquier otro
        # modo (entrenamiento) corremos headless para ir a maxima velocidad.
        if self.render_mode == "human":
            # Ventana visible: NO forzamos el driver dummy de video.
            # (El audio lo dejamos activo para oir el juego.)
            os.environ.pop("SDL_VIDEODRIVER", None)
            os.environ.pop("SDL_AUDIODRIVER", None)
        else:
            # Headless: sin ventana ni audio.
            os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
            os.environ.setdefault("SDL_AUDIODRIVER", "dummy")


        if GAME_DIR not in sys.path:
            sys.path.insert(0, GAME_DIR)
        os.chdir(GAME_DIR)

        import pygame as pg
        self._pg = pg

        # Tiempo con el reloj somulado
        pg.time.get_ticks = lambda: self._sim_clock

        # Soporte para el modo visible ("human").
        self._human_clock = pg.time.Clock()  # limita FPS en la ventana
        self._human_fps = 60                  # velocidad de reproduccion
        self._human_quit = False              # True si cierran la ventana

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

        # Contadores para el reward de progreso/estancamiento/tiempo.
        self._best_x = float(player.rect.x)   # mejor avance alcanzado (record)
        self._stuck_steps = 0                 # pasos seguidos sin batir el record
        self._last_score = int(player.score)  # score previo, para premiar su aumento

        # Bloques que entregan power-up (hongo/flor). Guardamos las referencias
        # una vez para premiar el momento en que Mario los activa (golpea).
        self._bonus_blocks = self._find_bonus_blocks()
        self._bonus_activated_prev = 0        # cuantos llevaba activados

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

        # Fin por estancamiento: si lleva demasiado sin avanzar, cortamos el
        # episodio como si "muriera". Evita que quedarse quieto sea viable y
        # refuerza el objetivo de terminar en el menor tiempo posible.
        if not terminated and self._stuck_steps >= self._max_stuck_steps:
            terminated = True
            status = "death"
            reward -= 50.0  # mismo castigo que una muerte normal

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

        # Modo "human" para ver la repeticion a 60fps, de resto se omite
        if self.render_mode == "human":
            self._draw_human_frame()

    def _draw_human_frame(self):
        pg = self._pg
        # Procesar eventos de la ventana para que el SO no la marque "colgada"
        # y para poder cerrarla con la X.
        for e in pg.event.get():
            if e.type == pg.QUIT:
                self._human_quit = True
        self.core.render()
        # Limita la velocidad a FPS jugables.
        self._human_clock.tick(self._human_fps)

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

    def _find_bonus_blocks(self):
        #Localizacion de bloques de poderes 
        game_map = self.core.get_map()
        blocks = []
        for column in game_map.map:
            for block in column:
                if block != 0 and getattr(block, "typeID", None) == 22:
                    if getattr(block, "bonus", None) == "mushroom":
                        blocks.append(block)
        return blocks

    def _count_bonus_activated(self):
        #Cuantos bloques de poderes han sido activados
        return sum(1 for b in self._bonus_blocks if getattr(b, "isActivated", False))

    def _compute_reward(self, terminated):
        #Sistema de recompensas
        game_map = self.core.get_map()
        player = game_map.get_player()
        x = float(player.rect.x)

        status = self._episode_status()

        # 1) Progreso instantaneo hacia la derecha.
        dx = x - self._last_x
        self._last_x = x
        self._max_x = max(self._max_x, x)
        reward = dx * 1.0

        # 2) Bono por batir el record de avance del episodio. Evita que se quede quieto
        if x > self._best_x:
            reward += (x - self._best_x) * 1.0
            self._best_x = x
            self._stuck_steps = 0
        else:
            # Si no mejora el record, cuenta como estancamiento.
            self._stuck_steps += 1

        # 3) Castigo por existir
        reward -= 0.1

        # 4) Castigo adicional creciente por estancarse.
        if self._stuck_steps > 0:
            reward -= 0.05

        # 5) Bono por PUNTUACION ganada (score)
        score = int(player.score)
        d_score = score - self._last_score
        self._last_score = score
        if d_score > 0:
            reward += d_score * 0.05

        # 5b) Bono por GOLPEAR un bloque de power-up (sube score)
        bonus_now = self._count_bonus_activated()
        if bonus_now > self._bonus_activated_prev:
            reward += 15.0 * (bonus_now - self._bonus_activated_prev)
            self._bonus_activated_prev = bonus_now

        if status == "win":
            # 6) Bono base por ganar mas acumulado bono por rapidez
            time_left = float(getattr(game_map, "time", 0))
            reward += 500.0 + time_left * 5.0
        elif status == "death":
            reward -= 50.0  # castigo por morir

        return reward, status

    def _obs_dim(self):
        # Tamaño del vector de observacion: 6 escalares del jugador + grid de mobs/powerups
        self._grid_rows = 7
        self._grid_cols = 9
        cells = self._grid_rows * self._grid_cols
        return 6 + cells + cells

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

        grid = self._local_grid(game_map, player) #grid bloques solidos
        mob_grid = self._mob_grid(game_map, player) #grid de enemigos y powerups

        #Vector unificado: escalares + bloques + mobs (campo de vision completo)
        return np.concatenate([
            scalars, grid.flatten(), mob_grid.flatten()
        ]).astype(np.float32)

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

    # Clases de mob consideradas PELIGRO (el agente debe esquivar o pisar).
    _ENEMY_NAMES = ("Goombas", "Koopa")
    # Clases de mob consideradas BUENAS (el agente deberia buscar).
    _GOODIE_NAMES = ("Mushroom", "Flower")

    def _mob_grid(self, game_map, player):
        #Identificador de enemigos y poderes en el campo de vision
        grid = np.zeros((self._grid_rows, self._grid_cols), dtype=np.float32)

        px = player.rect.x // 32
        py = player.rect.y // 32
        half_r = self._grid_rows // 2
        half_c = self._grid_cols // 2

        for mob in getattr(game_map, "mobs", []):
            rect = getattr(mob, "rect", None)
            if rect is None:
                continue
            # Posicion del mob relativa a Mario, en celdas de la ventana.
            c = (rect.x // 32) - px + half_c
            r = (rect.y // 32) - py + half_r
            if 0 <= r < self._grid_rows and 0 <= c < self._grid_cols:
                name = type(mob).__name__
                if name in self._ENEMY_NAMES:
                    grid[r, c] = 1.0
                elif name in self._GOODIE_NAMES:
                    grid[r, c] = -1.0
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
        # En modo "human" el frame ya se dibuja dentro de cada tick del juego
        if self.render_mode == "rgb_array":
            # Devuelve el frame actual como arreglo
            frame = self._pg.surfarray.array3d(self.core.screen)
            return np.transpose(frame, (1, 0, 2))
        return None

    def close(self):
        if self.core is not None:
            with contextlib.suppress(Exception):
                self._pg.quit()
            self.core = None
