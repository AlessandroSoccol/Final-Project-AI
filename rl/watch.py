import argparse
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

# Reutilizamos la logica ya probada de evaluate.py para no duplicar criterios.
from evaluate import (
    make_evaluation_env,
    read_training_config,
    resolve_model_path,
)


def base_env(vec_env):
    inner = getattr(vec_env, "venv", vec_env)  # baja por VecNormalize si aplica
    return inner.envs[0]

#Ayudas en terminal
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Muestra al agente de Mario jugando en una ventana."
    )
    parser.add_argument(
        "--model", type=Path, default=None,
        help="Ruta del .zip a ver; si se omite usa el campeon global.",
    )
    parser.add_argument(
        "--episodes", type=int, default=3,
        help="Numero de intentos a mostrar seguidos (por defecto 3).",
    )
    parser.add_argument(
        "--fps", type=int, default=None,
        help="Velocidad de reproduccion en la ventana (por defecto la del env).",
    )
    parser.add_argument(
        "--frame-skip", type=int, default=None,
        help="Sobrescribe frame_skip (por defecto lee config.json del modelo).",
    )
    parser.add_argument(
        "--max-steps", type=int, default=None,
        help="Sobrescribe max_steps (por defecto lee config.json del modelo).",
    )
    parser.add_argument(
        "--deterministic", action="store_true",
        help="Usar siempre la accion mas probable. Por defecto se MUESTREA la "
             "politica (estocastico), que es como el agente juega bien.",
    )
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error("--episodes debe ser >= 1")
    return args


def main() -> int:
    args = parse_args()

    model_path = resolve_model_path(args.model)
    config, run_dir = read_training_config(model_path)

    frame_skip = (
        args.frame_skip if args.frame_skip is not None
        else int(config.get("frame_skip", 4))
    )
    max_steps = (
        args.max_steps if args.max_steps is not None
        else int(config.get("max_steps", 3000))
    )

    print(f"[watch] Modelo: {model_path}")
    print(f"[watch] Mostrando {args.episodes} intento(s) | frame_skip={frame_skip} "
          f"| max_steps={max_steps}")
    print("[watch] Cierra la ventana para terminar antes de tiempo.")

    model = PPO.load(str(model_path), device="cpu")

    # Entorno en modo ventana ("human"): abre la pantalla del juego real.
    env = make_evaluation_env(frame_skip, max_steps, run_dir, render=True)

    underlying = base_env(env)

    # Permite ajustar la velocidad de reproduccion desde la linea de comandos.
    if args.fps is not None:
        underlying._human_fps = max(1, args.fps)

    deterministic = args.deterministic

    try:
        for episode in range(1, args.episodes + 1):
            observation = env.reset()
            done = False
            total_reward = 0.0
            steps = 0
            info = {}

            while not done:
                action, _ = model.predict(observation, deterministic=deterministic)
                observation, rewards, dones, infos = env.step(action)
                total_reward += float(rewards[0])
                steps += 1
                done = bool(dones[0])
                info = infos[0]

                # Si el jugador cerro la ventana, salir.
                if getattr(underlying, "_human_quit", False):
                    print("\n[watch] Ventana cerrada por el usuario.")
                    return 0

            status = info.get("status", "running")
            resultado = {
                "win": "GANO (llego al flag)",
                "death": "murio",
                "running": "se acabo el tiempo/pasos",
            }.get(status, status)

            print(
                f"[watch] Intento {episode}/{args.episodes}: {resultado} | "
                f"avance_max={info.get('max_x', 0.0):.0f}px | "
                f"x_final={info.get('x_pos', 0.0):.0f}px | "
                f"pasos={steps} | reward={total_reward:.1f}"
            )
    finally:
        env.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
