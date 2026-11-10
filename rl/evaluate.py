from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from mario_env import ACTIONS, GAME_DIR, MarioEnv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"

CSV_COLUMNS = [
    "agente", "episodio", "semilla", "recompensa_total", "pasos", "estado_final",
    "victoria", "muerte", "limite_pasos", "max_x", "x_final", "monedas",
    "puntuacion", "vidas", "tiempo_simulado_s",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evalua un agente PPO de Mario con episodios independientes."
    )
    parser.add_argument(
        "--model", type=Path, default=None,
        help="Ruta del .zip de PPO; si se omite busca el modelo mas reciente en models/.",
    )
    parser.add_argument(
        "--episodes", type=int, default=30, help="Numero de episodios por agente (30)."
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Semilla base reproducible (42)."
    )
    parser.add_argument(
        "--frame-skip", type=int, default=None,
        help="Sobrescribe el frame_skip del entrenamiento (por defecto lee config.json).",
    )
    parser.add_argument(
        "--max-steps", type=int, default=None,
        help="Sobrescribe max_steps del entrenamiento (por defecto lee config.json).",
    )
    parser.add_argument(
        "--stochastic", action="store_true",
        help="Muestrea la politica PPO en vez de usar la accion mas probable.",
    )
    parser.add_argument(
        "--baseline-random", action="store_true",
        help="Compara los mismos episodios contra un agente aleatorio.",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="Carpeta de salida (por defecto models/<corrida>/evaluations/<fecha>_...).",
    )
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error("--episodes debe ser >= 1")
    if args.frame_skip is not None and args.frame_skip < 1:
        parser.error("--frame-skip debe ser >= 1")
    if args.max_steps is not None and args.max_steps < 1:
        parser.error("--max-steps debe ser >= 1")
    return args


def resolve_model_path(requested: Path | None) -> Path:
    if requested is not None:
        candidate = requested.expanduser().resolve()
        if candidate.suffix != ".zip" and candidate.with_suffix(".zip").is_file():
            candidate = candidate.with_suffix(".zip")
        if not candidate.is_file():
            raise FileNotFoundError(f"Modelo no encontrado: {candidate}")
        return candidate

    candidates = list(MODELS_DIR.glob("*/best_model.zip"))
    candidates += list(MODELS_DIR.glob("*/final_model.zip"))
    if not candidates:
        raise FileNotFoundError(
            "No hay modelos entrenados en models/. Ejecuta train.py primero "
            "o especifica --model RUTA_AL_MODELO.zip."
        )

    newest_run = max(candidates, key=lambda p: p.stat().st_mtime).parent
    best = newest_run / "best_model.zip"
    return best if best.is_file() else newest_run / "final_model.zip"


def read_training_config(model_path: Path) -> tuple[dict, Path]:
    for directory in model_path.parents:
        config_path = directory / "config.json"
        if config_path.is_file():
            with config_path.open("r", encoding="utf-8") as handle:
                return json.load(handle), directory
        if directory == MODELS_DIR or directory == PROJECT_ROOT:
            break
    print("[aviso] No se encontro config.json; se usan los parametros por defecto.")
    return {}, model_path.parent


def make_evaluation_env(frame_skip: int, max_steps: int, run_dir: Path):
    env = DummyVecEnv([
        lambda: MarioEnv(frame_skip=frame_skip, max_steps=max_steps)
    ])

    stats_path = run_dir / "vecnormalize.pkl"
    if stats_path.is_file():
        try:
            env = VecNormalize.load(str(stats_path), env)
        except Exception:
            env.close()
            raise
        env.training = False
        env.norm_reward = False
        print(f"[evaluate] Estadisticas de normalizacion: {stats_path}")
    else:
        print("[evaluate] Sin vecnormalize.pkl: observaciones y recompensas originales.")
    return env


def run_episodes(
    env, model, agent_name: str, episodes: int, seed: int, deterministic: bool
) -> tuple[list[dict], Counter]:
    results = []
    counts: Counter = Counter()

    for episode in range(1, episodes + 1):
        episode_seed = seed + episode - 1
        env.seed(episode_seed)
        observation = env.reset()
        done = False
        total_reward = 0.0
        steps = 0
        info = {}
        rng = np.random.default_rng(episode_seed)

        while not done:
            if model is None:
                action = np.array([rng.integers(len(ACTIONS))], dtype=np.int64)
            else:
                action, _ = model.predict(observation, deterministic=deterministic)

            action_index = int(np.asarray(action).reshape(-1)[0])
            counts[action_index] += 1
            observation, rewards, dones, infos = env.step(action)
            total_reward += float(rewards[0])
            steps += 1
            done = bool(dones[0])
            info = infos[0]

        status = info.get("status", "running")
        if status == "win":
            terminal_state = "win"
        elif status == "death":
            terminal_state = "death"
        elif info.get("TimeLimit.truncated", False) or status == "running":
            terminal_state = "timeout"
        else:
            terminal_state = "other"

        result = {
            "agente": agent_name,
            "episodio": episode,
            "semilla": episode_seed,
            "recompensa_total": round(total_reward, 4),
            "pasos": steps,
            "estado_final": terminal_state,
            "victoria": int(terminal_state == "win"),
            "muerte": int(terminal_state == "death"),
            "limite_pasos": int(terminal_state == "timeout"),
            "max_x": round(float(info.get("max_x", 0.0)), 2),
            "x_final": round(float(info.get("x_pos", 0.0)), 2),
            "monedas": int(info.get("coins", 0)),
            "puntuacion": int(info.get("score", 0)),
            "vidas": int(info.get("lives", 0)),
            "tiempo_simulado_s": round(float(info.get("sim_clock", 0)) / 1000.0, 3),
        }
        results.append(result)
        print(
            f"[{agent_name}] {episode:02d}/{episodes} | {terminal_state:7s} "
            f"| reward={total_reward:9.2f} | x_max={result['max_x']:8.1f} "
            f"| pasos={steps}"
        )

    return results, counts


def wilson_interval(wins: int, total: int, z: float = 1.96) -> list[float]:
    p = wins / total
    denom = 1 + (z * z) / total
    center = (p + (z * z) / (2 * total)) / denom
    margin = z * math.sqrt(p * (1 - p) / total + (z * z) / (4 * total * total)) / denom
    return [round(max(0.0, center - margin), 4), round(min(1.0, center + margin), 4)]


def mean_sd(rows: list[dict], key: str) -> dict:
    values = np.array([float(row[key]) for row in rows], dtype=float)
    return {
        "media": round(float(np.mean(values)), 4),
        "desviacion_estandar": round(
            float(np.std(values, ddof=1)) if len(values) > 1 else 0.0, 4
        ),
    }


def summarize(rows: list[dict], action_counts: Counter) -> dict:
    total = len(rows)
    wins = sum(row["victoria"] for row in rows)
    deaths = sum(row["muerte"] for row in rows)
    timeouts = sum(row["limite_pasos"] for row in rows)
    other = total - wins - deaths - timeouts
    decisions = sum(action_counts.values())

    return {
        "episodios": total,
        "victorias": wins,
        "muertes": deaths,
        "truncados": timeouts,
        "otros": other,
        "tasa_victorias": round(wins / total, 4),
        "ic_95_tasa_victorias_wilson": wilson_interval(wins, total),
        "recompensa": mean_sd(rows, "recompensa_total"),
        "avance_maximo_px": mean_sd(rows, "max_x"),
        "pasos": mean_sd(rows, "pasos"),
        "monedas": mean_sd(rows, "monedas"),
        "puntuacion": mean_sd(rows, "puntuacion"),
        "acciones": {
            str(action): {
                "cantidad": action_counts[action],
                "proporcion": round(action_counts[action] / decisions, 4) if decisions else 0.0,
            }
            for action in range(len(ACTIONS))
        },
    }


def save_results(output_dir: Path, results: list[dict], summary: dict) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "episodes.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(results)
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)


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
    if frame_skip < 1 or max_steps < 1:
        raise ValueError("frame_skip y max_steps deben ser enteros positivos")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    default_output = run_dir / "evaluations" / f"{model_path.stem}_{timestamp}_seed{args.seed}"
    output_dir = (args.output_dir or default_output).expanduser().resolve()

    if not (Path(GAME_DIR) / "Core.py").is_file():
        raise FileNotFoundError(
            f"Falta el juego en {GAME_DIR}. Se necesita game/MarioPygame/Next/Core.py "
            "y sus recursos; solicita al equipo la carpeta del juego."
        )

    set_random_seed(args.seed)
    print(f"[evaluate] Modelo: {model_path}")
    print(f"[evaluate] Episodios: {args.episodes}, seed: {args.seed}, "
          f"frame_skip: {frame_skip}, max_steps: {max_steps}")
    model = PPO.load(str(model_path), device="cpu")
    env = make_evaluation_env(frame_skip, max_steps, run_dir)
    try:
        if (model.observation_space != env.observation_space
                or model.action_space != env.action_space):
            raise ValueError("Espacios del modelo y MarioEnv incompatibles: "
                             "revisa el modelo, las acciones y las observaciones.")

        ppo_rows, ppo_actions = run_episodes(
            env, model, "PPO", args.episodes, args.seed,
            deterministic=not args.stochastic,
        )
        all_rows = list(ppo_rows)
        summaries = {"PPO": summarize(ppo_rows, ppo_actions)}

        if args.baseline_random:
            rand_rows, rand_actions = run_episodes(
                env, None, "aleatorio", args.episodes, args.seed, deterministic=False
            )
            all_rows.extend(rand_rows)
            summaries["aleatorio"] = summarize(rand_rows, rand_actions)
    finally:
        env.close()

    report = {
        "proyecto": "Final-Project-AI / Mario",
        "algoritmo": "PPO",
        "modelo": str(model_path),
        "corrida": run_dir.name,
        "config_entrenamiento": config,
        "config_evaluacion": {
            "episodios_por_agente": args.episodes,
            "seed_base": args.seed,
            "frame_skip": frame_skip,
            "max_steps": max_steps,
            "ppo_deterministico": not args.stochastic,
            "baseline_aleatoria": args.baseline_random,
            "recompensas_sin_normalizacion": True,
        },
        "resumen": summaries,
    }
    save_results(output_dir, all_rows, report)

    print("\n=== RESULTADOS ===")
    for name, summary in summaries.items():
        lo, hi = summary["ic_95_tasa_victorias_wilson"]
        print(f"{name}: victorias={summary['victorias']}/{summary['episodios']} "
              f"({summary['tasa_victorias']:.1%}; IC95% [{lo:.1%}, {hi:.1%}]) | "
              f"reward={summary['recompensa']['media']:.2f} | "
              f"avance={summary['avance_maximo_px']['media']:.1f} px")
    print(f"[evaluate] CSV:  {output_dir / 'episodes.csv'}")
    print(f"[evaluate] JSON: {output_dir / 'summary.json'}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(f"[error] {exc}") from exc
