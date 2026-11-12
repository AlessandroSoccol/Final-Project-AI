import argparse
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

RL_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = RL_DIR.parent
RUNS_DIR = PROJECT_ROOT / "runs"
FIGURES_ROOT = RL_DIR / "figuras"

# Posicion X de la bandera (meta) en px; sirve para convertir avance -> %.
GOAL_X = 6336.0

TAG_ADVANCE = "rollout/mario_max_x"
TAG_WINRATE = "rollout/mario_win_rate"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Grafica la curva de aprendizaje en una corrida"
    )
    parser.add_argument(
        "--run", type=str, default=None,
        help="Nombre de la carpeta en runs"
             "Por defecto usa la corrida mas reciente.",
    )
    parser.add_argument(
        "--show", action="store_true",
        help="Mostrar la figura en pantalla ademas de guardarla.",
    )
    return parser.parse_args()


def find_latest_run() -> Path:
    runs = [d for d in RUNS_DIR.glob("*") if d.is_dir()]
    if not runs:
        raise FileNotFoundError(
            f"No hay corridas en {RUNS_DIR}. Entrena con train.py primero."
        )
    return max(runs, key=lambda d: d.stat().st_mtime)


def load_series(run_dir: Path, tag: str):
    ea = EventAccumulator(str(run_dir))
    ea.Reload()
    available = ea.Tags()["scalars"]
    if tag not in available:
        raise KeyError(
            f"El tag '{tag}' no esta en {run_dir.name}. "
            f"Tags disponibles: {available}"
        )
    events = ea.Scalars(tag)
    steps = [e.step for e in events]
    values = [e.value for e in events]
    return steps, values


def main() -> int:
    args = parse_args()
    if not args.show:
        matplotlib.use("Agg")

    run_dir = (RUNS_DIR / args.run) if args.run else find_latest_run()
    if not run_dir.is_dir():
        raise FileNotFoundError(f"No existe la corrida: {run_dir}")
    print(f"[plot] Corrida: {run_dir.name}")

    steps_adv, adv_px = load_series(run_dir, TAG_ADVANCE)
    steps_win, win_rate = load_series(run_dir, TAG_WINRATE)

    # Convertir a porcentajes.
    adv_pct = [min(100.0, v / GOAL_X * 100.0) for v in adv_px]
    win_pct = [v * 100.0 for v in win_rate]

    # Dos curvas en una figura, compartiendo el eje X logaritmico.
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(steps_adv, adv_pct, label="% de avance (hacia la meta)",
            color="#1f77b4", linewidth=2)
    ax.plot(steps_win, win_pct, label="% de exito (nivel completado)",
            color="#2ca02c", linewidth=2)

    ax.set_xscale("log")  # eje X logaritmico: pasos de entrenamiento
    ax.set_xlabel("Pasos de entrenamiento (escala logaritmica)")
    ax.set_ylabel("Porcentaje (%)")
    ax.set_ylim(0, 105)
    ax.set_title(f"Curva de aprendizaje — {run_dir.name}")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="upper left")

    figures_dir = FIGURES_ROOT / run_dir.name
    figures_dir.mkdir(parents=True, exist_ok=True)
    out_path = figures_dir / "curva_aprendizaje.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    print(f"[plot] Figura guardada en: {out_path}")

    if args.show:
        plt.show()
    plt.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
