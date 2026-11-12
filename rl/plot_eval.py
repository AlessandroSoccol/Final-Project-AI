import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt

RL_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = RL_DIR.parent
MODELS_DIR = PROJECT_ROOT / "models"
FIGURES_ROOT = RL_DIR / "figuras"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Grafica los resultados de evaluate.py (episodes.csv / summary.json)."
    )
    parser.add_argument(
        "--eval-dir", type=Path, default=None,
        help="Carpeta de evaluacion a graficar; si se omite usa la mas reciente.",
    )
    parser.add_argument(
        "--show", action="store_true",
        help="Mostrar las figuras en pantalla ademas de guardarlas.",
    )
    return parser.parse_args()


def find_latest_eval_dir() -> Path:
    #La carpeta mas recientemente evaluada
    candidates = list(MODELS_DIR.glob("*/evaluations/*/episodes.csv"))
    candidates += list(MODELS_DIR.glob("evaluations/*/episodes.csv"))
    
    #Si no encuentra candidatos a graficar, solicita al usuario evaluarlos
    if not candidates:
        raise FileNotFoundError(
            "No se encontraron evaluaciones. Ejecuta evaluate.py primero, o "
            "pasa --eval-dir con la ruta a una carpeta que tenga episodes.csv."
        )
    newest = max(candidates, key=lambda p: p.stat().st_mtime)
    return newest.parent


def load_episodes(eval_dir: Path) -> list[dict]:
    csv_path = eval_dir / "episodes.csv"
    if not csv_path.is_file():
        raise FileNotFoundError(f"No existe {csv_path}")
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def load_summary(eval_dir: Path) -> dict:
    json_path = eval_dir / "summary.json"
    if not json_path.is_file():
        return {}
    with json_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def rows_by_agent(rows: list[dict]) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row.get("agente", "agente")].append(row)
    return grouped


def plot_metric_per_episode(grouped, key, ylabel, title, out_path):
    #Grafica una metrica pro episodio, una linea por agente
    plt.figure(figsize=(9, 5))
    for agent, rows in grouped.items():
        episodes = [int(r["episodio"]) for r in rows]
        values = [float(r[key]) for r in rows]
        plt.plot(episodes, values, marker="o", markersize=3, label=agent)
    plt.xlabel("Episodio")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()


def plot_state_distribution(grouped, out_path):
    agents = list(grouped.keys())
    wins = [sum(int(r["victoria"]) for r in grouped[a]) for a in agents]
    deaths = [sum(int(r["muerte"]) for r in grouped[a]) for a in agents]
    timeouts = [sum(int(r["limite_pasos"]) for r in grouped[a]) for a in agents]

    plt.figure(figsize=(7, 5))
    x = range(len(agents))
    plt.bar(x, wins, label="Victorias", color="#2ca02c")
    plt.bar(x, deaths, bottom=wins, label="Muertes", color="#d62728")
    bottom2 = [w + d for w, d in zip(wins, deaths)]
    plt.bar(x, timeouts, bottom=bottom2, label="Timeout", color="#7f7f7f")
    plt.xticks(list(x), agents)
    plt.ylabel("Episodios")
    plt.title("Distribucion de resultados por agente")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()


def plot_agent_summary(summary, out_path):
    #Toma los datos de summary.json y de alli compara la taza de victorias y el avance medio
    resumen = summary.get("resumen", {})
    if not resumen:
        return False

    agents = list(resumen.keys())
    win_rates = [resumen[a].get("tasa_victorias", 0.0) * 100 for a in agents]
    avances = [resumen[a].get("avance_maximo_px", {}).get("media", 0.0) for a in agents]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))

    ax1.bar(agents, win_rates, color="#2ca02c")
    ax1.set_ylabel("Tasa de victorias (%)")
    ax1.set_title("Victorias por agente")
    ax1.set_ylim(0, 100)
    for i, v in enumerate(win_rates):
        ax1.text(i, v + 1, f"{v:.1f}%", ha="center")

    ax2.bar(agents, avances, color="#1f77b4")
    ax2.set_ylabel("Avance maximo medio (px)")
    ax2.set_title("Avance medio por agente")
    for i, v in enumerate(avances):
        ax2.text(i, v, f"{v:.0f}", ha="center", va="bottom")

    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()
    return True


def main() -> int:
    args = parse_args()
    if not args.show:
        matplotlib.use("Agg")  # backend sin ventana para solo guardar PNG

    eval_dir = (args.eval_dir.expanduser().resolve()
                if args.eval_dir else find_latest_eval_dir())
    print(f"[plot] Carpeta de evaluacion: {eval_dir}")

    rows = load_episodes(eval_dir)
    summary = load_summary(eval_dir)
    grouped = rows_by_agent(rows)

    # Crea una subcarpeta para no confundir resultados
    figures_dir = FIGURES_ROOT / eval_dir.name
    figures_dir.mkdir(parents=True, exist_ok=True)

    plot_metric_per_episode(
        grouped, "max_x", "Avance maximo (px)",
        "Avance por episodio", figures_dir / "avance_por_episodio.png",
    )
    plot_metric_per_episode(
        grouped, "recompensa_total", "Recompensa total",
        "Recompensa por episodio", figures_dir / "recompensa_por_episodio.png",
    )
    plot_state_distribution(grouped, figures_dir / "distribucion_estados.png")
    has_summary = plot_agent_summary(summary, figures_dir / "resumen_agentes.png")

    print(f"[plot] Figuras guardadas en: {figures_dir}")
    print("  - avance_por_episodio.png")
    print("  - recompensa_por_episodio.png")
    print("  - distribucion_estados.png")
    if has_summary:
        print("  - resumen_agentes.png")

    if args.show:
        import matplotlib.image as mpimg
        for png in sorted(figures_dir.glob("*.png")):
            img = mpimg.imread(png)
            plt.figure(figsize=(9, 5))
            plt.imshow(img)
            plt.axis("off")
            plt.title(png.name)
        plt.show()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
