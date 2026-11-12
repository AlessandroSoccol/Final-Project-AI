"""
champion.py

Gestiona el "campeon global": un unico modelo persistente que guarda el MEJOR
resultado historico de todas las corridas de entrenamiento.

Hay dos carriles de modelo en el proyecto:

  1) Carril efimero  -> models/<run-name>/   (cada corrida arranca de CERO y
                        guarda su propio final_model.zip / best_model.zip)
  2) Carril campeon  -> models/champion/     (UNICO; solo se actualiza cuando
                        una corrida produce algo MEJOR que el campeon actual)

Criterio de "mejor" (criterio B):
    1. Mayor tasa de victorias (fraccion de episodios que llegan al flag).
    2. En caso de empate, mayor avance horizontal medio (max_x).

El campeon guarda:
    models/champion/champion.zip          -> los pesos del modelo (PPO.save)
    models/champion/vecnormalize.pkl      -> estadisticas de normalizacion
    models/champion/champion_meta.json    -> metricas y procedencia
"""

import json
import os
import shutil
from datetime import datetime


CHAMPION_DIRNAME = "champion"
CHAMPION_MODEL = "champion.zip"
CHAMPION_VECNORM = "vecnormalize.pkl"
CHAMPION_META = "champion_meta.json"


def champion_dir(models_dir: str) -> str:
    return os.path.join(models_dir, CHAMPION_DIRNAME)


def champion_model_path(models_dir: str) -> str:
    return os.path.join(champion_dir(models_dir), CHAMPION_MODEL)


def load_champion_meta(models_dir: str) -> dict | None:
    """Devuelve las metricas del campeon actual, o None si no existe."""
    meta_path = os.path.join(champion_dir(models_dir), CHAMPION_META)
    if not os.path.isfile(meta_path):
        return None
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def is_better(candidate: dict, current: dict | None) -> bool:
    """
    Decide si 'candidate' supera al campeon 'current' (criterio B).

    candidate / current son dicts con al menos:
        - "win_rate"   (float, 0..1)
        - "mean_max_x" (float, pixeles)
    """
    # Sin campeon previo, cualquier candidato valido es mejor.
    if current is None:
        return True

    c_win = float(candidate.get("win_rate", 0.0))
    p_win = float(current.get("win_rate", 0.0))

    # 1) Mas victorias gana.
    if c_win != p_win:
        return c_win > p_win

    # 2) Empate en victorias -> mayor avance medio gana.
    c_x = float(candidate.get("mean_max_x", 0.0))
    p_x = float(current.get("mean_max_x", 0.0))
    return c_x > p_x


def promote(
    models_dir: str,
    model_src: str,
    vecnorm_src: str | None,
    candidate_meta: dict,
    run_name: str,
) -> None:
    """
    Copia el modelo (y su vecnormalize) a models/champion/ y escribe el meta.

    model_src   : ruta al .zip del modelo de la corrida que gano.
    vecnorm_src : ruta al vecnormalize.pkl de esa corrida (o None).
    candidate_meta : metricas medidas del candidato.
    run_name    : nombre de la corrida de la que proviene (trazabilidad).
    """
    dest = champion_dir(models_dir)
    os.makedirs(dest, exist_ok=True)

    # Normaliza la ruta del modelo (PPO.save a veces recibe sin .zip).
    if not model_src.endswith(".zip") and os.path.isfile(model_src + ".zip"):
        model_src = model_src + ".zip"

    shutil.copyfile(model_src, os.path.join(dest, CHAMPION_MODEL))

    if vecnorm_src and os.path.isfile(vecnorm_src):
        shutil.copyfile(vecnorm_src, os.path.join(dest, CHAMPION_VECNORM))

    meta = dict(candidate_meta)
    meta["run_name"] = run_name
    meta["promoted_at"] = datetime.now().isoformat(timespec="seconds")
    with open(os.path.join(dest, CHAMPION_META), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


def maybe_promote(
    models_dir: str,
    model_src: str,
    vecnorm_src: str | None,
    candidate_meta: dict,
    run_name: str,
    verbose: bool = True,
) -> bool:
    """
    Promueve el candidato a campeon SOLO si supera al campeon actual.

    Devuelve True si hubo promocion.
    """
    current = load_champion_meta(models_dir)
    if is_better(candidate_meta, current):
        promote(models_dir, model_src, vecnorm_src, candidate_meta, run_name)
        if verbose:
            prev = "ninguno" if current is None else (
                f"win_rate={current.get('win_rate', 0.0):.2%}, "
                f"avance={current.get('mean_max_x', 0.0):.0f}px"
            )
            print(
                f"[champion] NUEVO CAMPEON promovido desde '{run_name}': "
                f"win_rate={candidate_meta.get('win_rate', 0.0):.2%}, "
                f"avance={candidate_meta.get('mean_max_x', 0.0):.0f}px "
                f"(anterior: {prev})"
            )
        return True

    if verbose:
        print(
            f"[champion] '{run_name}' NO supera al campeon actual "
            f"(candidato win_rate={candidate_meta.get('win_rate', 0.0):.2%}, "
            f"avance={candidate_meta.get('mean_max_x', 0.0):.0f}px). "
            "Se conserva el campeon existente."
        )
    return False
