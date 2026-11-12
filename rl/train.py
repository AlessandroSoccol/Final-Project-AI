import argparse
import json
import os
import sys
import time

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize

from callbacks import make_callbacks
from champion import maybe_promote
from mario_env import MarioEnv

RL_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(RL_DIR)
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
TB_DIR = os.path.join(PROJECT_ROOT, "runs")

PPO_PARAMS = dict(
    learning_rate=2.5e-4,
    n_steps=512,
    batch_size=256,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.03,  # mas exploracion: evita que la politica colapse a una accion
    vf_coef=0.5,
    max_grad_norm=0.5,
)

POLICY_KWARGS = dict(net_arch=dict(pi=[64, 64], vf=[64, 64]))


def parse_args():
    p = argparse.ArgumentParser(description="Entrena PPO en MarioEnv")
    p.add_argument("--timesteps", type=int, default=2_000_000)
    p.add_argument("--n-envs", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--frame-skip", type=int, default=4)
    p.add_argument("--max-steps", type=int, default=3000)
    p.add_argument("--run-name", type=str, default=None)
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--no-norm-reward", action="store_true")
    p.add_argument(
        "--no-champion", action="store_true",
        help="No comparar ni actualizar el campeon global al terminar.",
    )
    return p.parse_args()


def align_to_envs(freq, n_envs):
    return max(n_envs, (freq // n_envs) * n_envs)


def make_env(args):
    return make_vec_env(
        MarioEnv,
        n_envs=args.n_envs,
        seed=args.seed,
        vec_env_cls=SubprocVecEnv,
        env_kwargs=dict(frame_skip=args.frame_skip, max_steps=args.max_steps),
    )


def wrap_normalize(env, resume):
    if resume:
        stats = os.path.join(os.path.dirname(os.path.abspath(resume)), "vecnormalize.pkl")
        if os.path.exists(stats):
            env = VecNormalize.load(stats, env)
            env.training = True
            return env
    return VecNormalize(env, norm_obs=False, norm_reward=True, gamma=PPO_PARAMS["gamma"])


def main():
    args = parse_args()
    torch.set_num_threads(1)

    run_name = args.run_name or f"ppo_seed{args.seed}_{time.strftime('%Y%m%d_%H%M%S')}"
    run_dir = os.path.join(MODELS_DIR, run_name)
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(TB_DIR, exist_ok=True)

    print(f"[train] corrida: {run_name} | {args.n_envs} entornos | {args.timesteps:,} pasos")

    env = make_env(args)
    use_norm = not args.no_norm_reward
    if use_norm:
        env = wrap_normalize(env, args.resume)

    if args.resume:
        model = PPO.load(args.resume, env=env, device="cpu", tensorboard_log=TB_DIR)
    else:
        model = PPO(
            "MlpPolicy",
            env,
            policy_kwargs=POLICY_KWARGS,
            tensorboard_log=TB_DIR,
            seed=args.seed,
            device="cpu",
            verbose=1,
            **PPO_PARAMS,
        )

    config = dict(vars(args), run_name=run_name, ppo_params=PPO_PARAMS,
                  policy_kwargs=POLICY_KWARGS)
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    callbacks, metrics_cb = make_callbacks(
        save_path=run_dir,
        checkpoint_freq=max(1, 100_000 // args.n_envs),
        best_check_freq=align_to_envs(10_000, args.n_envs),
        progress_freq=align_to_envs(10_000, args.n_envs),
        window=100,
        verbose=1,
    )

    final_path = os.path.join(run_dir, "final_model")
    start = time.time()
    try:
        model.learn(
            total_timesteps=args.timesteps,
            callback=callbacks,
            tb_log_name=run_name,
            reset_num_timesteps=args.resume is None,
        )
    except KeyboardInterrupt:
        print("\n[train] interrumpido, guardando modelo...")
    finally:
        model.save(final_path)
        if use_norm:
            env.save(os.path.join(run_dir, "vecnormalize.pkl"))
        env.close()

    print(f"[train] listo en {(time.time() - start) / 60:.1f} min")
    print(f"[train] modelo final: {final_path}.zip")
    print(f"[train] mejor modelo: {os.path.join(run_dir, 'best_model')}.zip")

    # --- Carril campeon global ------------------------------------------
    # Comparamos las metricas de esta corrida con el campeon historico y,
    # si lo superamos (mas victorias; desempate por avance), lo reemplazamos.
    # Usamos best_model si existe (suele ser mejor que el final); si no, final.
    if not args.no_champion:
        candidate_meta = metrics_cb.current_metrics()
        best_model_path = os.path.join(run_dir, "best_model.zip")
        candidate_model = best_model_path if os.path.exists(best_model_path) else final_path
        vecnorm_path = os.path.join(run_dir, "vecnormalize.pkl") if use_norm else None
        maybe_promote(
            models_dir=MODELS_DIR,
            model_src=candidate_model,
            vecnorm_src=vecnorm_path,
            candidate_meta=candidate_meta,
            run_name=run_name,
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
