#!/usr/bin/env python3
"""
pso_optimize.py — Particle Swarm Optimization over simulation config parameters.

Each particle is a candidate Config (vector of continuous parameters). All
particles are evaluated in parallel — one worker process per particle — by
running a short simulation trial and scoring on love emergence, social
structure, and population survival.

The swarm learns from personal bests and the global best, converging on the
parameter region most likely to produce bonded pairs (love).

Usage
─────
    python pso_optimize.py                      # 16 particles, 20 iters, 15k ticks
    python pso_optimize.py --particles 8        # fewer workers (less RAM)
    python pso_optimize.py --iters 30           # more generations
    python pso_optimize.py --eval-ticks 20000   # longer, more accurate trials
    python pso_optimize.py --out-dir my_run     # custom output directory

Outputs
───────
    pso_results/pso_log.json        — full generation history
    pso_results/best_config.json    — best parameters found
"""

import argparse
import json
import os
import sys
import time
import multiprocessing as mp

import numpy as np

# ── Make project root importable inside worker processes ─────────────────────
_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


# ─────────────────────────────────────────────────────────────────────────────
# Search space
#
# Each row: (param_name, lower_bound, upper_bound, is_integer)
#
# FIXED (not searched): max_visible_agents, num_signals, hidden_size
#   — changing these alters brain input/output dimensions, making configs
#     incompatible across particles.
# ─────────────────────────────────────────────────────────────────────────────
PARAMS = [
    # World & resources
    ("resource_regen_rate",   0.01,   0.12,  False),
    ("patch_peak",            4.0,   14.0,  False),
    ("num_resource_patches",  4,     20,    True),
    # Threats
    ("threat_damage",         5.0,   35.0,  False),
    ("threat_prob_per_tick",  5e-4,   6e-3, False),
    # Agent metabolism
    ("metabolic_cost",        0.10,   0.60,  False),
    ("eat_amount",            2.0,    8.0,  False),
    # Social economy — the key levers for love emergence
    ("share_amount",          3.0,   15.0,  False),
    ("attack_steal",          2.0,   12.0,  False),
    ("attack_cost",           1.0,   10.0,  False),
    # Reproduction
    ("mate_cost",             8.0,   30.0,  False),
    ("mate_energy_threshold", 40.0,  80.0,  False),
    # Perception & memory
    ("vision_radius",         8.0,   25.0,  False),
    ("memory_capacity",       8,     30,    True),
]

NAMES  = [p[0] for p in PARAMS]
LO     = np.array([p[1] for p in PARAMS], dtype=float)
HI     = np.array([p[2] for p in PARAMS], dtype=float)
IS_INT = [p[3] for p in PARAMS]
N_DIM  = len(PARAMS)


# ─────────────────────────────────────────────────────────────────────────────
# Config helpers
# ─────────────────────────────────────────────────────────────────────────────

def pos_to_config(pos: np.ndarray):
    """Convert a PSO position vector to a Config object."""
    from config import Config
    cfg = Config()
    for i, (name, lo, hi, is_int) in enumerate(PARAMS):
        val = float(np.clip(pos[i], lo, hi))
        if is_int:
            val = int(round(val))
        setattr(cfg, name, val)
    # Keep brain architecture fixed across all particles
    cfg.max_visible_agents = 8
    cfg.num_signals        = 8
    cfg.hidden_size        = 48
    # Disable checkpointing and rendering in trial runs
    cfg.checkpoint_interval = 10 ** 9
    cfg.render_interval     = 10 ** 9
    return cfg


def current_config_pos() -> np.ndarray:
    """Read current config.py defaults into a position vector (warm-start)."""
    from config import Config
    cfg = Config()
    pos = np.zeros(N_DIM)
    for i, (name, lo, hi, is_int) in enumerate(PARAMS):
        pos[i] = float(np.clip(getattr(cfg, name), lo, hi))
    return pos


# ─────────────────────────────────────────────────────────────────────────────
# Fitness evaluation  (runs inside a worker process)
# ─────────────────────────────────────────────────────────────────────────────

def _evaluate_worker(args):
    """
    Run one simulation trial and return (fitness, summary_dict).

    stdout/stderr are redirected to /dev/null so parallel workers don't
    produce interleaved output on the console.
    """
    particle_id, pos, seed, eval_ticks, work_dir = args

    # Silence simulation output from this worker
    _null = open(os.devnull, "w")
    _stdout, _stderr = sys.stdout, sys.stderr
    sys.stdout = sys.stderr = _null

    try:
        from config import Config
        from environment.world import World
        from simulation.runner import Simulation
        from simulation.metrics import MetricsTracker

        cfg = pos_to_config(pos)
        cfg.max_ticks       = eval_ticks
        cfg.metrics_interval = max(250, eval_ticks // 60)

        rng     = np.random.default_rng(seed)
        world   = World(cfg, rng=rng)
        p_dir   = os.path.join(work_dir, f"p{particle_id:02d}")
        os.makedirs(p_dir, exist_ok=True)
        metrics = MetricsTracker(cfg, out_dir=os.path.join(p_dir, "metrics"))
        sim     = Simulation(cfg, world, metrics=metrics, renderer=None, seed=seed)
        sim.run(ticks=eval_ticks)
        history = metrics.history

    except Exception as exc:
        sys.stdout, sys.stderr = _stdout, _stderr
        _null.close()
        return -1000.0, {"error": str(exc)[:120], "particle": particle_id}
    finally:
        sys.stdout, sys.stderr = _stdout, _stderr
        _null.close()

    if not history:
        return -900.0, {"error": "no metrics recorded", "particle": particle_id}

    # ── Extinction penalty ────────────────────────────────────────────────────
    final_pop   = history[-1].get("pop", 0)
    survival    = final_pop / max(1, cfg.initial_population)
    if final_pop < 15:
        return -500.0 + survival * 20.0, {"extinction": True, "particle": particle_id}

    # ── Evaluate over the tail (last 40% of snapshots) for stable signal ─────
    tail = history[int(len(history) * 0.6):]

    bonded_pairs = float(np.mean([h.get("bonded_pairs",       0) for h in tail]))
    mean_bond    = float(np.mean([h.get("mean_bond_score",    0) for h in tail]))
    max_bond     = float(np.max( [h.get("max_bond_score",     0) for h in tail]))
    same_val     = float(np.mean([h.get("same_signal_mean_valence", 0) for h in tail]))
    tribal_bias  = float(np.mean([abs(h.get("tribal_bias",    0)) for h in tail]))
    ritual_conv  = float(np.mean([h.get("ritual_convergence", 0) for h in tail]))

    # ── Fitness ───────────────────────────────────────────────────────────────
    # Love terms dominate; proxy metrics (same_val, max_bond) give gradient
    # even before the first threshold-crossing bond appears.
    fitness = (
        50.0  * bonded_pairs +    # primary: actual bonded pairs
        100.0 * mean_bond    +    # bond strength (0–1 × 100)
        40.0  * max_bond     +    # any pair ever close to bonding
         8.0  * max(0.0, same_val) +  # reward positive intra-group valence
         2.0  * tribal_bias  +    # reward social structure
         1.0  * ritual_conv  +    # reward ritual behaviour
         3.0  * survival          # keep population alive
    )

    summary = {
        "particle":     particle_id,
        "bonded_pairs": round(bonded_pairs, 3),
        "mean_bond":    round(mean_bond,    5),
        "max_bond":     round(max_bond,     5),
        "same_val":     round(same_val,     5),
        "tribal_bias":  round(tribal_bias,  4),
        "ritual_conv":  round(ritual_conv,  3),
        "survival":     round(survival,     3),
        "fitness":      round(fitness,      4),
    }
    return float(fitness), summary


# ─────────────────────────────────────────────────────────────────────────────
# PSO
# ─────────────────────────────────────────────────────────────────────────────

class PSO:
    """
    Standard PSO with:
      - Inertia weight linearly annealed from w_hi → w_lo  (explore → exploit)
      - Cognitive weight c1 pulls each particle toward its personal best
      - Social weight c2 pulls each particle toward the global best
      - Velocity clamped to ±20% of each parameter's range per step
      - Particle 0 warm-started at the current config.py defaults
    """

    def __init__(self, n_particles: int, n_iterations: int,
                 eval_ticks: int, out_dir: str, seed: int = 42):
        self.n_particles  = n_particles
        self.n_iterations = n_iterations
        self.eval_ticks   = eval_ticks
        self.out_dir      = out_dir
        self.rng          = np.random.default_rng(seed)
        os.makedirs(out_dir, exist_ok=True)

        # PSO hyper-parameters (Clerc & Kennedy 2002 canonical values)
        self.w_hi = 0.90    # starting inertia — high exploration
        self.w_lo = 0.35    # ending inertia   — high exploitation
        self.c1   = 1.494   # cognitive (personal best)
        self.c2   = 1.494   # social    (global  best)

        # ── Initialise swarm ─────────────────────────────────────────────────
        span = HI - LO
        self.positions  = self.rng.uniform(LO, HI, size=(n_particles, N_DIM))
        # Particle 0: warm-start at current config.py defaults
        self.positions[0] = current_config_pos()
        # Small random initial velocities (±5% of range)
        self.velocities = self.rng.uniform(-0.05 * span, 0.05 * span,
                                           size=(n_particles, N_DIM))

        self.pbest_pos = self.positions.copy()
        self.pbest_fit = np.full(n_particles, -np.inf)
        self.gbest_pos = self.positions[0].copy()
        self.gbest_fit = -np.inf

        self.log: list = []

    # ── Inertia schedule ─────────────────────────────────────────────────────
    def _w(self, iteration: int) -> float:
        t = iteration / max(1, self.n_iterations - 1)
        return self.w_hi + t * (self.w_lo - self.w_hi)

    # ── Main loop ─────────────────────────────────────────────────────────────
    def run(self, pool: mp.Pool) -> dict:
        log_path      = os.path.join(self.out_dir, "pso_log.json")
        best_cfg_path = os.path.join(self.out_dir, "best_config.json")

        for iteration in range(self.n_iterations):
            t0  = time.time()
            w   = self._w(iteration)
            gen = iteration + 1

            print(f"\n{'═' * 65}")
            print(f"  Generation {gen}/{self.n_iterations}  "
                  f"│  w={w:.3f}  │  best so far: {self.gbest_fit:.4f}")
            print(f"{'═' * 65}")

            # ── Evaluate all particles in parallel ────────────────────────────
            seeds = [int(self.rng.integers(0, 2 ** 31))
                     for _ in range(self.n_particles)]
            tasks = [
                (i, self.positions[i], seeds[i], self.eval_ticks, self.out_dir)
                for i in range(self.n_particles)
            ]
            results = pool.map(_evaluate_worker, tasks)

            # ── Update personal and global bests ──────────────────────────────
            newly_improved = False
            for i, (fit, summary) in enumerate(results):
                tag = f"  P{i:02d}  fit={fit:9.3f}"
                if "error" in summary:
                    tag += f"  [ERR: {summary['error'][:50]}]"
                elif summary.get("extinction"):
                    tag += "  [EXTINCT]"
                else:
                    tag += (f"  bonds={summary['bonded_pairs']:.2f}"
                            f"  max_bond={summary['max_bond']:.4f}"
                            f"  val={summary['same_val']:+.4f}"
                            f"  surv={summary['survival']:.2f}")
                print(tag)

                if fit > self.pbest_fit[i]:
                    self.pbest_fit[i] = fit
                    self.pbest_pos[i] = self.positions[i].copy()

                if fit > self.gbest_fit:
                    self.gbest_fit = fit
                    self.gbest_pos = self.positions[i].copy()
                    newly_improved = True

            # ── Velocity + position update ────────────────────────────────────
            r1  = self.rng.uniform(0, 1, (self.n_particles, N_DIM))
            r2  = self.rng.uniform(0, 1, (self.n_particles, N_DIM))
            cog = self.c1 * r1 * (self.pbest_pos - self.positions)
            soc = self.c2 * r2 * (self.gbest_pos - self.positions)

            self.velocities = w * self.velocities + cog + soc
            v_max = 0.20 * (HI - LO)                          # ±20% per step
            self.velocities = np.clip(self.velocities, -v_max, v_max)
            self.positions  = np.clip(self.positions + self.velocities, LO, HI)

            # ── Log & persist every generation ───────────────────────────────
            elapsed  = time.time() - t0
            best_cfg = pos_to_config(self.gbest_pos)
            entry = {
                "generation":      gen,
                "best_fitness":    float(self.gbest_fit),
                "elapsed_s":       round(elapsed, 1),
                "improved":        newly_improved,
                "best_params":     {n: getattr(best_cfg, n) for n in NAMES},
                "all_fitnesses":   [round(float(r[0]), 4) for r in results],
                "all_summaries":   [r[1] for r in results],
            }
            self.log.append(entry)

            with open(log_path, "w") as f:
                json.dump(self.log, f, indent=2)
            with open(best_cfg_path, "w") as f:
                json.dump(entry["best_params"], f, indent=2)

            marker = "★ NEW BEST" if newly_improved else "─"
            print(f"\n  {marker}  fitness={self.gbest_fit:.4f}  "
                  f"wall={elapsed:.0f}s")
            print("  Best params this gen:")
            for n in NAMES:
                print(f"    {n:<30s} {getattr(best_cfg, n):.5g}")

        return {
            "best_fitness": float(self.gbest_fit),
            "best_params":  {n: getattr(pos_to_config(self.gbest_pos), n)
                             for n in NAMES},
            "log": self.log,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def _print_final(result: dict) -> None:
    from config import Config
    defaults = Config()

    print("\n" + "=" * 65)
    print("  PSO COMPLETE")
    print("=" * 65)
    print(f"  Best fitness : {result['best_fitness']:.4f}\n")
    print(f"  {'Parameter':<30s}  {'Current':>10s}  →  {'Optimal':>10s}")
    print(f"  {'-'*30}  {'-'*10}     {'-'*10}")
    for name, val in result["best_params"].items():
        current = getattr(defaults, name)
        changed = "  ←" if current != val else ""
        print(f"  {name:<30s}  {str(current):>10s}  →  {str(val):>10s}{changed}")

    print(f"\n  Full log    → pso_results/pso_log.json")
    print(f"  Best config → pso_results/best_config.json")
    print(f"\n  To apply the best config, update config.py with the")
    print(f"  values marked ← above (or run apply_pso_config.py).")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="PSO parameter optimiser for the emotion evolution simulation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--particles",   type=int, default=min(16, mp.cpu_count()),
                    help="Swarm size (one worker process per particle)")
    ap.add_argument("--iters",       type=int, default=20,
                    help="Number of PSO generations")
    ap.add_argument("--eval-ticks",  type=int, default=15_000,
                    help="Simulation ticks per fitness evaluation")
    ap.add_argument("--out-dir",     type=str, default="pso_results",
                    help="Directory for logs and best_config.json")
    ap.add_argument("--seed",        type=int, default=42,
                    help="Master RNG seed")
    args = ap.parse_args()

    print("=" * 65)
    print("  PSO Parameter Optimiser — Human Emotion Evolution Sim")
    print("=" * 65)
    print(f"  Particles   : {args.particles}")
    print(f"  Generations : {args.iters}")
    print(f"  Eval ticks  : {args.eval_ticks:,} per trial")
    print(f"  Dimensions  : {N_DIM}  ({', '.join(NAMES)})")
    print(f"  CPUs        : {mp.cpu_count()}")
    print(f"  Output dir  : {args.out_dir}/")
    print(f"  Total evals : {args.particles * args.iters:,}")
    est_min = (args.eval_ticks / 60) * args.iters / 60  # rough: ~60 ticks/s
    print(f"  Est. time   : ~{est_min:.0f} min  (varies with hardware)")
    print("=" * 65)

    pso = PSO(
        n_particles  = args.particles,
        n_iterations = args.iters,
        eval_ticks   = args.eval_ticks,
        out_dir      = args.out_dir,
        seed         = args.seed,
    )

    with mp.Pool(processes=args.particles) as pool:
        result = pso.run(pool)

    _print_final(result)


if __name__ == "__main__":
    mp.set_start_method("fork", force=True)
    main()
