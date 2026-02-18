"""
Entry point for the Human Emotion Evolution simulation.

Quick start
───────────
    python main.py                        # run with defaults
    python main.py --ticks 50000          # shorter run
    python main.py --no-render            # headless (faster)
    python main.py --resume checkpoints/tick_00005000.pkl
    python main.py --seed 1234            # reproducible run

What you're watching
────────────────────
Agents start with random neural networks.  Their only hard-coded drive is
surviving long enough to reproduce.  Over thousands of generations you should
observe four categories of emergent behaviour:

  LOVE        – stable bonded pairs / small cooperative clusters form
  TRIBALISM   – agents converge on sub-groups distinguished by the signal
                they broadcast; intra-group valence rises, inter-group falls
  RITUAL      – under stress, agents converge on a specific signal/action even
                without direct resource payoff  (proto-religion)
  DOMINANCE   – energy inequality rises; some agents consistently extract
                resources from others (social hierarchy)

None of this is programmed in.  The agents' neural networks discover these
strategies because they improve survival odds.
"""

import argparse
import sys
import numpy as np

from config import Config
from environment.world import World
from simulation.runner import Simulation
from simulation.metrics import MetricsTracker
from visualization.renderer import Renderer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Human emotion evolution via multi-agent RL + evolution",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--ticks", type=int, default=None,
                   help="Number of ticks to run (overrides config)")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed for reproducibility")
    p.add_argument("--population", type=int, default=None,
                   help="Initial population size (overrides config)")
    p.add_argument("--world-size", type=float, default=None,
                   help="World side length (overrides config)")
    p.add_argument("--no-render", action="store_true",
                   help="Disable visualisation (faster headless mode)")
    p.add_argument("--render-interval", type=int, default=None,
                   help="Ticks between rendered frames (overrides config)")
    p.add_argument("--metrics-interval", type=int, default=None,
                   help="Ticks between metric snapshots (overrides config)")
    p.add_argument("--resume", type=str, default=None,
                   help="Path to checkpoint .pkl file to resume from")
    p.add_argument("--out-dir", type=str, default=".",
                   help="Base output directory for frames/metrics/checkpoints")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # ── Config ────────────────────────────────────────────────────────────────
    cfg = Config()
    if args.population:
        cfg.initial_population = args.population
    if args.world_size:
        cfg.world_size = args.world_size
    if args.render_interval:
        cfg.render_interval = args.render_interval
    if args.metrics_interval:
        cfg.metrics_interval = args.metrics_interval

    print("=" * 65)
    print("  Human Emotion Evolution Simulation")
    print("=" * 65)
    print(f"  World size    : {cfg.world_size} × {cfg.world_size}")
    print(f"  Population    : {cfg.initial_population} (max {cfg.max_population})")
    print(f"  Brain dims    : {cfg.input_size()} → {cfg.hidden_size} → {cfg.output_size()}")
    print(f"  Signals       : {cfg.num_signals}  (tribal / communication channel)")
    print(f"  Mutation rate : {cfg.mutation_rate}")
    print(f"  Seed          : {args.seed}")
    print("=" * 65)

    # ── World ─────────────────────────────────────────────────────────────────
    rng = np.random.default_rng(args.seed)
    world = World(cfg, rng=rng)

    # ── Metrics ───────────────────────────────────────────────────────────────
    import os
    metrics_dir = os.path.join(args.out_dir, "metrics")
    metrics = MetricsTracker(cfg, out_dir=metrics_dir)

    # ── Renderer ──────────────────────────────────────────────────────────────
    renderer = None
    if not args.no_render:
        frames_dir = os.path.join(args.out_dir, "frames")
        renderer = Renderer(cfg, metrics, save_dir=frames_dir)

    # ── Simulation ────────────────────────────────────────────────────────────
    if args.resume:
        print(f"\nResuming from checkpoint: {args.resume}")
        sim = Simulation.load_checkpoint(args.resume, cfg, world)
        sim.metrics = metrics
        sim.renderer = renderer
    else:
        sim = Simulation(cfg, world, metrics=metrics, renderer=renderer,
                         seed=args.seed)

    ticks = args.ticks or cfg.max_ticks
    print(f"\nRunning for {ticks:,} ticks…  (Ctrl-C to stop early)\n")

    try:
        sim.run(ticks=ticks)
    except KeyboardInterrupt:
        print("\n[Interrupted by user]")

    # ── Save final state ──────────────────────────────────────────────────────
    metrics.save(os.path.join(metrics_dir, "history.json"))
    print("\nDone.")
    print(f"  Frames saved to  : {os.path.join(args.out_dir, 'frames')}")
    print(f"  Metrics saved to : {os.path.join(metrics_dir, 'history.json')}")
    print(f"  Checkpoints in   : checkpoints/")


if __name__ == "__main__":
    main()
