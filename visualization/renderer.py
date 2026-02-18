"""
Real-time visualisation using matplotlib.

Layout (2×2 grid)
─────────────────
  ┌─────────────────┬─────────────────┐
  │  World view     │  Emotion radar  │
  │  (agents +      │  (4 metrics     │
  │   resources +   │   over time)    │
  │   threats)      │                 │
  ├─────────────────┼─────────────────┤
  │  Signal dist.   │  Energy dist.   │
  │  (tribalism)    │  (dominance)    │
  └─────────────────┴─────────────────┘

Agent colours in the world view encode their dominant signal (tribe).
Agent size encodes energy.
Threat zones shown as red semi-transparent circles.
Bond lines drawn between bonded pairs (high familiarity + positive valence).
"""

from __future__ import annotations

import numpy as np
from typing import List, TYPE_CHECKING

try:
    import matplotlib
    matplotlib.use("Agg")   # non-interactive backend – works in headless envs
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from matplotlib.collections import LineCollection
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

if TYPE_CHECKING:
    from agents.agent import Agent
    from environment.world import World
    from simulation.metrics import MetricsTracker

# 8 distinct colours for the 8 tribe signals
TRIBE_COLORS = [
    "#e6194B",  # red
    "#3cb44b",  # green
    "#4363d8",  # blue
    "#f58231",  # orange
    "#911eb4",  # purple
    "#42d4f4",  # cyan
    "#f032e6",  # magenta
    "#bfef45",  # lime
]


class Renderer:

    def __init__(self, cfg, metrics: "MetricsTracker", save_dir: str = "frames"):
        self.cfg = cfg
        self.metrics = metrics
        self.save_dir = save_dir
        self.enabled = HAS_MPL

        if not self.enabled:
            print("[Renderer] matplotlib not available – skipping visualisation.")
            return

        import os
        os.makedirs(save_dir, exist_ok=True)

        self.fig, self.axes = plt.subplots(2, 2, figsize=(14, 11))
        self.fig.patch.set_facecolor("#1a1a2e")
        for ax in self.axes.flat:
            ax.set_facecolor("#16213e")
            for spine in ax.spines.values():
                spine.set_edgecolor("#e0e0e0")
            ax.tick_params(colors="#e0e0e0")
            ax.yaxis.label.set_color("#e0e0e0")
            ax.xaxis.label.set_color("#e0e0e0")
            ax.title.set_color("#e0e0e0")

        self._frame_idx = 0

    def draw(self, tick: int, agents: List["Agent"], world: "World") -> None:
        if not self.enabled or not agents:
            return

        self.fig.suptitle(
            f"Human Emotion Evolution  –  Tick {tick:,}  |  Pop {len(agents)}",
            color="white", fontsize=13, y=0.98,
        )

        self._draw_world(self.axes[0, 0], agents, world, tick)
        self._draw_emotion_timelines(self.axes[0, 1])
        self._draw_signal_distribution(self.axes[1, 0], agents)
        self._draw_energy_distribution(self.axes[1, 1], agents)

        self.fig.tight_layout(rect=[0, 0, 1, 0.96])

        fname = f"{self.save_dir}/frame_{self._frame_idx:06d}.png"
        self.fig.savefig(fname, dpi=90, facecolor=self.fig.get_facecolor())
        self._frame_idx += 1

        for ax in self.axes.flat:
            ax.cla()
            ax.set_facecolor("#16213e")
            for spine in ax.spines.values():
                spine.set_edgecolor("#e0e0e0")
            ax.tick_params(colors="#e0e0e0")

    # ── World view ────────────────────────────────────────────────────────────

    def _draw_world(self, ax, agents: List["Agent"], world: "World",
                    tick: int) -> None:
        ax.set_title("World", fontsize=10)
        ax.set_xlim(0, world.size)
        ax.set_ylim(0, world.size)
        ax.set_aspect("equal")

        # Resource heatmap (greyscale)
        ax.imshow(
            world.resources.T,
            origin="lower",
            extent=[0, world.size, 0, world.size],
            cmap="YlGn",
            alpha=0.55,
            vmin=0,
            vmax=self.cfg.patch_peak,
        )

        # Threat zones
        for threat in world.threats:
            circle = plt.Circle(
                (threat.x, threat.y), threat.radius,
                color="red", alpha=0.25, linewidth=0,
            )
            ax.add_patch(circle)

        # Bond lines between close bonded pairs
        id_to_agent = {a.id: a for a in agents}
        lines = []
        for agent in agents:
            for other_id, rec in agent.memory.all_records().items():
                if other_id <= agent.id:
                    continue
                other = id_to_agent.get(other_id)
                if other is None:
                    continue
                rec_o = other.memory.get(agent.id)
                if rec_o is None:
                    continue
                fam = (rec["familiarity"] + rec_o["familiarity"]) / 2
                val = (rec["valence"] + rec_o["valence"]) / 2
                if fam > 0.5 and val > 0.3:
                    lines.append([(agent.x, agent.y), (other.x, other.y)])

        if lines:
            lc = LineCollection(lines, colors="white", linewidths=0.5, alpha=0.3)
            ax.add_collection(lc)

        # Agents (colour = tribe signal, size = energy)
        xs = np.array([a.x for a in agents])
        ys = np.array([a.y for a in agents])
        sizes = np.array([max(5, a.energy * 0.3) for a in agents])
        colors = [TRIBE_COLORS[a.signal % len(TRIBE_COLORS)] for a in agents]

        ax.scatter(xs, ys, c=colors, s=sizes, alpha=0.8, linewidths=0)

    # ── Emotion timelines ─────────────────────────────────────────────────────

    def _draw_emotion_timelines(self, ax) -> None:
        ax.set_title("Emergent behaviour over time", fontsize=10)
        if len(self.metrics.history) < 2:
            ax.text(0.5, 0.5, "Collecting data…",
                    ha="center", va="center", color="grey", transform=ax.transAxes)
            return

        ticks = [r["tick"] for r in self.metrics.history]

        def ts(key, default=0):
            return [r.get(key, default) for r in self.metrics.history]

        ax.plot(ticks, ts("mean_bond_score"), label="Love (bond score)",
                color="#ff6b9d", linewidth=1.5)
        ax.plot(ticks, [max(0, v) for v in ts("tribal_bias")],
                label="Tribal bias", color="#ffd93d", linewidth=1.5)
        # Ritual convergence (0-1 range)
        ax.plot(ticks, ts("ritual_convergence"), label="Ritual convergence",
                color="#c77dff", linewidth=1.5)
        # Dominance (Gini, 0-1 range)
        ax.plot(ticks, ts("energy_gini"), label="Dominance (Gini)",
                color="#ff4d4d", linewidth=1.5)

        ax.set_xlabel("Tick")
        ax.set_ylabel("Normalised score")
        ax.legend(fontsize=7, facecolor="#16213e", labelcolor="white",
                  framealpha=0.7)
        ax.set_ylim(0, 1.05)
        for s in ax.spines.values():
            s.set_edgecolor("#e0e0e0")

    # ── Signal distribution ───────────────────────────────────────────────────

    def _draw_signal_distribution(self, ax, agents: List["Agent"]) -> None:
        ax.set_title("Tribe signal distribution  (colour = signal)", fontsize=10)
        if not agents:
            return
        from collections import Counter
        counts = Counter(a.signal for a in agents)
        signals = list(range(self.cfg.num_signals))
        vals = [counts.get(s, 0) for s in signals]
        colors = [TRIBE_COLORS[s % len(TRIBE_COLORS)] for s in signals]
        ax.bar(signals, vals, color=colors, edgecolor="none")
        ax.set_xlabel("Signal (tribal marker)")
        ax.set_ylabel("# agents")
        ax.set_xticks(signals)

    # ── Energy distribution ───────────────────────────────────────────────────

    def _draw_energy_distribution(self, ax, agents: List["Agent"]) -> None:
        ax.set_title("Energy distribution  (dominance / inequality)", fontsize=10)
        if not agents:
            return
        energies = [a.energy for a in agents]
        ax.hist(energies, bins=30, color="#4cc9f0", edgecolor="none", alpha=0.85)
        ax.axvline(np.mean(energies), color="yellow", linewidth=1,
                   linestyle="--", label=f"mean={np.mean(energies):.1f}")
        ax.set_xlabel("Energy")
        ax.set_ylabel("# agents")
        ax.legend(fontsize=8, facecolor="#16213e", labelcolor="white",
                  framealpha=0.7)
