"""
Emergent behaviour metrics.

None of the four "emotions" we track are programmed into the agents.
They are *detected* here from behavioural patterns.

┌────────────────────┬─────────────────────────────────────────────────────┐
│ Emergent behaviour │ How we detect it                                    │
├────────────────────┼─────────────────────────────────────────────────────┤
│ Love / Bonding     │ Pairs with high mutual familiarity + positive        │
│                    │ valence that consistently remain within              │
│                    │ close proximity (co-presence score).                 │
├────────────────────┼─────────────────────────────────────────────────────┤
│ Tribalism          │ Cluster agents by dominant signal.  Measure          │
│                    │ intra-cluster vs inter-cluster sharing and attack    │
│                    │ rates.  High ratio → tribal in-group bias.          │
├────────────────────┼─────────────────────────────────────────────────────┤
│ Superstition/      │ Under stress (stress > 0.5), which signal do agents  │
│ Religion           │ tend to emit?  Low entropy across agents →           │
│                    │ convergence on a "ritual" signal.                   │
│                    │ Also: do certain locations become avoided even        │
│                    │ though resources are still there?                   │
├────────────────────┼─────────────────────────────────────────────────────┤
│ Dominance          │ Gini coefficient of energy distribution.            │
│ Hierarchy          │ Social network of attack outcomes (who beats whom). │
│                    │ High Gini + persistent hierarchy → dominance.       │
└────────────────────┴─────────────────────────────────────────────────────┘
"""

from __future__ import annotations

import json
import os
from collections import defaultdict, Counter
from typing import List, Dict, Any, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from agents.agent import Agent
    from environment.world import World
    from evolution.genetic import GeneticEngine


def _gini(values: np.ndarray) -> float:
    """Gini coefficient of an array of non-negative values."""
    if len(values) == 0:
        return 0.0
    v = np.sort(values.astype(float))
    n = len(v)
    cumv = np.cumsum(v)
    return float((2 * np.sum((np.arange(1, n + 1)) * v) - (n + 1) * cumv[-1])
                 / (n * cumv[-1] + 1e-9))


def _entropy(counts: np.ndarray) -> float:
    """Shannon entropy of a count vector."""
    p = counts / (counts.sum() + 1e-9)
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p)))


class MetricsTracker:

    def __init__(self, cfg, out_dir: str = "metrics"):
        self.cfg = cfg
        self.out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)
        self.history: List[Dict[str, Any]] = []

    # ── Top-level record ──────────────────────────────────────────────────────

    def record(self, tick: int, agents: List["Agent"],
               world: "World", ga: "GeneticEngine") -> Dict[str, Any]:
        row: Dict[str, Any] = {"tick": tick}

        row.update(self._population_stats(agents))
        row.update(self._bonding(agents))
        row.update(self._tribalism(agents))
        row.update(self._superstition(agents))
        row.update(self._dominance(agents))
        row.update(ga.population_stats(agents))

        self.history.append(row)
        self._print_summary(row)
        return row

    # ── Population basics ─────────────────────────────────────────────────────

    def _population_stats(self, agents: List["Agent"]) -> dict:
        if not agents:
            return {"pop": 0, "mean_age": 0, "mean_energy": 0}
        ages = [a.age for a in agents]
        energies = [a.energy for a in agents]
        return {
            "pop": len(agents),
            "mean_age": float(np.mean(ages)),
            "mean_energy": float(np.mean(energies)),
            "mean_stress": float(np.mean([a.stress for a in agents])),
        }

    # ── Love / Bonding ────────────────────────────────────────────────────────

    def _bonding(self, agents: List["Agent"]) -> dict:
        """
        A bond is detected when BOTH agents have a high-familiarity,
        positive-valence record of each other.

        bond_score = mean(familiarity_A_of_B + familiarity_B_of_A) / 2
                       × mean(valence_A_of_B + valence_B_of_A) / 2
        only counted when both valences are positive.
        """
        id_to_agent = {a.id: a for a in agents}
        bond_scores = []
        bonded_pairs = 0

        for agent in agents:
            for other_id, rec in agent.memory.all_records().items():
                if other_id <= agent.id:  # avoid double-counting
                    continue
                other = id_to_agent.get(other_id)
                if other is None:
                    continue
                rec_other = other.memory.get(agent.id)
                if rec_other is None:
                    continue

                # Mutual familiarity and valence
                fam = (rec["familiarity"] + rec_other["familiarity"]) / 2
                val = (rec["valence"] + rec_other["valence"]) / 2
                if val > 0.1 and fam > 0.2:
                    score = fam * val
                    bond_scores.append(score)
                    if fam > 0.5 and val > 0.3:
                        bonded_pairs += 1

        return {
            "bonded_pairs": bonded_pairs,
            "mean_bond_score": float(np.mean(bond_scores)) if bond_scores else 0.0,
            "max_bond_score": float(np.max(bond_scores)) if bond_scores else 0.0,
        }

    # ── Tribalism ─────────────────────────────────────────────────────────────

    def _tribalism(self, agents: List["Agent"]) -> dict:
        """
        Group agents by their current signal (0-7).
        Count sharing events and attacks within vs across groups.

        We approximate this from the social memory:
        - Positive valence memories likely stem from sharing → intra/inter sharing
        - Negative valence memories likely stem from attacks → intra/inter aggression

        True cross-group tracking would require logging each interaction;
        here we use a proxy: are agents with the same signal more likely to
        have positive valence records of each other?
        """
        if not agents:
            return {"tribal_bias": 0.0, "num_tribes": 0, "dominant_tribe_frac": 0.0}

        id_to_signal = {a.id: a.signal for a in agents}

        # Count signal distribution
        signal_counts = Counter(a.signal for a in agents)
        num_tribes = len(signal_counts)
        dominant_frac = max(signal_counts.values()) / len(agents)

        # Tribal bias: average valence when same signal vs different signal
        same_val, diff_val = [], []
        for agent in agents:
            my_sig = agent.signal
            for other_id, rec in agent.memory.all_records().items():
                other_sig = id_to_signal.get(other_id)
                if other_sig is None:
                    continue
                if other_sig == my_sig:
                    same_val.append(rec["valence"])
                else:
                    diff_val.append(rec["valence"])

        same_mean = float(np.mean(same_val)) if same_val else 0.0
        diff_mean = float(np.mean(diff_val)) if diff_val else 0.0
        tribal_bias = same_mean - diff_mean   # positive = favour same group

        return {
            "tribal_bias": tribal_bias,
            "num_tribes": num_tribes,
            "dominant_tribe_frac": dominant_frac,
            "same_signal_mean_valence": same_mean,
            "diff_signal_mean_valence": diff_mean,
        }

    # ── Superstition / Religion ───────────────────────────────────────────────

    def _superstition(self, agents: List["Agent"]) -> dict:
        """
        Under high stress, which signals do agents emit?
        If all agents converge on the same signal under stress, a ritual has emerged.

        Also track: do agents share a common stress-action pattern (the ritual)?
        ritual_signal = mode of signals emitted under stress
        ritual_convergence = fraction of stressed agents using the ritual signal
        """
        stress_signals = []
        for agent in agents:
            for entry in agent.stress_action_log[-50:]:  # recent 50 stress events
                stress_signals.append(entry["signal"])
            # Clear to avoid unbounded growth
            agent.stress_action_log = agent.stress_action_log[-50:]

        if not stress_signals:
            return {"ritual_signal": -1, "ritual_convergence": 0.0,
                    "stress_signal_entropy": float(np.log2(self.cfg.num_signals))}

        counts = np.zeros(self.cfg.num_signals)
        for s in stress_signals:
            counts[s] += 1

        entropy = _entropy(counts)
        max_entropy = float(np.log2(self.cfg.num_signals))
        ritual_signal = int(np.argmax(counts))
        convergence = float(counts[ritual_signal] / len(stress_signals))

        return {
            "ritual_signal": ritual_signal,
            "ritual_convergence": convergence,
            "stress_signal_entropy": entropy,
            "max_possible_entropy": max_entropy,
        }

    # ── Dominance hierarchy ───────────────────────────────────────────────────

    def _dominance(self, agents: List["Agent"]) -> dict:
        """
        Gini coefficient of energy → wealth inequality → dominance.
        Also measure: fraction of agents with > 2× mean energy.
        """
        if not agents:
            return {"energy_gini": 0.0, "dominant_fraction": 0.0}

        energies = np.array([a.energy for a in agents])
        gini = _gini(energies)
        mean_e = energies.mean()
        dominant_frac = float(np.mean(energies > 2 * mean_e))

        # Attack index: total attacks / total agents
        total_attacks = sum(a.total_attacked for a in agents)
        attack_index = total_attacks / (len(agents) + 1e-9)

        return {
            "energy_gini": gini,
            "dominant_fraction": dominant_frac,
            "mean_attack_stolen": attack_index,
        }

    # ── Output ────────────────────────────────────────────────────────────────

    def _print_summary(self, row: dict) -> None:
        t = row.get("tick", 0)
        print(
            f"\n── Tick {t:>7d} ──────────────────────────────────────────────\n"
            f"  Population  : {row.get('pop',0):>4d}  |  "
            f"Mean energy: {row.get('mean_energy', 0):>5.1f}  |  "
            f"Mean stress: {row.get('mean_stress', 0):.2f}\n"
            f"  [LOVE]      bonded pairs={row.get('bonded_pairs',0):>3d}  "
            f"mean bond score={row.get('mean_bond_score',0):.3f}\n"
            f"  [TRIBE]     tribal bias={row.get('tribal_bias',0):>+.3f}  "
            f"num_tribes={row.get('num_tribes',0):>2d}  "
            f"dominant={row.get('dominant_tribe_frac',0):.0%}\n"
            f"  [RITUAL]    ritual_signal={row.get('ritual_signal',-1):>2d}  "
            f"convergence={row.get('ritual_convergence',0):.0%}  "
            f"entropy={row.get('stress_signal_entropy',0):.2f}"
            f"/{row.get('max_possible_entropy',3):.2f}\n"
            f"  [DOMINANCE] energy_gini={row.get('energy_gini',0):.3f}  "
            f"dominant_frac={row.get('dominant_fraction',0):.0%}\n"
            f"  [GENETICS]  generation={row.get('generation',0):>6d}  "
            f"diversity={row.get('genome_diversity',0):.4f}"
        )

    def save(self, filename: str = "metrics/history.json") -> None:
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, "w") as f:
            json.dump(self.history, f, indent=2)
        print(f"  [metrics saved → {filename}]")
