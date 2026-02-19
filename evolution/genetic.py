"""
Genetic algorithm engine.

Key design decisions
────────────────────
1. *Continuous reproduction* – agents reproduce sexually during the simulation
   whenever both are willing and energetic enough.  There is no artificial
   "generation boundary."  This is closer to how biology actually works and
   allows for overlapping generations (grandparents and grandchildren coexist).

2. *Tournament selection* – when two parents are needed (e.g. to replace a
   dead agent to keep population near target), we run independent tournaments
   among the LIVING agents.  The better a genome has been at keeping its
   current owner alive, the more likely it is chosen.

3. *Uniform crossover* – each gene is independently drawn from either parent
   with equal probability.  This lets good partial solutions from both parents
   recombine freely.

4. *Gaussian mutation* – selected genome entries receive additive Gaussian
   noise.  The mutation rate and std are hyperparameters that evolution
   itself could in principle optimise (not implemented here for simplicity).

5. *Elitism guard* – the single best genome seen so far is kept in memory.
   If the living population drops below 10, one fresh agent is seeded from
   this elite genome to prevent total extinction.
"""

from __future__ import annotations

import numpy as np
from typing import List, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from config import Config
    from agents.agent import Agent


class GeneticEngine:

    def __init__(self, cfg: "Config", rng: np.random.Generator | None = None):
        self.cfg = cfg
        self.rng = rng or np.random.default_rng()
        self.elite_genome: Optional[np.ndarray] = None
        self.elite_fitness: float = -1.0
        self.generation: int = 0   # incremented each time a new agent is born

    # ── Selection ─────────────────────────────────────────────────────────────

    def tournament_select(self, agents: List["Agent"]) -> "Agent":
        """Pick the best agent from a random subset of size k."""
        k = min(self.cfg.tournament_size, len(agents))
        contestants = self.rng.choice(agents, size=k, replace=False)
        return max(contestants, key=lambda a: a.fitness)

    def regional_tournament_select(self, agents: List["Agent"],
                                   near_x: float, near_y: float) -> "Agent":
        """Tournament selection biased toward agents near (near_x, near_y).

        Creates regional gene pools: successful behaviours spread locally,
        not globally.  If too few nearby candidates, falls back to global.
        """
        from environment.world import World as _W   # avoid circular import
        radius = self.cfg.spawn_radius
        nearby = [
            a for a in agents
            if abs(a.x - near_x) < radius and abs(a.y - near_y) < radius
        ]
        pool = nearby if len(nearby) >= 2 else agents
        k = min(self.cfg.tournament_size, len(pool))
        contestants = self.rng.choice(pool, size=k, replace=False)
        return max(contestants, key=lambda a: a.fitness)

    # ── Reproduction (called by runner when two agents mate) ──────────────────

    def crossover(self, genome_a: np.ndarray,
                  genome_b: np.ndarray) -> np.ndarray:
        """Uniform crossover: each gene independently from either parent."""
        mask = self.rng.random(genome_a.shape) < 0.5
        child = np.where(mask, genome_a, genome_b)
        return child

    def mutate(self, genome: np.ndarray) -> np.ndarray:
        """Gaussian mutation with per-gene probability."""
        genome = genome.copy()
        mask = self.rng.random(genome.shape) < self.cfg.mutation_rate
        genome[mask] += self.rng.normal(0, self.cfg.mutation_std, mask.sum())
        return genome

    def make_child_genome(self, parent_a: "Agent",
                          parent_b: "Agent") -> Tuple[np.ndarray, int]:
        """Full pipeline: crossover + mutate.  Returns (genome, child_hidden_size).

        NEAT-like architecture evolution
        ─────────────────────────────────
        Each agent carries its own hidden_size.  The child inherits one
        parent's size (50/50), with a small probability of growing or
        shrinking by 1 neuron.  Parents with the same hidden_size do normal
        crossover; mismatched parents contribute one parent's genome (zero-
        padded / truncated via Brain.resize_genome) before mutation.
        """
        hs_a = parent_a.brain.hs
        hs_b = parent_b.brain.hs

        # Inherit one parent's architecture
        child_hs = int(hs_a if self.rng.random() < 0.5 else hs_b)

        # Occasional architectural mutation (±1 neuron)
        if self.rng.random() < 0.02:
            child_hs = int(np.clip(
                child_hs + self.rng.choice([-1, 1]),
                self.cfg.hidden_size_min,
                self.cfg.hidden_size_max,
            ))

        if hs_a == hs_b == child_hs:
            # Same architecture: normal weight crossover
            ga = parent_a.brain.get_genome()
            gb = parent_b.brain.get_genome()
            if self.rng.random() < self.cfg.crossover_rate:
                child_genome = self.crossover(ga, gb)
            else:
                child_genome = ga.copy() if parent_a.fitness >= parent_b.fitness else gb.copy()
        else:
            # Architecture mismatch: clone the matching parent; resize if needed
            if hs_a == child_hs:
                child_genome = parent_a.brain.get_genome().copy()
            elif hs_b == child_hs:
                child_genome = parent_b.brain.get_genome().copy()
            else:
                # Neither matches — resize the fitter parent's genome
                src = parent_a if parent_a.fitness >= parent_b.fitness else parent_b
                child_genome = src.brain.resize_genome(child_hs)

        child_genome = self.mutate(child_genome)
        return child_genome, child_hs

    # ── Population maintenance ────────────────────────────────────────────────

    def update_elite(self, agents: List["Agent"]) -> None:
        """Update the best-ever genome from current living population."""
        for a in agents:
            if a.fitness > self.elite_fitness:
                self.elite_fitness = a.fitness
                self.elite_genome = a.brain.get_genome().copy()

    def replacement_genome(self, living_agents: List["Agent"],
                           near_x: float = None,
                           near_y: float = None) -> Tuple[np.ndarray, int]:
        """Produce a (genome, hidden_size) for a replacement agent.

        Uses regional tournament selection when a spawn location is given,
        so behaviours spread locally rather than globally.
        """
        default_hs = self.cfg.hidden_size
        if len(living_agents) >= 2:
            if near_x is not None:
                p1 = self.regional_tournament_select(living_agents, near_x, near_y)
                p2 = self.regional_tournament_select(living_agents, near_x, near_y)
            else:
                p1 = self.tournament_select(living_agents)
                p2 = self.tournament_select(living_agents)
            genome, hs = self.make_child_genome(p1, p2)
            return genome, hs
        elif self.elite_genome is not None:
            return self.mutate(self.elite_genome.copy()), default_hs
        else:
            return None, default_hs   # caller will use a random genome

    # ── Statistics ────────────────────────────────────────────────────────────

    def population_stats(self, agents: List["Agent"]) -> dict:
        if not agents:
            return {}
        fitnesses = [a.fitness for a in agents]
        # Genome diversity: group by hidden_size so we only stack same-shape arrays.
        # Average the per-group diversity weighted by group size.
        groups: dict = {}
        for a in agents:
            hs = a.brain.hs
            groups.setdefault(hs, []).append(a.brain.get_genome())
        diversity_vals = []
        for hs_group in groups.values():
            if len(hs_group) > 1:
                mat = np.stack(hs_group)
                diversity_vals.append(float(np.mean(np.std(mat, axis=0))))
        genome_diversity = float(np.mean(diversity_vals)) if diversity_vals else 0.0
        return {
            "mean_fitness":     float(np.mean(fitnesses)),
            "max_fitness":      float(np.max(fitnesses)),
            "genome_diversity": genome_diversity,
            "population_size":  len(agents),
            "generation":       self.generation,
        }
