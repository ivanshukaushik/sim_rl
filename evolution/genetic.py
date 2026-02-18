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
                          parent_b: "Agent") -> np.ndarray:
        """Full pipeline: select crossover strategy, crossover, mutate."""
        ga = parent_a.brain.get_genome()
        gb = parent_b.brain.get_genome()

        if self.rng.random() < self.cfg.crossover_rate:
            child_genome = self.crossover(ga, gb)
        else:
            # Clone the higher-fitness parent
            child_genome = ga.copy() if parent_a.fitness >= parent_b.fitness else gb.copy()

        child_genome = self.mutate(child_genome)
        return child_genome

    # ── Population maintenance ────────────────────────────────────────────────

    def update_elite(self, agents: List["Agent"]) -> None:
        """Update the best-ever genome from current living population."""
        for a in agents:
            if a.fitness > self.elite_fitness:
                self.elite_fitness = a.fitness
                self.elite_genome = a.brain.get_genome().copy()

    def replacement_genome(self, living_agents: List["Agent"]) -> np.ndarray:
        """
        Produce a genome to seed a replacement agent when population falls.
        Uses tournament selection from the living pool; falls back to elite
        if the pool is too small.
        """
        if len(living_agents) >= 2:
            p1 = self.tournament_select(living_agents)
            p2 = self.tournament_select(living_agents)
            return self.mutate(self.crossover(
                p1.brain.get_genome(), p2.brain.get_genome()
            ))
        elif self.elite_genome is not None:
            return self.mutate(self.elite_genome.copy())
        else:
            return None   # caller will use a random genome

    # ── Statistics ────────────────────────────────────────────────────────────

    def population_stats(self, agents: List["Agent"]) -> dict:
        if not agents:
            return {}
        genomes = np.stack([a.brain.get_genome() for a in agents])
        return {
            "mean_fitness": float(np.mean([a.fitness for a in agents])),
            "max_fitness": float(np.max([a.fitness for a in agents])),
            "genome_diversity": float(np.mean(np.std(genomes, axis=0))),
            "population_size": len(agents),
        }
