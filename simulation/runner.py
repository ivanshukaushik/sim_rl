"""
Main simulation loop.

Each tick:
  1. World step (resource regen, threats)
  2. Spatial indexing – find each agent's visible neighbours
  3. Agent step (obs → brain → actions + move + eat + signal update)
  4. Resolve interactions (sharing, attacking, mating)
  5. Apply threat damage
  6. Cull dead agents
  7. Maintain population (spawn replacements / let sexual reproduction happen)
  8. Log metrics periodically
  9. Checkpoint periodically
"""

from __future__ import annotations

import os
import pickle
import time
import numpy as np
from typing import List, Optional, TYPE_CHECKING

from agents.agent import Agent
from evolution.genetic import GeneticEngine

if TYPE_CHECKING:
    from config import Config
    from environment.world import World
    from simulation.metrics import MetricsTracker
    from visualization.renderer import Renderer


class Simulation:

    def __init__(self, cfg: "Config",
                 world: "World",
                 metrics: Optional["MetricsTracker"] = None,
                 renderer: Optional["Renderer"] = None,
                 seed: int = 42):

        self.cfg = cfg
        self.world = world
        self.metrics = metrics
        self.renderer = renderer
        self.rng = np.random.default_rng(seed)
        self.ga = GeneticEngine(cfg, rng=self.rng)

        # ── Spawn initial population ────────────────────────────────────
        self.agents: List[Agent] = []
        for _ in range(cfg.initial_population):
            a = Agent(cfg, rng=np.random.default_rng(self.rng.integers(1 << 31)))
            self.agents.append(a)

        self.tick = 0
        self._wall_start = time.time()

    # ── Spatial index ─────────────────────────────────────────────────────────

    def _visible_neighbours(self, agent: Agent) -> List[Agent]:
        """Return agents within vision_radius, sorted by distance, excluding self."""
        result = []
        for other in self.agents:
            if other.id == agent.id or not other.alive:
                continue
            d = self.world.toroidal_dist(agent.x, agent.y, other.x, other.y)
            if d <= self.cfg.vision_radius:
                result.append((d, other))
        result.sort(key=lambda x: x[0])
        return [a for _, a in result]

    # ── Interaction resolution ────────────────────────────────────────────────

    def _resolve_interactions(self, agent: Agent, actions: dict,
                               visible: List[Agent]) -> None:
        """
        Handle sharing, attacking, and mating.

        We operate on the first eligible neighbour in each category to keep
        things O(k) per agent rather than O(k²).
        """
        if not visible:
            return

        nearest = visible[0]

        # ── Share ─────────────────────────────────────────────────────────
        if (actions.get("share", 0) > 0.5
                and agent.energy > self.cfg.share_amount + 10):
            agent.energy -= self.cfg.share_amount
            agent.total_shared += self.cfg.share_amount
            nearest.receive_share(self.cfg.share_amount, agent.id, self.tick)

        # ── Attack ────────────────────────────────────────────────────────
        elif actions.get("attack", 0) > 0.5:
            agent.energy -= self.cfg.attack_cost
            stolen = nearest.take_attack_damage(self.cfg.attack_steal,
                                                agent.id, self.tick)
            agent.receive_attack_reward(stolen, nearest.id, self.tick)

        # ── Mate ──────────────────────────────────────────────────────────
        if (actions.get("mate", 0) > 0.5
                and agent.can_mate()
                and nearest.can_mate()):
            # Both agents must be willing
            nearest_obs = nearest.build_observation(self.world, self._visible_neighbours(nearest))
            nearest_raw = nearest.brain.forward(nearest_obs)
            nearest_actions = nearest.brain.decode_actions(nearest_raw, self.cfg.max_speed)
            if nearest_actions.get("mate", 0) > 0.5:
                self._reproduce(agent, nearest)

    def _reproduce(self, parent_a: Agent, parent_b: Agent) -> None:
        """Create an offspring near the parents."""
        if len(self.agents) >= self.cfg.max_population:
            return

        child_genome = self.ga.make_child_genome(parent_a, parent_b)
        cx = (parent_a.x + parent_b.x) / 2 + self.rng.uniform(-2, 2)
        cy = (parent_a.y + parent_b.y) / 2 + self.rng.uniform(-2, 2)
        cx, cy = self.world.wrap(cx, cy)

        child = Agent(
            self.cfg,
            genome=child_genome,
            x=cx, y=cy,
            rng=np.random.default_rng(self.rng.integers(1 << 31)),
            parent_ids=(parent_a.id, parent_b.id),
        )

        parent_a.spend_mate_energy()
        parent_b.spend_mate_energy()
        parent_a.offspring_count += 1
        parent_b.offspring_count += 1

        self.agents.append(child)
        self.ga.generation += 1

    def _maintain_population(self) -> None:
        """If population falls below threshold, spawn replacement agents."""
        target = self.cfg.initial_population
        while len(self.agents) < target:
            genome = self.ga.replacement_genome(self.agents)
            x = self.rng.uniform(0, self.cfg.world_size)
            y = self.rng.uniform(0, self.cfg.world_size)
            a = Agent(self.cfg, genome=genome, x=x, y=y,
                      rng=np.random.default_rng(self.rng.integers(1 << 31)))
            self.agents.append(a)

    # ── Main step ─────────────────────────────────────────────────────────────

    def step(self) -> None:
        self.tick += 1

        # 1. World update
        self.world.step()

        # 2. Collect per-agent actions
        actions_map = {}
        visible_map = {}
        for agent in self.agents:
            if not agent.alive:
                continue
            vis = self._visible_neighbours(agent)
            visible_map[agent.id] = vis
            # Update co-presence memory for all visible agents
            for other in vis:
                agent.update_coexistence(other.id, self.tick)
            actions_map[agent.id] = agent.step(self.world, vis)

        # 3. Resolve interactions
        for agent in self.agents:
            if not agent.alive:
                continue
            actions = actions_map.get(agent.id, {})
            vis = visible_map.get(agent.id, [])
            self._resolve_interactions(agent, actions, vis)

        # 4. Apply threat damage to agents inside threat zones
        for threat in self.world.threats:
            for agent in self.agents:
                if not agent.alive:
                    continue
                d = self.world.toroidal_dist(agent.x, agent.y, threat.x, threat.y)
                if d <= threat.radius:
                    # Damage already partially applied in agent.step() via threats_at()
                    # Full damage on the tick the threat spawns
                    if threat.ticks_remaining == self.cfg.threat_duration:
                        agent.energy -= threat.damage * 0.5
                        if agent.energy <= 0:
                            agent.alive = False

        # 5. Cull dead agents, update elite
        self.ga.update_elite(self.agents)
        self.agents = [a for a in self.agents if a.alive]

        # 6. Maintain population
        self._maintain_population()

        # 7. Metrics
        if self.metrics and self.tick % self.cfg.metrics_interval == 0:
            self.metrics.record(self.tick, self.agents, self.world, self.ga)

        # 8. Render
        if self.renderer and self.tick % self.cfg.render_interval == 0:
            self.renderer.draw(self.tick, self.agents, self.world)

        # 9. Checkpoint
        if self.tick % self.cfg.checkpoint_interval == 0:
            self._save_checkpoint()

    # ── Run loop ──────────────────────────────────────────────────────────────

    def run(self, ticks: Optional[int] = None) -> None:
        total = ticks or self.cfg.max_ticks
        for _ in range(total):
            self.step()
            if self.tick % 2000 == 0:
                elapsed = time.time() - self._wall_start
                print(f"  tick {self.tick:>7d} | pop {len(self.agents):>4d} | "
                      f"gen {self.ga.generation:>6d} | "
                      f"wall {elapsed:.0f}s")

    # ── Checkpointing ─────────────────────────────────────────────────────────

    def _save_checkpoint(self, path: str = "checkpoints") -> None:
        os.makedirs(path, exist_ok=True)
        fname = os.path.join(path, f"tick_{self.tick:08d}.pkl")
        state = {
            "tick": self.tick,
            "genomes": [a.brain.get_genome() for a in self.agents],
            "positions": [(a.x, a.y) for a in self.agents],
            "energies": [a.energy for a in self.agents],
            "ages": [a.age for a in self.agents],
            "signals": [a.signal for a in self.agents],
            "fitnesses": [a.fitness for a in self.agents],
            "elite_genome": self.ga.elite_genome,
            "generation": self.ga.generation,
        }
        with open(fname, "wb") as f:
            pickle.dump(state, f)
        print(f"  [checkpoint saved → {fname}]")

    @classmethod
    def load_checkpoint(cls, path: str, cfg: "Config", world: "World") -> "Simulation":
        with open(path, "rb") as f:
            state = pickle.load(f)
        sim = cls(cfg, world)
        # Re-create agents from saved genomes
        sim.agents = []
        for g, (x, y), e, age, sig in zip(
                state["genomes"], state["positions"],
                state["energies"], state["ages"], state["signals"]):
            a = Agent(cfg, genome=g, x=x, y=y,
                      rng=np.random.default_rng())
            a.energy = e
            a.age = age
            a.signal = sig
            sim.agents.append(a)
        sim.tick = state["tick"]
        sim.ga.elite_genome = state.get("elite_genome")
        sim.ga.generation = state.get("generation", 0)
        return sim
