"""
2D continuous world with uneven resource patches and random threat events.

Design philosophy
─────────────────
* Resources are distributed as overlapping Gaussian blobs, creating feast /
  famine gradients.  Scarcity is the engine that drives all interesting
  social behaviour.

* Threats are short-lived events that hit a random location and damage nearby
  agents.  Agents that learn to predict or avoid threats survive longer.
  When a threat co-occurs with a particular agent behaviour (e.g. emitting
  signal 3), Hebbian learning can reinforce that behaviour – the seed of
  superstition / proto-religion.

* The world is *toroidal* (wraps at edges) so no agent ever gets "stuck in a
  corner" – every location is topologically equivalent.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from config import Config


@dataclass
class Threat:
    """A localised damaging event (predator, storm, earthquake…)."""
    x: float
    y: float
    radius: float
    damage: float
    ticks_remaining: int


class World:
    def __init__(self, cfg: "Config", rng: np.random.Generator | None = None):
        self.cfg = cfg
        self.rng = rng or np.random.default_rng()
        self.size = cfg.world_size

        # ── Resource grid ────────────────────────────────────────────────────
        # resource_capacity[i,j] = max energy at cell (i,j)
        # resources[i,j]         = current energy at cell (i,j)
        grid = int(self.size)
        self.resource_capacity = np.zeros((grid, grid), dtype=np.float32)
        self._init_resources(grid)
        self.resources = self.resource_capacity.copy() * 0.6   # start at 60 %

        # ── Active threats (routine) ─────────────────────────────────────────
        self.threats: List[Threat] = []

        # ── Catastrophic events (rare, large, episodic) ───────────────────────
        # Separate from routine threats so agents can build distinct episodic
        # memories of rare disasters — the substrate for Whitehouse-style ritual.
        self.catastrophes: List[Threat] = []

        # ── Tick counter ─────────────────────────────────────────────────────
        self.tick = 0

    # ── Initialisation ───────────────────────────────────────────────────────

    def _init_resources(self, grid: int) -> None:
        """Place Gaussian resource patches at random positions."""
        xs = np.arange(grid)
        ys = np.arange(grid)
        XX, YY = np.meshgrid(xs, ys, indexing="ij")

        for _ in range(self.cfg.num_resource_patches):
            cx = self.rng.uniform(0, self.size)
            cy = self.rng.uniform(0, self.size)
            peak = self.cfg.patch_peak
            r = self.cfg.patch_radius

            # Toroidal distance to handle wrap-around
            dx = np.minimum(np.abs(XX - cx), self.size - np.abs(XX - cx))
            dy = np.minimum(np.abs(YY - cy), self.size - np.abs(YY - cy))
            dist_sq = dx ** 2 + dy ** 2
            self.resource_capacity += peak * np.exp(-dist_sq / (2 * r ** 2))

        # Normalise so peak is exactly patch_peak
        if self.resource_capacity.max() > 0:
            self.resource_capacity *= (self.cfg.patch_peak / self.resource_capacity.max())

    # ── Per-tick update ──────────────────────────────────────────────────────

    def step(self) -> List[Threat]:
        """Advance the world by one tick.  Returns list of newly spawned threats."""
        self.tick += 1

        # Resource regeneration toward capacity
        delta = self.cfg.resource_regen_rate * (self.resource_capacity - self.resources)
        self.resources = np.clip(self.resources + delta, 0.0, self.resource_capacity)

        # Age out existing threats
        still_active = []
        for t in self.threats:
            t.ticks_remaining -= 1
            if t.ticks_remaining > 0:
                still_active.append(t)
        self.threats = still_active

        # Possibly spawn a new routine threat
        new_threats: List[Threat] = []
        if self.rng.random() < self.cfg.threat_prob_per_tick:
            t = Threat(
                x=self.rng.uniform(0, self.size),
                y=self.rng.uniform(0, self.size),
                radius=self.cfg.threat_radius,
                damage=self.cfg.threat_damage,
                ticks_remaining=self.cfg.threat_duration,
            )
            self.threats.append(t)
            new_threats.append(t)

        # Possibly spawn a catastrophic event (rare, large, near-lethal)
        # These create distinct episodic memories — the seed of ritual/religion
        for cat in list(self.catastrophes):
            cat.ticks_remaining -= 1
        self.catastrophes = [c for c in self.catastrophes if c.ticks_remaining > 0]
        if self.rng.random() < self.cfg.catastrophe_prob_per_tick:
            self.catastrophes.append(Threat(
                x=self.rng.uniform(0, self.size),
                y=self.rng.uniform(0, self.size),
                radius=self.cfg.catastrophe_radius,
                damage=self.cfg.catastrophe_damage,
                ticks_remaining=self.cfg.catastrophe_duration,
            ))

        return new_threats

    # ── Spatial queries ──────────────────────────────────────────────────────

    def get_resource(self, x: float, y: float) -> float:
        """Return current resource level at a continuous position."""
        xi = int(x) % int(self.size)
        yi = int(y) % int(self.size)
        return float(self.resources[xi, yi])

    def get_capacity(self, x: float, y: float) -> float:
        xi = int(x) % int(self.size)
        yi = int(y) % int(self.size)
        return float(self.resource_capacity[xi, yi])

    def consume(self, x: float, y: float, amount: float) -> float:
        """Eat up to *amount* from position (x,y). Returns actual amount eaten."""
        xi = int(x) % int(self.size)
        yi = int(y) % int(self.size)
        available = self.resources[xi, yi]
        eaten = min(available, amount)
        self.resources[xi, yi] -= eaten
        return float(eaten)

    def toroidal_dist(self, x1: float, y1: float, x2: float, y2: float) -> float:
        """Euclidean distance accounting for world wrap-around."""
        dx = abs(x1 - x2)
        dy = abs(y1 - y2)
        dx = min(dx, self.size - dx)
        dy = min(dy, self.size - dy)
        return float(np.sqrt(dx * dx + dy * dy))

    def toroidal_delta(self, x1: float, y1: float,
                       x2: float, y2: float) -> Tuple[float, float]:
        """Signed (dx, dy) from (x1,y1) toward (x2,y2) on a toroidal world."""
        dx = x2 - x1
        dy = y2 - y1
        if abs(dx) > self.size / 2:
            dx -= np.sign(dx) * self.size
        if abs(dy) > self.size / 2:
            dy -= np.sign(dy) * self.size
        return float(dx), float(dy)

    def wrap(self, x: float, y: float) -> Tuple[float, float]:
        """Wrap coordinates to [0, size)."""
        return float(x % self.size), float(y % self.size)

    def threats_at(self, x: float, y: float) -> float:
        """Total routine threat damage at position."""
        total = 0.0
        for t in self.threats:
            if self.toroidal_dist(x, y, t.x, t.y) <= t.radius:
                total += t.damage
        return total

    def catastrophe_exposure_at(self, x: float, y: float) -> float:
        """Total catastrophic event damage at position (0 if none active)."""
        total = 0.0
        for c in self.catastrophes:
            if self.toroidal_dist(x, y, c.x, c.y) <= c.radius:
                total += c.damage
        return total

    def local_resource_density(self, x: float, y: float,
                                radius: float = 5.0) -> float:
        """Mean resource density within *radius* of (x,y)."""
        xi = int(x) % int(self.size)
        yi = int(y) % int(self.size)
        r = int(radius)
        total = 0.0
        count = 0
        for di in range(-r, r + 1):
            for dj in range(-r, r + 1):
                if di * di + dj * dj <= r * r:
                    ii = (xi + di) % int(self.size)
                    jj = (yi + dj) % int(self.size)
                    total += self.resources[ii, jj]
                    count += 1
        return total / max(count, 1)
