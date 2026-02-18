"""
Agent – the living unit of the simulation.

State
─────
  position      continuous (x, y) on the toroidal world
  energy        0-100 – reaches 0 → dies
  age           ticks lived – reaches max_age → dies
  stress        0-1 – rises near threats / low energy, decays otherwise
  signal        current discrete signal being broadcast (0-7)

Social memory
─────────────
Each agent remembers up to `memory_capacity` other agents it has encountered.
For each remembered agent it tracks:
  - familiarity   : how much total time spent near them (normalised 0-1)
  - valence       : net sentiment, −1 (hate) to +1 (love)
  - co_presence   : raw tick counter of time spent within vision together
  - last_seen     : world tick of most recent encounter

This memory is the substrate for love/hate to emerge:
  * Positive valence = the other agent has helped / not harmed me.
  * Familiarity rise above a threshold + positive valence → an agent that
    preferentially seeks this individual → "bonded pair" / love.
  * Familiarity rise + negative valence → active avoidance / hate.
  * Agents never start knowing anyone – all social knowledge is earned
    within a lifetime through direct experience.

Stress and superstition
───────────────────────
Stress spikes when a threat is nearby or when energy is low.
Under stress, the Hebbian learning rate is multiplied by a factor,
making associations formed during stressful moments especially sticky.
If an agent happens to be performing action X (e.g. emitting signal 5)
when a threat hits and it SURVIVES, that association is reinforced.
Over many generations this can produce ritual behaviour: agents that
systematically perform X when stressed, even without direct resource benefit.
"""

from __future__ import annotations

import numpy as np
from collections import OrderedDict
from typing import Dict, List, Optional, TYPE_CHECKING

from agents.brain import Brain, INPUT_SIZE, OUTPUT_SIZE, N_OWN_STATE, N_PER_AGENT

if TYPE_CHECKING:
    from config import Config
    from environment.world import World

# Global agent ID counter
_NEXT_ID = 0


def _new_id() -> int:
    global _NEXT_ID
    _NEXT_ID += 1
    return _NEXT_ID


class SocialMemory:
    """Fixed-capacity LRU cache of social records."""

    def __init__(self, capacity: int):
        self.capacity = capacity
        # agent_id → {"familiarity": float, "valence": float, "co_presence": int, "last_seen": int}
        self._records: OrderedDict[int, dict] = OrderedDict()

    def update(self, other_id: int, valence_delta: float,
               current_tick: int) -> None:
        """Call each tick an agent is visible; valence_delta is the sentiment increment."""
        if other_id not in self._records:
            if len(self._records) >= self.capacity:
                # Evict least-recently seen
                self._records.popitem(last=False)
            self._records[other_id] = {
                "familiarity": 0.0,
                "valence": 0.0,
                "co_presence": 0,
                "last_seen": current_tick,
            }
        rec = self._records[other_id]
        rec["co_presence"] += 1
        rec["familiarity"] = min(1.0, rec["co_presence"] / 500.0)  # saturates at 500 ticks
        rec["valence"] = float(np.clip(rec["valence"] + valence_delta * 0.05, -1.0, 1.0))
        rec["last_seen"] = current_tick
        # Move to end (most recent)
        self._records.move_to_end(other_id)

    def get(self, other_id: int) -> Optional[dict]:
        return self._records.get(other_id)

    def top_k(self, k: int) -> List[dict]:
        """Return up to k records sorted by familiarity descending."""
        recs = sorted(self._records.values(),
                      key=lambda r: r["familiarity"], reverse=True)
        return recs[:k]

    def all_records(self) -> dict:
        return dict(self._records)


class Agent:
    def __init__(self, cfg: "Config", genome: Optional[np.ndarray] = None,
                 x: Optional[float] = None, y: Optional[float] = None,
                 rng: Optional[np.random.Generator] = None,
                 parent_ids: Optional[tuple] = None):

        self.cfg = cfg
        self.rng = rng or np.random.default_rng()
        self.id = _new_id()
        self.parent_ids = parent_ids

        # Position
        self.x = float(x if x is not None else self.rng.uniform(0, cfg.world_size))
        self.y = float(y if y is not None else self.rng.uniform(0, cfg.world_size))

        # Vitals
        self.energy = cfg.initial_energy
        self.age = 0
        self.alive = True
        self.stress = 0.0
        self.signal = 0                # currently emitted signal (0 – num_signals-1)
        self.mate_cooldown = 0         # ticks before can mate again

        # Social
        self.memory = SocialMemory(cfg.memory_capacity)

        # Neural network
        self.brain = Brain(
            genome=genome,
            hidden_size=cfg.hidden_size,
            input_size=cfg.input_size(),
            output_size=cfg.output_size(),
            rng=self.rng,
        )

        # Logging fields (used by metrics)
        self.energy_history: List[float] = []
        self.position_history: List[tuple] = []  # sparse, sampled
        self.stress_action_log: List[dict] = []   # for superstition detection
        self.ticks_survived = 0
        self.offspring_count = 0
        self.total_shared = 0.0
        self.total_attacked = 0.0

    # ── Observation construction ──────────────────────────────────────────────

    def build_observation(self, world: "World",
                          visible_agents: List["Agent"]) -> np.ndarray:
        """
        Build the 77-dimensional input vector for the brain.

        visible_agents should be pre-sorted by distance (closest first),
        already filtered to only include agents within vision_radius.
        """
        obs = np.zeros(self.cfg.input_size(), dtype=np.float64)

        # ── Own state ─────────────────────────────────────────────────────
        obs[0] = self.energy / self.cfg.max_energy
        obs[1] = self.age / self.cfg.max_age
        # Signal one-hot
        obs[2 + self.signal] = 1.0   # indices 2-9
        obs[10] = self.stress
        obs[11] = min(1.0, world.get_resource(self.x, self.y) / self.cfg.patch_peak)

        # ── Visible neighbours ─────────────────────────────────────────────
        n_show = min(len(visible_agents), self.cfg.max_visible_agents)
        for i in range(n_show):
            other = visible_agents[i]
            base = N_OWN_STATE + i * N_PER_AGENT

            dx, dy = world.toroidal_delta(self.x, self.y, other.x, other.y)
            obs[base + 0] = dx / self.cfg.vision_radius
            obs[base + 1] = dy / self.cfg.vision_radius
            # Their signal one-hot
            obs[base + 2 + other.signal] = 1.0   # base+2 to base+9
            obs[base + 10] = other.energy / self.cfg.max_energy

            rec = self.memory.get(other.id)
            if rec is not None:
                obs[base + 11] = rec["familiarity"]
                obs[base + 12] = rec["valence"]
            # else stays 0,0 = unknown stranger

        return obs

    # ── Per-tick action step ──────────────────────────────────────────────────

    def step(self, world: "World", visible_agents: List["Agent"]) -> dict:
        """
        1. Build observation
        2. Forward pass through brain (updates Hebbian weights as side-effect)
        3. Decode actions
        4. Apply movement and energy costs
        5. Update stress
        6. Return action dict for runner to resolve interactions
        """
        if not self.alive:
            return {}

        self.age += 1
        self.ticks_survived += 1

        # Build obs
        obs = self.build_observation(world, visible_agents)

        # Stress modulation of Hebbian learning
        # Under stress, the Hebbian learning rate is amplified
        original_hebb = self.brain.hebb_rates.copy()
        if self.stress > 0.4:
            self.brain.hebb_rates = np.clip(
                self.brain.hebb_rates * (1 + self.stress * 2), -0.1, 0.1
            )

        # Forward pass
        raw_out = self.brain.forward(obs)

        # Restore (the rates themselves are evolved; we only boosted temporarily)
        self.brain.hebb_rates = original_hebb

        # Decode
        actions = Brain.decode_actions(raw_out, self.cfg.max_speed)

        # ── Movement ─────────────────────────────────────────────────────
        dist = np.sqrt(actions["dx"] ** 2 + actions["dy"] ** 2)
        self.x, self.y = world.wrap(self.x + actions["dx"],
                                    self.y + actions["dy"])
        move_cost = dist * self.cfg.move_cost_per_unit

        # ── Metabolic drain ───────────────────────────────────────────────
        self.energy -= self.cfg.metabolic_cost + move_cost

        # ── Eat ───────────────────────────────────────────────────────────
        if actions["eat"] > 0.5:
            gained = world.consume(self.x, self.y, self.cfg.eat_amount)
            self.energy = min(self.cfg.max_energy, self.energy + gained)

        # ── Update signal ─────────────────────────────────────────────────
        self.signal = actions["signal"]

        # ── Update stress ─────────────────────────────────────────────────
        threat_exposure = world.threats_at(self.x, self.y)
        energy_stress = max(0.0, 1.0 - (self.energy / (self.cfg.max_energy * 0.4)))
        raw_stress = min(1.0, threat_exposure / self.cfg.threat_damage + energy_stress)
        # Stress decays toward raw_stress with a lag
        self.stress = float(np.clip(0.85 * self.stress + 0.15 * raw_stress, 0.0, 1.0))

        # ── Log stress-action pairs (for superstition analysis) ───────────
        if self.stress > 0.5:
            self.stress_action_log.append({
                "signal": self.signal,
                "eat": actions["eat"] > 0.5,
                "stress": self.stress,
                "x": self.x,
                "y": self.y,
            })

        # ── Mate cooldown ─────────────────────────────────────────────────
        if self.mate_cooldown > 0:
            self.mate_cooldown -= 1

        # ── Death checks ──────────────────────────────────────────────────
        if threat_exposure > 0:
            self.energy -= threat_exposure * 0.02  # partial damage each tick

        if self.energy <= 0 or self.age >= self.cfg.max_age:
            self.alive = False

        return actions

    # ── Social interaction helpers ────────────────────────────────────────────

    def receive_share(self, amount: float, sender_id: int, tick: int) -> None:
        self.energy = min(self.cfg.max_energy, self.energy + amount)
        self.memory.update(sender_id, valence_delta=+1.0, current_tick=tick)

    def take_attack_damage(self, amount: float, attacker_id: int,
                           tick: int) -> float:
        """Return actual amount stolen."""
        stolen = min(self.energy, amount)
        self.energy -= stolen
        self.memory.update(attacker_id, valence_delta=-1.0, current_tick=tick)
        if self.energy <= 0:
            self.alive = False
        return stolen

    def receive_attack_reward(self, stolen: float, victim_id: int,
                              tick: int) -> None:
        self.energy = min(self.cfg.max_energy, self.energy + stolen)
        self.memory.update(victim_id, valence_delta=-0.3, current_tick=tick)
        self.total_attacked += stolen

    def update_coexistence(self, other_id: int, tick: int) -> None:
        """Call each tick two agents are in sight of each other.

        Peaceful proximity builds mild positive association — the 'mere exposure'
        effect. In real mammals, simply being near familiar non-threatening
        conspecifics triggers oxytocin release and reduces cortisol. A small
        positive delta here allows bonds to form through sustained coexistence
        without requiring active sharing.
        """
        self.memory.update(other_id, valence_delta=0.1, current_tick=tick)

    # ── Reproduction ──────────────────────────────────────────────────────────

    def can_mate(self) -> bool:
        return (self.alive
                and self.energy >= self.cfg.mate_energy_threshold
                and self.mate_cooldown == 0)

    def spend_mate_energy(self) -> None:
        self.energy -= self.cfg.mate_cost
        self.mate_cooldown = 200  # ticks before mating again

    # ── Fitness (used by GA as tiebreaker, not primary selection criterion) ───

    @property
    def fitness(self) -> float:
        return float(self.ticks_survived + self.offspring_count * 500)

    def __repr__(self) -> str:
        return (f"Agent(id={self.id}, age={self.age}, "
                f"energy={self.energy:.1f}, alive={self.alive})")
