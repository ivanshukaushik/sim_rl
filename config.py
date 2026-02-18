"""
Simulation configuration.

All hyperparameters live here. Change these to explore different evolutionary
pressures and see how emergent behaviours shift.
"""
from dataclasses import dataclass, field


@dataclass
class Config:
    # ── World ─────────────────────────────────────────────────────────────────
    world_size: float = 100.0        # Width and height of the arena (continuous)
    num_resource_patches: int = 8    # Number of gaussian food blobs
    patch_peak: float = 8.0          # Max resource density at patch centre
    patch_radius: float = 12.0       # Std-dev of gaussian patch (bigger = gentler gradient)
    resource_regen_rate: float = 0.04 # Fraction of (capacity - current) restored each tick

    # Threats – random damaging events that create selection pressure
    # and a substrate for superstitious/religious behaviour
    threat_prob_per_tick: float = 0.002   # Probability a new threat spawns each tick
    threat_radius: float = 6.0            # Spatial radius of damage
    threat_damage: float = 25.0           # Energy removed from agents inside radius
    threat_duration: int = 10             # Ticks the threat persists

    # ── Population ───────────────────────────────────────────────────────────
    initial_population: int = 120
    max_population: int = 400

    # ── Agent biology ─────────────────────────────────────────────────────────
    max_energy: float = 100.0
    initial_energy: float = 55.0
    max_age: int = 2500               # Ticks before natural death

    vision_radius: float = 14.0       # How far an agent can see
    max_visible_agents: int = 5       # Max neighbours fed into the brain
    num_signals: int = 8              # Discrete communication channel (tribal marker)

    # Memory – how many unique agents to remember
    memory_capacity: int = 12

    # ── Energy economy ────────────────────────────────────────────────────────
    metabolic_cost: float = 0.4       # Energy drained each tick just to exist
    move_cost_per_unit: float = 0.15  # Extra cost per unit of distance moved
    max_speed: float = 2.2

    eat_amount: float = 3.5           # Max energy extracted from world per eat action
    share_amount: float = 5.0         # Energy transferred from sharer to receiver
    attack_steal: float = 8.0         # Energy stolen on successful attack
    attack_cost: float = 3.0          # Energy cost to the attacker regardless of outcome
    mate_cost: float = 18.0           # Energy cost of reproducing
    mate_energy_threshold: float = 65.0   # Minimum energy to be willing to mate

    # ── Neural network ────────────────────────────────────────────────────────
    hidden_size: int = 48

    # ── Evolution ─────────────────────────────────────────────────────────────
    mutation_rate: float = 0.025      # Fraction of genome entries mutated
    mutation_std: float = 0.12        # Std-dev of gaussian noise applied to mutated genes
    crossover_rate: float = 0.6       # Probability of using crossover (vs pure clone+mutate)
    tournament_size: int = 5          # k in k-tournament selection

    # ── Simulation ────────────────────────────────────────────────────────────
    max_ticks: int = 200_000
    metrics_interval: int = 500       # Log metrics every N ticks
    render_interval: int = 200        # Re-draw visualisation every N ticks
    checkpoint_interval: int = 5_000  # Save checkpoint every N ticks

    # ── Derived (computed at runtime, do not set manually) ────────────────────
    # INPUT_SIZE  = 12 own-state + 13 × max_visible_agents = 77
    # OUTPUT_SIZE = 14
    # GENOME_SIZE = computed in brain.py

    def input_size(self) -> int:
        return 12 + 13 * self.max_visible_agents

    def output_size(self) -> int:
        return 14
