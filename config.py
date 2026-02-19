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
    num_resource_patches: int = 13   # More distributed food → less desperation-driven conflict
    patch_peak: float = 12.6554          # Max resource density at patch centre
    patch_radius: float = 12.0       # Std-dev of gaussian patch (bigger = gentler gradient)
    resource_regen_rate: float = 0.0289538 # Fraction of (capacity - current) restored each tick

    # Threats – random damaging events that create selection pressure
    # and a substrate for superstitious/religious behaviour
    threat_prob_per_tick: float = 0.00221844   # Probability a new threat spawns each tick
    threat_radius: float = 6.0            # Spatial radius of damage
    threat_damage: float = 35           # Real predators/hazards dangerous but rarely one-shot lethal
    threat_duration: int = 10             # Ticks the threat persists

    # ── Population ───────────────────────────────────────────────────────────
    initial_population: int = 150
    max_population: int = 600

    # ── Agent biology ─────────────────────────────────────────────────────────
    max_energy: float = 100.0
    initial_energy: float = 55.0
    max_age: int = 2500               # Ticks before natural death

    vision_radius: float = 18.8715       # Wider social awareness (real primates scan broadly)
    max_visible_agents: int = 8       # Track more neighbours; real primates monitor group
    num_signals: int = 8              # Discrete communication channel (tribal marker)

    # Memory – social brain hypothesis: primates track many relationships
    memory_capacity: int = 30

    # ── Energy economy ────────────────────────────────────────────────────────
    metabolic_cost: float = 0.29672      # Slightly efficient baseline; real metabolism is adaptive
    move_cost_per_unit: float = 0.15  # Extra cost per unit of distance moved
    max_speed: float = 2.2

    eat_amount: float = 5.25188           # Max energy extracted from world per eat action
    share_amount: float = 3.37901         # Generous sharing; real grooming/food-sharing alliances
    attack_steal: float = 2         # Real fights net less than expected (injury, resistance)
    attack_cost: float = 6.19301          # Aggression risks injury; ~17 ticks of metabolic cost
    mate_cost: float = 13.4798           # Energy cost of reproducing
    mate_energy_threshold: float = 61.2207   # Bonded pairs reproduce at lower individual thresholds

    # ── Catastrophic events (rare, large-scale — substrate for episodic ritual)
    catastrophe_prob_per_tick: float = 0.00005  # ~1 per 20k ticks
    catastrophe_radius: float = 35.0            # covers ~35% of world width
    catastrophe_damage: float = 75.0            # near-lethal; forces strong memory trace
    catastrophe_duration: int = 50              # lingers — agents must flee or die

    # ── Spatial / regional ────────────────────────────────────────────────────
    # Replacement agents born near existing neighbours, not globally random.
    # This creates regional gene pools and local "cultures".
    spawn_radius: float = 20.0

    # ── Cultural inheritance ───────────────────────────────────────────────────
    # Fraction of parent Hebbian fast-weights passed to offspring at birth.
    # Models cultural transmission: children begin life pre-primed by what
    # their parents learned, but only partially (fast weights are mostly reset).
    cultural_inheritance: float = 0.1

    # ── NEAT-like architecture evolution ──────────────────────────────────────
    # Hidden-layer size is heritable and mutable; populations evolve toward
    # more or less cognitively complex agents under selection pressure.
    hidden_size_min: int = 16
    hidden_size_max: int = 96

    # ── Neural network ────────────────────────────────────────────────────────
    hidden_size: int = 48   # starting default; individual agents can diverge

    # ── Evolution ─────────────────────────────────────────────────────────────
    mutation_rate: float = 0.025      # Fraction of genome entries mutated
    mutation_std: float = 0.12        # Std-dev of gaussian noise applied to mutated genes
    crossover_rate: float = 0.6       # Probability of using crossover (vs pure clone+mutate)
    tournament_size: int = 5          # k in k-tournament selection

    # ── Simulation ────────────────────────────────────────────────────────────
    max_ticks: int = 500_000
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
