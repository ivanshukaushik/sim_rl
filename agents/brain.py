"""
Agent neural network with Hebbian (within-lifetime) plasticity.

Architecture
────────────
                   ┌────────────────────────────────────┐
  observation ──►  │ FC1 (input→hidden)   tanh           │
   (77 dims)       │ FC2 (hidden→hidden)  tanh            │  ← slow weights, EVOLVED
                   │ FC2_plastic                          │  ← fast weights, RESET at birth,
                   │   updated via Hebbian rule each step │    but LEARNING RATE evolved
                   │ FC3 (hidden→output)  raw             │
                   └────────────────────────────────────┘
                              ↓
                          action vector (14 dims)

Two timescales
──────────────
* Slow weights (W1, W2, W3, biases, hebb_rates) are the GENOME and change only
  through inter-generational selection and mutation.

* Fast / plastic weights (plastic_W2) start at zero each new life and drift
  via Hebbian updates.  This lets an agent learn WITHIN its lifetime:
  - Which neighbours are trustworthy (love / hate)
  - Which places are dangerous (proto-religion / superstition)
  - How to respond to stress signals

The `hebb_rates` vector (one per hidden neuron) is also EVOLVED.  Positive
values create standard Hebbian potentiation; negative values create
anti-Hebbian (inhibitory) traces.  Evolution can discover which neurons
*should* be plastic and which should stay fixed.

Observation format (INPUT_SIZE = 12 + 13 × max_visible)
─────────────────
  [0]     energy / max_energy
  [1]     age / max_age
  [2-9]   own signal (one-hot, 8 bits)
  [10]    stress level (0-1)
  [11]    local resource density (normalised)
  for each of up to 5 visible neighbours:
    [base+0]   relative_x / vision_radius
    [base+1]   relative_y / vision_radius
    [base+2-9] their signal (one-hot, 8 bits)
    [base+10]  their energy / max_energy
    [base+11]  familiarity (0-1)
    [base+12]  valence (-1 to 1)

Action format (OUTPUT_SIZE = 14)
────────────────────────────────
  [0]   dx  (tanh, scaled by max_speed)
  [1]   dy  (tanh, scaled by max_speed)
  [2]   eat probability (sigmoid)
  [3]   share probability (sigmoid)
  [4]   attack probability (sigmoid)
  [5-12] signal logits (softmax → argmax = signal to emit)
  [13]  mate willingness (sigmoid)
"""

from __future__ import annotations

import numpy as np
from typing import Tuple

# ── Architecture constants ────────────────────────────────────────────────────
N_OWN_STATE = 12
N_PER_AGENT = 13
MAX_VIS = 5
INPUT_SIZE = N_OWN_STATE + N_PER_AGENT * MAX_VIS   # 77
OUTPUT_SIZE = 14
HIDDEN_SIZE = 48   # default; overridable via Config

# Genome layout sizes (computed dynamically in Brain.genome_size())


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -10, 10)))


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max())
    return e / e.sum()


class Brain:
    """Pure-NumPy neural network with Hebbian plasticity."""

    def __init__(self, genome: np.ndarray | None = None,
                 hidden_size: int = HIDDEN_SIZE,
                 input_size: int = INPUT_SIZE,
                 output_size: int = OUTPUT_SIZE,
                 rng: np.random.Generator | None = None):

        self.hs = hidden_size
        self.ins = input_size
        self.outs = output_size
        self.rng = rng or np.random.default_rng()

        if genome is None:
            genome = self._random_genome()

        self._unpack(genome)

        # Plastic (within-lifetime) fast weights – reset on birth
        self.plastic_W2 = np.zeros((self.hs, self.hs), dtype=np.float64)

    # ── Genome management ─────────────────────────────────────────────────────

    def genome_size(self) -> int:
        return (
            self.ins * self.hs   # W1
            + self.hs            # b1
            + self.hs * self.hs  # W2
            + self.hs            # b2
            + self.hs * self.outs # W3
            + self.outs          # b3
            + self.hs            # hebb_rates (one per hidden neuron)
        )

    def _random_genome(self) -> np.ndarray:
        size = (
            self.ins * self.hs
            + self.hs
            + self.hs * self.hs
            + self.hs
            + self.hs * self.outs
            + self.outs
            + self.hs
        )
        # Xavier init for weight matrices, small random for the rest
        g = self.rng.standard_normal(size) * 0.1
        return g

    def _unpack(self, genome: np.ndarray) -> None:
        idx = 0

        def take(n: int) -> np.ndarray:
            nonlocal idx
            v = genome[idx: idx + n]
            idx += n
            return v.copy()

        self.W1 = take(self.ins * self.hs).reshape(self.ins, self.hs)
        self.b1 = take(self.hs)
        self.W2 = take(self.hs * self.hs).reshape(self.hs, self.hs)
        self.b2 = take(self.hs)
        self.W3 = take(self.hs * self.outs).reshape(self.hs, self.outs)
        self.b3 = take(self.outs)
        # hebb_rates: evolved, can be positive or negative
        # We store raw values; actual rate = 0.02 * tanh(raw)
        self._hebb_raw = take(self.hs)
        self.hebb_rates = 0.02 * np.tanh(self._hebb_raw)  # range ≈ (-0.02, +0.02)

    def get_genome(self) -> np.ndarray:
        return np.concatenate([
            self.W1.flatten(),
            self.b1,
            self.W2.flatten(),
            self.b2,
            self.W3.flatten(),
            self.b3,
            self._hebb_raw,
        ])

    # ── Forward pass ──────────────────────────────────────────────────────────

    def forward(self, obs: np.ndarray) -> np.ndarray:
        """
        Run one step.  Updates plastic weights as a side-effect.
        Returns raw output vector (caller decodes into actions).
        """
        obs = obs.astype(np.float64)

        # Layer 1
        h1 = np.tanh(obs @ self.W1 + self.b1)

        # Layer 2 – combine slow (evolved) + fast (Hebbian) weights
        W2_eff = self.W2 + self.plastic_W2
        h2 = np.tanh(h1 @ W2_eff + self.b2)

        # Hebbian update:  ΔW_ij = η_j * pre_i * post_j
        dw = np.outer(h1, h2)                          # (hs, hs)
        dw *= self.hebb_rates[np.newaxis, :]            # scale each column
        self.plastic_W2 = np.clip(self.plastic_W2 + dw, -1.5, 1.5)

        # Output layer (raw, decoded by caller)
        out = h2 @ self.W3 + self.b3
        return out

    # ── Action decoding ───────────────────────────────────────────────────────

    @staticmethod
    def decode_actions(out: np.ndarray, max_speed: float) -> dict:
        """Turn raw output vector into a named action dictionary."""
        dx = float(np.tanh(out[0]) * max_speed)
        dy = float(np.tanh(out[1]) * max_speed)
        eat = float(_sigmoid(out[2]))
        share = float(_sigmoid(out[3]))
        attack = float(_sigmoid(out[4]))
        signal_probs = _softmax(out[5:13])
        signal = int(np.argmax(signal_probs))
        mate = float(_sigmoid(out[13]))
        return {
            "dx": dx,
            "dy": dy,
            "eat": eat,
            "share": share,
            "attack": attack,
            "signal": signal,
            "signal_probs": signal_probs,
            "mate": mate,
        }

    # ── Lifetime reset ────────────────────────────────────────────────────────

    def reset_lifetime(self) -> None:
        """Called when a new agent is born – clears Hebbian fast weights."""
        self.plastic_W2[:] = 0.0
