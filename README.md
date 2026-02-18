# Human Emotion Evolution Simulation

A multi-agent simulation asking: **can complex emotions like love, tribalism,
religion, and dominance emerge naturally from agents whose only drive is
survival?**

## The Hypothesis

Emotions aren't special.  They are evolutionary strategies:

| Emotion | Survival value |
|---|---|
| **Love / bonding** | Cooperative pair survives better than isolated individual |
| **Tribalism** | In-group coordination beats inter-group competition in resource-scarce environments |
| **Religion / ritual** | Calming behaviours under stress conserve energy; shared rituals enable group coordination |
| **Dominance / hate** | Controlling resource patches outcompetes weaker agents |

None of these are coded in.  The agents have no "love" output.  The simulation
*measures* these as emergent behavioural patterns.

## Architecture

```
Environment   2D toroidal world, uneven Gaussian resource patches,
              random threat events (predators / storms)

Agent brain   77-input → 48-hidden → 14-output neural network
              Layer 2 has Hebbian plastic weights that update within a
              lifetime (fast timescale) allowing within-life social learning

Genome        6830 real numbers: all slow weights + evolved Hebbian rates

Evolution     Continuous sexual reproduction (both parents must be willing
              and energetic), uniform crossover, Gaussian mutation
              → no artificial "generation" boundaries

Metrics       Four detectors run every N ticks and log:
              - Bond scores (love)
              - Tribal bias (tribalism)
              - Stress-signal entropy (ritual / religion)
              - Energy Gini coefficient (dominance)
```

## Quick Start

```bash
pip install -r requirements.txt

# Default run (200k ticks)
python main.py

# Short test run, no rendering
python main.py --ticks 5000 --no-render

# Resume from checkpoint
python main.py --resume checkpoints/tick_00005000.pkl
```

## Output

- `frames/`       – PNG frames (stitch to video with `ffmpeg`)
- `metrics/`      – JSON history of all four emotion metrics
- `checkpoints/`  – Pickled simulation state every 5000 ticks

## What to look for

Run the simulation for ~50k+ ticks.  You should observe:

1. **Tribal signal clusters** – the bar chart (panel 3) converges from a
   uniform distribution to one or two dominant signals.  Agents with the
   same signal will develop positive valence; different signals, negative.

2. **Bond lines** – white lines in the world view connecting pairs with high
   mutual familiarity.  These pairs share food and stay near each other.

3. **Ritual convergence rising** – agents under stress converge on emitting
   the same signal even though it has no direct effect.  This is the
   precursor to religious/superstitious behaviour.

4. **Gini coefficient rising then stabilising** – inequality grows as some
   agents discover that controlling resource patches is more efficient than
   sharing.  The stable level reflects the tension between exploitation and
   cooperative strategies.
