# Genesis World — Playground

Small architecture-level playground, separate from the day-numbered
CholecT50 challenge (`day01_...` onward), in the same spirit as
`playground/object-centric-world-models` and `playground/neural-physics-engine`.
The design-philosophy discussion itself (Genesis as a Simulation
Platform vs. Dreamer as a World Model, the relevance of Rigid/Soft
Body/Cloth/Fluid to surgery) lives in the Day35 LinkedIn post; this
folder is a small piece of code to check that understanding by actually
running the library, not just reading its README.

**Library:** [`genesis-world`](https://pypi.org/project/genesis-world/)
1.2.1 (PyPI), run on CPU backend on an Apple M2.

## What's here

Three self-contained scripts, run inside a local `.venv` (Genesis needs
Python ≥3.10; the machine's default `python3` was 3.9, so this uses
`pyenv`'s 3.11.5):

- **`01_rigid_body_falling_sphere.py`** — the simplest possible scene: a
  Rigid Body sphere (radius 0.1) falls from z=1.0 onto a ground plane.
  Renders an offscreen camera to `falling_sphere.gif`.
- **`02_gravity_parameter_change.py`** — same scene, run twice with only
  `gravity` changed (Earth vs. Moon), to check that the simulation
  actually responds to the physics parameter rather than replaying a
  fixed animation.
- **`03_soft_body_elastic_cube.py`** — a Soft Body cube (`MPM.Elastic`,
  Material Point Method, 8,000 particles) dropped from z=0.8, softened
  to E=20,000 (default is 300,000) so the squash-and-bounce is visible.
  Renders to `soft_body_cube.gif`.

```
python3.11 -m venv .venv
source .venv/bin/activate
pip install genesis-world torch imageio
python 01_rigid_body_falling_sphere.py
```

## What it found

**Rigid Body vs. Soft Body, same drop, same ground:**

| | Rigid Body (sphere) | Soft Body (cube, `MPM.Elastic`) |
|---|---|---|
| ![sphere](falling_sphere.gif) | falls, settles at z≈0.0996 (its own radius) and **stays perfectly still** | falls, **squashes** on impact (footprint 0.200→0.282), then **bounces** back up to z≈0.575 |
| ![cube](soft_body_cube.gif) | | |

The sphere never bounces — the default Rigid material has zero
restitution, so contact just kills its velocity. The elastic cube does
the opposite: it stores energy while squashed and releases it as a
bounce. Same "drop it on the ground" setup, opposite outcome, purely
from swapping one `material=` argument.

**Gravity parameter check** (`02_gravity_parameter_change.py`): with
Earth gravity (-9.81) the sphere settles by step 48; with Moon gravity
(-1.62, about 1/6) it takes until step 106. The simulation responds to
the physical parameter in the expected direction and rough magnitude —
evidence this is an actual physics calculation, not a canned animation.

## Not in scope here

No robot/articulated-body scenes, no Fluid or Cloth solver, no
Sim-to-Real transfer. This only checks that Rigid Body and Soft Body
behave differently under the same setup, and that changing one
parameter changes the outcome — the two claims the Day35 notes were
built on before touching any code.
