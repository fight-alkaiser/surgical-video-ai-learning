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

## Not in scope here (Day35)

No robot/articulated-body scenes, no Fluid or Cloth solver, no
Sim-to-Real transfer. This only checks that Rigid Body and Soft Body
behave differently under the same setup, and that changing one
parameter changes the outcome — the two claims the Day35 notes were
built on before touching any code.

## Day36: toward a pseudo surgical field

Day35 only ever dropped one object onto an open floor. Day36 asks a
different question: can Genesis pseudo-construct something like a
surgical field — a bounded cavity with multiple objects, some attached
to each other, some grasped? An initial plan of "change one Environment
parameter (gravity/friction/floor) at a time" turned out not to
actually build toward that; it only taught individual API knobs. The
plan pivoted to Genesis's official `examples/coupling/` and
`examples/collision/`, which combine different physics types on
purpose — much closer to what a surgical field actually is.

| # | Script | Surgical analogy | Physics concept | Result |
|---|---|---|---|---|
| 1 | `04_cloth_on_rigid.py` | Omentum draping over an organ | PBD Cloth, one-way Cloth-Rigid coupling | ![cloth on rigid](cloth_on_rigid.gif) Cloth falls and drapes over a fixed sphere, no attachment point |
| 2 | `05_cloth_attached_adhesion.py` | Adhesion and its release | Partial constraint propagation through a deforming Cloth | ![cloth adhesion](cloth_attached_adhesion.gif) A few particles fixed to a moving box; rest of the cloth folds dynamically, falls free once released |
| 3 | `06_soft_body_tethered.py` | An organ held by a ligament/mesentery | Local constraint propagating through a deformable (MPM) body | ![soft body tethered](soft_body_tethered.gif) Only the top particles of an elastic cube are tethered to a moving rigid box; z-span stretches/compresses 0.13–0.17 as it's driven up and down |
| 4 | `08_multi_body_pile.py` | Multiple organs packed together | Multi-body contact, static friction | ![multi body pile](multi_body_pile.gif) Boxes start mildly overlapping and jostle into a stable pile |
| 5 | `09_bounded_cavity.py` | The abdominal cavity itself | Boundary conditions constrain the reachable state space | ![bounded cavity](bounded_cavity.gif) Same pile as #4, now boxed in by four walls; walls add no new force law, just narrow what configurations are possible |
| 6 | `07_grasp_soft_body_colab.ipynb` | An instrument grasping tissue | Contact-force grasping of a deformable body | ![grasp soft body](grasp_soft_body.gif) A Franka arm grips a soft MPM cube with 1N finger force and lifts it; the cube visibly dents where the fingers press in |

Scripts 1–5 ran locally on CPU (Apple M2) inside the same `.venv` as
Day35. Script 6 (Franka + IK control on a soft body) is heavy and
originally GPU-targeted in Genesis's own example, so it was run on
Google Colab's free T4 GPU instead — the notebook is checked in but not
runnable from this repo's local `.venv`.

**Honest caveat on #2:** the box was only meant to *carry* the adhered
cloth corner sideways, but `set_dofs_velocity` didn't sustain a
constant velocity across steps the way expected (or the cloth's own
tension pulled back quickly) — the box moved only ~0.03 in the
commanded direction instead of traversing the scene. The cloth's own
dynamic folding and the attach/release contrast are still visible, but
"the adhesion visibly drags the box" is not something this run
demonstrates.

**Where this leaves the surgical-field question:** these are six
separate, isolated mechanisms — draping, partial attachment, tethering,
crowding, confinement, grasping — each pseudo-modeling one piece of a
real surgical field, not one combined scene where all of them coexist
(one bounded cavity with several organs, some adhered, some grasped, at
once). Combining them is a natural next step but wasn't attempted here.

## Not in scope here (Day36)

No fluid solver (blood/irrigation), no photorealistic rendering (this
is purely mechanical — a rendering engine like Unreal would sit on top
of, not replace, this layer), no single scene combining all six
mechanisms at once.
