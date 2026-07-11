import numpy as np
import torch
import genesis as gs

# ----------------------------------------
# What this script demonstrates
#
# Day38: the capstone of the Day36-37 Genesis arc. Day36 built six
# separate mechanisms (draping, adhesion, tethering, crowding,
# confinement, grasping) as isolated scripts. This combines five of
# them (grasping is simplified from a full Franka arm to two directly-
# driven rigid "jaws", to stay CPU-feasible) into ONE scene with a
# concrete situation:
#
#   A laparoscopic cholecystectomy moment: the gallbladder (a soft
#   MPM body) sits in the bounded abdominal cavity, tethered at one
#   edge by the cystic duct/mesentery (a fixed rigid anchor pulling on
#   a slice of the organ's particles). Neighboring organs crowd the
#   cavity. A patch of omentum (Cloth) is adhered to a neighboring
#   organ and draped over the gallbladder -- then the adhesion is
#   released partway through, as if it had just been dissected.
#   Finally, two simplified forceps jaws close in and grasp the
#   gallbladder.
#
# The point isn't a faithful surgical simulation -- it's checking
# whether Genesis can hold all these mechanisms in one scene at once
# without the solvers interfering with each other, closing out the
# open question Day36 left ("still six separate scenes, not one").
# The Emitter+Rigid bug found on Day37 doesn't apply here since
# nothing uses scene.add_emitter().
# ----------------------------------------

gs.init(backend=gs.cpu)

scene = gs.Scene(
    show_viewer=False,
    # dt=2e-2 (carried over from the Rigid/Cloth-only Day36 scripts)
    # was simply too coarse for MPM: it made the organ collapse to a
    # single point within ~20 steps. Isolation tests confirmed the
    # root cause precisely -- dt=2e-3 (Day36 script06's proven-stable
    # MPM setting) keeps the tethered organ cohesive, with or without
    # the jaws teleporting via set_pos. It wasn't "too many mechanisms
    # combined" -- it was one wrong timestep carried over by habit.
    sim_options=gs.options.SimOptions(dt=2e-3, substeps=20),
    pbd_options=gs.options.PBDOptions(particle_size=1e-2),
    mpm_options=gs.options.MPMOptions(
        lower_bound=(-0.4, -0.4, 0.0),
        upper_bound=(0.4, 0.4, 0.6),
        grid_density=64,
    ),
    vis_options=gs.options.VisOptions(
        ambient_light=(0.6, 0.6, 0.6),
        background_color=(0.9, 0.9, 0.9),
    ),
)

rigid_default = gs.materials.Rigid(needs_coup=True)

# ----- the abdominal cavity: floor + four walls -----
scene.add_entity(gs.morphs.Plane())

wall_half_extent = 0.4
wall_height = 0.35
for pos, normal in [
    ((wall_half_extent, 0, wall_height), (-1, 0, 0)),
    ((-wall_half_extent, 0, wall_height), (1, 0, 0)),
    ((0, wall_half_extent, wall_height), (0, -1, 0)),
    ((0, -wall_half_extent, wall_height), (0, 1, 0)),
]:
    scene.add_entity(gs.morphs.Plane(pos=pos, normal=normal))

# ----- neighboring organs: crowding, static -----
for x, y in [(-0.25, 0.18), (-0.22, -0.2)]:
    scene.add_entity(
        gs.morphs.Box(pos=(x, y, 0.06), size=(0.12, 0.1, 0.12), fixed=True),
        material=rigid_default,
        surface=gs.surfaces.Default(color=(0.55, 0.25, 0.2, 1.0)),
    )

# ----- the gallbladder: soft MPM body -----
organ = scene.add_entity(
    material=gs.materials.MPM.Elastic(E=3e4, nu=0.3, rho=1000),
    morph=gs.morphs.Box(pos=(0.0, 0.0, 0.14), size=(0.16, 0.16, 0.16)),
    surface=gs.surfaces.Default(color=(0.3, 0.55, 0.25, 1.0), vis_mode="particle"),
)

# ----- mesentery/cystic duct anchor: fixed rigid point pulling one edge -----
anchor = scene.add_entity(
    gs.morphs.Box(pos=(0.3, 0.0, 0.14), size=(0.04, 0.04, 0.04), fixed=True),
    material=rigid_default,
    surface=gs.surfaces.Default(color=(0.8, 0.8, 0.2, 1.0)),
)

# ----- adhesion target: a small neighboring organ the omentum sticks to -----
adhesion_anchor = scene.add_entity(
    gs.morphs.Box(pos=(-0.05, 0.16, 0.2), size=(0.05, 0.05, 0.05), fixed=True),
    material=rigid_default,
    surface=gs.surfaces.Default(color=(0.8, 0.8, 0.2, 1.0)),
)

# ----- omentum patch draping over the gallbladder, adhered at one corner -----
cloth = scene.add_entity(
    gs.morphs.Mesh(pos=(-0.05, 0.05, 0.32), scale=0.35, file="meshes/cloth.obj"),
    gs.materials.PBD.Cloth(),
    gs.surfaces.Default(color=(0.85, 0.75, 0.5, 1.0)),
)

# ----- simplified forceps jaws: two rigid boxes driven toward the organ -----
jaw1 = scene.add_entity(
    gs.morphs.Box(pos=(0.0, 0.16, 0.14), size=(0.03, 0.03, 0.08)),
    material=rigid_default,
    surface=gs.surfaces.Default(color=(0.75, 0.75, 0.78, 1.0)),
)
jaw2 = scene.add_entity(
    gs.morphs.Box(pos=(0.0, -0.16, 0.14), size=(0.03, 0.03, 0.08)),
    material=rigid_default,
    surface=gs.surfaces.Default(color=(0.75, 0.75, 0.78, 1.0)),
)

camera = scene.add_camera(
    res=(360, 280),
    pos=(0.85, -0.75, 0.65),
    lookat=(0.0, 0.0, 0.12),
    fov=45,
)

scene.build()

# Tether a slice of the organ's rightmost particles to the fixed anchor
mask = organ.get_particles_in_bbox((0.03, -0.08, 0.06), (0.09, 0.08, 0.22))
organ.set_particle_constraints(mask, anchor.links[0].idx, stiffness=3e3)

# Adhere the nearest corner of the cloth to the adhesion_anchor box
adhesion_particles = [0, 1, 2, 3]
cloth.fix_particles_to_link(adhesion_anchor.link_start, particles_idx_local=adhesion_particles)

frames = []


def render(step_i):
    if step_i % 3 == 0:
        rgb, _, _, _ = camera.render()
        frames.append(rgb)


# Phase 1: let everything settle (organ tethered, cloth draping+adhered,
# jaws open, neighbors static) -- 20 steps
for i in range(20):
    scene.step()
    render(i)

# Phase 2: dissect the adhesion -- release the cloth from the neighbor
cloth.release_particle(adhesion_particles)
for i in range(20, 35):
    scene.step()
    render(i)

# Phase 3: close the forceps jaws onto the gallbladder
n_grasp_steps = 30
jaw1_start, jaw1_end = 0.16, 0.065
jaw2_start, jaw2_end = -0.16, -0.065
for i in range(n_grasp_steps):
    t = i / (n_grasp_steps - 1)
    y1 = jaw1_start + t * (jaw1_end - jaw1_start)
    y2 = jaw2_start + t * (jaw2_end - jaw2_start)
    jaw1.set_pos(torch.tensor([0.0, y1, 0.14], device=gs.device))
    jaw2.set_pos(torch.tensor([0.0, y2, 0.14], device=gs.device))
    scene.step()
    render(35 + i)

# Phase 4: hold the grasp briefly
for i in range(10):
    scene.step()
    render(65 + i)

import imageio
imageio.mimsave("combined_surgical_field.gif", frames, duration=0.06)
print(f"Saved combined_surgical_field.gif ({len(frames)} frames)")
