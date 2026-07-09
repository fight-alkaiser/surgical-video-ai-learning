import math
import genesis as gs
import genesis.utils.geom as gu

# ----------------------------------------
# What this script demonstrates
#
# Day36 goal: can Genesis pseudo-model a surgical field? Step 2 of 6:
# a handful of Cloth particles are permanently fixed to a moving Rigid
# Body box; the rest of the cloth hangs free. As the box moves, the
# fixed points follow it exactly while the free cloth drapes and swings.
#
# Surgical analogy: adhesion. Unlike 04_cloth_on_rigid.py (cloth just
# resting on a fixed sphere, free to slide off), here specific points
# are permanently bonded -- exactly what an adhesion is: a small area
# stuck together while the surrounding tissue stays mobile.
#
# Physics: partial constraint propagation. Fixing only a few particles
# still shapes the whole cloth's motion, because the PBD solver
# propagates that constraint through the cloth's internal
# stretch/bending springs to every other particle.
#
# Based on Genesis's official examples/coupling/cloth_attached_to_rigid.py,
# adapted to CPU backend + offscreen rendering, and shortened (fewer
# steps, fewer moves) since Cloth (PBD) is slow on CPU (~1 step/sec).
# ----------------------------------------

gs.init(backend=gs.cpu)

dt = 2e-2
particle_size = 1e-2

scene = gs.Scene(
    show_viewer=False,
    sim_options=gs.options.SimOptions(dt=dt, substeps=10),
    pbd_options=gs.options.PBDOptions(particle_size=particle_size),
    vis_options=gs.options.VisOptions(
        ambient_light=(0.6, 0.6, 0.6),
        background_color=(0.9, 0.9, 0.9),
    ),
)

rigid_material = gs.materials.Rigid(needs_coup=True, coup_friction=0.0)

scene.add_entity(gs.morphs.Plane(), rigid_material)

box_morph = gs.morphs.Box(pos=[0.25, 0.25, 0.25], size=[0.2, 0.2, 0.2])
box = scene.add_entity(box_morph, rigid_material)

cloth_pos = (0.25, 0.25, 0.25 + 0.1 + particle_size)
cloth = scene.add_entity(
    gs.morphs.Mesh(pos=cloth_pos, scale=0.7, file="meshes/cloth.obj"),
    gs.materials.PBD.Cloth(),
    gs.surfaces.Default(color=(0.85, 0.25, 0.25, 1.0)),
)

camera = scene.add_camera(
    res=(320, 240),
    pos=(1.5, -0.5, 1.1),
    lookat=(0.4, 0.4, 0.25),
    fov=50,
)

scene.build()

# Fix a handful of particles at one corner of the cloth to the box --
# an "adhesion patch" -- and leave the rest free.
adhesion_particles = [0, 1, 2, 3]
box_link_idx = box.link_start
cloth.fix_particles_to_link(box_link_idx, particles_idx_local=adhesion_particles)

frames = []


def run_and_record(num_steps, render_every=3):
    for i in range(num_steps):
        scene.step()
        if i % render_every == 0:
            rgb, _, _, _ = camera.render()
            frames.append(rgb)
            box_pos = box.get_pos().cpu().numpy()
            print(f"  box pos: {box_pos}")


# Phase 1: settle under gravity with the adhesion patch fixed.
run_and_record(30)

# Phase 2: move the box -- the adhered corner follows exactly, the
# rest of the cloth swings and drapes freely.
box.set_dofs_velocity([0.0, 0.8, 0.0], dofs_idx_local=[0, 1, 2])
run_and_record(30)

box.set_dofs_velocity([0.0, 0.0, 0.0], dofs_idx_local=[0, 1, 2])
run_and_record(15)

# Phase 3: release the adhesion -- the whole cloth should now fall
# freely, no longer following the box.
cloth.release_particle(adhesion_particles)
run_and_record(30)

import imageio
imageio.mimsave("cloth_attached_adhesion.gif", frames, duration=0.06)
print("Saved cloth_attached_adhesion.gif")
