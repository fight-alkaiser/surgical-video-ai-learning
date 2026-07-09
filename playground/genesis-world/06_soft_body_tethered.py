import torch
import genesis as gs

# ----------------------------------------
# What this script demonstrates
#
# Day36 goal: can Genesis pseudo-model a surgical field? Step 3 of 6:
# a Soft Body (MPM.Elastic) cube has only its TOP particles attached to
# a moving Rigid Body box; the rest of the cube's mass hangs free below
# it. The rigid box is driven up and down.
#
# Surgical analogy: an organ held by a ligament or mesentery at one
# point while the rest of its (soft, deformable) mass hangs free and
# swings/stretches as the anchor point moves.
#
# Physics: constraint propagation through a deformable body, not just a
# rigid one. Unlike 05 (a rigid box carrying a whole cloth patch), here
# only ~particles at the top are constrained and the object itself is
# deformable, so the anchor motion has to visibly stretch/compress the
# soft body rather than just carry it rigidly.
#
# Based on Genesis's official examples/coupling/rigid_mpm_attachment.py,
# adapted to CPU backend + offscreen rendering.
# ----------------------------------------

gs.init(backend=gs.cpu)

scene = gs.Scene(
    show_viewer=False,
    sim_options=gs.options.SimOptions(dt=2e-3, substeps=20),
    mpm_options=gs.options.MPMOptions(
        lower_bound=(-1.0, -1.0, 0.0),
        upper_bound=(1.0, 1.0, 1.5),
        grid_density=64,
    ),
    vis_options=gs.options.VisOptions(
        ambient_light=(0.6, 0.6, 0.6),
        background_color=(0.9, 0.9, 0.9),
    ),
)

scene.add_entity(gs.morphs.Plane())

rigid_box = scene.add_entity(
    gs.morphs.Box(pos=(0.0, 0.0, 0.55), size=(0.12, 0.12, 0.05), fixed=False),
)

organ = scene.add_entity(
    material=gs.materials.MPM.Elastic(E=5e4, nu=0.3, rho=1000),
    morph=gs.morphs.Box(pos=(0.0, 0.0, 0.35), size=(0.15, 0.15, 0.15)),
    surface=gs.surfaces.Default(color=(0.85, 0.35, 0.3, 1.0)),
)

camera = scene.add_camera(
    res=(320, 240),
    pos=(1.0, -0.9, 0.75),
    lookat=(0.0, 0.0, 0.4),
    fov=40,
)

scene.build()

# Attach only the top slice of the organ's particles to the rigid box
# -- the "ligament" -- leaving the rest of the soft body free.
mask = organ.get_particles_in_bbox((-0.08, -0.08, 0.41), (0.08, 0.08, 0.44))
organ.set_particle_constraints(mask, rigid_box.links[0].idx, stiffness=1e5)

frames = []
n_steps = 220
initial_z = 0.55

for i in range(n_steps):
    z_offset = 0.12 * (1 - abs((i % 200) - 100) / 100.0)
    target_qpos = torch.tensor(
        [0.0, 0.0, initial_z + z_offset, 1.0, 0.0, 0.0, 0.0], device=gs.device
    )
    rigid_box.set_qpos(target_qpos)
    scene.step()
    if i % 4 == 0:
        rgb, _, _, _ = camera.render()
        frames.append(rgb)
        z = organ.get_particles_pos().cpu().numpy()[:, 2]
        print(f"step {i:4d} | organ z-range: {z.min():.3f} - {z.max():.3f} (span {z.max()-z.min():.3f})")

import imageio
imageio.mimsave("soft_body_tethered.gif", frames, duration=0.05)
print("Saved soft_body_tethered.gif")
