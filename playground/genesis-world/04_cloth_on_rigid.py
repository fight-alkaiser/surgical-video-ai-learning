import genesis as gs

# ----------------------------------------
# What this script demonstrates
#
# Day36 goal: can Genesis pseudo-model a surgical field? Step 1 of 6:
# a Cloth sheet (PBD.Cloth) falls onto a fixed Rigid Body sphere and
# drapes over it, staying in contact but not attached anywhere.
#
# Surgical analogy: the omentum (or peritoneum) draping loosely over
# an organ, following its shape from gravity + contact alone, with no
# fixed attachment point yet (that comes in 05_cloth_attached_rigid.py).
#
# Physics: PBD (Position Based Dynamics) cloth simulation, and a
# one-way Cloth-Rigid coupling (the cloth reacts to the sphere, the
# sphere is fixed so it doesn't react back).
#
# Based on Genesis's official examples/coupling/cloth_on_rigid.py,
# adapted to CPU backend + offscreen rendering instead of the
# interactive viewer.
# ----------------------------------------

gs.init(backend=gs.cpu)

scene = gs.Scene(
    show_viewer=False,
    sim_options=gs.options.SimOptions(dt=2e-3, substeps=10),
    pbd_options=gs.options.PBDOptions(particle_size=1e-2),
    vis_options=gs.options.VisOptions(
        ambient_light=(0.6, 0.6, 0.6),
        background_color=(0.9, 0.9, 0.9),
    ),
)

frictionless_rigid = gs.materials.Rigid(needs_coup=True, coup_friction=0.0)

scene.add_entity(gs.morphs.Plane(), material=frictionless_rigid)

organ = scene.add_entity(
    morph=gs.morphs.Sphere(radius=0.2, pos=(0.0, 0.0, 0.0), fixed=True),
    material=frictionless_rigid,
)

cloth = scene.add_entity(
    material=gs.materials.PBD.Cloth(),
    morph=gs.morphs.Mesh(
        file="meshes/cloth.obj",
        scale=1.0,
        pos=(0.0, 0.0, 0.3),
        euler=(180.0, 0.0, 0.0),
    ),
    surface=gs.surfaces.Default(color=(0.2, 0.4, 0.8, 1.0)),
)

camera = scene.add_camera(
    res=(320, 240),
    pos=(1.0, -1.0, 0.7),
    lookat=(0.0, 0.0, 0.0),
    fov=40,
)

scene.build()

frames = []
num_steps = 300

for i in range(num_steps):
    scene.step()
    if i % 6 == 0:
        rgb, _, _, _ = camera.render()
        frames.append(rgb)

import imageio
imageio.mimsave("cloth_on_rigid.gif", frames, duration=0.04)
print("Saved cloth_on_rigid.gif")
