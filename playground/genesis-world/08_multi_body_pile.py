import genesis as gs
import numpy as np

# ----------------------------------------
# What this script demonstrates
#
# Day36 goal: can Genesis pseudo-model a surgical field? Step 5 of 6:
# several Rigid Body boxes are dropped into a pile and left to settle,
# instead of the earlier scripts' single falling object.
#
# Surgical analogy: multiple organs packed into the abdominal cavity,
# pressing against and supporting each other rather than existing in
# isolation.
#
# Physics: multi-body contact. With several bodies touching at once,
# the solver has to resolve many simultaneous contact constraints, and
# static friction (not just kinetic) becomes what keeps the pile from
# collapsing once it settles.
#
# Based on Genesis's official examples/collision/pyramid.py, adapted
# to CPU backend + offscreen rendering instead of the interactive
# viewer, and fewer boxes for a shorter/lighter run.
# ----------------------------------------

gs.init(backend=gs.cpu)

scene = gs.Scene(
    show_viewer=False,
    sim_options=gs.options.SimOptions(dt=0.01),
    vis_options=gs.options.VisOptions(
        ambient_light=(0.6, 0.6, 0.6),
        background_color=(0.9, 0.9, 0.9),
    ),
)

scene.add_entity(gs.morphs.Plane())

box_size = 0.15
num_cubes = 4
# Slightly less than box_size: boxes start mildly interpenetrating, so
# the solver has to push them apart and let them jostle into a stable
# resting configuration, instead of starting already-stable.
box_spacing = 0.9 * box_size
box_pos_offset = np.array([-0.3, 0.0, 0.05])

for i in range(num_cubes):
    for j in range(num_cubes - i):
        scene.add_entity(
            gs.morphs.Box(
                size=[box_size, box_size, box_size],
                pos=box_pos_offset + box_spacing * np.array([i + 0.5 * j, 0, j]),
            ),
        )

camera = scene.add_camera(
    res=(320, 240),
    pos=(1.3, -1.3, 1.0),
    lookat=(0.0, 0.0, 0.2),
    fov=45,
)

scene.build()

frames = []
num_steps = 150

for i in range(num_steps):
    scene.step()
    if i % 3 == 0:
        rgb, _, _, _ = camera.render()
        frames.append(rgb)

import imageio
imageio.mimsave("multi_body_pile.gif", frames, duration=0.04)
print("Saved multi_body_pile.gif")
