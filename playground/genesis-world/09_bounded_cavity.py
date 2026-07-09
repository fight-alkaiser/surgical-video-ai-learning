import genesis as gs
import numpy as np

# ----------------------------------------
# What this script demonstrates
#
# Day36 goal: can Genesis pseudo-model a surgical field? Step 6 of 6:
# the same jostling box pile as 08_multi_body_pile.py, but now boxed
# in by four vertical walls (Plane entities) instead of an open floor.
#
# Surgical analogy: the abdominal cavity itself. Organs don't just
# settle on an open surface -- they are constrained by the surrounding
# cavity wall, which changes what configurations they can settle into.
#
# Physics: boundary conditions. The walls don't add any new force law
# -- they're just more Rigid Body contact surfaces -- but they shrink
# the reachable state space, which is the whole point of a "field": not
# a new kind of physics, but a shaped container for the same physics.
#
# Not based on a single official example -- built by adding extra
# gs.morphs.Plane() walls (rotated 90 degrees) around the pile from
# 08_multi_body_pile.py.
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

# Four walls forming a square cavity, each a vertical Plane whose
# normal points inward. wall_half_extent is how far each wall is from
# the center -- small enough that the settling boxes will bump into it.
wall_half_extent = 0.35
wall_height = 0.6

scene.add_entity(gs.morphs.Plane(pos=(wall_half_extent, 0, wall_height), normal=(-1, 0, 0)))
scene.add_entity(gs.morphs.Plane(pos=(-wall_half_extent, 0, wall_height), normal=(1, 0, 0)))
scene.add_entity(gs.morphs.Plane(pos=(0, wall_half_extent, wall_height), normal=(0, -1, 0)))
scene.add_entity(gs.morphs.Plane(pos=(0, -wall_half_extent, wall_height), normal=(0, 1, 0)))

box_size = 0.12
num_cubes = 4
box_spacing = 0.9 * box_size
box_pos_offset = np.array([-0.2, 0.0, 0.05])

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
    pos=(1.1, -1.1, 0.9),
    lookat=(0.0, 0.0, 0.15),
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
imageio.mimsave("bounded_cavity.gif", frames, duration=0.04)
print("Saved bounded_cavity.gif")
