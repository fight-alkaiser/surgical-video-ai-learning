import genesis as gs

# ----------------------------------------
# What this script demonstrates
#
# Day37, step 2: does Genesis's Fluid (SPH) solver actually couple with
# Rigid Body objects, or does the liquid just pass through them? A
# block of liquid falls into the same walled basin as
# 10_fluid_sph_liquid.py, but this time there's a fixed rigid box
# sitting in the middle of the floor. If the coupling works, the
# liquid should be diverted around/over the box instead of overlapping
# it.
#
# Still no surgical framing -- pure capability check: can two
# different solvers (SPH liquid, Rigid box) interact with each other
# in the same scene at all.
# ----------------------------------------

gs.init(backend=gs.cpu)

scene = gs.Scene(
    show_viewer=False,
    sim_options=gs.options.SimOptions(dt=4e-3, substeps=10),
    sph_options=gs.options.SPHOptions(
        lower_bound=(-0.4, -0.4, 0.0),
        upper_bound=(0.4, 0.4, 0.8),
        particle_size=0.015,
    ),
    vis_options=gs.options.VisOptions(
        ambient_light=(0.6, 0.6, 0.6),
        background_color=(0.9, 0.9, 0.9),
    ),
)

scene.add_entity(gs.morphs.Plane())

wall_half_extent = 0.35
wall_height = 0.3
scene.add_entity(gs.morphs.Plane(pos=(wall_half_extent, 0, wall_height), normal=(-1, 0, 0)))
scene.add_entity(gs.morphs.Plane(pos=(-wall_half_extent, 0, wall_height), normal=(1, 0, 0)))
scene.add_entity(gs.morphs.Plane(pos=(0, wall_half_extent, wall_height), normal=(0, -1, 0)))
scene.add_entity(gs.morphs.Plane(pos=(0, -wall_half_extent, wall_height), normal=(0, 1, 0)))

# A fixed rigid obstacle sitting on the basin floor, off-center so the
# falling liquid block (still centered at x=0) has to flow around it.
obstacle = scene.add_entity(
    gs.morphs.Box(pos=(0.12, 0.0, 0.06), size=(0.12, 0.12, 0.12), fixed=True),
    material=gs.materials.Rigid(needs_coup=True),
    surface=gs.surfaces.Default(color=(0.85, 0.35, 0.3, 1.0)),
)

liquid = scene.add_entity(
    material=gs.materials.SPH.Liquid(),
    morph=gs.morphs.Box(pos=(-0.1, 0.0, 0.4), size=(0.2, 0.2, 0.2)),
    surface=gs.surfaces.Default(color=(0.2, 0.5, 0.9, 0.9), vis_mode="particle"),
)

camera = scene.add_camera(
    res=(320, 240),
    pos=(1.1, -1.1, 0.9),
    lookat=(0.0, 0.0, 0.15),
    fov=45,
)

scene.build()

frames = []
num_steps = 220

for i in range(num_steps):
    scene.step()
    if i % 3 == 0:
        rgb, _, _, _ = camera.render()
        frames.append(rgb)
        if i % 30 == 0:
            pos = liquid.get_particles_pos().cpu().numpy()
            near_obstacle = ((pos[:, 0] > 0.06) & (pos[:, 0] < 0.18) & (pos[:, 1] > -0.06) & (pos[:, 1] < 0.06)).sum()
            print(f"step {i:4d} | particles occupying obstacle footprint: {near_obstacle}")

import imageio
imageio.mimsave("fluid_sph_rigid_obstacle.gif", frames, duration=0.05)
print("Saved fluid_sph_rigid_obstacle.gif")
