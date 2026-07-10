import genesis as gs

# ----------------------------------------
# What this script demonstrates
#
# Day37: Genesis also claims Fluid as one of its physics types
# (Rigid / Soft / Fluid / Cloth / Granular Material), which Day35-36
# never touched. This is the simplest possible check: drop a block of
# liquid (SPH -- Smoothed Particle Hydrodynamics) into a walled
# container and see if it behaves like a fluid (spreads out, splashes,
# settles into a puddle/pool shape) rather than like a Soft Body (which
# keeps its own shape and just deforms).
#
# No surgical framing attempted here on purpose -- this is a pure
# capability survey. Whether/how it's relevant to a surgical world
# model is a question for after seeing what it actually does.
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

# Four walls forming a small basin, same trick as Day36's bounded_cavity.
wall_half_extent = 0.35
wall_height = 0.3
scene.add_entity(gs.morphs.Plane(pos=(wall_half_extent, 0, wall_height), normal=(-1, 0, 0)))
scene.add_entity(gs.morphs.Plane(pos=(-wall_half_extent, 0, wall_height), normal=(1, 0, 0)))
scene.add_entity(gs.morphs.Plane(pos=(0, wall_half_extent, wall_height), normal=(0, -1, 0)))
scene.add_entity(gs.morphs.Plane(pos=(0, -wall_half_extent, wall_height), normal=(0, 1, 0)))

liquid = scene.add_entity(
    material=gs.materials.SPH.Liquid(),
    morph=gs.morphs.Box(
        pos=(0.0, 0.0, 0.4),
        size=(0.2, 0.2, 0.2),
    ),
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
num_steps = 200

for i in range(num_steps):
    scene.step()
    if i % 3 == 0:
        rgb, _, _, _ = camera.render()
        frames.append(rgb)
        pos = liquid.get_particles_pos().cpu().numpy() if hasattr(liquid, "get_particles_pos") else None
        if pos is not None and i % 30 == 0:
            print(
                f"step {i:4d} | z: {pos[:,2].min():.3f}-{pos[:,2].max():.3f} | "
                f"xy spread: {pos[:,0].max()-pos[:,0].min():.3f} x {pos[:,1].max()-pos[:,1].min():.3f}"
            )

import imageio
imageio.mimsave("fluid_sph_liquid.gif", frames, duration=0.05)
print("Saved fluid_sph_liquid.gif")
