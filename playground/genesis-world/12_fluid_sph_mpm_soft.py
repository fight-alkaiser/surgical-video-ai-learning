import genesis as gs

# ----------------------------------------
# What this script demonstrates
#
# Day37, step 3: same walled basin, but the obstacle from
# 11_fluid_sph_rigid_obstacle.py is replaced with a Soft Body (MPM
# Elastic) block instead of a Rigid one. Does the falling liquid just
# flow around it like the rigid box did, or does it also push/deform
# the soft block since the soft block itself can move and change
# shape? Two solvers that both deal in deformable/moving matter (SPH
# liquid, MPM elastic) interacting is a different kind of coupling
# question than liquid-vs-immovable-rigid.
#
# Still a pure capability check, no surgical framing forced.
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
    mpm_options=gs.options.MPMOptions(
        lower_bound=(-0.4, -0.4, 0.0),
        upper_bound=(0.4, 0.4, 0.8),
        grid_density=64,
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

soft_block = scene.add_entity(
    material=gs.materials.MPM.Elastic(E=2e4, nu=0.3),
    morph=gs.morphs.Box(pos=(0.12, 0.0, 0.11), size=(0.12, 0.12, 0.12)),
    surface=gs.surfaces.Default(color=(0.85, 0.35, 0.3, 1.0), vis_mode="particle"),
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
            soft_pos = soft_block.get_particles_pos().cpu().numpy()
            centroid = soft_pos.mean(axis=0)
            print(f"step {i:4d} | soft block centroid: x={centroid[0]:.3f} y={centroid[1]:.3f} z={centroid[2]:.3f}")

import imageio
imageio.mimsave("fluid_sph_mpm_soft.gif", frames, duration=0.05)
print("Saved fluid_sph_mpm_soft.gif")
