import genesis as gs

# ----------------------------------------
# What this script demonstrates
#
# A Soft Body (MPM.Elastic material) cube falling onto a rigid
# Plane. Unlike the Rigid Body sphere in 01/02, this object is
# made of many particles (Material Point Method) and should
# visibly deform/squash on impact instead of staying a perfect
# sphere shape -- the same MPM idea the Day35 notes describe as
# relevant to soft tissue (liver, gallbladder) in a future
# Surgical World Model.
# ----------------------------------------

gs.init(backend=gs.cpu)

scene = gs.Scene(
    show_viewer=False,
    sim_options=gs.options.SimOptions(dt=0.002, substeps=8),
    mpm_options=gs.options.MPMOptions(
        lower_bound=(-1.0, -1.0, 0.0),
        upper_bound=(1.0, 1.0, 2.0),
    ),
    vis_options=gs.options.VisOptions(
        ambient_light=(0.6, 0.6, 0.6),
        background_color=(0.9, 0.9, 0.9),
    ),
)

scene.add_entity(gs.morphs.Plane())

# Default E (Young's modulus) is 300000 -- stiff, like rubber. Dropping
# it two orders of magnitude makes the cube jelly-soft, so the squash on
# impact is large enough to see clearly, and dropping from higher up
# gives it more impact energy to deform with.
cube = scene.add_entity(
    morph=gs.morphs.Box(
        pos=(0.0, 0.0, 0.8),
        size=(0.2, 0.2, 0.2),
    ),
    material=gs.materials.MPM.Elastic(E=20000.0, nu=0.3),
)

camera = scene.add_camera(
    res=(320, 240),
    pos=(1.2, -1.2, 1.0),
    lookat=(0.0, 0.0, 0.15),
    fov=40,
)

scene.build()

frames = []
num_steps = 400
render_every = 5

for i in range(num_steps):
    scene.step()
    if i % render_every == 0:
        rgb, _, _, _ = camera.render()
        frames.append(rgb)

        state = cube.get_particles_pos().cpu().numpy()
        z_values = state[:, 2]
        xy_spread = state[:, :2].max(axis=0) - state[:, :2].min(axis=0)
        print(
            f"step {i:4d} | min z = {z_values.min():.3f} | "
            f"max z = {z_values.max():.3f} | "
            f"xy footprint = {xy_spread[0]:.3f} x {xy_spread[1]:.3f}"
        )

import imageio
imageio.mimsave("soft_body_cube.gif", frames, duration=0.04)
print("\nSaved soft_body_cube.gif")
print(
    "\nIf this behaved like the Rigid Body sphere, the xy footprint "
    "would stay ~0.2 x 0.2 forever. A squash/spread on impact would "
    "mean the Soft Body material is actually deforming."
)
