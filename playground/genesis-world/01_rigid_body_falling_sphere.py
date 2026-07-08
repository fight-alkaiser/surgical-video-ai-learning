import genesis as gs

# ----------------------------------------
# What this script demonstrates
#
# The simplest possible Genesis scene from the README quickstart:
# a Rigid Body (a sphere) falling under gravity onto a ground Plane.
# No viewer window (headless machine) -- instead we attach an
# offscreen Camera and save rendered frames to check that the
# physics actually ran, not just that the API calls didn't crash.
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

plane = scene.add_entity(gs.morphs.Plane())

sphere = scene.add_entity(
    gs.morphs.Sphere(
        pos=(0.0, 0.0, 1.0),
        radius=0.1,
    ),
)

camera = scene.add_camera(
    res=(320, 240),
    pos=(1.5, -1.5, 1.2),
    lookat=(0.0, 0.0, 0.3),
    fov=40,
)

scene.build()

# ----------------------------------------
# Step the simulation and record the sphere's height (z) at
# each step, plus a handful of rendered frames, so we can see
# it actually fall and settle on the plane rather than just
# trusting that scene.step() ran without error.
# ----------------------------------------

heights = []
frames = []
num_steps = 90

for i in range(num_steps):
    scene.step()

    pos = sphere.get_pos()
    heights.append(float(pos[2]))

    rgb, _, _, _ = camera.render()
    frames.append(rgb)

print("Step | Sphere height (z)")
for i in range(0, num_steps, 10):
    print(f"{i:4d} | {heights[i]:.4f}")

print(f"\nFinal height (last step): {heights[-1]:.4f}")
print(f"Radius: 0.1  ->  resting height should settle near 0.10")

import imageio
imageio.mimsave("falling_sphere.gif", frames, duration=0.05)
print("\nSaved falling_sphere.gif")
