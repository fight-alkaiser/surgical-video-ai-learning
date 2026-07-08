import genesis as gs

# ----------------------------------------
# What this script demonstrates
#
# Same falling-sphere scene as 01_rigid_body_falling_sphere.py,
# but changing one parameter -- gravity -- to check that the
# physics actually responds to it as expected, not just that a
# hardcoded animation plays. Earth gravity vs. Moon gravity
# (~1/6th) should make the sphere fall much more slowly and take
# longer to settle.
# ----------------------------------------

EARTH_GRAVITY = (0.0, 0.0, -9.81)
MOON_GRAVITY = (0.0, 0.0, -1.62)


def run(gravity, label, num_steps=150):

    gs.init(backend=gs.cpu)

    scene = gs.Scene(
        show_viewer=False,
        sim_options=gs.options.SimOptions(dt=0.01, gravity=gravity),
    )

    scene.add_entity(gs.morphs.Plane())
    sphere = scene.add_entity(
        gs.morphs.Sphere(pos=(0.0, 0.0, 1.0), radius=0.1),
    )

    scene.build()

    heights = []
    for i in range(num_steps):
        scene.step()
        heights.append(float(sphere.get_pos()[2]))

    print(f"\n--- {label} (gravity={gravity}) ---")
    for i in range(0, num_steps, 15):
        print(f"step {i:4d} | z = {heights[i]:.4f}")

    settle_step = next(
        (i for i, h in enumerate(heights) if abs(h - 0.1) < 0.001),
        None,
    )
    print(f"Settled (z≈0.10) at step: {settle_step}")

    gs.destroy()


run(EARTH_GRAVITY, "Earth gravity")
run(MOON_GRAVITY, "Moon gravity (~1/6)")
