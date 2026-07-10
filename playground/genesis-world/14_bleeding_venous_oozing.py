import genesis as gs

# ----------------------------------------
# What this script demonstrates
#
# The venous counterpart to 13_bleeding_arterial_spurt.py: instead of
# a pulsatile, high-speed jet, this uses the same Emitter API with a
# slow, constant, low-speed seep -- no pulsation, low enough speed
# that gravity dominates almost immediately and the liquid just wells
# up and dribbles down rather than arcing across the basin.
#
# Same emitter API, same basin, only the emission parameters (speed,
# direction, constant vs pulsed) differ -- deliberately, to isolate
# what "arterial vs venous" actually means physically: pressure
# (speed) and pulsatility, not a different fluid or a different solver.
# ----------------------------------------

gs.init(backend=gs.cpu)

scene = gs.Scene(
    show_viewer=False,
    sim_options=gs.options.SimOptions(dt=4e-3, substeps=10),
    sph_options=gs.options.SPHOptions(
        lower_bound=(-0.5, -0.5, 0.0),
        upper_bound=(0.5, 0.5, 0.8),
        particle_size=0.012,
    ),
    vis_options=gs.options.VisOptions(
        ambient_light=(0.6, 0.6, 0.6),
        background_color=(0.9, 0.9, 0.9),
    ),
)

scene.add_entity(gs.morphs.Plane())

wall_half_extent = 0.45
wall_height = 0.3
scene.add_entity(gs.morphs.Plane(pos=(wall_half_extent, 0, wall_height), normal=(-1, 0, 0)))
scene.add_entity(gs.morphs.Plane(pos=(-wall_half_extent, 0, wall_height), normal=(1, 0, 0)))
scene.add_entity(gs.morphs.Plane(pos=(0, wall_half_extent, wall_height), normal=(0, -1, 0)))
scene.add_entity(gs.morphs.Plane(pos=(0, -wall_half_extent, wall_height), normal=(0, 1, 0)))

blood = gs.materials.SPH.Liquid(rho=1060.0, mu=0.0035)
emitter = scene.add_emitter(
    material=blood,
    max_particles=20000,
    surface=gs.surfaces.Default(color=(0.55, 0.05, 0.08, 1.0), vis_mode="particle"),
)

camera = scene.add_camera(
    res=(320, 240),
    pos=(1.3, -0.9, 0.7),
    lookat=(0.0, 0.0, 0.15),
    fov=45,
)

scene.build()

NOZZLE_POS = (-0.1, 0.0, 0.18)
DIRECTION = (0.15, 0.0, -1.0)   # mostly straight down -- just welling up and dribbling
SPEED = 0.12                    # much slower than the arterial jet's 1.8-3.0 m/s

# At this low a speed, the emitter's default droplet_length (computed
# from speed * dt) works out smaller than one particle, so it would
# accumulate silently for ~25 steps before emitting anything -- the
# ooze would look like it's barely happening at all. Forcing a small
# fixed droplet_length keeps a slow, low *speed* (so it drips instead
# of arcing) while still emitting a visible trickle every step.
DROPLET_LENGTH = 0.012

frames = []
num_steps = 260

for i in range(num_steps):
    emitter.emit(
        droplet_shape="circle",
        droplet_size=0.02,
        droplet_length=DROPLET_LENGTH,
        pos=NOZZLE_POS,
        direction=DIRECTION,
        speed=SPEED,
    )
    scene.step()
    if i % 3 == 0:
        rgb, _, _, _ = camera.render()
        frames.append(rgb)

import imageio
imageio.mimsave("bleeding_venous_oozing.gif", frames, duration=0.045)
print("Saved bleeding_venous_oozing.gif")
