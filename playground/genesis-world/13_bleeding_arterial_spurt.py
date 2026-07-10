import math
import numpy as np
import genesis as gs

# ----------------------------------------
# What this script demonstrates
#
# Day37 follow-up: the earlier fluid scripts (10-12) just dropped one
# static block of liquid. That's too simple to say anything about
# bleeding patterns. Genesis has a proper `Emitter` API
# (`scene.add_emitter` + `emitter.emit(...)` called every step) that
# injects a continuous stream of new particles with a controllable
# speed and direction -- much closer to how a vessel actually bleeds.
#
# This script models ARTERIAL bleeding: a forceful, PULSATILE jet.
# Instead of a constant speed, the emission speed is modulated by a
# sine wave over time (mimicking the cardiac pulse) so the jet surges
# and eases rather than flowing at one constant rate, and the base
# speed is high enough to arc across open space before landing.
#
# Compare with 14_bleeding_venous_oozing.py, which uses the same
# emitter API but a slow, constant, low-speed seep instead.
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

blood = gs.materials.SPH.Liquid(rho=1060.0, mu=0.0035)  # roughly blood's density/viscosity vs water
emitter = scene.add_emitter(
    material=blood,
    max_particles=20000,
    surface=gs.surfaces.Default(color=(0.65, 0.02, 0.05, 1.0), vis_mode="particle"),
)

camera = scene.add_camera(
    res=(320, 240),
    pos=(1.3, -0.9, 0.7),
    lookat=(0.0, 0.0, 0.15),
    fov=45,
)

scene.build()

NOZZLE_POS = (-0.3, 0.0, 0.2)
DIRECTION = (1.0, 0.0, 0.4)   # up and outward, like a vessel stump under pressure
BASE_SPEED = 1.8              # m/s -- high enough to arc across the basin
PULSE_AMPLITUDE = 1.2
PULSE_PERIOD_STEPS = 45       # one "heartbeat" every 45 steps

frames = []
num_steps = 260

for i in range(num_steps):
    pulse = max(0.0, math.sin(2 * math.pi * i / PULSE_PERIOD_STEPS))
    speed = BASE_SPEED + PULSE_AMPLITUDE * pulse
    emitter.emit(
        droplet_shape="circle",
        droplet_size=0.03,
        pos=NOZZLE_POS,
        direction=DIRECTION,
        speed=speed,
    )
    scene.step()
    if i % 3 == 0:
        rgb, _, _, _ = camera.render()
        frames.append(rgb)
    if i % 40 == 0:
        print(f"step {i:4d} | pulse speed: {speed:.2f} m/s")

import imageio
imageio.mimsave("bleeding_arterial_spurt.gif", frames, duration=0.045)
print("Saved bleeding_arterial_spurt.gif")
