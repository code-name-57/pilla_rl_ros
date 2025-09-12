import numpy as np
import torch

import genesis as gs
from genesis.sensors.imu import IMUOptions

gs.init()
GRAVITY = -10.0
DT = 1e-2
BIAS = (0.1, 0.2, 0.3)

scene = gs.Scene(
    sim_options=gs.options.SimOptions(
        dt=DT,
        substeps=1,
        gravity=(0.0, 0.0, GRAVITY),
    ),
    profiling_options=gs.options.ProfilingOptions(show_FPS=False),
    viewer_options=gs.options.ViewerOptions(
        max_FPS=int(0.5 / 0.02),
        camera_pos=(2.0, 0.0, 2.5),
        camera_lookat=(0.0, 0.0, 0.5),
        camera_fov=40,
    ),
    vis_options=gs.options.VisOptions(rendered_envs_idx=list(range(1))),
    rigid_options=gs.options.RigidOptions(
        dt=0.02,
        constraint_solver=gs.constraint_solver.Newton,
        enable_collision=True,
        enable_joint_limit=True,
    ),
    show_viewer=True,
)

scene.add_entity(gs.morphs.Plane())

box = scene.add_entity(
    morph=gs.morphs.Box(
        size=(0.1, 0.1, 0.1),
        pos=(0.0, 0.0, 0.2),
    ),
)
imu_biased = scene.add_sensor(IMUOptions(entity_idx=box.idx, accelerometer_bias=BIAS, gyroscope_bias=BIAS))
imu_delayed = scene.add_sensor(IMUOptions(entity_idx=box.idx, read_delay=DT * 2))

scene.build()

# box is in freefall
for _ in range(10):
    scene.step()
    print(imu_biased.read())
    print(imu_delayed.read())

    # IMU should calculate "classical linear acceleration" using the local frame without accounting for gravity
    # acc_classical_lin_z = - theta_dot ** 2 - cos(theta) * g
   