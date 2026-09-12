"""
Central config. Edit OBJECTS / DESTINATIONS to match your scene.
"""

import math

# --- Franka Panda (pybullet_data/franka_panda/panda.urdf) -------------------
# The URDF has 12 joints. 0-6 are the arm. 7 and 8 are fixed. 9 and 10 are the
# two gripper fingers. 11 is "panda_grasptarget", the frame between the
# fingertips -- that is what you solve IK for.
NUM_JOINTS = 7
END_EFFECTOR_LINK = 11
GRIPPER_JOINTS = (9, 10)

# Each finger travels 0 to 0.04 m, so total opening is 0 to 0.08 m.
GRIPPER_OPEN = 0.08
GRIPPER_CLOSED = 0.0
GRIPPER_FORCE = 40

# A sane starting configuration. All-zeros puts the Panda in a straight-up
# near-singular pose, which makes the first IK solve return garbage.
HOME_POSE = [0.0, -0.4, 0.0, -2.0, 0.0, 1.6, 0.785]

# Orientation the gripper holds while picking (fingers pointing down). This is
# what pybullet's own Panda demo uses for link 11. If the gripper comes in
# sideways in your scene, try [math.pi, 0, 0] instead.
GRIPPER_DOWN_EULER = [math.pi, 0.0, 0.0]
USE_ORIENTATION = True   # set False to fall back to position-only IK

# --- Scene ------------------------------------------------------------------
# Blocks are cube_small.urdf at globalScaling 0.5, so ~0.025 m cubes resting
# with their centre near z = 0.0125.
OBJECTS = {
    "red block": [0.5, 0.0, 0.02],
    "blue block": [0.5, 0.2, 0.02],
    "green block": [0.5, -0.2, 0.02],
}

# From the arm's point of view (+x forward), +y is its left.
DESTINATIONS = {
    "left bin": [0.35, 0.35, 0.02],
    "right bin": [0.35, -0.35, 0.02],
    "bin": [0.35, 0.35, 0.02],   # fallback
}

PREGRASP_HEIGHT = 0.15   # hover height above an object before descending
LIFT_HEIGHT = 0.25       # how high to carry the object
GRASP_Z_OFFSET = 0.005   # nudge the grasp frame up so fingers clear the floor
PLACE_Z_OFFSET = 0.04    # release slightly above the bin so blocks drop in
