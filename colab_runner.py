"""
Colab front-end. Headless pybullet on a background thread, commands pushed in
from notebook cells.

    import colab_runner as cr
    cr.start()
    cr.send_command("pick up the red block and put it in the left bin")
    cr.show_view()
    cr.send_command("pick up the blue block and put it in the right bin")  # interrupts
    cr.send_command("resume")
"""

import queue
import threading
import time

import numpy as np
import pybullet as p
import pybullet_data
import matplotlib.pyplot as plt

import arm_control
from arm_control import (
    SIM_LOCK, build_pick_place_waypoints, run_pick_place_task,
    reset_to_home, object_position, ee_position, detach, check_ik,
    execute_cartesian, open_gripper, close_gripper, grasp_succeeded,
)
from arm_config import (
    OBJECTS, DESTINATIONS, PREGRASP_HEIGHT, LIFT_HEIGHT, GRASP_Z_OFFSET,
)

BLOCK_COLORS = {
    "red block": [1, 0, 0, 1],
    "blue block": [0, 0, 1, 1],
    "green block": [0, 1, 0, 1],
}

_arm_id = None
_object_ids = {}
_command_queue = queue.Queue()
_task_stack = []
_worker_thread = None
_status = "not started"
_started = False


def _add_bin_marker(position, color):
    vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.06, 0.06, 0.005],
                              rgbaColor=color)
    p.createMultiBody(baseMass=0, baseVisualShapeIndex=vis,
                      basePosition=[position[0], position[1], 0.005])


def _setup_sim():
    global _arm_id, _object_ids
    with SIM_LOCK:
        if p.isConnected():
            p.resetSimulation()
        else:
            p.connect(p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.8)
        p.setTimeStep(1.0 / 240.0)
        p.loadURDF("plane.urdf")

        _arm_id = p.loadURDF("franka_panda/panda.urdf",
                             basePosition=[0, 0, 0], useFixedBase=True)

        _object_ids = {}
        for name, pos in OBJECTS.items():
            body_id = p.loadURDF("cube_small.urdf", basePosition=pos,
                                 globalScaling=0.5)
            p.changeVisualShape(body_id, -1,
                                rgbaColor=BLOCK_COLORS.get(name, [0.6, 0.6, 0.6, 1]))
            p.changeDynamics(body_id, -1, lateralFriction=1.2, spinningFriction=0.01)
            _object_ids[name] = body_id

        for link in (9, 10):
            p.changeDynamics(_arm_id, link, lateralFriction=1.5)

        _add_bin_marker(DESTINATIONS["left bin"], [0.2, 0.8, 0.2, 0.6])
        _add_bin_marker(DESTINATIONS["right bin"], [0.8, 0.6, 0.2, 0.6])

    reset_to_home(_arm_id)
    for _ in range(150):
        with SIM_LOCK:
            p.stepSimulation()


def _get_interrupt_flag():
    return not _command_queue.empty()


def _build_task_state(task):
    object_name = task["object"]
    destination_name = task["destination"]
    object_id = _object_ids[object_name]

    plan = build_pick_place_waypoints(
        object_position(object_id),
        DESTINATIONS[destination_name],
        pregrasp_height=PREGRASP_HEIGHT, lift_height=LIFT_HEIGHT,
    )
    return {
        "object_id": object_id,
        "waypoints": plan["waypoints"],
        "grasp_at_index": plan["grasp_at_index"],
        "release_at_index": plan["release_at_index"],
        "constraint_id": None,
        "start_index": 0,
        "label": f"{object_name} -> {destination_name}",
    }


def _worker_loop():
    global _status
    from groq_planner import parse_command

    _status = "idle"
    while True:
        try:
            command_text = _command_queue.get(timeout=0.01)
        except queue.Empty:
            arm_control.step_idle(1)
            continue

        if command_text.lower() == "resume":
            if not _task_stack:
                print("[worker] Nothing to resume.")
                continue
            task_state = _task_stack.pop()
            print(f"[worker] Resuming: {task_state['label']}")
        else:
            task = parse_command(command_text)
            if task.get("action") != "pick_place":
                print(f"[worker] Couldn't turn that into a task: {task}")
                continue
            task_state = _build_task_state(task)
            print(f"[worker] New task: {task_state['label']}")

        _status = f"running: {task_state['label']}"
        try:
            result = run_pick_place_task(_arm_id, task_state, _get_interrupt_flag)
        except Exception as exc:
            print(f"[worker] Task failed: {exc!r}")
            _status = "idle"
            continue

        if result == "interrupted":
            _task_stack.append(task_state)
            print(f"[worker] Interrupted: {task_state['label']} (send 'resume')")
        else:
            print(f"[worker] Finished: {task_state['label']}")
        _status = "idle"


# --------------------------------------------------------------------------
# notebook API
# --------------------------------------------------------------------------

def start(realtime=True):
    """Call once per session. Safe to call again -- it resets the scene."""
    global _worker_thread, _started
    arm_control.STEP_DELAY = (1.0 / 240.0) if realtime else 0.0

    _setup_sim()
    if not _started:
        _worker_thread = threading.Thread(target=_worker_loop, daemon=True)
        _worker_thread.start()
        _started = True

    print("Arm ready.")
    print("  objects:     ", list(OBJECTS.keys()))
    print("  destinations:", list(DESTINATIONS.keys()))


def send_command(text):
    _command_queue.put(text)


def status():
    print(_status)
    if _task_stack:
        print(f"{len(_task_stack)} interrupted task(s): " +
              ", ".join(t["label"] for t in _task_stack))


def reset():
    for t in _task_stack:
        detach(t.get("constraint_id"))
    _task_stack.clear()
    while not _command_queue.empty():
        _command_queue.get()
    _setup_sim()
    print("Scene reset.")


def show_view(width=480, height=360, yaw=50, pitch=-35, distance=1.3):
    with SIM_LOCK:
        view = p.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=[0.35, 0, 0.1], distance=distance,
            yaw=yaw, pitch=pitch, roll=0, upAxisIndex=2)
        proj = p.computeProjectionMatrixFOV(
            fov=60, aspect=width / height, nearVal=0.1, farVal=3.0)
        _, _, rgb, _, _ = p.getCameraImage(
            width, height, view, proj, renderer=p.ER_TINY_RENDERER)
    rgb = np.reshape(np.array(rgb, dtype=np.uint8), (height, width, 4))[:, :, :3]
    plt.figure(figsize=(6, 4.5))
    plt.imshow(rgb)
    plt.axis("off")
    plt.show()


def watch(seconds=8, interval=2):
    end = time.time() + seconds
    while time.time() < end:
        show_view()
        time.sleep(interval)


def ik_check():
    """Reachability check. error_m should be well under 0.01 everywhere."""
    for name, oid in _object_ids.items():
        print(f"{name:12s}", check_ik(_arm_id, object_position(oid)))
    for name, pos in DESTINATIONS.items():
        print(f"{name:12s}", check_ik(_arm_id, pos))


def diagnose(object_name="red block"):
    """
    Drives the arm onto one block and reports exactly what happened. Run this
    when you get a 'no contact' warning -- it tells you whether the problem is
    IK accuracy, the z offset, or the gripper orientation.

    Only run it when the arm is idle.
    """
    oid = _object_ids[object_name]
    block = object_position(oid)
    target = [block[0], block[1], block[2] + GRASP_Z_OFFSET]

    print(f"block at        {[round(v, 4) for v in block]}")
    print(f"grasp target    {[round(v, 4) for v in target]}")
    print(f"open-loop IK    {check_ik(_arm_id, target)}")

    open_gripper(_arm_id)
    for _ in execute_cartesian(_arm_id, [target[0], target[1], target[2] + 0.15]):
        pass
    for _ in execute_cartesian(_arm_id, target):
        pass

    reached = ee_position(_arm_id)
    print(f"EE actually at  {[round(v, 4) for v in reached]}")
    print(f"closed-loop err {round(sum((a - b) ** 2 for a, b in zip(reached, target)) ** 0.5, 4)} m")

    close_gripper(_arm_id)
    with SIM_LOCK:
        contacts = p.getContactPoints(bodyA=_arm_id, bodyB=oid)
    print(f"contacts        {len(contacts)} (links {[c[3] for c in contacts]})")
    print(f"grasp_succeeded {grasp_succeeded(_arm_id, oid)}")
    show_view()
