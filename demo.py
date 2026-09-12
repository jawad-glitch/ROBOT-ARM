"""
Video recording and scripted demos for the Colab notebook.

    import colab_runner as cr
    import demo

    demo.record(cr, "put the red block in the left bin")
    demo.interrupt_demo(cr)

Both write an mp4 and display it inline.
"""

import base64
import threading
import time

import numpy as np
import pybullet as p
from IPython.display import HTML, display

import arm_control
from arm_control import SIM_LOCK, object_position
from arm_config import DESTINATIONS


def grab_frame(width=400, height=300, yaw=50, pitch=-35, distance=1.3):
    """One RGB frame of the current scene."""
    with SIM_LOCK:
        view = p.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=[0.35, 0, 0.1], distance=distance,
            yaw=yaw, pitch=pitch, roll=0, upAxisIndex=2)
        proj = p.computeProjectionMatrixFOV(
            fov=60, aspect=width / height, nearVal=0.1, farVal=3.0)
        _, _, rgb, _, _ = p.getCameraImage(
            width, height, view, proj, renderer=p.ER_TINY_RENDERER)
    return np.reshape(np.array(rgb, dtype=np.uint8), (height, width, 4))[:, :, :3]


def play(path):
    """Embed an mp4 in the notebook output."""
    data = base64.b64encode(open(path, "rb").read()).decode()
    display(HTML(f'<video width="480" controls loop autoplay muted>'
                 f'<source src="data:video/mp4;base64,{data}" type="video/mp4">'
                 f'</video>'))


def _film(path, duration, fps, size, stop_when_idle, cr):
    import imageio
    w, h = size
    writer = imageio.get_writer(path, fps=fps, macro_block_size=None,
                                codec="libx264", quality=8)
    start, n = time.time(), 0
    try:
        while time.time() - start < duration:
            writer.append_data(grab_frame(w, h))
            n += 1
            if stop_when_idle and cr._status == "idle" and time.time() - start > 2:
                break
            time.sleep(1.0 / fps)
        for _ in range(int(fps * 1.5)):          # tail so the ending is visible
            writer.append_data(grab_frame(w, h))
            n += 1
            time.sleep(1.0 / fps)
    finally:
        writer.close()
    print(f"{n} frames, {n / fps:.1f}s -> {path}")
    play(path)
    return path


def record(cr, command, path="arm.mp4", fps=10, size=(400, 300), max_seconds=120):
    """Sends one command and films until it finishes."""
    arm_control.STEP_DELAY = 1.0 / 240.0        # real-time, or there is nothing to film
    cr.send_command(command)
    t0 = time.time()
    while cr._status == "idle" and time.time() - t0 < 10:
        time.sleep(0.02)
    return _film(path, max_seconds, fps, size, True, cr)


def interrupt_demo(cr,
                   first="the red one needs to go to the bin on the left",
                   second="actually, put the blue block in the right bin instead",
                   path="interrupt_demo.mp4", fps=10, size=(400, 300),
                   duration=130, lift_threshold=0.08):
    """
    The full story in one continuous take: a task starts, a second command cuts
    in the moment the block is lifted, the new task completes, then the
    original resumes and finishes.

    Resets the scene first so it is reproducible.
    """
    arm_control.STEP_DELAY = 1.0 / 240.0

    cr._task_stack.clear()
    while not cr._command_queue.empty():
        cr._command_queue.get()
    cr.start()
    time.sleep(1.0)

    def fire_interrupt():
        red = cr._object_ids["red block"]
        t0 = time.time()
        while time.time() - t0 < 60:
            if object_position(red)[2] > lift_threshold:
                cr.send_command(second)
                return
            time.sleep(0.05)

    def fire_resume():
        t0 = time.time()
        while time.time() - t0 < 60:            # wait for the new task to start
            if "blue" in cr._status:
                break
            time.sleep(0.1)
        while time.time() - t0 < 150:           # wait for it to finish
            if cr._status == "idle":
                break
            time.sleep(0.1)
        time.sleep(2.0)
        cr.send_command("resume")

    threading.Thread(target=fire_interrupt, daemon=True).start()
    threading.Thread(target=fire_resume, daemon=True).start()

    cr.send_command(first)
    out = _film(path, duration, fps, size, False, cr)
    report(cr)
    return out


def report(cr):
    """Where every block actually ended up — the proof the demo worked."""
    import math
    print(f"status: {cr._status}")
    for name, oid in cr._object_ids.items():
        pos = object_position(oid)
        near = min(DESTINATIONS, key=lambda d: math.dist(pos[:2], DESTINATIONS[d][:2]))
        dist = math.dist(pos[:2], DESTINATIONS[near][:2])
        where = near if dist < 0.08 else "on the table"
        print(f"  {name:12s} {[round(v, 3) for v in pos]}  ->  {where} ({dist:.3f} m)")
