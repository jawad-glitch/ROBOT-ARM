"""
Streamlit demo for the interruptible pick-and-place arm.

Runs the simulation synchronously and renders the result as an animation, so
there are no background threads to fight Streamlit's rerun model. The
interrupt is demonstrated by choosing the point at which a second command
arrives, which is deterministic and reproducible for a judge.

    streamlit run app.py
"""

import io
import math
import os

import numpy as np
import streamlit as st

st.set_page_config(page_title="Interruptible Robot Arm", page_icon="🦾",
                   layout="wide")

import pybullet as p
import pybullet_data

import arm_control as ac
from arm_control import (
    build_pick_place_waypoints, run_pick_place_task, reset_to_home,
    object_position,
)
from arm_config import (
    OBJECTS, DESTINATIONS, PREGRASP_HEIGHT, LIFT_HEIGHT,
)
from groq_planner import parse_command

BLOCK_COLORS = {
    "red block": [1, 0, 0, 1],
    "blue block": [0, 0, 1, 1],
    "green block": [0, 1, 0, 1],
}


# --------------------------------------------------------------------------
# simulation
# --------------------------------------------------------------------------

def setup_scene():
    if p.isConnected():
        p.resetSimulation()
    else:
        p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.8)
    p.setTimeStep(1.0 / 240.0)
    p.loadURDF("plane.urdf")

    arm = p.loadURDF("franka_panda/panda.urdf", basePosition=[0, 0, 0],
                     useFixedBase=True)

    ids = {}
    for name, pos in OBJECTS.items():
        b = p.loadURDF("cube_small.urdf", basePosition=pos, globalScaling=0.5)
        p.changeVisualShape(b, -1, rgbaColor=BLOCK_COLORS.get(name, [0.6, 0.6, 0.6, 1]))
        p.changeDynamics(b, -1, lateralFriction=1.2, spinningFriction=0.01)
        ids[name] = b
    for link in (9, 10):
        p.changeDynamics(arm, link, lateralFriction=1.5)

    for pos, col in ((DESTINATIONS["left bin"], [0.2, 0.8, 0.2, 0.6]),
                     (DESTINATIONS["right bin"], [0.8, 0.6, 0.2, 0.6])):
        vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.06, 0.06, 0.005],
                                  rgbaColor=col)
        p.createMultiBody(baseMass=0, baseVisualShapeIndex=vis,
                          basePosition=[pos[0], pos[1], 0.005])

    ac.STEP_DELAY = 0.0
    reset_to_home(arm)
    for _ in range(150):
        p.stepSimulation()
    return arm, ids


def frame(width=420, height=320):
    view = p.computeViewMatrixFromYawPitchRoll(
        cameraTargetPosition=[0.35, 0, 0.1], distance=1.3,
        yaw=50, pitch=-35, roll=0, upAxisIndex=2)
    proj = p.computeProjectionMatrixFOV(fov=60, aspect=width / height,
                                        nearVal=0.1, farVal=3.0)
    _, _, rgb, _, _ = p.getCameraImage(width, height, view, proj,
                                       renderer=p.ER_TINY_RENDERER)
    return np.reshape(np.array(rgb, dtype=np.uint8), (height, width, 4))[:, :, :3]


def make_task(arm, ids, obj, dest):
    plan = build_pick_place_waypoints(
        object_position(ids[obj]), DESTINATIONS[dest],
        pregrasp_height=PREGRASP_HEIGHT, lift_height=LIFT_HEIGHT)
    return {
        "object_id": ids[obj], "waypoints": plan["waypoints"],
        "grasp_at_index": plan["grasp_at_index"],
        "release_at_index": plan["release_at_index"],
        "constraint_id": None, "start_index": 0,
        "label": f"{obj} -> {dest}",
    }


def run_capturing(arm, task, interrupt_after=None, every=25, frames=None):
    """Runs a task, capturing a frame every `every` sim steps. If
    interrupt_after is an int, the interrupt flag turns true after that many
    checks, which is how the demo triggers a mid-task switch."""
    if frames is None:
        frames = []
    calls = {"n": 0}

    def flag():
        calls["n"] += 1
        if calls["n"] % every == 0:
            frames.append(frame())
        if interrupt_after is None:
            return False
        return calls["n"] > interrupt_after

    result = run_pick_place_task(arm, task, flag)
    frames.append(frame())
    return result, frames


def to_gif(frames, fps=12):
    import imageio
    buf = io.BytesIO()
    imageio.mimsave(buf, frames, format="GIF", fps=fps, loop=0)
    return buf.getvalue()


# --------------------------------------------------------------------------
# UI
# --------------------------------------------------------------------------

st.title("🦾 Interruptible Pick-and-Place Arm")
st.caption("Natural language in, robot motion out — and it can change its "
           "mind mid-task without restarting.")

with st.sidebar:
    st.header("Setup")
    key = st.text_input("Groq API key", type="password",
                        help="Leave blank to use the offline keyword parser.")
    if key:
        os.environ["GROQ_API_KEY"] = key
    st.caption(f"Model: `{os.environ.get('GROQ_MODEL', 'openai/gpt-oss-20b')}`")
    st.divider()
    st.write("**Objects**: " + ", ".join(OBJECTS))
    st.write("**Destinations**: left bin, right bin")

tab1, tab2 = st.tabs(["Single command", "Interrupt demo"])

with tab1:
    cmd = st.text_input("Command",
                        "the red one needs to go to the bin on the left",
                        key="single")
    if st.button("Run", type="primary", key="run1"):
        with st.spinner("Parsing..."):
            task = parse_command(cmd)
        st.json(task)
        if task.get("action") != "pick_place":
            st.error("Not a pick-and-place command.")
        else:
            with st.spinner("Simulating..."):
                arm, ids = setup_scene()
                ts = make_task(arm, ids, task["object"], task["destination"])
                result, frames = run_capturing(arm, ts)
                pos = object_position(ids[task["object"]])
                err = math.dist(pos[:2], DESTINATIONS[task["destination"]][:2])
            st.image(to_gif(frames), caption=f"{ts['label']} — {result}")
            st.metric("Placement error", f"{err * 100:.1f} cm")

with tab2:
    st.write("A first command starts, a second arrives mid-task. The arm "
             "abandons the first job, completes the new one, then resumes "
             "the original from wherever the block ended up.")
    c1, c2 = st.columns(2)
    first = c1.text_input("First command",
                          "pick up the red block and put it in the left bin")
    second = c2.text_input("Interrupting command",
                           "actually, put the blue block in the right bin")
    when = st.slider("Interrupt after (flag checks)", 100, 800, 450, 50)

    if st.button("Run interrupt demo", type="primary", key="run2"):
        arm, ids = setup_scene()
        frames = []
        with st.spinner("Task A..."):
            t1 = parse_command(first)
            ts1 = make_task(arm, ids, t1["object"], t1["destination"])
            r1, frames = run_capturing(arm, ts1, interrupt_after=when,
                                       frames=frames)
        st.write(f"**Task A** `{ts1['label']}` → **{r1}** "
                 f"at waypoint {ts1['start_index']}")

        with st.spinner("Task B..."):
            t2 = parse_command(second)
            ts2 = make_task(arm, ids, t2["object"], t2["destination"])
            r2, frames = run_capturing(arm, ts2, frames=frames)
        st.write(f"**Task B** `{ts2['label']}` → **{r2}**")

        if r1 == "interrupted":
            with st.spinner("Resuming task A..."):
                r3, frames = run_capturing(arm, ts1, frames=frames)
            st.write(f"**Task A resumed** → **{r3}**")

        st.image(to_gif(frames), caption="One continuous run")
        cols = st.columns(3)
        for col, (name, oid) in zip(cols, ids.items()):
            pos = object_position(oid)
            near = min(DESTINATIONS, key=lambda d: math.dist(pos[:2], DESTINATIONS[d][:2]))
            d = math.dist(pos[:2], DESTINATIONS[near][:2])
            col.metric(name, near if d < 0.08 else "table", f"{d * 100:.1f} cm")

st.divider()
st.caption("Franka Panda in PyBullet · command parsing via Groq · "
           "closed-loop cartesian control with interrupt and resume")
