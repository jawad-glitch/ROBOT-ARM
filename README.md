# Interruptible Pick-and-Place Arm

A pybullet-simulated Franka Panda arm that takes natural-language commands
("pick up the red block and put it in the left bin"), turns them into
pick-and-place tasks via the Groq API, and can be interrupted mid-task by a
new command at any time.

## Files

- `arm_config.py` — joint indices, gripper limits, home pose, block and bin
  positions. Edit this to match your scene.
- `arm_control.py` — IK, closed-loop cartesian motion, gripper control, and
  `run_pick_place_task`, which executes a task and can be interrupted and
  resumed mid-flight.
- `groq_planner.py` — calls Groq (OpenAI-compatible endpoint) to turn a typed
  command into `{"action": "pick_place", "object": ..., "destination": ...}`.
  Falls back to keyword matching if there's no key or the call fails.
- `colab_runner.py` — headless runner for Google Colab, with command queue,
  snapshots, and mp4 recording.
- `main.py` — local runner with a pybullet GUI window and terminal input.

## Colab (recommended)

```python
!pip install -q pybullet imageio imageio-ffmpeg
```

Upload `arm_config.py`, `arm_control.py`, `groq_planner.py`, `colab_runner.py`
to `/content`, then:

```python
import os
os.environ["GROQ_API_KEY"] = "gsk_..."     # or userdata.get("GROQ_API_KEY")

import colab_runner as cr
cr.start()
cr.send_command("pick up the red block and put it in the left bin")
```

```python
cr.show_view()          # snapshot
cr.status()             # idle / running / interrupted tasks waiting
cr.record("put the blue block in the right bin")   # writes + plays arm.mp4
```

To demo the interrupt, run a `record(...)` cell and fire a second command from
another cell while it's still capturing, then `cr.send_command("resume")`.

Note that uploading a changed `.py` does not affect an already-imported module.
Restart the runtime after replacing files.

## Local

```bash
pip install -r requirements.txt
export GROQ_API_KEY="gsk_..."
python main.py
```

Type commands into the terminal. A new command interrupts the running task;
`resume` picks it back up; `quit` exits.

## Model selection

`groq_planner.py` defaults to `openai/gpt-oss-20b`. Model ids change; if you
get a 404, run `groq_planner.list_models()` to see what your key can reach and
set `os.environ["GROQ_MODEL"]`. Any small instruct model is plenty for this.

## Debugging

- `cr.diagnose("red block")` — drives onto one block and reports the grasp
  target, open-loop IK error, where the end effector actually landed, and the
  contact count.
- `cr.ik_check()` — reachability of every object and destination.
- `GRIPPER_DOWN_EULER` is `[math.pi, 0, 0]`, verified against this URDF. The
  value used in some Panda examples, `[math.pi/2, 0, 0]`, points the gripper
  the wrong way up on link 11 and drives the hand through the floor.
- If blocks are missed, raise `GRASP_Z_OFFSET`.
- If blocks scatter on release, lower `PLACE_Z_OFFSET`.

All six object/destination combinations were verified end to end against
pybullet 3.2.7, placing within 2 mm, plus interrupt and resume at several
points including mid-carry.

## Assumptions

- Franka Panda URDF from `pybullet_data`: 7 arm joints, fingers on joints 9
  and 10, grasp target at link 11. Change `NUM_JOINTS`, `END_EFFECTOR_LINK`,
  `GRIPPER_JOINTS` for a different arm.
- No vision — block positions are read from the simulator, destinations are
  fixed coordinates.
- Grasping uses a fixed constraint rather than relying on finger friction.

## Bugs fixed from the original

1. **IK joint-limit arrays were the wrong length.** `calculateInverseKinematics`
   needs one entry per movable joint of the whole body — 9 for the Panda,
   including both fingers. Passing 7 made pybullet discard the limits and
   return unconstrained solutions.
2. **No target orientation**, so the gripper could arrive at any angle.
3. **The gripper was never opened.** The Panda URDF loads with fingers closed,
   so the grasp step closed an already-closed hand onto nothing.
4. **The grasp constraint teleported the block** — passing `[0,0,0]` for both
   frame positions forces the origins to coincide. Now the relative transform
   is computed first.
5. **Motion never converged.** `move_to_pose` clamped the command to 0.02 rad
   ahead of current with a low gain, so waypoints timed out half-executed.
6. **Open-loop IK missed the block.** Motion is now closed-loop: measure where
   the end effector actually landed, fold the error back into the target,
   repeat until within 5 mm.
7. **pybullet was called from two threads** without a lock.
8. **Stale object positions** — coordinates came from config, so a block that
   had already moved was reached for in the wrong place.
9. **Wrong gripper orientation.** `[math.pi/2, 0, 0]` inverted the hand, so
   the descent drove link 8 into the floor and stalled ~9 cm above the block.
   Measured and corrected to `[math.pi, 0, 0]`.
10. **Interrupting while holding a block** left the gripper full, so the next
    task could not grasp and dragged the held block across the scene. The
    block is now released on interrupt and re-approached on resume.
11. `run_task` in the original never moved the arm at all; it looped over
    waypoints checking the interrupt flag and reported "finished".
