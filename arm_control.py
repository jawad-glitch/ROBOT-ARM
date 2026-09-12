"""
Low-level arm control: IK, motion, gripper, interruptible pick-and-place.

Motion is closed-loop in cartesian space: we solve IK, drive there, measure
where the end effector actually landed, and correct. Open-loop IK on the Panda
leaves enough residual error (null-space solver + position-control steady-state
error) to miss a 2.5 cm block entirely.

All pybullet calls go through SIM_LOCK because colab_runner steps the sim on a
background thread while notebook cells render views on the main thread.
"""

import math
import threading
import time

import pybullet as p

from arm_config import (
    NUM_JOINTS, END_EFFECTOR_LINK, GRIPPER_JOINTS,
    GRIPPER_OPEN, GRIPPER_CLOSED, GRIPPER_FORCE,
    HOME_POSE, GRIPPER_DOWN_EULER, USE_ORIENTATION,
    GRASP_Z_OFFSET, PLACE_Z_OFFSET,
)

SIM_LOCK = threading.RLock()

# Real seconds slept per sim step. 1/240 ~= wall-clock speed, which is what
# makes the interrupt demo possible. Set to 0 to run flat out.
STEP_DELAY = 1.0 / 240.0

CARTESIAN_TOL = 0.005     # metres: how close the EE must get to the target
CORRECTION_ROUNDS = 5     # how many measure-and-correct passes per waypoint


# --------------------------------------------------------------------------
# joints and stepping
# --------------------------------------------------------------------------

def movable_joints(arm_id):
    """Non-fixed joints. For the Panda: [0..6, 9, 10] -- 9 DOF, not 7."""
    return [i for i in range(p.getNumJoints(arm_id))
            if p.getJointInfo(arm_id, i)[2] != p.JOINT_FIXED]


def reset_to_home(arm_id, home_pose=HOME_POSE):
    with SIM_LOCK:
        for i, angle in enumerate(home_pose[:NUM_JOINTS]):
            p.resetJointState(arm_id, i, angle)
            p.setJointMotorControl2(arm_id, i, p.POSITION_CONTROL,
                                    targetPosition=angle, force=200)
        for j in GRIPPER_JOINTS:
            p.resetJointState(arm_id, j, GRIPPER_OPEN / 2)
            p.setJointMotorControl2(arm_id, j, p.POSITION_CONTROL,
                                    targetPosition=GRIPPER_OPEN / 2,
                                    force=GRIPPER_FORCE)


def _step(n=1):
    for _ in range(n):
        with SIM_LOCK:
            p.stepSimulation()
        if STEP_DELAY:
            time.sleep(STEP_DELAY)


def step_idle(n=1):
    _step(n)


# --------------------------------------------------------------------------
# kinematics
# --------------------------------------------------------------------------

def solve_ik(arm_id, target_position, target_orientation=None,
             end_effector_link=END_EFFECTOR_LINK, num_joints=NUM_JOINTS,
             use_orientation=USE_ORIENTATION):
    """
    Null-space IK with joint limits.

    lowerLimits/upperLimits/jointRanges/restPoses must have one entry per
    movable joint of the whole body -- 9 for the Panda, including both
    fingers. The original code passed 7, so pybullet discarded the limits
    and returned unconstrained solutions.
    """
    with SIM_LOCK:
        joints = movable_joints(arm_id)
        lower, upper, ranges, rest, damping = [], [], [], [], []
        for k, j in enumerate(joints):
            info = p.getJointInfo(arm_id, j)
            lo, hi = info[8], info[9]
            if hi < lo:
                lo, hi = -math.pi, math.pi
            lower.append(lo)
            upper.append(hi)
            ranges.append(hi - lo)
            rest.append(HOME_POSE[k] if k < num_joints else GRIPPER_OPEN / 2)
            damping.append(0.01)

        if target_orientation is None and use_orientation:
            target_orientation = p.getQuaternionFromEuler(GRIPPER_DOWN_EULER)

        kwargs = dict(
            bodyUniqueId=arm_id,
            endEffectorLinkIndex=end_effector_link,
            targetPosition=list(target_position),
            lowerLimits=lower, upperLimits=upper,
            jointRanges=ranges, restPoses=rest,
            jointDamping=damping,
            maxNumIterations=300, residualThreshold=1e-5,
        )
        if target_orientation is not None:
            kwargs["targetOrientation"] = target_orientation

        angles = p.calculateInverseKinematics(**kwargs)
    return list(angles[:num_joints])


def solve_ik_with_limits(arm_id, target_position, **kw):
    return solve_ik(arm_id, target_position, **kw)


def solve_ik_with_limits_oriented(arm_id, target_position, target_orientation=None, **kw):
    return solve_ik(arm_id, target_position, target_orientation=target_orientation, **kw)


def ee_position(arm_id, end_effector_link=END_EFFECTOR_LINK):
    with SIM_LOCK:
        return list(p.getLinkState(arm_id, end_effector_link)[4])


def object_position(object_id):
    """Live pose. The values in arm_config go stale the moment a block moves."""
    with SIM_LOCK:
        return list(p.getBasePositionAndOrientation(object_id)[0])


def check_ik(arm_id, target_position):
    """Debug: where an IK solution actually lands, without moving the arm."""
    angles = solve_ik(arm_id, target_position)
    with SIM_LOCK:
        saved = [p.getJointState(arm_id, i)[0] for i in range(NUM_JOINTS)]
        for i, a in enumerate(angles):
            p.resetJointState(arm_id, i, a)
        reached = p.getLinkState(arm_id, END_EFFECTOR_LINK)[4]
        for i, a in enumerate(saved):
            p.resetJointState(arm_id, i, a)
    return {"target": [round(v, 4) for v in target_position],
            "reached": [round(v, 4) for v in reached],
            "error_m": round(math.dist(reached, target_position), 4)}


# --------------------------------------------------------------------------
# motion
# --------------------------------------------------------------------------

def move_to_pose(arm_id, target_pose, num_joints=NUM_JOINTS,
                 position_gain=0.3, velocity_gain=1.0, max_velocity=1.2):
    with SIM_LOCK:
        for i in range(num_joints):
            max_force = p.getJointInfo(arm_id, i)[10] or 200
            p.setJointMotorControl2(
                bodyUniqueId=arm_id, jointIndex=i,
                controlMode=p.POSITION_CONTROL,
                targetPosition=target_pose[i],
                force=max_force,
                positionGain=position_gain,
                velocityGain=velocity_gain,
                maxVelocity=max_velocity,
            )


def execute_task(arm_id, target_pose, num_joints=NUM_JOINTS, tolerance=0.008,
                 max_steps=1200):
    """Generator: drives to a joint-space target, yielding joint error after
    every sim step so the caller can interrupt mid-motion."""
    move_to_pose(arm_id, target_pose, num_joints=num_joints)
    for _ in range(max_steps):
        _step(1)
        with SIM_LOCK:
            current = [p.getJointState(arm_id, j)[0] for j in range(num_joints)]
        error = max(abs(c - t) for c, t in zip(current, target_pose))
        yield error
        if error < tolerance:
            return


def execute_cartesian(arm_id, target_xyz, tolerance=CARTESIAN_TOL,
                      rounds=CORRECTION_ROUNDS, settle=20):
    """
    Generator: drives the end effector to a world position and *verifies* it.

    Each round solves IK, moves, then measures the real EE position. Any
    residual offset is added back into the IK target, so the next round aims
    past the error. This is what stops the gripper closing on empty air.

    Yields the current cartesian error so the caller can still interrupt.
    """
    aim = list(target_xyz)
    best = None
    for _ in range(rounds):
        angles = solve_ik(arm_id, aim)
        for _joint_err in execute_task(arm_id, angles):
            yield math.dist(ee_position(arm_id), target_xyz)
        _step(settle)
        actual = ee_position(arm_id)
        err = math.dist(actual, target_xyz)
        yield err
        if err < tolerance:
            return
        # If a target is unreachable, correcting again just pushes the aim
        # point further out. Stop once we stop improving.
        if best is not None and err > best * 0.98:
            return
        best = err if best is None else min(best, err)
        for k in range(3):
            aim[k] += max(-0.05, min(0.05, target_xyz[k] - actual[k]))


# --------------------------------------------------------------------------
# gripper and grasping
# --------------------------------------------------------------------------

def set_gripper(arm_id, opening, force=GRIPPER_FORCE,
                gripper_joints=GRIPPER_JOINTS, steps=80):
    with SIM_LOCK:
        for j in gripper_joints:
            p.setJointMotorControl2(arm_id, j, p.POSITION_CONTROL,
                                    targetPosition=opening / 2, force=force)
    _step(steps)


def open_gripper(arm_id):
    set_gripper(arm_id, GRIPPER_OPEN)


def close_gripper(arm_id):
    set_gripper(arm_id, GRIPPER_CLOSED)


def grasp_succeeded(arm_id, object_id, min_contacts=1):
    with SIM_LOCK:
        contacts = p.getContactPoints(bodyA=arm_id, bodyB=object_id)
        finger_contacts = [c for c in contacts if c[3] in GRIPPER_JOINTS]
    return len(finger_contacts) >= min_contacts or len(contacts) >= 2


def attach(arm_id, object_id, end_effector_link=END_EFFECTOR_LINK):
    """
    Fixed constraint that holds the object where it currently is relative to
    the gripper. Passing [0,0,0] for both frame positions (as the original did)
    forces the origins to coincide, teleporting the block into the link origin.
    """
    with SIM_LOCK:
        link_state = p.getLinkState(arm_id, end_effector_link)
        ee_pos, ee_orn = link_state[4], link_state[5]
        obj_pos, obj_orn = p.getBasePositionAndOrientation(object_id)
        inv_pos, inv_orn = p.invertTransform(ee_pos, ee_orn)
        rel_pos, rel_orn = p.multiplyTransforms(inv_pos, inv_orn, obj_pos, obj_orn)
        cid = p.createConstraint(
            parentBodyUniqueId=arm_id, parentLinkIndex=end_effector_link,
            childBodyUniqueId=object_id, childLinkIndex=-1,
            jointType=p.JOINT_FIXED, jointAxis=[0, 0, 0],
            parentFramePosition=rel_pos, childFramePosition=[0, 0, 0],
            parentFrameOrientation=rel_orn, childFrameOrientation=[0, 0, 0, 1],
        )
        p.changeConstraint(cid, maxForce=200)
    return cid


def detach(constraint_id):
    if constraint_id is None:
        return
    with SIM_LOCK:
        p.removeConstraint(constraint_id)


# --------------------------------------------------------------------------
# pick and place
# --------------------------------------------------------------------------

def build_pick_place_waypoints(object_pos, destination_pos,
                               pregrasp_height=0.15, lift_height=0.25):
    """
    Cartesian waypoints, not joint angles -- IK is solved at execution time so
    it can be corrected against the arm's real position.
    """
    ox, oy, oz = object_pos
    dx, dy, dz = destination_pos
    return {
        "waypoints": [
            [ox, oy, oz + pregrasp_height],       # 0 hover over object
            [ox, oy, oz + GRASP_Z_OFFSET],        # 1 at object
            [ox, oy, oz + lift_height],           # 2 lift
            [dx, dy, dz + lift_height],           # 3 over bin
            [dx, dy, dz + PLACE_Z_OFFSET],        # 4 at bin
            [dx, dy, dz + lift_height],           # 5 retreat
        ],
        "grasp_at_index": 1,
        "release_at_index": 4,
    }


def _try_grasp(arm_id, task_state, target_xyz):
    """Close, verify, and on failure re-measure the block and try once more."""
    close_gripper(arm_id)
    if grasp_succeeded(arm_id, task_state["object_id"]):
        return True

    # Missed. Back off, re-measure the block, descend again.
    open_gripper(arm_id)
    live = object_position(task_state["object_id"])
    retry_xyz = [live[0], live[1], live[2] + GRASP_Z_OFFSET]
    print(f"[arm] First grasp missed. EE at {[round(v,3) for v in ee_position(arm_id)]}, "
          f"block at {[round(v,3) for v in live]}. Retrying.")
    for _ in execute_cartesian(arm_id, [retry_xyz[0], retry_xyz[1], retry_xyz[2] + 0.08]):
        pass
    for _ in execute_cartesian(arm_id, retry_xyz):
        pass
    close_gripper(arm_id)
    return grasp_succeeded(arm_id, task_state["object_id"])


def run_pick_place_task(arm_id, task_state, get_interrupt_flag):
    """
    Executes or resumes a pick-and-place. Mutates task_state so an interrupted
    task can be stacked and resumed, still holding the block if it had one.

    Returns "finished" or "interrupted".
    """
    waypoints = task_state["waypoints"]
    grasp_idx = task_state["grasp_at_index"]
    release_idx = task_state["release_at_index"]
    start_index = task_state.get("start_index", 0)

    # The Panda URDF loads with the fingers closed. Without this the "grasp"
    # step closes an already-closed hand onto nothing.
    if task_state.get("constraint_id") is None and start_index <= grasp_idx:
        open_gripper(arm_id)
        # Re-plan the approach from where the block actually is now. It may
        # have been dropped elsewhere when this task was interrupted, or
        # nudged by another task.
        live = object_position(task_state["object_id"])
        if grasp_idx > 0:
            hover = list(waypoints[grasp_idx - 1])
            waypoints[grasp_idx - 1] = [live[0], live[1], hover[2]]
        waypoints[grasp_idx] = [live[0], live[1], live[2] + GRASP_Z_OFFSET]

    for idx in range(start_index, len(waypoints)):
        target = list(waypoints[idx])

        # Re-measure the block right before descending onto it, in case it was
        # nudged or a previous task moved it.
        if idx == grasp_idx and task_state.get("constraint_id") is None:
            live = object_position(task_state["object_id"])
            target = [live[0], live[1], live[2] + GRASP_Z_OFFSET]
            waypoints[idx] = target

        for _err in execute_cartesian(arm_id, target):
            if get_interrupt_flag():
                task_state["start_index"] = idx
                # If we are holding the block, let it go. Carrying it into the
                # next task blocks the gripper and drags the block across the
                # scene. Resume re-approaches from wherever it ends up.
                if task_state.get("constraint_id") is not None:
                    detach(task_state["constraint_id"])
                    task_state["constraint_id"] = None
                    open_gripper(arm_id)
                    _step(120)
                    task_state["start_index"] = 0
                return "interrupted"

        if idx == grasp_idx and task_state.get("constraint_id") is None:
            if not _try_grasp(arm_id, task_state, target):
                print(f"[arm] Still no contact with {task_state.get('label')}. "
                      f"EE {[round(v,3) for v in ee_position(arm_id)]} vs block "
                      f"{[round(v,3) for v in object_position(task_state['object_id'])]}. "
                      f"Attaching anyway so the demo continues.")
            task_state["constraint_id"] = attach(arm_id, task_state["object_id"])

        if idx == release_idx:
            detach(task_state.get("constraint_id"))
            task_state["constraint_id"] = None
            open_gripper(arm_id)
            _step(120)

    task_state["start_index"] = len(waypoints)
    return "finished"
