"""One control owner, bounded archival I/O, manual episode readiness."""
from __future__ import annotations
import time
import hashlib
from pathlib import Path

from .archive import EpisodeWriter
from .core import Synchronizer
from .controls import EpisodeControls, State


def metadata_for(robot, schema, task, source_name):
    if not task.strip():
        raise ValueError("A nonempty task instruction is required")
    acquisition = {}
    if not robot.synthetic:
        cfg = robot.cfg
        acquisition = {"operator_calibration_sha256": hashlib.sha256(Path(cfg["operator_calibration"]).read_bytes()).hexdigest(),
                       "arm_mapping": cfg.get("arm_mapping", {}), "camera_settings": cfg["cameras"],
                       "safety_profile": {k: cfg[k] for k in ("lower_rad", "upper_rad", "max_velocity_rad_s",
                                                              "max_tracking_error_rad", "episode_start_rad")}}
    return {"contract": {**schema.contract(), **robot.contract_extra}, "acquisition_profile": acquisition,
            "task": task, "synthetic": robot.synthetic, "source": source_name,
            "clock": "PC perf_counter_ns (monotonic); camera receipt is NOT exposure time",
            "alignment": "observe first; then transmit accepted joint target on the same tick",
            "action_semantics": "post-software-limit target, successful driver write return; NOT motor arrival acknowledgement",
            "raw_scope": "selected decision-rate VR samples and camera frames; not every transport packet"}


def warmup(robot, source, schema, controls=None, require_start=True, timeout_s=15):
    observation = robot.observe()
    if require_start:
        robot.check_start(observation.state)
    source.reset(observation.state)
    robot.safety.reset(observation.state)
    if hasattr(source, "prepare"):
        source.prepare(observation)  # Cold model warmup with no actuation.
    sync = Synchronizer(schema)
    deadline = time.monotonic() + timeout_s
    consecutive = 0
    while time.monotonic() < deadline:
        if controls and controls.poll() == "stop":
            raise KeyboardInterrupt("Stopped during warmup")
        try:
            observation = robot.observe()
            command = source.latest(observation)
            sync.validate(observation, command, time.perf_counter_ns())
            consecutive += 1
            if consecutive >= 5:
                if hasattr(source, "on_ready"):
                    source.on_ready()  # Discard policy chunks consumed by warmup.
                return
        except LookupError:
            consecutive = 0
        time.sleep(1 / robot.fps)
    raise TimeoutError("Fresh VR/three-camera warmup timed out; no motion was sent")


def tick(robot, source, schema, sync, writer=None, index=0, max_dispatch_s=None):
    if writer:
        writer.check()
    observation = robot.observe()
    proposed = source.latest(observation)
    decision_ns = time.perf_counter_ns()
    sync.validate(observation, proposed, decision_ns)
    accepted = robot.safety.apply(proposed.position, observation.state)
    started = time.perf_counter_ns()
    transmitted = schema.vector(robot.dispatch(accepted))
    finished = time.perf_counter_ns()
    if not (abs(transmitted - accepted) < 1e-6).all():
        raise RuntimeError("Backend changed the safety-filtered action; reject and fix the command contract")
    robot.safety.commit(transmitted)
    if max_dispatch_s is not None and (finished - started) / 1e9 > max_dispatch_s:
        raise RuntimeError("Serial dispatch exceeded timing budget; reject episode and lower the verified control rate")
    if writer:
        writer.append(observation, proposed, transmitted, decision_ns, started, finished, index)
    return transmitted


def record_mock(robot, source, schema, root, task, episodes=3, frame_count=30, realtime=True):
    if not robot.synthetic or episodes < 1 or frame_count < 1:
        raise ValueError("Automatic recording is only permitted with mock hardware and positive counts")
    paths = []
    for _ in range(episodes):
        source.reset(robot.observe().state)
        robot.safety.reset(robot.observe().state)
        sync = Synchronizer(schema)
        writer = EpisodeWriter(root, metadata_for(robot, schema, task, "mock"))
        try:
            deadline = time.monotonic()
            for index in range(frame_count):
                tick(robot, source, schema, sync, writer, index)
                if not realtime:
                    # Fast mode is mock-only: let disk catch up instead of
                    # deliberately overflowing the bounded real-time queue.
                    writer.queue.join()
                    writer.check()
                deadline += 1 / robot.fps
                if realtime:
                    time.sleep(max(0, deadline - time.monotonic()))
            paths.append(str(writer.finish()))
        except BaseException as exc:
            writer.finish("rejected", str(exc))
            raise
    return paths


def interactive_session(robot, source, schema, root, task, controls, execution=False):
    """Preview and inference never produce demonstration training episodes."""
    machine = EpisodeControls()
    writer = None
    sync = Synchronizer(schema)
    deadline = time.monotonic()
    frame_index = 0
    print("READY: right/n=start, left/r=preview toggle; recording: right/n=save, left/r=reject.")
    print("Center double-tap/q stops the session. Pedal is NOT an emergency stop.")
    try:
        while machine.state != State.STOPPED:
            event = controls.poll()
            transition = machine.handle(event)
            if transition == "stop":
                break
            if transition in ("save", "reject", "hold"):
                if writer:
                    path = writer.finish("accepted" if transition == "save" else "rejected", "operator decision")
                    print(f"{transition.upper()}: {path}")
                    writer = None
                source.reset(robot.observe().state)
                controls.clear_pending()
                print("RESETTING: outputs held. Left/r enables supervised preview; right/n confirms the start pose.")
            elif transition in ("start", "preview"):
                if execution and transition == "preview":
                    machine.state = State.RESETTING
                    print("Policy execution has no VR preview. Prepare the start pose in a separate teleop session.")
                    continue
                print("Checking fresh inputs; no new target is sent during warmup.")
                warmup(robot, source, schema, controls, require_start=transition == "start")
                sync = Synchronizer(schema)
                frame_index = 0
                if transition == "start" and not execution:
                    writer = EpisodeWriter(root, metadata_for(robot, schema, task, "metaquest"))
                deadline = time.monotonic()
                print("RUNNING POLICY" if execution else machine.state.value.upper())
            if machine.state in (State.RECORDING, State.PREVIEW):
                if time.monotonic() - deadline > .04:
                    raise RuntimeError("Control loop deadline missed; no catch-up command burst is allowed")
                tick(robot, source, schema, sync, writer, frame_index, max_dispatch_s=1 / robot.fps)
                frame_index += 1
                deadline += 1 / robot.fps
                time.sleep(max(0, deadline - time.monotonic()))
            else:
                time.sleep(.02)
    except BaseException as exc:
        if writer:
            path = writer.finish("rejected", str(exc))
            writer = None
            print(f"REJECTED (fault): {path}")
        raise
    finally:
        if writer:
            print(f"INCOMPLETE (session ended before save): {writer.finish('incomplete', 'session stop')}")
        source.reset(robot.observe().state) if robot.synthetic else None
