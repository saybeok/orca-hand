"""Mock and opt-in hardware adapters. Only the control thread calls robot I/O."""
from __future__ import annotations
import hashlib
import time
from pathlib import Path
import numpy as np

from .core import CAMERAS, CameraSample, Command, Observation, SafetyFilter, mock_safety, read_json


class MockRobot:
    synthetic = True

    def __init__(self, schema, fps=15):
        self.schema = schema
        self.fps = fps
        self.state = np.clip(np.zeros(schema.size), schema.lower, schema.upper)
        self.sequence = 0
        self.safety = mock_safety(schema, fps)
        self.contract_extra = {"calibration_id": "synthetic-only", "fps": fps,
                               "image_shapes": {k: [64, 96, 3] for k in CAMERAS}}

    def connect(self):
        pass

    def observe(self):
        started = time.perf_counter_ns()
        state = self.state.copy()
        finished = time.perf_counter_ns()
        self.sequence += 1
        cameras = {}
        for i, name in enumerate(CAMERAS):
            image = np.full((64, 96, 3), 35 + i * 55, dtype=np.uint8)
            image[:, self.sequence % 96, :] = 255
            cameras[name] = CameraSample(image, self.sequence, time.perf_counter_ns())
        return Observation(state, started, finished, cameras)

    def dispatch(self, target):
        self.state = self.schema.vector(target)
        return self.state.copy()

    def check_start(self, state):
        self.schema.vector(state)

    def close(self):
        pass


class MockSource:
    def __init__(self, schema):
        self.schema = schema
        self.index = 0

    def reset(self, state):
        self.origin = self.schema.vector(state)
        self.index = 0

    def latest(self, observation):
        self.index += 1
        target = self.origin + .15 * np.sin(self.index / 15 + np.arange(self.schema.size) / 4)
        return Command(np.clip(target, self.schema.lower, self.schema.upper), time.perf_counter_ns(),
                       raw={"kind": "synthetic", "sample": self.index})

    def close(self):
        pass


def hardware_preflight(config_path, schema, acknowledged=False):
    """No robot imports, serial access, torque or network before this gate."""
    config_path = Path(config_path).resolve()
    cfg = read_json(config_path)
    if not acknowledged or cfg.get("hardware_reviewed") is not True:
        raise ValueError("Hardware locked: review the template and pass --enable-hardware explicitly")
    if cfg.get("robot_type") != "am200_3dof_orca17" or schema.size != 20:
        raise ValueError("This backend is 3 arm + 17 hand = 20D, not SO-101 22D")
    if cfg.get("stop_mode") != "retain_last_goal_requires_physical_support":
        raise ValueError("A reviewed physical stop/support procedure is required")
    if not str(cfg.get("calibration_id", "")).strip():
        raise ValueError("A stable calibration ID is required")
    for key in ("hand_config", "arm_config", "model_path", "operator_calibration"):
        path = Path(cfg[key])
        path = path if path.is_absolute() else config_path.parent / path
        if not path.is_file():
            raise ValueError(f"Missing {key}: {path}")
        cfg[key] = str(path.resolve())
    if set(cfg["cameras"]) != set(CAMERAS):
        raise ValueError("Three named cameras required")
    if len({str(c["index"]) for c in cfg["cameras"].values()}) != 3:
        raise ValueError("Camera device IDs must be distinct")
    fps = cfg["fps"]
    safety = SafetyFilter(schema, cfg["lower_rad"], cfg["upper_rad"],
                          cfg["max_velocity_rad_s"], fps, cfg["max_tracking_error_rad"])
    start = schema.vector(cfg["episode_start_rad"])
    tolerance = schema.vector(cfg["episode_start_tolerance_rad"])
    if np.any(tolerance <= 0):
        raise ValueError("Start pose tolerances must be positive")
    safety.reset(start)
    if any(float(c.get("fps", 0)) < fps for c in cfg["cameras"].values()):
        raise ValueError("Cameras must run at least as fast as the recording rate")
    return cfg, safety


class HardwareRobot:
    synthetic = False

    def __init__(self, schema, config_path, acknowledged=False):
        self.schema = schema
        self.cfg, self.safety = hardware_preflight(config_path, schema, acknowledged)
        self.fps = self.cfg["fps"]
        self.sink = None
        self.cameras = None
        self.contract_extra = self._contract()

    def _contract(self):
        hand_dir = Path(self.cfg["hand_config"]).parent
        files = {k: Path(self.cfg[k]) for k in ("hand_config", "arm_config", "model_path")}
        files.update({f"hand_calibration/{f.name}": f for f in sorted(hand_dir.glob("*calibrat*.yaml"))})
        return {"calibration_id": self.cfg["calibration_id"], "fps": self.fps,
                "calibration_sha256": {k: hashlib.sha256(f.read_bytes()).hexdigest() for k, f in files.items()},
                "image_shapes": {k: [v["height"], v["width"], 3] for k, v in self.cfg["cameras"].items()}}

    def connect(self):
        import yaml
        from orca_teleop.am200.hw_sink import Am200HardwareSink
        from orca_teleop.cameras import CameraManager, OpenCVCameraConfig
        arm_cfg = yaml.safe_load(Path(self.cfg["arm_config"]).read_text(encoding="utf-8"))
        hand_cfg = yaml.safe_load(Path(self.cfg["hand_config"]).read_text(encoding="utf-8"))
        if any(not hand_cfg.get(k) for k in ("port", "motor_type", "baudrate")):
            raise ValueError("Explicit ORCA port, motor_type and baudrate required; no port auto-detection")
        if str(hand_cfg["motor_type"]).lower() != "feetech":
            raise ValueError("Learning strict-write adapter currently validated only for the Feetech hand")
        if arm_cfg.get("motor_zero_rad") is None or arm_cfg.get("boot_model_limits_rad") is None:
            raise ValueError("Persist motor_zero_rad and boot_model_limits_rad in the arm config first")
        if arm_cfg.get("max_step_from_home_rad") not in (None, False):
            raise ValueError("Disable legacy arm clipping; the common SafetyFilter owns command limits")
        boot = np.asarray(arm_cfg["boot_model_limits_rad"], dtype=float)
        if (boot.shape != (3, 2) or np.any(boot[:, 0] < self.safety.lower[:3])
                or np.any(boot[:, 1] > self.safety.upper[:3])):
            raise ValueError("Arm boot limits must be within reviewed learning limits")
        self.cameras = CameraManager([OpenCVCameraConfig(name=name, **spec)
                                      for name, spec in self.cfg["cameras"].items()])
        self.cameras.open()  # Fail camera checks before opening any motor bus.
        self.cameras.ensure_live()
        if {k: list(v) for k, v in self.cameras.shapes.items()} != self.contract_extra["image_shapes"]:
            raise ValueError("Actual camera dimensions differ from the reviewed profile")
        self.sink = Am200HardwareSink(hand_config_path=self.cfg["hand_config"],
                                     arm_config_path=self.cfg["arm_config"],
                                     model_path=self.cfg["model_path"], learning_mode=True)
        self.sink.connect()
        if tuple(self.sink.joint_ids) != self.schema.names:
            raise ValueError("Hardware joint ordering differs from dataset schema")
        if self._contract() != self.contract_extra:
            raise ValueError("Driver changed calibration/config at connection; review and restart with the stable profile")

    def observe(self):
        start = time.perf_counter_ns()
        state = self.schema.vector(self.sink.read_joint_state())
        finish = time.perf_counter_ns()
        samples = self.cameras.capture_samples()
        return Observation(state, start, finish,
                           {k: CameraSample(v.image, v.sequence, v.received_ns) for k, v in samples.items()})

    def dispatch(self, target):
        from orca_teleop.am200.canonical import RobotJointCommand
        return self.sink.dispatch_action(RobotJointCommand.now(self.schema.names, target))

    def check_start(self, state):
        error = np.abs(self.schema.vector(state) - self.cfg["episode_start_rad"])
        if np.any(error > self.cfg["episode_start_tolerance_rad"]):
            raise RuntimeError("Return to the reviewed episode start pose in preview mode first")

    def close(self):
        if self.cameras:
            self.cameras.close()
        if self.sink:
            self.sink.close()
        print("Hardware teardown retains the last motor goal. Support the arm and use the physical stop procedure.")
