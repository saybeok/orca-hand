"""Strict joint-space and timestamp contracts shared by capture and inference."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import numpy as np

CAMERAS = ("top", "front", "wrist")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def write_json(path, value):
    """Atomic replacement only of a pipeline-owned metadata file."""
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


@dataclass(frozen=True)
class Schema:
    names: tuple[str, ...]
    lower: np.ndarray
    upper: np.ndarray

    @classmethod
    def load(cls, path):
        data = read_json(path)
        joints = data["canonical_order"]
        if [j["index"] for j in joints] != list(range(data["total_dof"])):
            raise ValueError("Joint map indices must be contiguous and ordered")
        names = tuple(j["role"] for j in joints)
        bounds = np.asarray([j["range_rad"] for j in joints], dtype=float)
        if len(set(names)) != len(names) or bounds.shape != (len(names), 2):
            raise ValueError("Invalid joint map")
        if not np.isfinite(bounds).all() or np.any(bounds[:, 0] >= bounds[:, 1]):
            raise ValueError("Invalid joint limits")
        return cls(names, bounds[:, 0], bounds[:, 1])

    @property
    def size(self):
        return len(self.names)

    def vector(self, value):
        value = np.asarray(value, dtype=np.float64)
        if value.shape != (self.size,) or not np.isfinite(value).all():
            raise ValueError(f"Expected {self.size} finite model-radian joints, got {value.shape}")
        return value.copy()

    def contract(self):
        return {"joint_names": list(self.names), "units": "model_radians",
                "action_type": "absolute_joint_position_target", "cameras": list(CAMERAS),
                "state_semantics": "driver_reported_joint_position_not_independent_fingertip_sensing"}


@dataclass(frozen=True)
class CameraSample:
    image: np.ndarray
    sequence: int
    received_ns: int


@dataclass(frozen=True)
class Observation:
    state: np.ndarray
    read_started_ns: int
    read_finished_ns: int
    cameras: dict[str, CameraSample]


@dataclass(frozen=True)
class Command:
    position: np.ndarray
    received_ns: int
    source_timestamp_ns: int = 0
    raw: dict | None = None


class SafetyFilter:
    """No IK: hard-limit + per-tick velocity clamp in canonical joint space."""
    def __init__(self, schema, lower, upper, velocity, fps, max_tracking_error):
        self.schema = schema
        self.lower = schema.vector(lower)
        self.upper = schema.vector(upper)
        self.velocity = schema.vector(velocity)
        self.tracking = schema.vector(max_tracking_error)
        if not np.isfinite(fps) or fps <= 0:
            raise ValueError("fps must be positive")
        self.dt = 1 / fps
        if (np.any(self.lower >= self.upper) or np.any(self.lower < schema.lower)
                or np.any(self.upper > schema.upper) or np.any(self.velocity <= 0)
                or np.any(self.tracking <= 0)):
            raise ValueError("Safety limits must be finite, positive and within model limits")
        self.last = None

    def reset(self, state):
        state = self.schema.vector(state)
        if np.any(state < self.lower) or np.any(state > self.upper):
            raise ValueError("Measured start pose is outside reviewed safety limits")
        self.last = state

    def apply(self, target, state):
        target, state = self.schema.vector(target), self.schema.vector(state)
        if self.last is None:
            raise RuntimeError("Safety filter must be reset from a measured pose")
        if np.any(np.abs(state - self.last) > self.tracking):
            raise RuntimeError("Tracking error: stop issuing new targets and inspect hardware")
        target = np.clip(target, self.lower, self.upper)
        return np.clip(target, self.last - self.velocity * self.dt, self.last + self.velocity * self.dt)

    def commit(self, accepted):
        # Called only after all command writes return successfully.
        self.last = self.schema.vector(accepted)


class Synchronizer:
    """PC monotonic receipt times, NOT hardware exposure synchronization."""
    def __init__(self, schema, max_age_ms=120, max_camera_skew_ms=50):
        self.schema = schema
        self.age_ns = int(max_age_ms * 1e6)
        self.skew_ns = int(max_camera_skew_ms * 1e6)
        self.sequences = {}
        self.last_tick = None
        self.shapes = {}

    def validate(self, observation, command, now_ns):
        self.schema.vector(observation.state)
        self.schema.vector(command.position)
        if self.last_tick is not None and now_ns <= self.last_tick:
            raise ValueError("Non-monotonic decision timestamp")
        if not observation.read_started_ns <= observation.read_finished_ns <= now_ns:
            raise ValueError("Invalid state read interval")
        if set(observation.cameras) != set(CAMERAS):
            raise ValueError("Exactly top/front/wrist images are required; no substitution")
        times = [command.received_ns, observation.read_started_ns]
        camera_times = []
        for name, sample in observation.cameras.items():
            image = sample.image
            if image.dtype != np.uint8 or image.ndim != 3 or image.shape[-1] != 3:
                raise ValueError(f"{name}: expected HWC uint8 RGB")
            if name in self.shapes and self.shapes[name] != image.shape:
                raise ValueError(f"{name}: image shape changed")
            if sample.sequence <= self.sequences.get(name, -1):
                raise RuntimeError(f"{name}: repeated/reversed camera frame")
            times.append(sample.received_ns)
            camera_times.append(sample.received_ns)
        if any(t <= 0 or t > now_ns or now_ns - t > self.age_ns for t in times):
            raise RuntimeError("Stale/future observation or command; episode is invalid")
        if max(camera_times) - min(camera_times) > self.skew_ns:
            raise RuntimeError("Camera receipt skew exceeds configured budget")
        self.sequences = {k: v.sequence for k, v in observation.cameras.items()}
        self.shapes = {k: v.image.shape for k, v in observation.cameras.items()}
        self.last_tick = now_ns


def mock_safety(schema, fps):
    return SafetyFilter(schema, schema.lower, schema.upper, np.full(schema.size, .6),
                        fps, np.full(schema.size, .5))
