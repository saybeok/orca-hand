"""Quest -> existing retargeter + arm IK -> timestamped canonical targets."""
from __future__ import annotations
import queue
import multiprocessing
import threading
import time
import numpy as np

from .core import Command


class QuestSource:
    def __init__(self, schema, config):
        # This source receives frames from the existing MetaQuest publisher.
        # It does not start a second robot consumer or touch any serial bus.
        from orca_teleop.am200.composer import Am200ActionComposer, ArmMappingConfig
        from orca_teleop.am200.operator_cal import OperatorCalibration
        from orca_teleop.ingress.server import IngressServer
        from orca_teleop.pipeline import TeleopQueues, retargeter_worker
        calibration = OperatorCalibration.load(config["operator_calibration"])
        if calibration is None or calibration.handedness != "right":
            raise ValueError("A right-hand operator calibration file is required")
        self.calibration = calibration
        self.schema = schema
        self.lock = threading.Lock()
        self.latest_command = None
        self.last_frame = None
        self.reset_ns = 0
        self.error = None
        self.ready_count = 0
        self.stop = threading.Event()
        self.retarget_reset = threading.Event()
        self.composer = Am200ActionComposer.from_model_path(
            config["model_path"], config=ArmMappingConfig(**config.get("arm_mapping", {})),
            calibration=calibration)
        self.queues = TeleopQueues(queue.Queue(maxsize=2), queue.Queue(maxsize=2))
        self.server = IngressServer(self.queues.landmarks_q, self.stop, port=config.get("ingress_port", 50051))
        self.port = self.server.start()
        if self.port == 0:
            self.server.stop()
            raise RuntimeError("Failed to bind the Quest ingress port")
        self.worker = threading.Thread(target=retargeter_worker, kwargs=dict(
            queues=self.queues, stop_event=self.stop, model_path=config["hand_config"],
            retargeter_backend="adaptive_analytical", landmark_source="metaquest",
            action_composer=self._compose, retargeter_reset_event=self.retarget_reset), daemon=True, name="quest-retarget-and-IK")
        self.publisher = None
        self.publisher_reset = None
        self.worker.start()
        if config.get("launch_quest_publisher", True):
            from orca_teleop.pipeline import _metaquest_publisher
            context = multiprocessing.get_context("spawn")
            self.publisher_reset = context.Event()
            self.publisher = context.Process(target=_metaquest_publisher, kwargs=dict(
                port=self.port, handedness="right", quest_host="0.0.0.0",
                quest_port=config.get("quest_web_port", 8443), fps=30,
                wrist_enabled=True, reset_event=self.publisher_reset, clutch_enabled=True), daemon=True)
            try:
                self.publisher.start()
            except Exception:
                self.close()
                raise
        else:
            self.close()
            raise ValueError("Learning capture requires the owned Quest publisher for per-episode wrist resets")

    def _compose(self, frame, hand_action):
        from orca_teleop.am200.canonical import RobotJointCommand
        with self.lock:
            self.last_frame = frame
            if self.retarget_reset.is_set() or (self.publisher_reset is not None and self.publisher_reset.is_set()):
                return hand_action
            if frame.received_monotonic_ns < self.reset_ns:
                return hand_action
            try:
                if (frame.handedness != "right" or frame.wrist_transform is None
                        or not np.isfinite(frame.keypoints).all()
                        or not np.isfinite(frame.wrist_transform).all()):
                    raise ValueError("A finite right-hand wrist pose and landmarks are required")
                if self.ready_count == 0:
                    distance = np.linalg.norm(frame.wrist_transform[:3, 3] - self.calibration.neutral_wrist_flu_m)
                    if distance > .1:
                        raise ValueError("Start with the VR wrist within 10 cm of its calibrated neutral")
                action = self.composer(frame, hand_action)
                if not isinstance(action, RobotJointCommand) or tuple(action.joint_ids) != self.schema.names:
                    raise ValueError("Hand-only or wrong-order commands cannot enter a full-robot dataset")
                self.latest_command = Command(self.schema.vector(action.position), frame.received_monotonic_ns,
                                              frame.timestamp_ns, {
                    "keypoints_m": frame.keypoints.tolist(), "wrist_transform": frame.wrist_transform.tolist(),
                    "wrist_angle_degrees": frame.wrist_angle_degrees,
                    "clutch_supported": frame.clutch_supported, "clutch_engaged": frame.clutch_engaged})
                self.ready_count += 1
                self.error = None
                return action
            except Exception as exc:
                self.error = exc
                self.latest_command = None
                raise

    def reset(self, state):
        with self.lock:
            self.reset_ns = time.perf_counter_ns()
            self.composer.reset(arm_position=self.schema.vector(state)[:3])
            self.latest_command = None
            self.ready_count = 0
            self.error = None
            self.retarget_reset.set()
            if self.publisher_reset is not None:
                self.publisher_reset.set()
            while True:
                try:
                    self.queues.landmarks_q.get_nowait()
                except queue.Empty:
                    break

    def latest(self, observation):
        if self.publisher is not None and not self.publisher.is_alive():
            raise RuntimeError("Quest publisher process stopped")
        if not self.worker.is_alive():
            raise RuntimeError("Quest retargeter worker stopped")
        # Upstream actions are not robot commands here; only the callback publishes
        # validated full-robot targets. Drain its diagnostic queue to avoid blockage.
        while not self.queues.actions_q.empty():
            self.queues.actions_q.get_nowait()
        with self.lock:
            if self.error:
                raise RuntimeError(f"Quest input invalid: {self.error}")
            if self.latest_command is None or self.ready_count < 5:
                raise LookupError("Waiting for five fresh full-robot Quest samples")
            return self.latest_command

    def close(self):
        self.stop.set()
        self.server.stop()
        self.worker.join(timeout=3)
        if self.publisher is not None and self.publisher.pid is not None:
            # Only this runner's publisher child is terminated, never another
            # user's process. It has no motor connection.
            self.publisher.terminate()
            self.publisher.join(timeout=3)
