"""Hardware sink for AM-ARM200 (3 DoF) + ORCA hand on separate Feetech buses."""

from __future__ import annotations

import logging
import queue
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
from orca_core import OrcaHand, OrcaJointPositions

from orca_teleop.am200.arm_client import AmArmFeetech, default_arm_config_path
from orca_teleop.am200.canonical import ARM_ROLES, CanonicalJointMap, RobotJointCommand
from orca_teleop.am200.sim_env import resolve_model_path
from orca_teleop.cameras import CameraManager, OpenCVCameraConfig
from orca_teleop.constants import MOTION_NUM_STEPS
from orca_teleop.pipeline import _SHUTDOWN, RecordableSink, SinkObservation

logger = logging.getLogger(__name__)

STALE_COMMAND_WARN_S = 2.0


class Am200HardwareSink(RecordableSink):
    """Drives the physical arm + hand from the canonical 20-DoF command.

    Arm joints use connect-time motor angles as hardware zero; model radians
    from IK are applied as signed offsets. Hand joints use ``orca_core`` physical
    degrees via :class:`OrcaHand`.
    """

    def __init__(
        self,
        hand_config_path: str | Path | None = None,
        arm_config_path: str | Path | None = None,
        model_path: str | Path | None = None,
        camera_configs: list[OpenCVCameraConfig] | None = None,
        enable_arm: bool = True,
        enable_hand: bool = True,
    ) -> None:
        self._hand_config_path = hand_config_path
        self._arm_config_path = Path(arm_config_path) if arm_config_path else default_arm_config_path()
        self._model_path = resolve_model_path(model_path)
        self._enable_arm = enable_arm
        self._enable_hand = enable_hand
        self._cameras = CameraManager(camera_configs or [])

        self._hand: OrcaHand | None = None
        self._arm: AmArmFeetech | None = None
        self._map: CanonicalJointMap | None = None
        self._target: np.ndarray | None = None
        self._last_command_ns: int | None = None
        self._stale_logged = False

    def connect(self) -> None:
        import mujoco

        if self._hand_config_path is None:
            import os

            env_path = os.environ.get("ORCA_MODEL_PATH")
            if not env_path:
                raise RuntimeError("Pass hand_config_path or set $ORCA_MODEL_PATH")
            self._hand_config_path = env_path

        model = mujoco.MjModel.from_xml_path(str(self._model_path))
        joint_map_path = self._model_path.parent / "joint_map.json"
        self._map = CanonicalJointMap.load(joint_map_path, model)
        self._target = np.zeros(self._map.dof, dtype=np.float64)

        if self._enable_hand:
            hand = OrcaHand(str(self._hand_config_path))
            ok, msg = hand.connect(interactive=False)
            if not ok:
                raise RuntimeError(f"hand connect failed: {msg}")
            hand.init_joints()
            self._hand = hand
            self._target[self._map.hand_slice] = self._map.hand_from_physical(self.home_position())

        if self._enable_arm:
            arm = AmArmFeetech(self._arm_config_path)
            arm.connect()
            self._arm = arm

        self._cameras.open()
        self._cameras.ensure_live()
        logger.info(
            "Am200HardwareSink connected: arm=%s hand=%s dof=%d",
            self._enable_arm,
            self._enable_hand,
            self._map.dof,
        )

    @property
    def model_path(self) -> Path:
        return self._model_path

    @property
    def joint_ids(self) -> list[str]:
        assert self._map is not None
        return list(self._map.roles)

    @property
    def retarget_model_path(self) -> str | None:
        return None if self._hand_config_path is None else str(self._hand_config_path)

    @property
    def camera_shapes(self) -> dict[str, tuple[int, int, int]]:
        return self._cameras.shapes

    def get_observation(self) -> SinkObservation:
        return SinkObservation(
            joint_state=self._read_joint_state(),
            images=self._cameras.capture(),
        )

    def _read_joint_state(self) -> np.ndarray:
        assert self._map is not None
        state = np.zeros(self._map.dof, dtype=np.float32)
        if self._arm is not None:
            state[self._map.arm_slice] = self._arm.read_model_rad().astype(np.float32)
        if self._hand is not None:
            phys = self._hand.get_joint_position()
            state[self._map.hand_slice] = self._map.hand_from_physical(phys).astype(np.float32)
        return state

    def set_target(self, action: RobotJointCommand | OrcaJointPositions) -> None:
        assert self._map is not None and self._target is not None
        if isinstance(action, RobotJointCommand):
            if tuple(action.joint_ids) != self._map.roles:
                raise ValueError("RobotJointCommand joint_ids mismatch")
            self._target = self._map.clip(action.position)
        elif isinstance(action, OrcaJointPositions):
            target = self._target.copy()
            target[self._map.hand_slice] = self._map.hand_from_physical(action)
            self._target = self._map.clip(target)
        else:
            raise TypeError(f"unsupported action type {type(action).__name__}")
        self._last_command_ns = time.monotonic_ns()
        self._stale_logged = False

    def dispatch_action(self, action: RobotJointCommand | OrcaJointPositions) -> None:
        self.set_target(action)
        self._apply()

    def _apply(self) -> None:
        assert self._map is not None and self._target is not None
        if self._arm is not None:
            self._arm.write_model_rad(self._target[self._map.arm_slice])
        if self._hand is not None:
            phys = self._map.hand_to_physical(self._target[self._map.hand_slice])
            self._hand.set_joint_positions(phys, num_steps=MOTION_NUM_STEPS)

    def home_position(self) -> OrcaJointPositions:
        assert self._hand is not None or self._hand_config_path is not None
        if self._hand is not None:
            cfg = self._hand.config
        else:
            cfg = OrcaHand(str(self._hand_config_path)).config
        positions = dict(cfg.neutral_position)
        positions["wrist"] = 0.0
        return OrcaJointPositions(positions)

    def go_home(self) -> None:
        assert self._map is not None
        self._target = np.zeros(self._map.dof, dtype=np.float64)
        if self._hand is not None:
            self._target[self._map.hand_slice] = self._map.hand_from_physical(self.home_position())
        self._apply()

    def run_loop(
        self,
        actions_q: queue.Queue[Any],
        stop_event: threading.Event,
    ) -> None:
        while not stop_event.is_set():
            try:
                item = actions_q.get(timeout=0.05)
            except queue.Empty:
                self._warn_if_stale()
                continue
            if item is _SHUTDOWN:
                break
            try:
                self.dispatch_action(item)
            except Exception:
                logger.exception("Am200HardwareSink rejected/failed an action")

    def _warn_if_stale(self) -> None:
        if self._stale_logged or self._last_command_ns is None:
            return
        idle_s = (time.monotonic_ns() - self._last_command_ns) / 1e9
        if idle_s >= STALE_COMMAND_WARN_S:
            logger.warning("No action for %.1fs — holding last target", idle_s)
            self._stale_logged = True

    def close(self) -> None:
        try:
            self._cameras.close()
            if self._hand is not None:
                self._hand.disable_torque()
                self._hand.disconnect()
            if self._arm is not None:
                self._arm.disconnect(disable_torque=True)
        except Exception:
            logger.exception("Am200HardwareSink.close() error")
        finally:
            self._hand = None
            self._arm = None
