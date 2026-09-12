"""Sim-backed sink for the combined 20-DoF AM-ARM200 + ORCA right hand robot.

The upstream ``OrcaHandSimSink`` cannot drive this robot: it only knows how to
build the hand-only ``orca_sim`` envs, and its actuator mapping requires the
model's actuators to be a 1:1 cover of the orca_core hand config, so the three
arm actuators make it raise. This sink speaks the canonical 20-DoF vector
instead, and accepts hand-only ``OrcaJointPositions`` as well so the existing
ingress and retargeter stack drives the hand while the arm holds its target.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from orca_core import OrcaHand, OrcaJointPositions

from orca_teleop.am200.canonical import CanonicalJointMap, RobotJointCommand
from orca_teleop.am200.sim_env import AmArm200OrcaRightEnv, resolve_model_path
from orca_teleop.cameras import CameraManager, OpenCVCameraConfig
from orca_teleop.pipeline import _SHUTDOWN, RecordableSink, SinkObservation
from orca_teleop.utils import RateTicker

logger = logging.getLogger(__name__)

DEFAULT_RENDER_FPS = 30
# How long the sink coasts on the last target before it says so. Teleop input
# stopping is normal (the operator lowers their hand); silence is not.
STALE_COMMAND_WARN_S = 2.0


@dataclass(frozen=True)
class SimCameraConfig:
    name: str = "frontal"
    width: int = 320
    height: int = 240


class Am200SimSink(RecordableSink):
    """Steps the combined model from the actions queue.

    Actions may be a full-robot :class:`RobotJointCommand` in the canonical
    order, or an :class:`OrcaJointPositions` for the hand alone. Either way the
    sink holds one 20-DoF target vector in the model's radian convention and
    writes it to every actuator each tick, so a dropped or partial command
    leaves the robot where it was rather than relaxing.
    """

    def __init__(
        self,
        model_path: str | Path | None = None,
        hand_config_path: str | Path | None = None,
        render_mode: str = "human",
        camera_config: SimCameraConfig | None = None,
        camera_configs: list[OpenCVCameraConfig] | None = None,
    ) -> None:
        self._model_path = resolve_model_path(model_path)
        self._joint_map_path = self._model_path.parent / "joint_map.json"
        self._hand_config_path = hand_config_path
        self._render_mode = render_mode
        self._camera_config = camera_config or SimCameraConfig()
        self._cameras = CameraManager(camera_configs or [])

        self._env: AmArm200OrcaRightEnv | None = None
        self._map: CanonicalJointMap | None = None
        self._hand_config: Any | None = None
        self._target: np.ndarray | None = None
        self._dt = 1.0 / DEFAULT_RENDER_FPS
        self._renderer: Any | None = None
        self._record_camera: Any | None = None
        self._last_command_ns: int | None = None
        self._stale_logged = False

    def connect(self) -> None:
        import mujoco

        env = AmArm200OrcaRightEnv(
            model_path=self._model_path,
            render_mode=self._render_mode,
        )
        env.reset()

        self._env = env
        self._map = CanonicalJointMap.load(self._joint_map_path, env.model)
        self._hand_config = self._load_hand_config()
        self._target = self._home_vector()

        self._dt = 1.0 / float(env.metadata.get("render_fps", DEFAULT_RENDER_FPS))

        self._renderer = mujoco.Renderer(
            env.model,
            height=self._camera_config.height,
            width=self._camera_config.width,
        )
        self._record_camera = mujoco.MjvCamera()
        mujoco.mjv_defaultFreeCamera(env.model, self._record_camera)

        if self._camera_config.name in self._cameras.names:
            raise ValueError(
                f"Real camera name {self._camera_config.name!r} collides with the sim "
                "render camera name; rename one so both observations get distinct keys."
            )
        self._cameras.open()
        self._cameras.ensure_live()

        self.go_home()

        logger.info(
            "Am200SimSink connected: model=%s dof=%d (arm %d + hand %d) "
            "frame_skip=%d dt=%.3fs cameras=%s",
            self._model_path.name,
            self._map.dof,
            self._map.arm_dof,
            self._map.hand_dof,
            env.frame_skip,
            self._dt,
            self.camera_shapes,
        )

    @property
    def model_path(self) -> Path:
        """The combined MJCF this sink drives, so a composer can share it."""
        return self._model_path

    @property
    def joint_ids(self) -> list[str]:
        assert self._map is not None, "connect() must be called first"
        return list(self._map.roles)

    @property
    def retarget_model_path(self) -> str | None:
        """Lets the pipeline resolve the retargeter's hand config from the sink."""
        return None if self._hand_config_path is None else str(self._hand_config_path)

    @property
    def camera_shapes(self) -> dict[str, tuple[int, int, int]]:
        return {
            self._camera_config.name: (
                self._camera_config.height,
                self._camera_config.width,
                3,
            ),
            **self._cameras.shapes,
        }

    def get_observation(self) -> SinkObservation:
        return SinkObservation(
            joint_state=self._read_joint_state(),
            images={self._camera_config.name: self._render(), **self._cameras.capture()},
        )

    def _read_joint_state(self) -> np.ndarray:
        """Measured joint positions in the canonical order, in model radians."""
        assert self._env is not None and self._map is not None
        return self._map.from_qpos(self._env.data.qpos).astype(np.float32)

    def _render(self) -> np.ndarray:
        assert self._env is not None and self._renderer is not None
        self._renderer.update_scene(self._env.data, camera=self._record_camera)
        return np.asarray(self._renderer.render(), dtype=np.uint8)

    def set_target(self, action: RobotJointCommand | OrcaJointPositions) -> None:
        """Update the held target without stepping the sim."""
        assert self._map is not None and self._target is not None

        if isinstance(action, RobotJointCommand):
            if tuple(action.joint_ids) != self._map.roles:
                raise ValueError(
                    "RobotJointCommand joint_ids do not match the canonical order; "
                    f"got {len(action.joint_ids)} ids starting {action.joint_ids[:3]}"
                )
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
        self._step()

    def _step(self) -> None:
        assert self._env is not None and self._map is not None and self._target is not None
        self._env.step(self._map.to_ctrl(self._target).astype(np.float32))

    def _home_vector(self) -> np.ndarray:
        """Arm at zero, hand at the orca_core neutral pose, in model radians."""
        assert self._map is not None and self._hand_config is not None
        home = np.zeros(self._map.dof, dtype=np.float64)
        home[self._map.hand_slice] = self._map.hand_from_physical(self.home_position())
        return self._map.clip(home)

    def home_position(self) -> OrcaJointPositions:
        assert self._hand_config is not None, "connect() must be called first"
        positions = dict(self._hand_config.neutral_position)
        positions["wrist"] = 0.0
        return OrcaJointPositions(positions)

    def go_home(self) -> None:
        assert self._map is not None
        self._target = self._home_vector()
        self._step()

    def run_loop(
        self,
        actions_q: queue.Queue[Any],
        stop_event: threading.Event,
    ) -> None:
        assert self._env is not None, "connect() must be called before run_loop()"
        ticker = RateTicker(dt=self._dt)

        while not stop_event.is_set():
            shutdown = False
            try:
                item = actions_q.get_nowait()
                if item is _SHUTDOWN:
                    shutdown = True
                else:
                    try:
                        self.set_target(item)
                    except (TypeError, ValueError):
                        logger.exception("Am200SimSink rejected an action; holding last target")
            except queue.Empty:
                pass

            if shutdown:
                break

            self._warn_if_stale()

            try:
                self._step()
            except Exception:
                logger.exception("combined env.step() failed")
                break

            ticker.tick()

    def _warn_if_stale(self) -> None:
        if self._stale_logged or self._last_command_ns is None:
            return
        idle_s = (time.monotonic_ns() - self._last_command_ns) / 1e9
        if idle_s >= STALE_COMMAND_WARN_S:
            logger.warning(
                "No action for %.1fs — holding the last target. Check the publisher.", idle_s
            )
            self._stale_logged = True

    def close(self) -> None:
        if self._env is None:
            return
        try:
            self._cameras.close()
            if self._renderer is not None:
                self._renderer.close()
            self._env.close()
        except Exception:
            logger.exception("Am200SimSink.close() encountered an error")
        finally:
            self._env = None
            self._renderer = None
            self._record_camera = None

    def _load_hand_config(self) -> Any:
        """Load the orca_core hand config that defines the hand's joint order."""
        if self._hand_config_path is None:
            import os

            env_path = os.environ.get("ORCA_MODEL_PATH")
            if not env_path:
                raise RuntimeError(
                    "No hand config given. Pass hand_config_path or set $ORCA_MODEL_PATH "
                    "to the orca_core config.yaml for this hand."
                )
            self._hand_config_path = env_path
        return OrcaHand(str(self._hand_config_path)).config
