"""Gymnasium env for the combined AM-ARM200 + ORCA right hand model.

Reuses ``orca_sim``'s ``BaseOrcaHandEnv`` for reset/step/render rather than
reimplementing them; the only orca_sim-specific behaviour it has to escape is
scene lookup, which normally resolves a filename inside the installed package.
"""

from __future__ import annotations

import os
from pathlib import Path

from orca_sim.envs import BaseOrcaHandEnv

MODEL_PATH_ENV = "AM200_MODEL_PATH"
DEFAULT_MODEL_PATH = (
    Path(os.environ.get("ORCA_HAND_ROOT", Path.home() / "orca_hand"))
    / "orca_am200"
    / "robot_model"
    / "am_arm200_orca_right.xml"
)

# The arm and hand actuators are position servos, so a control step should
# advance about as much sim time as the sink's 30 Hz loop consumes in wall time.
# Otherwise the sim falls behind the operator and teleop feels sluggish.
_TARGET_CONTROL_HZ = 30.0


def resolve_model_path(model_path: str | Path | None = None) -> Path:
    path = Path(model_path or os.environ.get(MODEL_PATH_ENV) or DEFAULT_MODEL_PATH)
    if not path.is_file():
        raise FileNotFoundError(
            f"Combined model not found at {path}. Pass --model-path or set ${MODEL_PATH_ENV}."
        )
    return path.resolve()


class AmArm200OrcaRightEnv(BaseOrcaHandEnv):
    """The 20-DoF combined embodiment: 3-DoF arm plus the 17-DoF ORCA v1 hand."""

    def __init__(
        self,
        model_path: str | Path | None = None,
        frame_skip: int | None = None,
        render_mode: str | None = None,
    ) -> None:
        path = resolve_model_path(model_path)

        # BaseOrcaHandEnv resolves `scene_file` under orca_sim's packaged scenes
        # via `SCENES_ROOT / version / scene_file`. An absolute scene_file makes
        # that join return the path unchanged, which is how this out-of-tree
        # model gets loaded without duplicating the base env. `version` still has
        # to name a real orca_sim version for the lookup to run at all.
        super().__init__(
            scene_file=str(path),
            version="v1",
            frame_skip=frame_skip or 1,
            render_mode=render_mode,
        )
        if self.scene_path != path:
            raise RuntimeError(
                f"orca_sim resolved the scene to {self.scene_path} instead of {path}; "
                "its scene lookup changed and this env needs updating."
            )

        self.version = "am_arm200_orca_right_v1"
        if frame_skip is None:
            self.frame_skip = max(1, round(1.0 / (_TARGET_CONTROL_HZ * self.model.opt.timestep)))
