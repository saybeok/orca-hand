#!/usr/bin/env bash
# Quest 3S (or webcam) -> retargeter -> AM-ARM200 + ORCA right hand v1 in MuJoCo.
#
# All 20 joints follow the operator: the hand from the retargeter, the arm from
# position IK on the wrist pose. Three arm joints can track wrist *position*
# only, not its rotation. A webcam sends no wrist pose, so the arm holds still.
#
# Default: first-frame relative follow. Optional absolute mapping:
#   python orca_am200/scripts/calibrate_operator.py
#
# With --clutch, pinch your OTHER hand to let the arm follow and release to park
# it, so you can reposition without dragging the robot.
#
# For --source quest the headset needs a secure context. Over USB:
#   adb devices                       # the Quest must appear as 'device'
#   adb reverse tcp:8765 tcp:8765
# then open http://localhost:8765 in Quest Browser and tap Start hand tracking.
#
# Usage:
#   ./scripts/teleop_am200.sh                            # Quest, arm + hand
#   ./scripts/teleop_am200.sh --clutch                   # off-hand pinch clutch
#   ./scripts/teleop_am200.sh --no-arm                   # hand only
#   ./scripts/teleop_am200.sh --arm-scale 0.35           # gentler arm motion
#   ./scripts/teleop_am200.sh --source webcam --show-video
set -euo pipefail

source "$(dirname "$(readlink -f "$0")")/../env.sh"
cd "$ORCA_HAND_ROOT/orca_teleop"

# Defaults the caller can override, since a repeated flag would otherwise be
# passed twice and only silently work because argparse keeps the last one.
source_arg="quest"
hand_arg="right"
args=()
while [ $# -gt 0 ]; do
  case "$1" in
    --source) source_arg="$2"; shift 2 ;;
    --hand)   hand_arg="$2";   shift 2 ;;
    *)        args+=("$1");    shift   ;;
  esac
done

exec .venv/bin/python scripts/teleop_am200_sim.py \
  --source "$source_arg" \
  --hand "$hand_arg" \
  --hand-config "$ORCA_MODEL_PATH" \
  --urdf_path "$ORCA_URDF_PATH" \
  --retarget-config "$ORCA_RETARGET_CONFIG" \
  ${args[@]+"${args[@]}"}
