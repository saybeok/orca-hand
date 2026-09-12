#!/usr/bin/env bash
# Quest / webcam -> IK + retarget -> physical AM-ARM200 + ORCA hand.
#
# Ports (this unit):
#   hand  /dev/ttyACM0  (configs/orcahand-right-v1-feetech)
#   arm   /dev/ttyACM1  (configs/am-arm200-3dof)
#
# Quest (arm moves):
#   adb devices && adb reverse tcp:8765 tcp:8765
#   ./scripts/teleop_am200_hw.sh --clutch
# Off-hand (left when teleoping right):
#   pinch  = arm clutch (follow while held)
#   fist   = re-run hand retarget auto-scale (~1 s neutral pose)
#
# Webcam (hand only; arm holds connect pose — no wrist pose on MediaPipe):
#   ./scripts/teleop_am200_hw.sh --source webcam --show-video --no-mirror --camera-id 0
#
# Hand-only on hardware (arm bus closed):
#   ./scripts/teleop_am200_hw.sh --no-arm --source webcam --show-video --no-mirror --camera-id 0
set -euo pipefail

source "$(dirname "$(readlink -f "$0")")/../env.sh"
cd "$ORCA_HAND_ROOT/orca_teleop"

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

exec .venv/bin/python scripts/teleop_am200_hw.py \
  --source "$source_arg" \
  --hand "$hand_arg" \
  --hand-config "$ORCA_MODEL_PATH" \
  --urdf_path "$ORCA_URDF_PATH" \
  --retarget-config "$ORCA_RETARGET_CONFIG" \
  ${args[@]+"${args[@]}"}
