# ORCA Hand v1 (Feetech, right) setup

Teleoperation stack for a v1 ORCA Hand with Feetech HLS servos, driven from a
USB webcam through MediaPipe.

## Layout

```
orca_core/               hardware control, tensioning, calibration (editable install)
orcahand_description/    URDF/MJCF models
orca_teleop/             retargeting pipeline, owns the .venv used by everything
configs/
  orcahand-right-v1-feetech/config.yaml   hand config (Feetech variant of the v1 right hand)
  adaptive_analytical_orca_v1.yaml        retarget config (v1 joint names)
scripts/                 wrappers that source env.sh and pass the right paths
env.sh                   environment for every command
```

## Setup notes

Everything runs from the single virtualenv at `orca_teleop/.venv` (Python 3.11).
`orca_core` is installed there as an editable checkout, so the calibration
scripts and the teleop pipeline always agree on one version.

Three things about this machine required deviating from the upstream READMEs:

- **`~/.bashrc` sources ROS 2 Humble and CUDA.** `PYTHONPATH` then shadows venv
  packages, and `LD_LIBRARY_PATH` makes the loader bind ROS's `libeigenpy.so`
  instead of the cmeel-bundled one, which breaks `import pinocchio` with an
  undefined `EIGENPY_ARRAY_API` symbol. `env.sh` unsets both.
- **`orca_teleop`'s lockfile pins `orca_core` 0.2.1**, which predates the
  Feetech baudrate probing and USB VID tables. The local clone (0.4.0) is
  installed over it with `uv pip install -e ../orca_core --no-deps`. Re-running
  `uv sync` reverts that, so reapply it afterwards. `--no-deps` matters: a plain
  editable install re-resolves numpy and breaks the pinocchio wheel.
- **Meshcat is not in any extra.** The error message suggests
  `--extra urdf-viz`, which does not exist; `uv pip install meshcat` is the fix.

## Hardware config

The bundled `orca_core/models/v1/orcahand-right/config.yaml` assumes Dynamixel
motors at 3M baud. The copy in `configs/` changes only the transport:

| field | value | why |
| --- | --- | --- |
| `motor_type` | `feetech` | selects `FeetechClient` |
| `baudrate` | `1000000` | the only rate in `MOTOR_BAUD_RATES[feetech]` |
| `control_mode` | `current_based_position` | Feetech honours mode 5 as a torque limit |
| `max_current` | `400` | mA maps onto the 0-1000 torque register, so 40% |

Joint maps, ROMs, neutral pose and the calibration sequence are unchanged from
the v1 hand.

This hand's adapter is a QinHeng `1a86:55d3`, a CH343/CH9102 that binds
`cdc-acm` and therefore appears as **`/dev/ttyACM0`**, not the `/dev/ttyUSB0`
you get from the older CH340 chips. `port: auto` resolves it either way, since
orca_core matches on the USB vendor ID rather than the device name.

Motor layout, derived from `joint_to_motor_map` (sign stripped):

| | ID | model |
| --- | --- | --- |
| wrist | 17 | HLS3930 |
| fingers | 16..1 | HLS3915 |

The highest ID attaches directly to the board, then each lower ID chains off the
one above it: board -> 17 -> 16 -> ... -> 1.

## Retarget config

`joint_name_aliases` upstream renames `thumb_pip` to `thumb_cmc`, which only
fits the v2 joint vocabulary. The v1 URDF names that joint `right_thumb_pip`,
exactly as the v1 hand config does, so `configs/adaptive_analytical_orca_v1.yaml`
empties the alias map. The frame block needs no change: the default
`robot_frames` in the upstream config is already the v1 layout.

`orca_teleop/src/orca_teleop/urdf_viz.py` had the same alias hardcoded; it now
picks whichever of the two names the URDF actually defines, so it works for both
v1 and v2.

## Running

```bash
scripts/smoke_retarget.py   # offline: synthetic hand, no camera or hardware
scripts/teleop_viz.sh       # webcam -> Meshcat viewer at http://127.0.0.1:7000/static/
```

Bring-up, in order, with the hand plugged in and powered:

```bash
scripts/hand.sh detect      # confirm the port and which motors answer
scripts/hand.sh configure   # one-time: burn IDs/baud into a fresh motor chain
scripts/hand.sh tension     # hold taut, tension tendons by hand, Ctrl+C when done
scripts/hand.sh calibrate   # 28-step limit sweep, writes calibration.yaml
scripts/hand.sh neutral     # move to the neutral pose
scripts/teleop_hw.sh        # webcam -> physical hand
```

`configure` is interactive: it prompts for one motor at a time and, because
Feetech needs an unpowered hotplug, asks you to unplug and replug USB between
motors. Motors that already carry their target ID are skipped, so it is safe to
re-run after an interruption. Skip it entirely if the chain is already
programmed — `detect` will list the configured IDs.

Add `--mock` to `tension`, `calibrate` or `neutral` for a dry run without
hardware.

Hold your hand still in frame at the start of either teleop command: the
retargeter emits nothing until it has auto-scaled, which takes about 30 frames.
