"""Offline-first command line. No hardware/ML work occurs merely by importing it."""
from __future__ import annotations
import argparse
import importlib.metadata
import json
from pathlib import Path
import shlex
import sys

from .core import Schema, write_json

REPO = Path(__file__).resolve().parents[3]
DEFAULT_MAP = REPO / "orca_am200" / "robot_model" / "joint_map.json"


def parser():
    p = argparse.ArgumentParser(description="ORCA+AM200 / foot pedal / ACT & pi05 joint-space pipeline")
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("doctor", help="Read-only dependency report; never opens a device")
    sub.add_parser("schema", help="Print the exact joint order and model ranges (not approved safety limits)")
    for command in ("record", "run-policy"):
        c = sub.add_parser(command)
        c.add_argument("--joint-map", default=str(DEFAULT_MAP))
        c.add_argument("--backend", choices=["mock", "hardware"], default="mock")
        c.add_argument("--hardware-config")
        c.add_argument("--enable-hardware", action="store_true")
        c.add_argument("--task", default="Pick up one tile from the blanket, lift it, and place it flat at the marked target.")
        c.add_argument("--pedal-device", help="Linux evdev device path; omitted = n/r/q + Enter")
        if command == "record":
            c.add_argument("--raw-root", required=True)
            c.add_argument("--mock-episodes", type=int, default=0)
            c.add_argument("--frames", type=int, default=30)
            c.add_argument("--fast", action="store_true", help="Skip mock-only real-time sleeps")
        else:
            c.add_argument("--checkpoint", required=True)
            c.add_argument("--device", default="cpu")
            c.add_argument("--allow-synthetic", action="store_true")
    for command in ("inspect", "export"):
        c = sub.add_parser(command)
        c.add_argument("--raw-root", required=True)
        c.add_argument("--allow-synthetic", action="store_true")
        c.add_argument("--seed", type=int, default=42)
        c.add_argument("--output", required=command == "export")
    c = sub.add_parser("train")
    c.add_argument("--dataset", required=True, help="Export root containing paran_manifest.json")
    c.add_argument("--model", choices=["act", "pi05"], required=True)
    c.add_argument("--output", required=True)
    c.add_argument("--device", default="cuda")
    c.add_argument("--steps", type=int, default=20000)
    c.add_argument("--batch-size", type=int)
    c.add_argument("--base-checkpoint", default="lerobot/pi05_base")
    c.add_argument("--allow-synthetic", action="store_true")
    c.add_argument("--execute", action="store_true", help="Actually launch training; default only prints arguments")
    c = sub.add_parser("evaluate")
    c.add_argument("--dataset", required=True)
    c.add_argument("--checkpoint", required=True)
    c.add_argument("--output", required=True)
    c.add_argument("--split", choices=["validation", "test"], default="validation")
    c.add_argument("--device", default="cpu")
    c.add_argument("--max-batches", type=int, default=100)
    c.add_argument("--allow-synthetic", action="store_true")
    c = sub.add_parser("quest-publisher", help="Start VR publisher only; no robot connection")
    c.add_argument("--ingress-port", type=int, default=50051)
    c.add_argument("--web-port", type=int, default=8443)
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        dispatch(args)
    except (ValueError, RuntimeError, OSError, ImportError, TimeoutError) as exc:
        print(f"STOPPED: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc


def dispatch(args):
    if args.command == "doctor":
        versions = {}
        for name in ("numpy", "lerobot", "torch", "opencv-python", "mujoco", "grpcio", "evdev"):
            try:
                versions[name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                versions[name] = "not installed"
        print(json.dumps({"python": sys.version, "versions": versions,
                          "required_lerobot": "0.4.4", "hardware_opened": False}, indent=2))
    elif args.command == "schema":
        schema = Schema.load(DEFAULT_MAP)
        print(json.dumps({**schema.contract(), "model_lower_rad": schema.lower.tolist(),
                          "model_upper_rad": schema.upper.tolist(),
                          "warning": "Model ranges are not reviewed hardware safety limits."}, indent=2))
    elif args.command == "evaluate":
        from .lerobot_io import evaluate_offline
        print(json.dumps(evaluate_offline(args.dataset, args.checkpoint, args.output, args.split,
                                          args.device, args.max_batches, args.allow_synthetic), indent=2))
    elif args.command in ("inspect", "export"):
        from .lerobot_io import inspect_raw, export_dataset
        if args.command == "inspect":
            report = inspect_raw(args.raw_root, args.allow_synthetic, args.seed)
            if args.output:
                if Path(args.output).exists():
                    raise FileExistsError("Report already exists; choose a new path")
                write_json(args.output, report)
        else:
            report = export_dataset(args.raw_root, args.output, args.allow_synthetic, args.seed)
        print(json.dumps({k: v for k, v in report.items() if k != "train_vector_stats"}, indent=2))
    elif args.command == "train":
        from .lerobot_io import training_command, execute_training
        cmd, manifest = training_command(args.dataset, args.model, args.output, args.device, args.steps,
                                         args.batch_size, args.base_checkpoint, args.allow_synthetic)
        print("Training arguments (no shell execution):\n" + shlex.join(cmd))
        if args.execute:
            execute_training(cmd, manifest, args.output)
        else:
            print("DRY RUN: no model downloaded, GPU job launched, or dataset uploaded. Add --execute after review.")
    elif args.command == "quest-publisher":
        from orca_teleop.pipeline import _metaquest_publisher
        _metaquest_publisher(port=args.ingress_port, handedness="right", quest_host="0.0.0.0",
                             quest_port=args.web_port, fps=30, wrist_enabled=True, clutch_enabled=True)
    else:
        run_robot(args)


def run_robot(args):
    from .backends import MockRobot, MockSource, HardwareRobot
    from .controls import ControlInput
    from .runner import record_mock, interactive_session, metadata_for
    schema = Schema.load(args.joint_map)
    if args.backend == "hardware":
        if not args.hardware_config:
            raise ValueError("Hardware config is required")
        if getattr(args, "mock_episodes", 0) or getattr(args, "fast", False):
            raise ValueError("Automatic/fast modes cannot be used with hardware")
        if getattr(args, "allow_synthetic", False):
            raise ValueError("Synthetic checkpoints are not permitted on hardware")
        robot = HardwareRobot(schema, args.hardware_config, args.enable_hardware)
    else:
        robot = MockRobot(schema)
    source = controls = None
    try:
        if args.command == "run-policy":
            from .lerobot_io import CheckpointPolicy
            source = CheckpointPolicy(args.checkpoint, schema, metadata_for(robot, schema, args.task, "policy")["contract"],
                                      args.task, args.device, args.allow_synthetic)
        elif robot.synthetic:
            source = MockSource(schema)
        else:
            from .quest import QuestSource
            source = QuestSource(schema, robot.cfg)
            print(f"Quest ingress listening at {source.port}; the owned Quest web publisher is starting.")
        robot.connect()
        if args.command == "record" and args.mock_episodes:
            paths = record_mock(robot, source, schema, args.raw_root, args.task, args.mock_episodes, args.frames, not args.fast)
            print(json.dumps({"synthetic": True, "episodes": paths}, indent=2))
        else:
            controls = ControlInput(args.pedal_device)
            controls.start()
            interactive_session(robot, source, schema, getattr(args, "raw_root", None), args.task,
                                controls, execution=args.command == "run-policy")
    finally:
        # Attempt all cleanup steps, even if one background component fails.
        for component in (controls, source, robot):
            if component is not None:
                try:
                    component.close()
                except Exception as exc:
                    print(f"Cleanup warning: {exc}", file=sys.stderr)
