"""LeRobot v0.4.4 dataset / training / checkpoint boundary (lazy ML imports)."""
from __future__ import annotations
import importlib.metadata
from pathlib import Path
import subprocess
import sys
import time
import numpy as np

from .archive import accepted_episodes, frames, make_split, vector_statistics
from .core import CAMERAS, Command, fingerprint, read_json, write_json

LEROBOT_VERSION = "0.4.4"


def require_lerobot():
    try:
        version = importlib.metadata.version("lerobot")
    except importlib.metadata.PackageNotFoundError as exc:
        raise RuntimeError("Install the learning extra in a separate training environment; see learning/README.md") from exc
    if version != LEROBOT_VERSION:
        raise RuntimeError(f"Expected LeRobot {LEROBOT_VERSION}, found {version}; run compatibility tests before upgrading")


def inspect_raw(root, allow_synthetic=False, seed=42):
    episodes = accepted_episodes(root, allow_synthetic)
    split = make_split(episodes, seed)
    contract = episodes[0][1]["contract"]
    n = len(contract["joint_names"])
    count = 0
    for path, metadata in episodes:
        previous = None
        sequences = {}
        for data, row in frames(path, metadata):
            for key in ("state", "action", "proposed_action"):
                if data[key].shape != (n,) or not np.isfinite(data[key]).all():
                    raise ValueError(f"Invalid {key} in {path}")
            for camera in CAMERAS:
                value = data[f"image_{camera}"]
                if value.dtype != np.uint8 or list(value.shape) != contract["image_shapes"][camera]:
                    raise ValueError(f"Invalid {camera} frame in {path}")
                if row["camera_sequence"][camera] <= sequences.get(camera, -1):
                    raise ValueError("Repeated camera frame in an accepted archive")
            sequences = row["camera_sequence"]
            tick = row["decision_ns"]
            if previous is not None:
                interval = (tick - previous) / 1e9
                if interval <= 0:
                    raise ValueError("Reversed timestamp")
                if not metadata.get("synthetic") and abs(interval - 1 / contract["fps"]) > .04:
                    raise ValueError("Irregular real-data timing; do not silently resample across a gap")
            if not row["state_read_finished_ns"] <= tick <= row["dispatch_started_ns"] <= row["dispatch_finished_ns"]:
                raise ValueError("Invalid observation/action ordering")
            previous = tick
            count += 1
    return {"status": "validated", "contract": contract, "contract_sha256": fingerprint(contract),
            "episodes": [str(path.resolve()) for path, _ in episodes], "frames": count,
            "synthetic": any(meta.get("synthetic") for _, meta in episodes),
            "split": split, "seed": seed, "train_vector_stats": vector_statistics(episodes, split["train"])}


def export_dataset(root, destination, allow_synthetic=False, seed=42, dataset_class=None):
    destination = Path(destination).resolve()
    if destination.exists():
        raise FileExistsError("Export destination must be new; existing datasets are never overwritten")
    report = inspect_raw(root, allow_synthetic, seed)
    episodes = accepted_episodes(root, allow_synthetic)
    if dataset_class is None:
        require_lerobot()
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        dataset_class = LeRobotDataset
    contract = report["contract"]
    names = contract["joint_names"]
    features = {key: {"dtype": "float32", "shape": (len(names),), "names": names}
                for key in ("observation.state", "action")}
    features.update({f"observation.images.{k}": {"dtype": "video", "shape": tuple(contract["image_shapes"][k]),
                                                "names": ["height", "width", "channels"]} for k in CAMERAS})
    destination.mkdir(parents=True, exist_ok=False)
    report["status"] = "exporting"
    report["datasets"] = {}
    write_json(destination / "paran_manifest.json", report)
    # Separate datasets eliminate normalization leakage into the training split.
    for split, indices in report["split"].items():
        repo_id = f"local/paran_{split}"
        split_root = destination / split
        dataset = dataset_class.create(repo_id=repo_id, root=split_root, fps=contract["fps"],
                                       features=features, robot_type="am200_3dof_orca17", use_videos=True,
                                       image_writer_threads=0, vcodec="h264")
        try:
            for index in indices:
                path, metadata = episodes[index]
                for data, _ in frames(path, metadata):
                    frame = {"observation.state": data["state"], "action": data["action"], "task": metadata["task"]}
                    frame.update({f"observation.images.{k}": data[f"image_{k}"] for k in CAMERAS})
                    dataset.add_frame(frame)
                dataset.save_episode(parallel_encoding=False)
        finally:
            dataset.finalize()  # v3 parquet footers are required before reload.
        stats_path = split_root / "meta" / "stats.json"
        stats = read_json(stats_path)
        stats.update(vector_statistics(episodes, indices))  # Includes pi05 q01/q99.
        write_json(stats_path, stats)
        report["datasets"][split] = {"repo_id": repo_id, "root": str(split_root), "raw_indices": indices}
    report["status"] = "complete"
    report["lerobot_version"] = LEROBOT_VERSION
    write_json(destination / "paran_manifest.json", report)
    return report


def training_command(export_root, model, output, device="cuda", steps=20000, batch_size=None,
                     checkpoint="lerobot/pi05_base", allow_synthetic=False):
    manifest = read_json(Path(export_root) / "paran_manifest.json")
    if manifest["status"] != "complete":
        raise ValueError("Dataset export did not complete")
    if manifest["synthetic"] and not allow_synthetic:
        raise ValueError("Synthetic training is test-only; pass --allow-synthetic explicitly")
    if model not in ("act", "pi05") or steps < 1:
        raise ValueError("Use act or pi05 and a positive step count")
    if Path(output).exists():
        raise FileExistsError("Choose a new training output directory")
    n = len(manifest["contract"]["joint_names"])
    if model == "pi05" and n > 32:
        raise ValueError("pi05 base max_action_dim=32 cannot represent this schema")
    dataset = manifest["datasets"]["train"]
    command = [sys.executable, "-m", "lerobot.scripts.lerobot_train",
               f"--dataset.repo_id={dataset['repo_id']}", f"--dataset.root={dataset['root']}",
               f"--policy.type={model}", f"--policy.device={device}", "--policy.push_to_hub=false",
               f"--output_dir={Path(output).resolve()}", f"--steps={steps}",
               f"--batch_size={batch_size or (8 if model == 'act' else 1)}",
               "--num_workers=0", "--wandb.enable=false", "--eval_freq=0", "--seed=42",
               f"--save_freq={min(5000, steps)}", "--policy.chunk_size=50", "--policy.n_action_steps=10"]
    if model == "pi05":
        # type + pretrained_path builds features from OUR dataset instead of
        # silently inheriting another robot's camera and state feature schema.
        command.extend([f"--policy.pretrained_path={checkpoint}", "--policy.dtype=bfloat16",
                        "--policy.gradient_checkpointing=true", "--policy.max_state_dim=32",
                        "--policy.max_action_dim=32"])
    return command, manifest


def evaluate_offline(export_root, checkpoint, output, split="validation", device="cpu", max_batches=100,
                     allow_synthetic=False):
    """Held-out offline model loss only; not a physical grasp-success metric."""
    if split not in ("validation", "test") or max_batches < 1:
        raise ValueError("Use a held-out split and a positive batch limit")
    output = Path(output)
    if output.exists():
        raise FileExistsError("Evaluation report already exists")
    manifest = read_json(Path(export_root) / "paran_manifest.json")
    if manifest["status"] != "complete":
        raise ValueError("Dataset export is incomplete")
    from .core import Schema
    names = tuple(manifest["contract"]["joint_names"])
    # Evaluation only needs the dimension/order validator, not actuator limits.
    schema = Schema(names, np.full(len(names), -np.inf), np.full(len(names), np.inf))
    adapter = CheckpointPolicy(checkpoint, schema, manifest["contract"], "evaluation", device, allow_synthetic)
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    info = manifest["datasets"][split]
    fps = manifest["contract"]["fps"]
    deltas = {"action": [i / fps for i in adapter.policy.config.action_delta_indices]}
    dataset = LeRobotDataset(repo_id=info["repo_id"], root=info["root"], delta_timestamps=deltas)
    loader = adapter.torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    losses = []
    with adapter.torch.inference_mode():
        for index, batch in enumerate(loader):
            if index >= max_batches:
                break
            loss, _ = adapter.policy.forward(adapter.pre(batch))
            value = float(loss.detach().cpu())
            if not np.isfinite(value):
                raise RuntimeError("Non-finite held-out loss")
            losses.append(value)
    if not losses:
        raise ValueError("No held-out batches")
    report = {"split": split, "batches": len(losses), "mean_model_loss": float(np.mean(losses)),
              "checkpoint": str(Path(checkpoint).resolve()), "synthetic": manifest["synthetic"],
              "note": "Model-specific normalized loss; do not compare ACT vs pi05 loss as grasp success."}
    write_json(output, report)
    return report


def execute_training(command, manifest, output):
    require_lerobot()
    # No shell, hub uploads, auto-resume or credentials stored in provenance.
    subprocess.run(command, check=True)
    checkpoints = list(Path(output).glob("checkpoints/*/pretrained_model"))
    if not checkpoints:
        raise RuntimeError("Training returned without a saved checkpoint")
    contract = {"contract": manifest["contract"], "contract_sha256": manifest["contract_sha256"],
                "synthetic": manifest["synthetic"], "lerobot_version": LEROBOT_VERSION,
                "dataset_split": manifest["split"], "train_vector_stats": manifest["train_vector_stats"]}
    for checkpoint in checkpoints:
        write_json(checkpoint / "paran_contract.json", contract)
    write_json(Path(output) / "paran_run.json", {**contract, "command": command})


class CheckpointPolicy:
    """Saved processors + select_action's own chunk queue, never arbitrary [0,0]."""
    def __init__(self, checkpoint, schema, contract, task, device="cpu", allow_synthetic=False):
        require_lerobot()
        import torch
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.policies.factory import get_policy_class, make_pre_post_processors
        checkpoint = Path(checkpoint).resolve()
        saved = read_json(checkpoint / "paran_contract.json")
        if fingerprint(contract) != saved["contract_sha256"] or saved["contract"] != contract:
            raise ValueError("Checkpoint and live robot/cameras/calibration contracts differ")
        if saved["synthetic"] and not allow_synthetic:
            raise ValueError("Synthetic-trained checkpoints are forbidden on real hardware")
        for filename in ("policy_preprocessor.json", "policy_postprocessor.json"):
            if not (checkpoint / filename).is_file():
                raise ValueError(f"Saved normalization processor missing: {filename}")
        config = PreTrainedConfig.from_pretrained(str(checkpoint))
        if config.type not in ("act", "pi05"):
            raise ValueError("Only ACT and pi05 are supported")
        expected_images = {f"observation.images.{k}" for k in CAMERAS}
        actual_images = {k for k in config.input_features if k.startswith("observation.images.")}
        if actual_images != expected_images or tuple(config.input_features["observation.state"].shape) != (schema.size,):
            raise ValueError("Checkpoint input schema mismatch; no image fallback or state padding allowed")
        if tuple(config.output_features["action"].shape) != (schema.size,):
            raise ValueError("Checkpoint output schema mismatch")
        config.device = device
        self.policy = get_policy_class(config.type).from_pretrained(str(checkpoint), config=config)
        self.policy.to(device).eval()
        self.pre, self.post = make_pre_post_processors(
            policy_cfg=config, pretrained_path=str(checkpoint),
            preprocessor_overrides={"device_processor": {"device": device}})
        self.torch, self.schema, self.task = torch, schema, task
        self.expected_shapes = contract["image_shapes"]
        self.policy.reset()

    def reset(self, state):
        self.policy.reset()
        for processor in (self.pre, self.post):
            if hasattr(processor, "reset"):
                processor.reset()

    def prepare(self, observation):
        self.latest(observation)
        self.reset(observation.state)

    def on_ready(self):
        self.reset(None)

    def latest(self, observation):
        torch = self.torch
        if set(observation.cameras) != set(CAMERAS):
            raise ValueError("Missing camera")
        batch = {"observation.state": torch.from_numpy(self.schema.vector(observation.state).astype(np.float32)),
                 "task": self.task}
        for camera, sample in observation.cameras.items():
            if list(sample.image.shape) != self.expected_shapes[camera] or sample.image.dtype != np.uint8:
                raise ValueError(f"Invalid live image {camera}")
            batch[f"observation.images.{camera}"] = torch.from_numpy(sample.image.copy()).permute(2, 0, 1).float() / 255
        started = time.perf_counter_ns()
        with torch.inference_mode():
            action = self.post(self.policy.select_action(self.pre(batch)))
        array = action.detach().cpu().numpy()
        if array.shape != (1, self.schema.size):
            raise ValueError(f"Expected one batched joint action, not an unconsumed chunk: {array.shape}")
        return Command(self.schema.vector(array[0]), started, raw={"kind": "policy"})
