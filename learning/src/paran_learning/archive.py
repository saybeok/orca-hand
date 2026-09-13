"""Append-only raw episodes with a bounded writer and explicit rejection status."""
from __future__ import annotations

import hashlib
import json
import queue
import threading
import time
import uuid
from pathlib import Path
import numpy as np

from .core import CAMERAS, fingerprint, read_json, write_json


class EpisodeWriter:
    def __init__(self, root, metadata, capacity=8):
        self.path = Path(root) / (time.strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:10])
        self.path.mkdir(parents=True, exist_ok=False)
        self.metadata = {**metadata, "format_version": 2, "status": "incomplete",
                         "created_unix_ns": time.time_ns(), "frames": []}
        write_json(self.path / "manifest.json", self.metadata)
        self.queue = queue.Queue(maxsize=capacity)
        self.error = None
        self.count = 0
        self.closed = False
        self.thread = threading.Thread(target=self._run, name="raw-episode-writer", daemon=True)
        self.thread.start()

    def _write_frame(self, index, arrays, timing):
        name = f"frame_{index:06d}.npz"
        path = self.path / name
        # Images and all vectors are stored without pickle, in their native units.
        with path.open("xb") as stream:
            # Also embed timing in each immutable frame so a process crash
            # before manifest finalization does not erase the timing evidence.
            np.savez(stream, **arrays, timing_json=np.asarray(json.dumps(timing, allow_nan=False)))
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        self.metadata["frames"].append({"file": name, "sha256": digest, **timing})

    def _run(self):
        while True:
            item = self.queue.get()
            try:
                if item is None:
                    return
                if self.error is None:
                    self._write_frame(*item)
            except Exception as exc:
                self.error = exc
            finally:
                self.queue.task_done()

    def check(self):
        if self.error is not None:
            raise RuntimeError(f"Raw writer failed: {self.error}") from self.error
        if self.closed:
            raise RuntimeError("Episode is already closed")

    def append(self, observation, proposed, accepted, decision_ns, dispatch_started_ns,
               dispatch_finished_ns, frame_index):
        self.check()
        arrays = {"state": observation.state.copy().astype(np.float32),
                  "action": np.asarray(accepted, dtype=np.float32).copy(),
                  "proposed_action": np.asarray(proposed.position, dtype=np.float32).copy()}
        arrays.update({f"image_{k}": v.image.copy() for k, v in observation.cameras.items()})
        timing = {"frame_index": frame_index, "decision_ns": decision_ns,
                  "state_read_started_ns": observation.read_started_ns,
                  "state_read_finished_ns": observation.read_finished_ns,
                  "command_received_ns": proposed.received_ns,
                  "source_timestamp_ns": proposed.source_timestamp_ns,
                  "dispatch_started_ns": dispatch_started_ns,
                  "dispatch_finished_ns": dispatch_finished_ns,
                  "camera_received_ns": {k: v.received_ns for k, v in observation.cameras.items()},
                  "camera_sequence": {k: v.sequence for k, v in observation.cameras.items()},
                  "selected_vr_sample": proposed.raw,
                  "command_clipped": bool(not np.allclose(proposed.position, accepted, atol=1e-7))}
        try:
            self.queue.put_nowait((self.count, arrays, timing))
        except queue.Full as exc:
            self.error = RuntimeError("Raw writer queue overflow: episode must be rejected")
            raise self.error from exc
        self.count += 1

    def finish(self, status="accepted", reason=""):
        if self.closed:
            return self.path
        if status not in ("accepted", "rejected", "incomplete"):
            raise ValueError("Unknown episode status")
        self.closed = True
        try:
            self.queue.put(None, timeout=10)
            self.thread.join(timeout=30)
            if self.thread.is_alive():
                raise RuntimeError("Writer did not finish; raw episode left incomplete")
        except Exception:
            # Initial manifest remains incomplete and cannot be exported.
            raise
        if self.error is not None or not self.count:
            status = "rejected"
            reason = str(self.error or "empty episode")
        self.metadata.update(status=status, reason=reason, completed_unix_ns=time.time_ns(),
                             frame_count=len(self.metadata["frames"]))
        write_json(self.path / "manifest.json", self.metadata)
        return self.path


def accepted_episodes(root, allow_synthetic=False):
    episodes = []
    for path in sorted(Path(root).glob("*/manifest.json")):
        data = read_json(path)
        if data["status"] != "accepted":
            continue
        if data.get("synthetic") and not allow_synthetic:
            raise ValueError("Synthetic data is present; use a separate root or --allow-synthetic for tests")
        if not data["frames"] or data["frame_count"] != len(data["frames"]):
            raise ValueError(f"Invalid frame count: {path}")
        episodes.append((path.parent, data))
    if not episodes:
        raise ValueError("No accepted episodes")
    expected = fingerprint(episodes[0][1]["contract"])
    for path, meta in episodes:
        if fingerprint(meta["contract"]) != expected:
            raise ValueError(f"Mixed robot/camera/calibration contracts: {path}")
    return episodes


def frames(path, metadata):
    for index, row in enumerate(metadata["frames"]):
        name = row["file"]
        if name != f"frame_{index:06d}.npz" or row["frame_index"] != index:
            raise ValueError("Non-contiguous or unsafe raw frame path")
        filename = path / name
        if hashlib.sha256(filename.read_bytes()).hexdigest() != row["sha256"]:
            raise ValueError(f"Corrupt frame: {filename}")
        with np.load(filename, allow_pickle=False) as data:
            if metadata.get("format_version", 1) >= 2:
                embedded = json.loads(str(data["timing_json"]))
                if embedded != {k: v for k, v in row.items() if k not in ("file", "sha256")}:
                    raise ValueError("Manifest timing differs from immutable frame timing")
            yield {k: data[k] for k in data.files}, row


def make_split(episodes, seed=42):
    """Stable episode-level split; never split adjacent frames across sets."""
    if len(episodes) < 3:
        raise ValueError("At least 3 accepted episodes required for train/validation/test")
    order = np.random.default_rng(seed).permutation(len(episodes)).tolist()
    held_out = max(1, round(len(order) * .1))
    return {"train": sorted(order[2 * held_out:]),
            "validation": sorted(order[:held_out]), "test": sorted(order[held_out:2 * held_out])}


def vector_statistics(episodes, indices):
    values = {"observation.state": [], "action": []}
    for i in indices:
        path, metadata = episodes[i]
        for data, _ in frames(path, metadata):
            values["observation.state"].append(data["state"])
            values["action"].append(data["action"])
    result = {}
    for key, rows in values.items():
        array = np.stack(rows).astype(np.float64)
        if not np.isfinite(array).all():
            raise ValueError("Non-finite training statistics")
        result[key] = {"mean": array.mean(0).tolist(), "std": array.std(0).tolist(),
                       "min": array.min(0).tolist(), "max": array.max(0).tolist(),
                       "q01": np.quantile(array, .01, axis=0).tolist(),
                       "q99": np.quantile(array, .99, axis=0).tolist(), "count": [len(array)]}
    return result
