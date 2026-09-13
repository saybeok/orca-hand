"""No robot, network, GPU or LeRobot installation required for these tests."""
import ast
from dataclasses import replace
import json
import logging
from pathlib import Path
import sys
import tempfile
import threading
import types
import unittest
from unittest.mock import Mock
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "learning" / "src"))

from paran_learning.core import Schema, SafetyFilter, Synchronizer, Command, CameraSample, Observation, CAMERAS, read_json, write_json
from paran_learning.backends import MockRobot, MockSource, hardware_preflight
from paran_learning.controls import PedalDecoder, EpisodeControls, State
from paran_learning.archive import EpisodeWriter, accepted_episodes, frames, make_split
from paran_learning.runner import record_mock, tick, metadata_for
from paran_learning.lerobot_io import inspect_raw, export_dataset, training_command
from paran_learning.lerobot_io import CheckpointPolicy

SCHEMA = Schema.load(REPO / "orca_am200" / "robot_model" / "joint_map.json")


def sample(now=1_000_000_000, seq=1):
    cams = {k: CameraSample(np.zeros((2, 3, 3), np.uint8), seq, now - 1_000) for k in CAMERAS}
    return Observation(np.zeros(20), now - 2_000, now - 1_000, cams), Command(np.zeros(20), now - 1_000)


class CoreTests(unittest.TestCase):
    def test_current_robot_is_20_not_22(self):
        self.assertEqual(SCHEMA.size, 20)
        self.assertEqual(SCHEMA.names[:3], ("arm.shoulder_pan", "arm.shoulder_lift", "arm.elbow_flex"))
        self.assertEqual(SCHEMA.names[-1], "hand.wrist")

    def test_strict_vector_no_padding(self):
        for bad in (np.zeros(17), np.zeros(22), np.zeros((1, 20)), np.full(20, np.nan)):
            with self.assertRaises(ValueError):
                SCHEMA.vector(bad)

    def test_velocity_limit_and_commit(self):
        robot = MockRobot(SCHEMA)
        safety = robot.safety
        safety.reset(robot.state)
        action = safety.apply(np.ones(20) * 10, robot.state)
        self.assertLessEqual(np.max(abs(action - robot.state)), .6 / 15 + 1e-12)
        np.testing.assert_array_equal(safety.last, robot.state)
        safety.commit(action)
        np.testing.assert_array_equal(safety.last, action)

    def test_tracking_error_stops(self):
        robot = MockRobot(SCHEMA)
        robot.safety.reset(robot.state)
        with self.assertRaises(RuntimeError):
            robot.safety.apply(robot.state, robot.state + 2)

    def test_invalid_limits(self):
        with self.assertRaises(ValueError):
            SafetyFilter(SCHEMA, SCHEMA.lower - 1, SCHEMA.upper, np.ones(20), 15, np.ones(20))

    def test_no_implicit_initial_target(self):
        with self.assertRaises(RuntimeError):
            MockRobot(SCHEMA).safety.apply(np.zeros(20), np.zeros(20))

    def test_sync_valid(self):
        observation, command = sample()
        Synchronizer(SCHEMA).validate(observation, command, 1_000_000_000)

    def test_stale_command(self):
        observation, command = sample()
        with self.assertRaises(RuntimeError):
            Synchronizer(SCHEMA).validate(observation, replace(command, received_ns=1), 1_000_000_000)

    def test_future_command(self):
        observation, command = sample()
        with self.assertRaises(RuntimeError):
            Synchronizer(SCHEMA).validate(observation, replace(command, received_ns=2_000_000_000), 1_000_000_000)

    def test_missing_camera_no_substitution(self):
        observation, command = sample()
        del observation.cameras["wrist"]
        with self.assertRaises(ValueError):
            Synchronizer(SCHEMA).validate(observation, command, 1_000_000_000)

    def test_duplicate_camera(self):
        sync = Synchronizer(SCHEMA)
        sync.validate(*sample(), 1_000_000_000)
        with self.assertRaises(RuntimeError):
            sync.validate(*sample(now=1_010_000_000), 1_010_000_000)

    def test_camera_skew(self):
        observation, command = sample()
        observation.cameras["top"] = replace(observation.cameras["top"], received_ns=930_000_000)
        with self.assertRaises(RuntimeError):
            Synchronizer(SCHEMA).validate(observation, command, 1_000_000_000)

    def test_non_rgb_camera(self):
        observation, command = sample()
        observation.cameras["top"] = replace(observation.cameras["top"], image=np.zeros((2, 3), np.uint8))
        with self.assertRaises(ValueError):
            Synchronizer(SCHEMA).validate(observation, command, 1_000_000_000)

    def test_image_shape_cannot_change_mid_episode(self):
        sync = Synchronizer(SCHEMA)
        sync.validate(*sample(), 1_000_000_000)
        observation, command = sample(now=1_010_000_000, seq=2)
        observation.cameras["top"] = replace(observation.cameras["top"], image=np.zeros((4, 6, 3), np.uint8))
        with self.assertRaises(ValueError):
            sync.validate(observation, command, 1_010_000_000)

    def test_policy_warmup_queue_is_reset(self):
        adapter = CheckpointPolicy.__new__(CheckpointPolicy)
        adapter.policy = Mock()
        adapter.pre, adapter.post = Mock(), Mock()
        adapter.on_ready()
        adapter.policy.reset.assert_called_once()
        adapter.pre.reset.assert_called_once()
        adapter.post.reset.assert_called_once()

    def test_template_stays_locked(self):
        with self.assertRaisesRegex(ValueError, "Hardware locked"):
            hardware_preflight(REPO / "configs" / "learning" / "hardware.template.json", SCHEMA, True)


class ControlTests(unittest.TestCase):
    def test_mapping(self):
        decoder = PedalDecoder()
        self.assertEqual(decoder.feed("KEY_C", 1, 1), "advance")
        self.assertEqual(decoder.feed("KEY_A", 1, 1), "reject")

    def test_repeat_release_bounce_ignored(self):
        decoder = PedalDecoder()
        decoder.feed("KEY_C", 1, 1)
        for value, now in ((2, 1.1), (0, 1.2), (1, 1.29)):
            self.assertIsNone(decoder.feed("KEY_C", value, now))
        self.assertEqual(decoder.feed("KEY_C", 1, 1.4), "advance")

    def test_doubletap(self):
        decoder = PedalDecoder()
        self.assertIsNone(decoder.feed("KEY_B", 1, 1))
        self.assertEqual(decoder.feed("KEY_B", 1, 1.5), "stop")

    def test_doubletap_timeout(self):
        decoder = PedalDecoder()
        decoder.feed("KEY_B", 1, 1)
        self.assertIsNone(decoder.feed("KEY_B", 1, 4))

    def test_other_pedal_cancels_confirmation(self):
        decoder = PedalDecoder()
        decoder.feed("KEY_B", 1, 1)
        decoder.feed("KEY_A", 1, 1.4)
        self.assertIsNone(decoder.feed("KEY_B", 1, 1.8))

    def test_save_requires_new_start(self):
        control = EpisodeControls()
        self.assertEqual(control.handle("advance"), "start")
        self.assertEqual(control.handle("advance"), "save")
        self.assertEqual(control.state, State.RESETTING)
        self.assertIsNone(control.handle(None))
        self.assertEqual(control.handle("advance"), "start")

    def test_reject_and_preview(self):
        control = EpisodeControls()
        control.handle("advance")
        self.assertEqual(control.handle("reject"), "reject")
        self.assertEqual(control.handle("reject"), "preview")
        self.assertEqual(control.handle("reject"), "hold")
        self.assertEqual(control.handle("stop"), "stop")


class FakeDataset:
    """Test double for the inspected LeRobot v0.4.4 public dataset API."""
    instances = []

    @classmethod
    def create(cls, **kwargs):
        self = cls()
        self.kwargs = kwargs
        self.root = Path(kwargs["root"])
        (self.root / "meta").mkdir(parents=True)
        self.rows = []
        self.saved = 0
        self.finalized = False
        cls.instances.append(self)
        return self

    def add_frame(self, frame):
        self.rows.append(frame)

    def save_episode(self, parallel_encoding):
        self.saved += 1

    def finalize(self):
        self.finalized = True
        write_json(self.root / "meta" / "stats.json", {})


class ArchiveTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="paran-tests-")
        self.root = Path(self.temp.name)
        self.robot = MockRobot(SCHEMA)
        self.source = MockSource(SCHEMA)

    def tearDown(self):
        self.temp.cleanup()

    def populate(self):
        return record_mock(self.robot, self.source, SCHEMA, self.root / "raw", "lift tile", 3, 8, False)

    def test_mock_record_inspect(self):
        self.populate()
        report = inspect_raw(self.root / "raw", True)
        self.assertEqual(report["frames"], 24)
        self.assertEqual(report["contract"]["units"], "model_radians")
        self.assertEqual(len(report["train_vector_stats"]["action"]["q01"]), 20)

    def test_state_precedes_action_and_accepted_is_limited(self):
        self.populate()
        path, meta = accepted_episodes(self.root / "raw", True)[0]
        data, timing = next(frames(path, meta))
        self.assertFalse(np.array_equal(data["state"], data["action"]))
        self.assertTrue(timing["command_clipped"])
        self.assertLessEqual(timing["state_read_finished_ns"], timing["dispatch_started_ns"])

    def test_synthetic_never_enters_real_dataset_implicitly(self):
        self.populate()
        with self.assertRaises(ValueError):
            accepted_episodes(self.root / "raw")

    def test_train_val_test_disjoint(self):
        self.populate()
        episodes = accepted_episodes(self.root / "raw", True)
        split = make_split(episodes)
        sets = [set(x) for x in split.values()]
        self.assertEqual(set.union(*sets), {0, 1, 2})
        self.assertTrue(sets[0].isdisjoint(sets[1]) and sets[0].isdisjoint(sets[2]))
        self.assertEqual(split, make_split(episodes))

    def test_too_few_episodes_rejected(self):
        with self.assertRaises(ValueError):
            make_split([1, 2])

    def test_raw_checksum(self):
        self.populate()
        path, meta = accepted_episodes(self.root / "raw", True)[0]
        with (path / meta["frames"][0]["file"]).open("ab") as stream:
            stream.write(b"corruption")
        with self.assertRaisesRegex(ValueError, "Corrupt"):
            list(frames(path, meta))

    def test_empty_episode_rejected(self):
        writer = EpisodeWriter(self.root, metadata_for(self.robot, SCHEMA, "tile", "mock"))
        path = writer.finish()
        self.assertEqual(read_json(path / "manifest.json")["status"], "rejected")

    def test_incomplete_preserved_not_accepted(self):
        writer = EpisodeWriter(self.root, metadata_for(self.robot, SCHEMA, "tile", "mock"))
        self.source.reset(self.robot.state)
        self.robot.safety.reset(self.robot.state)
        tick(self.robot, self.source, SCHEMA, Synchronizer(SCHEMA), writer)
        path = writer.finish("incomplete", "operator quit")
        self.assertEqual(read_json(path / "manifest.json")["status"], "incomplete")
        self.assertTrue((path / "frame_000000.npz").exists())

    def test_writer_failure_cannot_be_saved_as_success(self):
        writer = EpisodeWriter(self.root, metadata_for(self.robot, SCHEMA, "tile", "mock"))
        writer._write_frame = Mock(side_effect=OSError("disk failure"))
        self.source.reset(self.robot.state)
        self.robot.safety.reset(self.robot.state)
        tick(self.robot, self.source, SCHEMA, Synchronizer(SCHEMA), writer)
        path = writer.finish()
        self.assertEqual(read_json(path / "manifest.json")["status"], "rejected")

    def test_backend_target_mutation_rejected(self):
        self.source.reset(self.robot.state)
        self.robot.safety.reset(self.robot.state)
        self.robot.dispatch = lambda value: value + .1
        with self.assertRaisesRegex(RuntimeError, "Backend changed"):
            tick(self.robot, self.source, SCHEMA, Synchronizer(SCHEMA))

    def test_failed_write_does_not_commit_target(self):
        self.source.reset(self.robot.state)
        self.robot.safety.reset(self.robot.state)
        previous = self.robot.safety.last.copy()
        self.robot.dispatch = Mock(side_effect=OSError("serial failure"))
        with self.assertRaises(OSError):
            tick(self.robot, self.source, SCHEMA, Synchronizer(SCHEMA))
        np.testing.assert_array_equal(self.robot.safety.last, previous)

    def test_queue_overflow_rejects_without_silent_drop(self):
        writer = EpisodeWriter(self.root, metadata_for(self.robot, SCHEMA, "tile", "mock"), capacity=1)
        entered, release = threading.Event(), threading.Event()
        original = writer._write_frame
        def blocked(*args):
            entered.set()
            release.wait(timeout=3)
            original(*args)
        writer._write_frame = blocked
        self.source.reset(self.robot.state)
        self.robot.safety.reset(self.robot.state)
        sync = Synchronizer(SCHEMA)
        try:
            tick(self.robot, self.source, SCHEMA, sync, writer)
            self.assertTrue(entered.wait(timeout=1))
            tick(self.robot, self.source, SCHEMA, sync, writer, 1)
            with self.assertRaisesRegex(RuntimeError, "overflow"):
                tick(self.robot, self.source, SCHEMA, sync, writer, 2)
        finally:
            release.set()
            path = writer.finish()
        self.assertEqual(read_json(path / "manifest.json")["status"], "rejected")

    def test_export_common_schema_and_statistics(self):
        self.populate()
        FakeDataset.instances.clear()
        report = export_dataset(self.root / "raw", self.root / "dataset", True, dataset_class=FakeDataset)
        self.assertEqual(report["status"], "complete")
        self.assertEqual(len(FakeDataset.instances), 3)
        for dataset in FakeDataset.instances:
            self.assertTrue(dataset.finalized)
            self.assertEqual(dataset.kwargs["features"]["action"]["shape"], (20,))
            self.assertEqual(set(dataset.rows[0]), {"observation.state", "action", "task"} | {f"observation.images.{k}" for k in CAMERAS})
        stats = read_json(self.root / "dataset" / "train" / "meta" / "stats.json")
        self.assertEqual(stats["action"]["count"], [8])  # train only, not all 24 rows
        self.assertEqual(stats["action"], report["train_vector_stats"]["action"])

    def test_both_training_recipes_use_same_train_split(self):
        self.populate()
        export_dataset(self.root / "raw", self.root / "dataset", True, dataset_class=FakeDataset)
        act, _ = training_command(self.root / "dataset", "act", self.root / "act", allow_synthetic=True)
        pi, _ = training_command(self.root / "dataset", "pi05", self.root / "pi", allow_synthetic=True)
        self.assertEqual([x for x in act if x.startswith("--dataset")], [x for x in pi if x.startswith("--dataset")])
        self.assertIn("--policy.push_to_hub=false", act)
        self.assertIn("--policy.pretrained_path=lerobot/pi05_base", pi)
        self.assertNotIn("--policy.path=lerobot/pi05_base", pi)

    def test_export_never_overwrites(self):
        with self.assertRaises(FileExistsError):
            export_dataset(self.root, self.root, dataset_class=FakeDataset)


def isolated_method(file, class_name, method, namespace):
    """Exercise changed driver methods without importing/opening hardware stacks."""
    tree = ast.parse(file.read_text(encoding="utf-8"))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name)
    function = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == method)
    function.decorator_list = []
    module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), function], type_ignores=[])
    ast.fix_missing_locations(module)
    exec(compile(module, str(file), "exec"), namespace)
    return namespace[method]


class DriverRegressionTests(unittest.TestCase):
    def sync_method(self):
        return isolated_method(REPO / "orca_core/orca_core/hardware/feetech_client.py", "FeetechClient", "write_positions_sync",
                               {"logging": logging, "COMM_SUCCESS": 0})

    def client(self, result, strict=True):
        return types.SimpleNamespace(_check_connected=Mock(), _bus_lock=threading.Lock(), _default_speed=1,
            _default_acc=1, _default_torque=1, pos_scale=1, strict_io=strict, _rad_to_raw=lambda x, _: int(x),
            _clamp_position=lambda x: min(4095, max(0, x)), _flush_input_buffer=Mock(),
            packet_handler=types.SimpleNamespace(SyncWritePosEx=Mock(), groupSyncWrite=types.SimpleNamespace(
                clearParam=Mock(), txPacket=Mock(return_value=result))))

    def test_strict_serial_error_raises(self):
        with self.assertRaises(RuntimeError):
            self.sync_method()(self.client(-1), [1], np.array([12.]))

    def test_legacy_serial_behavior_preserved(self):
        self.sync_method()(self.client(-1, strict=False), [1], np.array([12.]))

    def test_motor_clamp_detected_before_transmission(self):
        client = self.client(0)
        with self.assertRaises(ValueError):
            self.sync_method()(client, [1], np.array([9999.]))
        client.packet_handler.groupSyncWrite.txPacket.assert_not_called()

    def test_cached_arm_read_rejected(self):
        method = isolated_method(REPO / "orca_teleop/src/orca_teleop/am200/arm_client.py", "AmArmFeetech", "read_motor_rad", {"np": np})
        driver = types.SimpleNamespace(_strict_io=True, _client=types.SimpleNamespace(
            read_position_velocity_current=lambda: types.SimpleNamespace(position=np.zeros(3)), last_read_ok=False))
        with self.assertRaises(RuntimeError):
            method(driver)

    def test_mode_change_cannot_enable_old_goal_in_strict_mode(self):
        method = isolated_method(REPO / "orca_core/orca_core/hardware/feetech_client.py", "FeetechClient", "set_operating_mode",
                                 {"logging": logging, "COMM_SUCCESS": 0, "SMS_STS_MODE": 33})
        driver = self.client(0)
        driver.set_torque_enabled = Mock(return_value=[])
        driver.packet_handler.write1ByteTxRx = Mock(return_value=(0, 0))
        method(driver, [1, 2], 5)
        driver.set_torque_enabled.assert_called_once_with([1, 2], False)

    def test_hand_boot_preloads_present_pose_before_enable(self):
        method = isolated_method(REPO / "orca_core/orca_core/hardware_hand.py", "OrcaHand", "init_joints", {})
        calls = []
        driver = types.SimpleNamespace(_motor_client=types.SimpleNamespace(strict_io=True, last_read_ok=True),
            calibrated=True, config=types.SimpleNamespace(control_mode="position", max_current=50, motor_ids=[1]),
            set_control_mode=lambda _: calls.append("mode"), set_max_current=lambda _: None,
            get_motor_pos=lambda: np.array([1.2]),
            write_motor_pos=lambda ids, q: calls.append("present_goal"),
            enable_torque=lambda: calls.append("enable") or [], _compute_wrap_offsets_dict=lambda: None)
        method(driver, move_to_neutral=False)
        self.assertEqual(calls, ["mode", "present_goal", "enable"])

    def test_persistent_arm_zero_not_replaced_by_current_pose(self):
        clients = []
        def factory(*args, **kwargs):
            client = types.SimpleNamespace(connect=Mock(), disconnect=Mock(), set_operating_mode=Mock(),
                read_position_velocity_current=lambda: types.SimpleNamespace(position=np.ones(3) * 1.2),
                last_read_ok=True, write_desired_pos=Mock(), set_torque_enabled=Mock(return_value=[]))
            clients.append(client)
            return client
        method = isolated_method(REPO / "orca_teleop/src/orca_teleop/am200/arm_client.py", "AmArmFeetech", "connect",
                                 {"np": np, "FeetechClient": factory, "CURRENT_BASED_POSITION": 5, "logger": logging})
        driver = types.SimpleNamespace(_cfg={"port": "FAKE", "motor_zero_rad": [1., 1., 1.],
                                             "boot_model_limits_rad": [[-.5, .5]] * 3},
            _motor_ids=[1, 2, 3], _strict_io=True, _signs=np.ones(3), _speed=1, _max_torque=1)
        method(driver)
        np.testing.assert_array_equal(driver._motor_home, np.ones(3))
        clients[0].write_desired_pos.assert_called_once()


if __name__ == "__main__":
    unittest.main()
