"""Focus-independent PCsensor controls based on lerobot-footpedal's mapping.

Right KEY_C -> n, left KEY_A -> r, center KEY_B double-tap -> q.
Debounce/double-tap design attribution: ValentinJaeung (MIT), see THIRD_PARTY.
"""
from __future__ import annotations
from enum import Enum
import queue
import threading
import time


class State(str, Enum):
    READY = "ready"
    RECORDING = "recording"
    RESETTING = "resetting"
    PREVIEW = "preview"
    STOPPED = "stopped"


class EpisodeControls:
    def __init__(self):
        self.state = State.READY

    def handle(self, event):
        if event == "stop":
            self.state = State.STOPPED
            return "stop"
        if self.state == State.RECORDING:
            if event in ("advance", "reject"):
                self.state = State.RESETTING
                return "save" if event == "advance" else "reject"
        elif event == "advance" and self.state in (State.READY, State.RESETTING, State.PREVIEW):
            self.state = State.RECORDING
            return "start"
        elif event == "reject" and self.state in (State.READY, State.RESETTING):
            self.state = State.PREVIEW
            return "preview"
        elif event == "reject" and self.state == State.PREVIEW:
            self.state = State.RESETTING
            return "hold"
        return None


class PedalDecoder:
    def __init__(self, debounce=.3, minimum_gap=.15, double_tap_window=2):
        self.debounce = debounce
        self.minimum_gap = minimum_gap
        self.window = double_tap_window
        self.last = {}
        self.armed = None

    def feed(self, key, pressed, now):
        # Only key-down (1), not release (0) or repeat (2).
        if pressed != 1 or key not in ("KEY_A", "KEY_B", "KEY_C"):
            return None
        if now - self.last.get(key, float("-inf")) < self.debounce:
            return None
        self.last[key] = now
        if key != "KEY_B":
            self.armed = None
            return {"KEY_A": "reject", "KEY_C": "advance"}[key]
        if self.armed is not None and self.minimum_gap <= now - self.armed <= self.window:
            self.armed = None
            return "stop"
        self.armed = now
        return None


class ControlInput:
    """Direct evdev queue (no UInput injection or window-focus dependency)."""
    def __init__(self, pedal_path=None):
        self.events = queue.Queue(maxsize=32)
        self.stop_event = threading.Event()
        self.device = None
        self.error = None
        self.pedal_path = pedal_path

    def start(self):
        if self.pedal_path:
            import evdev
            self.device = evdev.InputDevice(self.pedal_path)
            self.device.grab()
        self.thread = threading.Thread(target=self._run, daemon=True, name="episode-controls")
        self.thread.start()

    def _emit(self, event):
        if event:
            self.events.put_nowait(event)

    def _run(self):
        try:
            if self.device:
                import select
                import evdev
                decoder = PedalDecoder()
                while not self.stop_event.is_set():
                    readable, _, _ = select.select([self.device.fd], [], [], .1)
                    if not readable:
                        continue
                    for event in self.device.read():
                        if event.type == evdev.ecodes.EV_KEY:
                            key = evdev.ecodes.KEY.get(event.code, "")
                            self._emit(decoder.feed(key, event.value, time.monotonic()))
            else:
                # Deliberately line based: portable and never steals global keyboard focus.
                while not self.stop_event.is_set():
                    key = input("n+Enter 시작/저장, r+Enter 거절, q+Enter 종료 > ").strip().lower()
                    self._emit({"n": "advance", "r": "reject", "q": "stop"}.get(key))
        except EOFError:
            self._emit("stop")
        except Exception as exc:
            if not self.stop_event.is_set():
                self.error = exc

    def poll(self):
        if self.error:
            raise RuntimeError(f"Control input failed/disconnected: {self.error}") from self.error
        try:
            return self.events.get_nowait()
        except queue.Empty:
            return None

    def clear_pending(self):
        while not self.events.empty():
            self.events.get_nowait()

    def close(self):
        self.stop_event.set()
        if self.device:
            self.thread.join(timeout=1)
            self.device.ungrab()
            self.device.close()
