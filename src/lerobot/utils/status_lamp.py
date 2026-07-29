"""NexArm recording status lamp — drives NeoPixel ring via serial.

Reuses the same ESP32 + serial protocol as ~/.claude/lamp.py.
Only active when --lamp_port is configured.

Firmware command:  countdown:r,g,b:N
  - r,g,b = LED color (stage indicator)
  - N     = number of LEDs to light (0~24, represents remaining time)

Thread safety: all serial I/O runs in a background daemon thread.
The main record loop is NEVER blocked by lamp communication.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from typing import Callable

logger = logging.getLogger(__name__)

# Color constants: color = stage
COLOR_RECORD = "255,50,50"    # Red    = recording
COLOR_RESET = "255,165,0"     # Orange = resetting
COLOR_DONE = "0,255,0"        # Green  = all episodes done

NUM_LEDS = 24


class StatusLamp:
    """Thread-safe lamp controller for recording status.

    All serial I/O is offloaded to a background daemon thread via a queue.
    Calling countdown() / off() / done() from the record loop is
    non-blocking — they just enqueue a command and return immediately.
    """

    def __init__(self, port: str = "COM4", baud: int = 115200):
        self.port = port
        self.baud = baud
        self._queue: queue.Queue[str | None] = queue.Queue(maxsize=4)
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def _worker(self) -> None:
        """Background thread: drain queue, send commands one at a time."""
        while True:
            cmd = self._queue.get()
            if cmd is None:  # shutdown signal
                break
            try:
                import serial

                with serial.Serial(self.port, self.baud, timeout=1, dsrdtr=False) as s:
                    s.dtr = False
                    time.sleep(0.15)
                    s.write((cmd + "\n").encode())
            except Exception:
                pass

    def _enqueue(self, cmd: str) -> None:
        """Enqueue a command, dropping the oldest if queue is full."""
        if self._queue.full():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                pass
        self._queue.put_nowait(cmd)

    def countdown(self, color: str, remaining_ratio: float) -> None:
        """Update countdown display (non-blocking).

        Args:
            color: "r,g,b" string matching the current stage.
            remaining_ratio: 1.0 = just started, 0.0 = time's up.
        """
        ratio = max(0.0, min(1.0, remaining_ratio))
        leds = round(NUM_LEDS * ratio)
        self._enqueue(f"countdown:{color}:{leds}")

    def done(self) -> None:
        """All episodes complete — solid green."""
        self._enqueue(f"on:{COLOR_DONE}")

    def off(self) -> None:
        """Turn off the lamp."""
        self._enqueue("off")

    def shutdown(self) -> None:
        """Stop the background thread. Call once at program exit."""
        self._queue.put(None)
        self._thread.join(timeout=2)


def make_lamp_progress_cb(lamp: StatusLamp, color: str) -> Callable[[float], None]:
    """Create a progress callback for record_loop().

    Returns a callable that throttles lamp updates to ~1 Hz
    so we don't flood the serial queue at 30 fps.
    The callback itself is non-blocking (just enqueues).
    """
    last_sent = -1.0  # force first send immediately

    def _cb(remaining_ratio: float) -> None:
        nonlocal last_sent
        now = time.monotonic()
        if now - last_sent >= 1.0:
            lamp.countdown(color, remaining_ratio)
            last_sent = now

    return _cb
