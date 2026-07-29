"""Find the LED lamp's serial port by hot-plug detection.

Usage:
  1. Plug in the LED lamp (ESP32)
  2. Run:  python find_lamp_port.py
  3. Unplug the LED lamp
  4. Press Enter
  5. The disappeared port is the lamp's port
"""

from __future__ import annotations

import sys


def get_port_names() -> set[str]:
    try:
        from serial.tools.list_ports import comports
    except ImportError:
        print("pyserial is required: pip install pyserial")
        sys.exit(1)
    return {p.device for p in comports()}


def main() -> None:
    print("LED lamp port finder")
    print("=" * 40)

    before = get_port_names()
    if not before:
        print("No serial ports detected. Is pyserial installed?")
        sys.exit(1)

    print(f"Current ports: {', '.join(sorted(before))}")
    print()
    print(">>> Unplug the LED lamp now, then press Enter <<<")
    input()

    after = get_port_names()
    disappeared = before - after
    appeared = after - before

    if disappeared:
        if len(disappeared) == 1:
            port = next(iter(disappeared))
            print(f"\nLED lamp port: {port}")
            print(f"\nUse: --lamp_port={port}")
        else:
            print(f"\nMultiple ports disappeared: {', '.join(sorted(disappeared))}")
            print("Re-run and unplug only the lamp.")
    elif appeared:
        # Reverse mode: user plugged in instead of unplugged
        print(f"\nNo port disappeared, but new port(s) appeared: {', '.join(sorted(appeared))}")
        print("Did you plug in instead of unplug? The lamp port is likely one of these.")
    else:
        print("\nNo port change detected. Make sure you unplugged the lamp.")


if __name__ == "__main__":
    main()
