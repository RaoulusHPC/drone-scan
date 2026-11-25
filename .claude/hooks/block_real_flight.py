#!/usr/bin/env python3
"""PreToolUse hook: Claude Code must never command a real drone.

Reads the hook JSON from stdin. Exit code 2 blocks the Bash call and
returns the stderr message to Claude. Real flights are started by a human
in their own terminal only.
"""

import json
import re
import sys

BLOCKED = [
    r"--real-flight",
    r"\bfly-real\b",
    r"udp:192\.168\.10\.1",  # DJI/Ryze Tello default address
    r"--master[= ]+(/dev/tty|udp:(?!127\.0\.0\.1)|tcp:(?!127\.0\.0\.1))",  # MAVLink to non-local endpoints
]

try:
    payload = json.load(sys.stdin)
except json.JSONDecodeError:
    sys.exit(0)

tool_input = payload.get("tool_input") if isinstance(payload, dict) else None
command = tool_input.get("command") if isinstance(tool_input, dict) else None
if not isinstance(command, str):
    sys.exit(0)
for pattern in BLOCKED:
    if re.search(pattern, command):
        print(
            "Blockiert: Befehle an echte Drohnen sind für Claude Code verboten "
            "(CLAUDE.md, Regel 4). Nutze ArduPilot-SITL (127.0.0.1) oder bitte den "
            "Menschen, den Flug selbst im Terminal zu starten.",
            file=sys.stderr,
        )
        sys.exit(2)
sys.exit(0)
