"""Tests for the PreToolUse hook that blocks commands to real drones (KAN-22, AC 2).

Blocked command strings are assembled from fragments so that the hook does not
already block the Bash call that runs this test file.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

HOOK = Path(__file__).resolve().parents[2] / ".claude" / "hooks" / "block_real_flight.py"

REAL_FLIGHT_FLAG = "--real" + "-flight"
FLY_REAL = "fly" + "-real"
TELLO_ADDR = "udp:192.168." + "10.1"
MASTER = "--" + "master"


def run_hook(stdin: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(HOOK)],
        input=stdin,
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )


def bash_payload(command: str) -> str:
    return json.dumps({"tool_name": "Bash", "tool_input": {"command": command}})


def test_real_flight_flag_is_blocked_with_exit_2() -> None:
    """Given a Bash command containing the real-flight flag,
    When the hook evaluates it,
    Then it exits with code 2 and explains the block on stderr.
    """
    result = run_hook(bash_payload(f"drone-scan fly --mission m.yaml {REAL_FLIGHT_FLAG}"))

    assert result.returncode == 2
    assert "Blockiert" in result.stderr
    assert "SITL" in result.stderr


@pytest.mark.parametrize(
    "command",
    [
        f"drone-scan {FLY_REAL} --mission m.yaml",
        f"python -m tello_ctl --addr {TELLO_ADDR}:8889",
        f"mavproxy.py {MASTER} /dev/ttyUSB0",
        f"mavproxy.py {MASTER}=/dev/ttyACM0",
        f"mavproxy.py {MASTER} udp:10.0.0.5:14550",
        f"mavproxy.py {MASTER} tcp:192.168.1.20:5760",
    ],
)
def test_commands_to_real_drones_are_blocked(command: str) -> None:
    """Given a command that targets a real drone (flag, Tello address, non-local MAVLink),
    When the hook evaluates it,
    Then it exits with code 2.
    """
    assert run_hook(bash_payload(command)).returncode == 2


@pytest.mark.parametrize(
    "command",
    [
        "ls -la",
        f"mavproxy.py {MASTER} udp:127.0.0.1:14550",
        f"mavproxy.py {MASTER} tcp:127.0.0.1:5760",
        "uv run pytest -m sitl",
    ],
)
def test_harmless_and_sitl_commands_are_allowed(command: str) -> None:
    """Given a harmless command or a MAVLink connection to local SITL,
    When the hook evaluates it,
    Then it exits with code 0.
    """
    result = run_hook(bash_payload(command))

    assert result.returncode == 0
    assert result.stderr == ""


@pytest.mark.parametrize(
    "stdin",
    [
        "",
        "not json",
        "[]",
        '"x"',
        json.dumps({"tool_name": "Bash"}),
        json.dumps({"tool_input": None}),
        json.dumps({"tool_input": {"command": None}}),
        json.dumps({"tool_input": ["ls"]}),
    ],
)
def test_malformed_or_empty_payload_is_allowed(stdin: str) -> None:
    """Given an empty, malformed or command-less hook payload,
    When the hook evaluates it,
    Then it does not crash (no traceback) and exits with code 0.
    """
    result = run_hook(stdin)

    assert result.returncode == 0
    assert "Traceback" not in result.stderr


def test_hook_is_registered_for_bash_in_project_settings() -> None:
    """Given the project settings in .claude/settings.json,
    When Claude Code loads its PreToolUse hooks,
    Then block_real_flight.py is registered for the Bash tool.
    """
    settings_path = HOOK.parents[1] / "settings.json"
    settings = json.loads(settings_path.read_text(encoding="utf-8"))

    bash_commands = [
        hook["command"]
        for entry in settings["hooks"]["PreToolUse"]
        if "Bash" in entry["matcher"].split("|")
        for hook in entry["hooks"]
    ]
    assert any("block_real_flight.py" in cmd for cmd in bash_commands)


# Known gaps in the --master pattern, found in the KAN-22 flight-safety review.
# strict=True: once the hook is hardened (follow-up ticket), these turn into
# XPASS failures and must be moved into the regular parametrized tests above.
@pytest.mark.xfail(strict=True, reason="hook gap: MAVProxy udpout/udpin to non-local hosts")
@pytest.mark.parametrize(
    "command",
    [
        f"mavproxy.py {MASTER}=udpout:10.0.0.5:14550",
        f"mavproxy.py {MASTER} udpin:0.0.0.0:14550",
    ],
)
def test_known_gap_mavproxy_udp_variants_are_blocked(command: str) -> None:
    """Given a MAVProxy connection via udpout/udpin to a non-local endpoint,
    When the hook evaluates it,
    Then it should exit with code 2 (currently not the case).
    """
    assert run_hook(bash_payload(command)).returncode == 2


@pytest.mark.xfail(strict=True, reason="hook gap: serial devices outside /dev/tty*")
def test_known_gap_serial_by_id_is_blocked() -> None:
    """Given a MAVLink connection to a serial device under /dev/serial/by-id,
    When the hook evaluates it,
    Then it should exit with code 2 (currently not the case).
    """
    command = f"mavproxy.py {MASTER} /dev/serial/by-id/usb-ArduPilot-if00"
    assert run_hook(bash_payload(command)).returncode == 2
