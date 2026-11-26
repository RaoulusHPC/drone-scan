# SPDX-License-Identifier: AGPL-3.0-or-later
"""REST API, GPU job queue (concurrency 1), progress via SSE and storage management."""

from __future__ import annotations

import ipaddress


def is_loopback(host: str) -> bool:
    """True for addresses only reachable from this machine (127.0.0.0/8, ::1, localhost)."""
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host.strip("[]")).is_loopback
    except ValueError:
        return False
