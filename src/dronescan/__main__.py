# SPDX-License-Identifier: AGPL-3.0-or-later
"""`python -m dronescan` = `drone-scan` (used by the job queue to start stages, KAN-52)."""

from dronescan.cli import app

app(prog_name="drone-scan")
