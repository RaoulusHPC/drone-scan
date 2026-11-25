# SPDX-License-Identifier: AGPL-3.0-or-later
"""Capture-guidance hints (German, user-facing) shared by the stage and quality checks.

One wording per real-world cause (KAN-33, KAN-34), so a pilot reading several hints never
sees two slightly different sentences for the same advice.
"""

from __future__ import annotations

HINT_MOVE_SLOWLY = (
    "Kamera während der Aufnahme langsam und stetig durch den Raum bewegen (≤ 0,5 m/s)."
)
HINT_RECORD_LONGER = "Länger aufnehmen: mindestens 30 Sekunden pro Raum."
HINT_LIGHTS_ON = "Licht einschalten, damit die Bilder nicht verwackeln."
HINT_DAYLIGHT = "Licht einschalten und möglichst bei Tageslicht aufnehmen."
HINT_AVOID_BACKLIGHT = (
    "Nicht direkt gegen Fenster oder helles Licht filmen, Lichtquelle besser im Rücken halten."
)
HINT_KEEP_MOVING = (
    "Nicht an einer Stelle verweilen, stetig weiterbewegen und neue Blickwinkel zeigen."
)
