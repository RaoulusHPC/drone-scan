# SPDX-License-Identifier: AGPL-3.0-or-later
"""Stage ``compress``: SPZ scene for playback on phones (KAN-42, US-027, Q-7).

Reads ``recon/splat/scene.ply``, drops Gaussians below ``compress.min_opacity``, keeps at
most ``compress.max_gaussians`` (the most important by opacity × area), reduces the SH
degree to ``compress.sh_degree`` and writes ``compress/scene.spz`` (SPZ version 2, RUB).
CPU only (numpy); a separate stage so that changing these values never retrains.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar

import structlog

from dronescan.common import error_catalog as catalog
from dronescan.common.config import Settings
from dronescan.common.stage import (
    MANIFEST_NAME,
    WRITE_HINT,
    InputMissingError,
    QualityError,
    Stage,
    StageContext,
    StageError,
    load_manifest,
)
from dronescan.recon.job import PLY_NAME
from dronescan.recon.stage import SPLAT_DIR, ReconStage

SCENE_NAME = "scene.spz"

RECON_MISSING_HINTS = catalog.hints("compress.recon_missing")
UNREADABLE_HINTS = catalog.hints("compress.scene_unreadable")
EMPTY_HINTS = catalog.hints("compress.empty")

log = structlog.get_logger(__name__)


class CompressStage(Stage):
    name: ClassVar[str] = "compress"
    version: ClassVar[str] = "1"

    def inputs(self, scan_dir: Path) -> list[Path]:
        recon = scan_dir / ReconStage.name
        manifest = load_manifest(recon / MANIFEST_NAME)
        ply = recon / SPLAT_DIR / PLY_NAME
        if manifest is None or manifest.status != "ok" or not ply.is_file():
            raise InputMissingError(
                "Die Szene fehlt: Stufe 'recon' ist nicht erfolgreich gelaufen.",
                RECON_MISSING_HINTS,
                code="compress.recon_missing",
            )
        return [ply]

    def params(self, settings: Settings) -> dict[str, Any]:
        return settings.compress.model_dump()

    def tool_versions(self) -> dict[str, str]:
        from dronescan.export.spz import VERSION

        return {"spz_format": str(VERSION)}

    def run(self, ctx: StageContext) -> None:
        # numpy: lazy, keeps the CLI start fast
        from dronescan.export.prune import cap, drop_non_finite, drop_transparent, reduce_sh
        from dronescan.export.spz import write_spz
        from dronescan.recon.scene import SceneError, read_ply

        cfg = ctx.settings.compress
        ply = ctx.inputs[0]
        scene_path = ctx.out_dir / SCENE_NAME
        try:
            scene_path.unlink(missing_ok=True)  # a failed run must not leave an old scene
        except OSError as exc:
            raise StageError(
                "Ergebnisse konnten nicht geschrieben werden.",
                [WRITE_HINT],
                code="common.write_failed",
            ) from exc
        try:
            splats = read_ply(ply)
        except SceneError as exc:
            raise StageError(str(exc), UNREADABLE_HINTS, code="compress.scene_unreadable") from exc
        finite = drop_non_finite(splats)
        visible = drop_transparent(finite, cfg.min_opacity)
        kept = cap(visible, cfg.max_gaussians)
        ctx.metrics.update(
            gaussians_in=len(splats),
            gaussians_non_finite=len(splats) - len(finite),
            gaussians_transparent=len(finite) - len(visible),
            gaussians_over_cap=len(visible) - len(kept),
            gaussians_out=len(kept),
            sh_degree_in=splats.sh_degree,
        )
        if len(kept) == 0:
            raise QualityError(
                "Die Szene enthält keine sichtbaren Gaussians.", EMPTY_HINTS, code="compress.empty"
            )
        scene = reduce_sh(kept, cfg.sh_degree)
        try:
            size = write_spz(scene_path, scene)
        except OSError as exc:
            raise StageError(
                "Ergebnisse konnten nicht geschrieben werden.",
                [WRITE_HINT],
                code="common.write_failed",
            ) from exc
        ply_size = ply.stat().st_size
        ctx.metrics.update(
            sh_degree_out=scene.sh_degree,
            ply_bytes=ply_size,
            spz_bytes=size,
            compression_ratio=round(ply_size / size, 1),
        )
        log.info("compress_done", gaussians=len(scene), spz_mb=round(size / 2**20, 1))
