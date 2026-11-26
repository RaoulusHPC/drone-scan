# SPDX-License-Identifier: AGPL-3.0-or-later
"""Contract between stage ``recon`` (host) and the trainer in the worker container.

The stage writes a `TrainJob` as JSON, the trainer answers with a `TrainResult` JSON next to
the splat. Both sides import only this module and pydantic, so no PyTorch is needed on the host.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, PositiveInt

JOB_NAME = "job.json"
RESULT_NAME = "metrics.json"
PLY_NAME = "scene.ply"


class TrainExit:
    """Exit codes of the trainer (same meaning as the stage contract)."""

    OK = 0
    ERROR = 1
    INPUT_MISSING = 10
    OOM = 30


class TrainJob(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    profile: str
    vram_limit_gb: float = Field(gt=0)
    iterations: PositiveInt
    max_gaussians: PositiveInt
    max_image_edge_px: PositiveInt
    sh_degree: int = Field(ge=0, le=3)
    holdout_every: int = Field(ge=2)
    packed: bool = False
    seed: int = 0
    eval_at: list[PositiveInt] = []  # extra held-out evaluations during training (quality curve)
    log_every: PositiveInt = 500


class Evaluation(BaseModel):
    step: int
    psnr_db: float
    ssim: float
    gaussians: int
    seconds: float


class TrainResult(BaseModel):
    """``metrics.json`` of the trainer. Numbers only, no image content."""

    model_config = ConfigDict(extra="forbid")

    ok: bool
    error: str | None = None
    vram_limit_gb: float
    device: str | None = None
    capability: list[int] = []
    iterations: int = 0
    max_gaussians: int = 0
    image_edge_px: int = 0
    packed: bool = False
    images_train: int = 0
    images_holdout: int = 0
    init_points: int = 0  # points of the SfM cloud
    init_points_used: int = 0  # after thinning to max_gaussians
    gaussians: int = 0
    psnr_db: float | None = None  # mean over the held-out frames
    ssim: float | None = None
    train_psnr_db: float | None = None  # mean over a sample of training views
    train_seconds: float = 0.0
    seconds_per_iteration: float | None = None
    peak_allocated_mib: float | None = None
    peak_reserved_mib: float | None = None
    curve: list[Evaluation] = []
