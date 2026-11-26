# SPDX-License-Identifier: AGPL-3.0-or-later
"""Gaussian-splatting trainer of stage ``recon`` (runs in the worker container, KAN-41).

``python -m dronescan.recon.train --scene DIR --raw DIR --job FILE --out DIR`` trains 3D
Gaussians with gsplat (MCMC strategy) on an undistorted COLMAP scene, under the PyTorch
allocator limit of the job, in fp32 (Pascal has no fast fp16, ADR-0010). Every n-th image is
held out; PSNR/SSIM on them go to ``metrics.json``, the splat to ``scene.ply``. No image
content is written.

Written from the gsplat API (Apache-2.0), based on the spike trainer of KAN-19
(tools/spike01/train_gsplat.py). Learning rates, the SSIM weight 0.2, the MCMC regularisation
0.01 and the initial opacity/scale follow the gsplat example trainer
(examples/simple_trainer.py, nerfstudio project, Apache-2.0); no code was copied from it.
Exit codes: 0 ok, 1 error, 10 input missing, 30 out of memory (see `TrainExit`).
"""

from __future__ import annotations

import argparse
import math
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as functional
from gsplat import rasterization
from gsplat.strategy import MCMCStrategy
from pydantic import ValidationError

from dronescan.common.config import load_settings
from dronescan.common.gpu import apply_vram_limit
from dronescan.common.stage import is_oom
from dronescan.recon.job import PLY_NAME, RESULT_NAME, Evaluation, TrainExit, TrainJob, TrainResult
from dronescan.recon.scene import (
    SceneError,
    View,
    read_raw,
    read_scene,
    split_holdout,
    write_ply,
)

SH_C0 = 0.28209479177387814  # zeroth spherical-harmonics basis function
PSNR_MAX_DB = 100.0  # identical images: a finite value, JSON has no infinity
SCENE_SCALE_MARGIN = 1.1  # scene radius = largest camera distance from the centre, plus 10 %
MIN_SCENE_SCALE = 1e-3  # a single camera position must not give a zero scale
MIN_NEIGHBOUR_DISTANCE = 1e-6  # duplicate SfM points: avoid log(0) for the initial scale
SINGLE_POINT_SCALE = 0.01  # initial size of a lone point, relative to the scene scale
ADAM_EPS = 1e-15  # as in the gsplat example trainer (tiny gradients of small Gaussians)
JOB_MISMATCH = (
    "Der Trainingsauftrag passt nicht zum Worker-Image. Abhilfe: Image neu bauen mit "
    "`docker compose build worker`."
)
MIB = 1024**2
SSIM_WEIGHT = 0.2
OPACITY_REG = 0.01  # MCMC regularisation of opacities and scales
SCALE_REG = 0.01
INIT_OPACITY = 0.5
INIT_SCALE = 0.1
SH_DEGREE_INTERVAL = 1000  # raise the SH degree by one every this many steps
REFINE_STOP_FRACTION = 0.8  # MCMC relocation/growth stops after 80 % of the iterations
POSITION_LR_FINAL_FACTOR = 0.01  # position learning rate decays to 1 % over the run
TRAIN_VIEW_SAMPLES = 8  # training views evaluated for the overfitting check
# Memory of one chunk of the nearest-neighbour search: a [chunk, P] distance matrix plus the
# top-k buffers. Budgeted in bytes so the start does not grow with the SfM point count (vram-guard).
NEIGHBOUR_BUDGET_BYTES = 256 * MIB
_NEIGHBOUR_BYTES_PER_ENTRY = 4 * 3  # fp32 distances, cdist/topk temporaries
# The exact search is O(P²): 3.6 s for 100k points, ~20 min for 1.5 M on the GTX 1060 (KAN-41).
# Above this size a Morton-order window search is used (O(P)).
EXACT_KNN_MAX_POINTS = 100_000
KNN_WINDOW = 16  # neighbours on each side along the Morton curve
LEARNING_RATES = {
    "means": 1.6e-4,  # times the scene scale
    "scales": 5e-3,
    "quats": 1e-3,
    "opacities": 5e-2,
    "sh0": 2.5e-3,
    "shN": 2.5e-3 / 20,
}
_SSIM_C1 = 0.01**2
_SSIM_C2 = 0.03**2
_SSIM_WINDOW = 11
_SSIM_SIGMA = 1.5


@dataclass
class Data:
    views: list[View]
    viewmats: torch.Tensor  # [M, 4, 4] world to camera, on the GPU
    intrinsics: torch.Tensor  # [M, 3, 3] on the GPU
    images: list[torch.Tensor]  # uint8 [H, W, 3] on the CPU
    points: torch.Tensor  # [P, 3]
    colors: torch.Tensor  # [P, 3]
    scene_scale: float


# ---------- metrics (own SSIM: Wang et al. 2004, gaussian window 11, sigma 1.5) ----------


def _window(channels: int, device: torch.device) -> torch.Tensor:
    axis = torch.arange(_SSIM_WINDOW, dtype=torch.float32, device=device) - (_SSIM_WINDOW - 1) / 2
    gauss = torch.exp(-(axis**2) / (2 * _SSIM_SIGMA**2))
    gauss = gauss / gauss.sum()
    kernel = (gauss[:, None] * gauss[None, :])[None, None]
    return kernel.expand(channels, 1, _SSIM_WINDOW, _SSIM_WINDOW).contiguous()


def ssim_tensor(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean SSIM of two [H, W, 3] images in 0..1 (differentiable)."""
    x, y = prediction.permute(2, 0, 1)[None], target.permute(2, 0, 1)[None]
    window = _window(3, x.device)

    def blur(image: torch.Tensor) -> torch.Tensor:
        return functional.conv2d(image, window, padding=_SSIM_WINDOW // 2, groups=3)

    mu_x, mu_y = blur(x), blur(y)
    var_x = blur(x * x) - mu_x**2
    var_y = blur(y * y) - mu_y**2
    covariance = blur(x * y) - mu_x * mu_y
    numerator = (2 * mu_x * mu_y + _SSIM_C1) * (2 * covariance + _SSIM_C2)
    denominator = (mu_x**2 + mu_y**2 + _SSIM_C1) * (var_x + var_y + _SSIM_C2)
    return (numerator / denominator).mean()


def psnr(prediction: torch.Tensor, target: torch.Tensor) -> float:
    mse = float(torch.mean((prediction - target) ** 2))
    return PSNR_MAX_DB if mse == 0.0 else min(PSNR_MAX_DB, -10.0 * math.log10(mse))


# ---------- data ----------


def _rotation(quat_wxyz: tuple[float, float, float, float]) -> torch.Tensor:
    w, x, y, z = quat_wxyz
    return torch.tensor(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=torch.float32,
    )


def _load_image(raw_dir: Path, original: View, view: View) -> torch.Tensor:
    image = torch.from_numpy(read_raw(raw_dir / original.raw_name, original.width, original.height))
    if (view.width, view.height) == (original.width, original.height):
        return image
    resized = functional.interpolate(
        image.permute(2, 0, 1)[None].float(), size=(view.height, view.width), mode="area"
    )
    return resized[0].permute(1, 2, 0).round().clamp(0, 255).to(torch.uint8).contiguous()


def load_data(scene_dir: Path, raw_dir: Path, max_edge: int, device: torch.device) -> Data:
    scene = read_scene(scene_dir)
    originals = scene.views
    views = [view.scaled(max_edge) for view in originals]
    viewmats, intrinsics, centres = [], [], []
    for view in views:
        rotation = _rotation(view.quat_wxyz)
        translation = torch.tensor(view.translation, dtype=torch.float32)
        viewmat = torch.eye(4)
        viewmat[:3, :3], viewmat[:3, 3] = rotation, translation
        viewmats.append(viewmat)
        intrinsics.append(torch.tensor([[view.fx, 0, view.cx], [0, view.fy, view.cy], [0, 0, 1]]))
        centres.append(-rotation.T @ translation)
    centre_stack = torch.stack(centres)
    scene_scale = (
        float((centre_stack - centre_stack.mean(0)).norm(dim=1).max()) * SCENE_SCALE_MARGIN
    )
    return Data(
        views=views,
        viewmats=torch.stack(viewmats).to(device),
        intrinsics=torch.stack(intrinsics).float().to(device),
        images=[_load_image(raw_dir, o, v) for o, v in zip(originals, views, strict=True)],
        points=torch.from_numpy(scene.points),
        colors=torch.from_numpy(scene.colors),
        scene_scale=max(scene_scale, MIN_SCENE_SCALE),
    )


# ---------- model ----------


def neighbour_chunk(count: int) -> int:
    """Points per chunk of the nearest-neighbour search within `NEIGHBOUR_BUDGET_BYTES`."""
    return max(1, min(count, NEIGHBOUR_BUDGET_BYTES // (count * _NEIGHBOUR_BYTES_PER_ENTRY)))


def _exact_neighbours(points: torch.Tensor, k: int) -> torch.Tensor:
    """Distances to the ``k`` nearest other points [P, k]; O(P²), chunked by a byte budget."""
    chunk = neighbour_chunk(len(points))
    nearest = []
    for start in range(0, len(points), chunk):
        # matmul form (fast; fp32 rounding at mm distances is fine for an initial size,
        # the direct form takes 40x longer on the GTX 1060)
        distances = torch.cdist(points[start : start + chunk], points)
        nearest.append(distances.topk(k + 1, largest=False).values[:, 1:])
        del distances
    return torch.cat(nearest)


def _spread_bits(values: torch.Tensor) -> torch.Tensor:
    """Insert two zero bits between the lowest 21 bits (3D Morton code, int64)."""
    x = values & 0x1FFFFF
    for shift, mask in ((32, 0x1F00000000FFFF), (16, 0x1F0000FF0000FF), (8, 0x100F00F00F00F00F),
                        (4, 0x10C30C30C30C30C3), (2, 0x1249249249249249)):  # fmt: skip
        x = (x | (x << shift)) & mask
    return x


def _window_neighbours(points: torch.Tensor, k: int) -> torch.Tensor:
    """Approximate distances to the ``k`` nearest points [P, k] in O(P · window).

    Points are sorted along a Morton (Z-order) curve and compared only with their
    ``KNN_WINDOW`` predecessors and successors in that order. Neighbours across a cell border
    can be far apart on the curve, so the search is repeated on a grid shifted by half the
    extent and the smaller distances are kept. Good enough for the initial Gaussian size,
    which the optimiser adapts anyway.
    """
    low, high = points.min(0).values, points.max(0).values
    extent = (high - low).max().clamp_min(MIN_NEIGHBOUR_DISTANCE)
    best = torch.full((len(points), k), torch.inf, device=points.device)
    for shift in (0.0, 0.5):
        grid = ((points - low) / extent + shift) / (1.0 + shift) * (2**21 - 1)
        cells = grid.long()
        codes = _spread_bits(cells[:, 0]) | (_spread_bits(cells[:, 1]) << 1)
        codes = codes | (_spread_bits(cells[:, 2]) << 2)
        order = codes.argsort()
        ordered = points[order]
        found = torch.full((len(points), k), torch.inf, device=points.device)
        for offset in range(1, KNN_WINDOW + 1):
            for other in (ordered.roll(offset, 0), ordered.roll(-offset, 0)):
                distance = (ordered - other).norm(dim=1)  # wraps at the ends: only too large
                found = torch.cat([found, distance[:, None]], dim=1).topk(k, largest=False).values
        best[order] = torch.minimum(best[order], found)
    return best


def neighbour_distances(points: torch.Tensor, k: int) -> torch.Tensor:
    """Distances to the ``k`` nearest other points: exact for small clouds, else approximate."""
    if len(points) <= EXACT_KNN_MAX_POINTS:
        return _exact_neighbours(points, k)
    return _window_neighbours(points, k)


def limit_points(data: Data, max_points: int, seed: int) -> None:
    """Thin the SfM cloud to ``max_points`` (MCMC only grows: more start points than the cap
    would keep the model above it for the whole run)."""
    if len(data.points) > max_points:
        keep = torch.randperm(len(data.points), generator=torch.Generator().manual_seed(seed))
        keep = keep[:max_points].sort().values
        data.points, data.colors = data.points[keep], data.colors[keep]


def init_splats(data: Data, sh_degree: int, device: torch.device) -> torch.nn.ParameterDict:
    points, colors = data.points.to(device), data.colors.to(device)
    count = len(points)
    if count > 1:
        nearest = neighbour_distances(points, min(3, count - 1))
        dist_avg = nearest.pow(2).mean(dim=-1).sqrt().clamp_min(MIN_NEIGHBOUR_DISTANCE)
    else:
        dist_avg = torch.full((count,), data.scene_scale * SINGLE_POINT_SCALE, device=device)
    rest = (sh_degree + 1) ** 2 - 1
    return torch.nn.ParameterDict(
        {
            "means": torch.nn.Parameter(points),
            "scales": torch.nn.Parameter(torch.log(dist_avg * INIT_SCALE)[:, None].repeat(1, 3)),
            "quats": torch.nn.Parameter(
                functional.normalize(torch.randn(count, 4, device=device), dim=-1)
            ),
            "opacities": torch.nn.Parameter(
                torch.logit(torch.full((count,), INIT_OPACITY, device=device))
            ),
            "sh0": torch.nn.Parameter(((colors - 0.5) / SH_C0)[:, None, :]),
            "shN": torch.nn.Parameter(torch.zeros(count, rest, 3, device=device)),
        }
    )


def render(
    params: torch.nn.ParameterDict, data: Data, index: int, sh_degree: int, packed: bool
) -> tuple[torch.Tensor, dict[str, Any]]:
    view = data.views[index]
    colors, _alphas, info = rasterization(
        means=params["means"],
        quats=params["quats"],
        scales=torch.exp(params["scales"]),
        opacities=torch.sigmoid(params["opacities"]),
        colors=torch.cat([params["sh0"], params["shN"]], dim=1),
        viewmats=data.viewmats[index : index + 1],
        Ks=data.intrinsics[index : index + 1],
        width=view.width,
        height=view.height,
        sh_degree=sh_degree,
        packed=packed,
        rasterize_mode="classic",
        render_mode="RGB",
    )
    return colors[0].clamp(0.0, 1.0), info


def evaluate(
    params: torch.nn.ParameterDict, data: Data, indices: list[int], sh_degree: int, packed: bool
) -> tuple[float, float]:
    """Mean PSNR (dB) and SSIM over ``indices``."""
    scores_psnr, scores_ssim = [], []
    with torch.no_grad():
        for index in indices:
            prediction, _ = render(params, data, index, sh_degree, packed)
            target = data.images[index].to(prediction.device).float() / 255.0
            scores_psnr.append(psnr(prediction, target))
            scores_ssim.append(float(ssim_tensor(prediction, target)))
    return sum(scores_psnr) / len(scores_psnr), sum(scores_ssim) / len(scores_ssim)


def save_ply(params: torch.nn.ParameterDict, path: Path) -> None:
    with torch.no_grad():
        arrays = {name: params[name].detach().float().cpu().numpy() for name in params}
    write_ply(
        path,
        means=arrays["means"],
        sh0=arrays["sh0"],
        sh_rest=arrays["shN"],
        opacity_logits=arrays["opacities"],
        log_scales=arrays["scales"],
        quats_wxyz=arrays["quats"],
    )


# ---------- training ----------


def train(job: TrainJob, scene_dir: Path, raw_dir: Path, out: Path, result: TrainResult) -> None:
    """Train and fill ``result``; raises on failure (OOM included)."""
    device = torch.device("cuda")
    settings = load_settings(job.profile).model_copy(update={"vram_limit_gb": job.vram_limit_gb})
    apply_vram_limit(settings)  # before the first allocation
    torch.cuda.reset_peak_memory_stats()
    torch.manual_seed(job.seed)
    result.device = torch.cuda.get_device_name()
    result.capability = list(torch.cuda.get_device_capability())

    data = load_data(scene_dir, raw_dir, job.max_image_edge_px, device)
    train_ids, holdout = split_holdout(len(data.views), job.holdout_every)
    if not train_ids or not holdout:
        raise SceneError("Zu wenige Bilder für Training und Validierung.")
    result.images_train, result.images_holdout = len(train_ids), len(holdout)
    result.init_points = len(data.points)
    limit_points(data, job.max_gaussians, job.seed)
    result.init_points_used = len(data.points)
    result.image_edge_px = max(max(view.width, view.height) for view in data.views)

    params = init_splats(data, job.sh_degree, device)
    optimizers = {
        name: torch.optim.Adam(
            [
                {
                    "params": [params[name]],
                    "lr": rate * (data.scene_scale if name == "means" else 1.0),
                    "name": name,
                }
            ],
            eps=ADAM_EPS,
        )
        for name, rate in LEARNING_RATES.items()
    }
    scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizers["means"], gamma=POSITION_LR_FINAL_FACTOR ** (1.0 / job.iterations)
    )
    strategy = MCMCStrategy(
        cap_max=job.max_gaussians,
        refine_stop_iter=int(REFINE_STOP_FRACTION * job.iterations),
        verbose=False,
    )
    strategy.check_sanity(params, optimizers)
    state = strategy.initialize_state()

    generator = torch.Generator().manual_seed(job.seed)
    eval_at = set(job.eval_at)
    started = time.perf_counter()
    for step in range(job.iterations):
        index = train_ids[int(torch.randint(len(train_ids), (1,), generator=generator))]
        target = data.images[index].to(device, non_blocking=True).float() / 255.0
        sh_degree = min(step // SH_DEGREE_INTERVAL, job.sh_degree)
        prediction, info = render(params, data, index, sh_degree, job.packed)
        loss = (1 - SSIM_WEIGHT) * (prediction - target).abs().mean()
        loss = loss + SSIM_WEIGHT * (1.0 - ssim_tensor(prediction, target))
        loss = loss + OPACITY_REG * torch.sigmoid(params["opacities"]).mean()
        loss = loss + SCALE_REG * torch.exp(params["scales"]).mean()
        loss.backward()
        for optimizer in optimizers.values():
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        scheduler.step()
        strategy.step_post_backward(
            params, optimizers, state, step, info, lr=scheduler.get_last_lr()[0]
        )
        done = step + 1
        if done in eval_at:
            value_psnr, value_ssim = evaluate(params, data, holdout, sh_degree, job.packed)
            result.curve.append(
                Evaluation(
                    step=done,
                    psnr_db=round(value_psnr, 3),
                    ssim=round(value_ssim, 4),
                    gaussians=len(params["means"]),
                    seconds=round(time.perf_counter() - started, 1),
                )
            )
            print(result.curve[-1].model_dump_json(), flush=True)
        if done % job.log_every == 0 or done == job.iterations:
            torch.cuda.synchronize()
            print(
                f'{{"step": {done}, "gaussians": {len(params["means"])}, '
                f'"seconds": {time.perf_counter() - started:.1f}}}',
                flush=True,
            )
    torch.cuda.synchronize()
    result.train_seconds = round(time.perf_counter() - started, 2)
    result.seconds_per_iteration = round(result.train_seconds / job.iterations, 4)

    value_psnr, value_ssim = evaluate(params, data, holdout, job.sh_degree, job.packed)
    sample = train_ids[:: max(1, len(train_ids) // TRAIN_VIEW_SAMPLES)]
    train_psnr, _ = evaluate(params, data, sample, job.sh_degree, job.packed)
    result.psnr_db, result.ssim = round(value_psnr, 3), round(value_ssim, 4)
    result.train_psnr_db = round(train_psnr, 3)
    result.gaussians = len(params["means"])
    save_ply(params, out / PLY_NAME)
    result.ok = True


def _peaks(result: TrainResult) -> None:
    if torch.cuda.is_available() and torch.cuda.is_initialized():
        result.peak_allocated_mib = round(torch.cuda.max_memory_allocated() / MIB, 1)
        result.peak_reserved_mib = round(torch.cuda.max_memory_reserved() / MIB, 1)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Gaussian-Splatting-Training (Stufe recon).")
    parser.add_argument(
        "--scene", type=Path, required=True, help="undistorted COLMAP model (binary or text)"
    )
    parser.add_argument("--raw", type=Path, required=True, help="directory with <name>.rgb")
    parser.add_argument("--job", type=Path, required=True, help="TrainJob JSON")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)
    try:
        job = TrainJob.model_validate_json(args.job.read_text(encoding="utf-8"))
    except (OSError, ValidationError) as exc:  # host and image from different versions
        traceback.print_exc()
        failure = TrainResult(
            ok=False, vram_limit_gb=0.0, error=f"{JOB_MISMATCH} ({type(exc).__name__})"
        )
        (out / RESULT_NAME).write_text(failure.model_dump_json(indent=2), encoding="utf-8")
        return TrainExit.ERROR
    result = TrainResult(
        ok=False,
        vram_limit_gb=job.vram_limit_gb,
        iterations=job.iterations,
        max_gaussians=job.max_gaussians,
        packed=job.packed,
    )
    code = TrainExit.OK
    try:
        if not args.raw.is_dir():
            raise SceneError("Verzeichnis der Rohbilder fehlt.")
        train(job, args.scene, args.raw, out, result)
    except (SceneError, FileNotFoundError) as exc:
        result.error, code = str(exc), TrainExit.INPUT_MISSING
    except Exception as exc:  # report every failure in metrics.json, then exit non-zero
        traceback.print_exc()
        result.error = f"{type(exc).__name__}: {str(exc).splitlines()[0] if str(exc) else ''}"
        code = TrainExit.OOM if is_oom(exc) else TrainExit.ERROR
    _peaks(result)
    (out / RESULT_NAME).write_text(result.model_dump_json(indent=2), encoding="utf-8")
    print(result.model_dump_json(exclude={"curve"}), flush=True)
    return code


if __name__ == "__main__":
    sys.exit(main())
