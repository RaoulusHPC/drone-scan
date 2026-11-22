"""Minimal gsplat trainer for spike KAN-19 (runs in the worker container).

Trains 3D Gaussians from a COLMAP-undistorted text model and raw RGB frames, on the
GTX 1060 (sm_61, fp32) under the allocator limit of the profile (5.0 GiB), and reports
time, memory and PSNR/SSIM on held-out frames as one JSON file. Written from the gsplat
API (Apache-2.0) and the structure of its example trainer, not copied.

Usage (see tools/spike01/run_room.py):
    python -m tools.spike01.train_gsplat --scene DIR --raw DIR --out DIR --width W ...
Exit codes follow the stage contract: 0 ok, 10 input missing, 30 out of memory.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from gsplat import rasterization
from gsplat.strategy import DefaultStrategy, MCMCStrategy
from tools.spike01.colmap_text import (
    Image,
    parse_cameras_txt,
    parse_images_txt,
    parse_points3d_txt,
)
from tools.spike01.frames import scale_pinhole
from tools.spike01.metrics import psnr, ssim, ssim_tensor

from dronescan.common.config import load_settings
from dronescan.common.gpu import apply_vram_limit

SH_C0 = 0.28209479177387814
MIB = 1024**2


@dataclass
class Dataset:
    names: list[str]
    viewmats: torch.Tensor  # [M, 4, 4] world to camera
    intrinsics: torch.Tensor  # [3, 3]
    width: int
    height: int
    images: list[torch.Tensor]  # uint8 [H, W, 3] on the CPU
    points: torch.Tensor  # [P, 3]
    point_colors: torch.Tensor  # [P, 3] in 0..1
    scene_scale: float


def quat_to_rotation(w: float, x: float, y: float, z: float) -> torch.Tensor:
    return torch.tensor(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=torch.float32,
    )


def load_dataset(scene: Path, raw: Path, width: int) -> Dataset:
    cameras = parse_cameras_txt((scene / "cameras.txt").read_text())
    images: list[Image] = sorted(
        parse_images_txt((scene / "images.txt").read_text()), key=lambda image: image.name
    )
    points = parse_points3d_txt((scene / "points3D.txt").read_text())
    camera = scale_pinhole(next(iter(cameras.values())), new_width=width)
    fx, fy, cx, cy = camera.params
    intrinsics = torch.tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=torch.float32)

    viewmats, frames, centres = [], [], []
    for image in images:
        rotation = quat_to_rotation(*image.quat_wxyz)
        translation = torch.tensor(image.translation, dtype=torch.float32)
        viewmat = torch.eye(4)
        viewmat[:3, :3] = rotation
        viewmat[:3, 3] = translation
        viewmats.append(viewmat)
        centres.append(-rotation.T @ translation)
        data = torch.frombuffer(
            bytearray((raw / f"{image.name}.rgb").read_bytes()), dtype=torch.uint8
        )
        frames.append(data.reshape(camera.height, camera.width, 3))
    centre_stack = torch.stack(centres)
    scene_scale = float((centre_stack - centre_stack.mean(0)).norm(dim=1).max()) * 1.1
    return Dataset(
        names=[image.name for image in images],
        viewmats=torch.stack(viewmats),
        intrinsics=intrinsics,
        width=camera.width,
        height=camera.height,
        images=frames,
        points=torch.tensor([p.xyz for p in points], dtype=torch.float32),
        point_colors=torch.tensor([p.rgb for p in points], dtype=torch.float32) / 255.0,
        scene_scale=max(scene_scale, 1e-3),
    )


def random_points(data: Dataset, count: int, seed: int) -> None:
    """Memory stress mode: replace the SfM cloud by ``count`` random points in its bounding box."""
    generator = torch.Generator().manual_seed(seed)
    low, high = data.points.min(dim=0).values, data.points.max(dim=0).values
    data.points = low + (high - low) * torch.rand(count, 3, generator=generator)
    data.point_colors = torch.rand(count, 3, generator=generator)


def init_splats(
    data: Dataset,
    *,
    sh_degree: int,
    init_opacity: float,
    init_scale: float,
    device: torch.device,
    stress: bool = False,
) -> torch.nn.ParameterDict:
    points = data.points.to(device)
    colors = data.point_colors.to(device)
    if stress:  # a neighbour search over millions of points would itself need gigabytes
        volume = (points.max(0).values - points.min(0).values).prod()
        dist_avg = torch.full(
            (len(points),), float((volume / len(points)) ** (1 / 3)), device=device
        )
    else:
        nearest = []
        for start in range(0, len(points), 2048):  # 3 nearest neighbours, chunked to bound VRAM
            distances = torch.cdist(points[start : start + 2048], points)
            nearest.append(distances.topk(4, largest=False).values[:, 1:])
        dist_avg = torch.cat(nearest).pow(2).mean(dim=-1).sqrt().clamp_min(1e-6)
    count = len(points)
    sh_rest = (sh_degree + 1) ** 2 - 1
    return torch.nn.ParameterDict(
        {
            "means": torch.nn.Parameter(points),
            "scales": torch.nn.Parameter(torch.log(dist_avg * init_scale)[:, None].repeat(1, 3)),
            "quats": torch.nn.Parameter(
                torch.nn.functional.normalize(torch.randn(count, 4, device=device), dim=-1)
            ),
            "opacities": torch.nn.Parameter(
                torch.logit(torch.full((count,), init_opacity, device=device))
            ),
            "sh0": torch.nn.Parameter(((colors - 0.5) / SH_C0)[:, None, :]),
            "shN": torch.nn.Parameter(torch.zeros(count, sh_rest, 3, device=device)),
        }
    )


def make_optimizers(
    params: torch.nn.ParameterDict, scene_scale: float, iterations: int
) -> tuple[dict[str, torch.optim.Optimizer], torch.optim.lr_scheduler.LRScheduler]:
    rates = {
        "means": 1.6e-4 * scene_scale,
        "scales": 5e-3,
        "quats": 1e-3,
        "opacities": 5e-2,
        "sh0": 2.5e-3,
        "shN": 2.5e-3 / 20,
    }
    optimizers = {
        name: torch.optim.Adam([{"params": [params[name]], "lr": rate, "name": name}], eps=1e-15)
        for name, rate in rates.items()
    }
    # position learning rate decays to 1 % over the run, as in the reference trainer
    scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizers["means"], gamma=0.01 ** (1.0 / iterations)
    )
    return optimizers, scheduler


def render(
    params: torch.nn.ParameterDict,
    data: Dataset,
    index: int,
    sh_degree: int,
    *,
    packed: bool,
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, Any]]:
    colors, _alphas, info = rasterization(
        means=params["means"],
        quats=params["quats"],
        scales=torch.exp(params["scales"]),
        opacities=torch.sigmoid(params["opacities"]),
        colors=torch.cat([params["sh0"], params["shN"]], dim=1),
        viewmats=data.viewmats[index : index + 1].to(device),
        Ks=data.intrinsics[None].to(device),
        width=data.width,
        height=data.height,
        sh_degree=sh_degree,
        packed=packed,
        absgrad=False,
        rasterize_mode="classic",
        render_mode="RGB",
    )
    return colors[0].clamp(0.0, 1.0), info


def evaluate(
    params: torch.nn.ParameterDict,
    data: Dataset,
    indices: list[int],
    sh_degree: int,
    *,
    packed: bool,
    device: torch.device,
) -> dict[str, float]:
    scores_psnr, scores_ssim = [], []
    with torch.no_grad():
        for index in indices:
            prediction, _ = render(params, data, index, sh_degree, packed=packed, device=device)
            target = data.images[index].to(device).float() / 255.0
            scores_psnr.append(psnr(prediction, target))
            scores_ssim.append(ssim(prediction, target))
    return {
        "psnr": round(sum(scores_psnr) / len(scores_psnr), 3),
        "ssim": round(sum(scores_ssim) / len(scores_ssim), 4),
    }


def save_renders(
    params: torch.nn.ParameterDict,
    data: Dataset,
    indices: list[int],
    sh_degree: int,
    out: Path,
    *,
    packed: bool,
    device: torch.device,
) -> None:
    """Raw RGB renders next to the held-out frames, for the PO's local visual check only."""
    out.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        for index in indices:
            prediction, _ = render(params, data, index, sh_degree, packed=packed, device=device)
            name = data.names[index]
            (out / f"{name}.render.rgb").write_bytes(
                (prediction * 255).round().byte().cpu().numpy().tobytes()
            )
            (out / f"{name}.gt.rgb").write_bytes(data.images[index].numpy().tobytes())


def train(args: argparse.Namespace) -> dict[str, Any]:
    device = torch.device("cuda")
    settings = load_settings(args.profile)
    apply_vram_limit(settings, device.index)
    torch.manual_seed(args.seed)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()

    data = load_dataset(Path(args.scene), Path(args.raw), args.width)
    if args.init_random:
        random_points(data, args.init_random, args.seed)
    holdout = [i for i in range(len(data.names)) if i % args.holdout_every == 0]
    train_ids = [i for i in range(len(data.names)) if i not in set(holdout)]
    mcmc = args.strategy == "mcmc"
    params = init_splats(
        data,
        sh_degree=args.sh_degree,
        init_opacity=0.5 if mcmc else 0.1,
        init_scale=0.1 if mcmc else 1.0,
        device=device,
        stress=bool(args.init_random),
    )
    optimizers, scheduler = make_optimizers(params, data.scene_scale, args.iterations)
    strategy: DefaultStrategy | MCMCStrategy
    if mcmc:
        strategy = MCMCStrategy(
            cap_max=args.cap, refine_stop_iter=int(0.8 * args.iterations), verbose=False
        )
        state = strategy.initialize_state()
    else:
        strategy = DefaultStrategy(
            refine_stop_iter=int(0.5 * args.iterations),
            reset_every=min(3000, max(args.iterations // 10, 300)),
            verbose=False,
        )
        state = strategy.initialize_state(scene_scale=data.scene_scale)
    strategy.check_sanity(params, optimizers)
    prepare_seconds = time.perf_counter() - started

    generator = torch.Generator().manual_seed(args.seed)
    log: list[dict[str, float]] = []
    train_started = time.perf_counter()
    for step in range(args.iterations):
        index = train_ids[int(torch.randint(len(train_ids), (1,), generator=generator))]
        target = data.images[index].to(device, non_blocking=True).float() / 255.0
        sh_degree = min(step // 1000, args.sh_degree)
        prediction, info = render(params, data, index, sh_degree, packed=args.packed, device=device)
        loss = 0.8 * (prediction - target).abs().mean() + 0.2 * (
            1.0 - ssim_tensor(prediction.permute(2, 0, 1)[None], target.permute(2, 0, 1)[None])
        )
        if mcmc:
            loss = (
                loss
                + 0.01 * torch.sigmoid(params["opacities"]).mean()
                + 0.01 * torch.exp(params["scales"]).mean()
            )
        if isinstance(strategy, DefaultStrategy):
            strategy.step_pre_backward(params, optimizers, state, step, info)
        loss.backward()
        for optimizer in optimizers.values():
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        scheduler.step()
        if isinstance(strategy, MCMCStrategy):
            strategy.step_post_backward(
                params, optimizers, state, step, info, lr=scheduler.get_last_lr()[0]
            )
        else:
            strategy.step_post_backward(params, optimizers, state, step, info, packed=args.packed)
        if (step + 1) % args.log_every == 0 or step + 1 == args.iterations:
            torch.cuda.synchronize()
            log.append(
                {
                    "step": step + 1,
                    "seconds": round(time.perf_counter() - train_started, 2),
                    "gaussians": len(params["means"]),
                    "loss": round(float(loss.detach()), 4),
                    "peak_allocated_mib": round(torch.cuda.max_memory_allocated() / MIB, 1),
                    "peak_reserved_mib": round(torch.cuda.max_memory_reserved() / MIB, 1),
                }
            )
            print(json.dumps(log[-1]), flush=True)
    torch.cuda.synchronize()
    train_seconds = time.perf_counter() - train_started

    final_degree = args.sh_degree
    held_out = evaluate(params, data, holdout, final_degree, packed=args.packed, device=device)
    train_views = evaluate(
        params,
        data,
        train_ids[:: max(1, len(train_ids) // 8)],
        final_degree,
        packed=args.packed,
        device=device,
    )
    if args.save_renders:
        save_renders(
            params,
            data,
            holdout[: args.save_renders],
            final_degree,
            Path(args.out) / "renders",
            packed=args.packed,
            device=device,
        )
    free, total = torch.cuda.mem_get_info()
    return {
        "ok": True,
        "device": torch.cuda.get_device_name(0),
        "capability": list(torch.cuda.get_device_capability(0)),
        "strategy": args.strategy,
        "cap": args.cap if mcmc else None,
        "iterations": args.iterations,
        "train_width": data.width,
        "train_height": data.height,
        "sh_degree": args.sh_degree,
        "packed": args.packed,
        "images_total": len(data.names),
        "images_train": len(train_ids),
        "images_holdout": len(holdout),
        "init_points": len(data.points),
        "final_gaussians": len(params["means"]),
        "prepare_seconds": round(prepare_seconds, 2),
        "train_seconds": round(train_seconds, 2),
        "seconds_per_iteration": round(train_seconds / args.iterations, 4),
        "extrapolated_30k_minutes": round(train_seconds / args.iterations * 30000 / 60, 1),
        "peak_allocated_mib": round(torch.cuda.max_memory_allocated() / MIB, 1),
        "peak_reserved_mib": round(torch.cuda.max_memory_reserved() / MIB, 1),
        "device_used_after_mib": round((total - free) / MIB, 1),
        "heldout": held_out,
        "train_views": train_views,
        "log": log,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scene", required=True, help="COLMAP text model of the undistorted images"
    )
    parser.add_argument("--raw", required=True, help="directory with <image name>.rgb files")
    parser.add_argument("--out", required=True)
    parser.add_argument("--width", type=int, required=True, help="training image width in px")
    parser.add_argument("--iterations", type=int, default=7000)
    parser.add_argument("--strategy", choices=["default", "mcmc"], default="default")
    parser.add_argument("--cap", type=int, default=1_000_000, help="Gaussian cap (mcmc only)")
    parser.add_argument("--sh-degree", type=int, default=3)
    parser.add_argument("--packed", action="store_true")
    parser.add_argument(
        "--init-random",
        type=int,
        default=0,
        help="memory stress mode: start from N random Gaussians instead of the SfM cloud",
    )
    parser.add_argument("--holdout-every", type=int, default=8)
    parser.add_argument("--profile", default="low_vram_6gb")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=500)
    parser.add_argument(
        "--save-renders", type=int, default=0, help="number of held-out renders to write"
    )
    args = parser.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for required in ("cameras.txt", "images.txt", "points3D.txt"):
        if not (Path(args.scene) / required).is_file():
            print(f"missing input: {required}", file=sys.stderr)
            return 10
    try:
        result = train(args)
        code = 0
    except torch.cuda.OutOfMemoryError as error:
        result = {
            "ok": False,
            "error": "out of memory",
            "detail": str(error).splitlines()[0],
            "peak_allocated_mib": round(torch.cuda.max_memory_allocated() / MIB, 1),
            "peak_reserved_mib": round(torch.cuda.max_memory_reserved() / MIB, 1),
            "iterations": args.iterations,
            "train_width": args.width,
            "strategy": args.strategy,
            "cap": args.cap,
        }
        code = 30
    (out / "metrics.json").write_text(json.dumps(result, indent=2))
    summary = {k: v for k, v in result.items() if k != "log"}
    print(json.dumps(summary), flush=True)
    return code


if __name__ == "__main__":
    sys.exit(main())
