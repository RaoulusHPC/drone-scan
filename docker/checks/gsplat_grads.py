"""Gradient check for the gsplat Pascal patch (docker/patches/gsplat-v1.5.3-pascal.patch).

Runs inside the worker container (`docker compose run -T worker python - < this file`).
Compares the backward pass of the patched projection kernels (3DGS and 2DGS, fused and
packed) with a float64 PyTorch reference and packed with fused, as relative L2 errors per
gradient. Tolerances instead of exact values. Synthetic data only. Prints one JSON line.

Why float64 and L2: gsplat's 3DGS kernels match float64 to float32 precision (~1e-6), so a
broken reduction (missing or doubled atomicAdd) shows up as a large error. The 2DGS kernels
approximate the gradients of quats/scales/viewmats by ~0.2-2 % per gaussian (not caused by the
patch: `means`, which uses the same reduction, is exact, and packed equals fused), so those
get wider limits.
"""

import json
import math
import os
import sys

import torch
from gsplat.cuda._torch_impl import _fully_fused_projection
from gsplat.cuda._torch_impl_2dgs import _fully_fused_projection_2dgs
from gsplat.cuda._wrapper import (
    fully_fused_projection,
    fully_fused_projection_2dgs,
    quat_scale_to_covar_preci,
)

WIDTH = HEIGHT = 256


def limit_vram() -> None:
    limit_gb = float(os.environ.get("DRONESCAN_VRAM_LIMIT_GB", "5.0"))
    total = torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction(min(1.0, limit_gb * 1024**3 / total), 0)


def rotation(axis: torch.Tensor, angle: float) -> torch.Tensor:
    axis = axis / axis.norm()
    x, y, z = axis.tolist()
    k = torch.tensor([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
    return torch.eye(3) + math.sin(angle) * k + (1 - math.cos(angle)) * (k @ k)


def scene(n: int, c: int, seed: int) -> dict[str, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    means = torch.rand(n, 3, generator=g) * 2 - 1
    means[:, 2] *= 0.5
    quats = torch.nn.functional.normalize(torch.randn(n, 4, generator=g), dim=-1)
    scales = torch.exp(torch.rand(n, 3, generator=g) * 1.5 - 3.0)
    viewmats = torch.eye(4).repeat(c, 1, 1)
    for i in range(c):
        viewmats[i, :3, :3] = rotation(torch.randn(3, generator=g), 0.1 * (i + 1))
        viewmats[i, :3, 3] = torch.tensor([0.1 * i, -0.05 * i, 4.0])
    ks = torch.tensor([[300.0, 0.0, WIDTH / 2], [0.0, 300.0, HEIGHT / 2], [0.0, 0.0, 1.0]])
    data = {"means": means, "quats": quats, "scales": scales, "viewmats": viewmats}
    out = {k: v.cuda().requires_grad_(True) for k, v in data.items()}
    out["Ks"] = ks.repeat(c, 1, 1).cuda()
    return out


INPUTS = ("viewmats", "quats", "scales", "means")


def grads(outputs: list[torch.Tensor], weights: list[torch.Tensor], s: dict[str, torch.Tensor]):
    loss = sum((o * w).sum() for o, w in zip(outputs, weights, strict=True))
    return dict(zip(INPUTS, torch.autograd.grad(loss, [s[k] for k in INPUTS]), strict=True))


def random_weights(outputs: list[torch.Tensor], valid: torch.Tensor, seed: int):
    g = torch.Generator(device="cuda").manual_seed(seed)
    weights = []
    for o in outputs:
        mask = valid.reshape(valid.shape + (1,) * (o.dim() - valid.dim()))
        weights.append(torch.randn(o.shape, generator=g, device="cuda") * mask)
    return weights


def rel_l2(got: torch.Tensor, ref: torch.Tensor) -> float:
    got, ref = got.double(), ref.double()
    return ((got - ref).norm() / ref.norm().clamp_min(1e-12)).item()


def compare(name: str, got: dict, ref: dict, limits: dict, report: dict) -> None:
    for key in INPUTS:
        err = rel_l2(got[key], ref[key])
        report[f"{name}:{key}"] = max(err, report.get(f"{name}:{key}", 0.0))
        assert err <= limits[key], f"{name}:{key} rel. L2 error {err:.2e} > {limits[key]:.0e}"


def to_f64(s: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    out = {k: v.detach().double().requires_grad_(k in INPUTS) for k, v in s.items() if k != "Ks"}
    out["Ks"] = s["Ks"].double()
    return out


STRICT = dict.fromkeys(INPUTS, 1e-4)  # float32 noise level is ~1e-6
LIMITS_2DGS = {"means": 1e-4, "viewmats": 1e-2, "scales": 1e-2, "quats": 5e-2}


def check_3dgs(s: dict[str, torch.Tensor], report: dict) -> None:
    args = (s["viewmats"], s["Ks"], WIDTH, HEIGHT)
    radii, *outs = fully_fused_projection(s["means"], None, s["quats"], s["scales"], *args)[:4]
    covars, _ = quat_scale_to_covar_preci(s["quats"], s["scales"], triu=False)
    ref_radii = _fully_fused_projection(s["means"], covars, *args)[0]
    valid = (radii > 0).all(-1) & (ref_radii > 0).all(-1)
    assert valid.sum() > 0, "no visible Gaussians"
    w = random_weights(outs, valid, 1)
    fused = grads(outs, w, s)

    s64 = to_f64(s)
    covars64, _ = quat_scale_to_covar_preci(s64["quats"], s64["scales"], triu=False)
    args64 = (s64["viewmats"], s64["Ks"], WIDTH, HEIGHT)
    _, *outs64 = _fully_fused_projection(s64["means"], covars64, *args64)[:4]
    ref = grads(outs64, [x.double() for x in w], s64)
    compare("3dgs_fused_vs_f64", fused, ref, STRICT, report)

    packed = fully_fused_projection(
        s["means"], None, s["quats"], s["scales"], *args, packed=True
    )
    _, cam, gid, _, *p_outs = packed[:7]
    p_w = [wi[cam, gid] for wi in w]  # same upstream gradients (already masked by valid)
    compare("3dgs_packed_vs_fused", grads(p_outs, p_w, s), fused, STRICT, report)


def check_2dgs(s: dict[str, torch.Tensor], report: dict) -> None:
    args = (s["means"], s["quats"], s["scales"], s["viewmats"], s["Ks"], WIDTH, HEIGHT)
    radii, *outs = fully_fused_projection_2dgs(*args)
    ref_radii = _fully_fused_projection_2dgs(*args)[0]
    valid = (radii > 0).all(-1) & (ref_radii > 0).all(-1)
    assert valid.sum() > 0, "no visible Gaussians"
    w = random_weights(outs, valid, 2)
    fused = grads(outs, w, s)

    s64 = to_f64(s)
    args64 = (s64["means"], s64["quats"], s64["scales"], s64["viewmats"], s64["Ks"], WIDTH, HEIGHT)
    _, *outs64 = _fully_fused_projection_2dgs(*args64)
    ref = grads(outs64, [x.double() for x in w], s64)
    compare("2dgs_fused_vs_f64", fused, ref, LIMITS_2DGS, report)

    packed = fully_fused_projection_2dgs(*args, packed=True)
    _, cam, gid, _, *p_outs = packed
    p_w = [wi[cam, gid] for wi in w]  # same upstream gradients (already masked by valid)
    compare("2dgs_packed_vs_fused", grads(p_outs, p_w, s), fused, STRICT, report)


def main() -> int:
    limit_vram()
    report: dict = {"capability": list(torch.cuda.get_device_capability(0))}
    # Few Gaussians and several cameras: threads of one warp share Gaussian and camera ids,
    # which is exactly where the original kernels reduce with labeled_partition.
    for n, c, seed in ((8, 4, 0), (2000, 3, 1), (2000, 3, 7)):
        s = scene(n, c, seed)
        check_3dgs(s, report)
        check_2dgs(s, report)
    report["peak_vram_mb"] = round(torch.cuda.max_memory_allocated(0) / 2**20, 1)
    print(json.dumps({"ok": True, **report}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
