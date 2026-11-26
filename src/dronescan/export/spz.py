# SPDX-License-Identifier: AGPL-3.0-or-later
"""SPZ (version 2) writer and reader for Gaussian splats, numpy only (KAN-42).

SPZ is Niantic's compressed splat format (MIT, github.com/nianticlabs/spz), read by the
common web viewers. File = gzip of a 16-byte header followed by the attributes, each for all
points: positions (24-bit fixed point), alphas, colours, scales, rotations, SH coefficients.

SPZ stores the RUB frame (x right, y up, z back); our splats are in the COLMAP frame
(RDF: x right, y down, z forward). The conversion is a 180° rotation about x, see
`to_rub`; it is its own inverse.
"""

from __future__ import annotations

import gzip
import struct
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from dronescan.recon.scene import Splats

MAGIC = 0x5053474E  # "NGSP"
VERSION = 2
FRACTIONAL_BITS = 12  # position resolution 1/4096 m, range ±2048 m
_HEADER = struct.Struct("<IIIBBBB")
_COLOR_SCALE = 0.15  # SH DC -> uint8 as in the reference implementation
_SCALE_OFFSET, _SCALE_STEP = 10.0, 16.0  # log scale -> uint8
_SH1_BUCKET, _SH_REST_BUCKET = 8, 16  # 5 bits for degree 1, 4 bits for higher degrees
_SH_DEGREE1 = 3  # coefficients of degree 1
_GZIP_LEVEL = 9

# Sign of each real SH basis function under (x, y, z) -> (x, -y, -z), in the order of the
# 3DGS reference (degree 1: y, z, x; degree 2: xy, yz, 3z²-1, xz, x²-y²; degree 3: ...).
_SH_FLIP_RDF_RUB = np.array([-1, -1, 1, -1, 1, 1, -1, 1, -1, 1, -1, -1, 1, -1, 1], dtype=np.float32)


def to_rub(splats: Splats) -> Splats:
    """Rotate splats by 180° about x (COLMAP RDF <-> SPZ RUB); the inverse is the same call."""
    flip = np.array([1, -1, -1], dtype=np.float32)
    quat_flip = np.array([1, 1, -1, -1], dtype=np.float32)  # q' = S q S with S = diag(1,-1,-1)
    rest = splats.sh_rest.shape[1]
    return Splats(
        means=splats.means * flip,
        sh0=splats.sh0,
        sh_rest=splats.sh_rest * _SH_FLIP_RDF_RUB[:rest, None],
        opacity_logits=splats.opacity_logits,
        log_scales=splats.log_scales,
        quats_wxyz=splats.quats_wxyz * quat_flip,
    )


def _to_uint8(values: NDArray[np.floating]) -> NDArray[np.uint8]:
    return np.clip(np.round(values), 0, 255).astype(np.uint8)


def _quantize_sh(values: NDArray[np.float32], bucket: int) -> NDArray[np.uint8]:
    q = np.round(values.astype(np.float64) * 128.0 + 128.0)
    q = np.floor((q + bucket / 2) / bucket) * bucket
    return np.clip(q, 0, 255).astype(np.uint8)


def encode(splats: Splats) -> bytes:
    """SPZ file content (gzip, deterministic) for splats in the COLMAP frame."""
    rub = to_rub(splats)
    count, rest = len(rub), rub.sh_rest.shape[1]

    fixed = np.round(rub.means.astype(np.float64) * (1 << FRACTIONAL_BITS)).astype(np.int64)
    fixed = np.clip(fixed, -(1 << 23), (1 << 23) - 1) & 0xFFFFFF
    positions = np.stack([(fixed >> shift) & 0xFF for shift in (0, 8, 16)], axis=-1)

    alphas = _to_uint8(255.0 / (1.0 + np.exp(-rub.opacity_logits.astype(np.float64))))
    colors = _to_uint8(rub.sh0[:, 0, :] * (_COLOR_SCALE * 255.0) + 127.5)
    scales = _to_uint8((rub.log_scales + _SCALE_OFFSET) * _SCALE_STEP)

    quats = rub.quats_wxyz.astype(np.float64)
    quats /= np.maximum(np.linalg.norm(quats, axis=1, keepdims=True), 1e-12)
    sign = np.where(quats[:, :1] < 0, -1.0, 1.0)  # w >= 0, so w follows from x, y, z
    rotations = _to_uint8(quats[:, 1:] * sign * 127.5 + 127.5)

    sh = np.empty((count, rest, 3), dtype=np.uint8)
    sh[:, :_SH_DEGREE1] = _quantize_sh(rub.sh_rest[:, :_SH_DEGREE1], _SH1_BUCKET)
    sh[:, _SH_DEGREE1:] = _quantize_sh(rub.sh_rest[:, _SH_DEGREE1:], _SH_REST_BUCKET)

    header = _HEADER.pack(MAGIC, VERSION, count, rub.sh_degree, FRACTIONAL_BITS, 0, 0)
    body = b"".join(
        a.astype(np.uint8).tobytes() for a in (positions, alphas, colors, scales, rotations, sh)
    )
    return gzip.compress(header + body, compresslevel=_GZIP_LEVEL, mtime=0)


def write_spz(path: Path, splats: Splats) -> int:
    """Write ``path`` atomically; returns the file size in bytes."""
    data = encode(splats)
    tmp = path.with_suffix(".tmp")
    tmp.write_bytes(data)
    tmp.replace(path)
    return len(data)


def decode(data: bytes) -> Splats:
    """Splats in the COLMAP frame from SPZ version 2 content (raises ValueError if invalid)."""
    try:
        raw = gzip.decompress(data)
    except (OSError, EOFError) as exc:
        raise ValueError("not a gzip file") from exc
    if len(raw) < _HEADER.size:
        raise ValueError("SPZ header truncated")
    magic, version, count, degree, bits, _flags, _ = _HEADER.unpack_from(raw)
    if magic != MAGIC or version != VERSION or degree > 3:
        raise ValueError(f"unsupported SPZ (magic {magic:#x}, version {version})")
    rest = (degree + 1) ** 2 - 1
    sizes = [count * n for n in (9, 1, 3, 3, 3, rest * 3)]
    if len(raw) != _HEADER.size + sum(sizes):
        raise ValueError("SPZ size does not match its header")
    offsets = np.cumsum([_HEADER.size, *sizes])
    pos, alpha, color, scale, rot, sh = (
        np.frombuffer(raw, np.uint8, size, offset)
        for size, offset in zip(sizes, offsets[:-1], strict=True)
    )

    b = pos.reshape(count, 3, 3).astype(np.int64)
    fixed = b[..., 0] | (b[..., 1] << 8) | (b[..., 2] << 16)
    fixed = np.where(fixed & 0x800000, fixed - (1 << 24), fixed)
    xyz = rot.reshape(count, 3).astype(np.float64) / 127.5 - 1.0
    w = np.sqrt(np.maximum(0.0, 1.0 - np.sum(xyz * xyz, axis=1, keepdims=True)))
    a = alpha.astype(np.float64).clip(0.5, 254.5) / 255.0  # 0 and 255 are not invertible
    rub = Splats(
        means=(fixed / float(1 << bits)).astype(np.float32),
        sh0=((color.reshape(count, 1, 3) / 255.0 - 0.5) / _COLOR_SCALE).astype(np.float32),
        sh_rest=((sh.reshape(count, rest, 3).astype(np.float32) - 128.0) / 128.0),
        opacity_logits=np.log(a / (1.0 - a)).astype(np.float32),
        log_scales=(scale.reshape(count, 3) / _SCALE_STEP - _SCALE_OFFSET).astype(np.float32),
        quats_wxyz=np.concatenate([w, xyz], axis=1).astype(np.float32),
    )
    return to_rub(rub)


def read_spz(path: Path) -> Splats:
    return decode(path.read_bytes())
