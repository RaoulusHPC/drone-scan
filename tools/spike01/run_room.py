"""Run one reference room through ingest, SfM and gsplat training on the GTX 1060 (host).

Spike KAN-19. Orchestrates ffmpeg (host), the `colmap` and `worker` containers, one stage at a
time (no parallel GPU jobs), and records per stage the wall time and the GPU memory taken on
GPU 0 (nvidia-smi: whole-process total including the CUDA context, minus the display's baseline).

Privacy: recordings, frames, COLMAP models, splats and renders stay in --work (outside the
repository and outside workspace/). Only numbers are printed and written to results.json.

Example:
    python tools/spike01/run_room.py --video data/samples/kurz_1.mp4 --room kurz_1 \
        --work /tmp/.../spike01 ingest sfm prep train
"""

from __future__ import annotations

import argparse
import fcntl
import json
import shutil
import subprocess
import sys
import threading
import time
from collections.abc import Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.spike01.colmap_text import (  # noqa: E402
    parse_cameras_txt,
    parse_images_txt,
    registration_stats,
)
from tools.spike01.frames import (  # noqa: E402
    analysis_size,
    colmap_option_args,
    display_size,
    ffmpeg_select_expression,
    scale_pinhole,
    select_keyframes,
)

GPU_LOCK: Path | None = None  # set in main(): one GPU stage at a time across all running jobs
GPU_INDEX = "0"  # GTX 1060 (nvidia-smi / PCI bus order); the RTX 2070 SUPER is not used
ANALYSIS_FPS = 6
ANALYSIS_EDGE = 480


class GpuSampler:
    """Polls the used memory of GPU 0 while a stage runs (MiB above the idle baseline)."""

    def __init__(self) -> None:
        self.baseline = self._used()
        self.peak = self.baseline
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    @staticmethod
    def _used() -> int:
        out = subprocess.run(
            [
                "nvidia-smi",
                "-i",
                GPU_INDEX,
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        return int(out.strip())

    def _run(self) -> None:
        while not self._stop.is_set():
            self.peak = max(self.peak, self._used())
            self._stop.wait(0.4)

    def __enter__(self) -> GpuSampler:
        self._thread.start()
        return self

    def __exit__(self, *_: object) -> None:
        self._stop.set()
        self._thread.join()
        self.peak = max(self.peak, self._used())

    @property
    def over_baseline_mib(self) -> int:
        return self.peak - self.baseline


class Recorder:
    """Appends one entry per stage to <room>/results.json (numbers only)."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.data: dict[str, object] = json.loads(path.read_text()) if path.is_file() else {}

    def set(self, key: str, value: object) -> None:
        self.data[key] = value
        self.path.write_text(json.dumps(self.data, indent=2))

    @contextmanager
    def stage(self, name: str, *, gpu: bool) -> Iterator[None]:
        lock = None
        if gpu and GPU_LOCK is not None:  # GPU jobs never run in parallel (CLAUDE.md rule 2)
            lock = GPU_LOCK.open("w")
            fcntl.flock(lock, fcntl.LOCK_EX)  # waits if another job holds the GPU
        started = time.perf_counter()
        sampler = GpuSampler() if gpu else None
        if sampler:
            sampler.__enter__()
        try:
            yield
        finally:
            entry: dict[str, object] = {"seconds": round(time.perf_counter() - started, 1)}
            if sampler:
                sampler.__exit__()
                entry["gpu_over_baseline_mib"] = sampler.over_baseline_mib
                entry["gpu_baseline_mib"] = sampler.baseline
            if lock:
                lock.close()  # releases the lock
            stages = self.data.setdefault("stages", {})
            assert isinstance(stages, dict)
            stages[name] = entry
            self.set("stages", stages)
            print(f"[{name}] {entry}", flush=True)


def run(cmd: Sequence[str | Path]) -> subprocess.CompletedProcess[str]:
    """Run a command from the repository root (docker compose finds compose.yaml there)."""
    result = subprocess.run(
        [str(part) for part in cmd], cwd=ROOT, capture_output=True, text=True, check=False
    )
    if result.returncode != 0:
        tail = (result.stderr or result.stdout)[-1500:]
        raise SystemExit(f"command failed ({result.returncode}): {cmd[0]} ...\n{tail}")
    return result


def compose(service: str, work: Path, *command: str, extra: Sequence[str] = ()) -> list[str]:
    return [
        "docker", "compose", "run", "--rm", "-T",
        "-v", f"{work}:/work",
        "-v", f"{ROOT / 'tools'}:/tools:ro",
        "-e", "PYTHONPATH=/",
        *extra,
        service, *command,
    ]  # fmt: skip


def probe(video: Path) -> dict[str, float]:
    out = run([
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=width,height:stream_side_data=rotation:format=duration",
        "-of", "json", video,
    ]).stdout  # fmt: skip
    data = json.loads(out)
    stream = data["streams"][0]
    rotation = int(stream.get("side_data_list", [{}])[0].get("rotation", 0))
    width, height = display_size(int(stream["width"]), int(stream["height"]), rotation)
    return {
        "coded_width": int(stream["width"]),
        "coded_height": int(stream["height"]),
        "rotation": rotation,
        "display_width": width,
        "display_height": height,
        "duration_s": round(float(data["format"]["duration"]), 2),
    }


def stage_ingest(args: argparse.Namespace, room: Path, rec: Recorder) -> None:
    info = probe(Path(args.video))
    rec.set("video", info)
    width, height = int(info["display_width"]), int(info["display_height"])
    small_w, small_h = analysis_size(width, height, long_edge=ANALYSIS_EDGE)
    with rec.stage("ingest", gpu=False):
        analysis = room / "analysis.gray"
        run([
            "ffmpeg", "-v", "error", "-y", "-i", args.video,
            "-vf", f"fps={ANALYSIS_FPS},scale={small_w}:{small_h},format=gray",
            "-f", "rawvideo", "-pix_fmt", "gray", analysis,
        ])  # fmt: skip
        out = run(
            compose("worker", room.parent, "python", "-m", "tools.spike01.sharpness",
                    f"/work/{room.name}/analysis.gray", str(small_w), str(small_h))
        ).stdout  # fmt: skip
        scores = json.loads(out.strip().splitlines()[-1])
        window = max(1, round(ANALYSIS_FPS / args.fps))
        picked = select_keyframes(scores, window=window, min_rel_sharpness=args.min_sharpness)

        frames = room / "frames"
        shutil.rmtree(frames, ignore_errors=True)
        frames.mkdir(parents=True)
        long_edge = args.colmap_edge
        scale = f"scale={long_edge}:-2" if width >= height else f"scale=-2:{long_edge}"
        script = room / "select.ffscript"
        script.write_text(f"fps={ANALYSIS_FPS},select='{ffmpeg_select_expression(picked)}',{scale}")
        run([
            "ffmpeg", "-v", "error", "-y", "-i", args.video,
            "-filter_script:v", script, "-fps_mode", "passthrough", "-qscale:v", "2",
            frames / "f_%05d.jpg",
        ])  # fmt: skip
    sorted_scores = sorted(scores)
    rec.set(
        "ingest",
        {
            "analysis_fps": ANALYSIS_FPS,
            "analysis_frames": len(scores),
            "window": window,
            "min_rel_sharpness": args.min_sharpness,
            "windows_total": -(-len(scores) // window),
            "frames_selected": len(picked),
            "frames_written": len(list(frames.glob("*.jpg"))),
            "sharpness_median": sorted_scores[len(scores) // 2],
            "sharpness_min": sorted_scores[0],
            "sharpness_max": sorted_scores[-1],
            "colmap_edge_px": long_edge,
        },
    )


def colmap(room: Path, *command: str) -> None:
    run(compose("colmap", room.parent, "colmap", *command))


def stage_sfm(args: argparse.Namespace, room: Path, rec: Recorder) -> None:
    work = f"/work/{room.name}"
    frames_total = len(list((room / "frames").glob("*.jpg")))
    db = f"{work}/db.db"
    for leftover in (room / "db.db", room / "sparse", room / "sparse_txt", room / "undist"):
        if leftover.is_dir():
            shutil.rmtree(leftover)
        else:
            leftover.unlink(missing_ok=True)
    (room / "sparse").mkdir()

    with rec.stage("sfm_features", gpu=True):
        colmap(
            room, "feature_extractor", "--database_path", db, "--image_path", f"{work}/frames",
            "--ImageReader.camera_model", "SIMPLE_RADIAL", "--ImageReader.single_camera", "1",
            "--FeatureExtraction.use_gpu", "1", "--FeatureExtraction.gpu_index", GPU_INDEX,
            "--FeatureExtraction.max_image_size", str(args.colmap_edge),
            "--SiftExtraction.max_num_features", str(args.max_features),
            *colmap_option_args("SiftExtraction", args.sift_opt),
        )  # fmt: skip
    matcher_options = (
        ["--SequentialMatching.overlap", str(args.overlap)] if args.matcher == "sequential" else []
    )
    with rec.stage("sfm_matching", gpu=True):
        colmap(
            room, f"{args.matcher}_matcher", "--database_path", db,
            "--FeatureMatching.use_gpu", "1", "--FeatureMatching.gpu_index", GPU_INDEX,
            *matcher_options,
        )  # fmt: skip
    with rec.stage("sfm_mapper", gpu=False):
        colmap(
            room, "mapper", "--database_path", db, "--image_path", f"{work}/frames",
            "--output_path", f"{work}/sparse", "--Mapper.ba_use_gpu", "0",
            *colmap_option_args("Mapper", args.mapper_opt),
        )  # fmt: skip

    models = []
    for model in sorted(p for p in (room / "sparse").iterdir() if p.is_dir()):
        out = room / "sparse_txt" / model.name
        out.mkdir(parents=True)
        colmap(
            room, "model_converter", "--input_path", f"{work}/sparse/{model.name}",
            "--output_path", f"{work}/sparse_txt/{model.name}", "--output_type", "TXT",
        )  # fmt: skip
        images = parse_images_txt((out / "images.txt").read_text())
        stats = registration_stats(images, frames_total)
        models.append(
            {"model": model.name, "registered": stats.registered, "percent": stats.percent}
        )
    if not models:
        rec.set("sfm", {"models": [], "registered_percent": 0.0, "frames_total": frames_total})
        raise SystemExit("SfM failed: COLMAP produced no model")
    best = max(models, key=lambda item: int(item["registered"]))  # type: ignore[call-overload]
    rec.set(
        "sfm",
        {
            "frames_total": frames_total,
            "models": models,
            "best_model": best["model"],
            "registered": best["registered"],
            "registered_percent": best["percent"],
        },
    )
    print(f"[sfm] registered {best['registered']}/{frames_total} = {best['percent']} %", flush=True)


def stage_prep(args: argparse.Namespace, room: Path, rec: Recorder) -> None:
    sfm = rec.data["sfm"]
    assert isinstance(sfm, dict)
    work = f"/work/{room.name}"
    with rec.stage("undistort", gpu=False):
        colmap(
            room, "image_undistorter", "--image_path", f"{work}/frames",
            "--input_path", f"{work}/sparse/{sfm['best_model']}", "--output_path", f"{work}/undist",
            "--output_type", "COLMAP",
        )  # fmt: skip
        (room / "undist" / "sparse_txt").mkdir(exist_ok=True)
        colmap(
            room, "model_converter", "--input_path", f"{work}/undist/sparse",
            "--output_path", f"{work}/undist/sparse_txt", "--output_type", "TXT",
        )  # fmt: skip
    scene = room / "undist" / "sparse_txt"
    camera = next(iter(parse_cameras_txt((scene / "cameras.txt").read_text()).values()))
    scaled = scale_pinhole(camera, new_width=args.train_width)
    raw = room / "raw" / f"w{args.train_width}"
    shutil.rmtree(raw, ignore_errors=True)
    raw.mkdir(parents=True)

    def convert(image: Path) -> None:
        run([
            "ffmpeg", "-v", "error", "-y", "-i", image,
            "-vf", f"scale={scaled.width}:{scaled.height}",
            "-f", "rawvideo", "-pix_fmt", "rgb24", raw / f"{image.name}.rgb",
        ])  # fmt: skip

    with rec.stage(f"prep_raw_w{args.train_width}", gpu=False), ThreadPoolExecutor(16) as pool:
        list(pool.map(convert, sorted((room / "undist" / "images").glob("*.jpg"))))
    rec.set(
        f"prep_w{args.train_width}",
        {
            "undistorted_width": camera.width,
            "undistorted_height": camera.height,
            "train_width": scaled.width,
            "train_height": scaled.height,
            "images": len(list(raw.glob("*.rgb"))),
        },
    )


def stage_train(args: argparse.Namespace, room: Path, rec: Recorder) -> None:
    tag = args.tag or f"{args.strategy}_w{args.train_width}_i{args.iterations}"
    out = f"/work/{room.name}/train_{tag}"
    cmd = [
        "python", "-m", "tools.spike01.train_gsplat",
        "--scene", f"/work/{room.name}/undist/sparse_txt",
        "--raw", f"/work/{room.name}/raw/w{args.train_width}", "--out", out,
        "--width", str(args.train_width), "--iterations", str(args.iterations),
        "--strategy", args.strategy, "--cap", str(args.cap), "--sh-degree", str(args.sh_degree),
        "--save-renders", str(args.save_renders),
        "--log-every", str(max(100, args.iterations // 10)),
    ]  # fmt: skip
    if args.packed:
        cmd.append("--packed")
    if args.init_random:
        cmd += ["--init-random", str(args.init_random)]
    stage = f"train_{tag}"
    with rec.stage(stage, gpu=True):
        result = subprocess.run(
            compose("worker", room.parent, *cmd, extra=["-e", "DRONESCAN_VRAM_LIMIT_GB=5.0"]),
            cwd=ROOT, capture_output=True, text=True, check=False,
        )  # fmt: skip
    metrics_path = room / f"train_{tag}" / "metrics.json"
    metrics = json.loads(metrics_path.read_text()) if metrics_path.is_file() else {"ok": False}
    metrics.pop("log", None)
    stages = rec.data["stages"]
    assert isinstance(stages, dict)
    metrics["gpu_over_baseline_mib"] = stages[stage]["gpu_over_baseline_mib"]
    metrics["exit_code"] = result.returncode
    rec.set(f"train_{tag}", metrics)
    print(f"[train {tag}] {json.dumps(metrics)}", flush=True)
    if result.returncode not in (0, 30):
        raise SystemExit(f"training failed ({result.returncode}):\n{result.stderr[-1500:]}")


def stage_compare(args: argparse.Namespace, room: Path, rec: Recorder) -> None:
    """Side-by-side PNGs (held-out frame | render) for the PO's local visual check only."""
    tag = args.tag or f"{args.strategy}_w{args.train_width}_i{args.iterations}"
    renders = room / f"train_{tag}" / "renders"
    prep = rec.data[f"prep_w{args.train_width}"]
    assert isinstance(prep, dict)
    size = f"{prep['train_width']}x{prep['train_height']}"
    target = room / "compare"
    target.mkdir(exist_ok=True)
    for render in sorted(renders.glob("*.render.rgb")):
        name = render.name.removesuffix(".render.rgb")
        run([
            "ffmpeg", "-v", "error", "-y",
            "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", size, "-i", renders / f"{name}.gt.rgb",
            "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", size, "-i", render,
            "-filter_complex", "hstack", target / f"{tag}_{name}.png",
        ])  # fmt: skip
    print(f"[compare] {len(list(target.glob('*.png')))} comparison images (local only)", flush=True)


STAGES = {
    "ingest": stage_ingest,
    "sfm": stage_sfm,
    "prep": stage_prep,
    "train": stage_train,
    "compare": stage_compare,
}


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("stages", nargs="+", choices=list(STAGES))
    parser.add_argument("--video", required=True)
    parser.add_argument("--room", required=True)
    parser.add_argument("--work", required=True, help="private work directory outside the repo")
    parser.add_argument("--fps", type=float, default=2.0, help="target keyframe rate")
    parser.add_argument("--min-sharpness", type=float, default=0.4)
    parser.add_argument("--colmap-edge", type=int, default=1600)
    parser.add_argument("--max-features", type=int, default=8192)
    parser.add_argument("--overlap", type=int, default=10)
    parser.add_argument("--matcher", choices=["sequential", "exhaustive"], default="sequential")
    parser.add_argument(
        "--sift-opt",
        action="append",
        default=[],
        help="SiftExtraction option, e.g. peak_threshold=0.003",
    )
    parser.add_argument(
        "--mapper-opt", action="append", default=[], help="Mapper option, e.g. init_min_tri_angle=8"
    )
    parser.add_argument("--train-width", type=int, default=1600)
    parser.add_argument("--iterations", type=int, default=7000)
    parser.add_argument("--strategy", choices=["default", "mcmc"], default="default")
    parser.add_argument("--cap", type=int, default=1_000_000)
    parser.add_argument("--sh-degree", type=int, default=3)
    parser.add_argument("--packed", action="store_true")
    parser.add_argument(
        "--init-random", type=int, default=0, help="memory stress mode, see trainer"
    )
    parser.add_argument("--save-renders", type=int, default=0)
    parser.add_argument("--tag", default="")
    args = parser.parse_args()

    work = Path(args.work)
    if ROOT in work.resolve().parents or work.resolve() == ROOT:
        raise SystemExit("--work must be outside the repository (private recordings)")
    work.mkdir(parents=True, exist_ok=True)
    global GPU_LOCK
    GPU_LOCK = work / ".gpu.lock"
    room = work / args.room
    room.mkdir(parents=True, exist_ok=True)
    recorder = Recorder(room / "results.json")
    for name in args.stages:
        STAGES[name](args, room, recorder)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
