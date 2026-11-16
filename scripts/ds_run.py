"""Run one bundle through deepstream-app in a DeepStream container, for the GPU checks.

Copies the bundle (without engines or libraries) to build/ds_runs/<name>/bundle, builds its parser, runs
deepstream-app on a video with KITTI output, and records the steady-state FPS. Options: --dump (nvinfer
dump-input-tensor=1, for scripts/check_color_range.py and scripts/same_input_check.py), --mp4 (encoded output with
boxes, and masks for segmentation bundles).

    python scripts/ds_run.py --image nvcr.io/nvidia/deepstream:7.0-triton-multiarch --bundle PATH --name ds7_fp16 \
        [--video /opt/nvidia/deepstream/deepstream/samples/streams/sample_720p.mp4] [--dump] [--mp4]

The video can be a path inside the container (the samples) or a file in this repo. Results:
build/ds_runs/<name>/{kitti/, ip_tensor_dump.bin, out.mp4, run.log, summary.json}.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SAMPLE = "/opt/nvidia/deepstream/deepstream/samples/streams/sample_720p.mp4"


def app_config(video_uri: str, width: int, height: int, gie: str, kitti: str, mp4: str | None, masks: bool) -> str:
    sink = (f"type=3\ncontainer=1\ncodec=1\nenc-type=0\nsync=0\nbitrate=6000000\noutput-file={mp4}"
            if mp4 else "type=1\nsync=0")
    return f"""[application]
enable-perf-measurement=1
perf-measurement-interval-sec=1
gie-kitti-output-dir={kitti}

[source0]
enable=1
type=3
uri={video_uri}
num-sources=1
gpu-id=0

[streammux]
gpu-id=0
batch-size=1
batched-push-timeout=40000
width={width}
height={height}
live-source=0

[primary-gie]
enable=1
gpu-id=0
batch-size=1
gie-unique-id=1
config-file={gie}

[osd]
enable=1
gpu-id=0
border-width=2
text-size=12
display-mask={1 if masks else 0}

[tiled-display]
enable=0

[sink0]
enable=1
{sink}
"""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--image", required=True, help="DeepStream Docker image")
    ap.add_argument("--bundle", type=Path, required=True, help="bundle folder (rfdetr-deepstream build -o ...)")
    ap.add_argument("--name", required=True, help="run name: build/ds_runs/<name>")
    ap.add_argument("--video", default=SAMPLE, help="video path in the container, or a file inside this repo")
    ap.add_argument("--dump", action="store_true", help="dump nvinfer's input tensor (large: W*H*12 bytes per frame)")
    ap.add_argument("--mp4", action="store_true", help="write out.mp4 with the detections drawn")
    ap.add_argument("--tf32", action="store_true", help="allow TF32 (default: NVIDIA_TF32_OVERRIDE=0, for parity)")
    args = ap.parse_args()

    if not str(args.video).startswith("/opt/") and not Path(args.video).exists():
        raise SystemExit(f"{args.video} not found")
    run = ROOT / "build" / "ds_runs" / args.name
    shutil.rmtree(run, ignore_errors=True)
    (run / "kitti").mkdir(parents=True)
    shutil.copytree(args.bundle, run / "bundle", ignore=shutil.ignore_patterns("*.engine", "*.so", "*_dump.txt"))
    cfg_path = run / "bundle" / "config_infer_rfdetr.txt"
    cfg = cfg_path.read_text()
    if args.dump:
        cfg = cfg.replace("[property]\n", "[property]\ndump-input-tensor=1\n", 1)
        cfg_path.write_text(cfg)
    onnx_name = re.search(r"^onnx-file=(.+)$", cfg, re.M).group(1)
    m = re.search(r"_(\d+)x(\d+)\.onnx$", onnx_name)
    if not m:
        raise SystemExit("bundle has no --frame-size; this helper expects one (nvstreammux size = frame size)")
    width, height = int(m.group(1)), int(m.group(2))
    masks = "output-instance-mask=1" in cfg
    pose = (run / "bundle" / "pose").exists()
    if pose:
        raise SystemExit("keypoint bundles need nvdsvideotemplate: use scripts/test_deepstream_version.py")

    w = f"/work/build/ds_runs/{args.name}"
    video = Path(args.video)
    extra_mounts = []
    if str(video).startswith("/opt/"):                      # inside the container (the samples)
        uri = f"file://{args.video}"
    elif video.resolve().is_relative_to(ROOT):
        uri = f"file:///work/{video.resolve().relative_to(ROOT)}"
    else:                                                    # elsewhere on the host: mount its folder
        extra_mounts = ["-v", f"{video.resolve().parent}:/video:ro"]
        uri = f"file:///video/{video.name}"
    (run / "app.txt").write_text(app_config(uri, width, height, f"{w}/bundle/config_infer_rfdetr.txt",
                                            f"{w}/kitti", f"{w}/out.mp4" if args.mp4 else None, masks))
    cmd = (f"make -C {w}/bundle/parser >/dev/null && cd {w} && "
           f"deepstream-app -c {w}/app.txt")
    docker = ["docker", "run", "--rm", "--gpus", "all", "-u", f"{os.getuid()}:{os.getgid()}", "-e", "HOME=/tmp",
              "-e", "NVIDIA_DRIVER_CAPABILITIES=compute,utility,video", "-v", f"{ROOT}:/work", *extra_mounts, "-w", w]
    if not args.tf32:
        docker += ["-e", "NVIDIA_TF32_OVERRIDE=0"]
    docker += ["--entrypoint", "bash", args.image, "-c", cmd]
    print(f"running {args.name}: {args.image}, {width}x{height}{', dump' if args.dump else ''}"
          f"{', mp4' if args.mp4 else ''}{', masks' if masks else ''} (first run builds the engine: minutes)")
    proc = subprocess.run(docker, capture_output=True, text=True)
    log = proc.stdout + proc.stderr
    (run / "run.log").write_text(log)

    fps = [float(x) for x in re.findall(r"\*\*PERF:\s+([\d.]+)\s+\(", log)]
    steady = fps[2:-1] if len(fps) > 4 else fps  # skip warm-up and the last partial interval
    summary = {
        "name": args.name, "image": args.image, "bundle": str(args.bundle), "video": args.video,
        "exit_code": proc.returncode, "frames_with_kitti": len(list((run / "kitti").glob("*.txt"))),
        "fps_steady_mean": round(sum(steady) / len(steady), 2) if steady else None,
        "engine": next(iter(re.findall(r"([\w.-]+\.engine)", log)), None),
        "dump": str(run / "ip_tensor_dump.bin") if (run / "ip_tensor_dump.bin").exists() else None,
    }
    (run / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    if proc.returncode != 0:
        print(f"deepstream-app failed, see {run / 'run.log'}")
    return proc.returncode


if __name__ == "__main__":
    raise SystemExit(main())
