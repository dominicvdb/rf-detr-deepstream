"""Test the parser, pose library and bundles against one DeepStream version (Docker image).

Everything version-specific (engines, compiled libraries, outputs) goes to build/<tag>/; the summary goes to
results/<tag>.json. Run from ~/rfdetr-deepstream (WSL) with the venv active:

    python scripts/test_deepstream_version.py --tag ds7.0 --image nvcr.io/nvidia/deepstream:7.0-triton-multiarch

Prerequisites (created by the CLI / earlier steps):
    models/nano_1280x720/        rfdetr-deepstream build nano --frame-size 1280x720 -o models/nano_1280x720
    models/keypoint_1920x1080/   rfdetr-deepstream build keypoint --frame-size 1920x1080 -o models/keypoint_1920x1080
    models/keypoint/rfdetr-keypoint-preview.onnx  (unfolded export, scripts/export_keypoint.py)
    data/sample_720p.mp4, data/sample_walk.mov    (copied from the DeepStream samples)

Steps: versions, build (parser + pose), constant-folding check (trtexec on the unfolded keypoint ONNX),
detection end to end + parity, detection same-input + scaler check, keypoint end to end + parity,
keypoint same-input. All DeepStream runs use NVIDIA_TF32_OVERRIDE=0.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SAMPLES = "/opt/nvidia/deepstream/deepstream/samples/streams"
DET_BUNDLE = ROOT / "models/nano_1280x720"
KP_BUNDLE = ROOT / "models/keypoint_1920x1080"
UNFOLDED_KP_ONNX = "models/keypoint/rfdetr-keypoint-preview.onnx"
DET_VIDEO = ("data/sample_720p.mp4", 1280, 720)
DET_CLIP = "data/sample_720p_100.mp4"   # first 100 frames, for the input-tensor dump
KP_VIDEO = ("data/sample_walk.mov", 1920, 1080)
KP_DUMP_FRAMES = 100
POSE_PROPS = [
    "threshold:0.5", "trace-alpha:0.2", "keypoints-per-class:0,17", "labels:background;person",
    "skeleton:15-13;13-11;16-14;14-12;11-12;5-11;6-12;5-6;5-7;6-8;7-9;8-10;1-2;0-1;0-2;1-3;2-4;3-5;4-6",
]


class Runner:
    def __init__(self, image: str, build: Path):
        self.image = image
        self.logs = build / "logs"
        self.logs.mkdir(parents=True, exist_ok=True)

    def docker(self, cmd: str, log: str, workdir: str = "/work", gpu: bool = True) -> tuple[int, str, float]:
        args = ["docker", "run", "--rm", "-u", f"{os.getuid()}:{os.getgid()}", "-e", "HOME=/tmp",
                "-e", "NVIDIA_TF32_OVERRIDE=0", "-e", "NVIDIA_DRIVER_CAPABILITIES=compute,utility,video",
                "-v", f"{ROOT}:/work", "-w", workdir]
        if gpu:
            args += ["--gpus", "all"]
        args += ["--entrypoint", "bash", self.image, "-c", cmd]
        start = time.time()
        proc = subprocess.run(args, capture_output=True, text=True)
        elapsed = time.time() - start
        out = proc.stdout + proc.stderr
        (self.logs / f"{log}.log").write_text(f"$ {cmd}\n\n{out}")
        return proc.returncode, out, elapsed

    @staticmethod
    def python(script: str, *script_args: str) -> int:
        return subprocess.run([sys.executable, str(ROOT / "scripts" / script), *script_args], cwd=ROOT,
                              capture_output=True, text=True).returncode


def step(name: str):
    print(f"--- {name}", flush=True)


def copy_bundle(src: Path, dst: Path) -> None:
    """Copy a bundle without engines or build output: engines and libraries are per TensorRT/DeepStream version."""
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst, ignore=shutil.ignore_patterns("*.engine", "*.so", "*_dump.txt"))


def write_dump_config(bundle: Path) -> str:
    cfg = (bundle / "config_infer_rfdetr.txt").read_text().replace("[property]\n", "[property]\ndump-input-tensor=1\n", 1)
    (bundle / "config_infer_rfdetr_dump.txt").write_text(cfg)
    return "config_infer_rfdetr_dump.txt"


def deepstream_app_config(uri: str, width: int, height: int, gie_config: str, kitti_dir: str,
                          output_file: str | None) -> str:
    sink = (f"type=3\ncontainer=1\ncodec=1\nenc-type=0\nsync=0\nbitrate=4000000\noutput-file={output_file}"
            if output_file else "type=1\nsync=0")
    return f"""[application]
enable-perf-measurement=1
perf-measurement-interval-sec=5
gie-kitti-output-dir={kitti_dir}

[source0]
enable=1
type=3
uri={uri}
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
config-file={gie_config}

[osd]
enable=1
gpu-id=0
border-width=2
text-size=12
text-color=1;1;1;1
text-bg-color=0.3;0.3;0.3;1
font=Serif

[tiled-display]
enable=0

[sink0]
enable=1
{sink}

[tests]
file-loop=0
"""


def pose_pipeline(video: str, width: int, height: int, config: str, lib: str, dump_dir: str,
                  output_mp4: str | None, max_frames: int | None = None) -> str:
    limit = f"identity eos-after={max_frames + 1} ! " if max_frames else ""
    props = " ".join(f'customlib-props="{p}"' for p in POSE_PROPS + [f"dump-dir:{dump_dir}"])
    tail = (f"nvvideoconvert ! nvdsosd ! nvvideoconvert ! nvv4l2h264enc bitrate=8000000 ! h264parse ! qtmux ! "
            f"filesink location={output_mp4}" if output_mp4 else "fakesink")
    return (f"gst-launch-1.0 filesrc location={video} ! qtdemux ! h264parse ! nvv4l2decoder ! {limit}m.sink_0 "
            f"nvstreammux name=m batch-size=1 width={width} height={height} ! nvinfer config-file-path={config} ! "
            f"nvdsvideotemplate customlib-name={lib} {props} ! {tail}")


def load_json(path: Path):
    return json.loads(path.read_text()) if path.exists() else None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", required=True, help="short name, e.g. ds8.0 (folder and results file name)")
    ap.add_argument("--image", required=True, help="DeepStream Docker image")
    ap.add_argument("--skip", nargs="*", default=[],
                    choices=["folding", "det", "det-same", "kp", "kp-same"], help="steps to skip")
    args = ap.parse_args()

    build = ROOT / "build" / args.tag
    build.mkdir(parents=True, exist_ok=True)
    w = f"/work/build/{args.tag}"
    run = Runner(args.image, build)
    results: dict = {"tag": args.tag, "image": args.image, "date": time.strftime("%Y-%m-%d")}

    # ---- Versions -------------------------------------------------------------------------------------------
    step("versions")
    _, out, _ = run.docker("deepstream-app --version-all", "versions")
    for key, pattern in (("deepstream", r"DeepStreamSDK ([\d.]+)"), ("tensorrt", r"TensorRT Version: ([\d.]+)"),
                         ("cuda_runtime", r"CUDA Runtime Version: ([\d.]+)"), ("cuda_driver", r"CUDA Driver Version: ([\d.]+)")):
        m = re.search(pattern, out)
        results[key] = m.group(1) if m else None
    _, gpu, _ = run.docker("nvidia-smi --query-gpu=name --format=csv,noheader", "gpu")
    results["gpu"] = gpu.strip().splitlines()[-1] if gpu.strip() else None
    print(f"    DeepStream {results['deepstream']}, TensorRT {results['tensorrt']}, GPU {results['gpu']}")

    # ---- Bundles and builds ---------------------------------------------------------------------------------
    step("build parser + pose")
    copy_bundle(DET_BUNDLE, build / "det")
    copy_bundle(KP_BUNDLE, build / "kp")
    det_dump_cfg = write_dump_config(build / "det")
    kp_dump_cfg = write_dump_config(build / "kp")
    results["build"] = {}
    for name, sub in (("parser", "det/parser"), ("pose", "kp/pose")):
        code, out, _ = run.docker(f"make -C {w}/{sub}", f"build_{name}", gpu=False)
        warnings = len(re.findall(r"warning:", out))
        ok = code == 0 and any((build / sub).glob("*.so"))
        results["build"][name] = {"ok": ok, "warnings": warnings}
        print(f"    {name}: {'ok' if ok else 'FAILED'} ({warnings} warnings)")
    if not all(b["ok"] for b in results["build"].values()):
        (ROOT / "results").mkdir(exist_ok=True)
        (ROOT / "results" / f"{args.tag}.json").write_text(json.dumps(results, indent=2))
        print("Build failed, see logs in", run.logs)
        return 1

    # ---- Constant folding still needed? ---------------------------------------------------------------------
    if "folding" not in args.skip:
        step("trtexec on the unfolded keypoint ONNX")
        code, out, _ = run.docker(
            f"T=$(ls /usr/src/tensorrt/bin/trtexec 2>/dev/null || which trtexec); "
            f"$T --onnx=/work/{UNFOLDED_KP_ONNX} --noTF32 --skipInference", "trtexec_unfolded")
        builds = "PASSED" in out
        results["unfolded_keypoint_onnx_builds"] = builds
        print(f"    unfolded keypoint ONNX builds: {builds}")

    # ---- Detection end to end -------------------------------------------------------------------------------
    if "det" not in args.skip:
        step("detection end to end (deepstream-app)")
        out_dir = build / "det_out"
        shutil.rmtree(out_dir, ignore_errors=True)
        (out_dir / "kitti").mkdir(parents=True)
        (build / "det_app.txt").write_text(deepstream_app_config(
            f"file://{SAMPLES}/sample_720p.mp4", DET_VIDEO[1], DET_VIDEO[2], f"{w}/det/config_infer_rfdetr.txt",
            f"{w}/det_out/kitti", f"{w}/det_out/rfdetr_nano.mp4"))
        code, out, secs = run.docker(f"deepstream-app -c {w}/det_app.txt", "det_run")
        perf = re.findall(r"\*\*PERF:\s+[\d.]+ \(([\d.]+)\)", out)
        results["det_fps"] = float(perf[-1]) if perf else None
        results["det_run_ok"] = code == 0 and "App run successful" in out
        run.python("parity_deepstream.py", "--video", DET_VIDEO[0], "--kitti", str(out_dir / "kitti"),
                   "--labels", str(build / "det/labels.txt"), "--json", str(build / "det_parity.json"))
        results["det_parity"] = load_json(build / "det_parity.json")
        p = results["det_parity"] or {}
        print(f"    run ok: {results['det_run_ok']}, fps {results['det_fps']}, "
              f"match {p.get('match_rate')}, IoU {p.get('iou_mean')}")

    # ---- Detection same input + scaler ----------------------------------------------------------------------
    if "det-same" not in args.skip:
        step("detection same-input + scaler check")
        if not (ROOT / DET_CLIP).exists():
            run.docker(f"gst-launch-1.0 filesrc location={SAMPLES}/sample_720p.mp4 ! qtdemux ! h264parse ! "
                       f"identity eos-after=101 ! qtmux ! filesink location=/work/{DET_CLIP}", "make_clip", gpu=False)
        out_dir = build / "det_dump"
        shutil.rmtree(out_dir, ignore_errors=True)
        (out_dir / "kitti").mkdir(parents=True)
        (build / "det_dump_app.txt").write_text(deepstream_app_config(
            f"file:///work/{DET_CLIP}", DET_VIDEO[1], DET_VIDEO[2], f"{w}/det/{det_dump_cfg}",
            f"{w}/det_dump/kitti", None))
        run.docker(f"deepstream-app -c {w}/det_dump_app.txt", "det_dump_run", workdir=f"{w}/det_dump")
        dump = out_dir / "ip_tensor_dump.bin"
        onnx = next((build / "det").glob("*.onnx"))
        run.python("same_input_check.py", "--task", "detect", "--bin", str(dump), "--onnx", str(onnx),
                   "--out-dir", str(out_dir / "kitti"), "--labels", str(build / "det/labels.txt"),
                   "--json", str(build / "det_same_input.json"))
        run.python("scaler_check.py", "--bin", str(dump), "--video", DET_CLIP, "--json", str(build / "scaler.json"))
        dump.unlink(missing_ok=True)
        results["det_same_input"] = load_json(build / "det_same_input.json")
        results["scaler"] = load_json(build / "scaler.json")
        s, sc = results["det_same_input"] or {}, results["scaler"] or {}
        print(f"    same input: match {s.get('match_rate')}, IoU {s.get('iou_mean')}; "
              f"scaler unchanged: {sc.get('unchanged')} (best shift {sc.get('best_shift')})")

    # ---- Keypoints end to end -------------------------------------------------------------------------------
    if "kp" not in args.skip:
        step("keypoints end to end (gst-launch + nvdsvideotemplate)")
        out_dir = build / "kp_out"
        shutil.rmtree(out_dir, ignore_errors=True)
        (out_dir / "dump").mkdir(parents=True)
        lib = f"{w}/kp/pose/libnvds_rfdetr_pose.so"
        # First run builds the engine; the second run is timed.
        run.docker(pose_pipeline(f"/work/{KP_VIDEO[0]}", KP_VIDEO[1], KP_VIDEO[2], f"{w}/kp/config_infer_rfdetr.txt",
                                 lib, "/tmp", None, max_frames=5), "kp_engine_build")
        code, out, _ = run.docker(pose_pipeline(
            f"/work/{KP_VIDEO[0]}", KP_VIDEO[1], KP_VIDEO[2], f"{w}/kp/config_infer_rfdetr.txt", lib,
            f"{w}/kp_out/dump", f"{w}/kp_out/walk_pose.mp4"), "kp_run")
        m = re.search(r"Execution ended after (\d+):(\d+):([\d.]+)", out)
        secs = int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3)) if m else None
        frames = len(list((out_dir / "dump").glob("*.txt")))
        results["kp_run_ok"] = code == 0 and frames > 0
        results["kp_fps"] = frames / secs if secs else None
        run.python("parity_keypoints.py", "--video", KP_VIDEO[0], "--dump", str(out_dir / "dump"),
                   "--json", str(build / "kp_parity.json"))
        results["kp_parity"] = load_json(build / "kp_parity.json")
        p = results["kp_parity"] or {}
        print(f"    run ok: {results['kp_run_ok']}, fps {results['kp_fps']}, match {p.get('match_rate')}, "
              f"keypoints median {p.get('kp_dist_median')} px")

    # ---- Keypoints same input -------------------------------------------------------------------------------
    if "kp-same" not in args.skip:
        step("keypoints same-input")
        out_dir = build / "kp_dump"
        shutil.rmtree(out_dir, ignore_errors=True)
        (out_dir / "dump").mkdir(parents=True)
        run.docker(pose_pipeline(f"/work/{KP_VIDEO[0]}", KP_VIDEO[1], KP_VIDEO[2], f"{w}/kp/{kp_dump_cfg}",
                                 f"{w}/kp/pose/libnvds_rfdetr_pose.so", f"{w}/kp_dump/dump", None,
                                 max_frames=KP_DUMP_FRAMES), "kp_dump_run", workdir=f"{w}/kp_dump")
        dump = out_dir / "ip_tensor_dump.bin"
        onnx = next((build / "kp").glob("*.onnx"))
        run.python("same_input_check.py", "--task", "keypoints", "--bin", str(dump), "--onnx", str(onnx),
                   "--out-dir", str(out_dir / "dump"), "--json", str(build / "kp_same_input.json"))
        dump.unlink(missing_ok=True)
        results["kp_same_input"] = load_json(build / "kp_same_input.json")
        s = results["kp_same_input"] or {}
        print(f"    same input: match {s.get('match_rate')}, IoU {s.get('iou_mean')}, "
              f"keypoints {s.get('kp_dist_mean')} px")

    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / f"{args.tag}.json").write_text(json.dumps(results, indent=2))
    print(f"\nResults: results/{args.tag}.json   Logs: {run.logs.relative_to(ROOT)}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
