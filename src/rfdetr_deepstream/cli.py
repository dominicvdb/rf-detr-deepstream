"""Command line interface: `rfdetr-deepstream build ...`."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from rfdetr_deepstream.bundle import PRETRAINED, BundleError, build_bundle


def _frame_size(value: str) -> tuple[int, int]:
    try:
        w, h = (int(v) for v in value.lower().split("x"))
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected WIDTHxHEIGHT, e.g. 1280x720, got {value!r}") from None
    if w <= 0 or h <= 0:
        raise argparse.ArgumentTypeError("width and height must be positive")
    return w, h


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rfdetr-deepstream", description="Run RF-DETR models in NVIDIA DeepStream.")
    sub = parser.add_subparsers(dest="command", required=True)

    b = sub.add_parser(
        "build",
        help="build a DeepStream bundle (ONNX, labels, nvinfer config, parser source) from an RF-DETR model",
        description="Build a DeepStream bundle from an RF-DETR detection or keypoint model.",
    )
    b.add_argument("model", help=f"pretrained name ({', '.join(PRETRAINED)}), a checkpoint (.pth/.ckpt), or an .onnx")
    b.add_argument("-o", "--output", type=Path, required=True, help="output folder for the bundle")
    b.add_argument("--frame-size", type=_frame_size, metavar="WxH",
                   help="size of the frames nvinfer receives (nvstreammux width x height). Builds the preprocessing "
                        "into the ONNX for exact parity with predict(). Strongly recommended.")
    names = b.add_mutually_exclusive_group()
    names.add_argument("--class-names", type=Path, metavar="FILE",
                       help="class names, one per line (required for a custom .onnx; overrides a checkpoint's names)")
    names.add_argument("--coco", action="store_true", help="the .onnx is an official COCO-pretrained model")
    b.add_argument("--threshold", type=float, default=0.5, help="detection threshold (default 0.5)")
    b.add_argument("--precision", choices=["fp32", "fp16"], default="fp32", help="TensorRT precision (default fp32)")
    b.add_argument("--batch-size", type=int, default=1, metavar="N",
                   help="maximum number of streams per inference call (nvinfer batch-size; default 1). "
                        "Above 1 the ONNX gets a dynamic batch dimension, unless --static-batch is given.")
    b.add_argument("--static-batch", action="store_true",
                   help="export with a fixed batch of --batch-size instead of a dynamic one. Needed for keypoint "
                        "models on DeepStream 7.0 (TensorRT 8.6), which can't build them with a dynamic batch.")
    b.add_argument("--parser-lib", type=Path, metavar="PATH",
                   help="detection: use an already built parser library instead of shipping the parser source")
    b.add_argument("--keypoints-per-class", type=_int_list, metavar="N,N,..",
                   help="keypoint .onnx only: keypoint count per class slot (default 0,17 for the official model)")
    return parser


def _int_list(value: str) -> list[int]:
    try:
        return [int(v) for v in value.split(",") if v.strip()]
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected comma-separated integers, got {value!r}") from None


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "build":
        try:
            build_bundle(
                model=args.model,
                output_dir=args.output,
                frame_size=args.frame_size,
                class_names_file=args.class_names,
                coco=args.coco,
                threshold=args.threshold,
                precision=args.precision,
                parser_lib=args.parser_lib,
                keypoints_per_class=args.keypoints_per_class,
                batch_size=args.batch_size,
                static_batch=args.static_batch,
            )
        except BundleError as e:
            print(f"error: {e}", file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
