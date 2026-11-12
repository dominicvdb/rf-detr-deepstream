# rfdetr-deepstream

Run [RF-DETR](https://github.com/roboflow/rf-detr) models (detection, later keypoints) in NVIDIA DeepStream.

**Status: early development.**

Goal: a tool that turns an RF-DETR `.pth` or `.onnx` into a ready-to-run DeepStream bundle (ONNX, labels, `nvinfer` config), plus a C++ `nvinfer` parser library built for each DeepStream version and platform.

Tested so far: DeepStream 7.0 (TensorRT 8.6) on an RTX 4060 under WSL2.
