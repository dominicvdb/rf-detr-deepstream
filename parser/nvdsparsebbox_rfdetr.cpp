// RF-DETR detection parser for DeepStream nvinfer.
//
// nvinfer calls NvDsInferParseRfdetr() once per frame, after TensorRT has run the model.
// It hands us the raw output tensors and expects a list of boxes back.
//
// This is a C++ translation of src/rfdetr_deepstream/reference.py::decode(), which in turn mirrors
// rfdetr 1.12.1 (export/_runtime/decode.py). Step numbers below match the Python reference.
//
// Differences from the Python reference, both required by DeepStream:
//   - Boxes are returned in NETWORK input pixels (e.g. 384x384). nvinfer scales them to the video frame.
//   - The threshold is per class, from the nvinfer config ([class-attrs-all] / [class-attrs-N]
//     pre-cluster-threshold). A class whose threshold is >= 1.0 can never be detected (a sigmoid score is
//     always < 1), so we drop it before selection. That is how a background slot is excluded:
//     the Python tool writes "pre-cluster-threshold=1.0" for that class into the config.

#include <algorithm>  // std::sort, std::min, std::max
#include <cmath>      // std::exp
#include <cstring>    // std::strcmp
#include <iostream>   // std::cerr (error messages)
#include <vector>     // std::vector (a growable array, like a Python list)

#include "nvdsinfer_custom_impl.h"  // NVIDIA's parser API: the structs and the function type nvinfer expects

// An unnamed namespace makes everything inside it private to this file, like a leading underscore
// in Python, except the compiler enforces it. Only NvDsInferParseRfdetr (below) is visible to nvinfer.
namespace {

// Name of the outputs in the RF-DETR ONNX export. Always match by name, never by shape:
// with 3 classes, "labels" is (Q, 4), the same shape as "dets".
constexpr const char* kBoxesLayer = "dets";     // (Q, 4)  normalised cx, cy, w, h
constexpr const char* kLogitsLayer = "labels";  // (Q, C)  raw class logits
constexpr unsigned int kBoxSize = 4;

// One (query, class) pair that passed the threshold.
struct Candidate {
  float score;             // sigmoid probability
  unsigned int flatIndex;  // position in the flattened (Q, C) logits: query * C + class
};

// Look up an output layer by name. Returns a POINTER to the layer inside the vector, or nullptr if
// there is no layer with that name. Returning a pointer avoids copying the struct, and nullptr is
// C++'s way of saying "nothing", like returning None in Python.
const NvDsInferLayerInfo* findLayer(const std::vector<NvDsInferLayerInfo>& layers, const char* name) {
  for (const NvDsInferLayerInfo& layer : layers) {  // like: for layer in layers
    if (std::strcmp(layer.layerName, name) == 0) {   // C strings are compared with strcmp; 0 means equal
      return &layer;                                 // & = "address of": hand out a pointer to it
    }
  }
  return nullptr;
}

// Step 1 for one value: logit -> probability. Clip first so exp() cannot overflow a float.
float sigmoid(float x) {
  x = std::min(std::max(x, -88.0f), 88.0f);  // the f suffix makes the literal a float, not a double
  return 1.0f / (1.0f + std::exp(-x));
}

}  // namespace

// extern "C" turns off C++ "name mangling", so the function's name in the compiled .so is exactly
// "NvDsInferParseRfdetr". nvinfer finds it at runtime by that name (parse-bbox-func-name in the config).
//
// Parameters: "const X&" = read-only reference (no copy, cannot be modified).
//             "X&" (objectList) = a reference we CAN modify; that is how we return our boxes.
extern "C" bool NvDsInferParseRfdetr(std::vector<NvDsInferLayerInfo> const& outputLayersInfo,
                                     NvDsInferNetworkInfo const& networkInfo,
                                     NvDsInferParseDetectionParams const& detectionParams,
                                     std::vector<NvDsInferObjectDetectionInfo>& objectList) {
  // ---- Find and validate the two output tensors --------------------------------------------------
  const NvDsInferLayerInfo* boxesLayer = findLayer(outputLayersInfo, kBoxesLayer);
  const NvDsInferLayerInfo* logitsLayer = findLayer(outputLayersInfo, kLogitsLayer);
  if (boxesLayer == nullptr || logitsLayer == nullptr) {
    std::cerr << "RF-DETR parser: outputs \"" << kBoxesLayer << "\" and \"" << kLogitsLayer
              << "\" not found. The engine has:\n";
    for (const NvDsInferLayerInfo& layer : outputLayersInfo) {
      std::cerr << "  - " << layer.layerName << "\n";
    }
    return false;  // returning false makes nvinfer report a parsing error
  }

  // "->" reads a member through a pointer: boxesLayer->dataType is (*boxesLayer).dataType.
  if (boxesLayer->dataType != FLOAT || logitsLayer->dataType != FLOAT) {
    std::cerr << "RF-DETR parser: expected FLOAT outputs.\n";
    return false;
  }

  // nvinfer strips the batch dimension: per frame we get dets (Q, 4) and labels (Q, C).
  const NvDsInferDims& boxDims = boxesLayer->inferDims;
  const NvDsInferDims& logitDims = logitsLayer->inferDims;
  if (boxDims.numDims != 2 || logitDims.numDims != 2 || boxDims.d[1] != kBoxSize ||
      boxDims.d[0] != logitDims.d[0]) {
    std::cerr << "RF-DETR parser: unexpected output shapes (expected dets (Q, 4) and labels (Q, C)).\n";
    return false;
  }
  const unsigned int numQueries = logitDims.d[0];  // 300 for the official models
  const unsigned int numClasses = logitDims.d[1];  // 91 for the official COCO models

  if (numClasses != detectionParams.numClassesConfigured ||
      detectionParams.perClassPreclusterThreshold.size() < numClasses) {
    std::cerr << "RF-DETR parser: the model has " << numClasses
              << " class slots, but the nvinfer config says num-detected-classes="
              << detectionParams.numClassesConfigured << ".\n";
    return false;
  }

  // The tensors are flat blocks of floats in (CPU) memory. nvinfer gives us an untyped pointer
  // (void*: "an address, type unknown"); we tell the compiler it points at floats.
  // const float* = pointer to floats we promise not to modify.
  const float* boxes = static_cast<const float*>(boxesLayer->buffer);
  const float* logits = static_cast<const float*>(logitsLayer->buffer);

  // ---- Steps 1, 2, 5: sigmoid, skip excluded classes, threshold ------------------------------------
  // We threshold BEFORE sorting. This gives exactly the same result as rfdetr's "top-k, then
  // threshold" (see docs), but we only have to sort the few pairs that pass, not all Q*C = 27300.
  std::vector<Candidate> candidates;
  for (unsigned int q = 0; q < numQueries; ++q) {
    for (unsigned int c = 0; c < numClasses; ++c) {
      const float threshold = detectionParams.perClassPreclusterThreshold[c];
      if (threshold >= 1.0f) {
        continue;  // Step 2: excluded class (e.g. background slot)
      }
      const unsigned int flatIndex = q * numClasses + c;  // Step 3: row-major position of (q, c)
      const float score = sigmoid(logits[flatIndex]);     // logits[i] reads the i-th float after the pointer
      if (score > threshold) {
        candidates.push_back({score, flatIndex});  // like list.append(...)
      }
    }
  }

  // ---- Step 4: highest score first; ties -> lower flat index first; keep at most Q ----------------
  // The [](...) { ... } is a lambda: an inline function, like Python's lambda, here telling sort how
  // to compare two candidates.
  std::sort(candidates.begin(), candidates.end(), [](const Candidate& a, const Candidate& b) {
    if (a.score != b.score) {
      return a.score > b.score;
    }
    return a.flatIndex < b.flatIndex;
  });
  if (candidates.size() > numQueries) {
    candidates.resize(numQueries);
  }

  // ---- Step 6: normalised (cx, cy, w, h) -> (left, top, width, height) in network pixels ----------
  const float netW = static_cast<float>(networkInfo.width);
  const float netH = static_cast<float>(networkInfo.height);

  objectList.reserve(objectList.size() + candidates.size());  // allocate once instead of growing
  for (const Candidate& cand : candidates) {
    const unsigned int q = cand.flatIndex / numClasses;  // integer division, like // in Python
    const unsigned int c = cand.flatIndex % numClasses;

    // Pointer arithmetic: box points to the 4 floats of query q (q * 4 floats after the start).
    const float* box = boxes + q * kBoxSize;
    const float cx = box[0], cy = box[1], w = box[2], h = box[3];

    const float x1 = std::clamp((cx - w / 2) * netW, 0.0f, netW);  // std::clamp(v, lo, hi), like np.clip
    const float y1 = std::clamp((cy - h / 2) * netH, 0.0f, netH);
    const float x2 = std::clamp((cx + w / 2) * netW, 0.0f, netW);
    const float y2 = std::clamp((cy + h / 2) * netH, 0.0f, netH);

    NvDsInferObjectDetectionInfo obj{};  // {} = zero-initialise every field
    obj.classId = c;
    obj.left = x1;
    obj.top = y1;
    obj.width = x2 - x1;
    obj.height = y2 - y1;
    obj.detectionConfidence = cand.score;
    objectList.push_back(obj);
  }
  return true;
}

// NVIDIA's macro: makes the compiler check that our function matches the type nvinfer expects
// (NvDsInferParseCustomFunc). A mismatch becomes a compile error instead of a crash at runtime.
CHECK_CUSTOM_PARSE_FUNC_PROTOTYPE(NvDsInferParseRfdetr);
