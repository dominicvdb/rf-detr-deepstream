// RF-DETR detection parser for DeepStream nvinfer (parse-bbox-func-name=NvDsInferParseRfdetr).
//
// C++ port of rfdetr_deepstream/reference.py::decode(), which mirrors rfdetr 1.12.1
// (export/_runtime/decode.py). Step numbers match the Python reference.
//
// Differences from the Python reference, both required by DeepStream:
//   - Boxes are returned in network input pixels; nvinfer scales them to the frame.
//   - The threshold is per class (pre-cluster-threshold in the nvinfer config). A class with a threshold
//     >= 1.0 can never be detected and is skipped before selection; that is how a background slot is excluded.

#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <vector>

#include "nvdsinfer_custom_impl.h"

namespace {

// Output names of the RF-DETR ONNX export. Matched by name, never by shape:
// with 3 classes, "labels" has the same shape as "dets".
constexpr const char* kBoxesLayer = "dets";     // (Q, 4)  normalised cx, cy, w, h
constexpr const char* kLogitsLayer = "labels";  // (Q, C)  raw class logits
constexpr unsigned int kBoxSize = 4;

struct Candidate {
  float score;
  unsigned int flatIndex;  // query * numClasses + class
};

const NvDsInferLayerInfo* findLayer(const std::vector<NvDsInferLayerInfo>& layers, const char* name) {
  for (const NvDsInferLayerInfo& layer : layers) {
    if (std::strcmp(layer.layerName, name) == 0) {
      return &layer;
    }
  }
  return nullptr;
}

// Clipped to +-88 so exp() cannot overflow a float.
float sigmoid(float x) {
  x = std::min(std::max(x, -88.0f), 88.0f);
  return 1.0f / (1.0f + std::exp(-x));
}

}  // namespace

extern "C" bool NvDsInferParseRfdetr(std::vector<NvDsInferLayerInfo> const& outputLayersInfo,
                                     NvDsInferNetworkInfo const& networkInfo,
                                     NvDsInferParseDetectionParams const& detectionParams,
                                     std::vector<NvDsInferObjectDetectionInfo>& objectList) {
  // ---- Find and validate the output tensors ----------------------------------------------------------
  const NvDsInferLayerInfo* boxesLayer = findLayer(outputLayersInfo, kBoxesLayer);
  const NvDsInferLayerInfo* logitsLayer = findLayer(outputLayersInfo, kLogitsLayer);
  if (boxesLayer == nullptr || logitsLayer == nullptr) {
    std::cerr << "RF-DETR parser: outputs \"" << kBoxesLayer << "\" and \"" << kLogitsLayer
              << "\" not found. The engine has:\n";
    for (const NvDsInferLayerInfo& layer : outputLayersInfo) {
      std::cerr << "  - " << layer.layerName << "\n";
    }
    return false;
  }

  if (boxesLayer->dataType != FLOAT || logitsLayer->dataType != FLOAT) {
    std::cerr << "RF-DETR parser: expected FLOAT outputs.\n";
    return false;
  }

  // nvinfer strips the batch dimension: dets (Q, 4), labels (Q, C).
  const NvDsInferDims& boxDims = boxesLayer->inferDims;
  const NvDsInferDims& logitDims = logitsLayer->inferDims;
  if (boxDims.numDims != 2 || logitDims.numDims != 2 || boxDims.d[1] != kBoxSize ||
      boxDims.d[0] != logitDims.d[0]) {
    std::cerr << "RF-DETR parser: unexpected output shapes (expected dets (Q, 4) and labels (Q, C)).\n";
    return false;
  }
  const unsigned int numQueries = logitDims.d[0];
  const unsigned int numClasses = logitDims.d[1];

  if (numClasses != detectionParams.numClassesConfigured ||
      detectionParams.perClassPreclusterThreshold.size() < numClasses) {
    std::cerr << "RF-DETR parser: the model has " << numClasses
              << " class slots, but the nvinfer config says num-detected-classes="
              << detectionParams.numClassesConfigured << ".\n";
    return false;
  }

  const float* boxes = static_cast<const float*>(boxesLayer->buffer);
  const float* logits = static_cast<const float*>(logitsLayer->buffer);

  // ---- Steps 1, 2, 5: sigmoid, skip excluded classes, threshold -------------------------------------
  // Thresholding before the top-k gives the same result as rfdetr's top-k-then-threshold,
  // and only the passing pairs need sorting instead of all Q * C.
  std::vector<Candidate> candidates;
  for (unsigned int q = 0; q < numQueries; ++q) {
    for (unsigned int c = 0; c < numClasses; ++c) {
      const float threshold = detectionParams.perClassPreclusterThreshold[c];
      if (threshold >= 1.0f) {
        continue;
      }
      const unsigned int flatIndex = q * numClasses + c;
      const float score = sigmoid(logits[flatIndex]);
      if (score > threshold) {
        candidates.push_back({score, flatIndex});
      }
    }
  }

  // ---- Step 4: highest score first, ties by lower flat index (as rfdetr); keep at most Q -----------
  std::sort(candidates.begin(), candidates.end(), [](const Candidate& a, const Candidate& b) {
    if (a.score != b.score) {
      return a.score > b.score;
    }
    return a.flatIndex < b.flatIndex;
  });
  if (candidates.size() > numQueries) {
    candidates.resize(numQueries);
  }

  // ---- Step 6: normalised (cx, cy, w, h) -> (left, top, width, height) in network pixels -----------
  const float netW = static_cast<float>(networkInfo.width);
  const float netH = static_cast<float>(networkInfo.height);

  objectList.reserve(objectList.size() + candidates.size());
  for (const Candidate& cand : candidates) {
    const unsigned int q = cand.flatIndex / numClasses;
    const unsigned int c = cand.flatIndex % numClasses;

    const float* box = boxes + q * kBoxSize;
    const float cx = box[0], cy = box[1], w = box[2], h = box[3];

    const float x1 = std::clamp((cx - w / 2) * netW, 0.0f, netW);
    const float y1 = std::clamp((cy - h / 2) * netH, 0.0f, netH);
    const float x2 = std::clamp((cx + w / 2) * netW, 0.0f, netW);
    const float y2 = std::clamp((cy + h / 2) * netH, 0.0f, netH);

    NvDsInferObjectDetectionInfo obj{};
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

CHECK_CUSTOM_PARSE_FUNC_PROTOTYPE(NvDsInferParseRfdetr);
