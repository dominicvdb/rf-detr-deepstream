// RF-DETR parsers for DeepStream nvinfer.
//   Detection:     parse-bbox-func-name=NvDsInferParseRfdetr
//   Segmentation:  parse-bbox-instance-mask-func-name=NvDsInferParseRfdetrSeg  (network-type=3, output-instance-mask=1)
//
// C++ port of rfdetr_deepstream/reference.py::decode() and decode_masks(), which mirror rfdetr 1.12.1
// (export/_runtime/decode.py, models/postprocess.py). Step numbers match the Python reference.
//
// Differences from the Python reference, all required by DeepStream:
//   - Boxes are returned in network input pixels; nvinfer scales them to the frame.
//   - The threshold is per class (pre-cluster-threshold in the nvinfer config). A class with a threshold
//     >= 1.0 can never be detected and is skipped before selection; that is how a background slot is excluded.
//   - Segmentation: a mask covers only its box (DeepStream's instance-mask convention, NvOSD_MaskParams), at network
//     resolution, holding the probability sigmoid(logit). With segmentation-threshold=0.5 in the config, a pixel is
//     "in" exactly when the Python reference's upsampled logit is > 0.

#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <vector>

#include "nvdsinfer_custom_impl.h"

namespace {

// Output names of the RF-DETR ONNX export. Matched by name, never by shape:
// with 3 classes, "labels" has the same shape as "dets".
constexpr const char* kBoxesLayer = "dets";     // (Q, 4)       normalised cx, cy, w, h
constexpr const char* kLogitsLayer = "labels";  // (Q, C)       raw class logits
constexpr const char* kMasksLayer = "masks";    // (Q, Hm, Wm)  raw mask logits over the whole network input
constexpr unsigned int kBoxSize = 4;

struct Candidate {
  float score;
  unsigned int flatIndex;  // query * numClasses + class
};

struct Box {
  float x1, y1, x2, y2;  // network pixels
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

// Finds and validates "dets" and "labels". Returns false (after printing why) if they don't fit.
bool getBoxAndLogitLayers(const std::vector<NvDsInferLayerInfo>& outputLayersInfo,
                          const NvDsInferParseDetectionParams& detectionParams, const float*& boxes,
                          const float*& logits, unsigned int& numQueries, unsigned int& numClasses) {
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
  numQueries = logitDims.d[0];
  numClasses = logitDims.d[1];

  if (numClasses != detectionParams.numClassesConfigured ||
      detectionParams.perClassPreclusterThreshold.size() < numClasses) {
    std::cerr << "RF-DETR parser: the model has " << numClasses
              << " class slots, but the nvinfer config says num-detected-classes="
              << detectionParams.numClassesConfigured << ".\n";
    return false;
  }

  boxes = static_cast<const float*>(boxesLayer->buffer);
  logits = static_cast<const float*>(logitsLayer->buffer);
  return true;
}

// Steps 1-5: the (query, class) pairs to keep, best first.
std::vector<Candidate> selectCandidates(const float* logits, unsigned int numQueries, unsigned int numClasses,
                                        const NvDsInferParseDetectionParams& detectionParams) {
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
  return candidates;
}

// ---- Step 6: normalised (cx, cy, w, h) -> pixel corners in network pixels, clipped ----------------------
Box toPixels(const float* box, float netW, float netH) {
  const float cx = box[0], cy = box[1], w = box[2], h = box[3];
  return {std::clamp((cx - w / 2) * netW, 0.0f, netW), std::clamp((cy - h / 2) * netH, 0.0f, netH),
          std::clamp((cx + w / 2) * netW, 0.0f, netW), std::clamp((cy + h / 2) * netH, 0.0f, netH)};
}

// One axis of PyTorch's bilinear upsample (align_corners=False): src = (dst + 0.5) * in/out - 0.5, clamped at 0.
struct AxisWeights {
  std::vector<unsigned int> i0, i1;
  std::vector<float> l0, l1;
};

AxisWeights bilinearAxis(unsigned int first, unsigned int count, unsigned int outSize, unsigned int inSize) {
  AxisWeights a;
  a.i0.resize(count);
  a.i1.resize(count);
  a.l0.resize(count);
  a.l1.resize(count);
  const float scale = static_cast<float>(inSize) / static_cast<float>(outSize);
  for (unsigned int k = 0; k < count; ++k) {
    const float src = std::max(scale * (static_cast<float>(first + k) + 0.5f) - 0.5f, 0.0f);
    const unsigned int i0 = static_cast<unsigned int>(src);
    a.i0[k] = i0;
    a.i1[k] = i0 < inSize - 1 ? i0 + 1 : i0;
    a.l1[k] = src - static_cast<float>(i0);
    a.l0[k] = 1.0f - a.l1[k];
  }
  return a;
}

// The pixels a box's mask covers: every pixel the box touches, clipped, at least 1x1. x1/y1 exclusive.
// Same rule as reference.py::mask_box_region.
void maskRegion(const Box& b, unsigned int netW, unsigned int netH, unsigned int& x0, unsigned int& y0,
                unsigned int& x1, unsigned int& y1) {
  x0 = std::min(static_cast<unsigned int>(std::max(std::floor(b.x1), 0.0f)), netW - 1);
  y0 = std::min(static_cast<unsigned int>(std::max(std::floor(b.y1), 0.0f)), netH - 1);
  x1 = std::min(std::max(static_cast<unsigned int>(std::ceil(b.x2)), x0 + 1), netW);
  y1 = std::min(std::max(static_cast<unsigned int>(std::ceil(b.y2)), y0 + 1), netH);
}

}  // namespace

extern "C" bool NvDsInferParseRfdetr(std::vector<NvDsInferLayerInfo> const& outputLayersInfo,
                                     NvDsInferNetworkInfo const& networkInfo,
                                     NvDsInferParseDetectionParams const& detectionParams,
                                     std::vector<NvDsInferObjectDetectionInfo>& objectList) {
  const float *boxes = nullptr, *logits = nullptr;
  unsigned int numQueries = 0, numClasses = 0;
  if (!getBoxAndLogitLayers(outputLayersInfo, detectionParams, boxes, logits, numQueries, numClasses)) {
    return false;
  }
  const std::vector<Candidate> candidates = selectCandidates(logits, numQueries, numClasses, detectionParams);

  const float netW = static_cast<float>(networkInfo.width);
  const float netH = static_cast<float>(networkInfo.height);
  objectList.reserve(objectList.size() + candidates.size());
  for (const Candidate& cand : candidates) {
    const unsigned int q = cand.flatIndex / numClasses;
    const Box b = toPixels(boxes + q * kBoxSize, netW, netH);

    NvDsInferObjectDetectionInfo obj{};
    obj.classId = cand.flatIndex % numClasses;
    obj.left = b.x1;
    obj.top = b.y1;
    obj.width = b.x2 - b.x1;
    obj.height = b.y2 - b.y1;
    obj.detectionConfidence = cand.score;
    objectList.push_back(obj);
  }
  return true;
}

extern "C" bool NvDsInferParseRfdetrSeg(std::vector<NvDsInferLayerInfo> const& outputLayersInfo,
                                        NvDsInferNetworkInfo const& networkInfo,
                                        NvDsInferParseDetectionParams const& detectionParams,
                                        std::vector<NvDsInferInstanceMaskInfo>& objectList) {
  const float *boxes = nullptr, *logits = nullptr;
  unsigned int numQueries = 0, numClasses = 0;
  if (!getBoxAndLogitLayers(outputLayersInfo, detectionParams, boxes, logits, numQueries, numClasses)) {
    return false;
  }

  const NvDsInferLayerInfo* masksLayer = findLayer(outputLayersInfo, kMasksLayer);
  if (masksLayer == nullptr) {
    std::cerr << "RF-DETR parser: output \"" << kMasksLayer << "\" not found. Is this a segmentation model, and is "
              << "\"" << kMasksLayer << "\" in output-blob-names?\n";
    return false;
  }
  const NvDsInferDims& maskDims = masksLayer->inferDims;
  if (masksLayer->dataType != FLOAT || maskDims.numDims != 3 || maskDims.d[0] != numQueries) {
    std::cerr << "RF-DETR parser: expected FLOAT masks of shape (Q, Hm, Wm) with Q=" << numQueries << ".\n";
    return false;
  }
  const unsigned int maskH = maskDims.d[1];
  const unsigned int maskW = maskDims.d[2];
  const float* masks = static_cast<const float*>(masksLayer->buffer);

  const std::vector<Candidate> candidates = selectCandidates(logits, numQueries, numClasses, detectionParams);

  const unsigned int netW = networkInfo.width;
  const unsigned int netH = networkInfo.height;
  objectList.reserve(objectList.size() + candidates.size());
  for (const Candidate& cand : candidates) {
    const unsigned int q = cand.flatIndex / numClasses;
    const Box b = toPixels(boxes + q * kBoxSize, static_cast<float>(netW), static_cast<float>(netH));

    NvDsInferInstanceMaskInfo obj{};
    obj.classId = cand.flatIndex % numClasses;
    obj.left = b.x1;
    obj.top = b.y1;
    obj.width = b.x2 - b.x1;
    obj.height = b.y2 - b.y1;
    obj.detectionConfidence = cand.score;

    // ---- Step 7: the query's mask logits, upsampled bilinearly to network size, inside the box only ----
    unsigned int x0, y0, x1, y1;
    maskRegion(b, netW, netH, x0, y0, x1, y1);
    const unsigned int w = x1 - x0, h = y1 - y0;
    const AxisWeights ax = bilinearAxis(x0, w, netW, maskW);
    const AxisWeights ay = bilinearAxis(y0, h, netH, maskH);
    const float* m = masks + static_cast<size_t>(q) * maskH * maskW;

    // nvinfer takes ownership and releases it with delete[].
    float* out = new float[static_cast<size_t>(w) * h];
    for (unsigned int r = 0; r < h; ++r) {
      const float* row0 = m + static_cast<size_t>(ay.i0[r]) * maskW;
      const float* row1 = m + static_cast<size_t>(ay.i1[r]) * maskW;
      for (unsigned int c = 0; c < w; ++c) {
        // Same summation order as PyTorch's CPU kernel and reference.py::upsample_mask_logits.
        const float top = ax.l0[c] * row0[ax.i0[c]] + ax.l1[c] * row0[ax.i1[c]];
        const float bottom = ax.l0[c] * row1[ax.i0[c]] + ax.l1[c] * row1[ax.i1[c]];
        out[static_cast<size_t>(r) * w + c] = sigmoid(ay.l0[r] * top + ay.l1[r] * bottom);
      }
    }
    obj.mask = out;
    obj.mask_width = w;
    obj.mask_height = h;
    obj.mask_size = static_cast<unsigned int>(sizeof(float) * w * h);
    objectList.push_back(obj);
  }
  return true;
}

CHECK_CUSTOM_PARSE_FUNC_PROTOTYPE(NvDsInferParseRfdetr);
CHECK_CUSTOM_INSTANCE_MASK_PARSE_FUNC_PROTOTYPE(NvDsInferParseRfdetrSeg);
