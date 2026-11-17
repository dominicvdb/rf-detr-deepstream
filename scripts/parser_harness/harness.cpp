// Runs the real RF-DETR nvinfer parsers (rfdetr_deepstream/parser) on raw model outputs, without DeepStream.
// Build (any C++17 compiler; uses the stand-in header in this folder):
//     make -C scripts/parser_harness
// Usage:
//     harness det IN N Q C netW netH thr0 [thr1 ...]       IN: per image, dets (Q*4) then labels (Q*C), float32
//     harness seg IN N Q C Hm Wm netW netH OUT thr0 [...]   IN: per image, dets, labels, masks (Q*Hm*Wm), float32
// Prints one line per object: image class left top width height confidence [mask_width mask_height].
// seg also appends every mask (mask_width * mask_height float32, in printed order) to OUT.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "nvdsinfer_custom_impl.h"

extern "C" bool NvDsInferParseRfdetr(std::vector<NvDsInferLayerInfo> const&, NvDsInferNetworkInfo const&,
                                     NvDsInferParseDetectionParams const&, std::vector<NvDsInferObjectDetectionInfo>&);
extern "C" bool NvDsInferParseRfdetrSeg(std::vector<NvDsInferLayerInfo> const&, NvDsInferNetworkInfo const&,
                                        NvDsInferParseDetectionParams const&, std::vector<NvDsInferInstanceMaskInfo>&);

static NvDsInferLayerInfo layer(const char* name, std::vector<float>& data, std::vector<unsigned int> dims) {
  NvDsInferLayerInfo l{};
  l.dataType = FLOAT;
  l.layerName = name;
  l.buffer = data.data();
  l.inferDims.numDims = dims.size();
  l.inferDims.numElements = 1;
  for (size_t i = 0; i < dims.size(); ++i) {
    l.inferDims.d[i] = dims[i];
    l.inferDims.numElements *= dims[i];
  }
  return l;
}

static bool readAll(FILE* f, std::vector<float>& v) { return std::fread(v.data(), 4, v.size(), f) == v.size(); }

int main(int argc, char** argv) {
  if (argc < 3) return 2;
  const bool seg = std::strcmp(argv[1], "seg") == 0;
  const int fixedArgs = seg ? 11 : 8;
  if (argc < fixedArgs + 1) return 2;
  FILE* in = std::fopen(argv[2], "rb");
  if (in == nullptr) return 2;
  const unsigned n = std::atoi(argv[3]), q = std::atoi(argv[4]), c = std::atoi(argv[5]);
  const unsigned hm = seg ? std::atoi(argv[6]) : 0, wm = seg ? std::atoi(argv[7]) : 0;
  const int a = seg ? 8 : 6;
  NvDsInferNetworkInfo net{(unsigned)std::atoi(argv[a]), (unsigned)std::atoi(argv[a + 1]), 3};
  FILE* maskOut = seg ? std::fopen(argv[10], "wb") : nullptr;
  NvDsInferParseDetectionParams params;
  params.numClassesConfigured = c;
  for (int i = fixedArgs; i < argc; ++i) params.perClassPreclusterThreshold.push_back(std::atof(argv[i]));

  std::vector<float> dets(q * 4), labels(q * c), masks(static_cast<size_t>(q) * hm * wm);
  for (unsigned img = 0; img < n; ++img) {
    if (!readAll(in, dets) || !readAll(in, labels) || (seg && !readAll(in, masks))) return 3;
    // Reversed layer order on purpose: the parser must match layers by name.
    std::vector<NvDsInferLayerInfo> layers = {layer("labels", labels, {q, c}), layer("dets", dets, {q, 4})};
    if (!seg) {
      std::vector<NvDsInferObjectDetectionInfo> out;
      if (!NvDsInferParseRfdetr(layers, net, params, out)) return 4;
      for (auto& o : out)
        std::printf("%u %u %.4f %.4f %.4f %.4f %.7f\n", img, o.classId, o.left, o.top, o.width, o.height,
                    o.detectionConfidence);
      continue;
    }
    layers.insert(layers.begin(), layer("masks", masks, {q, hm, wm}));
    std::vector<NvDsInferInstanceMaskInfo> out;
    if (!NvDsInferParseRfdetrSeg(layers, net, params, out)) return 4;
    for (auto& o : out) {
      if (o.mask_size != sizeof(float) * o.mask_width * o.mask_height) return 5;
      std::printf("%u %u %.4f %.4f %.4f %.4f %.7f %u %u\n", img, o.classId, o.left, o.top, o.width, o.height,
                  o.detectionConfidence, o.mask_width, o.mask_height);
      std::fwrite(o.mask, sizeof(float), static_cast<size_t>(o.mask_width) * o.mask_height, maskOut);
      delete[] o.mask;  // what nvinfer does
    }
  }
  if (maskOut != nullptr) std::fclose(maskOut);
  return 0;
}
