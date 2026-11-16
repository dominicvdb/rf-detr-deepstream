// Test driver for the detection parser (compiled by tests/conftest.py with the stand-in header in this folder).
// Usage: harness IN N Q C netW netH thr0 [thr1 ...]
//   IN: for each of N images, dets (Q*4 float32) then labels (Q*C float32).
// Prints one line per object: image class left top width height confidence. Exit code 4 = parser returned false.
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "nvdsinfer_custom_impl.h"

extern "C" bool NvDsInferParseRfdetr(std::vector<NvDsInferLayerInfo> const&, NvDsInferNetworkInfo const&,
                                     NvDsInferParseDetectionParams const&, std::vector<NvDsInferObjectDetectionInfo>&);

static NvDsInferLayerInfo layer(const char* name, std::vector<float>& data, unsigned int d0, unsigned int d1) {
  NvDsInferLayerInfo l{};
  l.dataType = FLOAT;
  l.layerName = name;
  l.buffer = data.data();
  l.inferDims.numDims = 2;
  l.inferDims.d[0] = d0;
  l.inferDims.d[1] = d1;
  l.inferDims.numElements = d0 * d1;
  return l;
}

int main(int argc, char** argv) {
  if (argc < 8) return 2;
  FILE* in = std::fopen(argv[1], "rb");
  if (in == nullptr) return 2;
  const unsigned n = std::atoi(argv[2]), q = std::atoi(argv[3]), c = std::atoi(argv[4]);
  NvDsInferNetworkInfo net{(unsigned)std::atoi(argv[5]), (unsigned)std::atoi(argv[6]), 3};
  NvDsInferParseDetectionParams params;
  params.numClassesConfigured = c;
  for (int i = 7; i < argc; ++i) params.perClassPreclusterThreshold.push_back(std::atof(argv[i]));

  std::vector<float> dets(q * 4), labels(q * c);
  for (unsigned img = 0; img < n; ++img) {
    if (std::fread(dets.data(), 4, dets.size(), in) != dets.size()) return 3;
    if (std::fread(labels.data(), 4, labels.size(), in) != labels.size()) return 3;
    // Reversed order on purpose: the parser must find the layers by name.
    std::vector<NvDsInferLayerInfo> layers = {layer("labels", labels, q, c), layer("dets", dets, q, 4)};
    std::vector<NvDsInferObjectDetectionInfo> out;
    if (!NvDsInferParseRfdetr(layers, net, params, out)) return 4;
    for (auto& o : out)
      std::printf("%u %u %.6f %.6f %.6f %.6f %.8f\n", img, o.classId, o.left, o.top, o.width, o.height,
                  o.detectionConfidence);
  }
  return 0;
}
