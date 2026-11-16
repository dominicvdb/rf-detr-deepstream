// Stand-in for DeepStream's nvdsinfer.h / nvdsinfer_custom_impl.h, for testing the parser on a machine without
// DeepStream (tests/cpp/harness.cpp). Only the types the RF-DETR parser uses, with DeepStream's field names.
// It is NOT used on the target: there the parser compiles against the real headers (parser/Makefile).
#pragma once
#include <vector>

typedef enum { FLOAT = 0, HALF = 1, INT8 = 2, INT32 = 3 } NvDsInferDataType;

#define NVDSINFER_MAX_DIMS 8
typedef struct {
  unsigned int numDims;
  unsigned int d[NVDSINFER_MAX_DIMS];
  unsigned int numElements;
} NvDsInferDims;

typedef struct {
  NvDsInferDataType dataType;
  NvDsInferDims inferDims;
  int bindingIndex;
  const char* layerName;
  void* buffer;
  int isInput;
} NvDsInferLayerInfo;

typedef struct {
  unsigned int width;
  unsigned int height;
  unsigned int channels;
} NvDsInferNetworkInfo;

typedef struct {
  unsigned int numClassesConfigured;
  std::vector<float> perClassPreclusterThreshold;
  std::vector<float> perClassPostclusterThreshold;
} NvDsInferParseDetectionParams;

typedef struct {
  unsigned int classId;
  float left;
  float top;
  float width;
  float height;
  float detectionConfidence;
} NvDsInferObjectDetectionInfo;

typedef struct {
  unsigned int classId;
  float left;
  float top;
  float width;
  float height;
  float detectionConfidence;
  float* mask;
  unsigned int mask_width;
  unsigned int mask_height;
  unsigned int mask_size;
} NvDsInferInstanceMaskInfo;

#define CHECK_CUSTOM_PARSE_FUNC_PROTOTYPE(f) static_assert(true, "")
#define CHECK_CUSTOM_INSTANCE_MASK_PARSE_FUNC_PROTOTYPE(f) static_assert(true, "")
