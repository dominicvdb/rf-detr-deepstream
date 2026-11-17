// RF-DETR keypoint (pose) post-processing for DeepStream, as an nvdsvideotemplate custom library.
//
// Pipeline: nvinfer (network-type=100, output-tensor-meta=1) ! nvdsvideotemplate customlib-name=libnvds_rfdetr_pose.so
//           customlib-props="threshold:0.5" ... ! nvdsosd
//
// For each frame, decodes nvinfer's raw output tensors (dets, labels, keypoints) like
// rfdetr_deepstream/reference.py::decode_keypoints() (mirrors rfdetr 1.12.1 models/postprocess.py) and attaches:
//   - NvDsObjectMeta per detection (box, label, fused score),
//   - RfdetrKeypoints user meta on each object (nvds_rfdetr_keypoints_meta.h),
//   - display meta (keypoint circles, skeleton lines) for nvdsosd.
// Model outputs are normalised, so coordinates are scaled to the frame (nvstreammux) size, independent of the
// network input size.
//
// customlib-props (key:value):
//   threshold:<float>            detection threshold on the fused score (default 0.5)
//   trace-alpha:<float>          keypoint-uncertainty score fusion (default 0.2, as rfdetr; 0 disables)
//   keypoints-per-class:<a,b,..> keypoint schema, one count per logits slot (default 0,17)
//   labels:<a;b;..>              class name per logits slot (default background;person)
//   skeleton:<i-j;k-l;..>        keypoint pairs to draw (default COCO 17-keypoint skeleton)
//   keypoint-threshold:<float>   minimum keypoint confidence to draw (default 0.5)
//   draw:<0|1>                   add display meta (default 1)
//   gie-id:<int>                 gie-unique-id of the nvinfer whose tensors to decode (default 1)
//   dump-dir:<path>              write one text file per frame with all detections (for parity tests)

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

// Order matters: nvdscustomlib_base.hpp needs CUDA and NvBufSurface declared first (as in NVIDIA's sample),
// and gstnvdsinfer.h relies on the GStreamer/glib types from the headers before it.
#include <cuda_runtime.h>
#include "nvbufsurface.h"
#include "nvdscustomlib_base.hpp"
#include "gstnvdsmeta.h"
#include "gstnvdsinfer.h"
#include "nvds_rfdetr_keypoints_meta.h"

namespace {

constexpr const char* kBoxesLayer = "dets";          // (Q, 4)  normalised cx, cy, w, h
constexpr const char* kLogitsLayer = "labels";       // (Q, C)  raw class logits
constexpr const char* kKeypointsLayer = "keypoints"; // (Q, C * Kmax, D), D >= 7

const std::vector<std::pair<int, int>> kCocoSkeleton = {
    {15, 13}, {13, 11}, {16, 14}, {14, 12}, {11, 12}, {5, 11}, {6, 12}, {5, 6}, {5, 7}, {6, 8},
    {7, 9},   {8, 10},  {1, 2},   {0, 1},   {0, 2},   {1, 3},  {2, 4},  {3, 5}, {4, 6}};

struct Detection {
  unsigned int classId;
  float score;  // fused
  float x1, y1, x2, y2;
  std::vector<float> kx, ky, kc;
};

float sigmoid(float x) {
  x = std::min(std::max(x, -88.0f), 88.0f);
  return 1.0f / (1.0f + std::exp(-x));
}

float logsumexp(const float* v, size_t n) {
  float m = -std::numeric_limits<float>::infinity();
  for (size_t i = 0; i < n; ++i) m = std::max(m, v[i]);
  double s = 0.0;
  for (size_t i = 0; i < n; ++i) s += std::exp(static_cast<double>(v[i] - m));
  return m + static_cast<float>(std::log(s));
}

// Log of the findability-weighted mean covariance trace, from the precision-Cholesky params in columns 4:7.
// kp points at n keypoints of d values each.
float keypointLogMeanTrace(const float* kp, unsigned int n, unsigned int d) {
  std::vector<float> weighted(n), logW(n);
  for (unsigned int k = 0; k < n; ++k) {
    const float* p = kp + k * d;
    const float logT1 = -2.0f * p[4];
    const float logT2 = -2.0f * p[6];
    const float logT3 = 2.0f * std::log(std::max(std::fabs(p[5]), 1e-12f)) + logT1 + logT2;
    const float terms[3] = {logT1, logT2, logT3};
    logW[k] = std::log(std::max(sigmoid(p[2]), 1e-12f));
    weighted[k] = logsumexp(terms, 3) + logW[k];
  }
  return logsumexp(weighted.data(), n) - logsumexp(logW.data(), n);
}

const NvDsInferLayerInfo* findLayer(const NvDsInferTensorMeta* meta, const char* name) {
  for (unsigned int i = 0; i < meta->num_output_layers; ++i) {
    if (std::strcmp(meta->output_layers_info[i].layerName, name) == 0) return &meta->output_layers_info[i];
  }
  return nullptr;
}

const float* hostBuffer(const NvDsInferTensorMeta* meta, const NvDsInferLayerInfo* layer) {
  return static_cast<const float*>(meta->out_buf_ptrs_host[layer - meta->output_layers_info]);
}

std::vector<std::string> split(const std::string& s, char sep) {
  std::vector<std::string> out;
  std::stringstream ss(s);
  std::string item;
  while (std::getline(ss, item, sep)) {
    if (!item.empty()) out.push_back(item);
  }
  return out;
}

gpointer copyKeypointsMeta(gpointer data, gpointer /*user_data*/) {
  NvDsUserMeta* userMeta = static_cast<NvDsUserMeta*>(data);
  return g_memdup2(userMeta->user_meta_data, sizeof(RfdetrKeypoints));
}

void releaseKeypointsMeta(gpointer data, gpointer /*user_data*/) {
  NvDsUserMeta* userMeta = static_cast<NvDsUserMeta*>(data);
  g_free(userMeta->user_meta_data);
  userMeta->user_meta_data = nullptr;
}

}  // namespace

class RfdetrPose : public DSCustomLibraryBase {
 public:
  bool SetInitParams(DSCustom_CreateParams* params) override;
  bool SetProperty(Property& prop) override;
  bool HandleEvent(GstEvent* /*event*/) override { return true; }
  char* QueryProperties() override;
  BufferResult ProcessBuffer(GstBuffer* inbuf) override;

 private:
  bool decodeFrame(const NvDsInferTensorMeta* meta, float frameW, float frameH, std::vector<Detection>& out);
  void attachFrame(NvDsBatchMeta* batchMeta, NvDsFrameMeta* frameMeta, const std::vector<Detection>& dets,
                   float frameW, float frameH);
  void dumpFrame(const NvDsFrameMeta* frameMeta, const std::vector<Detection>& dets);

  float threshold_ = 0.5f;
  float traceAlpha_ = 0.2f;
  float keypointThreshold_ = 0.5f;
  bool draw_ = true;
  unsigned int gieId_ = 1;
  std::vector<unsigned int> keypointsPerClass_ = {0, 17};
  std::vector<std::string> labels_ = {"background", "person"};
  std::vector<std::pair<int, int>> skeleton_ = kCocoSkeleton;
  std::string dumpDir_;
  NvDsMetaType keypointsMetaType_ = NVDS_INVALID_META;
  bool warnedMissing_ = false;
};

bool RfdetrPose::SetInitParams(DSCustom_CreateParams* params) {
  DSCustomLibraryBase::SetInitParams(params);
  keypointsMetaType_ = nvds_get_user_meta_type(const_cast<gchar*>(RFDETR_KEYPOINTS_META_STRING));
  return true;
}

bool RfdetrPose::SetProperty(Property& prop) {
  try {
    if (prop.key == "threshold") {
      threshold_ = std::stof(prop.value);
    } else if (prop.key == "trace-alpha") {
      traceAlpha_ = std::stof(prop.value);
    } else if (prop.key == "keypoint-threshold") {
      keypointThreshold_ = std::stof(prop.value);
    } else if (prop.key == "draw") {
      draw_ = std::stoi(prop.value) != 0;
    } else if (prop.key == "gie-id") {
      gieId_ = static_cast<unsigned int>(std::stoul(prop.value));
    } else if (prop.key == "keypoints-per-class") {
      keypointsPerClass_.clear();
      for (const auto& v : split(prop.value, ',')) keypointsPerClass_.push_back(std::stoul(v));
    } else if (prop.key == "labels") {
      labels_ = split(prop.value, ';');
    } else if (prop.key == "skeleton") {
      skeleton_.clear();
      for (const auto& pair : split(prop.value, ';')) {
        const auto ij = split(pair, '-');
        if (ij.size() == 2) skeleton_.emplace_back(std::stoi(ij[0]), std::stoi(ij[1]));
      }
    } else if (prop.key == "dump-dir") {
      dumpDir_ = prop.value;
    } else {
      g_printerr("rfdetr_pose: unknown property '%s'\n", prop.key.c_str());
      return false;
    }
  } catch (const std::exception& e) {
    g_printerr("rfdetr_pose: invalid value '%s' for '%s'\n", prop.value.c_str(), prop.key.c_str());
    return false;
  }
  return true;
}

char* RfdetrPose::QueryProperties() {
  return g_strdup(
      "threshold:<float> trace-alpha:<float> keypoint-threshold:<float> draw:<0|1> gie-id:<int> "
      "keypoints-per-class:<a,b,..> labels:<a;b;..> skeleton:<i-j;..> dump-dir:<path>");
}

BufferResult RfdetrPose::ProcessBuffer(GstBuffer* inbuf) {
  NvDsBatchMeta* batchMeta = gst_buffer_get_nvds_batch_meta(inbuf);
  if (batchMeta == nullptr) return BufferResult::Buffer_Ok;

  const float frameW = static_cast<float>(GST_VIDEO_INFO_WIDTH(&m_inVideoInfo));
  const float frameH = static_cast<float>(GST_VIDEO_INFO_HEIGHT(&m_inVideoInfo));

  for (NvDsMetaList* lf = batchMeta->frame_meta_list; lf != nullptr; lf = lf->next) {
    NvDsFrameMeta* frameMeta = static_cast<NvDsFrameMeta*>(lf->data);
    const NvDsInferTensorMeta* tensorMeta = nullptr;
    for (NvDsMetaList* lu = frameMeta->frame_user_meta_list; lu != nullptr; lu = lu->next) {
      NvDsUserMeta* userMeta = static_cast<NvDsUserMeta*>(lu->data);
      if (userMeta->base_meta.meta_type != NVDSINFER_TENSOR_OUTPUT_META) continue;
      const auto* meta = static_cast<const NvDsInferTensorMeta*>(userMeta->user_meta_data);
      if (meta->unique_id == gieId_) tensorMeta = meta;
    }
    if (tensorMeta == nullptr) {
      if (!warnedMissing_) {
        g_printerr("rfdetr_pose: no tensor meta from gie-id %u. Set output-tensor-meta=1 in the nvinfer config.\n",
                   gieId_);
        warnedMissing_ = true;
      }
      continue;
    }

    std::vector<Detection> dets;
    if (!decodeFrame(tensorMeta, frameW, frameH, dets)) return BufferResult::Buffer_Error;
    attachFrame(batchMeta, frameMeta, dets, frameW, frameH);
    if (!dumpDir_.empty()) dumpFrame(frameMeta, dets);
  }
  return BufferResult::Buffer_Ok;
}

bool RfdetrPose::decodeFrame(const NvDsInferTensorMeta* meta, float frameW, float frameH,
                             std::vector<Detection>& out) {
  const NvDsInferLayerInfo* boxesLayer = findLayer(meta, kBoxesLayer);
  const NvDsInferLayerInfo* logitsLayer = findLayer(meta, kLogitsLayer);
  const NvDsInferLayerInfo* kpLayer = findLayer(meta, kKeypointsLayer);
  if (boxesLayer == nullptr || logitsLayer == nullptr || kpLayer == nullptr) {
    g_printerr("rfdetr_pose: outputs dets, labels and keypoints are required.\n");
    return false;
  }
  if (boxesLayer->dataType != FLOAT || logitsLayer->dataType != FLOAT || kpLayer->dataType != FLOAT) {
    g_printerr("rfdetr_pose: expected FLOAT outputs.\n");
    return false;
  }
  const NvDsInferDims& bd = boxesLayer->inferDims;
  const NvDsInferDims& ld = logitsLayer->inferDims;
  const NvDsInferDims& kd = kpLayer->inferDims;
  if (bd.numDims != 2 || ld.numDims != 2 || kd.numDims != 3 || bd.d[1] != 4 || bd.d[0] != ld.d[0] ||
      kd.d[0] != ld.d[0]) {
    g_printerr("rfdetr_pose: unexpected output shapes.\n");
    return false;
  }
  const unsigned int numQueries = ld.d[0];
  const unsigned int numClasses = ld.d[1];
  const unsigned int d = kd.d[2];
  if (keypointsPerClass_.size() != numClasses) {
    g_printerr("rfdetr_pose: keypoints-per-class has %zu entries, the model has %u class slots.\n",
               keypointsPerClass_.size(), numClasses);
    return false;
  }
  const unsigned int kmax = *std::max_element(keypointsPerClass_.begin(), keypointsPerClass_.end());
  if (kd.d[1] != numClasses * kmax || kmax > RFDETR_MAX_KEYPOINTS) {
    g_printerr("rfdetr_pose: keypoints tensor has %u slots, expected %u classes x %u keypoints.\n", kd.d[1],
               numClasses, kmax);
    return false;
  }
  if (kmax > 0 && d < 3) {
    g_printerr("rfdetr_pose: keypoints tensor has %u values per keypoint, expected at least 3 (x, y, logit).\n", d);
    return false;
  }
  const bool fuse = traceAlpha_ > 0.0f && d >= 7;

  const float* boxes = hostBuffer(meta, boxesLayer);
  const float* logits = hostBuffer(meta, logitsLayer);
  const float* keypoints = hostBuffer(meta, kpLayer);

  // Steps 1-4: sigmoid over all slots (background included), top-k pairs by score, ties by lower flat index.
  // No threshold yet: predict() thresholds the fused score.
  const unsigned int numPairs = numQueries * numClasses;
  std::vector<unsigned int> order(numPairs);
  std::vector<float> scores(numPairs);
  for (unsigned int i = 0; i < numPairs; ++i) {
    order[i] = i;
    // NaN logits (e.g. FP16 overflow) would break the sort order; they sort last and are skipped below.
    scores[i] = std::isnan(logits[i]) ? -1.0f : sigmoid(logits[i]);
  }
  const unsigned int k = std::min(numQueries, numPairs);
  std::partial_sort(order.begin(), order.begin() + k, order.end(), [&](unsigned int a, unsigned int b) {
    if (scores[a] != scores[b]) return scores[a] > scores[b];
    return a < b;
  });

  const float maxScore = std::nextafter(1.0f, 0.0f);
  for (unsigned int r = 0; r < k; ++r) {
    const unsigned int q = order[r] / numClasses;
    const unsigned int c = order[r] % numClasses;
    const float score = scores[order[r]];
    if (score < 0.0f) continue;
    const unsigned int n = keypointsPerClass_[c];
    const float* kp = keypoints + (static_cast<size_t>(q) * numClasses * kmax + static_cast<size_t>(c) * kmax) * d;

    // Step 7: score fusion with keypoint uncertainty (trace 0 for classes without keypoints).
    float fused = score;
    if (fuse) {
      const float logMeanTrace = n > 0 ? keypointLogMeanTrace(kp, n, d) : 0.0f;
      fused = std::min(sigmoid(std::log(score) - traceAlpha_ * logMeanTrace), maxScore);
    }
    // Step 5: threshold on the fused score.
    if (!(fused > threshold_)) continue;

    // Step 6: normalised box -> frame pixels.
    const float* b = boxes + q * 4;
    Detection det;
    det.classId = c;
    det.score = fused;
    det.x1 = std::clamp((b[0] - b[2] / 2) * frameW, 0.0f, frameW);
    det.y1 = std::clamp((b[1] - b[3] / 2) * frameH, 0.0f, frameH);
    det.x2 = std::clamp((b[0] + b[2] / 2) * frameW, 0.0f, frameW);
    det.y2 = std::clamp((b[1] + b[3] / 2) * frameH, 0.0f, frameH);
    if (!std::isfinite(det.x1) || !std::isfinite(det.y1) || !std::isfinite(det.x2) || !std::isfinite(det.y2)) continue;
    for (unsigned int j = 0; j < n; ++j) {
      det.kx.push_back(kp[j * d + 0] * frameW);
      det.ky.push_back(kp[j * d + 1] * frameH);
      det.kc.push_back(sigmoid(kp[j * d + 2]));
    }
    out.push_back(std::move(det));
  }
  return true;
}

void RfdetrPose::attachFrame(NvDsBatchMeta* batchMeta, NvDsFrameMeta* frameMeta, const std::vector<Detection>& dets,
                             float frameW, float frameH) {
  NvDsDisplayMeta* display = nullptr;
  auto displayMeta = [&]() -> NvDsDisplayMeta* {
    if (display == nullptr || display->num_circles >= MAX_ELEMENTS_IN_DISPLAY_META ||
        display->num_lines >= MAX_ELEMENTS_IN_DISPLAY_META) {
      display = nvds_acquire_display_meta_from_pool(batchMeta);
      nvds_add_display_meta_to_frame(frameMeta, display);
    }
    return display;
  };
  const NvOSD_ColorParams pointColor = {1.0, 0.2, 0.2, 1.0};
  const NvOSD_ColorParams lineColor = {0.2, 1.0, 0.2, 1.0};

  for (const Detection& det : dets) {
    NvDsObjectMeta* obj = nvds_acquire_obj_meta_from_pool(batchMeta);
    obj->unique_component_id = gieId_;
    obj->class_id = static_cast<gint>(det.classId);
    obj->confidence = det.score;
    obj->object_id = UNTRACKED_OBJECT_ID;
    const std::string label = det.classId < labels_.size() ? labels_[det.classId] : std::to_string(det.classId);
    g_strlcpy(obj->obj_label, label.c_str(), MAX_LABEL_SIZE);

    NvOSD_RectParams& rect = obj->rect_params;
    rect.left = det.x1;
    rect.top = det.y1;
    rect.width = det.x2 - det.x1;
    rect.height = det.y2 - det.y1;
    rect.border_width = 2;
    rect.border_color = {0.2, 0.6, 1.0, 1.0};
    obj->detector_bbox_info.org_bbox_coords.left = rect.left;
    obj->detector_bbox_info.org_bbox_coords.top = rect.top;
    obj->detector_bbox_info.org_bbox_coords.width = rect.width;
    obj->detector_bbox_info.org_bbox_coords.height = rect.height;

    NvOSD_TextParams& text = obj->text_params;
    text.display_text = g_strdup_printf("%s %.2f", label.c_str(), det.score);
    text.x_offset = static_cast<unsigned int>(std::max(0.0f, rect.left));
    text.y_offset = static_cast<unsigned int>(std::max(0.0f, rect.top - 15.0f));
    text.font_params.font_name = const_cast<gchar*>("Serif");
    text.font_params.font_size = 10;
    text.font_params.font_color = {1.0, 1.0, 1.0, 1.0};
    text.set_bg_clr = 1;
    text.text_bg_clr = {0.0, 0.0, 0.0, 0.6};

    nvds_add_obj_meta_to_frame(frameMeta, obj, nullptr);

    if (!det.kx.empty()) {
      auto* kp = static_cast<RfdetrKeypoints*>(g_malloc0(sizeof(RfdetrKeypoints)));
      kp->num_keypoints = static_cast<uint32_t>(det.kx.size());
      for (size_t j = 0; j < det.kx.size(); ++j) {
        kp->x[j] = det.kx[j];
        kp->y[j] = det.ky[j];
        kp->confidence[j] = det.kc[j];
      }
      NvDsUserMeta* userMeta = nvds_acquire_user_meta_from_pool(batchMeta);
      userMeta->user_meta_data = kp;
      userMeta->base_meta.meta_type = keypointsMetaType_;
      userMeta->base_meta.copy_func = copyKeypointsMeta;
      userMeta->base_meta.release_func = releaseKeypointsMeta;
      nvds_add_user_meta_to_obj(obj, userMeta);
    }

    if (!draw_ || det.kx.empty()) continue;
    // Off-frame keypoints are common (people partly out of view) and are not drawn: nvdsosd takes unsigned
    // in-frame pixel coordinates. NaN coordinates fail these comparisons too.
    const auto visible = [&](int j) {
      return j >= 0 && static_cast<size_t>(j) < det.kx.size() && det.kc[j] >= keypointThreshold_ &&
             det.kx[j] >= 0.0f && det.kx[j] < frameW && det.ky[j] >= 0.0f && det.ky[j] < frameH;
    };
    for (const auto& [a, b] : skeleton_) {
      if (!visible(a) || !visible(b)) continue;
      NvDsDisplayMeta* dm = displayMeta();
      NvOSD_LineParams& line = dm->line_params[dm->num_lines++];
      line.x1 = static_cast<unsigned int>(det.kx[a]);
      line.y1 = static_cast<unsigned int>(det.ky[a]);
      line.x2 = static_cast<unsigned int>(det.kx[b]);
      line.y2 = static_cast<unsigned int>(det.ky[b]);
      line.line_width = 2;
      line.line_color = lineColor;
    }
    for (size_t j = 0; j < det.kx.size(); ++j) {
      if (!visible(static_cast<int>(j))) continue;
      NvDsDisplayMeta* dm = displayMeta();
      NvOSD_CircleParams& circle = dm->circle_params[dm->num_circles++];
      circle.xc = static_cast<unsigned int>(det.kx[j]);
      circle.yc = static_cast<unsigned int>(det.ky[j]);
      circle.radius = 3;
      circle.circle_color = pointColor;
      circle.has_bg_color = 1;
      circle.bg_color = pointColor;
    }
  }
}

void RfdetrPose::dumpFrame(const NvDsFrameMeta* frameMeta, const std::vector<Detection>& dets) {
  char path[4096];
  std::snprintf(path, sizeof(path), "%s/%02u_%06u.txt", dumpDir_.c_str(), frameMeta->pad_index,
                frameMeta->frame_num);
  FILE* f = std::fopen(path, "w");
  if (f == nullptr) return;
  // class_id score x1 y1 x2 y2 num_keypoints then x y confidence per keypoint
  for (const Detection& det : dets) {
    std::fprintf(f, "%u %.6f %.3f %.3f %.3f %.3f %zu", det.classId, det.score, det.x1, det.y1, det.x2, det.y2,
                 det.kx.size());
    for (size_t j = 0; j < det.kx.size(); ++j) std::fprintf(f, " %.3f %.3f %.5f", det.kx[j], det.ky[j], det.kc[j]);
    std::fprintf(f, "\n");
  }
  std::fclose(f);
}

extern "C" IDSCustomLibrary* CreateCustomAlgoCtx(DSCustom_CreateParams* /*params*/) { return new RfdetrPose(); }
