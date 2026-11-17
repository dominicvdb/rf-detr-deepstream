// Runs a GStreamer pipeline and dumps every object nvinfer attaches, with its instance mask, for
// scripts/parity_segmentation.py. The pipeline needs an nvinfer element named "infer"; the probe sits on its src
// pad, so rect_params and mask_params are exactly what nvinfer produced (nothing downstream has touched them).
//
//   mask_probe PIPELINE_FILE objects.txt masks.bin     (PIPELINE_FILE holds a gst-launch pipeline description)
//
// objects.txt, one line per object:
//   source_id frame_num class_id confidence left top width height mask_width mask_height threshold byte_offset
// masks.bin: the float masks back to back (byte_offset into it; no line when the object has no mask).
//
// Build (inside a DeepStream container): make -C scripts/mask_probe

#include <gst/gst.h>

#include <cstdio>

#include "gstnvdsmeta.h"

namespace {

FILE* gObjects = nullptr;
FILE* gMasks = nullptr;
size_t gOffset = 0;

GstPadProbeReturn onBuffer(GstPad*, GstPadProbeInfo* info, gpointer) {
  NvDsBatchMeta* batch = gst_buffer_get_nvds_batch_meta(GST_PAD_PROBE_INFO_BUFFER(info));
  if (batch == nullptr) {
    return GST_PAD_PROBE_OK;
  }
  for (NvDsMetaList* f = batch->frame_meta_list; f != nullptr; f = f->next) {
    const NvDsFrameMeta* frame = static_cast<NvDsFrameMeta*>(f->data);
    for (NvDsMetaList* o = frame->obj_meta_list; o != nullptr; o = o->next) {
      const NvDsObjectMeta* obj = static_cast<NvDsObjectMeta*>(o->data);
      const NvOSD_MaskParams& mask = obj->mask_params;
      const bool hasMask = mask.data != nullptr && mask.size > 0;
      std::fprintf(gObjects, "%u %d %d %.7f %.5f %.5f %.5f %.5f %u %u %.3f %zu\n", frame->source_id,
                   frame->frame_num, obj->class_id, obj->confidence, obj->rect_params.left, obj->rect_params.top,
                   obj->rect_params.width, obj->rect_params.height, hasMask ? mask.width : 0,
                   hasMask ? mask.height : 0, mask.threshold, gOffset);
      if (hasMask) {
        std::fwrite(mask.data, 1, mask.size, gMasks);
        gOffset += mask.size;
      }
    }
  }
  return GST_PAD_PROBE_OK;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 4) {
    std::fprintf(stderr, "usage: %s PIPELINE_FILE objects.txt masks.bin\n", argv[0]);
    return 2;
  }
  gst_init(&argc, &argv);
  gchar* description = nullptr;
  if (!g_file_get_contents(argv[1], &description, nullptr, nullptr)) {
    std::fprintf(stderr, "cannot read %s\n", argv[1]);
    return 2;
  }
  GError* error = nullptr;
  GstElement* pipeline = gst_parse_launch(description, &error);
  if (pipeline == nullptr) {
    std::fprintf(stderr, "bad pipeline: %s\n", error->message);
    return 2;
  }
  GstElement* infer = gst_bin_get_by_name(GST_BIN(pipeline), "infer");
  if (infer == nullptr) {
    std::fprintf(stderr, "the pipeline has no element named \"infer\"\n");
    return 2;
  }
  gObjects = std::fopen(argv[2], "w");
  gMasks = std::fopen(argv[3], "wb");
  if (gObjects == nullptr || gMasks == nullptr) {
    std::fprintf(stderr, "cannot open the output files\n");
    return 2;
  }
  GstPad* pad = gst_element_get_static_pad(infer, "src");
  gst_pad_add_probe(pad, GST_PAD_PROBE_TYPE_BUFFER, onBuffer, nullptr, nullptr);
  gst_object_unref(pad);

  gst_element_set_state(pipeline, GST_STATE_PLAYING);
  GstBus* bus = gst_element_get_bus(pipeline);
  GstMessage* msg = gst_bus_timed_pop_filtered(bus, GST_CLOCK_TIME_NONE,
                                               static_cast<GstMessageType>(GST_MESSAGE_ERROR | GST_MESSAGE_EOS));
  int code = 0;
  if (GST_MESSAGE_TYPE(msg) == GST_MESSAGE_ERROR) {
    GError* err = nullptr;
    gst_message_parse_error(msg, &err, nullptr);
    std::fprintf(stderr, "pipeline error: %s\n", err->message);
    code = 1;
  }
  gst_message_unref(msg);
  gst_element_set_state(pipeline, GST_STATE_NULL);
  gst_object_unref(bus);
  gst_object_unref(pipeline);
  std::fclose(gObjects);
  std::fclose(gMasks);
  return code;
}
