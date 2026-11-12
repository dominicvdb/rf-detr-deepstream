// Keypoint metadata attached by libnvds_rfdetr_pose.so to every NvDsObjectMeta it creates
// (in obj_meta->obj_user_meta_list). Include this header to read the keypoints downstream:
//
//   NvDsMetaType type = nvds_get_user_meta_type((gchar*)RFDETR_KEYPOINTS_META_STRING);
//   for (NvDsMetaList* l = obj_meta->obj_user_meta_list; l; l = l->next) {
//     NvDsUserMeta* um = (NvDsUserMeta*)l->data;
//     if (um->base_meta.meta_type != type) continue;
//     const RfdetrKeypoints* kp = (const RfdetrKeypoints*)um->user_meta_data;
//     ... kp->x[i], kp->y[i], kp->confidence[i] for i < kp->num_keypoints
//   }
#ifndef NVDS_RFDETR_KEYPOINTS_META_H
#define NVDS_RFDETR_KEYPOINTS_META_H

#include <stdint.h>

#define RFDETR_KEYPOINTS_META_STRING "RFDETR.KEYPOINTS.USER_META"
#define RFDETR_MAX_KEYPOINTS 64

typedef struct {
  uint32_t num_keypoints;
  float x[RFDETR_MAX_KEYPOINTS];           // frame pixels (nvstreammux resolution)
  float y[RFDETR_MAX_KEYPOINTS];
  float confidence[RFDETR_MAX_KEYPOINTS];  // findability, sigmoid of the keypoint head's logit
} RfdetrKeypoints;

#endif
