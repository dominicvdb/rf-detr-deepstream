# Builds libnvds_rfdetr_pose.so (nvdsvideotemplate customlib-name).
# Run inside a DeepStream container or on a machine/Jetson with DeepStream installed:
#     make          build
#     make clean    remove build output

DS_PATH   ?= /opt/nvidia/deepstream/deepstream
CUDA_PATH ?= /usr/local/cuda

CXX  ?= g++
PKGS := gstreamer-1.0 gstreamer-base-1.0 gstreamer-video-1.0

# DeepStream, nvdsvideotemplate and GStreamer headers are included as system headers.
CXXFLAGS := -std=c++17 -O2 -Wall -Wextra -fPIC \
	-isystem $(DS_PATH)/sources/includes \
	-isystem $(DS_PATH)/sources/gst-plugins/gst-nvdsvideotemplate/includes \
	-isystem $(CUDA_PATH)/include \
	$(subst -I,-isystem ,$(shell pkg-config --cflags $(PKGS)))
LDFLAGS := -shared -Wl,-no-undefined \
	-L$(CUDA_PATH)/lib64 -lcudart \
	-L$(DS_PATH)/lib -lnvdsgst_helper -lnvdsgst_meta -lnvds_meta -lnvbufsurface -lnvbufsurftransform \
	-lnvdsbufferpool -Wl,-rpath,$(DS_PATH)/lib \
	$(shell pkg-config --libs $(PKGS))

TARGET := libnvds_rfdetr_pose.so
SRCS   := nvdsrfdetr_pose.cpp

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(SRCS) nvds_rfdetr_keypoints_meta.h Makefile
	$(CXX) $(CXXFLAGS) $(SRCS) -o $@ $(LDFLAGS)

clean:
	rm -f $(TARGET)
