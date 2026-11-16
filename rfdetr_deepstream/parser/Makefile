# Builds libnvds_rfdetr.so (nvinfer custom-lib-path).
# Run inside a DeepStream container or on a machine/Jetson with DeepStream installed:
#     make          build
#     make clean    remove build output

DS_PATH   ?= /opt/nvidia/deepstream/deepstream
CUDA_PATH ?= /usr/local/cuda

CXX      ?= g++
# NVIDIA's headers are included as system headers: they trigger -Wextra warnings.
CXXFLAGS := -std=c++17 -O2 -Wall -Wextra -fPIC -isystem $(DS_PATH)/sources/includes -isystem $(CUDA_PATH)/include
LDFLAGS  := -shared

TARGET := libnvds_rfdetr.so
SRCS   := nvdsparsebbox_rfdetr.cpp

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(SRCS) Makefile
	$(CXX) $(CXXFLAGS) $(SRCS) -o $@ $(LDFLAGS)

clean:
	rm -f $(TARGET)
