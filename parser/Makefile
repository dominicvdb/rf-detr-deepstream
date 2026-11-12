# Builds libnvds_rfdetr.so, the shared library nvinfer loads via custom-lib-path.
# Run inside a DeepStream container (or on a machine/Jetson with DeepStream installed):
#     make            build
#     make clean      remove build output

# ?= means "use this value unless it is already set", e.g.: make DS_PATH=/other/path
DS_PATH   ?= /opt/nvidia/deepstream/deepstream
CUDA_PATH ?= /usr/local/cuda

CXX      ?= g++
# -std=c++17       C++ language version
# -O2              optimise
# -Wall -Wextra    turn on (almost) all warnings
# -fPIC            "position-independent code": required for code that goes into a shared library
# -isystem <dir>   extra folder to search for #include files, treated as third-party "system" headers:
#                  the compiler doesn't warn about code in them (NVIDIA's headers trigger -Wextra warnings)
CXXFLAGS := -std=c++17 -O2 -Wall -Wextra -fPIC -isystem $(DS_PATH)/sources/includes -isystem $(CUDA_PATH)/include
# -shared          produce a shared library (.so) instead of an executable
LDFLAGS  := -shared

TARGET := libnvds_rfdetr.so
SRCS   := nvdsparsebbox_rfdetr.cpp

.PHONY: all clean

all: $(TARGET)

# "target: dependencies" then a TAB-indented command. make reruns the command only when a
# dependency is newer than the target. $@ = the target name, $^ = all dependencies.
$(TARGET): $(SRCS) Makefile
	$(CXX) $(CXXFLAGS) $(SRCS) -o $@ $(LDFLAGS)

clean:
	rm -f $(TARGET)
