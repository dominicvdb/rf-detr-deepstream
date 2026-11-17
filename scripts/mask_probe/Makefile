DS_ROOT ?= /opt/nvidia/deepstream/deepstream
CXX ?= g++

mask_probe: mask_probe.cpp
	$(CXX) -std=c++17 -O2 -Wall -Wextra mask_probe.cpp -o mask_probe \
		$(shell pkg-config --cflags --libs gstreamer-1.0) \
		-isystem $(DS_ROOT)/sources/includes -isystem /usr/local/cuda/include \
		-L$(DS_ROOT)/lib -lnvdsgst_meta -lnvds_meta -Wl,-rpath,$(DS_ROOT)/lib

clean:
	rm -f mask_probe
