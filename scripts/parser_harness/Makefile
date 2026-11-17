# Builds the parser test harness on any machine (no DeepStream needed): the real parser source plus a stand-in
# for DeepStream's header. Output (harness) is git-ignored via build/.
CXX      ?= c++
CXXFLAGS := -std=c++17 -O2 -Wall -Wextra -I.
PARSER   := ../../rfdetr_deepstream/parser/nvdsparsebbox_rfdetr.cpp
OUT      := ../../build/parser_harness/harness

.PHONY: all clean
all: $(OUT)

$(OUT): harness.cpp nvdsinfer_custom_impl.h $(PARSER) Makefile
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $(PARSER) harness.cpp -o $@

clean:
	rm -f $(OUT)
