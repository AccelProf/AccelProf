CUDA_PATH      ?= /usr/local/cuda
SANITIZER_PATH ?= $(CUDA_PATH)/compute-sanitizer

CXX            ?= g++
NVCC           := $(CUDA_PATH)/bin/nvcc -ccbin $(CXX)

INCLUDE_FLAGS  := -I$(CUDA_PATH)/include -I$(SANITIZER_PATH)/include

LINK_FLAGS     := -L$(SANITIZER_PATH) -fPIC -shared
LINK_LIBS      := -lsanitizer-public

NVCC_FLAGS     := --fatbin --compile-as-tools-patch
NVCC_FLAGS     += $(INCLUDE_FLAGS)

ifeq ($(dbg),1)
    # NVCC_FLAGS += -g -G
	CXX_FLAGS  += -g -O0
endif

################################################################################

# architecture
TARGET_ARCH   := $(shell uname -m)

ifeq ($(TARGET_ARCH),aarch64)
    SMS        ?= 53 61 70 72 75 80 86 87 90
else
    SMS        ?= 52 60 70 75 80 86 90
endif

# Generate SASS code for each SM architecture listed in $(SMS)
$(foreach sm,$(SMS),$(eval GENCODE_FLAGS += -gencode arch=compute_$(sm),code=sm_$(sm)))

# Generate PTX code from the highest SM architecture in $(SMS) to guarantee forward-compatibility
HIGHEST_SM     := $(lastword $(sort $(SMS)))
GENCODE_FLAGS  += -gencode arch=compute_$(HIGHEST_SM),code=compute_$(HIGHEST_SM)

LIB_DIR        := lib

PATCH_DIR      := $(LIB_DIR)/gpu_patch
PATCH_SRC_DIR  := gpu_src
PATCH_FATBINS  := $(addprefix $(PATCH_DIR)/, $(patsubst %.cu, %.fatbin, $(notdir $(wildcard $(PATCH_SRC_DIR)/*.cu))))

INCLUDE_FLAGS  += -I$(PATCH_SRC_DIR)

################################################################################

# Target rules
all: dirs libs

dirs: $(PATCH_DIR)

libs: $(LIB_DIR)/libcompute_sanitizer.so $(PATCH_FATBINS)

$(PATCH_DIR):
	mkdir -p $@

$(LIB_DIR)/libcompute_sanitizer.so: compute_sanitizer.cpp
	$(CXX) $(CXX_FLAGS) $(INCLUDE_FLAGS) $(LINK_FLAGS) -o $@ $< $(LINK_LIBS)

$(PATCH_DIR)/%.fatbin:$(PATCH_SRC_DIR)/%.cu
	$(NVCC) $(NVCC_FLAGS) $(GENCODE_FLAGS) -o $@ -c $<

clean:
	rm -rf $(LIB_DIR)

clobber: clean
