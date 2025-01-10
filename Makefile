PROJECT := compute_sanitizer
CONFIGS := Makefile.config

include $(CONFIGS)

CUDA_PATH      ?= /usr/local/cuda
SANITIZER_PATH ?= $(CUDA_PATH)/compute-sanitizer

CXX            ?= g++
NVCC           := $(CUDA_PATH)/bin/nvcc -ccbin $(CXX)

CXX_FLAGS      ?=
INCLUDES       ?=
LDFLAGS        ?=
LINK_LIBS      ?=
NVCC_FLAGS     ?=
NVCC_INCS      ?=

INCLUDES       += -I$(CUDA_PATH)/include
NVCC_INCS      += -I$(CUDA_PATH)/include

INCLUDES       += -I$(SANITIZER_PATH)/include
NVCC_INCS      += -I$(SANITIZER_PATH)/include
LDFLAGS        += -L$(SANITIZER_PATH)
LINK_LIBS      += -lsanitizer-public

NVCC_FLAGS     += --fatbin --compile-as-tools-patch

CXX_FLAGS      += -std=c++17

ifeq ($(DEBUG), 1)
#	NVCC_FLAGS += -g -G
	CXX_FLAGS += -g -O0
else
	CXX_FLAGS += -O3
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
OBJ_DIR        := $(LIB_DIR)/obj
SRC_DIR        := src

PATCH_DIR      := $(LIB_DIR)/gpu_patch
PATCH_SRC_DIR  := gpu_src
PATCH_FATBINS  := $(addprefix $(PATCH_DIR)/, $(patsubst %.cu, %.fatbin, $(notdir $(wildcard $(PATCH_SRC_DIR)/*.cu))))
INCLUDES       += -I$(PATCH_SRC_DIR)/include
NVCC_INCS      += -I$(PATCH_SRC_DIR)/include

INCLUDES       += -I$(SANALYZER_DIR)/include
LDFLAGS        += -L$(SANALYZER_DIR)/lib -Wl,-rpath=$(SANALYZER_DIR)/lib
LINK_LIBS	   += -lsanalyzer

INCLUDES       += -I$(TENSOR_SCOPE_DIR)/include
LDFLAGS        += -L$(TENSOR_SCOPE_DIR)/lib -Wl,-rpath=$(TENSOR_SCOPE_DIR)/lib
LINK_LIBS	   += -ltensor_scope

SRCS := $(notdir $(wildcard $(SRC_DIR)/*.cpp))
OBJS := $(addprefix $(OBJ_DIR)/, $(patsubst %.cpp, %.o, $(SRCS)))

################################################################################

# Target rules
all: dirs libs

dirs: $(PATCH_DIR) $(OBJ_DIR)

libs: $(LIB_DIR)/lib$(PROJECT).so $(PATCH_FATBINS)

$(PATCH_DIR):
	mkdir -p $@

$(OBJ_DIR):
	mkdir -p $@

$(LIB_DIR)/lib$(PROJECT).so: $(OBJS)
	$(CXX) $(LDFLAGS) -fPIC -shared -o $@ $^ $(LINK_LIBS)

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp
	$(CXX) $(CXX_FLAGS) $(INCLUDES) -fPIC -o $@ -c $<

$(PATCH_DIR)/%.fatbin: $(PATCH_SRC_DIR)/%.cu
	$(NVCC) $(NVCC_FLAGS) $(NVCC_INCS) $(GENCODE_FLAGS) -o $@ -c $<

clean:
	rm -rf $(LIB_DIR)

clobber: clean
