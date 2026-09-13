PROJECT := compute_sanitizer
CONFIGS := Makefile.config

include $(CONFIGS)

SANITIZER_PATH ?= $(CUDA_PATH)/compute-sanitizer

CXX            ?= g++
NVCC           := $(CUDA_PATH)/bin/nvcc -ccbin $(CXX)

CXX_FLAGS      ?=
INCLUDES       ?=
LDFLAGS        ?=
LINK_LIBS      ?=
NVCC_FLAGS     ?=
NVCC_INCS      ?=

INCLUDES       += -Iinclude

INCLUDES       += -I$(CUDA_PATH)/include
NVCC_INCS      += -I$(CUDA_PATH)/include

INCLUDES       += -I$(SANITIZER_PATH)/include
NVCC_INCS      += -I$(SANITIZER_PATH)/include
LDFLAGS        += -L$(SANITIZER_PATH) -Wl,-rpath=$(SANITIZER_PATH)
LINK_LIBS      += -lsanitizer-public

# The injection library calls into the CUDA runtime (cudaStreamCreateWithFlags,
# cudaDeviceGetStreamPriorityRange, ...). Link it explicitly rather than relying
# on another dependency to drag it in.
LDFLAGS        += -L$(CUDA_PATH)/lib64 -Wl,-rpath=$(CUDA_PATH)/lib64
LINK_LIBS      += -lcudart

NVCC_FLAGS     += --fatbin --compile-as-tools-patch

CXX_FLAGS      += -std=c++17

ifeq ($(DEBUG), 1)
	CXX_FLAGS += -g
endif

OPT_LVL ?= 3
ifeq ($(OPT_LVL), 0)
	CXX_FLAGS += -O0
else ifeq ($(OPT_LVL), 1)
	CXX_FLAGS += -O1 -march=native
else ifeq ($(OPT_LVL), 2)
	CXX_FLAGS += -O2 -march=native
else ifeq ($(OPT_LVL), 3)
	CXX_FLAGS += -O3 -march=native
else
    $(error Invalid OPT_LVL=$(OPT_LVL), expected 0,1,2,3)
endif

ifneq ($(OPT_LVL),0)
    CXX_FLAGS += -march=native
endif

################################################################################

# architecture
TARGET_ARCH   := $(shell uname -m)

# Patch fatbins must carry SASS for the GPU being profiled. With only PTX for an
# older arch present (e.g. compute_90 on an sm_120 card) the patch load fails:
#     sanitizerAddPatchesFromFile(...) - error(999)
# so the target architecture has to be in this list.
ifeq ($(TARGET_ARCH),aarch64)
    SMS_WANTED ?= 75 80 86 87 89 90 100 120
else
    SMS_WANTED ?= 75 80 86 89 90 100 120
endif

# Drop architectures the installed toolkit cannot generate (sm_100/sm_120 need
# CUDA >= 12.8) so that one list works across toolkit versions.
NVCC_ARCHS    := $(shell $(CUDA_PATH)/bin/nvcc --list-gpu-arch 2>/dev/null | sed 's/compute_//')
ifeq ($(NVCC_ARCHS),)
    SMS        ?= $(SMS_WANTED)
else
    SMS        ?= $(filter $(NVCC_ARCHS),$(SMS_WANTED))
endif

# Generate SASS code for each SM architecture listed in $(SMS)
$(foreach sm,$(SMS),$(eval GENCODE_FLAGS += -gencode arch=compute_$(sm),code=sm_$(sm)))

# Generate PTX code from the highest SM architecture in $(SMS) to guarantee
# forward-compatibility. Sorted numerically: $(sort) is lexicographic and would
# rank 90 above 120.
HIGHEST_SM     := $(shell printf '%s\n' $(SMS) | sort -n | tail -1)
GENCODE_FLAGS  += -gencode arch=compute_$(HIGHEST_SM),code=compute_$(HIGHEST_SM)

LIB_DIR        := lib
OBJ_DIR        := $(LIB_DIR)/obj
SRC_DIR        := src

PATCH_DIR      := $(LIB_DIR)/gpu_patch
PATCH_FATBINS  := $(addprefix $(PATCH_DIR)/, $(patsubst %.cu, %.fatbin, $(notdir $(wildcard $(PATCH_SRC_DIR)/*.cu))))
INCLUDES       += -I$(PATCH_SRC_DIR)/include
NVCC_INCS      += -I$(PATCH_SRC_DIR)/include

INCLUDES       += -I$(SANALYZER_DIR)/include
LDFLAGS        += -L$(SANALYZER_DIR)/lib -Wl,-rpath=$(SANALYZER_DIR)/lib
LINK_LIBS	   += -lsanalyzer

# torch_scope is only built when ENABLE_TORCH=1; bin/build forwards that as
# ENABLE_TORCH_SCOPE so the backend links it only when it was actually built,
# and ENABLE_CS=1 ENABLE_TORCH=0 still links.
ENABLE_TORCH_SCOPE ?= 0
ifeq ($(ENABLE_TORCH_SCOPE),1)
INCLUDES       += -I$(TORCH_SCOPE_DIR)/include
LDFLAGS        += -L$(TORCH_SCOPE_DIR)/lib -Wl,-rpath=$(TORCH_SCOPE_DIR)/lib
LINK_LIBS	   += -ltorch_scope
CXX_FLAGS      += -DENABLE_TORCH_SCOPE
endif

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
	$(NVCC) $(NVCC_FLAGS) $(NVCC_INCS) $(GENCODE_FLAGS) -o $@ $<

clean:
	rm -rf $(LIB_DIR)

clobber: clean
