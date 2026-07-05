PROJECT := amd_rocm_torch

LIB := lib$(PROJECT).so

CXX ?= g++
CXX_FLAGS ?=
INCLUDES ?=
LDFLAGS ?=
LINK_LIBS ?=

SRC := amd_rocm_torch.cpp

TORCH_DIR = $(shell python3 -c "import torch; import os; print(os.path.dirname(torch.__file__))")
INCLUDES += -I$(TORCH_DIR)/include -I$(TORCH_DIR)/include/torch/csrc/api/include 
LDFLAGS += -L$(TORCH_DIR)/lib -Wl,-rpath=$(TORCH_DIR)/lib
LINK_LIBS += -lc10 -ltorch -ltorch_cpu

PYTHON_INCLUDE_DIR = $(shell python3 -c "import sysconfig; print(sysconfig.get_path('include'))")
PYTHON_LIB_DIR = $(shell python3 -c "import sysconfig; print(sysconfig.get_path('stdlib'))")
PYTHON_VERSION = $(shell python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
INCLUDES += -I$(PYTHON_INCLUDE_DIR)
LDFLAGS += -L$(PYTHON_LIB_DIR)/../ -Wl,-rpath=$(PYTHON_LIB_DIR)/../
LINK_LIBS += -lpython$(PYTHON_VERSION)

# --- Add CUDA (or HIP) component if present
# Try torch_cuda first; if not found, try torch_hip (ROCm)
ifneq ("$(wildcard $(TORCH_DIR)/lib/libtorch_cuda.so)","")
  LINK_LIBS += -ltorch_cuda -lc10_cuda
endif
ifneq ("$(wildcard $(TORCH_DIR)/lib/libtorch_hip.so)","")
  LINK_LIBS += -ltorch_hip
endif

# --- Common flags
CXX_FLAGS += -std=c++17 -fPIC

ifeq ($(DEBUG), 1)
  CXX_FLAGS += -g -O0
else
  CXX_FLAGS += -O2
endif

# for multi-process support
CXX_FLAGS += -fno-omit-frame-pointer
LDFLAGS += -Wl,-z,nodelete

# Needed by pinning + atfork paths
LINK_LIBS += -ldl -pthread

all: $(LIB)

$(LIB): $(SRC)
	$(CXX) $(CXX_FLAGS) $(INCLUDES) $(LDFLAGS) -shared $< -o $@ $(LINK_LIBS)

.PHONY: clean
clean:
	-@rm -f $(LIB)
