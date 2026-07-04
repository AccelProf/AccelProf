PROJECT := nv-nvbit
CONFIGS := Makefile.config

include $(CONFIGS)

CXX ?=
NVCC=nvcc -ccbin=$(CXX) -D_FORCE_INLINES
PTXAS=ptxas

NVCC_VER_REQ=10.1
NVCC_VER=$(shell $(NVCC) --version | grep release | cut -f2 -d, | cut -f3 -d' ')
NVCC_VER_CHECK=$(shell echo "${NVCC_VER} >= $(NVCC_VER_REQ)" | bc)

ifeq ($(NVCC_VER_CHECK),0)
$(error ERROR: nvcc version >= $(NVCC_VER_REQ) required to compile an nvbit tool! Instrumented applications can still use lower versions of nvcc.)
endif

PTXAS_VER_ADD_FLAG=12.3
PTXAS_VER=$(shell $(PTXAS) --version | grep release | cut -f2 -d, | cut -f3 -d' ')
PTXAS_VER_CHECK=$(shell echo "${PTXAS_VER} >= $(PTXAS_VER_ADD_FLAG)" | bc)

ifeq ($(PTXAS_VER_CHECK), 0)
MAXRREGCOUNT_FLAG=-maxrregcount=24
else
MAXRREGCOUNT_FLAG=
endif

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

NVBIT_PATH=core/
INCLUDES=-I$(NVBIT_PATH) -Iinclude/ -Itorch/include/

LIBS = -L$(NVBIT_PATH) -lnvbit
NVCC_PATH = -L $(subst bin/nvcc,lib64,$(shell which nvcc | tr -s /))

SANALYZER_INC = -I$(SANALYZER_DIR)/include
SANALYZER_LDFLAGS = -L$(SANALYZER_DIR)/lib -Xlinker -rpath -Xlinker $(SANALYZER_DIR)/lib
SANALYZER_LIB = -lsanalyzer

TORCH_SCOPE_INC = -I$(TORCH_SCOPE_DIR)/include
TORCH_SCOPE_LDFLAGS = -L$(TORCH_SCOPE_DIR)/lib -Xlinker -rpath -Xlinker $(TORCH_SCOPE_DIR)/lib
TORCH_SCOPE_LIB = -ltorch_scope

SRC_DIR := src/
OBJ_DIR := obj/
LIB_DIR := lib/

CPP_SRCS := $(notdir $(wildcard $(SRC_DIR)*.cpp $(SRC_DIR)*/*.cpp))
CU_SRCS := $(notdir $(wildcard $(SRC_DIR)*.cu $(SRC_DIR)*/*.cu))
OBJS := $(addprefix $(OBJ_DIR), $(patsubst %.cpp, %.o, $(CPP_SRCS)))
OBJS += $(addprefix $(OBJ_DIR), $(patsubst %.cu, %.o, $(CU_SRCS)))

ARCH ?= all

NVBIT_TOOL = $(LIB_DIR)/lib$(PROJECT).so

all: dirs $(NVBIT_TOOL)

dirs: $(OBJ_DIR) $(LIB_DIR) $(TORCH_OBJ_DIR)

$(OBJ_DIR):
	mkdir -p $@

$(LIB_DIR):
	mkdir -p $@

$(TORCH_OBJ_DIR):
	mkdir -p $@

$(NVBIT_TOOL): $(OBJS) $(NVBIT_PATH)/libnvbit.a
	$(NVCC) -arch=$(ARCH) $(DEBUG_FLAGS) $(OBJS) $(LIBS) $(NVCC_PATH) $(SANALYZER_LDFLAGS) $(TORCH_SCOPE_LDFLAGS) -lcuda -lcudart_static -shared -o $@ $(SANALYZER_LIB) $(TORCH_SCOPE_LIB)

$(OBJ_DIR)%.o: $(SRC_DIR)/backend/%.cu
	$(NVCC) -dc -c -std=c++17 $(INCLUDES) $(SANALYZER_INC) $(TORCH_SCOPE_INC) -Xptxas -cloning=no -Xcompiler -Wall -arch=$(ARCH) $(DEBUG_FLAGS) -Xcompiler -fPIC $< -o $@

$(OBJ_DIR)%.o:: $(SRC_DIR)/inj_fns/%.cu
	$(NVCC) $(INCLUDES) $(MAXRREGCOUNT_FLAG) -Xptxas -astoolspatch --keep-device-functions -arch=$(ARCH) -Xcompiler -Wall -Xcompiler -fPIC -c $< -o $@

clean:
	-rm -rf $(OBJ_DIR) $(LIB_DIR)
