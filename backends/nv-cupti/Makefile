PROJECT := nv-cupti
CONFIGS := Makefile.config

include $(CONFIGS)

# CUPTI is usually in $(CUDA_PATH)/extras/CUPTI; some installs put it in $(CUDA_PATH) directly.
CUPTI_PATH     ?= $(if $(wildcard $(CUDA_PATH)/extras/CUPTI/include/cupti.h),$(CUDA_PATH)/extras/CUPTI,$(CUDA_PATH))

CXX            ?= g++

CXX_FLAGS      ?=
INCLUDES       ?=
LDFLAGS        ?=
LINK_LIBS      ?=

INCLUDES       += -I$(CUDA_PATH)/include
INCLUDES       += -I$(CUPTI_PATH)/include
LDFLAGS        += -L$(CUPTI_PATH)/lib64 -Wl,-rpath=$(CUPTI_PATH)/lib64
LINK_LIBS      += -lcupti

INCLUDES       += -I$(SANALYZER_DIR)/include
LDFLAGS        += -L$(SANALYZER_DIR)/lib -Wl,-rpath=$(SANALYZER_DIR)/lib
LINK_LIBS      += -lsanalyzer

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

################################################################################

LIB_DIR        := lib
OBJ_DIR        := $(LIB_DIR)/obj
SRC_DIR        := src

SRCS := $(notdir $(wildcard $(SRC_DIR)/*.cpp))
OBJS := $(addprefix $(OBJ_DIR)/, $(patsubst %.cpp, %.o, $(SRCS)))

################################################################################

# Target rules
all: dirs libs

dirs: $(OBJ_DIR)

libs: $(LIB_DIR)/lib$(PROJECT).so

$(OBJ_DIR):
	mkdir -p $@

$(LIB_DIR)/lib$(PROJECT).so: $(OBJS)
	$(CXX) $(LDFLAGS) -fPIC -shared -o $@ $^ $(LINK_LIBS)

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp
	$(CXX) $(CXX_FLAGS) $(INCLUDES) -fPIC -o $@ -c $<

clean:
	rm -rf $(LIB_DIR)

clobber: clean
