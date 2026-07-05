PROJECT := rocm_callback
CONFIGS := Makefile.config

include $(CONFIGS)

# Compiler
HIPCC = hipcc
CXX = g++

# Directories
SRC_DIR = src
BUILD_DIR = build
LIB_DIR = $(BUILD_DIR)/lib
OBJ_DIR = $(BUILD_DIR)/obj

# Source files
SOURCES = $(notdir $(wildcard $(SRC_DIR)/*.cpp $(SRC_DIR)/*/*.cpp))
OBJECTS = $(addprefix $(OBJ_DIR)/, $(patsubst %.cpp, %.o, $(SOURCES)))

# Include directories
INCLUDE_FLAGS = -I/opt/rocm/include

SANALYZER_INC = -I$(SANALYZER_DIR)/include
SANALYZER_LDFLAGS = -L$(SANALYZER_DIR)/lib -Wl,-rpath=$(SANALYZER_DIR)/lib
SANALYZER_LIB = -lsanalyzer

# Linker flags
LIBS = -lrocprofiler-sdk -lrocprofiler-sdk-roctx -lpthread
LDFLAGS = -L/opt/rocm/lib -Wl,-rpath,/opt/rocm/lib $(LIBS)

# Compiler flags
CXX_FLAGS =
HIPFLAGS = -std=c++17 -fPIC -Wall

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

HIPFLAGS += $(CXX_FLAGS)


# Targets
all: dirs rocm_callback

rocm_callback: $(LIB_DIR)/librocm_callback.so

dirs: $(BUILD_DIR) $(LIB_DIR) $(OBJ_DIR)

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(LIB_DIR):
	mkdir -p $(LIB_DIR)


$(OBJ_DIR):
	mkdir -p $(OBJ_DIR)

# Build shared library
$(LIB_DIR)/lib$(PROJECT).so: $(OBJ_DIR)/$(PROJECT).o
	$(HIPCC) -shared -o $@ $^ $(LDFLAGS) $(SANALYZER_LDFLAGS) $(SANALYZER_LIB)

# Compile source files
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp
	$(HIPCC) $(HIPFLAGS) $(INCLUDE_FLAGS) $(SANALYZER_INC) -c $< -o $@

# Run tests
test: all
	LD_LIBRARY_PATH=$(LIB_DIR):$$LD_LIBRARY_PATH

# Clean build
clean:
	rm -rf $(BUILD_DIR)
