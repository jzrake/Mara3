# =====================================================================
# Mara build system
# =====================================================================
#
#
# External library dependencies: HDF5, MPI
#
#
# Notes
# -----
#
# - A useful resource for techniques to process Makefile dependencies:
# www.microhowto.info/howto/automatically_generate_makefile_dependencies.html
#
# - Using -O0 rather than -O3 during development may reduce compilation time
# significantly.


# Build configuration
# =====================================================================


# Default build macros
CXX      = mpicxx
CXXFLAGS = -std=c++17 -Wall -O0 -MMD -MP
LDFLAGS  = -lhdf5


# If a Makefile.in exists in this directory, then use it
-include Makefile.in


# Build macros
# =====================================================================
SRC         := $(wildcard src/*.cpp)
OBJ         := $(SRC:%.cpp=%.o)
DEP         := $(SRC:%.cpp=%.d)
EXE         := mara

EXAMPLE_SRC := $(wildcard examples/*.cpp) $(wildcard examples/advect_1d/*.cpp)
EXAMPLE_DEP := $(EXAMPLE_SRC:%.cpp=%.d)
EXAMPLE_EXE := $(EXAMPLE_SRC:%.cpp=%)


# Build rules
# =====================================================================
all: $(EXE) $(EXAMPLE_EXE) tutorial

$(EXE): $(OBJ)
	$(CXX) -o $@ $^ $(LDFLAGS)

$(EXAMPLE_EXE): CXXFLAGS += -Isrc

tutorial:
	$(MAKE) -C tutorial

clean:
	$(RM) $(OBJ) $(DEP) $(EXE) $(EXAMPLE_EXE) $(EXAMPLE_DEP)

.PHONY: tutorial

-include $(DEP)
