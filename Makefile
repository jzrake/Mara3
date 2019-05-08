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
SRC      := $(wildcard src/*.cpp)
OBJ      := $(SRC:%.cpp=%.o)
DEP      := $(SRC:%.cpp=%.d)


# Build rules
# =====================================================================
#
mara3: $(OBJ)
	$(CXX) -o $@ $^ $(LDFLAGS)

clean:
	$(RM) $(OBJ) $(DEP) mara3

-include $(DEP)
