CXX      ?= g++
CXXFLAGS  = -O3 -std=c++17 -Wall -Wextra -pthread -fopenmp -fPIC

LIBDIR   = engine
BUILDDIR = build
SRC      = $(wildcard $(LIBDIR)/src/*.cpp)
OBJ      = $(patsubst $(LIBDIR)/src/%.cpp,$(BUILDDIR)/%.o,$(SRC))
LIB      = $(LIBDIR)/libengine.so

all: $(LIB)

$(BUILDDIR):
	mkdir -p $(BUILDDIR)

$(BUILDDIR)/%.o: $(LIBDIR)/src/%.cpp $(wildcard $(LIBDIR)/src/*.hpp) $(wildcard $(LIBDIR)/include/astar/*.hpp) Makefile | $(BUILDDIR)
	$(CXX) $(CXXFLAGS) -I$(LIBDIR)/include -I$(LIBDIR)/src -c $< -o $@

$(LIB): $(OBJ)
	$(CXX) $(CXXFLAGS) -shared -o $@ $(OBJ)

.PHONY: all
