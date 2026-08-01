CXX      ?= g++
STD       = -std=c++17
WARN      = -Wall -Wextra
OPT      ?= -O3
PAR       = -pthread -fopenmp
CXXFLAGS ?= $(OPT) $(STD) $(WARN) $(PAR) -fPIC

LIBDIR = astar
SRC    = $(wildcard $(LIBDIR)/src/*.cpp)
OBJ    = $(SRC:.cpp=.o)
LIB    = $(LIBDIR)/libastar.so
ARGS  ?=

all: $(LIB)

$(LIBDIR)/src/%.o: $(LIBDIR)/src/%.cpp $(wildcard $(LIBDIR)/src/*.hpp) $(wildcard $(LIBDIR)/include/astar/*.hpp) Makefile
	$(CXX) $(CXXFLAGS) -I$(LIBDIR)/include -I$(LIBDIR)/src -c $< -o $@

$(LIB): $(OBJ)
	$(CXX) $(CXXFLAGS) -shared -o $@ $(OBJ)

run: $(LIB)
	python3 run_astar.py $(ARGS)

native:
	$(MAKE) clean
	$(MAKE) OPT="-O3 -march=native" $(LIB)

clean:
	rm -f $(OBJ) $(LIB)

.PHONY: all run native clean
