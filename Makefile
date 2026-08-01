CXX      ?= g++
CXXFLAGS  = -O3 -std=c++17 -Wall -Wextra -pthread -fopenmp -fPIC

LIBDIR = astar
SRC    = $(wildcard $(LIBDIR)/src/*.cpp)
OBJ    = $(SRC:.cpp=.o)
LIB    = $(LIBDIR)/libastar.so

all: $(LIB)

$(LIBDIR)/src/%.o: $(LIBDIR)/src/%.cpp $(wildcard $(LIBDIR)/src/*.hpp) $(wildcard $(LIBDIR)/include/astar/*.hpp) Makefile
	$(CXX) $(CXXFLAGS) -I$(LIBDIR)/include -I$(LIBDIR)/src -c $< -o $@

$(LIB): $(OBJ)
	$(CXX) $(CXXFLAGS) -shared -o $@ $(OBJ)

.PHONY: all
