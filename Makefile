# make              - собрать ./astar_port
# make run          - собрать и запустить; точки: make run ARGS="3000 3850 1500 1500"
#                     доп. флаги: --wastar=W (скорость за счёт качества), --threads=N
# make pgo          - сборка с profile-guided optimization (нужна data/ рядом)
# make portable     - без -march=native, бинарник переносим между машинами
# make debug        - отладочная сборка с санитайзерами
# make clean

CXX      ?= g++
STD       = -std=c++17
WARN      = -Wall -Wextra
OPT      ?= -O3 -march=native
PAR       = -pthread -fopenmp
CXXFLAGS ?= $(OPT) $(STD) $(WARN) $(PAR)

TARGET = astar_port
SRC    = astar_port.cpp
ARGS  ?=

all: $(TARGET)

$(TARGET): $(SRC) Makefile
	$(CXX) $(CXXFLAGS) -o $@ $(SRC)

run: $(TARGET)
	./$(TARGET) $(ARGS)

pgo: $(SRC)
	$(CXX) $(CXXFLAGS) -fprofile-generate -o $(TARGET) $(SRC)
	./$(TARGET) $(ARGS)
	$(CXX) $(CXXFLAGS) -fprofile-use -fprofile-correction -o $(TARGET) $(SRC)
	@echo "PGO-сборка готова: ./$(TARGET)"

portable:
	$(MAKE) clean
	$(MAKE) OPT="-O3" $(TARGET)

debug: $(SRC)
	$(CXX) -O0 -g $(STD) $(WARN) -pthread -fsanitize=address,undefined \
		-o $(TARGET)_debug $(SRC)

clean:
	rm -f $(TARGET) $(TARGET)_debug *.gcda

.PHONY: all run pgo portable debug clean
