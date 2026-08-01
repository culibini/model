# Сборка и запуск порта astar.
#
#   make              — собрать оптимизированный ./astar_port
#   make run          — собрать и запустить с маршрутом по умолчанию
#   make run ARGS="3000 3850 1500 1500"
#                     — свои точки (парами y x), можно добавить --threads=N
#   make pgo          — сборка с profile-guided optimization: компиляция с
#                       профилировкой, тренировочный прогон (нужна data/!),
#                       пересборка по собранному профилю. Даёт обычно 5-15%.
#   make portable     — без -march=native: бинарник можно переносить на
#                       другие машины (без AVX2 под конкретный процессор)
#   make debug        — отладочная сборка с санитайзерами (./astar_port_debug)
#   make clean
#
# Windows: работает с MinGW (mingw32-make) и в WSL как есть.

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
