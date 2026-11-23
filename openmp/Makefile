CXX      ?= g++
CXXFLAGS ?= -O2 -std=c++20 -Wall -fopenmp
CPPFLAGS += -Icommon

TASK_DIRS := $(sort $(wildcard task*))
BINS      := $(foreach d,$(TASK_DIRS),bin/$(word 1,$(subst _, ,$(d))))

.PHONY: all clean run-%

all: $(BINS)

# task01_minmax/main.cpp -> bin/task01
.SECONDEXPANSION:
bin/%: $$(wildcard %_*/main.cpp) common/bench.hpp | bin
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $< -o $@

bin runs data:
	mkdir -p $@

# make run-task01 ARGS="--sizes 1000,1000000 --threads 1,2,4"  ->  runs/task01.csv
run-%: bin/% | runs
	bash -o pipefail -c './bin/$* $(ARGS) | tee runs/$*.csv'

clean:
	rm -rf bin
