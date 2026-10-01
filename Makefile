# Parallelism is capped: the benchmark machine is shared.
JOBS ?= 4
CARGO ?= cargo
DATA ?= bench/data
PY ?= python3

.PHONY: build test lint fmt bench bench-kernels bench-ann clean

build:
	$(CARGO) build --release -j$(JOBS)

test:
	$(CARGO) test -j$(JOBS) -- --test-threads=$(JOBS)

lint:
	$(CARGO) fmt --check
	$(CARGO) clippy -j$(JOBS) --all-targets -- -D warnings

fmt:
	$(CARGO) fmt

# Kernel micro-benchmarks (seconds, no data needed).
bench-kernels: build
	./target/release/quiver bench kernels

# ANN benchmarks: downloads datasets into $(DATA) and writes bench/results/.
bench-ann: build
	bench/ann/run.sh $(DATA)

bench: bench-kernels

clean:
	$(CARGO) clean
