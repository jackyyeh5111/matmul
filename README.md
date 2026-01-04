# MatMul

## Benchmark
- https://github.com/google/benchmark

```bash
g++ benchmark.cpp -std=c++11 -isystem benchmark_dir/include -Lbenchmark_dir/build/src -lbenchmark -lpthread -o benchmark
```