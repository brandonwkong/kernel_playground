Kernel Playground

A personal learning repo for GPU kernel programming — profiling and writing
small ops (starting with RMSNorm) to build intuition for GPU performance:
timing methodology, memory-bound vs compute-bound behavior, kernel fusion,
and eventually hand-written Triton/CUDA kernels. Currently in "play" mode:
no deadlines, just working through the core loop of implement -> verify
correctness -> benchmark -> profile -> optimize.