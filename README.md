# jax ffi break

Running `pip install -e .`, then `python tests/cuda_examples_test.py` works, but commenting out `set(CMAKE_CUDA_ARCHITECTURES 90)` in CMakeLists.txt and replacing it with `set(CMAKE_CUDA_ARCHITECTURES 90a)` breaks when running the same commands.
